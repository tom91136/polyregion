#include <algorithm>
#include <cctype>

#include <catch2/catch_test_macros.hpp>

#include "polyfront/options_backend.hpp"
#include "polyfront/package_program.hpp"
#include "polyfront/polyc_client.hpp"
#include "polyfront/program_fragment.hpp"
#include "polyfront/resolved_sym_program_compilation.hpp"

#include "polyast_codec.h"

using namespace polyregion;

namespace {

polyast::FunctionDecl transform(const polyast::Sym &name, const std::string &variable, std::optional<int32_t> exactSizeInBytes = {}) {
  using namespace polyast;
  const auto typeVariable = Type::Var(variable, exactSizeInBytes);
  const auto type = typeVariable.widen();
  const auto pointer = Type::Ptr(type, TypeSpace::Global()).widen();
  const auto extent = ArgExtent::Elements(ArgSizeExpr::Param(2));
  return FunctionDecl(name, {typeVariable}, {},
                      {Arg(Named("in", pointer), {}, ArgBoundary(ArgAccess::Read(), extent)),
                       Arg(Named("out", pointer), {}, ArgBoundary(ArgAccess::Write(), extent)), Arg(Named("n", Type::IntS32()), {}, {})},
                      {}, {}, Type::Unit0(), FunctionAffinity::Host());
}

polyast::Function variant(polyast::FunctionDecl decl, const polyast::Sym &publicName) {
  using namespace polyast;
  return Function(std::move(decl), {}, FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), CallConvention::RegularCall(), publicName,
                  {"gpu"});
}

} // namespace

TEST_CASE("program fragments merge host orchestration with device entries") {
  using namespace polyast;
  const auto i32 = Type::IntS32().widen();
  const auto hostDecl = FunctionDecl(Sym({"dispatch"}), {}, {}, {}, {}, {}, Type::Unit0(), FunctionAffinity::Host());
  const auto kernelDecl = FunctionDecl(Sym({"kernel"}), {}, {}, {Arg(Named("out", Type::Ptr(i32, TypeSpace::Global())), {})}, {}, {},
                                       Type::Unit0(), FunctionAffinity::Offload());
  const auto hostKernel = Function(kernelDecl, {Stmt::Return(Expr::Alias(Term::Unit0Const()))}, FunctionVisibility::Internal(),
                                   FunctionFpMode::Relaxed(), CallConvention::OffloadEntry());
  const auto deviceKernel = hostKernel.withBody(
      {Stmt::Var(Named("device", i32), Expr::Alias(Term::IntS32Const(7)), false), Stmt::Return(Expr::Alias(Term::Unit0Const()))});
  const auto host =
      polyfront::packageProgram({Function(hostDecl, {Stmt::Return(Expr::Alias(Term::Unit0Const()))}, FunctionVisibility::Exported(),
                                          FunctionFpMode::Relaxed(), CallConvention::RegularCall()),
                                 hostKernel},
                                {});
  const auto device = polyfront::packageProgram({deviceKernel}, {});
  const auto merged = polyfront::package::mergeProgramFragments(host, device);
  REQUIRE(merged);
  REQUIRE_FALSE(merged.value->entry);
  REQUIRE(merged.value->functions.size() == 2);
  const auto kernel = merged.value->functions ^ aspartame::collect_first([&](const auto &function) -> std::optional<Function> {
                        return function.decl.name == kernelDecl.name ? std::optional{function} : std::nullopt;
                      });
  REQUIRE(kernel);
  CHECK(kernel->convention.is<CallConvention::OffloadEntry>());
  CHECK(kernel->collect_all<Term::IntS32Const>() == std::vector{Term::IntS32Const(7)});
}

TEST_CASE("program fragment merging keeps same-named host helpers distinct from device entries") {
  using namespace polyast;
  const auto decl = FunctionDecl(Sym({"helper"}), {}, {}, {}, {}, {}, Type::Unit0(), FunctionAffinity::Host());
  const auto host =
      polyfront::packageProgram({Function(decl, {Stmt::Return(Expr::Alias(Term::Unit0Const()))}, FunctionVisibility::Internal(),
                                          FunctionFpMode::Relaxed(), CallConvention::RegularCall())},
                                {});
  const auto device =
      polyfront::packageProgram({Function(decl.withAffinity(FunctionAffinity::Offload()), {Stmt::Return(Expr::Alias(Term::Unit0Const()))},
                                          FunctionVisibility::Internal(), FunctionFpMode::Relaxed(), CallConvention::OffloadEntry())},
                                {});
  const auto merged = polyfront::package::mergeProgramFragments(host, device);
  REQUIRE(merged);
  CHECK(merged.value->functions.size() == 2);
  CHECK(merged.value->functions ^ aspartame::exists([](const auto &function) { return function.decl.name == Sym({"helper"}); }));
  CHECK(merged.value->functions ^ aspartame::exists([](const auto &function) { return function.decl.name == Sym({"#device", "helper"}); }));
}

TEST_CASE("program fragment merging preserves overload signatures") {
  using namespace polyast;
  const auto name = Sym({"overloaded"});
  const auto make = [&](const Type::Any &type, int32_t value) {
    const auto decl = FunctionDecl(name, {}, {}, {Arg(Named("value", type), {})}, {}, {}, type, FunctionAffinity::Offload());
    return Function(decl, {Stmt::Return(Expr::Alias(Term::IntS32Const(value)))}, FunctionVisibility::Internal(), FunctionFpMode::Relaxed(),
                    CallConvention::RegularCall());
  };
  const auto device = polyfront::packageProgram({make(Type::IntS32(), 1), make(Type::Float32(), 2)}, {});
  const auto merged = polyfront::package::mergeProgramFragments(polyfront::packageProgram({}, {}), device);
  REQUIRE(merged);
  REQUIRE(merged.value->functions.size() == 2);
  CHECK(merged.value->functions[0].decl.args.front().named.tpe != merged.value->functions[1].decl.args.front().named.tpe);
}

TEST_CASE("program fragment struct renaming propagates through enclosing device types") {
  using namespace polyast;
  const auto a = Sym({"A"});
  const auto b = Sym({"B"});
  const auto hostA = StructDef(a, {}, {Named("value", Type::IntS32())}, {}, false);
  const auto deviceA = StructDef(a, {}, {Named("value", Type::Float32())}, {}, false);
  const auto bType = Type::Struct(b, {}).widen();
  const auto nestedA = Type::Struct(a, {}).widen();
  const auto hostB = StructDef(b, {}, {Named("nested", nestedA)}, {}, false);
  const auto deviceB = StructDef(b, {}, {Named("nested", nestedA)}, {}, false);
  const auto declaration =
      FunctionDecl(Sym({"helper"}), {}, {}, {Arg(Named("value", bType), {})}, {}, {}, Type::Unit0(), FunctionAffinity::Offload());
  const auto function = Function(declaration, {Stmt::Return(Expr::Alias(Term::Unit0Const()))}, FunctionVisibility::Internal(),
                                 FunctionFpMode::Relaxed(), CallConvention::RegularCall());
  const auto merged = polyfront::package::mergeProgramFragments(polyfront::packageProgram({function}, {hostA, hostB}),
                                                                polyfront::packageProgram({function}, {deviceA, deviceB}));
  REQUIRE(merged);
  CHECK(merged.value->defs ^ aspartame::exists([](const auto &definition) { return definition.name == Sym({"#device", "A"}); }));
  CHECK(merged.value->defs ^ aspartame::exists([](const auto &definition) { return definition.name == Sym({"#device", "B"}); }));
}

TEST_CASE("program fragment merging rejects mismatched offload entry ABIs") {
  using namespace polyast;
  const auto name = Sym({"kernel"});
  const auto entry = [&](const Type::Any &type) {
    const auto declaration =
        FunctionDecl(name, {}, {}, {Arg(Named("value", type), {})}, {}, {}, Type::Unit0(), FunctionAffinity::Offload());
    return Function(declaration, {Stmt::Return(Expr::Alias(Term::Unit0Const()))}, FunctionVisibility::Internal(), FunctionFpMode::Relaxed(),
                    CallConvention::OffloadEntry());
  };
  const auto merged = polyfront::package::mergeProgramFragments(polyfront::packageProgram({entry(Type::IntS32())}, {}),
                                                                polyfront::packageProgram({entry(Type::IntS64())}, {}));
  CHECK_FALSE(merged);
  CHECK(merged.errors ^ aspartame::exists([](const auto &error) { return error.find("offload entry ABI") != std::string::npos; }));
}

TEST_CASE("program fragment merging renames body-local entry types outside the entry ABI") {
  using namespace polyast;
  const auto name = Sym({"LocalState"});
  const auto type = Type::Struct(name, {}).widen();
  const auto declaration = FunctionDecl(Sym({"kernel"}), {}, {}, {}, {}, {}, Type::Unit0(), FunctionAffinity::Offload());
  const auto entry = Function(
      declaration, {Stmt::Var(Named("local", type), std::optional<Expr::Any>{}, true), Stmt::Return(Expr::Alias(Term::Unit0Const()))},
      FunctionVisibility::Internal(), FunctionFpMode::Relaxed(), CallConvention::OffloadEntry());
  const auto hostDefinition = StructDef(name, {}, {Named("value", Type::IntS32())}, {}, false);
  const auto deviceDefinition = StructDef(name, {}, {Named("value", Type::Float32())}, {}, false);
  const auto merged = polyfront::package::mergeProgramFragments(polyfront::packageProgram({entry}, {hostDefinition}),
                                                                polyfront::packageProgram({entry}, {deviceDefinition}));
  REQUIRE(merged);
  CHECK(merged.value->defs ^ aspartame::exists([](const auto &definition) { return definition.name == Sym({"#device", "LocalState"}); }));
}

TEST_CASE("resolved package Sym targets follow the consumer architecture") {
  using polyregion::compiletime::Target;
  const llvm::Triple x86("x86_64-unknown-linux-gnu"), riscv("riscv64-unknown-linux-gnu"), ppc("powerpc64le-unknown-linux-gnu");
  CHECK(polyfront::objectTargetFor(llvm::Triple("x86_64-apple-darwin")) == Target::Object_LLVM_x86_64);
  CHECK(polyfront::objectTargetFor(llvm::Triple("aarch64-apple-darwin")) == Target::Object_LLVM_AArch64);
  CHECK(polyfront::objectTargetFor(llvm::Triple("armv7-unknown-linux-gnueabihf")) == Target::Object_LLVM_ARM);
  CHECK(polyfront::objectTargetFor(riscv, riscv) == Target::Object_LLVM_HOST);
  CHECK(polyfront::objectTargetFor(ppc, ppc) == Target::Object_LLVM_HOST);
  CHECK_FALSE(polyfront::objectTargetFor(riscv, x86));
  CHECK(polyfront::objectCPUFor(llvm::Triple("aarch64-unknown-linux-gnu"), "", x86) == "generic");
  CHECK(polyfront::objectCPUFor(x86, "", x86) == "native");
  CHECK(polyfront::objectCPUFor(x86, "haswell", x86) == "haswell");
  CHECK(polyfront::objectTargetsCompatible(llvm::Triple("aarch64-apple-darwin"), llvm::Triple("arm64-apple-macosx26.0.0")));
  CHECK_FALSE(polyfront::objectTargetsCompatible(llvm::Triple("aarch64-unknown-linux-gnu"), llvm::Triple("arm64-apple-macosx26.0.0")));
  CHECK_FALSE(polyfront::objectTargetsCompatible(llvm::Triple("x86_64-unknown-linux"), llvm::Triple("x86_64-unknown-freebsd")));
  CHECK_FALSE(
      polyfront::objectTargetsCompatible(llvm::Triple("armv7-unknown-linux-gnueabi"), llvm::Triple("armv7-unknown-linux-gnueabihf")));

  llvm::LLVMContext context;
  llvm::Module consumer("consumer", context);
  consumer.setDataLayout("e-p:64:64");
  llvm::Function::Create(llvm::FunctionType::get(llvm::Type::getVoidTy(context), {llvm::PointerType::get(context, 0)}, false),
                         llvm::GlobalValue::ExternalLinkage, "entry", consumer);
  CHECK(polyfront::objectLayoutsCompatible(consumer, llvm::DataLayout("e-p:64:64-p270:32:32")));
  CHECK_FALSE(polyfront::objectLayoutsCompatible(consumer, llvm::DataLayout("e-p:32:32")));
}

TEST_CASE("package service resolves and compiles a typed package Sym") {
  using namespace polyast;
  using namespace polyast::dsl;
  const auto publicName = Sym({"bar", "transform"});
  const auto publicDecl = transform(publicName, "T");
  const auto implementation = variant(transform(Sym({"implementation", "transform_w4"}), "Element", 4), publicName);
  const auto remoteKernelName = Sym({"implementation", "transform_kernel"});
  const auto contextType = Type::Ptr(Type::IntU8(), TypeSpace::Global()).widen();
  const Named context("#context", contextType);
  auto remoteArgs = implementation.decl.args;
  const auto remoteExtent = ArgExtent::Elements(ArgSizeExpr::Param(3));
  remoteArgs[0] = remoteArgs[0].withBoundary(ArgBoundary(ArgAccess::Read(), remoteExtent));
  remoteArgs[1] = remoteArgs[1].withBoundary(ArgBoundary(ArgAccess::Write(), remoteExtent));
  remoteArgs.insert(remoteArgs.begin(), Arg(context, {}));
  const auto one = Term::IntU32Const(1).widen();
  const auto zero = Term::IntU32Const(0).widen();
  const auto remoteLaunch = Spec::RemoteLaunch(selectNamed(context).widen(), Term::Poison(Type::FnRef(remoteKernelName)).widen(), {}, one,
                                               one, one, zero, zero, zero, zero, {});
  const auto remoteImplementation = implementation.withDecl(implementation.decl.withArgs(remoteArgs))
                                        .withBody({let("launched") = Expr::SpecOp(remoteLaunch).widen(), ret()});
  const auto remoteKernel = Function(FunctionDecl(remoteKernelName, {}, {}, {}, {}, {}, Type::Unit0(), FunctionAffinity::Offload()),
                                     {ret()}, FunctionVisibility::Internal(), FunctionFpMode::Relaxed(), CallConvention::OffloadEntry());
  const auto pkg = Package(Interface(Sym({"foo"}), {publicDecl}, {}), polyfront::packageProgram({remoteImplementation, remoteKernel}, {}));
  const auto f32 = Type::Float32().widen();
  const auto f32Pointer = Type::Ptr(f32, TypeSpace::Global()).widen();
  const auto signature = InvokeSignature(publicName, {}, {}, {f32Pointer, f32Pointer, Type::IntS32()}, Type::Unit0());
  const auto request =
      PackageSymRequest(pkg, signature, {}, {}, {}, {"gpu"}, {PackageTypeSize(f32, 4)}, "__package_sym", PackageReturnConvention::Return());
  const auto compiled = polyfront::package::PolycClient::compileSym(request, {}, compiletime::Target::Object_LLVM_HOST, "native",
                                                                    {{compiletime::Target::Source_C_Metal1_0, "host"}}, 8);
  for (const auto &error : compiled.errors)
    UNSCOPED_INFO(error);
  REQUIRE(compiled);
  const auto &resolved = compiled.value->resolved;
  REQUIRE(resolved.program.entry);
  const auto &entry = *resolved.program.entry;
  REQUIRE_FALSE(entry.decl.args.empty());
  CHECK(entry.decl.args.front().named.symbol == "#context");
  CHECK(resolved.entryArgs
        == std::vector<PackageEntryArgBinding::Any>{PackageEntryArgBinding::Context(), PackageEntryArgBinding::CallValue(0),
                                                    PackageEntryArgBinding::CallValue(1), PackageEntryArgBinding::CallAddress(2)});
  CHECK(entry.collect_all<Spec::RemoteAlloc>().size() == 2);
  CHECK(entry.collect_all<Spec::RemoteMemcpy>().size() == 2);
  CHECK(entry.collect_all<Spec::RemoteFree>().size() == 2);
  CHECK(polyfront::package::validateResolvedSymProgram(request, resolved, signature.args, signature.rtn).empty());

  auto malformed = resolved;
  malformed.entryArgs[1] = PackageEntryArgBinding::CallValue(-1);
  CHECK_FALSE(polyfront::package::validateResolvedSymProgram(request, malformed, signature.args, signature.rtn).empty());

  const auto erasedResultPointer = Type::Ptr(Type::Nothing(), TypeSpace::Global()).widen();
  const auto concreteResultPointer = Type::Ptr(f32, TypeSpace::Global()).widen();
  const auto erasedRequest = request.withSignature(InvokeSignature(publicName, {}, {}, {f32}, Type::Nothing()))
                                 .withReturnConvention(PackageReturnConvention::OutParam(1));
  const auto erasedEntry = entry.withDecl(entry.decl.withArgs(
      {entry.decl.args.front(), Arg(Named("value", concreteResultPointer), {}), Arg(Named("result", concreteResultPointer), {})}));
  const auto erasedResolved =
      resolved.withProgram(resolved.program.withEntry(erasedEntry))
          .withEntryArgs({PackageEntryArgBinding::Context(), PackageEntryArgBinding::CallAddress(0), PackageEntryArgBinding::CallValue(1)});
  CHECK(polyfront::package::validateResolvedSymProgram(erasedRequest, erasedResolved, {f32, erasedResultPointer}, Type::Nothing()).empty());
  CHECK_FALSE(
      polyfront::package::validateResolvedSymProgram(erasedRequest, erasedResolved, {f32, concreteResultPointer}, Type::Nothing()).empty());

  CHECK_FALSE(compiled.value->hostObject.empty());
  REQUIRE(compiled.value->remoteObjects.size() == 1);
  CHECK_FALSE(compiled.value->remoteObjects.front().moduleImage.empty());
}

TEST_CASE("package-service wire envelopes reject a stale fingerprint") {
  using namespace polyast;
  const auto request = PackageLinkRequest(Interface(Sym({"foo"}), {}, {}), {}, {});
  auto bytes = packagelinkrequest_to_msgpack(request);
  auto hash = bytes.end();
  for (size_t i = 0; i + 32 <= bytes.size(); ++i) {
    if (bytes | aspartame::slice(i, i + 32) | aspartame::forall([](uint8_t value) { return std::isxdigit(value) != 0; })) {
      hash = bytes.begin() + static_cast<ptrdiff_t>(i);
      break;
    }
  }
  REQUIRE(hash != bytes.end());
  *hash = *hash == static_cast<uint8_t>('0') ? static_cast<uint8_t>('1') : static_cast<uint8_t>('0');
  CHECK_THROWS(packagelinkrequest_from_msgpack(bytes));
}

TEST_CASE("Program and Package payloads retain their persistent wire schemas") {
  using namespace polyast;
  const auto program = polyfront::packageProgram({}, {});
  const auto programBytes = hashed_program_to_msgpack(program);
  const auto packageBytes = package_to_msgpack(Package(Interface(Sym({"foo"}), {}, {}), program));
  CHECK(std::holds_alternative<std::string>(decodePackage(programBytes.data(), programBytes.data() + programBytes.size())));
  CHECK(std::holds_alternative<std::string>(decodeHashedProgram(packageBytes.data(), packageBytes.data() + packageBytes.size())));
}
