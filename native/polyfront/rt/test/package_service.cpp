#include <algorithm>
#include <cctype>

#include <catch2/catch_test_macros.hpp>

#include "polyfront/options_backend.hpp"
#include "polyfront/package_program.hpp"
#include "polyfront/polyc_client.hpp"
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
  for (auto it = bytes.begin(); it != bytes.end(); ++it) {
    if (static_cast<size_t>(bytes.end() - it) < 32) break;
    if (std::all_of(it, it + 32, [](uint8_t value) { return std::isxdigit(value) != 0; })) {
      hash = it;
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
