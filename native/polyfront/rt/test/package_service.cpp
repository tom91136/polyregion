#include "polyfront/package_service.hpp"

#include <algorithm>
#include <cctype>

#include <catch2/catch_test_macros.hpp>

#include "polyfront/options_backend.hpp"
#include "polyfront/package_program.hpp"
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

polyast::Package packaged(const polyast::FunctionDecl &declaration, polyast::Function implementation) {
  using namespace polyast;
  return Package(Interface(Sym({"foo"}), {declaration}, {}), polyfront::packageProgram({std::move(implementation)}, {}));
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

TEST_CASE("package service resolves a typed package Sym") {
  using namespace polyast;
  const auto publicName = Sym({"bar", "transform"});
  const auto publicDecl = transform(publicName, "T");
  const auto implementation = variant(transform(Sym({"implementation", "transform_w4"}), "Element", 4), publicName);
  const auto pkg = packaged(publicDecl, implementation);
  const auto f32 = Type::Float32().widen();
  const auto f32Pointer = Type::Ptr(f32, TypeSpace::Global()).widen();
  const auto signature = InvokeSignature(publicName, {}, {}, {f32Pointer, f32Pointer, Type::IntS32()}, Type::Unit0());
  const auto request =
      PackageSymRequest(pkg, signature, {}, {}, {}, {"gpu"}, {PackageTypeSize(f32, 4)}, "__package_sym", PackageReturnConvention::Return());
  const auto result = polyfront::package::PackageService::resolveSym(request);
  REQUIRE(result);
  REQUIRE(result.value->program.entry);
  const auto &entry = *result.value->program.entry;
  REQUIRE_FALSE(entry.decl.args.empty());
  CHECK(entry.decl.args.front().named.symbol == "#context");
  CHECK(result.value->entryArgs
        == std::vector<PackageEntryArgBinding::Any>{PackageEntryArgBinding::Context(), PackageEntryArgBinding::CallValue(0),
                                                    PackageEntryArgBinding::CallValue(1), PackageEntryArgBinding::CallAddress(2)});
  CHECK(entry.collect_all<Spec::RemoteAlloc>().size() == 2);
  CHECK(entry.collect_all<Spec::RemoteMemcpy>().size() == 2);
  CHECK(entry.collect_all<Spec::RemoteFree>().size() == 2);
  CHECK(polyfront::package::validateResolvedSymProgram(request, *result.value, signature.args, signature.rtn).empty());

  auto malformed = *result.value;
  malformed.entryArgs[1] = PackageEntryArgBinding::CallValue(-1);
  CHECK_FALSE(polyfront::package::validateResolvedSymProgram(request, malformed, signature.args, signature.rtn).empty());

  const auto erasedResultPointer = Type::Ptr(Type::Nothing(), TypeSpace::Global()).widen();
  const auto concreteResultPointer = Type::Ptr(f32, TypeSpace::Global()).widen();
  const auto erasedRequest = request.withSignature(InvokeSignature(publicName, {}, {}, {f32}, Type::Nothing()))
                                 .withReturnConvention(PackageReturnConvention::OutParam(1));
  const auto erasedEntry = entry.withDecl(entry.decl.withArgs(
      {entry.decl.args.front(), Arg(Named("value", concreteResultPointer), {}), Arg(Named("result", concreteResultPointer), {})}));
  const auto erasedResolved =
      result.value->withProgram(result.value->program.withEntry(erasedEntry))
          .withEntryArgs({PackageEntryArgBinding::Context(), PackageEntryArgBinding::CallAddress(0), PackageEntryArgBinding::CallValue(1)});
  CHECK(polyfront::package::validateResolvedSymProgram(erasedRequest, erasedResolved, {f32, erasedResultPointer}, Type::Nothing()).empty());
  CHECK_FALSE(
      polyfront::package::validateResolvedSymProgram(erasedRequest, erasedResolved, {f32, concreteResultPointer}, Type::Nothing()).empty());
}

TEST_CASE("package service transports a large linked package without truncation") {
  using namespace polyast;
  const auto publicName = Sym({"bar", "large"});
  const auto publicDecl = FunctionDecl(publicName, {}, {}, {}, {}, {}, Type::Unit0(), FunctionAffinity::Host());
  const auto implementationDecl = publicDecl.withName(Sym({"implementation", "large"}));
  std::vector<Function> functions;
  functions.reserve(4097);
  functions.emplace_back(implementationDecl, std::vector<Stmt::Any>{}, FunctionVisibility::Exported(), FunctionFpMode::Relaxed(),
                         CallConvention::RegularCall(), publicName);
  for (int i = 0; i < 4096; ++i) {
    const auto decl = publicDecl.withName(Sym({"helper", std::to_string(i)}));
    functions.emplace_back(decl, std::vector<Stmt::Any>{}, FunctionVisibility::Internal(), FunctionFpMode::Relaxed(),
                           CallConvention::RegularCall());
  }
  const auto request =
      PackageLinkRequest(Interface(Sym({"large"}), {publicDecl}, {}), {polyfront::packageProgram(std::move(functions), {})}, {});
  const auto result = polyfront::package::PackageService::linkPackage(request);
  REQUIRE(result);
  CHECK(result.value->program.functions.size() == 4097);
  CHECK(std::any_of(result.value->program.functions.begin(), result.value->program.functions.end(),
                    [](const auto &function) { return function.decl.name == Sym({"helper", "4095"}); }));
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
