#include <string>
#include <system_error>

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include "aspartame/all.hpp"

#include "polyfront/package.hpp"
#include "polyfront/package_emit.hpp"
#include "polyfront/package_program.hpp"

#include "ast.h"
#include "polyast_codec.h"

using namespace aspartame;

int main(int argc, char **argv) {
  using namespace polyregion::polyast;
  using namespace polyregion::polyast::dsl;
  const bool writeInputs = argc == 3 && std::string(argv[1]) == "--write-package-inputs";
  if (argc == 3 && std::string(argv[1]) == "--remove-prefix") {
    llvm::SmallString<256> directory(argv[2]);
    llvm::sys::path::remove_filename(directory);
    const auto prefix = llvm::sys::path::filename(argv[2]);
    const auto path = directory.empty() ? std::string{"."} : directory.str().str();
    std::error_code ec;
    for (llvm::sys::fs::directory_iterator it(path, ec), end; it != end && !ec; it.increment(ec))
      if (llvm::sys::path::filename(it->path()).starts_with(prefix)) llvm::sys::fs::remove(it->path());
    return ec ? 6 : 0;
  }
  if (argc == 3 && std::string(argv[1]) == "--assert-no-prefix") {
    llvm::SmallString<256> directory(argv[2]);
    llvm::sys::path::remove_filename(directory);
    const auto prefix = llvm::sys::path::filename(argv[2]);
    const auto path = directory.empty() ? std::string{"."} : directory.str().str();
    std::error_code ec;
    for (llvm::sys::fs::directory_iterator it(path, ec), end; it != end && !ec; it.increment(ec))
      if (llvm::sys::path::filename(it->path()).starts_with(prefix)) return 7;
    return ec ? 6 : 0;
  }
  if (argc == 5 && std::string(argv[1]) == "--assert-function-substring-count") {
    const auto source = llvm::MemoryBuffer::getFile(argv[2]);
    if (!source) return 8;
    const auto *begin = reinterpret_cast<const uint8_t *>((*source)->getBufferStart());
    const auto *end = reinterpret_cast<const uint8_t *>((*source)->getBufferEnd());
    const auto program = hashed_program_from_msgpack(begin, end);
    const std::string needle = argv[3];
    const auto actual = program.functions ^ count([&](const auto &fn) { return fqcn(fn.decl.name) ^ contains_slice(needle); });
    const auto expected = std::stoul(argv[4]);
    if (actual != expected) {
      llvm::errs() << "Expected " << expected << " functions containing `" << needle << "`, found " << actual << '\n';
      return 9;
    }
    return 0;
  }
  if (argc != 2 && !writeInputs) return 2;

  const auto publicName = Sym({"bar", "increment"});
  const auto implementationName = Sym({"bar", "implementation", "increment"});
  const auto copyName = Sym({"bar", "copy"});
  const auto copyImplementationName = Sym({"bar", "implementation", "copy"});
  const auto applyName = Sym({"bar", "apply"});
  const auto applyImplementationName = Sym({"bar", "implementation", "apply"});
  const auto i32 = Type::IntS32().widen();
  const auto publicDecl = FunctionDecl(publicName, {}, {}, {Arg(Named("x", i32), {})}, {}, {}, i32, FunctionAffinity::Host());
  const auto implementationDecl =
      FunctionDecl(implementationName, {}, {}, {Arg(Named("x", i32), {})}, {}, {}, i32, FunctionAffinity::Host());
  const auto x = NamedBuilder(Named("x", i32));
  const auto implementation = Function(implementationDecl, {ret(call(Intr::Add(x, Term::IntS32Const(1).widen(), i32)))},
                                       FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), CallConvention::RegularCall());
  const auto i32p = Type::Ptr(i32, TypeSpace::Global()).widen();
  const auto copyExtent = ArgExtent::Elements(ArgSizeExpr::Param(2));
  const auto copyDecl =
      FunctionDecl(copyName, {}, {},
                   {Arg(Named("in", i32p), {}, ArgBoundary(ArgAccess::Read(), copyExtent)),
                    Arg(Named("out", i32p), {}, ArgBoundary(ArgAccess::Write(), copyExtent)), Arg(Named("n", i32), {}, {})},
                   {}, {}, Type::Unit0(), FunctionAffinity::Host());
  const auto copyImplementationDecl = copyDecl.withName(copyImplementationName);
  const auto copyImplementation =
      Function(copyImplementationDecl, {ret()}, FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), CallConvention::RegularCall());
  const auto t = Type::Var("T").widen();
  const auto op = Type::Exec({}, {t}, t).widen();
  const auto applyDecl = FunctionDecl(applyName, {Type::Var("T")}, {}, {Arg(Named("x", t), {}), Arg(Named("op", op), {})}, {}, {}, t,
                                      FunctionAffinity::Host());
  const auto element = Type::Var("Element").widen();
  const auto applyImplementationDecl =
      FunctionDecl(applyImplementationName, {Type::Var("Element"), Type::Var("Op")}, {},
                   {Arg(Named("x", element), {}), Arg(Named("op", Type::Var("Op")), {})}, {}, {}, element, FunctionAffinity::Host());
  const auto applyX = NamedBuilder(Named("x", element));
  const auto applyImplementation =
      Function(applyImplementationDecl, {ret(Expr::Invoke(Type::Var("Op"), {}, {}, {applyX}, element).widen())},
               FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), CallConvention::RegularCall());
  const auto combineName = Sym({"bar", "combine"});
  const auto combineImplementationName = Sym({"bar", "implementation", "combine"});
  const auto combineDecl =
      FunctionDecl(combineName, {Type::Var("T")}, {}, {Arg(Named("x", t), {}), Arg(Named("left", op), {}), Arg(Named("right", op), {})}, {},
                   {}, t, FunctionAffinity::Host());
  const auto combineImplementationDecl =
      FunctionDecl(combineImplementationName, {Type::Var("Element"), Type::Var("Left"), Type::Var("Right")}, {},
                   {Arg(Named("x", element), {}), Arg(Named("left", Type::Var("Left")), {}), Arg(Named("right", Type::Var("Right")), {})},
                   {}, {}, element, FunctionAffinity::Host());
  const auto first = NamedBuilder(Named("first", element));
  const auto combineImplementation = Function(combineImplementationDecl,
                                              {let("first") = Expr::Invoke(Type::Var("Left"), {}, {}, {applyX}, element).widen(),
                                               ret(Expr::Invoke(Type::Var("Right"), {}, {}, {first}, element).widen())},
                                              FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), CallConvention::RegularCall());
  const auto capableName = Sym({"bar", "capable_increment"});
  const auto capableImplementationDecl = implementationDecl.withName(Sym({"bar", "implementation", "capable_increment"}));
  const auto capableImplementation = implementation.withDecl(capableImplementationDecl);
  const auto capableDecl = publicDecl.withName(capableName);
  const auto remoteName = Sym({"bar", "remote_increment"});
  const auto remoteImplementationName = Sym({"bar", "implementation", "remote_increment"});
  const auto remoteKernelName = Sym({"bar", "implementation", "remote_increment_kernel"});
  const auto remoteDecl = publicDecl.withName(remoteName);
  const auto contextType = Type::Ptr(Type::IntU8(), TypeSpace::Global()).widen();
  const Named context("#context", contextType);
  const auto remoteImplementationDecl =
      implementationDecl.withName(remoteImplementationName).withArgs({Arg(context, {}), Arg(Named("x", i32), {})});
  const auto one = Term::IntU32Const(1).widen();
  const auto zero = Term::IntU32Const(0).widen();
  const auto remoteLaunch = Spec::RemoteLaunch(/*context*/ selectNamed(context).widen(),
                                               /*kernel*/ Term::Poison(Type::FnRef(remoteKernelName)).widen(),
                                               /*tpeArgs*/ {},
                                               /*gridX*/ one,
                                               /*gridY*/ one,
                                               /*gridZ*/ one,
                                               /*blockX*/ zero,
                                               /*blockY*/ zero,
                                               /*blockZ*/ zero,
                                               /*shmem*/ zero,
                                               /*args*/ {});
  const auto remoteImplementation =
      Function(remoteImplementationDecl,
               {let("launched") = Expr::SpecOp(remoteLaunch).widen(), ret(call(Intr::Add(x, Term::IntS32Const(1).widen(), i32)))},
               FunctionVisibility::Exported(), FunctionFpMode::Relaxed(), CallConvention::RegularCall());
  const auto remoteKernel = Function(FunctionDecl(remoteKernelName, {}, {}, {}, {}, {}, Type::Unit0(), FunctionAffinity::Offload()),
                                     {ret()}, FunctionVisibility::Internal(), FunctionFpMode::Relaxed(), CallConvention::OffloadEntry());
  const auto interface = Interface(Sym({"foo"}), {publicDecl, copyDecl, applyDecl, combineDecl, capableDecl, remoteDecl}, {});
  const auto program = polyregion::polyfront::packageProgram(
      {implementation.withImplements(publicName), copyImplementation.withImplements(copyName),
       applyImplementation.withImplements(applyName), combineImplementation.withImplements(combineName),
       capableImplementation.withImplements(capableName).withRequiredCapabilities({"demo"}),
       remoteImplementation.withImplements(remoteName), remoteKernel},
      {});

  if (writeInputs) {
    llvm::SmallString<256> directory(argv[2]);
    if (llvm::sys::fs::create_directories(directory)) return 3;
    const auto write = [&](llvm::StringRef name, const std::vector<uint8_t> &bytes) {
      llvm::SmallString<256> path(directory);
      llvm::sys::path::append(path, name);
      std::error_code error;
      llvm::raw_fd_ostream out(path, error, llvm::sys::fs::OF_None);
      if (error) return false;
      out.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
      out.close();
      return !out.has_error();
    };
    return write("interface.polyast", interface_to_msgpack(interface)) && write("program.polyast", hashed_program_to_msgpack(program)) ? 0
                                                                                                                                       : 5;
  }

  const auto emitted = polyregion::polyfront::package::linkAndPublish(PackageLinkRequest(interface, {program}, {}), argv[1]);
  if (emitted) return 0;
  llvm::errs() << (emitted.errors ^ mk_string("\n")) << '\n';
  return 4;
}
