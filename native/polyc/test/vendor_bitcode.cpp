#include <fstream>

#include "llvm/ADT/SmallString.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include "catch2/catch_all.hpp"

#include "backend/llvmc.h"

using namespace polyregion::backend;

namespace {

struct ScratchDir {
  std::string path;
  explicit ScratchDir(const std::string &suffix) {
    llvm::SmallString<256> dir;
    if (auto ec = llvm::sys::fs::createUniqueDirectory("polyc-vendor-bitcode-" + suffix, dir))
      FAIL("ScratchDir createUniqueDirectory failed: " << ec.message());
    path = dir.str().str();
  }
  ScratchDir(const ScratchDir &) = delete;
  ScratchDir &operator=(const ScratchDir &) = delete;
  ~ScratchDir() { llvm::sys::fs::remove_directories(path); }
};

std::string joined(const std::string &dir, llvm::StringRef name) {
  llvm::SmallString<256> p(dir);
  llvm::sys::path::append(p, name);
  return p.str().str();
}

} // namespace

TEST_CASE("findInDirs picks the first match", "[vendor-bitcode]") {
  const ScratchDir a("first-match-a");
  const ScratchDir b("first-match-b");
  std::ofstream(joined(b.path, "libdevice.10.bc")) << "stub";
  CHECK(llvmc::findInDirs("libdevice.10.bc", {a.path, b.path}) == joined(b.path, "libdevice.10.bc"));
}

TEST_CASE("findInDirs returns empty when no dir contains the file", "[vendor-bitcode]") {
  const ScratchDir a("no-match");
  CHECK(llvmc::findInDirs("missing.bc", {a.path}).empty());
}

TEST_CASE("findInDirs skips empty dir entries", "[vendor-bitcode]") {
  const ScratchDir a("skip-empty");
  std::ofstream(joined(a.path, "libdevice.10.bc")) << "stub";
  const llvm::StringRef empty;
  CHECK(llvmc::findInDirs("libdevice.10.bc", {empty, a.path}) == joined(a.path, "libdevice.10.bc"));
}

TEST_CASE("linkVendorBitcodeFile merges definitions into target module", "[vendor-bitcode]") {
  const ScratchDir dir("link");
  const auto donorPath = joined(dir.path, "donor.bc");

  {
    llvm::LLVMContext ctx;
    llvm::Module m("donor", ctx);
    auto *fty = llvm::FunctionType::get(llvm::Type::getFloatTy(ctx), {llvm::Type::getFloatTy(ctx)}, false);
    auto *fn = llvm::Function::Create(fty, llvm::Function::ExternalLinkage, "donor", &m);
    llvm::IRBuilder<> b(llvm::BasicBlock::Create(ctx, "entry", fn));
    b.CreateRet(fn->getArg(0));
    std::error_code ec;
    llvm::raw_fd_ostream out(donorPath, ec);
    REQUIRE_FALSE(ec);
    llvm::WriteBitcodeToFile(m, out);
  }

  llvm::LLVMContext ctx;
  llvm::Module target("target", ctx);
  auto *fty = llvm::FunctionType::get(llvm::Type::getFloatTy(ctx), {llvm::Type::getFloatTy(ctx)}, false);
  llvm::Function::Create(fty, llvm::Function::ExternalLinkage, "donor", &target);

  REQUIRE(llvmc::linkVendorBitcodeFile(target, donorPath));

  auto *fn = target.getFunction("donor");
  REQUIRE(fn != nullptr);
  CHECK_FALSE(fn->isDeclaration());
  CHECK(fn->size() == 1);
}

TEST_CASE("linkVendorBitcodeFile fails gracefully on missing path", "[vendor-bitcode]") {
  llvm::LLVMContext ctx;
  llvm::Module target("target", ctx);
  CHECK_FALSE(llvmc::linkVendorBitcodeFile(target, "/nonexistent/path/foo.bc"));
}
