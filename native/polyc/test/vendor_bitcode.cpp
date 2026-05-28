#include <fstream>

#include "llvm/ADT/SmallString.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/raw_ostream.h"

#include "catch2/catch_all.hpp"

#include "backend/llvmc.h"

using namespace polyregion::backend;

namespace {

std::string scratchDir(const std::string &suffix) {
  llvm::SmallString<256> root;
  llvm::sys::path::system_temp_directory(/*ErasedOnReboot=*/true, root);
  const auto pid = std::to_string(llvm::sys::Process::getProcessId());
  llvm::sys::path::append(root, "polyc-vendor-bitcode-" + pid + "-" + suffix);
  llvm::sys::fs::remove_directories(root);
  if (auto ec = llvm::sys::fs::create_directories(root)) FAIL("scratchDir create_directories failed: " << ec.message());
  return root.str().str();
}

std::string joined(const std::string &dir, llvm::StringRef name) {
  llvm::SmallString<256> p(dir);
  llvm::sys::path::append(p, name);
  return p.str().str();
}

} // namespace

TEST_CASE("findInDirs picks the first match", "[vendor-bitcode]") {
  auto a = scratchDir("first-match-a");
  auto b = scratchDir("first-match-b");
  std::ofstream(joined(b, "libdevice.10.bc")) << "stub";
  CHECK(llvmc::findInDirs("libdevice.10.bc", {a, b}) == joined(b, "libdevice.10.bc"));
}

TEST_CASE("findInDirs returns empty when no dir contains the file", "[vendor-bitcode]") {
  auto a = scratchDir("no-match");
  CHECK(llvmc::findInDirs("missing.bc", {a}).empty());
}

TEST_CASE("findInDirs skips empty dir entries", "[vendor-bitcode]") {
  auto a = scratchDir("skip-empty");
  std::ofstream(joined(a, "libdevice.10.bc")) << "stub";
  const llvm::StringRef empty;
  CHECK(llvmc::findInDirs("libdevice.10.bc", {empty, a}) == joined(a, "libdevice.10.bc"));
}

TEST_CASE("linkVendorBitcodeFile merges definitions into target module", "[vendor-bitcode]") {
  auto dir = scratchDir("link");
  const auto donorPath = joined(dir, "donor.bc");

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
