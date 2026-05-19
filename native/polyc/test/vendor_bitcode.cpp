#include <filesystem>
#include <fstream>

#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"

#include "catch2/catch_all.hpp"

#include "backend/llvmc.h"

using namespace polyregion::backend;

namespace {

std::filesystem::path scratchDir(const std::string &suffix) {
  auto root = std::filesystem::temp_directory_path() / ("polyc-vendor-bitcode-" + suffix);
  std::filesystem::remove_all(root);
  std::filesystem::create_directories(root);
  return root;
}

} // namespace

TEST_CASE("findInDirs picks the first match", "[vendor-bitcode]") {
  auto a = scratchDir("first-match-a");
  auto b = scratchDir("first-match-b");
  std::ofstream(b / "libdevice.10.bc") << "stub";
  const std::string aStr = a.string(), bStr = b.string();
  CHECK(llvmc::findInDirs("libdevice.10.bc", {aStr, bStr}) == (b / "libdevice.10.bc").string());
}

TEST_CASE("findInDirs returns empty when no dir contains the file", "[vendor-bitcode]") {
  auto a = scratchDir("no-match");
  const std::string aStr = a.string();
  CHECK(llvmc::findInDirs("missing.bc", {aStr}).empty());
}

TEST_CASE("findInDirs skips empty dir entries", "[vendor-bitcode]") {
  auto a = scratchDir("skip-empty");
  std::ofstream(a / "libdevice.10.bc") << "stub";
  const llvm::StringRef empty;
  const std::string aStr = a.string();
  CHECK(llvmc::findInDirs("libdevice.10.bc", {empty, aStr}) == (a / "libdevice.10.bc").string());
}

TEST_CASE("linkVendorBitcodeFile merges definitions into target module", "[vendor-bitcode]") {
  auto dir = scratchDir("link");
  const auto donorPath = (dir / "donor.bc").string();

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
