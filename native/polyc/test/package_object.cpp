#include "llvm/AsmParser/Parser.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"

#include "catch2/catch_all.hpp"

#include "compiler.h"
#include "package_compiler.hpp"

using namespace polyregion;

namespace {

std::vector<int8_t> redundantBitcode() {
  static constexpr auto source = R"(
define i32 @sum(i32 %n) {
entry:
  %acc = alloca i32
  %i = alloca i32
  store i32 0, ptr %acc
  store i32 0, ptr %i
  br label %loop
loop:
  %iv = load i32, ptr %i
  %done = icmp sge i32 %iv, %n
  br i1 %done, label %exit, label %body
body:
  %a = load i32, ptr %acc
  %next = add i32 %a, %iv
  store i32 %next, ptr %acc
  %inc = add i32 %iv, 1
  store i32 %inc, ptr %i
  br label %loop
exit:
  %r = load i32, ptr %acc
  ret i32 %r
}
)";
  compiler::initialise();
  llvm::LLVMContext context;
  llvm::SMDiagnostic diagnostic;
  const auto module = llvm::parseAssemblyString(source, diagnostic, context);
  REQUIRE(module);
  llvm::SmallVector<char, 0> buffer;
  llvm::raw_svector_ostream stream(buffer);
  llvm::WriteBitcodeToFile(*module, stream);
  return {reinterpret_cast<const int8_t *>(buffer.data()), reinterpret_cast<const int8_t *>(buffer.data()) + buffer.size()};
}

} // namespace

TEST_CASE("host program bitcode is optimised at the requested level") {
  const auto bitcode = redundantBitcode();
  const auto target = compiletime::Target::Object_LLVM_HOST;
  const auto unoptimised = compiler::package::bitcodeToObject(bitcode, target, "native", compiletime::OptLevel::O0);
  const auto optimised = compiler::package::bitcodeToObject(bitcode, target, "native", compiletime::OptLevel::O3);
  REQUIRE_FALSE(unoptimised.empty());
  REQUIRE_FALSE(optimised.empty());
  CHECK(optimised.size() < unoptimised.size());
}
