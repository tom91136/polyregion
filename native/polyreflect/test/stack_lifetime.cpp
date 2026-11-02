#include "llvm/Analysis/ValueTracking.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Passes/PassBuilder.h"

#include <catch2/catch_test_macros.hpp>

#include "reflect-stack.h"

static constexpr auto IR = R"(
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture)

define i32 @fill(i32 %seed) {
entry:
  %a = alloca [32 x i32], align 4
  %b = alloca [32 x i32], align 4
  call void @llvm.lifetime.start.p0(i64 128, ptr %a)
  store i32 %seed, ptr %a, align 4
  %s = load i32, ptr %a, align 4
  call void @llvm.lifetime.end.p0(i64 128, ptr %a)
  %bp = getelementptr i8, ptr %b, i64 0
  call void @llvm.lifetime.start.p0(i64 128, ptr %b)
  store i32 %s, ptr %b, align 4
  %t = load i32, ptr %b, align 4
  call void @llvm.lifetime.end.p0(i64 128, ptr %bp)
  ret i32 %t
}
)";

TEST_CASE("lifetime markers of tracked allocas are dropped") {
  llvm::LLVMContext C;
  llvm::SMDiagnostic Err;
  auto M = llvm::parseAssemblyString(IR, Err, C);
  REQUIRE(M);

  llvm::ModuleAnalysisManager MAM;
  polyregion::polyreflect::ReflectStackPass Pass(false);
  Pass.run(*M, MAM);

  const auto *Fn = M->getFunction("fill");
  REQUIRE(Fn);

  std::size_t records = 0, markers = 0;
  for (const auto &I : llvm::instructions(*Fn)) {
    if (const auto *CB = llvm::dyn_cast<llvm::CallBase>(&I))
      if (const auto *Callee = CB->getCalledFunction(); Callee && Callee->getName() == "_rt_record") ++records;
    if (const auto *II = llvm::dyn_cast<llvm::IntrinsicInst>(&I))
      if (II->isLifetimeStartOrEnd() && llvm::isa<llvm::AllocaInst>(llvm::getUnderlyingObject(II->getArgOperand(0)))) ++markers;
  }
  CHECK(records == 2);
  CHECK(markers == 0);
}
