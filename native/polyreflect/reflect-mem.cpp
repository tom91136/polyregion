#include <unordered_set>

#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/LazyCallGraph.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/MemoryLocation.h"
#include "llvm/Analysis/ModuleSummaryAnalysis.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/AbstractCallSite.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/ConstantFold.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Value.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/LoopSimplify.h"
#include "llvm/Transforms/Utils/ScalarEvolutionExpander.h"

#include "aspartame/all.hpp"

#include "polyregion/interval.hpp"
#include "polyregion/llvm_dyn.hpp"
#include "polyregion/llvm_ir.hpp"
#include "reflect-mem.h"

#include <stack>

namespace {

using namespace aspartame;
using namespace polyregion;

const llvm::ConstantInt *getConstantInt(llvm::Value *v, const llvm::DataLayout &DL) {
  if (auto *CI = dyn_cast<llvm::ConstantInt>(v)) return CI;
  if (auto *CE = dyn_cast<llvm::Constant>(v)) {
    if (llvm::Constant *C = llvm::ConstantFoldConstant(CE, DL)) return dyn_cast<llvm::ConstantInt>(C);
  }
  if (auto *Inst = dyn_cast<llvm::Instruction>(v)) {
    if (llvm::Constant *C = llvm::ConstantFoldInstruction(Inst, DL)) return dyn_cast<llvm::ConstantInt>(C);
  }
  return {};
}

struct MemAccess {
  enum class Kind : char { ReadOnly = 'R', WriteOnly = 'W', ReadWrite = 'B' };

  Kind kind;
  llvm::Value *origin;
  llvm::Value *unitInBytes;
  llvm::Value *sizeInBytes;

  static Kind combine(const Kind &l, const Kind &r) {
    if (l == Kind::ReadOnly && r == Kind::ReadOnly) return Kind::ReadOnly;
    if (l == Kind::WriteOnly && r == Kind::WriteOnly) return Kind::WriteOnly;
    return Kind::ReadWrite;
  }

  static std::optional<MemAccess> resolve(llvm::LLVMContext &C, llvm::Instruction *I) {

    const bool isRead = I->mayReadFromMemory();
    const bool isWrite = I->mayWriteToMemory();
    if (!(isRead || isWrite)) return {}; // doesn't touch memory

    auto access = Kind::ReadWrite;
    if (isRead && !isWrite) access = Kind::ReadOnly;
    if (!isRead && isWrite) access = Kind::WriteOnly;

    const auto intConst = [&](const size_t size, const size_t value) {
      const auto ty = llvm::IntegerType::getIntNTy(C, size);
      return llvm::ConstantInt::get(ty, llvm::APInt(ty->getBitWidth(), value));
    };

    const auto singleUnit = [&](llvm::Value *operand, const llvm::Type *tpe) {
      const auto unitSize = intConst(64, tpe->getScalarSizeInBits() / 8);
      return MemAccess{
          .kind = access,
          .origin = operand,
          .unitInBytes = unitSize, //
          .sizeInBytes = unitSize  // unitSize * 1
      };
    };
    // XXX GEP doesn't actually have memory effects
    if (auto *LI = llvm::dyn_cast<llvm::LoadInst>(I)) return singleUnit(LI->getPointerOperand(), LI->getAccessType());
    else if (auto *SI = llvm::dyn_cast<llvm::StoreInst>(I)) return singleUnit(SI->getPointerOperand(), SI->getAccessType());
    else if (auto *ARMWI = llvm::dyn_cast<llvm::AtomicRMWInst>(I)) return singleUnit(ARMWI->getPointerOperand(), ARMWI->getAccessType());
    else if (auto *ACXI = llvm::dyn_cast<llvm::AtomicCmpXchgInst>(I)) return singleUnit(ACXI->getPointerOperand(), ACXI->getAccessType());
    else if (auto *VAI = llvm::dyn_cast<llvm::VAArgInst>(I)) return singleUnit(VAI->getPointerOperand(), VAI->getAccessType());
    else {
      if (const auto *CI = llvm::dyn_cast<llvm::CallInst>(I)) {
        switch (CI->getIntrinsicID()) {
          case llvm::Intrinsic::memcpy: [[fallthrough]];
          case llvm::Intrinsic::memmove: [[fallthrough]];
          case llvm::Intrinsic::memset: {
            return MemAccess{.kind = access,                 //
                             .origin = CI->getArgOperand(0), //
                             .unitInBytes = intConst(64, 1), //
                             .sizeInBytes = CI->getArgOperand(2)};
          }
          default: break;
        }
      }
    }
    return {};
  }
};

class CoarseGrainedTaintAnalysis {

  std::unordered_set<llvm::Value *> tainted;
  std::stack<llvm::Value *> workList;
  bool debug = false;

  bool isTainted(llvm::Value *V) const { return tainted.count(V); }

  bool isReturnTainted(llvm::Function *F) const {
    for (const auto &BB : *F) {
      if (const auto RI = llvm::dyn_cast<llvm::ReturnInst>(BB.getTerminator())) {
        if (RI->getReturnValue() && isTainted(RI->getReturnValue())) return true;
      }
    }
    return false;
  }

  void taint(llvm::Value *V) {
    if (tainted.insert(V).second) {
      workList.push(V);
      if (debug) {
        if (const auto I = llvm::dyn_cast<llvm::Instruction>(V))
          llvm::errs() << "[CGTA]       marking ins `" << *I << "` (from: " << I->getFunction()->getName() << ")\n";
        else if (const auto A = llvm::dyn_cast<llvm::Argument>(V))
          llvm::errs() << "[CGTA]       marking arg `" << *A << "` (from: " << A->getParent()->getName() << ")\n";
        else llvm::errs() << "[CGTA]       marking val`" << *V << "`\n";
      }
    }
  }

  static llvm::Value *unwrapAll(llvm::Value *V) {
    if (const auto CE = llvm::dyn_cast<llvm::ConstantExpr>(V)) {
      return unwrapAll(CE->getAsInstruction());
    }
    return V;
  }

  void propagateForward() {
    using Deferred = std::unordered_multimap<llvm::Function *, llvm::CallBase *>;

    const auto forwards = [&](Deferred &deferred) -> bool {
      while (!workList.empty()) {
        llvm::Value *V = workList.top();
        workList.pop();
        if (debug) llvm::errs() << "[CGTA]   propagating value `" << *V << "`\n";
        // 1. spanning propagation
        if (debug) llvm::errs() << "[CGTA]     spanning:\n";
        llvm_shared::visitDyn0(
            V,                                                                  //
            [&](llvm::GetElementPtrInst *I) { taint(I->getPointerOperand()); }, //
            [&](llvm::LoadInst *I) { taint(I->getPointerOperand()); },          //
            [&](llvm::StoreInst *I) {
              taint(I->getPointerOperand());
              if (I->getValueOperand()->getType()->isPointerTy()) taint(I->getValueOperand());
            },                                                                   //
            [&](llvm::ExtractValueInst *I) { taint(I->getAggregateOperand()); }, //
            [&](llvm::InsertValueInst *I) { taint(I->getAggregateOperand()); },  //
            [&](llvm::IntToPtrInst *I) { taint(I->getOperand(0)); },             //
            [&](llvm::PtrToIntInst *I) { taint(I->getOperand(0)); },             //
            [&](llvm::PHINode *I) {
              for (auto &U : I->incoming_values())
                taint(U.get());
            },
            [&](llvm::SelectInst *I) {
              taint(I->getTrueValue());
              taint(I->getFalseValue());
            });
        // 2. forwarding propagation
        for (llvm::User *U : V->users()) {
          if (debug) llvm::errs() << "[CGTA]     uses: `" << *U << "`\n";
          //  XXX do not taint the unwrapped instruction directly as it may be a constant
          llvm_shared::visitDyn0( //
              U,                  //
              [&](llvm::ConstantExpr *C) {
                // XXX do not use C->getAsInstruction() as it will insert it into the module!
                switch (C->getOpcode()) {
                  case llvm::Instruction::GetElementPtr: [[fallthrough]];
                  case llvm::Instruction::IntToPtr: [[fallthrough]];
                  case llvm::Instruction::PtrToInt: [[fallthrough]];
                  case llvm::Instruction::BitCast: [[fallthrough]];
                  case llvm::Instruction::PHI: [[fallthrough]];
                  case llvm::Instruction::Select: [[fallthrough]];
                  case llvm::Instruction::InsertValue: [[fallthrough]];
                  case llvm::Instruction::ExtractValue: [[fallthrough]];
                  case llvm::Instruction::Ret: [[fallthrough]];
                  case llvm::Instruction::Load: [[fallthrough]];
                  case llvm::Instruction::Store: taint(C);
                  default: break;
                }
              },                                            //
              [&](llvm::GetElementPtrInst *) { taint(U); }, //
              [&](llvm::IntToPtrInst *) { taint(U); },      //
              [&](llvm::PtrToIntInst *) { taint(U); },      //
              [&](llvm::BitCastInst *) { taint(U); },       //
              [&](llvm::PHINode *) { taint(U); },           //
              [&](llvm::SelectInst *) { taint(U); },        //
              [&](llvm::InsertValueInst *) { taint(U); },   //
              [&](llvm::ExtractValueInst *) { taint(U); },  //
              [&](llvm::ReturnInst *) { taint(U); },        //
              [&](llvm::LoadInst *) { taint(U); },          //
              [&](llvm::StoreInst *) { taint(U); },         //
              [&](llvm::CallBase *CB) {
                // XXX it's impossible to get a CallInst in a ConstantExpr, so this should be safe
                auto F = CB->getCalledFunction();
                if (!F) return; // XXX handle indirect calls later
                bool defer = false;
                for (size_t i = 0; i < CB->arg_size(); ++i) {
                  if (!isTainted(CB->getArgOperand(i))) continue;
                  if (i < F->arg_size()) {
                    taint(F->getArg(i));
                    defer = true;
                  }
                }
                if (defer && F->willReturn() && !F->getReturnType()->isVoidTy()) deferred.emplace(F, CB);
              });
        }
      }
      return !deferred.empty();
    };
    Deferred deferred;
    while (forwards(deferred)) {
      for (auto &[F, CB] : deferred) {
        if (isReturnTainted(F)) taint(CB);
      }
      deferred.clear();
    }
  }

public:
  static std::unordered_set<llvm::Value *> propagate(const std::vector<llvm::Function *> &Roots, //
                                                     const std::vector<llvm::Value *> &Values,   //
                                                     const bool debug = false) {
    CoarseGrainedTaintAnalysis t;
    t.debug = debug;
    std::unordered_set<llvm::Function *> visited;
    std::stack<llvm::Function *> workList;
    for (const auto R : Roots)
      workList.emplace(R);
    for (const auto V : Values)
      t.taint(V);
    t.propagateForward();
    while (!workList.empty()) {
      auto F = workList.top();
      workList.pop();
      if (!visited.insert(F).second) continue;
      if (debug) llvm::errs() << "[CGTA] function: `" << F->getName() << "`\n";
      for (auto &arg : F->args()) {
        if (!t.isTainted(&arg)) continue;
        if (debug) llvm::errs() << "[CGTA]   tainted arg: `" << arg << "`\n";
        for (const auto &U : F->uses()) {
          if (auto ACS = llvm::AbstractCallSite(&U)) {
            t.taint(ACS.getCallArgOperand(arg.getArgNo()));
            t.propagateForward();
            const auto call = ACS.getInstruction();
            if (t.isReturnTainted(call->getFunction())) t.taint(call);
            if (debug) llvm::errs() << "[CGTA]   push next: `" << call->getFunction()->getName() << "`\n";
            workList.push(call->getFunction());
          } else if (debug) {
            llvm::errs() << "[CGTA]   unknown use (failed ACS): \n";
            U->print(llvm::errs());
            llvm::errs() << "\n";
          }
        }
      }
    }
    return t.tainted;
  }
};

struct Target {
  MemAccess::Kind access;
  llvm::Instruction *insertPoint;
  llvm::Value *origin;
  llvm::Value *sizeInBytes;
  llvm::Value *unitInBytes;
};

struct CoalescedTarget {
  MemAccess::Kind access;
  llvm::Instruction *insertPoint;
  llvm::Value *origin;
  CoalescedTarget operator+(const CoalescedTarget &that) const {
    return CoalescedTarget{.access = MemAccess::combine(access, that.access), //
                           .insertPoint = !insertPoint->comesBefore(that.insertPoint) ? insertPoint : that.insertPoint,
                           .origin = origin};
  }
};

void callMap(llvm::IRBuilder<> &B, llvm::Module &M, llvm::Value *origin, llvm::Value *sizeInBytes, llvm::Value *unitInBytes,
             MemAccess::Kind kind) {
  const auto mapFnTy = llvm::FunctionType::get(
      B.getVoidTy(), llvm::ArrayRef<llvm::Type *>{B.getPtrTy(), B.getIntPtrTy(M.getDataLayout()), B.getIntPtrTy(M.getDataLayout())}, false);
  const auto fn = M.getOrInsertFunction(
      [&]() {
        switch (kind) {
          case MemAccess::Kind::ReadOnly: return "polyrt_map_read";
          case MemAccess::Kind::WriteOnly: return "polyrt_map_write";
          case MemAccess::Kind::ReadWrite: [[fallthrough]];
          default: return "polyrt_map_readwrite";
        }
      }(),
      mapFnTy);
  B.CreateCall(fn, {origin, sizeInBytes, unitInBytes});
}

void insertMapCalls(llvm::Module &M, //
                    llvm::FunctionAnalysisManager &FAM, std::vector<std::pair<llvm::Function *, std::vector<llvm::Value *>>> groups,
                    const bool verbose) {

  if (verbose) llvm::errs() << "[ReflectMemPass] Inserting map calls for " << groups.size() << " functions\n";

  std::unordered_map<llvm::BasicBlock *, std::vector<Target>> targets;
  for (auto &[F, Vs] : groups) {
    // TODO we need to run AA to group pointers that alias, with the following possible outcomes:
    //  a. Single group: ideal outcome
    //  b. More than one group:
    //    1. A single group has direct use from the argument, that group wins
    //    2. Multiple groups have direct use from argument, emit error
    if (verbose) llvm::errs() << "[ReflectMemPass] In " << (F ? F->getName() : llvm::StringRef("(top-level)")) << "\n";
    for (auto &V : Vs) {
      auto I = llvm::dyn_cast<llvm::Instruction>(V);
      if (!I) continue;

      const auto access = MemAccess::resolve(M.getContext(), I);
      if (!access) continue;
      auto &SE = FAM.getResult<llvm::ScalarEvolutionAnalysis>(*F);

      const auto emplaceNonSCEV = [&](const std::string &prefix) {
        if (verbose)
          llvm::errs() << "[ReflectMemPass]   " << static_cast<char>(access->kind) << " " << prefix << "@`" << *I << "`"
                       << " origin=`" << *access->origin << "`"
                       << " size=`" << *access->sizeInBytes << "`"
                       << " unit=`" << *access->unitInBytes << "`"
                       << "\n";
        targets[I->getParent()].emplace_back(Target{
            .access = access->kind,             //
            .insertPoint = I,                   //
            .origin = access->origin,           //
            .sizeInBytes = access->sizeInBytes, //
            .unitInBytes = access->unitInBytes, //
        });
      };

      if (auto *AddRec = llvm::dyn_cast_if_present<llvm::SCEVAddRecExpr>(SE.getSCEV(access->origin))) {
        if (!SE.hasLoopInvariantBackedgeTakenCount(AddRec->getLoop())) {
          emplaceNonSCEV("SCEV");
          continue;
        }
        const auto scLB = AddRec->getStart();
        const auto scStride = AddRec->getStepRecurrence(SE);
        const auto scTripCount = SE.getTripCountFromExitCount(
            /*ExitCount*/ SE.getExitCount(AddRec->getLoop(), AddRec->getLoop()->getExitingBlock()),
            // XXX 64 bit will overflow if trip count is UINT_MAX, LLVM defaults to a 65 bit APInt but that causes mulExpr to fail (!!)
            /*EvalTy*/ llvm::IntegerType::getIntNTy(M.getContext(), 64),
            /*L*/ AddRec->getLoop());
        const auto scUB = SE.getAddExpr(scLB, SE.getMulExpr(scStride, scTripCount));
        const auto predecessor = AddRec->getLoop()->getLoopPredecessor()->getTerminator();
        llvm::SCEVExpander Expander(SE, M.getDataLayout(), "scev");
        Expander.setInsertPoint(predecessor);
        const auto lb = Expander.expandCodeFor(scLB);
        const auto ub = Expander.expandCodeFor(scUB);
        llvm::IRBuilder<> B(predecessor);
        const auto ptrDiff = B.CreateSub(B.CreatePtrToInt(ub, B.getInt64Ty()), B.CreatePtrToInt(lb, B.getInt64Ty()));
        if (verbose)
          llvm::errs() << "[ReflectMemPass]   " << static_cast<char>(access->kind) << " InvariantSCEV@`" << *predecessor << "`"
                       << " origin=`" << *lb << "`"
                       << " size=`" << *ptrDiff << "`"
                       << " unit=`" << *access->unitInBytes << "`"
                       << "\n";
        targets[I->getParent()].emplace_back(Target{
            .access = access->kind,     //
            .insertPoint = predecessor, //
            .origin = lb,               //
            .sizeInBytes = ptrDiff,     //
            .unitInBytes = access->unitInBytes,
        });
      } else emplaceNonSCEV("Direct");
    }
  }

  for (auto &[BB, ts] : targets) {
    std::vector<Target> independent;
    IntervalMap<CoalescedTarget> coalesced;
    for (auto &t : ts) {
      int64_t offset = -1;
      const auto base = GetPointerBaseWithConstantOffset(t.origin, offset, M.getDataLayout());
      if (const auto constSize = getConstantInt(t.sizeInBytes, M.getDataLayout()); offset != -1 && constSize) {
        coalesced.insert({offset, constSize->getSExtValue()},
                         CoalescedTarget{.access = t.access, .insertPoint = t.insertPoint, .origin = base},
                         [](const CoalescedTarget &l, const CoalescedTarget &r) { return l + r; });
        independent.emplace_back(t);
      } else independent.emplace_back(t);
    }

    llvm::IRBuilder<> B(M.getContext());
    for (const auto &[range, ct] : coalesced.map) {
      if (verbose)
        llvm::errs() << "[ReflectMemPass] Insert coalesced: [" << range.offset << ", " << range.end() << ") ["
                     << static_cast<char>(ct.access) << "] `" << *ct.origin << "`\n";
      B.SetInsertPoint(ct.insertPoint);
      callMap(B, M, //
              B.CreateConstInBoundsGEP1_64(B.getInt8Ty(), ct.origin, range.offset), B.getInt64(range.size), B.getInt64(1), ct.access);
    }
    for (auto &t : independent) {

      if (verbose) llvm::errs() << "[ReflectMemPass] Insert independent: [" << static_cast<char>(t.access) << "] `" << *t.origin << "`\n";
      B.SetInsertPoint(t.insertPoint);
      callMap(B, M, t.origin, t.sizeInBytes, t.unitInBytes, t.access);
    }
  }
}

bool runSplice(llvm::Module &M, llvm::FunctionAnalysisManager &FAM, const bool verbose) {
  std::vector<llvm::Function *> roots;
  std::vector<llvm::Value *> values;
  llvm_shared::findValuesWithStringAnnotations(M, [&](llvm::Function &F, llvm::Value *V, llvm::StringRef Annotation) {
    if (!Annotation.starts_with("polyreflect-track")) return;
    const auto U = llvm::getUnderlyingObject(V, 0);
    if (auto GV = llvm::dyn_cast<llvm::GlobalVariable>(U)) {
      if (verbose) llvm::errs() << "[ReflectMemPass] Found annotated (GV) in " << F.getName() << ": `" << *GV << "`\n";
      roots.emplace_back(&F);
      values.emplace_back(GV);
    } else if (auto I = llvm::dyn_cast<llvm::Instruction>(U)) {
      if (verbose) llvm::errs() << "[ReflectMemPass] Found annotated (Ins) in " << F.getName() << ": `" << *I << "`\n";
      roots.emplace_back(&F);
      values.emplace_back(I);
    } else {
      if (verbose) llvm::errs() << "[ReflectMemPass] Skipping unknown annotated in " << F.getName() << ": `" << *V << "`\n";
    }
  });

  if (verbose) {
    llvm::errs() << "[ReflectMemPass] Found " << values.size() << " annotated values in module " << M.getName() << "\n";
    llvm::errs() << "[ReflectMemPass] Running CGTA\n";
  }
  const auto tainted = CoarseGrainedTaintAnalysis::propagate(roots, values, false);

  if (verbose) llvm::errs() << "[ReflectMemPass] CGTA yielded " << tainted.size() << " tainted values\n";

  const std::vector<std::pair<llvm::Function *, std::vector<llvm::Value *>>> groups =
      tainted //
      ^ group_by([](auto &V) -> llvm::Function * {
          if (auto I = llvm::dyn_cast<llvm::Instruction>(V)) return I->getFunction();
          return {};
        })                                                                            //
      ^ to_vector()                                                                   //
      ^ sort_by([](auto &F, auto &) { return F ? F->getName() : llvm::StringRef{}; }) //
      ^ map([](auto &F, auto &Ins) { return std::pair{F, Ins ^ to_vector()}; });

  if (verbose) {
    llvm::errs() << "[ReflectMemPass] CGTA summary:\n";
    for (auto &[F, Ins] : groups) {
      llvm::errs() << "[ReflectMemPass]   - " << (F ? F->getName() : llvm::StringRef("(top-level)")) << "\n";
      for (auto &I : Ins)
        llvm::errs() << "[ReflectMemPass]     `" << *I << "`\n";
    }
  }

  insertMapCalls(M, FAM, groups, verbose);

  if (verbose) llvm::errs() << "[ReflectMemPass] Completed for module " << M.getName() << "\n";
  if (verbose) llvm::errs() << M << "\n";
  return !groups.empty();
}

} // namespace

polyreflect::ReflectMemPass::ReflectMemPass(const bool verbose) : verbose(verbose) {}
llvm::PreservedAnalyses polyreflect::ReflectMemPass::run(llvm::Module &M, llvm::ModuleAnalysisManager &MAM) {
  auto &FAM = MAM.getResult<llvm::FunctionAnalysisManagerModuleProxy>(M).getManager();
  if (runSplice(M, FAM, verbose)) return llvm::PreservedAnalyses::none();
  return llvm::PreservedAnalyses::all();
}