#pragma once

#include "flang/Optimizer/Dialect/FIRAttr.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"

#include "aspartame/all.hpp"
#include "fmt/core.h"

#include "fexpr.h"
#include "ftypes.h"
#include "polyast.h"
#include "utils.h"

namespace polyregion::polyfc {

using namespace aspartame;

// a `do concurrent ... reduce(...)` rewritten for polyregion.core.transform_reduce: the loop lists its reduce operators, and the
// body's final store of each per-iteration accumulator into its host result slot names that accumulator's result field
inline constexpr llvm::StringLiteral CoreReduceAttr = "polyfc.core_reduce";
inline constexpr llvm::StringLiteral CoreReduceResultAttr = "polyfc.core_reduce_result";

struct Remapper {

  mlir::ModuleOp &M;
  mlir::DataLayout &L;
  mlir::Operation *perimeter;
  std::vector<polyast::Named> captureRoot;

  llvm::DenseMap<mlir::Value, FExpr> valuesLUT;
  llvm::DenseMap<mlir::Type, FType> typesLUT;
  llvm::MapVector<mlir::Value, polyast::Term::Select> captures;

  std::unordered_set<polyast::StructDef> syntheticDefs;
  std::unordered_map<polyast::Type::Struct, polyast::StructDef> defs;
  std::unordered_map<polyast::Type::Struct, std::variant<FBoxedMirror, FBoxedNoneMirror>> boxTypes;

  size_t loopSeq = 0;

  llvm::DenseMap<mlir::Value, polyast::Term::Select> aggLoadCache;

  std::vector<polyast::Stmt::Any> stmts;
  std::optional<polyast::Named> reduceResult;
  std::vector<polyast::Function> functions;
  // user procedures translated to polyast (recursive callees survive as fir.call -> Invoke); by mangled name
  std::unordered_map<std::string, polyast::Function> userFuncs;
  void handleFunc(mlir::func::FuncOp funcOp);
  polyast::Term::Select newVar(const std::variant<polyast::Expr::Any, polyast::Type::Any> &expr);
  polyast::Term::Any termOf(const polyast::Expr::Any &expr);

  Remapper(mlir::ModuleOp &M, mlir::DataLayout &L, mlir::Operation *perimeter, const std::vector<polyast::Named> &captureRoot);
  std::optional<FType> fTypeOf(const mlir::Type &type);
  mlir::Type resolveType(const polyast::Type::Any &tpe);
  polyast::StructLayout resolveLayout(const polyast::StructDef &def);
  polyast::Type::Any handleType(mlir::Type type, bool captureBoundary = false);
  FExpr handleValue(mlir::Value val, const std::optional<std::vector<polyast::Named>> &altRoot = {});
  polyast::Expr::Any handleValueAsScalar(mlir::Value val);

  template <typename T = polyast::Expr::Any> T handleValueAs(const mlir::Value val) {
    const auto expr = handleValue(val);
    return expr ^ get_maybe<T>() ^ fold([&]() -> T {
             if constexpr (std::is_same_v<T, polyast::Expr::Any>) {
               return polyast::Expr::Alias(polyast::Term::Poison(handleType(val.getType())));
             } else return T{};
           });
  }

  void handleOp(mlir::Operation *op);

  struct DoConcurrentRegion {

    enum class Locality : uint8_t { Default, Local, LocalInit, Reduce, Shared };

    struct Capture {
      polyast::Named named;
      mlir::Value value;
      Locality locality;
    };

    struct Reduction {
      polyast::Named named;
      polyast::Named partialArray;
      mlir::Value value;
      polydco::FReduction::Kind kind;
    };

    polyast::Program program;
    std::vector<std::pair<bool, polyast::StructLayout>> layouts;
    std::vector<Capture> captures;
    std::vector<Reduction> reductions;
    std::unordered_map<std::string, FBoxedMirror> boxes;
    polyast::StructLayout preludeLayout, captureLayout;
    std::optional<polyast::StructLayout> reductionLayout;

    struct CoreReduce {
      polyast::Type::Struct resultType;
      polyast::StructLayout resultLayout;
      polyast::Sym combine;
      std::vector<fir::ReduceOperationEnum> operations;
      std::vector<mlir::Value> slots;
    };
    std::optional<CoreReduce> coreReduce;
  };

  // with `bodyName`, the loop body becomes that closure function over the capture instead of an offload entry
  static DoConcurrentRegion createRegion(const std::string &name, bool gpu, mlir::ModuleOp &m, mlir::DataLayout &L, fir::DoLoopOp &op,
                                         const std::optional<polyast::Sym> &bodyName = {});
};

} // namespace polyregion::polyfc
