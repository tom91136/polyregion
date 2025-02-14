#pragma once

#include "fexpr.h"
#include "ftypes.h"
#include "polyast.h"
#include "utils.h"

#include "aspartame/all.hpp"
#include "fmt/core.h"

namespace polyregion::polyfc {

using namespace aspartame;

struct Remapper {

  mlir::ModuleOp &M;
  mlir::DataLayout &L;
  mlir::Operation *perimeter;
  std::vector<polyast::Named> captureRoot;

  llvm::DenseMap<mlir::Value, FExpr> valuesLUT;
  llvm::DenseMap<mlir::Type, FType> typesLUT;
  llvm::DenseMap<mlir::Value, polyast::Expr::Select> captures;

  std::unordered_set<polyast::StructDef> syntheticDefs;
  std::unordered_map<polyast::Type::Struct, polyast::StructDef> defs;
  std::unordered_map<polyast::Type::Struct, std::variant<FBoxedMirror, FBoxedNoneMirror>> boxTypes;

  std::vector<polyast::Stmt::Any> stmts;
  std::vector<polyast::Function> functions;
  polyast::Expr::Select newVar(const std::variant<polyast::Expr::Any, polyast::Type::Any> &expr);

  Remapper(mlir::ModuleOp &M, mlir::DataLayout &L, mlir::Operation *perimeter, const std::vector<polyast::Named> &captureRoot);
  std::optional<FType> fTypeOf(const mlir::Type &type);
  mlir::Type resolveType(const polyast::Type::Any &tpe);
  polyast::StructLayout resolveLayout(const polyast::StructDef &def);
  polyast::Type::Any handleType(mlir::Type type, bool captureBoundary = false);
  FExpr handleValue(mlir::Value val, const std::optional<std::vector<polyast::Named>> &altRoot = {});
  polyast::Expr::Select handleSelectExpr(mlir::Value val);
  polyast::Expr::Any handleValueAsScalar(mlir::Value val);

  template <typename T = polyast::Expr::Any> T handleValueAs(const mlir::Value val) {
    const auto expr = handleValue(val);
    return expr ^ get_maybe<T>() ^ fold([&] {
             if constexpr (std::is_same_v<T, polyast::Expr::Any>) {
               return polyast::Expr::Annotated(polyast::Expr::Poison(handleType(val.getType())), {},
                                               fmt::format("Value {} cannot be cast to Expr::Any", fRepr(expr)));
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
  };

  static DoConcurrentRegion createRegion(const std::string &name, bool gpu, mlir::ModuleOp &m, mlir::DataLayout &L, fir::DoLoopOp &op);
};

} // namespace polyregion::polyfc