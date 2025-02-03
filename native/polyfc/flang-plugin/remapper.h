#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "polyast.h"
#include "utils.h"

#include "aspartame/all.hpp"
#include "fmt/core.h"

namespace polyregion::polyfc {

using namespace aspartame;

struct Remapper {

  struct BoxedFArrayAggregate {
    polyast::Type::Any comp;
    polyast::Named addr, sizeInBytes, ranks, dims;
    explicit BoxedFArrayAggregate(const polyast::Type::Any &t);
    BoxedFArrayAggregate();
    polyast::StructDef def(const std::string &name) const;
  };

  struct FBoxedArray {
    polyast::Expr::Any base;
    BoxedFArrayAggregate aggregate;

    FBoxedArray(const polyast::Expr::Any &base, const BoxedFArrayAggregate &aggregate);
    FBoxedArray();
    polyast::Type::Any comp() const;
    polyast::Expr::Any addr() const;
    polyast::Expr::Any sizeInBytes() const;
    polyast::Expr::Any ranks() const;
    polyast::Expr::Any dims() const;
    polyast::Expr::Any dimAt(size_t rank) const;
  };

  struct ArrayCoord {
    polyast::Expr::Any array;
    polyast::Expr::Any offset;
    polyast::Type::Any comp;
  };

  struct FTuple {
    std::vector<polyast::Expr::Any> values;
  };

  struct FShapeShift {
    std::vector<polyast::Expr::Any> lowerBounds, extents;
    size_t rank() const { return lowerBounds.size(); }
    std::vector<polyast::Expr::Any> values() const { return lowerBounds ^ concat(extents); }
  };

  struct FShift {
    std::vector<polyast::Expr::Any> lowerBounds;
    size_t rank() const { return lowerBounds.size(); }
    std::vector<polyast::Expr::Any> values() const { return lowerBounds; }
    FShapeShift asShapeShift() const {
      return FShapeShift{lowerBounds, repeat(polyast::Expr::IntS64Const(1).widen()) | take(rank()) | to_vector()};
    }
  };

  struct FShape {
    std::vector<polyast::Expr::Any> extents;
    size_t rank() const { return extents.size(); }
    std::vector<polyast::Expr::Any> values() const { return extents; }
    FShapeShift asShapeShift() const {
      return FShapeShift{repeat(polyast::Expr::IntS64Const(1).widen()) | take(rank()) | to_vector(), extents};
    }
  };

  struct FSlice {
    std::vector<polyast::Expr::Any> lowerBounds, upperBounds, strides;
    size_t rank() const { return lowerBounds.size(); }
    std::vector<polyast::Expr::Any> values() const { return lowerBounds | concat(upperBounds) | concat(strides) | to_vector(); }
  };

  using FExpr = std::variant<polyast::Expr::Any, //
                             FBoxedArray,        //
                             FTuple,             //
                             FShift,             //
                             FShape,             //
                             FShapeShift,        //
                             FSlice,             //
                             ArrayCoord>;
  //  case BoxedFArray -> base
  //  case Expr::Any -> identity
  //  case _ -> poison expr

  mlir::MLIRContext *C;
  mlir::DataLayout &L;
  mlir::Operation *perimeter;
  polyast::Named captureRoot;

  llvm::DenseMap<mlir::Value, FExpr> valuesLUT;
  llvm::DenseMap<mlir::Value,polyast:: Expr::Select> captures;
  llvm::DenseMap<mlir::Type, polyast::Type::Any> typeLUT;

  llvm::DenseMap<mlir::Type, polyast::StructDef> defs;
  std::unordered_set<polyast::StructDef> syntheticDefs;

  llvm::DenseMap<mlir::Type, BoxedFArrayAggregate> boxedArrays;

  std::vector<polyast::Stmt::Any> stmts;
  std::vector<polyast::Function> functions;
  polyast::Expr::Select newVar(const polyast::Expr::Any &expr);

  Remapper(mlir::MLIRContext *C, mlir::DataLayout &L, mlir::Operation *perimeter, const polyast::Named &captureRoot);

  mlir::Type resolveType(const polyast::Type::Any &tpe);
  polyast::StructLayout resolveLayout(const polyast::StructDef &def);
  polyast::Type::Any handleType(mlir::Type type);
  FExpr handleValue(mlir::Value val);
  polyast::Expr::Select handleSelectExpr(mlir::Value val);

  template <typename T = polyast::Expr::Any> T handleValueAs(const mlir::Value val) {
    return handleValue(val) //
           ^ get_maybe<T>() //
           ^ fold([&] {     //
               if constexpr (std::is_same_v<T, polyast::Expr::Any>) {
                 return polyast::Expr::Annotated(polyast::Expr::Poison(handleType(val.getType())), {},
                                                 fmt::format("Value {} cannot be cast to the required type", show(val)));
               } else return T{};

             });
  }

  void handleOp(mlir::Operation *op);

  struct DoConcurrentRegion {
    polyast::Program program;
    std::vector<std::pair<bool, polyast::StructLayout>> layouts;
    std::vector<std::pair<polyast::Named, mlir::Value>> captures;
    polyast::StructLayout preludeLayout, captureLayout;
  };

  std::vector<std::pair<polyast::Named, mlir::Value>> findCaptures(mlir::Block *op);

  static DoConcurrentRegion createRegion(const std::string &name, bool gpu, mlir::DataLayout &L, fir::DoLoopOp &op);
};

} // namespace polyregion::polyfc