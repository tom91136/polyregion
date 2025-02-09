#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "polyast.h"
#include "utils.h"

#include "aspartame/all.hpp"
#include "fmt/core.h"

namespace polyregion::polyfc {

using namespace aspartame;

struct Remapper {

  // see flang/include/flang/Runtime/descriptor-consts.h

  struct FDescExtraMirror {
    polyast::Named derivedType{"derivedType", polyast::Type::Ptr(polyast::Type::IntU8(), {}, polyast::TypeSpace::Global())};
    polyast::Named typeParamValue{"typeParamValue", polyast::Type::IntS64()};
    static polyast::Type::Struct tpe();
    polyast::StructDef def() const;
  };

  struct FDimMirror {
    polyast::Named lowerBound{"lowerBound", polyast::Type::IntS64()};
    polyast::Named extent{"extent", polyast::Type::IntS64()};
    polyast::Named stride{"stride", polyast::Type::IntS64()};
    static polyast::Type::Struct tpe();
    polyast::StructDef def() const;
  };

  struct FBoxedMirror {
    polyast::Named addr;
    polyast::Named sizeInBytes{"sizeInBytes", polyast::Type::IntS64()};
    polyast::Named version{"version", polyast::Type::IntS32()};
    polyast::Named rank{"rank", polyast::Type::IntU8()};
    polyast::Named type{"type", polyast::Type::IntS8()};
    polyast::Named attributes{"attributes", polyast::Type::IntU8()};
    polyast::Named extra{"extra", polyast::Type::IntU8()};
    size_t ranks;
    polyast::Named dims;
    std::optional<polyast::Named> derivedTypeInfo;

    explicit FBoxedMirror(const polyast::Type::Any &t, size_t ranks);
    FBoxedMirror();
    polyast::Type::Any comp() const;
    polyast::Type::Struct tpe() const;
    polyast::StructDef def() const;
  };

  struct FBoxed {
    polyast::Expr::Any base;
    FBoxedMirror mirror;

    FBoxed(const polyast::Expr::Any &base, const FBoxedMirror &aggregate);
    FBoxed();
    polyast::Type::Any comp() const;
    polyast::Expr::Any addr() const;
    // polyast::Expr::Any sizeInBytes() const;
    // polyast::Expr::Any ranks() const;
    polyast::Expr::Any dims() const;
    polyast::Expr::Any dimAt(size_t rank) const;
  };

  struct FBoxedNone {
    polyast::Expr::Any base;
  };

  struct FBoxedNoneMirror {
    polyast::Named addr{"addr", polyast::Type::Ptr(polyast::Type::IntU8(), {}, polyast::TypeSpace::Global())};
    static polyast::Type::Struct tpe();
    polyast::StructDef def() const;
  };

  struct FArrayCoord {
    polyast::Expr::Any array;
    polyast::Expr::Any offset;
    polyast::Type::Any comp;
  };

  struct FFieldIndex {
    polyast::Named field{"invalid", polyast::Type::Nothing()};
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

  struct FVar {
    polyast::Expr::Select value;
  };

  struct FVarMirror {
    polyast::Type::Any comp;
  };

  using FType = std::variant<FBoxedMirror,     //
                             FBoxedNoneMirror, //
                             FVarMirror>;

  using FExpr = std::variant<polyast::Expr::Any, //
                             FVar,               //
                             FBoxed,             //
                             FBoxedNone,         //
                             FTuple,             //
                             FShift,             //
                             FShape,             //
                             FShapeShift,        //
                             FSlice,             //
                             FArrayCoord,        //
                             FFieldIndex>;

  static std::string fRepr(const FExpr &t);
  static std::string fRepr(const FType &t);

  //  case BoxedFArray -> base
  //  case polyast::Expr::Any -> identity
  //  case _ -> poison expr

  mlir::ModuleOp &m;
  mlir::DataLayout &L;
  mlir::Operation *perimeter;
  polyast::Named captureRoot;

  llvm::DenseMap<mlir::Value, FExpr> valuesLUT;
  llvm::DenseMap<mlir::Type, FType> typesLUT;
  llvm::DenseMap<mlir::Value, polyast::Expr::Select> captures;

  std::unordered_set<polyast::StructDef> syntheticDefs;
  std::unordered_map<polyast::Type::Struct, polyast::StructDef> defs;
  std::unordered_map<polyast::Type::Struct, std::variant<FBoxedMirror, FBoxedNoneMirror>> boxTypes;


  std::vector<polyast::Stmt::Any> stmts;
  std::vector<polyast::Function> functions;
  polyast::Expr::Select newVar(const polyast::Expr::Any &expr);

  Remapper(mlir::ModuleOp &m, mlir::DataLayout &L, mlir::Operation *perimeter, const polyast::Named &captureRoot);
  std::optional<FType> fTypeOf(const mlir::Type &type);
  mlir::Type resolveType(const polyast::Type::Any &tpe);
  polyast::StructLayout resolveLayout(const polyast::StructDef &def);
  polyast::Type::Any handleType(mlir::Type type, bool captureBoundary = false);
  FExpr handleValue(mlir::Value val);
  polyast::Expr::Select handleSelectExpr(mlir::Value val);

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
    polyast::Program program;
    std::vector<std::pair<bool, polyast::StructLayout>> layouts;
    std::vector<std::pair<polyast::Named, mlir::Value>> captures;
    std::unordered_map<std::string, FBoxedMirror> boxes;
    polyast::StructLayout preludeLayout, captureLayout;
  };

  std::vector<std::pair<polyast::Named, mlir::Value>> findCaptures(mlir::Block *op);

  static DoConcurrentRegion createRegion(const std::string &name, bool gpu, mlir::ModuleOp &m, mlir::DataLayout &L, fir::DoLoopOp &op);
};

} // namespace polyregion::polyfc