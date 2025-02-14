#pragma once

#include "ftypes.h"
#include "polyast.h"

namespace polyregion::polyfc {

using namespace aspartame;

polyast::Expr::Any selectAny(const polyast::Expr::Any &base, const polyast::Named &that);

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
  // see flang/include/flang/Runtime/descriptor-consts.h
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
  friend bool operator==(const FBoxedMirror &lhs, const FBoxedMirror &rhs);
};

struct FVarMirror {
  polyast::Type::Any comp;
};

// ====

struct FBoxed {
  polyast::Expr::Any base;
  FBoxedMirror mirror;

  FBoxed(const polyast::Expr::Any &base, const FBoxedMirror &aggregate);
  FBoxed();
  polyast::Type::Any comp() const;
  polyast::Expr::Any addr() const;
  polyast::Expr::Any dims() const;
  polyast::Expr::Any dimAt(size_t rank) const;
  friend bool operator==(const FBoxed &lhs, const FBoxed &rhs);
};

struct FBoxedNone {
  polyast::Expr::Any base;
  friend bool operator==(const FBoxedNone &lhs, const FBoxedNone &rhs);
};

struct FBoxedNoneMirror {
  polyast::Named addr{"addr", polyast::Type::Ptr(polyast::Type::IntU8(), {}, polyast::TypeSpace::Global())};
  static polyast::Type::Struct tpe();
  polyast::StructDef def() const;
  friend bool operator==(const FBoxedNoneMirror &lhs, const FBoxedNoneMirror &rhs);
};

struct FArrayCoord {
  polyast::Expr::Any array;
  polyast::Expr::Any offset;
  polyast::Type::Any comp;
  friend bool operator==(const FArrayCoord &lhs, const FArrayCoord &rhs);
};

struct FFieldIndex {
  polyast::Named field{"invalid", polyast::Type::Nothing()};
  friend bool operator==(const FFieldIndex &lhs, const FFieldIndex &rhs);
};

struct FTuple {
  std::vector<polyast::Expr::Any> values;
  friend bool operator==(const FTuple &lhs, const FTuple &rhs);
};

struct FShapeShift {
  std::vector<polyast::Expr::Any> lowerBounds, extents;
  size_t rank() const { return lowerBounds.size(); }
  friend bool operator==(const FShapeShift &lhs, const FShapeShift &rhs);
};

struct FShift {
  std::vector<polyast::Expr::Any> lowerBounds;
  size_t rank() const;
  FShapeShift asShapeShift() const;
  friend bool operator==(const FShift &lhs, const FShift &rhs);
};

struct FShape {
  std::vector<polyast::Expr::Any> extents;
  size_t rank() const;
  FShapeShift asShapeShift() const;
  friend bool operator==(const FShape &lhs, const FShape &rhs);
};

struct FSlice {
  std::vector<polyast::Expr::Any> lowerBounds, upperBounds, strides;
  size_t rank() const;
  friend bool operator==(const FSlice &lhs, const FSlice &rhs);
};

struct FVar {
  polyast::Expr::Select value;
  friend bool operator==(const FVar &lhs, const FVar &rhs);
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

std::string fRepr(const FExpr &t);
std::string fRepr(const FType &t);

std::function<polyast::Expr::Any(const polyast::Expr::Any &, const polyast::Expr::Any &)> reductionOp(const polydco::FReduction::Kind &k,
                                                                                                      const polyast::Type::Any &t);

polyast::Expr::Any reductionInit(const polydco::FReduction::Kind &k, const polyast::Type::Any &t);

} // namespace polyregion::polyfc