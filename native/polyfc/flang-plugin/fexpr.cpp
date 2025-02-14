#include <optional>
#include <vector>

#include "aspartame/all.hpp"
#include "fmt/format.h"
#include "magic_enum.hpp"

#include "fexpr.h"
#include "ftypes.h"

using namespace polyregion;
using namespace polyregion::polyast;
using namespace aspartame;

Expr::Any polyfc::selectAny(const Expr::Any &base, const Named &that) {
  return base.get<Expr::Select>() ^                                        //
         fold([&](const auto &s) { return selectNamed(s, that).widen(); }, //
              [&] {
                return Expr::Annotated(Expr::Poison(that.tpe), {}, fmt::format("Select of non-select base `{}`", repr(base))).widen();
              });
}
Type::Struct polyfc::FDescExtraMirror::tpe() { return Type::Struct("FDescExtra"); }
StructDef polyfc::FDescExtraMirror::def() const { return StructDef(tpe().name, {derivedType, typeParamValue}); }
Type::Struct polyfc::FDimMirror::tpe() { return Type::Struct("FDim"); }
StructDef polyfc::FDimMirror::def() const { return StructDef(tpe().name, {lowerBound, extent, stride}); }
polyfc::FBoxedMirror::FBoxedMirror(const Type::Any &t, size_t ranks)
    : addr("addr", t), //
      ranks(ranks), dims("dim", Type::Ptr(FDimMirror::tpe(), ranks, TypeSpace::Global())),
      derivedTypeInfo(t.is<Type::Struct>() ? std::optional{Named("descExtra", FDescExtraMirror::tpe())} : std::nullopt) {}

polyfc::FBoxedMirror::FBoxedMirror() : FBoxedMirror(Type::Nothing(), 0) {}
Type::Any polyfc::FBoxedMirror::comp() const {
  return addr.tpe.get<Type::Ptr>() ^ fold([&](auto &t) { return t.comp; }, [&] { return Type::Nothing().widen(); });
}

Type::Struct polyfc::FBoxedMirror::tpe() const { return Type::Struct(fmt::format("FBoxed<{}, {}>", repr(comp()), ranks)); }
StructDef polyfc::FBoxedMirror::def() const {
  return StructDef(tpe().name,
                   std::vector{addr,        //
                               sizeInBytes, //
                               version,     //
                               rank,        //
                               type,        //
                               attributes,  //
                               extra}       //
                       ^ append(dims)       //
                       ^ concat(derivedTypeInfo ^ to_vector()));
}

polyfc::FBoxed::FBoxed(const Expr::Any &base, const FBoxedMirror &aggregate) : base(base), mirror(aggregate) {}
polyfc::FBoxed::FBoxed() : FBoxed(Expr::Annotated(Expr::Poison(Type::Nothing()), {}, "Empty FBoxed"), {}) {}
Type::Any polyfc::FBoxed::comp() const { return mirror.comp(); }
Expr::Any polyfc::FBoxed::addr() const { return selectAny(base, mirror.addr); }
Expr::Any polyfc::FBoxed::dims() const { return selectAny(base, mirror.dims); }
Expr::Any polyfc::FBoxed::dimAt(const size_t rank) const {
  return Expr::Index(selectAny(base, mirror.dims), Expr::IntS64Const(rank), FDimMirror::tpe());
}
Type::Struct polyfc::FBoxedNoneMirror::tpe() { return Type::Struct{"FBoxedNone"}; }
StructDef polyfc::FBoxedNoneMirror::def() const { return StructDef{tpe().name, {addr}}; }

bool polyfc::operator==(const FBoxedMirror &lhs, const FBoxedMirror &rhs) {
  return lhs.addr == rhs.addr && lhs.ranks == rhs.ranks && lhs.dims == rhs.dims && lhs.derivedTypeInfo == rhs.derivedTypeInfo;
}
bool polyfc::operator==(const FBoxed &lhs, const FBoxed &rhs) { return lhs.base == rhs.base && lhs.mirror == rhs.mirror; }
bool polyfc::operator==(const FBoxedNone &lhs, const FBoxedNone &rhs) { return lhs.base == rhs.base; }
bool polyfc::operator==(const FBoxedNoneMirror &lhs, const FBoxedNoneMirror &rhs) { return lhs.addr == rhs.addr; }
bool polyfc::operator==(const FArrayCoord &lhs, const FArrayCoord &rhs) {
  return lhs.array == rhs.array && lhs.offset == rhs.offset && lhs.comp == rhs.comp;
}
bool polyfc::operator==(const FFieldIndex &lhs, const FFieldIndex &rhs) { return lhs.field == rhs.field; }
bool polyfc::operator==(const FTuple &lhs, const FTuple &rhs) { return lhs.values == rhs.values; }
bool polyfc::operator==(const FShapeShift &lhs, const FShapeShift &rhs) {
  return lhs.lowerBounds == rhs.lowerBounds && lhs.extents == rhs.extents;
}
bool polyfc::operator==(const FShift &lhs, const FShift &rhs) { return lhs.lowerBounds == rhs.lowerBounds; }
bool polyfc::operator==(const FShape &lhs, const FShape &rhs) { return lhs.extents == rhs.extents; }
bool polyfc::operator==(const FSlice &lhs, const FSlice &rhs) {
  return lhs.lowerBounds == rhs.lowerBounds && lhs.upperBounds == rhs.upperBounds && lhs.strides == rhs.strides;
}
bool polyfc::operator==(const FVar &lhs, const FVar &rhs) { return lhs.value == rhs.value; }

std::string polyfc::fRepr(const FExpr &t) {
  return t ^ fold_total([&](const Expr::Any &p) { return repr(p); }, [&](const FVar &p) { return fmt::format("FVar({})", repr(p.value)); },
                        [&](const FBoxed &p) { return fmt::format("FBoxed({})", repr(p.base)); },
                        [&](const FBoxedNone &p) { return fmt::format("FBoxedNone({})", repr(p.base)); },
                        [&](const FTuple &p) { return fmt::format("FTuple({})", p.values | mk_string(", ", show_repr)); },
                        [&](const FShift &p) { return fmt::format("FShift(.lowerBounds={})", p.lowerBounds | mk_string(", ", show_repr)); },
                        [&](const FShape &p) { return fmt::format("FShape(.extents={})", p.extents | mk_string(", ", show_repr)); },
                        [&](const FShapeShift &p) {
                          return fmt::format("FShapeShift(.lowerBounds={}, .extents={})", //
                                             p.lowerBounds | mk_string(", ", show_repr),  //
                                             p.extents | mk_string(", ", show_repr));
                        },
                        [&](const FSlice &p) {
                          return fmt::format("FSlice(.lowerBounds={}, .upperBounds={}, .strides={})", //
                                             p.lowerBounds | mk_string(", ", show_repr),              //
                                             p.upperBounds | mk_string(", ", show_repr),              //
                                             p.strides | mk_string(", ", show_repr));
                        },
                        [&](const FArrayCoord &p) {
                          return fmt::format("FArrayCoord(.array={}, .offset={}, .comp={})", repr(p.array), repr(p.offset), repr(p.comp));
                        },
                        [&](const FFieldIndex &p) { return fmt::format("FFieldIndex(.field={})", repr(p.field)); });
}

std::string polyfc::fRepr(const FType &t) {
  return t ^ fold_total([&](const FBoxedMirror &p) -> std::string { return fmt::format("FBoxedMirror<{}>", repr(p.comp())); },
                        [&](const FBoxedNoneMirror &) -> std::string { return "FBoxedNone"; },
                        [&](const FVarMirror &p) -> std::string { return fmt::format("FVar<{}>", repr(p.comp)); });
}

std::function<Expr::Any(const Expr::Any &, const Expr::Any &)> polyfc::reductionOp(const polydco::FReduction::Kind &k, const Type::Any &t) {
  switch (k) {
    case polydco::FReduction::Kind::Add: return [&](auto &l, auto &r) { return Expr::IntrOp(Intr::Add(l, r, t)); };
    case polydco::FReduction::Kind::Mul: return [&](auto &l, auto &r) { return Expr::IntrOp(Intr::Mul(l, r, t)); };

    case polydco::FReduction::Kind::Max: return [&](auto &l, auto &r) { return Expr::IntrOp(Intr::Max(l, r, t)); };
    case polydco::FReduction::Kind::Min: return [&](auto &l, auto &r) { return Expr::IntrOp(Intr::Min(l, r, t)); };

    case polydco::FReduction::Kind::IAnd: return [&](auto &l, auto &r) { return Expr::IntrOp(Intr::BAnd(l, r, t)); };
    case polydco::FReduction::Kind::IOr: return [&](auto &l, auto &r) { return Expr::IntrOp(Intr::BOr(l, r, t)); };
    case polydco::FReduction::Kind::IEor: return [&](auto &l, auto &r) { return Expr::IntrOp(Intr::BXor(l, r, t)); };

    case polydco::FReduction::Kind::And: return [&](auto &l, auto &r) { return Expr::IntrOp(Intr::LogicAnd(l, r)); };
    case polydco::FReduction::Kind::Or: return [&](auto &l, auto &r) { return Expr::IntrOp(Intr::LogicOr(l, r)); };
    case polydco::FReduction::Kind::Eqv: return [&](auto &l, auto &r) { return Expr::IntrOp(Intr::LogicEq(l, r)); };
    case polydco::FReduction::Kind::Neqv: return [&](auto &l, auto &r) { return Expr::IntrOp(Intr::LogicNeq(l, r)); };
    default:
      return [&](auto &, auto &) {
        return Expr::Annotated(Expr::Poison(t), {}, fmt::format("Unsupported type {} for {} reduction", repr(t), magic_enum::enum_name(k)));
      };
  }
}

Expr::Any polyfc::reductionInit(const polydco::FReduction::Kind &k, const Type::Any &t) {
  auto unsupported = [&]() {
    return Expr::Annotated(Expr::Poison(t), {}, fmt::format("Unsupported type {} for {} reduction", repr(t), magic_enum::enum_name(k)));
  };
  return t.kind().match_total(
      [&](const TypeKind::Integral &) -> Expr::Any {
        switch (k) {
          case polydco::FReduction::Kind::Add: return dsl::numeric(t, [](auto _) { return static_cast<decltype(_)>(0); });
          case polydco::FReduction::Kind::Mul: return dsl::numeric(t, [](auto _) { return static_cast<decltype(_)>(1); });

          case polydco::FReduction::Kind::Max: return dsl::numeric(t, [](auto _) { return std::numeric_limits<decltype(_)>::lowest(); });
          case polydco::FReduction::Kind::Min: return dsl::numeric(t, [](auto _) { return std::numeric_limits<decltype(_)>::max(); });

          case polydco::FReduction::Kind::IAnd: return dsl::numeric(t, [](auto _) { return static_cast<decltype(_)>(~0); });
          case polydco::FReduction::Kind::IOr: return dsl::numeric(t, [](auto _) { return static_cast<decltype(_)>(0); });
          case polydco::FReduction::Kind::IEor: return dsl::numeric(t, [](auto _) { return static_cast<decltype(_)>(0); });

          case polydco::FReduction::Kind::And: return t.is<Type::Bool1>() ? Expr::Bool1Const(true).widen() : unsupported();
          case polydco::FReduction::Kind::Or: return t.is<Type::Bool1>() ? Expr::Bool1Const(false).widen() : unsupported();
          case polydco::FReduction::Kind::Eqv: return t.is<Type::Bool1>() ? Expr::Bool1Const(true).widen() : unsupported();
          case polydco::FReduction::Kind::Neqv: return t.is<Type::Bool1>() ? Expr::Bool1Const(false).widen() : unsupported();
          default: return unsupported();
        }
      },
      [&](const TypeKind::Fractional &) -> Expr::Any {
        switch (k) {
          case polydco::FReduction::Kind::Add: return dsl::numeric(t, [](auto _) { return static_cast<decltype(_)>(0); });
          case polydco::FReduction::Kind::Mul: return dsl::numeric(t, [](auto _) { return static_cast<decltype(_)>(1); });
          case polydco::FReduction::Kind::Max: return dsl::numeric(t, [](auto _) { return std::numeric_limits<decltype(_)>::lowest(); });
          case polydco::FReduction::Kind::Min: return dsl::numeric(t, [](auto _) { return std::numeric_limits<decltype(_)>::max(); });
          default: return unsupported();
        }
      },
      [&](const TypeKind::None &) -> Expr::Any { return unsupported(); }, //
      [&](const TypeKind::Ref &) -> Expr::Any { return unsupported(); });
}
