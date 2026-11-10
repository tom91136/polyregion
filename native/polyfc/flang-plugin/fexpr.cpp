#include "fexpr.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

#include "aspartame/all.hpp"
#include "fmt/format.h"
#include "magic_enum/magic_enum.hpp"

#include "polyregion/error.h"

#include "ast.h"
#include "ftypes.h"

using namespace polyregion;
using namespace polyregion::polyast;
using namespace aspartame;

Expr::Any polyfc::selectAny(const Expr::Any &base, const Named &that) {
  // base may already wrap a Term::Select (via Expr::Alias). Unwrap and extend with a Field step.
  if (auto alias = base.get<Expr::Alias>()) {
    if (auto sel = alias->ref.get<Term::Select>()) {
      return Expr::Alias(selectField(*sel, that));
    }
  }
  raise(fmt::format("selecting field {} from non-variable base {} is not lowered", that.symbol, repr(base)));
}
Type::Struct polyfc::FDescExtraMirror::tpe() { return Type::Struct(Sym({"FDescExtra"}), {}); }
StructDef polyfc::FDescExtraMirror::def() const { return StructDef(tpe().name, {}, {derivedType, typeParamValue}, {}, false); }
Type::Struct polyfc::FDimMirror::tpe() { return Type::Struct(Sym({"FDim"}), {}); }
StructDef polyfc::FDimMirror::def() const { return StructDef(tpe().name, {}, {lowerBound, extent, stride}, {}, false); }
polyfc::FBoxedMirror::FBoxedMirror(const Type::Any &t, size_t ranks)
    : addr("addr", t), //
      ranks(ranks), dims("dim", Type::Arr(FDimMirror::tpe(), static_cast<int32_t>(ranks), TypeSpace::Global())),
      derivedTypeInfo(t.is<Type::Struct>() ? std::optional{Named("descExtra", FDescExtraMirror::tpe())} : std::nullopt) {}

polyfc::FBoxedMirror::FBoxedMirror() : FBoxedMirror(Type::Nothing(), 0) {}
Type::Any polyfc::FBoxedMirror::comp() const {
  return addr.tpe.get<Type::Ptr>() ^ fold([&](const auto &t) { return t.comp; }, [&] { return Type::Nothing().widen(); });
}

Type::Struct polyfc::FBoxedMirror::tpe() const {
  return Type::Struct(Sym({fmt::format("FBoxed<{}, {}>", canonicalName(comp()), ranks)}), {});
}
StructDef polyfc::FBoxedMirror::def() const {
  // XXX A zero-length `dims` lowers to OpTypeRuntimeArray, rejected by OpenCL Kernel SPIR-V.
  std::vector<Named> members{addr, sizeInBytes, version, rank, type, attributes, extra};
  if (ranks > 0) members.push_back(dims);
  return StructDef(tpe().name, {}, members ^ concat(derivedTypeInfo ^ to_vector()), {}, false);
}

polyfc::FBoxed::FBoxed(Expr::Any base, FBoxedMirror aggregate) : base(std::move(base)), mirror(std::move(aggregate)) {}
Type::Any polyfc::FBoxed::comp() const { return mirror.comp(); }
Expr::Any polyfc::FBoxed::addr() const { return selectAny(base, mirror.addr); }
Expr::Any polyfc::FBoxed::dims() const { return selectAny(base, mirror.dims); }
Expr::Any polyfc::FBoxed::dimAt(const size_t rank) const {
  // Index expects a Term::Any lhs; extract the Term from the Alias-wrapped Expr we built.
  const auto dims = selectAny(base, mirror.dims).get<Expr::Alias>()->ref;
  return Expr::Index(dims, Term::IntS64Const(rank), FDimMirror::tpe());
}
Type::Struct polyfc::FBoxedNoneMirror::tpe() { return Type::Struct(Sym({"FBoxedNone"}), {}); }
StructDef polyfc::FBoxedNoneMirror::def() const { return StructDef(tpe().name, {}, {addr}, {}, false); }

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

static std::string reprExtents(const std::vector<std::optional<Expr::Any>> &extents) {
  return extents | mk_string(", ", [](const auto &e) { return e ? repr(*e) : std::string{"*"}; });
}

std::string polyfc::fRepr(const FExpr &t) {
  return t //
         ^ fold_total([&](const Expr::Any &p) { return repr(p); }, [&](const FVar &p) { return fmt::format("FVar({})", repr(p.value)); },
                      [&](const FBoxed &p) { return fmt::format("FBoxed({})", repr(p.base)); },
                      [&](const FBoxedNone &p) { return fmt::format("FBoxedNone({})", repr(p.base)); },
                      [&](const FTuple &p) { return fmt::format("FTuple({})", p.values | mk_string(", ", show_repr)); },
                      [&](const FShift &p) { return fmt::format("FShift(.lowerBounds={})", p.lowerBounds | mk_string(", ", show_repr)); },
                      [&](const FShape &p) { return fmt::format("FShape(.extents={})", reprExtents(p.extents)); },
                      [&](const FShapeShift &p) {
                        return fmt::format("FShapeShift(.lowerBounds={}, .extents={})", //
                                           p.lowerBounds | mk_string(", ", show_repr),  //
                                           reprExtents(p.extents));
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
  return t //
         ^ fold_total([&](const FBoxedMirror &p) -> std::string { return fmt::format("FBoxedMirror<{}>", repr(p.comp())); },
                      [&](const FBoxedNoneMirror &) -> std::string { return "FBoxedNone"; },
                      [&](const FVarMirror &p) -> std::string { return fmt::format("FVar<{}>", repr(p.comp)); });
}

std::function<Expr::Any(const Term::Any &, const Term::Any &)> polyfc::reductionOp(const polydco::FReduction::Kind &k, const Type::Any &t) {
  switch (k) {
    case polydco::FReduction::Kind::Add: return [t](const auto &l, const auto &r) -> Expr::Any { return Expr::IntrOp(Intr::Add(l, r, t)); };
    case polydco::FReduction::Kind::Mul: return [t](const auto &l, const auto &r) -> Expr::Any { return Expr::IntrOp(Intr::Mul(l, r, t)); };
    case polydco::FReduction::Kind::Max: return [t](const auto &l, const auto &r) -> Expr::Any { return Expr::IntrOp(Intr::Max(l, r, t)); };
    case polydco::FReduction::Kind::Min: return [t](const auto &l, const auto &r) -> Expr::Any { return Expr::IntrOp(Intr::Min(l, r, t)); };
    case polydco::FReduction::Kind::IAnd:
      return [t](const auto &l, const auto &r) -> Expr::Any { return Expr::IntrOp(Intr::BAnd(l, r, t)); };
    case polydco::FReduction::Kind::IOr: return [t](const auto &l, const auto &r) -> Expr::Any { return Expr::IntrOp(Intr::BOr(l, r, t)); };
    case polydco::FReduction::Kind::IEor:
      return [t](const auto &l, const auto &r) -> Expr::Any { return Expr::IntrOp(Intr::BXor(l, r, t)); };
    case polydco::FReduction::Kind::And:
      return [](const auto &l, const auto &r) -> Expr::Any { return Expr::IntrOp(Intr::LogicAnd(l, r)); };
    case polydco::FReduction::Kind::Or: return [](const auto &l, const auto &r) -> Expr::Any { return Expr::IntrOp(Intr::LogicOr(l, r)); };
    case polydco::FReduction::Kind::Eqv: return [](const auto &l, const auto &r) -> Expr::Any { return Expr::IntrOp(Intr::LogicEq(l, r)); };
    case polydco::FReduction::Kind::Neqv:
      return [](const auto &l, const auto &r) -> Expr::Any { return Expr::IntrOp(Intr::LogicNeq(l, r)); };
    default: raise(fmt::format("reduction operator {} is not supported", magic_enum::enum_name(k)));
  }
}

Term::Any polyfc::reductionInit(const polydco::FReduction::Kind &k, const Type::Any &t) {
  auto unsupported = [&]() -> Term::Any { raise(fmt::format("reduction {} on {} is not supported", magic_enum::enum_name(k), repr(t))); };
  return t.kind().match_total(
      [&](const TypeKind::Integral &) -> Term::Any {
        switch (k) {
          case polydco::FReduction::Kind::Add: return dsl::numeric(t, [](const auto &_) { return static_cast<decltype(_)>(0); });
          case polydco::FReduction::Kind::Mul: return dsl::numeric(t, [](const auto &_) { return static_cast<decltype(_)>(1); });
          case polydco::FReduction::Kind::Max:
            return dsl::numeric(t, [](const auto &_) { return std::numeric_limits<std::decay_t<decltype(_)>>::lowest(); });
          case polydco::FReduction::Kind::Min:
            return dsl::numeric(t, [](const auto &_) { return std::numeric_limits<std::decay_t<decltype(_)>>::max(); });
          case polydco::FReduction::Kind::IAnd: return dsl::numeric(t, [](const auto &_) { return static_cast<decltype(_)>(~0); });
          case polydco::FReduction::Kind::IOr: return dsl::numeric(t, [](const auto &_) { return static_cast<decltype(_)>(0); });
          case polydco::FReduction::Kind::IEor: return dsl::numeric(t, [](const auto &_) { return static_cast<decltype(_)>(0); });
          case polydco::FReduction::Kind::And: return t.is<Type::Bool1>() ? Term::Bool1Const(true).widen() : unsupported();
          case polydco::FReduction::Kind::Or: return t.is<Type::Bool1>() ? Term::Bool1Const(false).widen() : unsupported();
          case polydco::FReduction::Kind::Eqv: return t.is<Type::Bool1>() ? Term::Bool1Const(true).widen() : unsupported();
          case polydco::FReduction::Kind::Neqv: return t.is<Type::Bool1>() ? Term::Bool1Const(false).widen() : unsupported();
          default: return unsupported();
        }
      },
      [&](const TypeKind::Fractional &) -> Term::Any {
        switch (k) {
          case polydco::FReduction::Kind::Add: return dsl::numeric(t, [](const auto &_) { return static_cast<decltype(_)>(0); });
          case polydco::FReduction::Kind::Mul: return dsl::numeric(t, [](const auto &_) { return static_cast<decltype(_)>(1); });
          case polydco::FReduction::Kind::Max:
            return dsl::numeric(t, [](const auto &_) { return std::numeric_limits<std::decay_t<decltype(_)>>::lowest(); });
          case polydco::FReduction::Kind::Min:
            return dsl::numeric(t, [](const auto &_) { return std::numeric_limits<std::decay_t<decltype(_)>>::max(); });
          default: return unsupported();
        }
      },
      [&](const TypeKind::None &) -> Term::Any { return unsupported(); },
      [&](const TypeKind::Ref &) -> Term::Any { return unsupported(); });
}
