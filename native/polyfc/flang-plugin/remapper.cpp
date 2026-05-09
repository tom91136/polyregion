#include "remapper.h"

#include <algorithm>
#include <cstdlib>
#include <optional>
#include <unordered_map>
#include <vector>

#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Dialect/Support/FIRContext.h"
#include "flang/Optimizer/Support/InternalNames.h"
#include "flang/Optimizer/Support/Utils.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Value.h"

#include "aspartame/all.hpp"
#include "fmt/format.h"
#include "magic_enum/magic_enum.hpp"

#include "polyregion/llvm_dyn.hpp"

#include "ftypes.h"
#include "parallel.h"
#include "rewriter.h"
#include "utils.h"

using namespace polyregion;
using namespace polyregion::polyast;
using namespace aspartame;

static auto asView(const mlir::Operation::operand_range range) {
  return view(range.getBase(), range.getBase() + range.size()) | map([](auto &x) { return x.get(); });
}

static Term::Select asSelectF(const Expr::Any &expr) {
  if (auto a = expr.template get<Expr::Alias>()) {
    if (auto s = a->ref.template get<Term::Select>()) return *s;
  }
  return Term::Select(Named("_invalid_select", expr.tpe()), {}, expr.tpe());
}

static Stmt::Any update0(const Expr::Any &expr, const Term::Any &rhsT) {
  if (expr.tpe().is<Type::Ptr>()) {
    auto sel = asSelectF(expr);
    auto idx = Term::IntU64Const(0);
    return Stmt::Update(sel, idx, rhsT).widen();
  }
  return Stmt::Mut(asSelectF(expr), Expr::Alias(rhsT)).widen();
}

static Expr::Any index0(const Expr::Any &rhs) {
  return rhs.tpe().get<Type::Ptr>() //
         ^
         map([&](auto &ptr) {
           auto base = rhs.template get<Expr::Alias>() ? rhs.template get<Expr::Alias>()->ref : Term::Any(Term::Poison(rhs.tpe()).widen());
           return Expr::Index(base, Term::IntU64Const(0), ptr.comp).widen();
         })                  //
         ^ get_or_else(rhs); //
}

static std::optional<mlir::Value> maybe(mlir::Value v) { return v ? std::optional{v} : std::nullopt; }

const static polyfc::FBoxedNoneMirror FBoxedNoneM;
const static polyfc::FDescExtraMirror FDescExtraM;
const static polyfc::FDimMirror FDimM;

polyfc::Remapper::Remapper(mlir::ModuleOp &M, mlir::DataLayout &L, mlir::Operation *perimeter, const std::vector<Named> &captureRoot)
    : M(M), L(L), perimeter(perimeter), captureRoot(captureRoot) {}

std::optional<polyfc::FType> polyfc::Remapper::fTypeOf(const mlir::Type &type) {
  if (const auto it = typesLUT.find(type); it != typesLUT.end()) return it->second;
  return {};
}

mlir::Type polyfc::Remapper::resolveType(const Type::Any &tpe) {
  const auto C = M.getContext();
  return tpe.match_total(                                                             //
      [&](const Type::Float16 &) -> mlir::Type { return mlir::Float16Type::get(C); }, //
      [&](const Type::Float32 &) -> mlir::Type { return mlir::Float32Type::get(C); }, //
      [&](const Type::Float64 &) -> mlir::Type { return mlir::Float64Type::get(C); }, //

      [&](const Type::IntU8 &) -> mlir::Type { return mlir::IntegerType::get(C, 8, mlir::IntegerType::Unsigned); },   //
      [&](const Type::IntU16 &) -> mlir::Type { return mlir::IntegerType::get(C, 16, mlir::IntegerType::Unsigned); }, //
      [&](const Type::IntU32 &) -> mlir::Type { return mlir::IntegerType::get(C, 32, mlir::IntegerType::Unsigned); }, //
      [&](const Type::IntU64 &) -> mlir::Type { return mlir::IntegerType::get(C, 64, mlir::IntegerType::Unsigned); }, //

      [&](const Type::IntS8 &) -> mlir::Type { return mlir::IntegerType::get(C, 8, mlir::IntegerType::Signed); },   //
      [&](const Type::IntS16 &) -> mlir::Type { return mlir::IntegerType::get(C, 16, mlir::IntegerType::Signed); }, //
      [&](const Type::IntS32 &) -> mlir::Type { return mlir::IntegerType::get(C, 32, mlir::IntegerType::Signed); }, //
      [&](const Type::IntS64 &) -> mlir::Type { return mlir::IntegerType::get(C, 64, mlir::IntegerType::Signed); }, //

      [&](const Type::Nothing &) -> mlir::Type { raise("Nothing type appeared in raising back to MLIR type"); }, //
      [&](const Type::Unit0 &) -> mlir::Type { return mlir::NoneType::get(C); },                                 //
      [&](const Type::Bool1 &) -> mlir::Type { return mlir::IntegerType::get(C, 1); },                           //

      [&](const Type::Struct &x) -> mlir::Type {
        const auto def = defs                    //
                         | values()              //
                         | concat(syntheticDefs) //
                         | find([&](auto &d) { return d.name == x.name; });
        if (!def) raise(fmt::format("Unseen struct type {}", repr(x)));
        return mlir::LLVM::LLVMStructType::getLiteral(C, def->members ^ map([&](auto &m) { return resolveType(m.tpe); }));
      },                                                                                                               //
      [&](const Type::Ptr &x) -> mlir::Type { return mlir::LLVM::LLVMPointerType::get(C); },                           //
      [&](const Type::Arr &x) -> mlir::Type { return mlir::LLVM::LLVMArrayType::get(resolveType(x.comp), x.length); }, //
      [&](const Type::Var &x) -> mlir::Type { raise(fmt::format("Type::Var unsupported: {}", repr(x))); },             //
      [&](const Type::Exec &x) -> mlir::Type { raise(fmt::format("Type::Exec unsupported: {}", repr(x))); }            //
  );
}

StructLayout polyfc::Remapper::resolveLayout(const StructDef &def) {
  const auto fields = def.members ^ map([&](auto &m) { return std::pair{m, resolveType(m.tpe)}; });
  const auto mirror = mlir::LLVM::LLVMStructType::getLiteral(M.getContext(), fields | values() | to_vector());
  std::vector<StructLayoutMember> ms;
  // XXX See slot calculation in `gepToByteOffset` from mlir/lib/Dialect/LLVMIR/IR/LLVMMemorySlot.cpp
  size_t offset = 0;
  for (const auto &[named, ty] : fields) {
    offset = llvm::alignTo(offset, L.getTypeABIAlignment(ty));
    const uint64_t size = L.getTypeSize(ty);
    ms.emplace_back(named, offset, size);
    offset += size;
  }
  const size_t size = L.getTypeSize(mirror);
  if (offset > size)
    raise(fmt::format(
        "Type offset mismatch for {}: field size arithmetic gave last element offset = {} but max size from mlir::DataLayout gave {}",
        repr(def), offset, size));
  return StructLayout(repr(def.name), size, L.getTypeABIAlignment(mirror), ms);
}

Term::Select polyfc::Remapper::newVar(const std::variant<Expr::Any, Type::Any> &x) {
  static size_t id = 0;
  return x ^ fold_total(
                 [&](const Expr::Any &expr) {
                   const Named name(fmt::format("v{}", id++), expr.tpe());
                   stmts.emplace_back(Stmt::Var(name, expr, /*isMutable*/ false));
                   return Term::Select(name, {}, name.tpe);
                 },
                 [&](const Type::Any &tpe) {
                   const Named name(fmt::format("v{}", id++), tpe);
                   stmts.emplace_back(Stmt::Var(name, std::optional<Expr::Any>{}, /*isMutable*/ true));
                   return Term::Select(name, {}, name.tpe);
                 });
}

Type::Any polyfc::Remapper::handleType(const mlir::Type type, const bool captureBoundary) {

  auto handleSeq = [&](fir::SequenceType t) -> Type::Any {
    const auto dynamic = t.hasDynamicExtents() || t.hasUnknownShape();
    return Type::Ptr(handleType(t.getEleTy()), TypeSpace::Global());
  };

  auto handleBox = [&](const fir::BoxType t) -> std::pair<Type::Any, FType> {
    if (fir::isBoxNone(t)) {
      typesLUT.insert({t, FBoxedNoneM});
      boxTypes.emplace(FBoxedNoneMirror::tpe(), FBoxedNoneM);
      defs.insert({FBoxedNoneMirror::tpe(), FBoxedNoneM.def()});
      return {FBoxedNoneMirror::tpe(), FBoxedNoneM};
    } else {
      const auto comp0 = handleType(t.getEleTy());
      const auto comp = comp0.is<Type::Ptr>() ? comp0 : Type::Ptr(comp0, TypeSpace::Global()); // add ptr, even if the component isn't
      FBoxedMirror mirror(comp, fir::getBoxRank(t));
      const auto def = mirror.def();
      Type::Struct ty(def.name, {});
      typesLUT.insert({t, mirror});
      boxTypes.emplace(ty, mirror);
      defs.insert({ty, def});
      syntheticDefs.insert(FDimM.def());
      return {ty, mirror};
    }
  };

  return llvm_shared::visitDyn<Type::Any>(
             type, //
             [&](mlir::IndexType) -> Type::Any { return Type::IntS64(); },
             [&](const mlir::IntegerType i) -> Type::Any {
               switch (i.getWidth()) {
                 case 1: return Type::Bool1().widen();
                 case 8: return i.isSigned() || i.isSignless() ? Type::IntS8().widen() : Type::IntU8().widen();
                 case 16: return i.isSigned() || i.isSignless() ? Type::IntS16().widen() : Type::IntU16().widen();
                 case 32: return i.isSigned() || i.isSignless() ? Type::IntS32().widen() : Type::IntU32().widen();
                 case 64: return i.isSigned() || i.isSignless() ? Type::IntS64().widen() : Type::IntU64().widen();
                 default: return Type::Nothing();
               }
             },
             [&](mlir::Float16Type) -> Type::Any { return Type::Float16(); }, //
             [&](mlir::Float32Type) -> Type::Any { return Type::Float32(); }, //
             [&](mlir::Float64Type) -> Type::Any { return Type::Float64(); }, //
             [&](fir::ReferenceType t) -> Type::Any {
               // XXX special case for Seq and Box, both are flattened to just !box<T> and not !ref<box<T>> or !ref<ref<T>>
               if (const auto seqTy = llvm::dyn_cast<fir::SequenceType>(t.getEleTy())) return handleSeq(seqTy);
               if (const auto boxTy = llvm::dyn_cast<fir::BoxType>(t.getEleTy())) { // record both
                 auto [tpe, boxed] = handleBox(boxTy);
                 typesLUT.insert({t, boxed});
                 return Type::Ptr(tpe, TypeSpace::Global());
               }
               // Otherwise use normal pointer semantic
               return Type::Ptr(handleType(t.getEleTy()), TypeSpace::Global());
             },
             [&](const fir::HeapType t) -> Type::Any { return handleType(t.getEleTy()); },
             [&](fir::SequenceType t) -> Type::Any {
               // Bare fir.array (no enclosing fir.ref) shows up as an inline aggregate, e.g. a
               // sized array field of a record type. handleSeq returns a Ptr (since fir.ref is
               // handled above), but inline arrays need Type::Arr so struct GEP indexes the
               // bytes that live alongside the other fields.
               if (!t.hasDynamicExtents() && !t.hasUnknownShape() && t.getDimension() == 1) {
                 return Type::Arr(handleType(t.getEleTy()), static_cast<int32_t>(t.getConstantArraySize()), TypeSpace::Global());
               }
               return handleSeq(t);
             },
             [&](const fir::CharacterType) -> Type::Any { return Type::Ptr(Type::IntU8(), TypeSpace::Global()); },
             [&](fir::RecordType t) -> Type::Any {
               const StructDef def(Sym({t.getName().str()}), {},
                                   t.getTypeList()                                                                //
                                       | map([&](auto &name, auto &tpe) { return Named(name, handleType(tpe)); }) //
                                       | to_vector(),
                                   std::vector<Type::Struct>{});
               const Type::Struct ty(def.name, {});
               defs.insert({ty, def});
               return ty;
             }, //
             [&](const fir::BoxType t) -> Type::Any {
               const auto box = handleBox(t).first;
               return captureBoundary ? Type::Ptr(box, TypeSpace::Global()) : box;
             }) //
         ^ get_or_else(Type::Nothing());
}

Term::Select polyfc::Remapper::handleSelectExpr(const mlir::Value val) {
  const auto expr = handleValueAs(val);
  // Term::Select isn't an alternative of Expr::Any anymore; unwrap from Expr::Alias if present.
  if (auto a = expr.template get<Expr::Alias>()) {
    if (auto s = a->ref.template get<Term::Select>()) return *s;
  }
  return newVar(expr);
}

Expr::Any polyfc::Remapper::handleValueAsScalar(const mlir::Value val) {

  const auto expr = handleValue(val);
  return expr ^
         fold_partial([&](const Expr::Any &x) { return index0(x); },                       //
                      [&](const FVar &x) { return index0(Expr::Alias(x.value).widen()); }, //
                      [](const FBoxed &x) { return index0(x.addr()); },                    //
                      [](const FArrayCoord &x) -> Expr::Any {
                        // FArrayCoord carries Expr::Any operands but Expr::Index now needs Term::Any.
                        // Approximate: extract the underlying Term when wrapped in Alias, else Poison.
                        auto unwrap = [](const Expr::Any &e) -> Term::Any {
                          if (auto a = e.template get<Expr::Alias>()) return a->ref;
                          return Term::Poison(e.tpe()).widen();
                        };
                        return Expr::Index(unwrap(x.array), unwrap(x.offset), x.comp).widen();
                      }) //
         ^ fold([&]() {  //
             return Expr::Alias(Term::Poison(handleType(val.getType()))).widen();
           });
}

static std::optional<std::string> resolveUniqueName(const fir::DeclareOp decl) {
  if (const auto strAttr = decl->getAttrOfType<mlir::StringAttr>("uniq_name")) {
    // TODO use fir::NameUniquer::deconstruct(v).second.name
    return strAttr.str();
  }
  return {};
}

static bool isEnclosedWithin(const mlir::Operation *parent, mlir::Operation *that) {
  for (; that; that = that->getParentOp()) {
    if (that == parent) return true;
  }
  return false;
}

struct BoxRoot {
  mlir::Value value;
  std::optional<std::string> name;

  static std::optional<BoxRoot> resolve(const mlir::Value &val) {
    return walk(val) ^ map([&](const BoxRoot &r) {
             return BoxRoot{r.value, //
                            r.name ^ or_else([&]() -> std::optional<std::string> {
                              std::optional<std::string> name;
                              for (const auto op : r.value.getUsers()) {
                                if (const auto decl = llvm::dyn_cast_if_present<fir::DeclareOp>(op)) {
                                  name = resolveUniqueName(decl);
                                  if (name) break;
                                }
                              }
                              return name;
                            })};
           });
  }

private:
  static std::optional<BoxRoot> walk(const mlir::Value &val) {
    using R = std::optional<BoxRoot>;
    auto walkMany = [](auto &op) -> R {
      if (const auto xs = asView(op.getOperands())                                             //
                          | collect([&](auto &v) -> R { return walk(v); })                     //
                          | distinct_by([&](auto &r) { return r.value.getAsOpaquePointer(); }) //
                          | to_vector();                                                       //
          xs.size() == 1)
        return xs[0];
      else if (xs.empty()) return {};
      else
        polyfc::raise(fmt::format("Bad root walk, op {} yielded multiple distinct roots: [{}]", polyfc::show(op),
                                  xs | mk_string(", ", [](auto &r) { return polyfc::show(r.value); })));
    };
    return llvm_shared::visitDyn<R>(
               val.getDefiningOp(), //
               [&](fir::DeclareOp x) -> R {
                 const auto name = resolveUniqueName(x);
                 return walkMany(x) //
                        ^ fold([&](auto &v) { return BoxRoot{v.value, v.name ^ or_else_maybe(name)}; },
                               [&]() { return BoxRoot{x.getResult(), name}; });
               }, //
               [&](fir::LoadOp x) -> R {
                 if (!fir::isa_box_type(x.getType())) return {};
                 return walk(x.getMemref());
               },                                                     //
               [&](fir::ShapeOp x) -> R { return walkMany(x); },      //
               [&](fir::ShiftOp x) -> R { return walkMany(x); },      //
               [&](fir::ShapeShiftOp x) -> R { return walkMany(x); }, //
               [&](fir::SliceOp x) -> R { return walkMany(x); },      //
               [&](fir::BoxDimsOp x) -> R { return walkMany(x); },    //
               [&](fir::ReboxOp x) -> R { return walkMany(x); })      //
           ^ flatten()                                                //
           ^ or_else([&]() -> R {                                     //
               return fir::isa_box_type(val.getType()) ? std::optional{BoxRoot{val, {}}} : std::nullopt;
             });
  }
};

polyfc::FExpr polyfc::Remapper::handleValue(const mlir::Value val, const std::optional<std::vector<Named>> &altRoot) {
  if (const auto it = valuesLUT.find(val); it != valuesLUT.end()) return it->second;
  if (const auto defOp = val.getDefiningOp()) {
    if (llvm::isa<mlir::arith::ConstantOp>(defOp) //
        || llvm::isa<fir::ShapeOp>(defOp)         //
        || llvm::isa<fir::ShiftOp>(defOp)         //
        || llvm::isa<fir::ShapeShiftOp>(defOp)    //
        || llvm::isa<fir::SliceOp>(defOp)         //
        || llvm::isa<fir::BoxDimsOp>(defOp)       //
    ) {
      handleOp(defOp);
      if (const auto it = valuesLUT.find(val); it != valuesLUT.end()) return it->second;
      else {
        return Expr::Alias(Term::Poison(handleType(val.getType())));
      }
    }

    if (!isEnclosedWithin(perimeter, defOp)) { // The value is from outside the region: handle captures
      // XXX For memrefs in general, it's possible to find two different defining ops that have the same root, for example,
      //   %7 = fir.load %6 : !fir.ref<!fir.box<!fir.array<?xf64>>>
      //   %8:3 = fir.box_dims %7, %c0 : (!fir.box<!fir.array<?xf64>>, index) -> (index, index, index)
      //   %9 = fir.shift %8#0 : (index) -> !fir.shift<1>
      //   %10 = fir.declare %7(%9) {...} : (!fir.box<!fir.array<?xf64>>, !fir.shift<1>) -> !fir.box<!fir.array<?xf64>>
      //   %11 = fir.rebox %10(%9) : (!fir.box<!fir.array<?xf64>>, !fir.shift<1>) -> !fir.box<!fir.array<?xf64>>
      // Both %7 and %11 could be the root, but %10 is what we really want as that has the correct var names.
      // When we encounter a memref where the defining op isn't a fir.declare, we'll trace the root too.

      auto bind = [&](const Term::Select &select, const mlir::Value x) -> FExpr {
        if (!altRoot) captures.insert({x, select});

        const auto selectExpr = Expr::Any(Expr::Alias(select));
        if (const auto tpe = fTypeOf(x.getType())) {
          const auto expr = *tpe ^ fold_total([&](const FBoxedMirror &m) -> FExpr { return FBoxed(selectExpr, m); },
                                              [&](const FBoxedNoneMirror &) -> FExpr { return FBoxedNone{selectExpr}; },
                                              [&](const FVarMirror &m) -> FExpr {
                                                // possibly an alloca is crossing the boundary...
                                                raise("FVar crossing the boundary!");
                                              });
          valuesLUT.insert({x, expr});
          return expr;
        } else {
          valuesLUT.insert({x, selectExpr});
          return selectExpr;
        }
      };

      static size_t id = 0;
      const auto tpe = handleType(val.getType(), true);
      return BoxRoot::resolve(val) //
             ^ fold(
                   [&](const BoxRoot &root) {
                     const Named field(root.name ^ fold([]() { return fmt::format("arg_{}", ++id); }), tpe);
                     auto expr = bind(dsl::Select(altRoot.value_or(captureRoot), field), root.value);
                     valuesLUT.insert({val, expr});
                     return expr;
                   },
                   [&]() {
                     const Named field(fmt::format("arg_{}", ++id), tpe);
                     return bind(dsl::Select(altRoot.value_or(captureRoot), field), val);
                   });
    }
  }
  return Expr::Alias(Term::Poison(handleType(val.getType())));
}

template <typename T> struct Bind {
  template <typename... Args> T operator()(Args &&...args) const { return T(std::forward<Args>(args)...); }
};

void polyfc::Remapper::handleOp(mlir::Operation *op) {
  const auto witness = [&](auto x, auto expr) -> void { valuesLUT.insert({x, expr}); };
  const auto intr2 = [&](auto x, auto ap) -> void {
    auto l = newVar(handleValueAsScalar(x.getLhs()));
    auto r = newVar(handleValueAsScalar(x.getRhs()));
    witness(x.getResult(), Expr::IntrOp(ap(l, r).widen()));
  };
  const auto intr2r = [&](auto x, auto ap) -> void {
    auto l = newVar(handleValueAsScalar(x.getLhs()));
    auto r = newVar(handleValueAsScalar(x.getRhs()));
    witness(x.getResult(), Expr::IntrOp(ap(l, r, handleType(x.getType())).widen()));
  };
  const auto poison = [&](auto x, const std::string &reason) -> void { witness(x, Expr::Alias(Term::Poison(handleType(x.getType())))); };
  const auto poison0 = [&](const std::string &reason) { ; };
  const auto push = [&](const auto &x) { stmts.emplace_back(x); };

  // push(Term::Unit0Const().widen());

  const auto handled = llvm_shared::visitDyn0(
      op, //
      [&](mlir::arith::ConstantOp c) {
        witness(c.getResult(), [&]() -> Expr::Any {
          const auto constVal = c.getValue();
          if (const auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(constVal)) {
            const auto tpe = handleType(intAttr.getType());
            if (tpe.is<Type::IntS8>()) return Expr::Alias(Term::IntS8Const(intAttr.getInt()));
            if (tpe.is<Type::IntS16>()) return Expr::Alias(Term::IntS16Const(intAttr.getInt()));
            if (tpe.is<Type::IntS32>()) return Expr::Alias(Term::IntS32Const(intAttr.getInt()));
            if (tpe.is<Type::IntS64>()) return Expr::Alias(Term::IntS64Const(intAttr.getInt()));
            if (tpe.is<Type::IntU8>()) return Expr::Alias(Term::IntU8Const(intAttr.getInt()));
            if (tpe.is<Type::IntU16>()) return Expr::Alias(Term::IntU16Const(intAttr.getInt()));
            if (tpe.is<Type::IntU32>()) return Expr::Alias(Term::IntU32Const(intAttr.getInt()));
            if (tpe.is<Type::IntU64>()) return Expr::Alias(Term::IntU64Const(intAttr.getInt()));
          } else if (const auto floatAttr = llvm::dyn_cast<mlir::FloatAttr>(constVal)) {
            const auto tpe = handleType(floatAttr.getType());
            if (tpe.is<Type::Float16>()) return Expr::Alias(Term::Float16Const(floatAttr.getValueAsDouble()));
            if (tpe.is<Type::Float32>()) return Expr::Alias(Term::Float32Const(floatAttr.getValueAsDouble()));
            if (tpe.is<Type::Float64>()) return Expr::Alias(Term::Float64Const(floatAttr.getValueAsDouble()));
          }
          return Expr::Alias(Term::Poison(handleType(constVal.getType())));
        }());
      },
      [&](fir::BoxDimsOp dims) {
        const auto boxSelect = handleValueAs<FBoxed>(dims.getVal());
        const auto rank = handleValueAs(dims.getDim());
        // Expr::Index now requires Term::Any operands; bind the box-dims base and rank to atom Selects.
        auto dimsBase = newVar(boxSelect.dims());
        auto rankT = newVar(rank);
        const auto dim = newVar(Expr::Index(dimsBase, rankT, FDimMirror::tpe()).widen());
        witness(dims.getLowerBound(), Expr::Alias(selectField(dim, FDimM.lowerBound)));
        witness(dims.getExtent(), Expr::Alias(selectField(dim, FDimM.extent)));
        witness(dims.getByteStride(), Expr::Alias(selectField(dim, FDimM.stride)));
      },
      [&](fir::ShiftOp s) {
        witness(s.getResult(),
                FShift{.lowerBounds = asView(s.getOrigins()) | map([&](auto &v) { return handleValueAs(v); }) | to_vector()});
      },

      [&](fir::ShapeOp s) {
        witness(s.getResult(), FShape{.extents = asView(s.getExtents()) | map([&](auto &v) { return handleValueAs(v); }) | to_vector()});
      },
      [&](fir::ShapeShiftOp s) {
        witness(s.getResult(), FShapeShift{.lowerBounds = s.getOrigins() ^ map([&](auto &v) { return handleValueAs(v); }),
                                           .extents = s.getExtents() ^ map([&](auto &v) { return handleValueAs(v); })});
      },
      [&](fir::SliceOp s) {
        const auto xs = (asView(s.getTriples()) | map([&](auto &v) { return handleValueAs(v); }) | to_vector()) ^ grouped(3) ^ transpose();
        witness(s.getResult(), FSlice{.lowerBounds = xs[0], .upperBounds = xs[1], .strides = xs[2]});
      },
      [&](mlir::arith::SelectOp x) {
        const auto cond = handleValueAsScalar(x.getCondition());
        const auto value = newVar(handleType(x.getType()));
        auto condT = newVar(cond);
        Term::Any oneT = dsl::integral(cond.tpe(), 1);
        auto eqT = newVar(Expr::IntrOp(Intr::LogicEq(condT, oneT)).widen());
        push(Stmt::Cond(eqT,                                                               //
                        {Stmt::Mut(value, handleValueAsScalar(x.getTrueValue())).widen()}, //
                        {Stmt::Mut(value, handleValueAsScalar(x.getFalseValue())).widen()})
                 .widen());
        witness(x.getResult(), Expr::Alias(value));
      },
      [&](mlir::arith::CmpIOp x) {
        switch (x.getPredicate()) {
          case mlir::arith::CmpIPredicate::eq: return intr2(x, Bind<Intr::LogicEq>());
          case mlir::arith::CmpIPredicate::ne: return intr2(x, Bind<Intr::LogicNeq>());

          case mlir::arith::CmpIPredicate::slt: return intr2(x, Bind<Intr::LogicLt>());
          case mlir::arith::CmpIPredicate::sle: return intr2(x, Bind<Intr::LogicLte>());
          case mlir::arith::CmpIPredicate::sgt: return intr2(x, Bind<Intr::LogicGt>());
          case mlir::arith::CmpIPredicate::sge: return intr2(x, Bind<Intr::LogicGte>());

          case mlir::arith::CmpIPredicate::ult: return intr2(x, Bind<Intr::LogicLt>());
          case mlir::arith::CmpIPredicate::ule: return intr2(x, Bind<Intr::LogicLte>());
          case mlir::arith::CmpIPredicate::ugt: return intr2(x, Bind<Intr::LogicGt>());
          case mlir::arith::CmpIPredicate::uge: return intr2(x, Bind<Intr::LogicGte>());
        }
      }, //

      [&](mlir::arith::CmpFOp x) {
        switch (x.getPredicate()) {
          case mlir::arith::CmpFPredicate::AlwaysFalse:
            return witness(x.getResult(), Expr::Any(Expr::Alias(dsl::integral(Type::Bool1(), false))));
          case mlir::arith::CmpFPredicate::AlwaysTrue:
            return witness(x.getResult(), Expr::Any(Expr::Alias(dsl::integral(Type::Bool1(), true))));

          case mlir::arith::CmpFPredicate::OEQ: return intr2(x, Bind<Intr::LogicEq>());
          case mlir::arith::CmpFPredicate::OGT: return intr2(x, Bind<Intr::LogicGt>());
          case mlir::arith::CmpFPredicate::OGE: return intr2(x, Bind<Intr::LogicGte>());
          case mlir::arith::CmpFPredicate::OLT: return intr2(x, Bind<Intr::LogicLt>());
          case mlir::arith::CmpFPredicate::OLE: return intr2(x, Bind<Intr::LogicLte>());
          case mlir::arith::CmpFPredicate::ONE: return intr2(x, Bind<Intr::LogicNeq>());

          case mlir::arith::CmpFPredicate::ORD: return poison(x.getResult(), "CmpF ORD Unimplemented");
          case mlir::arith::CmpFPredicate::UEQ: return poison(x.getResult(), "CmpF UEQ Unimplemented");
          case mlir::arith::CmpFPredicate::UGT: return poison(x.getResult(), "CmpF UGT Unimplemented");
          case mlir::arith::CmpFPredicate::UGE: return poison(x.getResult(), "CmpF UGE Unimplemented");
          case mlir::arith::CmpFPredicate::ULT: return poison(x.getResult(), "CmpF ULT Unimplemented");
          case mlir::arith::CmpFPredicate::ULE: return poison(x.getResult(), "CmpF ULE Unimplemented");
          case mlir::arith::CmpFPredicate::UNE: return poison(x.getResult(), "CmpF UNE Unimplemented");
          case mlir::arith::CmpFPredicate::UNO: return poison(x.getResult(), "CmpF UNO Unimplemented");
        }
      }, //

      [&](const mlir::arith::AddIOp x) { intr2r(x, Bind<Intr::Add>()); }, //
      [&](const mlir::arith::AddFOp x) { intr2r(x, Bind<Intr::Add>()); }, //

      [&](const mlir::arith::SubIOp x) { intr2r(x, Bind<Intr::Sub>()); }, //
      [&](const mlir::arith::SubFOp x) { intr2r(x, Bind<Intr::Sub>()); }, //

      [&](const mlir::arith::MulIOp x) { intr2r(x, Bind<Intr::Mul>()); }, //
      [&](const mlir::arith::MulFOp x) { intr2r(x, Bind<Intr::Mul>()); }, //

      [&](const mlir::arith::DivSIOp x) { intr2r(x, Bind<Intr::Div>()); }, //
      [&](const mlir::arith::DivUIOp x) { intr2r(x, Bind<Intr::Div>()); }, //
      [&](const mlir::arith::DivFOp x) { intr2r(x, Bind<Intr::Div>()); },

      [&](const mlir::arith::RemSIOp x) { intr2r(x, Bind<Intr::Rem>()); }, //
      [&](const mlir::arith::RemUIOp x) { intr2r(x, Bind<Intr::Rem>()); }, //
      [&](const mlir::arith::RemFOp x) { intr2r(x, Bind<Intr::Rem>()); },  //

      [&](const mlir::arith::MinSIOp x) { intr2r(x, Bind<Intr::Min>()); },    //
      [&](const mlir::arith::MinUIOp x) { intr2r(x, Bind<Intr::Min>()); },    //
      [&](const mlir::arith::MinimumFOp x) { intr2r(x, Bind<Intr::Min>()); }, //

      [&](const mlir::arith::MaxSIOp x) { intr2r(x, Bind<Intr::Max>()); },    //
      [&](const mlir::arith::MaxUIOp x) { intr2r(x, Bind<Intr::Max>()); },    //
      [&](const mlir::arith::MaximumFOp x) { intr2r(x, Bind<Intr::Max>()); }, //

      [&](fir::FieldIndexOp f) {
        const auto on = handleType(f.getOnType());
        if (const auto structTpe = on.get<Type::Struct>()) {
          if (const auto def = defs ^ get_maybe(*structTpe)) {
            const auto field = f.getFieldName().str();
            def->members ^ find([&](auto &n) { return n.symbol == field; }) ^
                fold([&](auto &n) { witness(f.getResult(), FFieldIndex{n}); },
                     [&] {
                       poison(f.getResult(), fmt::format("Unknown field name {} in {} from index (op=`{}`)", field, repr(*def), show(f)));
                     });
          } else poison(f.getResult(), fmt::format("Unknown field index on type {} (op=`{}`) ", show(f.getOnType()), show(f)));
        } else poison(f.getResult(), fmt::format("FieldIndexOp used against a non-struct type {} (op=`{}`)", repr(on), show(f)));
      },
      [&](fir::CoordinateOp c) {
        const auto handleBoxed = [&](const Expr::Any &base) {
          const auto indices = c.getIndices();
          if (indices.empty()) {
            poison(c.getResult(), fmt::format("CoordinateOf has no coordinate (op=`{}`)", show(c)));
            return;
          }
          Named field("invalid", Type::Nothing());
          const auto first = indices[0];
          if (auto v = mlir::dyn_cast_if_present<mlir::Value>(first)) {
            field = handleValueAs<FFieldIndex>(v).field;
          } else if (auto attr = mlir::dyn_cast_if_present<mlir::IntegerAttr>(first)) {
            const auto idx = static_cast<size_t>(attr.getInt());
            const auto baseTy = handleType(c.getBaseType());
            const auto resolveFieldByIdx = [&](const Type::Any &t) -> std::optional<Named> {
              if (auto s = t.get<Type::Struct>())
                if (auto def = defs ^ get_maybe(*s); def && idx < def->members.size()) return def->members[idx];
              return std::nullopt;
            };
            // The base type can be the record itself or a fir.ref/fir.box wrapping it.
            const auto resolved =         //
                resolveFieldByIdx(baseTy) //
                ^ or_else([&]() { return baseTy.get<Type::Ptr>() ^ flat_map([&](auto &p) { return resolveFieldByIdx(p.comp); }); });
            field = resolved ^
                    fold([](auto &n) { return n; }, [&]() { return Named(fmt::format("invalid_static_field_{}", idx), Type::Nothing()); });
          }
          const auto select = selectAny(base, field);
          const auto expr = field.tpe.get<Type::Ptr>()                                               //
                            ^ flat_map([&](auto &p) { return p.comp.template get<Type::Struct>(); }) //
                            ^ or_else([&]() { return field.tpe.get<Type::Struct>(); })               //
                            ^ flat_map([&](auto &s) { return boxTypes ^ get_maybe(s); })             //
                            ^ map([&](auto &m) { // we're pointing to a Ptr|FBox field, retain boxed semantic
                                return m ^ fold_total([&](const FBoxedMirror &bm) -> FExpr { return FBoxed(select, bm); },    //
                                                      [&](const FBoxedNoneMirror &) -> FExpr { return FBoxedNone{select}; }); //
                              })                                                                                              //
                            ^ fold([&]() -> FExpr { // pointing to scalar, like fir.alloca, use FVar semantic
                                if (auto a = select.template get<Expr::Alias>()) {
                                  if (auto s = a->ref.template get<Term::Select>()) return FVar{*s};
                                }
                                return Expr::Any(Expr::Alias(Term::Poison(select.tpe())));
                              });

          witness(c.getResult(), expr);
        };

        // TODO  (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
        if (const auto ref = handleValue(c.getRef()) ^ narrow<Expr::Any, FBoxed>()) {
          // handle case: (!fir.heap<!fir.type<T{x:f64}>>, !fir.field) -> !fir.ref<f64>
          *ref ^ foreach_total([&](const Expr::Any &e) { handleBoxed(e); }, [&](const FBoxed &e) { handleBoxed(e.addr()); });
        } else poison0(fmt::format("CoordinateOf ref value not an Expr|FBoxed, was {}", show(c)));
      },
      [&](fir::DeclareOp d) { witness(d.getResult(), handleValue(d.getMemref())); },
      [&](fir::ConvertOp c) {
        const auto as = handleType(c.getType());
        if (const auto from = handleValue(c.getOperand()); from ^ holds_any<FVar, FBoxed, FBoxedNone>()) {
          if (const auto tpe = fTypeOf(c.getType())) {
            *tpe ^ foreach_total([&](const FBoxedMirror &) { witness(c.getResult(), from); },
                                 [&](const FBoxedNoneMirror &) { witness(c.getResult(), from); },
                                 [&](const FVarMirror &) { witness(c.getResult(), from); });
          } else {
            from ^ foreach_partial(
                       [&](const FVar &e) {
                         // Expr::Cast now requires Term::Any.
                         witness(c.getResult(), Expr::Cast(e.value.widen(), as).widen());
                       },
                       [&](const FBoxed &e) {
                         auto base = newVar(index0(e.addr()));
                         witness(c.getResult(), Expr::Cast(base, as).widen());
                       },
                       [&](const FBoxedNone &e) {
                         poison(c.getResult(),
                                fmt::format("Cast source is an FBoxNone ({}) but output type is not an FType {}", fRepr(e), repr(as)));
                       });
          }
        } else {
          if (const auto tpe = fTypeOf(c.getType())) {
            // special case: if target type is a fType, retain lhs expression
            *tpe ^ foreach_total([&](const FBoxedMirror &) { witness(c.getResult(), from); },
                                 [&](const FBoxedNoneMirror &) { witness(c.getResult(), from); },
                                 [&](const FVarMirror &) { witness(c.getResult(), from); });
          } else {
            if (const auto expr = handleValueAs(c.getOperand()); expr.tpe() == as) witness(c.getResult(), expr);
            else {
              auto term = newVar(expr);
              witness(c.getResult(), Expr::Cast(term, as).widen());
            }
          }
        }
      },
      [&](fir::BoxAddrOp a) {
        const auto expr = handleValue(a.getVal());
        expr ^ narrow<FBoxed, Expr::Any>() ^
            fold(
                [&](auto &v) {
                  v ^ foreach_total([&](const FBoxed &e) { witness(a, e); }, //
                                    [&](const Expr::Any &e) { witness(a, e); });
                },
                [&]() { poison(a.getResult(), fmt::format("Unexpected expr ({}) on BoxAddr", fRepr(expr))); });
      },
      [&](fir::EmboxOp a) { witness(a, handleValue(a.getMemref())); },
      [&](fir::AddrOfOp a) {
        if (auto global = llvm::dyn_cast_if_present<fir::GlobalOp>(mlir::SymbolTable::lookupSymbolIn(M, a.getSymbol()))) {
          {
            const auto gName = Named(global.getSymName().str(), handleType(global.getType()));
            witness(a.getResult(), Expr::Alias(Term::Select(gName, {}, gName.tpe)));
          }
        } else {
          return poison(a.getResult(),
                        fmt::format("FIR AddrOf lookup of symbol {} is not a FIR global", a.getSymbol().getLeafReference().str()));
        }
      },
      [&](fir::CallOp a) {
        const auto args = asView(a.getArgs()) | to_vector();
        if (const auto symbol = a.getCallee()) {
          const auto name = symbol->getLeafReference().str();
          if (name == "_FortranAAssign") {
            // in flang/runtime/assign.cpp, @_FortranAAssign(!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
            if (args.size() != 4) {
              return poison0(fmt::format("While replacing intrinsics: expecting 4 argument(s) for {} but got {}", name, args.size()));
            }

            const auto rhsTye = handleType(args[1].getType());
            const auto lhs = handleValue(args[0]) ^ narrow<Expr::Any, FVar, FBoxed, FBoxedNone>();
            const auto rhs = handleValue(args[1]) ^ narrow<Expr::Any, FVar, FBoxed, FBoxedNone>();

            if (!lhs) return poison0(fmt::format("Unknown LHS {} in call to {}", show(args[0]), name));
            if (!rhs) return poison0(fmt::format("Unknown RHS {} in call to {}", show(args[1]), name));

            const auto rhs0 =
                *rhs ^ fold_total([&](const Expr::Any &x) -> Expr::Any { return Expr::Alias(Term::Poison(rhsTye)).widen(); },
                                  [&](const FVar &x) -> Expr::Any { return Expr::Alias(x.value).widen(); },
                                  [&](const FBoxed &x) -> Expr::Any { return Expr::Alias(Term::Poison(rhsTye)).widen(); },
                                  [&](const FBoxedNone &x) -> Expr::Any { return Expr::Alias(Term::Poison(rhsTye)).widen(); });

            *lhs ^ foreach_total(
                       [&](const Expr::Any &x) {
                         // Stmt::Mut needs Term::Select lhs.
                         push(Stmt::Mut(asSelectF(x), rhs0));
                       },                                                               //
                       [&](const FVar &x) { push(Stmt::Mut(x.value, rhs0)); },          //
                       [&](const FBoxed &x) { push(update0(x.addr(), newVar(rhs0))); }, //
                       [&](const FBoxedNone &x) { poison0("IMPL"); });
          } else poison0(fmt::format("Unimplemented intrinsic in {}", show(a)));
        } else poison0(fmt::format("Unknown callee in {}", show(a)));
      },

      [&](fir::ArrayCoorOp c) {
        // XXX see `XArrayCoorOpConversion` / `ArrayCoorConversion` in flang. Invariants:
        //   indices.size() == rank
        //   shape  empty || shape.size()  == rank
        //   shift  empty || shift.size()  == rank
        //   slice  empty || slice has 3*rank entries (lb, ub, stride)
        const auto seqTy = fir::unwrapUntilSeqType(c.getMemref().getType());
        if (!seqTy) return poison(c, fmt::format("Memref {} type does not contain a sequence type", show(c.getMemref().getType())));
        auto maybeRef = handleValue(c.getMemref());
        const auto ref = maybeRef ^ narrow<FBoxed, FVar, Expr::Any>();
        if (!ref) return poison(c, fmt::format("Memref is not a FBoxed|Expr|FVar (was {})", fRepr(maybeRef)));
        if (seqTy.hasUnknownShape()) return poison(c, "Unknown shape not yet implemented");

        const auto i64 = Type::IntS64();
        const auto asTerm = [&](const Expr::Any &e) { return Term::Any(newVar(e).widen()); };
        const auto onesT = [&](size_t n) {
          std::vector<Term::Any> out(n, Term::IntS64Const(1).widen());
          return out;
        };
        const size_t ranks = seqTy.getDimension();

        std::vector<Term::Any> actualShape;
        if (seqTy.hasDynamicExtents()) {
          const auto boxed = *ref ^ get_maybe<FBoxed>();
          if (!boxed) return poison(c, fmt::format("array {} has dynamic extent but is not boxed", show(c.getMemref())));
          for (size_t rank = 0; rank < ranks; ++rank)
            actualShape.emplace_back(asTerm(selectAny(Expr::Alias(newVar(boxed->dimAt(rank))).widen(), FDimM.extent)));
        } else {
          for (auto extent : seqTy.getShape())
            actualShape.emplace_back(Term::IntS64Const(extent).widen());
        }

        std::vector<Term::Any> indicesT;
        for (auto v : asView(c.getIndices()))
          indicesT.emplace_back(asTerm(handleValueAs(v)));

        std::vector<Term::Any> shapeT, shiftT;
        if (auto sh = c.getShape()) {
          if (auto val = handleValue(sh) ^ narrow<FShift, FShape, FShapeShift>()) {
            *val ^ foreach_total( //
                       [&](const FShift &x) {
                         shapeT = actualShape;
                         for (auto &b : x.lowerBounds)
                           shiftT.emplace_back(asTerm(b));
                       },
                       [&](const FShape &x) {
                         for (auto &e : x.extents)
                           shapeT.emplace_back(asTerm(e));
                         shiftT = onesT(ranks);
                       },
                       [&](const FShapeShift &x) {
                         for (auto &e : x.extents)
                           shapeT.emplace_back(asTerm(e));
                         for (auto &b : x.lowerBounds)
                           shiftT.emplace_back(asTerm(b));
                       });
          }
        }
        if (shapeT.empty()) shapeT = actualShape;
        if (shiftT.empty()) shiftT = onesT(ranks);

        std::vector<Term::Any> sliceLB, sliceStep;
        if (auto slice = c.getSlice()) {
          const auto val = handleValueAs<FSlice>(slice);
          for (auto &b : val.lowerBounds)
            sliceLB.emplace_back(asTerm(b));
          for (auto &s : val.strides)
            sliceStep.emplace_back(asTerm(s));
        }
        if (sliceLB.empty()) sliceLB = shiftT;
        if (sliceStep.empty()) sliceStep = onesT(ranks);

        if (indicesT.size() != ranks || shapeT.size() != ranks || shiftT.size() != ranks || sliceLB.size() != ranks ||
            sliceStep.size() != ranks) {
          return poison(c, fmt::format("rank mismatch in ArrayCoor (rank={}, sizes: idx={} shape={} shift={} sliceLB={} sliceStep={})",
                                       ranks, indicesT.size(), shapeT.size(), shiftT.size(), sliceLB.size(), sliceStep.size()));
        }

        // offset = sum_i (((idx[i] - shift[i]) * sliceStep[i] + (sliceLB[i] - shift[i])) * stride_i)
        // stride_i = product_{j<i} shape[j]
        Term::Any offset = Term::IntS64Const(0).widen();
        Term::Any stride = Term::IntS64Const(1).widen();
        for (size_t i = 0; i < ranks; ++i) {
          const auto idxMinusShift = Term::Any(newVar(Expr::IntrOp(Intr::Sub(indicesT[i], shiftT[i], i64)).widen()).widen());
          const auto scaled = Term::Any(newVar(Expr::IntrOp(Intr::Mul(idxMinusShift, sliceStep[i], i64)).widen()).widen());
          const auto sliceShift = Term::Any(newVar(Expr::IntrOp(Intr::Sub(sliceLB[i], shiftT[i], i64)).widen()).widen());
          const auto diff = Term::Any(newVar(Expr::IntrOp(Intr::Add(scaled, sliceShift, i64)).widen()).widen());
          const auto contribution = Term::Any(newVar(Expr::IntrOp(Intr::Mul(diff, stride, i64)).widen()).widen());
          offset = Term::Any(newVar(Expr::IntrOp(Intr::Add(offset, contribution, i64)).widen()).widen());
          stride = Term::Any(newVar(Expr::IntrOp(Intr::Mul(stride, shapeT[i], i64)).widen()).widen());
        }

        const auto offsetExpr = Expr::Alias(offset).widen();
        *ref ^ foreach_total( //
                   [&](const FBoxed &e) { witness(c.getResult(), FArrayCoord{e.addr(), offsetExpr, e.comp()}); },
                   [&](const FVar &e) {
                     witness(c.getResult(), FArrayCoord{Expr::Alias(e.value).widen(), offsetExpr, handleType(seqTy.getEleTy())});
                   },
                   [&](const Expr::Any &e) { witness(c.getResult(), FArrayCoord{e, offsetExpr, handleType(seqTy.getEleTy())}); });
      },

      [&](fir::AllocaOp a) {
        static size_t id;
        const Named named(fmt::format("alloca_{}", ++id), handleType(a.getInType()));
        push(Stmt::Var(named, std::optional<Expr::Any>{}, /*isMutable*/ true).widen());
        if (named.tpe.is<Type::Ptr>()) {
          witness(a.getResult(), Expr::Alias(Term::Select(named, {}, named.tpe)));
        } else {
          witness(a.getResult(), FVar{Term::Select(named, {}, named.tpe)});
        }
      },
      [&](fir::LoadOp l) {
        const auto expr = handleValue(l.getMemref());
        if (const auto ref = expr ^ narrow<Expr::Any, FVar, FBoxed, FArrayCoord>()) {
          *ref ^ foreach_total([&](const Expr::Any &e) { witness(l.getResult(), index0(e)); },               //
                               [&](const FVar &e) { witness(l.getResult(), Expr::Alias(e.value).widen()); }, //
                               [&](const FArrayCoord &e) {
                                 auto base = newVar(e.array);
                                 auto off = newVar(e.offset);
                                 witness(l.getResult(), Expr::Index(base, off, e.comp).widen());
                               },                                                    //
                               [&](const FBoxed &e) { witness(l.getResult(), e); }); //
        } else poison0(fmt::format("LoadOp RHS value not an Expr|FBoxed|FArrayCoord, was {}", fRepr(expr)));
      },
      [&](fir::StoreOp s) {
        const auto rhs = handleValueAs(s.getValue());
        if (const auto lhs = handleValue(s.getMemref()) ^ narrow<Expr::Any, FVar, FBoxed, FArrayCoord>()) {
          *lhs ^ foreach_total([&](const Expr::Any &e) { push(update0(e, newVar(rhs))); },    //
                               [&](const FVar &a) { push(Stmt::Mut(a.value, rhs).widen()); }, //
                               [&](const FArrayCoord &e) {
                                 auto rhsT = newVar(rhs);
                                 push(Stmt::Update(asSelectF(e.array), newVar(e.offset), rhsT).widen());
                               }, //
                               [&](const FBoxed &e) { push(update0(e.addr(), newVar(rhs))); });
        } else poison0(fmt::format("StoreOp LHS value not an Expr|FBoxed|FArrayCoord, was {}", show(s.getValue())));
      },
      [&](const fir::ResultOp r) {
        // (no-op result, formerly emitted Stmt::Comment)
      });
  if (!handled) {
    // (no-op for unhandled ops)
  }
}

static polydco::FReduction::Kind mapReductionKind(const fir::ReduceOperationEnum &k) {
  switch (k) {
    case fir::ReduceOperationEnum::Add: return polydco::FReduction::Kind::Add;
    case fir::ReduceOperationEnum::Multiply: return polydco::FReduction::Kind::Mul;

    case fir::ReduceOperationEnum::MAX: return polydco::FReduction::Kind::Max;
    case fir::ReduceOperationEnum::MIN: return polydco::FReduction::Kind::Min;

    case fir::ReduceOperationEnum::IAND: return polydco::FReduction::Kind::IAnd;
    case fir::ReduceOperationEnum::IOR: return polydco::FReduction::Kind::IOr;
    case fir::ReduceOperationEnum::IEOR: return polydco::FReduction::Kind::IEor;

    case fir::ReduceOperationEnum::AND: return polydco::FReduction::Kind::And;
    case fir::ReduceOperationEnum::OR: return polydco::FReduction::Kind::Or;
    case fir::ReduceOperationEnum::EQV: return polydco::FReduction::Kind::Eqv;
    case fir::ReduceOperationEnum::NEQV: return polydco::FReduction::Kind::Neqv;
    default: polyfc::raise(fmt::format("Unknown reduction kind {}", magic_enum::enum_name(k)));
  }
}

polyfc::Remapper::DoConcurrentRegion polyfc::Remapper::createRegion( //
    const std::string &name, bool gpu, mlir::ModuleOp &m, mlir::DataLayout &L, fir::DoLoopOp &op) {
  using namespace dsl;
  using namespace parallel_ops;

  const static std::string fnName = "_main";

  const static Named UpperBound("#upperBound", Long);
  const static Named LowerBound("#lowerBound", Long);
  const static Named Step("#step", Long);
  const static Named TripCount("#tripCount", Long);
  const static Named Begins("#begins", Ptr(Long));
  const static Named Ends("#ends", Ptr(Long));
  const static Named MappedInduction("#mappedInd", Long);

  const Type::Struct CaptureType(Sym({fmt::format("#Capture<{}>", name)}), {});
  const Named Capture("#capture", Ptr(CaptureType));

  const Type::Struct ReductionType(Sym({fmt::format("#Reduction<{}>", name)}), {});
  const Named Reduction("#reduction", Ptr(ReductionType));

  const StructDef preludeDef(Sym({"#Prelude"}), {},
                             gpu ? std::vector{LowerBound, UpperBound, Step, TripCount} //
                                 : std::vector{LowerBound, UpperBound, Step, TripCount, Begins, Ends},
                             std::vector<Type::Struct>{});
  const Named Prelude("#prelude", typeOf(preludeDef));

  op.getBody()->dump();
  Remapper r(m, L, op, {Capture});

  // Work out the reduction vars first, setting an alternative root to mask captures
  const auto exprWithReductions =
      op.getReduceAttrs() ^ to_vector() ^ flat_map([&](auto &attrs) {
        return asView(op.getReduceOperands()) //
               | zip(attrs)                   //
               | collect([&](auto &val, auto &attr) -> std::optional<std::pair<mlir::Value, polydco::FReduction::Kind>> {
                   if (const auto ra = llvm::dyn_cast<fir::ReduceAttr>(attr))
                     return std::pair{val, mapReductionKind(ra.getReduceOperation())};
                   return {};
                 })                                                                                                     //
               | collect([&](auto &val, auto &kind) -> std::optional<std::pair<FExpr, DoConcurrentRegion::Reduction>> { //
                   // A reduction locality capture should be a pointer to a scalar, boxed or otherwise
                   const auto expr = r.handleValue(val, std::vector<Named>{});
                   return expr                          //
                          ^ narrow<Expr::Any, FBoxed>() //
                          ^ flat_map([](auto &v) {      //
                              return v ^ fold_total(
                                             [](const Expr::Any &e) -> std::optional<Named> {
                                               // The new Term::Select carries (root, steps, tpe). The "last" component
                                               // is the leaf segment - either the last Field's name (with Select.tpe) or root.
                                               if (auto a = e.template get<Expr::Alias>()) {
                                                 if (auto sel = a->ref.template get<Term::Select>()) {
                                                   if (auto p = sel->tpe.template get<Type::Ptr>()) {
                                                     std::string leaf = sel->root.symbol;
                                                     for (auto &st : sel->steps) {
                                                       if (auto f = st.template get<PathStep::Field>()) leaf = f->name;
                                                     }
                                                     return Named(leaf, p->comp);
                                                   }
                                                 }
                                               }
                                               return std::nullopt;
                                             },
                                             [](const FBoxed &e) -> std::optional<Named> {
                                               if (auto a = e.base.template get<Expr::Alias>()) {
                                                 if (auto sel = a->ref.template get<Term::Select>()) {
                                                   std::string leaf = sel->root.symbol;
                                                   for (auto &st : sel->steps) {
                                                     if (auto f = st.template get<PathStep::Field>()) leaf = f->name;
                                                   }
                                                   return Named(leaf, e.comp());
                                                 }
                                               }
                                               return std::nullopt;
                                             });
                            }) //
                          ^ map([&](auto &target) {
                              const Named partial(fmt::format("#partial_{}", target.symbol), Ptr(target.tpe));
                              return std::pair{expr, DoConcurrentRegion::Reduction{.named = target,         //
                                                                                   .partialArray = partial, //
                                                                                   .value = val,            //
                                                                                   .kind = kind}};
                            });
                 }) //
               | to_vector();
      });

  for (auto &[expr, rd] : exprWithReductions) {
    for (auto &entry : r.valuesLUT) {
      if (entry.second == expr) entry.second = Expr::Any(Expr::Alias(Term::Select(rd.named, {}, rd.named.tpe)));
    }
  }

  // Then inject induction
  r.valuesLUT.insert({op.getInductionVar(), Expr::Any(Expr::Alias(selectNamed(MappedInduction)))});
  // Finally, map all operations
  for (auto &x : op.getBody()->getOperations()) {
    r.handleOp(&x);
  }

  // Captures are now populated
  const auto captures = r.captures //
                        | map([](auto &p) {
                            // The new Term::Select stores root + path-steps; pick the leaf Named (last Field, else root).
                            std::string leaf = p.second.root.symbol;
                            for (auto &s : p.second.steps) {
                              if (auto f = s.template get<PathStep::Field>()) leaf = f->name;
                            }
                            const Named leafNamed(leaf, p.second.tpe);
                            return DoConcurrentRegion::Capture{.named = leafNamed, //
                                                               .value = p.first,   //
                                                               .locality = DoConcurrentRegion::Locality::Default};
                          }) //
                        | to_vector();

  const StructDef capturesDef(CaptureType.name, {},
                              std::vector{Prelude}                                          //
                                  | concat(captures | map([](auto &c) { return c.named; })) //
                                  | to_vector(),
                              std::vector<Type::Struct>{});

  const StructDef reductionsDef(ReductionType.name, {}, //
                                exprWithReductions ^ map([&](auto &, auto &rd) { return rd.partialArray; }), std::vector<Type::Struct>{});

  r.syntheticDefs.emplace(preludeDef);
  r.syntheticDefs.emplace(capturesDef);
  if (!exprWithReductions.empty()) r.syntheticDefs.emplace(reductionsDef);

  const auto svrs = exprWithReductions ^ map([&](auto &, auto &rd) {
                      // SingleVarReduction.init is Term::Any. reductionInit returns Expr::Any (Alias-wrapped Term).
                      auto initExpr = reductionInit(rd.kind, rd.named.tpe);
                      Term::Any initTerm = initExpr.template get<Expr::Alias>() ? initExpr.template get<Expr::Alias>()->ref
                                                                                : Term::Any(Term::Poison(rd.named.tpe).widen());
                      return SingleVarReduction{.target = rd.named,                                   //
                                                .init = initTerm,                                     //
                                                .partialArray = Select({Reduction}, rd.partialArray), //
                                                .binaryOp = reductionOp(rd.kind, rd.named.tpe)};
                    });

  const auto params = gpu ? OpParams{GPUParams{.induction = MappedInduction,                         //
                                               .lowerBound = Select({Capture, Prelude}, LowerBound), //
                                               .step = Select({Capture, Prelude}, Step),             //
                                               .tripCount = Select({Capture, Prelude}, TripCount),   //
                                               .body = r.stmts}}                                     //
                          : OpParams{CPUParams{.induction = MappedInduction,                         //
                                               .lowerBound = Select({Capture, Prelude}, LowerBound), //
                                               .step = Select({Capture, Prelude}, Step),             //
                                               .begins = Select({Capture, Prelude}, Begins),         //
                                               .ends = Select({Capture, Prelude}, Ends),             //
                                               .body = r.stmts}};

  const Function entry = exprWithReductions.empty()             //
                             ? forEach(fnName, Capture, params) //
                             : reduce(fnName, Capture, Reduction, params, svrs);

  const Program program(entry, r.functions,
                        r.defs                        //
                            | values()                //
                            | concat(r.syntheticDefs) //
                            | to_vector(),
                        PassPhase::Initial());

  llvm::errs() << repr(program) << "\n";
  llvm::errs().flush();

  const auto defLayouts = program.defs | map([&](auto &d) { return r.resolveLayout(d); }) | to_vector();

  const auto findNamedLayout = [&](const auto &symbol) {
    const auto target = repr(symbol);
    return defLayouts ^ find([&](auto &d) { return d.name == target; }) ^ fold([&]() -> StructLayout { raise("Capture type missing"); });
  };

  return DoConcurrentRegion{
      .program = program,
      .layouts = defLayouts                                                                     //
                 | map([&](auto &l) { return std::pair{l.name == repr(capturesDef.name), l}; }) //
                 | to_vector(),
      .captures = captures,
      .reductions = exprWithReductions ^ map([](auto &, auto &rd) { return rd; }),
      .boxes = r.boxTypes //
               | collect([](auto &k, auto &f) {
                   return f ^ get_maybe<FBoxedMirror>() ^ map([&](auto &v) { return std::pair{repr(k.name), v}; });
                 })                        //
               | to<std::unordered_map>(), //
      .preludeLayout = findNamedLayout(preludeDef.name),
      .captureLayout = findNamedLayout(capturesDef.name),
      .reductionLayout = exprWithReductions.empty() ? std::optional<StructLayout>{} : findNamedLayout(reductionsDef.name),
  };
}