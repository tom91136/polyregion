#include <algorithm>
#include <cstdlib>
#include <optional>
#include <unordered_map>
#include <vector>

#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Dialect/Support/FIRContext.h"
#include "flang/Optimizer/Support/Utils.h"

#include "flang/Optimizer/Support/InternalNames.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Value.h"

#include "llvm/Support/Casting.h"

#include "aspartame/all.hpp"
#include "fmt/format.h"
#include "magic_enum.hpp"
#include "polyregion/llvm_dyn.hpp"

#include "ftypes.h"
#include "parallel.h"
#include "remapper.h"
#include "rewriter.h"
#include "utils.h"

using namespace polyregion;
using namespace polyregion::polyast;
using namespace aspartame;

static auto asView(const mlir::Operation::operand_range range) {
  return view(range.getBase(), range.getBase() + range.size()) | map([](auto &x) { return x.get(); });
}

static Stmt::Any update0(const Expr::Any &expr, const Expr::Any &rhs) {
  // XXX for reductions, we'll be storing to a ref that is now a scalar, so handle it accordingly
  return expr.tpe().is<Type::Ptr>() ? Stmt::Update(expr, Expr::IntU64Const(0), rhs).widen() : Stmt::Mut(expr, rhs).widen();
}

static Expr::Any index0(const Expr::Any &rhs) {
  // XXX for reductions, we'll be loading from a ref that is now a scalar, so handle it accordingly
  return rhs.tpe().get<Type::Ptr>()                                                                 //
         ^ map([&](auto &ptr) { return Expr::Index(rhs, Expr::IntU64Const(0), ptr.comp).widen(); }) //
         ^ get_or_else(rhs);                                                                        //
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
      }, //
      [&](const Type::Ptr &x) -> mlir::Type {
        if (x.length) return mlir::LLVM::LLVMArrayType::get(resolveType(x.comp), *x.length);
        return mlir::LLVM::LLVMPointerType::get(C);
      },                                                                         //
      [&](const Type::Annotated &x) -> mlir::Type { return resolveType(x.tpe); } //
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
  return StructLayout(def.name, size, L.getTypeABIAlignment(mirror), ms);
}

Expr::Select polyfc::Remapper::newVar(const std::variant<Expr::Any, Type::Any> &x) {
  static size_t id = 0;
  return x ^ fold_total(
                 [&](const Expr::Any &expr) {
                   const Named name(fmt::format("v{}", id++), expr.tpe());
                   stmts.emplace_back(Stmt::Var(name, expr));
                   return Expr::Select({}, name);
                 },
                 [&](const Type::Any &tpe) {
                   const Named name(fmt::format("v{}", id++), tpe);
                   stmts.emplace_back(Stmt::Var(name, {}));
                   return Expr::Select({}, name);
                 });
}

Type::Any polyfc::Remapper::handleType(const mlir::Type type, const bool captureBoundary) {

  auto handleSeq = [&](fir::SequenceType t) -> Type::Any {
    const auto dynamic = t.hasDynamicExtents() || t.hasUnknownShape();
    return Type::Ptr(handleType(t.getEleTy()),
                     !dynamic && !captureBoundary ? std::optional<int32_t>{t.getConstantArraySize()} : std::nullopt, TypeSpace::Global());
  };

  auto handleBox = [&](const fir::BoxType t) -> std::pair<Type::Any, FType> {
    if (fir::isBoxNone(t)) {
      typesLUT.insert({t, FBoxedNoneM});
      boxTypes.emplace(FBoxedNoneMirror::tpe(), FBoxedNoneM);
      defs.insert({FBoxedNoneMirror::tpe(), FBoxedNoneM.def()});
      return {FBoxedNoneMirror::tpe(), FBoxedNoneM};
    } else {
      const auto comp0 = handleType(t.getEleTy());
      const auto comp = comp0.is<Type::Ptr>() ? comp0 : Type::Ptr(comp0, {}, TypeSpace::Global()); // add ptr, even if the component isn't
      FBoxedMirror mirror(comp, fir::getBoxRank(t));
      const auto def = mirror.def();
      Type::Struct ty(def.name);
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
                 default: return Type::Annotated(Type::Nothing(), {}, fmt::format("Unsupported MLIR integer type {}", show(type)));
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
                 return Type::Ptr(tpe, {}, TypeSpace::Global());
               }
               // Otherwise use normal pointer semantic
               return Type::Ptr(handleType(t.getEleTy()), {}, TypeSpace::Global());
             },
             [&](const fir::HeapType t) -> Type::Any { return handleType(t.getEleTy()); },
             [&](const fir::SequenceType t) -> Type::Any { return handleSeq(t); },
             [&](const fir::CharacterType) -> Type::Any { return Type::Ptr(Type::IntU8(), {}, TypeSpace::Global()); },
             [&](fir::RecordType t) -> Type::Any {
               const StructDef def(t.getName().str(),
                                   t.getTypeList() ^ map([&](auto &name, auto &tpe) { return Named(name, handleType(tpe)); }));
               const Type::Struct ty(def.name);
               defs.insert({ty, def});
               return ty;
             }, //
             [&](const fir::BoxType t) -> Type::Any {
               const auto box = handleBox(t).first;
               return captureBoundary ? Type::Ptr(box, {}, TypeSpace::Global()) : box;
             }) //
         ^ get_or_else(Type::Annotated(Type::Nothing(), {}, fmt::format("ERROR: Unsupported MLIR type {}", show(type))).widen());
}

Expr::Select polyfc::Remapper::handleSelectExpr(const mlir::Value val) {
  const auto expr = handleValueAs(val);
  return expr.get<Expr::Select>() ^ fold([&]() { return newVar(expr); });
}

Expr::Any polyfc::Remapper::handleValueAsScalar(const mlir::Value val) {

  const auto expr = handleValue(val);
  return expr ^
         fold_partial([&](const Expr::Any &x) { return index0(x); },                                       //
                      [&](const FVar &x) { return index0(x.value); },                                      //
                      [](const FBoxed &x) { return index0(x.addr()); },                                    //
                      [](const FArrayCoord &x) { return Expr::Index(x.array, x.offset, x.comp).widen(); }) //
         ^ fold([&]() {                                                                                    //
             return Expr::Annotated(Expr::Poison(handleType(val.getType())), {},
                                    fmt::format("Value {} cannot be treated as a Expr::Any scalar", fRepr(expr)))
                 .widen();
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
        return Expr::Annotated(Expr::Poison(handleType(val.getType())), {},
                               fmt::format("ERROR: Constant or defined vector type did not yield a usable value {}", show(val)));
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

      auto bind = [&](const Expr::Select &select, const mlir::Value x) -> FExpr {
        if (!altRoot) captures.insert({x, select});

        if (const auto tpe = fTypeOf(x.getType())) {
          const auto expr = *tpe ^ fold_total([&](const FBoxedMirror &m) -> FExpr { return FBoxed(select, m); },
                                              [&](const FBoxedNoneMirror &) -> FExpr { return FBoxedNone{select}; },
                                              [&](const FVarMirror &m) -> FExpr {
                                                // possibly an alloca is crossing the boundary...
                                                raise("FVar crossing the boundary!");
                                              });
          valuesLUT.insert({x, expr});
          return expr;
        } else {
          valuesLUT.insert({x, select});
          return select;
        }
      };

      static size_t id = 0;
      const auto tpe = handleType(val.getType(), true);
      return BoxRoot::resolve(val) //
             ^ fold(
                   [&](const BoxRoot &root) {
                     const Named field(root.name ^ fold([]() { return fmt::format("arg_{}", ++id); }), tpe);
                     auto expr = bind(Expr::Select(altRoot.value_or(captureRoot), field), root.value);
                     valuesLUT.insert({val, expr});
                     return expr;
                   },
                   [&]() {
                     const Named field(fmt::format("arg_{}", ++id), tpe);
                     return bind(Expr::Select(altRoot.value_or(captureRoot), field), val);
                   });
    }
  }
  return Expr::Annotated(Expr::Poison(handleType(val.getType())), {}, fmt::format("ERROR: Unseen MLIR value {}", show(val)));
}

template <typename T> struct Bind {
  template <typename... Args> T operator()(Args &&...args) const { return T(std::forward<Args>(args)...); }
};

void polyfc::Remapper::handleOp(mlir::Operation *op) {
  const auto witness = [&](auto x, auto expr) -> void { valuesLUT.insert({x, expr}); };
  const auto intr2 = [&](auto x, auto ap) -> void {
    witness(x.getResult(), Expr::IntrOp(ap(handleValueAsScalar(x.getLhs()), //
                                           handleValueAsScalar(x.getRhs())) //
                                            .widen()));
  };
  const auto intr2r = [&](auto x, auto ap) -> void {
    witness(x.getResult(), Expr::IntrOp(ap(handleValueAsScalar(x.getLhs()), //
                                           handleValueAsScalar(x.getRhs()), //
                                           handleType(x.getType()))         //
                                            .widen()));
  };
  const auto poison = [&](auto x, const std::string &reason) -> void {
    witness(x, Expr::Poison(handleType(x.getType())));
    stmts.emplace_back(Stmt::Comment(fmt::format("ERR: {}", reason)));
  };
  const auto poison0 = [&](const std::string &reason) { stmts.emplace_back(Stmt::Comment(fmt::format("ERR: {}", reason))); };
  const auto push = [&](const auto &x) { stmts.emplace_back(x); };

  // push(Stmt::Comment(show(op)));

  const auto handled = llvm_shared::visitDyn0(
      op, //
      [&](mlir::arith::ConstantOp c) {
        witness(c.getResult(), [&]() -> Expr::Any {
          const auto constVal = c.getValue();
          if (const auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(constVal)) {
            const auto tpe = handleType(intAttr.getType());
            if (tpe.is<Type::IntS8>()) return Expr::IntS8Const(intAttr.getInt());
            if (tpe.is<Type::IntS16>()) return Expr::IntS16Const(intAttr.getInt());
            if (tpe.is<Type::IntS32>()) return Expr::IntS32Const(intAttr.getInt());
            if (tpe.is<Type::IntS64>()) return Expr::IntS64Const(intAttr.getInt());
            if (tpe.is<Type::IntU8>()) return Expr::IntU8Const(intAttr.getInt());
            if (tpe.is<Type::IntU16>()) return Expr::IntU16Const(intAttr.getInt());
            if (tpe.is<Type::IntU32>()) return Expr::IntU32Const(intAttr.getInt());
            if (tpe.is<Type::IntU64>()) return Expr::IntU64Const(intAttr.getInt());
          } else if (const auto floatAttr = llvm::dyn_cast<mlir::FloatAttr>(constVal)) {
            const auto tpe = handleType(floatAttr.getType());
            if (tpe.is<Type::Float16>()) return Expr::Float16Const(floatAttr.getValueAsDouble());
            if (tpe.is<Type::Float32>()) return Expr::Float32Const(floatAttr.getValueAsDouble());
            if (tpe.is<Type::Float64>()) return Expr::Float64Const(floatAttr.getValueAsDouble());
          }
          return Expr::Annotated(Expr::Poison(handleType(constVal.getType())), {},
                                 fmt::format("ERROR: Unsupported arith.constant op {}", show(c)));
        }());
      },
      [&](fir::BoxDimsOp dims) {
        const auto boxSelect = handleValueAs<FBoxed>(dims.getVal());
        const auto rank = handleValueAs(dims.getDim());

        const auto dim = newVar(Expr::Index(boxSelect.dims(), rank, FDimMirror::tpe()));
        witness(dims.getLowerBound(), selectNamed(dim, FDimM.lowerBound));
        witness(dims.getExtent(), selectNamed(dim, FDimM.extent));
        witness(dims.getByteStride(), selectNamed(dim, FDimM.stride));
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
        push(Stmt::Cond(Expr::IntrOp(Intr::LogicEq(cond, dsl::integral(cond.tpe(), 1))), //
                        {Stmt::Mut(value, handleValueAsScalar(x.getTrueValue()))},       //
                        {Stmt::Mut(value, handleValueAsScalar(x.getFalseValue()))}));
        witness(x.getResult(), value);
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
          case mlir::arith::CmpFPredicate::AlwaysFalse: return witness(x.getResult(), dsl::integral(Type::Bool1(), false));
          case mlir::arith::CmpFPredicate::AlwaysTrue: return witness(x.getResult(), dsl::integral(Type::Bool1(), true));

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
          if (const auto it = defs.find(*structTpe); it != defs.end()) {
            const auto field = f.getFieldName().str();
            const auto def = it->second;
            def.members ^ find([&](auto &n) { return n.symbol == field; }) ^
                fold([&](auto &n) { witness(f.getResult(), FFieldIndex{n}); },
                     [&] {
                       poison(f.getResult(), fmt::format("Unknown field name {} in {} from index (op=`{}`)", field, repr(def), show(f)));
                     });
          } else poison(f.getResult(), fmt::format("Unknown field index on type {} (op=`{}`) ", show(f.getOnType()), show(f)));
        } else poison(f.getResult(), fmt::format("FieldIndexOp used against a non-struct type {} (op=`{}`)", repr(on), show(f)));
      },
      [&](fir::CoordinateOp c) {
        const auto handleBoxed = [&](const Expr::Any &base) {
          // if (const auto x = t.get<Type::Struct>(); !x) poison(c.getResult(), fmt::format("FBOX !PTR IMPL {}# {}", show(c), repr(t)));
          const auto field = handleValueAs<FFieldIndex>(c.getCoor()[0]).field;
          const auto select = selectAny(base, field);
          const auto expr =
              field.tpe.get<Type::Ptr>()                                               //
              ^ flat_map([&](auto &p) { return p.comp.template get<Type::Struct>(); }) //
              ^ or_else([&]() { return field.tpe.get<Type::Struct>(); })               //
              ^ flat_map([&](auto &s) { return boxTypes ^ get_maybe(s); })             //
              ^ map([&](auto &m) { // we're pointing to a Ptr|FBox field, retain boxed semantic
                  return m ^ fold_total([&](const FBoxedMirror &bm) -> FExpr { return FBoxed(select, bm); },    //
                                        [&](const FBoxedNoneMirror &) -> FExpr { return FBoxedNone{select}; }); //
                })                                                                                              //
              ^ fold([&] { // pointing to scalar, like fir.alloca, use FVar semantic
                  return select.get<Expr::Select>() ^
                         fold([&](auto &s) -> FExpr { return FVar{s}; },
                              [&]() -> FExpr { return Expr::Annotated(Expr::Poison(select.tpe()), {}, "CoordinateOp non-select coord"); });
                });

          witness(c.getResult(), expr);
        };

        // TODO  (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
        if (const auto ref = handleValue(c.getRef()) ^ narrow<Expr::Any, FBoxed>()) {
          // handle case: (!fir.heap<!fir.type<T{x:f64}>>, !fir.field) -> !fir.ref<f64>
          *ref ^ foreach_total([&](const Expr::Any &e) { handleBoxed(e); }, [&](const FBoxed &e) { handleBoxed(e.addr()); });
        } else poison0(fmt::format("CoordinateOf ref value not an Expr|FBoxed, was {}", show(c)));
      },
      [&](fir::ConvertOp c) {
        const auto as = handleType(c.getType());
        stmts.emplace_back(Stmt::Comment(fmt::format("convert {} to {}", fRepr(handleValue(c.getOperand())), repr(as))));
        if (const auto from = handleValue(c.getOperand()); from ^ holds_any<FVar, FBoxed, FBoxedNone>()) {
          if (const auto tpe = fTypeOf(c.getType())) {
            *tpe ^ foreach_total([&](const FBoxedMirror &) { witness(c.getResult(), from); },
                                 [&](const FBoxedNoneMirror &) { witness(c.getResult(), from); },
                                 [&](const FVarMirror &) { witness(c.getResult(), from); });
          } else {
            from ^ foreach_partial([&](const FVar &e) { witness(c.getResult(), Expr::Cast(e.value, as)); },
                                   [&](const FBoxed &e) { witness(c.getResult(), Expr::Cast(index0(e.addr()), as)); },
                                   [&](const FBoxedNone &e) {
                                     poison(c.getResult(), fmt::format("Cast source is an FBoxNone ({}) but output type is not an FType {}",
                                                                       fRepr(e), repr(as)));
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
            else witness(c.getResult(), Expr::Cast(expr, as));
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
          witness(a.getResult(), Expr::Select({}, Named(global.getSymName().str(), handleType(global.getType()))));
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

            const auto rhs0 = *rhs ^ fold_total(
                                         [&](const Expr::Any &x) {
                                           return Expr::Annotated(Expr::Poison(rhsTye), {}, fmt::format("IMPL?? {}", repr(x))).widen();
                                         },
                                         [&](const FVar &x) { return x.value.widen(); },
                                         [&](const FBoxed &x) { return Expr::Annotated(Expr::Poison(rhsTye), {}, "IMPL").widen(); },
                                         [&](const FBoxedNone &x) { return Expr::Annotated(Expr::Poison(rhsTye), {}, "IMPL").widen(); });

            *lhs ^ foreach_total([&](const Expr::Any &x) { push(Stmt::Mut(x, rhs0)); },   //
                                 [&](const FVar &x) { push(Stmt::Mut(x.value, rhs0)); },  //
                                 [&](const FBoxed &x) { push(update0(x.addr(), rhs0)); }, //
                                 [&](const FBoxedNone &x) { poison0("IMPL"); });
          } else poison0(fmt::format("Unimplemented intrinsic in {}", show(a)));
        } else poison0(fmt::format("Unknown callee in {}", show(a)));
      },

      [&](fir::ArrayCoorOp c) {
        // XXX see `XArrayCoorOpConversion` in CodeGen.cpp
        // XXX `XArrayCoorOp` values are created by `ArrayCoorConversion` in PreCGRewrite.cpp
        // The overall invariant is:
        /// coor.getIndices().size() == rank
        /// coor.getShape().empty() || coor.getShape().size() == rank
        /// coor.getShift().empty() || coor.getShift().size() == rank
        /// coor.getSlice().empty() || coor.getSlice().size() == 3 * rank
        ///
        const auto seqTy = fir::unwrapUntilSeqType(c.getMemref().getType());
        if (!seqTy) {
          return poison(c, fmt::format("ERROR: Memref {} type does not contain a sequence type", show(static_cast<mlir::Type>(seqTy))));
        }
        auto maybeRef = handleValue(c.getMemref());
        const auto ref = maybeRef ^ narrow<FBoxed, FVar, Expr::Any>(); // BoxedArray | Expr::Any + seq type
        if (!ref) {
          return poison(c, fmt::format("ERROR: Memref is not a FBoxed|Expr, the type is {}, expr is {}",
                                       show(static_cast<mlir::Type>(seqTy)), fRepr(maybeRef)));
        }
        if (seqTy.hasUnknownShape()) {
          return poison(c, fmt::format("IMPL: Unknown shape not implemented yet, op={}", show(c)));
        }

        const auto ones = [](auto n) { return repeat(Expr::IntS64Const(1).widen()) | take(n) | to_vector(); };
        const size_t ranks = seqTy.getDimension();

        std::vector<Expr::Any> actualShape;
        if (seqTy.hasDynamicExtents()) {
          std::optional<FBoxed> boxed = *ref ^ get_maybe<FBoxed>();
          if (!boxed) {
            return poison(c, fmt::format("ERROR: array ({}) has dynamic extent but us not boxed", show(c.getMemref())));
          }
          actualShape =
              iota<size_t>(0, ranks) | map([&](auto r) { return selectAny(newVar(boxed->dimAt(r)), FDimM.extent); }) | to_vector();
        } else { // known extent (shape), in theory this could be boxed too, but we just ignore the dynamic shape
          actualShape = seqTy.getShape() | map([](auto extent) { return Expr::IntS64Const(extent).widen(); }) | to_vector();
        }

        const auto indices = asView(c.getIndices()) | map([&](auto x) { return handleValueAs(x); }) | to_vector();
        const auto [shape, shift] = maybe(c.getShape())                                                                                   //
                                    ^ flat_map([&](auto &v) { return handleValue(v) ^ narrow<FShift, FShape, FShapeShift>(); })           //
                                    ^ map([&](auto &v) {                                                                                  //
                                        return v ^ fold_total([&](const FShift &x) { return std::pair{actualShape, x.lowerBounds}; },     //
                                                              [&](const FShape &x) { return std::pair{x.extents, ones(ranks)}; },         //
                                                              [&](const FShapeShift &x) { return std::pair{x.extents, x.lowerBounds}; }); //
                                      })                                                                                                  //
                                    ^ fold([&] { return std::pair{actualShape, ones(ranks)}; });                                          //
        const auto [sliceLB, sliceStep] = maybe(c.getSlice())                                                                             //
                                          ^ map([&](auto &x) { return handleValueAs<FSlice>(x); })                                        //
                                          ^ map([&](auto &s) { return std::pair{s.lowerBounds, s.strides}; })                             //
                                          ^ fold([&] { return std::pair{shift, ones(ranks)}; });                                          //

        for (const auto &[name, xs] :
             {std::pair{"indices", indices}, {"shape", shape}, {"shift", shift}, {"sliceLB", sliceLB}, {"sliceStep", sliceStep}}) {
          if (xs.size() != ranks) {
            return poison(c, fmt::format("ERROR: invariant: {} array size != rank ({})", name, ranks));
          }
        }

        using namespace polyast::dsl;
        const auto offset = newVar(0_(Long));
        const auto stride = newVar(1_(Long));
        for (size_t i = 0; i < ranks; ++i) {                              // lb[i] === shift[i]
          auto idx = newVar(call(Intr::Sub(indices[i], shift[i], Long))); // idx = indices[i] - lb[i]
          auto diff = newVar(call(Intr::Add(                              // diff = idx * step[i] + (sliceLB[i] - lb[i]);
              call(Intr::Mul(idx, sliceStep[i], Long)),                   //
              call(Intr::Sub(sliceLB[i], shift[i], Long)),                //
              Long)));

          push(Mut(offset, call(Intr::Add(offset, call(Intr::Mul(diff, stride, Long)), Long)))); // offset += diff * stride[i]
          push(Mut(stride, call(Intr::Mul(stride, shape[i], Long))));                            // stride *= shape[i]
        }
        witness(c.getResult(), *ref ^ fold_total([&](const FBoxed &e) { return FArrayCoord{e.addr(), offset, e.comp()}; }, //
                                                 [&](const FVar &e) { return FArrayCoord{e.value, offset, handleType(seqTy.getEleTy())}; },
                                                 [&](const Expr::Any &e) { return FArrayCoord{e, offset, handleType(seqTy.getEleTy())}; }));
      },

      [&](fir::AllocaOp a) {
        static size_t id;
        const Named named(fmt::format("alloca_{}", ++id), handleType(a.getInType()));
        push(Stmt::Var(named, {}));
        if (named.tpe.is<Type::Ptr>()) {
          witness(a.getResult(), Expr::Select({}, named));
        } else {
          witness(a.getResult(), FVar{Expr::Select({}, named)});
        }
      },
      [&](fir::LoadOp l) {
        const auto expr = handleValue(l.getMemref());
        if (const auto ref = expr ^ narrow<Expr::Any, FVar, FBoxed, FArrayCoord>()) {
          *ref ^ foreach_total([&](const Expr::Any &e) { witness(l.getResult(), index0(e)); },                                //
                               [&](const FVar &e) { witness(l.getResult(), e.value); },                                       //
                               [&](const FArrayCoord &e) { witness(l.getResult(), Expr::Index(e.array, e.offset, e.comp)); }, //
                               [&](const FBoxed &e) { witness(l.getResult(), e); });                                          //
        } else poison0(fmt::format("LoadOp RHS value not an Expr|FBoxed|FArrayCoord, was {}", fRepr(expr)));
      },
      [&](fir::StoreOp s) {
        const auto rhs = handleValueAs(s.getValue());
        if (const auto lhs = handleValue(s.getMemref()) ^ narrow<Expr::Any, FVar, FBoxed, FArrayCoord>()) {
          *lhs ^ foreach_total([&](const Expr::Any &e) { push(update0(e, rhs)); },                        //
                               [&](const FVar &a) { push(Stmt::Mut(a.value, rhs)); },                     //
                               [&](const FArrayCoord &e) { push(Stmt::Update(e.array, e.offset, rhs)); }, //
                               [&](const FBoxed &e) {
                                 push(update0(e.addr(), rhs)); // TODO does this actually happen?
                               });
        } else poison0(fmt::format("StoreOp LHS value not an Expr|FBoxed|FArrayCoord, was {}", show(s.getValue())));
      },
      [&](const fir::ResultOp r) {
        //        if (const auto xs = r.getResults(); xs.empty()) {
        //          push(Stmt::Return(Expr::Unit0Const()));
        //        } else if (xs.size() > 1) {
        //          push(Stmt::Comment(fmt::format("ERROR: Multiple values in return: {}", show(r))));
        //        } else {
        //          push(Stmt::Return(handleValue(xs.front())));
        //        }
        push(Stmt::Comment(fmt::format("ignored op {}", show(r))));
      });
  if (!handled) {
    push(Stmt::Comment(fmt::format("ERROR: Unsupported MLIR op {}", show(op))).widen());
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
    case fir::ReduceOperationEnum::EIOR: return polydco::FReduction::Kind::IEor;

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

  const Type::Struct CaptureType(fmt::format("#Capture<{}>", name));
  const Named Capture("#capture", Ptr(CaptureType));

  const Type::Struct ReductionType(fmt::format("#Reduction<{}>", name));
  const Named Reduction("#reduction", Ptr(ReductionType));

  const StructDef preludeDef("#Prelude", gpu ? std::vector{LowerBound, UpperBound, Step, TripCount} //
                                             : std::vector{LowerBound, UpperBound, Step, TripCount, Begins, Ends});
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
                                             [](const Expr::Any &e) {
                                               return e.get<Select>() ^ flat_map([&](auto &s) {
                                                        return s.last.tpe.template get<Type::Ptr>() //
                                                               ^ map([&](auto &p) { return Named(s.last.symbol, p.comp); });
                                                      });
                                             },
                                             [](const FBoxed &e) {
                                               return e.base.get<Select>() ^ map([&](auto &s) { return Named(s.last.symbol, e.comp()); });
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
    llvm::errs() << "XXXX R " << fRepr(expr) << " = " << repr(rd.named) << repr(rd.named.tpe) << "\n";
    for (auto &entry : r.valuesLUT) {
      if (entry.second == expr) entry.second = Select({}, rd.named);
    }
  }

  for (auto &x : r.valuesLUT) {
    llvm::errs() << "@@@@R " << show(x.first) << " = " << fRepr(x.second) << "\n";
  }

  // Then inject induction
  r.valuesLUT.insert({op.getInductionVar(), selectNamed(MappedInduction)});
  // Finally, map all operations
  for (auto &x : op.getBody()->getOperations()) {
    r.handleOp(&x);
  }

  // Captures are now populated
  const auto captures = r.captures //
                        | map([](auto &p) {
                            return DoConcurrentRegion::Capture{.named = tail(p.second)[0], //
                                                               .value = p.first,           //
                                                               .locality = DoConcurrentRegion::Locality::Default};
                          }) //
                        | to_vector();

  const StructDef capturesDef(CaptureType.name, std::vector{Prelude}                                          //
                                                    | concat(captures | map([](auto &c) { return c.named; })) //
                                                    | to_vector());

  const StructDef reductionsDef(ReductionType.name, //
                                exprWithReductions ^ map([&](auto &, auto &rd) { return rd.partialArray; }));

  r.syntheticDefs.emplace(preludeDef);
  r.syntheticDefs.emplace(capturesDef);
  if (!exprWithReductions.empty()) r.syntheticDefs.emplace(reductionsDef);

  const auto svrs = exprWithReductions ^ map([&](auto &, auto &rd) {
                      return SingleVarReduction{.target = rd.named,                                   //
                                                .init = reductionInit(rd.kind, rd.named.tpe),         //
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

  const Program program(r.defs                        //
                            | values()                //
                            | concat(r.syntheticDefs) //
                            | to_vector(),            //
                        r.functions | append(entry) | to_vector());

  llvm::errs() << repr(program) << "\n";
  llvm::errs().flush();

  const auto defLayouts = program.structs | map([&](auto &d) { return r.resolveLayout(d); }) | to_vector();

  const auto findNamedLayout = [&](const auto &symbol) {
    return defLayouts ^ find([&](auto &d) { return d.name == symbol; }) ^ fold([&]() -> StructLayout { raise("Capture type missing"); });
  };

  return DoConcurrentRegion{
      .program = program,
      .layouts = defLayouts                                                               //
                 | map([&](auto &l) { return std::pair{l.name == capturesDef.name, l}; }) //
                 | to_vector(),
      .captures = captures,
      .reductions = exprWithReductions ^ map([](auto &, auto &rd) { return rd; }),
      .boxes =
          r.boxTypes                                                                                                                     //
          | collect([](auto &k, auto &f) { return f ^ get_maybe<FBoxedMirror>() ^ map([&](auto &v) { return std::pair{k.name, v}; }); }) //
          | to<std::unordered_map>(),                                                                                                    //
      .preludeLayout = findNamedLayout(preludeDef.name),
      .captureLayout = findNamedLayout(capturesDef.name),
      .reductionLayout = exprWithReductions.empty() ? std::optional<StructLayout>{} : findNamedLayout(reductionsDef.name),
  };
}