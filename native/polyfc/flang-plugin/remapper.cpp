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

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Value.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/IR/Constant.h"
#include "llvm/Support/Casting.h"

#include "aspartame/all.hpp"
#include "fmt/format.h"
#include "polyregion/llvm_dyn.hpp"

#include "remapper.h"
#include "rewriter.h"
#include "utils.h"

using namespace polyregion;
using namespace polyregion::polyast;
using namespace aspartame;

// const static Type::Struct FDimTy("FDim");
// const static Named FDimLowerBound("lowerBound", Type::IntS64());
// const static Named FDimExtent("extent", Type::IntS64());
// const static Named FDimStride("stride", Type::IntS64());
// const static StructDef FDimDef(FDimTy.name, {FDimLowerBound, FDimExtent, FDimStride});

// const static Type::Struct FBoxedNoneTy("FBoxedNone");
// const static Named FBoxedNoneAddr("addr", Type::Ptr(Type::IntU8(), {}, TypeSpace::Global()));
// const static StructDef FBoxedNoneDef(FBoxedNoneTy.name, {FBoxedNoneAddr});

const static polyfc::Remapper::FBoxedNoneMirror FBoxedNoneM;
const static polyfc::Remapper::FDescExtraMirror FDescExtraM;
const static polyfc::Remapper::FDimMirror FDimM;

Type::Struct polyfc::Remapper::FDescExtraMirror::tpe() { return Type::Struct("FDescExtra"); }
StructDef polyfc::Remapper::FDescExtraMirror::def() const { return StructDef(tpe().name, {derivedType, typeParamValue}); }
Type::Struct polyfc::Remapper::FDimMirror::tpe() { return Type::Struct("FDim"); }
StructDef polyfc::Remapper::FDimMirror::def() const { return StructDef(tpe().name, {lowerBound, extent, stride}); }
polyfc::Remapper::FBoxedMirror::FBoxedMirror(const Type::Any &t, size_t ranks)
    : addr("addr", t), //
      ranks(ranks), dims("dim", Type::Ptr(FDimMirror::tpe(), ranks, TypeSpace::Global())),
      derivedTypeInfo(t.is<Type::Struct>() ? std::optional{Named("descExtra", FDescExtraMirror::tpe())} : std::nullopt) {}

polyfc::Remapper::FBoxedMirror::FBoxedMirror() : FBoxedMirror(Type::Nothing(), 0) {}
Type::Any polyfc::Remapper::FBoxedMirror::comp() const {
  return addr.tpe.get<Type::Ptr>() ^ fold([&](auto &t) { return t.comp; }, [&] { return Type::Nothing().widen(); });
}

Type::Struct polyfc::Remapper::FBoxedMirror::tpe() const { return Type::Struct(fmt::format("FBoxed<{}, {}>", repr(comp()), ranks)); }
StructDef polyfc::Remapper::FBoxedMirror::def() const {
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

static Expr::Any selectAny(const Expr::Any &base, const Named &that) {
  return base.get<Expr::Select>() ^                                        //
         fold([&](const auto &s) { return selectNamed(s, that).widen(); }, //
              [&] { return Expr::Poison(that.tpe).widen(); });
}

polyfc::Remapper::FBoxed::FBoxed(const Expr::Any &base, const FBoxedMirror &aggregate) : base(base), mirror(aggregate) {}
polyfc::Remapper::FBoxed::FBoxed() : FBoxed(Expr::Poison(Type::Nothing()), {}) {}
Type::Any polyfc::Remapper::FBoxed::comp() const { return mirror.comp(); }
Expr::Any polyfc::Remapper::FBoxed::addr() const { return selectAny(base, mirror.addr); }
// Expr::Any polyfc::Remapper::FBoxed::sizeInBytes() const { return selectAny(base, mirror.sizeInBytes); }
// Expr::Any polyfc::Remapper::FBoxed::ranks() const { return selectAny(base, mirror.rank); }
Expr::Any polyfc::Remapper::FBoxed::dims() const { return selectAny(base, mirror.dims); }
Expr::Any polyfc::Remapper::FBoxed::dimAt(const size_t rank) const {
  return Expr::Index(selectAny(base, mirror.dims), Expr::IntS64Const(rank), FDimMirror::tpe());
}
Type::Struct polyfc::Remapper::FBoxedNoneMirror::tpe() { return Type::Struct{"FBoxedNone"}; }
StructDef polyfc::Remapper::FBoxedNoneMirror::def() const { return StructDef{tpe().name, {addr}}; }

polyfc::Remapper::Remapper(mlir::ModuleOp &m, mlir::DataLayout &L, mlir::Operation *perimeter, const Named &captureRoot)
    : m(m), L(L), perimeter(perimeter), captureRoot(captureRoot) {}

std::optional<polyfc::Remapper::FType> polyfc::Remapper::fTypeOf(const mlir::Type &type) {
  if (const auto it = typesLUT.find(type); it != typesLUT.end()) return it->second;
  return {};
}

mlir::Type polyfc::Remapper::resolveType(const Type::Any &tpe) {
  const auto C = m.getContext();
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
  const auto mirror = mlir::LLVM::LLVMStructType::getLiteral(m.getContext(), fields | values() | to_vector());
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

std::string polyfc::Remapper::fRepr(const FExpr &t) {
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

std::string polyfc::Remapper::fRepr(const FType &t) {
  return t ^ fold_total([&](const FBoxedMirror &p) -> std::string { return fmt::format("FBoxedMirror<{}>", repr(p.comp())); },
                        [&](const FBoxedNoneMirror &) -> std::string { return "FBoxedNone"; },
                        [&](const FVarMirror &p) -> std::string { return fmt::format("FVar<{}>", repr(p.comp)); });
}

Expr::Select polyfc::Remapper::newVar(const Expr::Any &expr) {
  static size_t id = 0;
  const Named name(fmt::format("v{}", id++), expr.tpe());
  stmts.emplace_back(Stmt::Var(name, expr));
  return Expr::Select({}, name);
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
             },                                                                     //
             [&](const fir::BoxType t) -> Type::Any { return handleBox(t).first; }) //
         ^ get_or_else(Type::Annotated(Type::Nothing(), {}, fmt::format("ERROR: Unsupported MLIR type {}", show(type))).widen());
}

Expr::Select polyfc::Remapper::handleSelectExpr(const mlir::Value val) {
  const auto expr = handleValueAs(val);
  return expr.get<Expr::Select>() ^ fold([&]() { return newVar(expr); });
}

static std::optional<std::string> resolveUniqueName(const mlir::Value value) {
  if (const auto decl = llvm::dyn_cast_if_present<fir::DeclareOp>(value.getDefiningOp())) {
    if (const auto strAttr = decl->getAttrOfType<mlir::StringAttr>("uniq_name")) {
      return strAttr.str();
    }
  }
  return {};
}

static bool isEnclosedWithin(const mlir::Operation *parent, mlir::Operation *that) {
  for (; that; that = that->getParentOp()) {
    if (that == parent) return true;
  }
  return false;
}

polyfc::Remapper::FExpr polyfc::Remapper::handleValue(const mlir::Value val) {
  if (const auto it = valuesLUT.find(val); it != valuesLUT.end()) return it->second;
  if (const auto defOp = val.getDefiningOp()) {
    if (llvm::isa<mlir::arith::ConstantOp>(defOp) || //
        llvm::isa<fir::ShapeOp>(defOp) ||            //
        llvm::isa<fir::ShiftOp>(defOp) ||            //
        llvm::isa<fir::ShapeShiftOp>(defOp) ||       //
        llvm::isa<fir::SliceOp>(defOp)) {
      handleOp(defOp);
      if (const auto it = valuesLUT.find(val); it != valuesLUT.end()) return it->second;
      else {
        return Expr::Annotated(Expr::Poison(handleType(val.getType())), {},
                               fmt::format("ERROR: Constant or defined vector type did not yield a usable value {}", show(val)));
      }
    }
    if (!isEnclosedWithin(perimeter, defOp)) {
      static size_t id = 0;
      const Named field(resolveUniqueName(val).value_or(fmt::format("arg_{}", ++id)), handleType(val.getType(), true));
      const Expr::Select select({captureRoot}, field);
      captures.insert({val, select});
      if (const auto tpe = fTypeOf(val.getType())) {
        const auto expr = *tpe ^ fold_total([&](const FBoxedMirror &m) -> FExpr { return FBoxed(select, m); },
                                            [&](const FBoxedNoneMirror &) -> FExpr { return FBoxedNone{select}; },
                                            [&](const FVarMirror &m) -> FExpr {
                                              // possibly an alloca is crossing the boundary...
                                              raise("FVar crossing the boundary!");
                                            });
        valuesLUT.insert({val, expr});
        return expr;
      } else {
        valuesLUT.insert({val, select});
        return select;
      }
    }
  }
  return Expr::Annotated(Expr::Poison(handleType(val.getType())), {}, fmt::format("ERROR: Unseen MLIR value {}", show(val)));
}

template <typename T> struct Bind {
  template <typename... Args> T operator()(Args &&...args) const { return T(std::forward<Args>(args)...); }
};

static auto asView(const mlir::Operation::operand_range range) {
  return view(range.getBase(), range.getBase() + range.size()) | map([](auto &x) { return x.get(); });
}

static std::optional<mlir::Value> maybe(mlir::Value v) { return v ? std::optional{v} : std::nullopt; }

void polyfc::Remapper::handleOp(mlir::Operation *op) {
  auto witness = [&](auto x, auto expr) -> void { valuesLUT.insert({x, expr}); };
  auto intr2 = [&](auto x, auto ap) -> void {
    witness(x.getResult(), Expr::IntrOp(ap(handleValueAs(x.getLhs()), handleValueAs(x.getRhs()), handleType(x.getType())).widen()));
  };

  auto poison = [&](auto x, const std::string &reason) -> void {
    witness(x, Expr::Poison(handleType(x.getType())));
    stmts.emplace_back(Stmt::Comment(reason));
  };

  auto poison0 = [&](const std::string &reason) { stmts.emplace_back(Stmt::Comment(reason)); };
  auto push = [&](const auto &x) { stmts.emplace_back(x); };

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

      [&](const mlir::arith::AddIOp x) { intr2(x, Bind<Intr::Add>()); }, //
      [&](const mlir::arith::AddFOp x) { intr2(x, Bind<Intr::Add>()); }, //

      [&](const mlir::arith::SubIOp x) { intr2(x, Bind<Intr::Sub>()); }, //
      [&](const mlir::arith::SubFOp x) { intr2(x, Bind<Intr::Sub>()); }, //

      [&](const mlir::arith::MulIOp x) { intr2(x, Bind<Intr::Mul>()); }, //
      [&](const mlir::arith::MulFOp x) { intr2(x, Bind<Intr::Mul>()); }, //

      [&](const mlir::arith::DivSIOp x) { intr2(x, Bind<Intr::Div>()); }, //
      [&](const mlir::arith::DivUIOp x) { intr2(x, Bind<Intr::Div>()); }, //
      [&](const mlir::arith::DivFOp x) { intr2(x, Bind<Intr::Div>()); },

      [&](const mlir::arith::RemSIOp x) { intr2(x, Bind<Intr::Rem>()); }, //
      [&](const mlir::arith::RemUIOp x) { intr2(x, Bind<Intr::Rem>()); }, //
      [&](const mlir::arith::RemFOp x) { intr2(x, Bind<Intr::Rem>()); },  //

      [&](const mlir::arith::MinSIOp x) { intr2(x, Bind<Intr::Min>()); },    //
      [&](const mlir::arith::MinUIOp x) { intr2(x, Bind<Intr::Min>()); },    //
      [&](const mlir::arith::MinimumFOp x) { intr2(x, Bind<Intr::Min>()); }, //

      [&](const mlir::arith::MaxSIOp x) { intr2(x, Bind<Intr::Max>()); },    //
      [&](const mlir::arith::MaxUIOp x) { intr2(x, Bind<Intr::Max>()); },    //
      [&](const mlir::arith::MaximumFOp x) { intr2(x, Bind<Intr::Max>()); }, //

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

        const auto handleBoxed = [&](const Type::Any &t, const Expr::Any &base) {
          // if (const auto x = t.get<Type::Struct>(); !x) poison(c.getResult(), fmt::format("FBOX !PTR IMPL {}# {}", show(c), repr(t)));
          const auto field = handleValueAs<FFieldIndex>(c.getCoor()[0]).field;
          const auto select = selectAny(base, field);
          const auto expr = field.tpe.get<Type::Ptr>()                                               //
                            ^ flat_map([&](auto &p) { return p.comp.template get<Type::Struct>(); }) //
                            ^ or_else(field.tpe.get<Type::Struct>())                                 //
                            ^ flat_map([&](auto &s) { return boxTypes ^ get_maybe(s); })             //
                            ^ map([&](auto &m) { // we're pointing to a Ptr|FBox field, retain boxed semantic
                                return m ^ fold_total([&](const FBoxedMirror &bm) -> FExpr { return FBoxed(select, bm); },    //
                                                      [&](const FBoxedNoneMirror &) -> FExpr { return FBoxedNone{select}; }); //
                              })                                                                                              //
                            ^ fold([&] { // pointing to scalar, like fir.alloca, use FVar semantic
                                return select.get<Expr::Select>() ^ fold([&](auto &s) -> FExpr { return FVar{s}; },
                                                                         [&]() -> FExpr { return Expr::Poison(select.tpe()); });
                              });

          witness(c.getResult(), expr);
        };

        // TODO  (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
        if (const auto ref = handleValue(c.getRef()) ^ narrow<Expr::Any, FBoxed>()) {
          // handle case: (!fir.heap<!fir.type<T{x:f64}>>, !fir.field) -> !fir.ref<f64>
          *ref ^ foreach_total([&](const Expr::Any &e) { handleBoxed(e.tpe(), e); },
                               [&](const FBoxed &e) { handleBoxed(e.comp(), e.addr()); });
        } else poison0(fmt::format("CoordinateOf ref value not an Expr|FBoxed, was {}", show(c)));
      },
      [&](fir::ConvertOp c) {
        const auto as = handleType(c.getType());

        stmts.emplace_back(Stmt::Comment(fmt::format("convert {} to {}", fRepr(handleValue(c.getOperand())), repr(as))));
        if (const auto from = handleValue(c.getOperand()); from ^ holds_any<FBoxed, FBoxedNone, FVar>()) {
          if (const auto tpe = fTypeOf(c.getType())) {
            *tpe ^ foreach_total([&](const FBoxedMirror &) { witness(c.getResult(), from); },
                                 [&](const FBoxedNoneMirror &) { witness(c.getResult(), from); },
                                 [&](const FVarMirror &) { witness(c.getResult(), from); });
          } else {
            poison(c.getResult(), fmt::format("Cast source is a FExpr {} but output type is not an FType {}", fRepr(from), repr(as)));
          }
        } else {
          if (const auto expr = handleValueAs(c.getOperand()); expr.tpe() == as) witness(c.getResult(), expr);
          else witness(c.getResult(), (Expr::Cast(expr, as)));
        }
      },
      [&](fir::BoxAddrOp a) { witness(a, handleValueAs<FBoxed>(a.getVal())); },
      [&](fir::EmboxOp a) { witness(a, handleValue(a.getMemref())); },
      [&](fir::AddrOfOp a) {
        if (auto global = llvm::dyn_cast_if_present<fir::GlobalOp>(mlir::SymbolTable::lookupSymbolIn(m, a.getSymbol()))) {
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

            const auto rhs0 =
                *rhs ^ fold_total(
                           [&](const Expr::Any &x) {
                             return Expr::Annotated(Expr::Annotated(Expr::Poison(rhsTye), {}, fmt::format("IMPL?? {}", repr(x)))).widen();
                           },
                           [&](const FVar &x) { return x.value.widen(); },
                           [&](const FBoxed &x) { return Expr::Annotated(Expr::Annotated(Expr::Poison(rhsTye), {}, "IMPL")).widen(); },
                           [&](const FBoxedNone &x) { return Expr::Annotated(Expr::Annotated(Expr::Poison(rhsTye), {}, "IMPL")).widen(); });

            *lhs ^ foreach_total([&](const Expr::Any &x) { poison0(fmt::format("IMPL! {} = {}", repr(x), repr(rhs0))); }, //
                                 [&](const FVar &x) { push(Stmt::Mut(x.value, rhs0)); },                                  //
                                 [&](const FBoxed &x) { push(Stmt::Update(x.addr(), Expr::IntU64Const(0), rhs0)); },      //
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
        auto x = handleValue(c.getMemref());
        const auto ref = x ^ narrow<FBoxed, FVar, Expr::Any>(); // BoxedArray | Expr::Any + seq type
        if (!ref) {
          return poison(c, fmt::format("ERROR: Memref is not a FBoxed|Expr, the type is {}, expr is {}",
                                       show(static_cast<mlir::Type>(seqTy)), fRepr(x)));
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
          actualShape = iota<size_t>(0, ranks) | map([&](auto r) { return selectAny(boxed->dimAt(r), FDimM.extent); }) | to_vector();
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
        if (const auto ref = handleValue(l.getMemref()) ^ narrow<Expr::Any, FBoxed, FArrayCoord>()) {
          *ref ^ foreach_total(
                     [&](const Expr::Any &e) { witness(l.getResult(), Expr::Index(e, Expr::IntU64Const(0), handleType(l.getType()))); },
                     [&](const FArrayCoord &e) { witness(l.getResult(), Expr::Index(e.array, e.offset, e.comp)); }, //
                     [&](const FBoxed &e) { witness(l.getResult(), e); });
        } else poison0(fmt::format("Load RHS value not an Expr|FBoxed|FArrayCoord, was {}", show(l.getMemref())));
      },
      [&](fir::StoreOp s) {
        const auto rhs = handleValueAs(s.getValue());
        if (const auto lhs = handleValue(s.getMemref()) ^ narrow<Expr::Any, FVar, FBoxed, FArrayCoord>()) {
          *lhs ^ foreach_total([&](const Expr::Any &e) { push(Stmt::Update(e, Expr::IntU64Const(0), rhs)); },
                               [&](const FVar &a) { push(Stmt::Mut(a.value, rhs)); },
                               [&](const FArrayCoord &e) { push(Stmt::Update(e.array, e.offset, rhs)); },
                               [&](const FBoxed &e) { // TODO does this actually happen?
                                 push(Stmt::Update(e.addr(), Expr::IntU64Const(0), rhs));
                               });
        } else poison0(fmt::format("Store LHS value not an Expr|FBoxed|FArrayCoord, was {}", show(s.getValue())));
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

static std::vector<mlir::Value> findCapturesInOrder(mlir::Block *block) {
  std::vector<mlir::Value> captures;
  const auto args = block->getArguments();
  for (mlir::Operation &op : *block) {
    for (mlir::Value arg : op.getOperands()) {
      if (const mlir::Block *origin = arg.getParentBlock()) {
        if (origin != block                   //
            && !llvm::is_contained(args, arg) //
            && !llvm::isa<mlir::arith::ConstantOp>(arg.getDefiningOp())) {
          captures.emplace_back(arg);
        }
      }
    }
  }
  return captures ^ distinct_by([](const mlir::Value &op) { return op.getAsOpaquePointer(); });
}

std::vector<std::pair<Named, mlir::Value>> polyfc::Remapper::findCaptures(mlir::Block *op) {

  // fir.shift(origins: (lb: Int)[N]) -> !shift<N>
  // fir.shape(extents: (extent: Int)[N]) -> !shape<N>
  // fir.shape_shift(pairs: (lb: Int, extent: Int)[N]) -> !shapeshift<N>
  // fir.slice(triples: (lb: Int, ub: Int, step: Int)[N], fields: !field[N]?, substr: Int[N]?) ->!slice<M>

  // fir.field_index(<field_id>, <on_type>, typeparams: Int[N]?) ->!field

  // !(index, index, index) boxdims

  return findCapturesInOrder(op) //
         | zip_with_index()      //
         | map([&](auto &v, auto i) {
             return std::pair{Named(resolveUniqueName(v).value_or(fmt::format("arg_{}", i)), handleType(v.getType())), v};
           }) //
         | to_vector();
}

polyfc::Remapper::DoConcurrentRegion polyfc::Remapper::createRegion( //
    const std::string &name, bool gpu, mlir::ModuleOp &m, mlir::DataLayout &L, fir::DoLoopOp &op) {
  using namespace dsl;

  const static Named UpperBound("#upperBound", Long);
  const static Named LowerBound("#lowerBound", Long);
  const static Named Step("#step", Long);
  const static Named TripCount("#tripCount", Long);

  const static Named Begins("#begins", Ptr(Long));
  const static Named Ends("#ends", Ptr(Long));

  const static Type::Struct CaptureType("#Capture");
  const static Named Capture("#capture", Ptr(CaptureType));
  const static Named MappedInduction("#mappedInd", Long);

  const StructDef preludeDef("#Prelude", gpu ? std::vector{LowerBound, UpperBound, Step, TripCount} //
                                             : std::vector{LowerBound, UpperBound, Step, TripCount, Begins, Ends});
  const Named Prelude("#prelude", typeOf(preludeDef));

  op.getBody()->dump();
  Remapper r(m, L, op, Capture);

  r.valuesLUT.insert({op.getInductionVar(), selectNamed(MappedInduction)});
  for (auto &x : op.getBody()->getOperations()) {
    r.handleOp(&x);
  }

  const auto captures = r.captures                                                           //
                        | map([](auto &p) { return std::pair{tail(p.second)[0], p.first}; }) //
                        | to_vector();

  const StructDef capturesDef(CaptureType.name, captures | keys() | prepend(Prelude) | to_vector());
  r.syntheticDefs.emplace(preludeDef);
  r.syntheticDefs.emplace(capturesDef);

  const Function entry =
      gpu ? Function(name, {Arg(Capture, {})}, Unit, //
                     std::vector<Stmt::Any>{

                         let("#gs") = Cast(call(Spec::GpuGlobalSize(0_(UInt))), Long), //
                         let("#i") = 0_(Long),                                         //
                         ForRange("#i"_(Long), Cast(call(Spec::GpuGlobalIdx(0_(UInt))), Long), Select({Capture, Prelude}, TripCount),
                                  "#gs"_(Long),
                                  {
                                      let(MappedInduction.symbol) =
                                          call(Intr::Add(Select({Capture, Prelude}, LowerBound),
                                                         call(Intr::Mul("#i"_(Long), Select({Capture, Prelude}, Step), Long)), Long)), //
                                      Block(r.stmts),
                                  }),
                         // Block( {
                         //   let(MappedInduction.symbol) =Cast(call(Spec::GpuGlobalIdx(0_(UInt))), Long), //
                         //   Block(r.stmts),
                         // }),
                         ret()}, //
                     {FunctionAttr::Entry(), FunctionAttr::Exported()})
          : Function(name, {Arg("#group"_(Long), {}), Arg(Capture, {})}, Unit, //
                     std::vector<Stmt::Any>{
                         let("#i") = 0_(Long),
                         ForRange("#i"_(Long),                                                      //
                                  Index(Select({Capture, Prelude}, Begins), "#group"_(Long), Long), //
                                  Index(Select({Capture, Prelude}, Ends), "#group"_(Long), Long),   //
                                  1_(Long),
                                  {
                                      let(MappedInduction.symbol) =
                                          call(Intr::Add(Select({Capture, Prelude}, LowerBound),
                                                         call(Intr::Mul("#i"_(Long), Select({Capture, Prelude}, Step), Long)), Long)), //
                                      Block(r.stmts),
                                  }),
                         ret()}, //
                     {FunctionAttr::Entry(), FunctionAttr::Exported()});

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
      .boxes =
          r.boxTypes                                                                                                                     //
          | collect([](auto &k, auto &f) { return f ^ get_maybe<FBoxedMirror>() ^ map([&](auto &v) { return std::pair{k.name, v}; }); }) //
          | to<std::unordered_map>(),                                                                                                    //
      .preludeLayout = findNamedLayout(preludeDef.name),
      .captureLayout = findNamedLayout(capturesDef.name),
  };
}
