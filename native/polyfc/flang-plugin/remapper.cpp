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

const static Named FDimLowerBound("lowerBound", Type::IntS64());
const static Named FDimExtent("extent", Type::IntS64());
const static Named FDimStride("stride", Type::IntS64());

const static Type::Struct FDimTy("FDim");
const static StructDef FDim("FDim", {FDimLowerBound, FDimExtent, FDimStride});

const static Named BoxedFArraySizeInBytes("sizeInBytes", Type::IntS64());
const static Named BoxedFArrayRanks("ranks", Type::IntS64());
const static Named BoxedFArrayDims("dims", Type::Ptr(FDimTy, {}, TypeSpace::Global()));

polyfc::Remapper::BoxedFArrayAggregate::BoxedFArrayAggregate(const Type::Any &t)
    : comp(t), addr("addr", t), sizeInBytes(BoxedFArraySizeInBytes), ranks(BoxedFArrayRanks), dims(BoxedFArrayDims) {}
polyfc::Remapper::BoxedFArrayAggregate::BoxedFArrayAggregate() : BoxedFArrayAggregate(Type::Nothing()) {}
StructDef polyfc::Remapper::BoxedFArrayAggregate::def(const std::string &name) const {
  return StructDef(name, {addr, sizeInBytes, ranks, dims});
}

static Expr::Any selectNamed(const Expr::Any &base, const Named &that) {
  return base.get<Expr::Select>() ^                                        //
         fold([&](const auto &s) { return selectNamed(s, that).widen(); }, //
              [&] { return Expr::Poison(that.tpe).widen(); });
}

polyfc::Remapper::FBoxedArray::FBoxedArray(const Expr::Any &base, const BoxedFArrayAggregate &aggregate)
    : base(base), aggregate(aggregate) {}
polyfc::Remapper::FBoxedArray::FBoxedArray() : FBoxedArray(Expr::Poison(Type::Nothing()), {}) {}
Type::Any polyfc::Remapper::FBoxedArray::comp() const { return aggregate.comp; }
Expr::Any polyfc::Remapper::FBoxedArray::addr() const { return selectNamed(base, aggregate.addr); }
Expr::Any polyfc::Remapper::FBoxedArray::sizeInBytes() const { return selectNamed(base, aggregate.sizeInBytes); }
Expr::Any polyfc::Remapper::FBoxedArray::ranks() const { return selectNamed(base, aggregate.ranks); }
Expr::Any polyfc::Remapper::FBoxedArray::dims() const { return selectNamed(base, aggregate.dims); }
Expr::Any polyfc::Remapper::FBoxedArray::dimAt(const size_t rank) const {
  return Expr::Index(selectNamed(base, aggregate.dims), Expr::IntS64Const(rank), FDimTy);
}

polyfc::Remapper::Remapper(mlir::MLIRContext *C, mlir::DataLayout &L, mlir::Operation *perimeter, const Named &captureRoot)
    : C(C), L(L), perimeter(perimeter), captureRoot(captureRoot) {}

mlir::Type polyfc::Remapper::resolveType(const Type::Any &tpe) {
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
        const auto def = defs                                    //
                         | map([](auto &p) { return p.second; }) //
                         | concat(syntheticDefs)                 //
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
  const auto mirror = mlir::LLVM::LLVMStructType::getLiteral(C, fields | values() | to_vector());
  const auto alignment = L.getTypeABIAlignment(mirror);
  std::vector<StructLayoutMember> ms;
  size_t offset = 0;
  for (const auto &[named, ty] : fields) {
    uint64_t size = L.getTypeSize(ty);
    ms.emplace_back(named, offset, size);
    offset += std::max(size, alignment);
  }
  return StructLayout(def.name, L.getTypeSize(mirror), alignment, ms);
}

Expr::Select polyfc::Remapper::newVar(const Expr::Any &expr) {
  static size_t id = 0;
  const Named name(fmt::format("v{}", id++), expr.tpe());
  stmts.emplace_back(Stmt::Var(name, expr));
  return Expr::Select({}, name);
}

Type::Any polyfc::Remapper::handleType(const mlir::Type type) {

  auto handleSeq = [&](const fir::SequenceType t) -> Type::Any {
    const auto dynamic = t.hasDynamicExtents() || t.hasUnknownShape();
    return Type::Ptr(handleType(t.getEleTy()), !dynamic ? std::optional<int32_t>{} : std::optional<int32_t>{}, TypeSpace::Global());
  };

  auto handleBox = [&](const fir::BoxType t) -> std::pair<Type::Any, BoxedFArrayAggregate> {
    const auto comp = handleType(t.getEleTy());
    const auto name = fmt::format("FArray<{}>", repr(comp));
    BoxedFArrayAggregate boxed(handleType(t.getEleTy()));
    boxedArrays.insert({t, boxed});
    defs.insert({t, boxed.def(name)});
    syntheticDefs.insert(FDim);
    return {Type::Struct(name), boxed};
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
             [&](const fir::HeapType t) -> Type::Any { return handleType(t.getEleTy()); },
             [&](fir::ReferenceType t) -> Type::Any {
               // XXX special case for Seq and Box, both are flattened
               if (const auto seqTy = llvm::dyn_cast<fir::SequenceType>(t.getEleTy())) return handleSeq(seqTy);
               if (const auto boxTy = llvm::dyn_cast<fir::BoxType>(t.getEleTy())) { // record both
                 auto [tpe, boxed] = handleBox(boxTy);
                 boxedArrays.insert({t, boxed});
                 return Type::Ptr(tpe, {}, TypeSpace::Global());
               }
               // Otherwise use normal pointer semantic
               return Type::Ptr(handleType(t.getEleTy()), {}, TypeSpace::Global());
             },
             [&](const fir::SequenceType t) -> Type::Any { return handleSeq(t); },
             //             [&](fir::RecordType t) -> Type::Any {
             //               for (auto x : t.getTypeList()) {
             //
             //                 StructLayoutMember()
             //               }
             //             }, //
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
      const Named field(resolveUniqueName(val).value_or(fmt::format("arg_{}", id)), handleType(val.getType()));
      const Expr::Select select({captureRoot}, field);
      captures.insert({val, select});
      if (const auto it = boxedArrays.find(val.getType()); it != boxedArrays.end()) {
        FBoxedArray arr(select, it->second);
        valuesLUT.insert({val, arr});
        return arr;
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

  auto poison = [&](auto x, const std::string &reason) {
    witness(x, Expr::Poison(handleType(x.getType())));
    stmts.emplace_back(Stmt::Comment(reason));
  };

  auto poison0 = [&](const std::string &reason) { stmts.emplace_back(Stmt::Comment(reason)); };

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
        const auto boxSelect = handleValueAs<FBoxedArray>(dims.getVal());
        const auto rank = handleValueAs(dims.getDim());

        const auto dim = newVar(Expr::Index(boxSelect.dims(), rank, FDimTy));
        witness(dims.getLowerBound(), selectNamed(dim, FDimLowerBound));
        witness(dims.getExtent(), selectNamed(dim, FDimExtent));
        witness(dims.getByteStride(), selectNamed(dim, FDimStride));
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

      [&](fir::ConvertOp c) { witness(c, Expr::Cast(handleValueAs(c.getOperand()), handleType(c.getType()))); },
      [&](fir::BoxAddrOp a) { witness(a, handleValueAs<FBoxedArray>(a.getVal())); },

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
          poison(c, fmt::format("ERROR: Memref {} type does not contain a sequence type", show(static_cast<mlir::Type>(seqTy))));
          return;
        }
        const auto ref = handleValue(c.getMemref()) ^ narrow<FBoxedArray, Expr::Any>(); // BoxedArray | Expr::Any + seq type
        if (!ref) {
          poison(c, fmt::format("ERROR: Memref is not a FBoxedArray|Expr, the type is {}", show(static_cast<mlir::Type>(seqTy))));
          return;
        }
        if (seqTy.hasUnknownShape()) {
          poison(c, fmt::format("IMPL: Unknown shape not implemented yet, op={}", show(c)));
          return;
        }

        const auto ones = [](auto n) { return repeat(Expr::IntS64Const(1).widen()) | take(n) | to_vector(); };
        const size_t ranks = seqTy.getDimension();

        std::vector<Expr::Any> actualShape;
        if (seqTy.hasDynamicExtents()) {
          std::optional<FBoxedArray> boxed = *ref ^ get_maybe<FBoxedArray>();
          if (!boxed) {
            poison(c, fmt::format("ERROR: array ({}) has dynamic extent but us not boxed", show(c.getMemref())));
            return;
          }
          actualShape = iota<size_t>(0, ranks) | map([&](auto r) { return selectNamed(boxed->dimAt(r), FDimExtent); }) | to_vector();
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
            poison(c, fmt::format("ERROR: invariant: {} array size != rank ({})", name, ranks));
            return;
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

          stmts.emplace_back(Mut(offset, call(Intr::Add(offset, call(Intr::Mul(diff, stride, Long)), Long)))); // offset += diff * stride[i]
          stmts.emplace_back(Mut(stride, call(Intr::Mul(stride, shape[i], Long))));                            // stride *= shape[i]
        }
        witness(c.getResult(), *ref ^ fold_total([&](const FBoxedArray &e) { return ArrayCoord{e.addr(), offset, e.comp()}; }, //
                                                 [&](const Expr::Any &e) { return ArrayCoord{e, offset, handleType(seqTy.getEleTy())}; }));
      },
      [&](fir::LoadOp l) {
        if (const auto ref = handleValue(l.getMemref()) ^ narrow<Expr::Any, FBoxedArray, ArrayCoord>()) {
          *ref ^ foreach_total(
                     [&](const Expr::Any &e) { witness(l.getResult(), Expr::Index(e, Expr::IntU64Const(0), handleType(l.getType()))); },
                     [&](const ArrayCoord &e) { poison(l.getResult(), "IMPL load array coord"); }, //
                     [&](const FBoxedArray &e) { witness(l.getResult(), e); });
        } else poison0(fmt::format("Load RHS value not an Expr|FBoxedArray|ArrayCoord, was {}", show(l.getMemref())));
      },
      [&](fir::StoreOp s) {
        const auto rhs = handleValueAs(s.getValue());
        if (const auto lhs = handleValue(s.getMemref()) ^ narrow<Expr::Any, FBoxedArray, ArrayCoord>()) {
          *lhs ^ foreach_total([&](const Expr::Any &e) { stmts.emplace_back(Stmt::Update(e, Expr::IntU64Const(0), rhs)); },
                               [&](const ArrayCoord &e) { stmts.emplace_back(Stmt::Update(e.array, e.offset, rhs)); },
                               [&](const FBoxedArray &e) { // TODO does this actually happen?
                                 stmts.emplace_back(Stmt::Update(e.addr(), Expr::IntU64Const(0), rhs));
                               });
        } else poison0(fmt::format("Store LHS value not an Expr|FBoxedArray|ArrayCoord, was {}", show(s.getValue())));
      },
      [&](const fir::ResultOp r) {
        //        if (const auto xs = r.getResults(); xs.empty()) {
        //          stmts.emplace_back(Stmt::Return(Expr::Unit0Const()));
        //        } else if (xs.size() > 1) {
        //          stmts.emplace_back(Stmt::Comment(fmt::format("ERROR: Multiple values in return: {}", show(r))));
        //        } else {
        //          stmts.emplace_back(Stmt::Return(handleValue(xs.front())));
        //        }
        stmts.emplace_back(Stmt::Comment(fmt::format("ignored op {}", show(r))));
      });
  if (!handled) {
    stmts.emplace_back(Stmt::Comment(fmt::format("ERROR: Unsupported MLIR op {}", show(op))).widen());
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
    const std::string &name, bool gpu, mlir::DataLayout &L, fir::DoLoopOp &op) {
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
  Remapper r(op.getContext(), L, op, Capture);

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

  const Program program(r.defs                                     //
                            | map([](auto p) { return p.second; }) //
                            | concat(r.syntheticDefs)              //
                            | to_vector(),                         //
                        r.functions | append(entry) | to_vector());

  llvm::errs() << repr(program) << "\n";

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
      .preludeLayout = findNamedLayout(preludeDef.name),
      .captureLayout = findNamedLayout(capturesDef.name),
  };
}
