#include <algorithm>
#include <cstdlib>
#include <optional>
#include <unordered_map>
#include <vector>

#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Support/InternalNames.h"
#include "flang/Optimizer/Support/Utils.h"
#include "flang/Optimizer/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/Path.h"

#include "aspartame/all.hpp"
#include "polyfront/options_backend.hpp"
#include "polyregion/types.h"

#include "codegen.h"
#include "mlir_utils.h"
#include "remapper.h"
#include "rewriter.h"
#include "utils.h"

using namespace aspartame;

namespace {

using namespace polyregion::polyfc;
using namespace polyregion;
using namespace mlir;

std::optional<std::string> resolveUniqueName(const Value value) {
  if (const auto decl = llvm::dyn_cast_if_present<fir::DeclareOp>(value.getDefiningOp())) {
    if (const auto strAttr = decl->getAttrOfType<StringAttr>("uniq_name")) {
      return strAttr.str();
    }
  }
  return {};
}

struct CharStarMirror final : AggregateMirror<1> {
  Field<LLVM::LLVMPointerType, 0> ptr;
  explicit CharStarMirror(MLIRContext *C) : AggregateMirror(C), ptr(C) {}
  const char *typeName() const override { return "CharStar"; }
  std::array<Type, 1> types() const override { return {ptr.widen()}; }
};

struct TypeLayoutMirror final : AggregateMirror<5> {
  Field<LLVM::LLVMPointerType, 0> name;
  Field<IntegerType, 1> sizeInBytes;
  Field<IntegerType, 2> alignmentInBytes;
  Field<IntegerType, 3> memberCount;
  Field<LLVM::LLVMPointerType, 4> members;
  explicit TypeLayoutMirror(MLIRContext *C)
      : AggregateMirror(C), name(C), sizeInBytes(C, 64), alignmentInBytes(C, 64), memberCount(C, 64), members(C) {}
  const char *typeName() const override { return "TypeLayout"; }
  std::array<Type, 5> types() const override {
    return {name.widen(), sizeInBytes.widen(), alignmentInBytes.widen(), memberCount.widen(), members.widen()};
  }
};

struct KernelObjectMirror final : AggregateMirror<6> {
  Field<IntegerType, 0> kind;
  Field<IntegerType, 1> structCount;
  Field<IntegerType, 2> featureCount;
  Field<LLVM::LLVMPointerType, 3> features;
  Field<IntegerType, 4> imageLength;
  Field<LLVM::LLVMPointerType, 5> image;
  explicit KernelObjectMirror(MLIRContext *C)
      : AggregateMirror(C), kind(C, 8), structCount(C, 8), featureCount(C, 64), features(C), imageLength(C, 64), image(C) {}
  const char *typeName() const override { return "KernelObject"; }
  std::array<Type, 6> types() const override {
    return {kind.widen(),         //
            structCount.widen(),  //
            featureCount.widen(), //
            features.widen(),     //
            imageLength.widen(),  //
            image.widen()};
  }
};

struct KernelBundleMirror final : AggregateMirror<7> {
  Field<LLVM::LLVMPointerType, 0> moduleName;
  Field<IntegerType, 1> objectCount;
  Field<LLVM::LLVMPointerType, 2> objects;
  Field<IntegerType, 3> structCount;
  Field<LLVM::LLVMPointerType, 4> structs;
  Field<IntegerType, 5> interfaceLayoutIdx;
  Field<LLVM::LLVMPointerType, 6> metadata;
  explicit KernelBundleMirror(MLIRContext *C)
      : AggregateMirror(C), moduleName(C), objectCount(C, 64), objects(C), structCount(C, 64), structs(C), interfaceLayoutIdx(C, 64),
        metadata(C) {}
  const char *typeName() const override { return "KernelBundle"; }
  std::array<Type, 7> types() const override {
    return {moduleName.widen(),         //
            objectCount.widen(),        //
            objects.widen(),            //
            structCount.widen(),        //
            structs.widen(),            //
            interfaceLayoutIdx.widen(), //
            metadata.widen()};
  }
};

struct AggregateMemberMirror final : AggregateMirror<6> {
  Field<LLVM::LLVMPointerType, 0> name;
  Field<IntegerType, 1> offsetInBytes;
  Field<IntegerType, 2> sizeInBytes;
  Field<IntegerType, 3> ptrIndirection;
  Field<IntegerType, 4> componentSize;
  Field<LLVM::LLVMPointerType, 5> type;
  explicit AggregateMemberMirror(MLIRContext *C)
      : AggregateMirror(C), name(C), offsetInBytes(C, 64), sizeInBytes(C, 64), ptrIndirection(C, 64), componentSize(C, 64), type(C) {}
  const char *typeName() const override { return "AggregateMember"; }
  std::array<Type, 6> types() const override {
    return {name.widen(), offsetInBytes.widen(), sizeInBytes.widen(), ptrIndirection.widen(), componentSize.widen(), type.widen()};
  }
};

struct FDimMirror final : AggregateMirror<3> {
  Field<IntegerType, 0> lowerBound;
  Field<IntegerType, 1> extent;
  Field<IntegerType, 2> stride;
  explicit FDimMirror(MLIRContext *C) : AggregateMirror(C), lowerBound(C, 64), extent(C, 64), stride(C, 64) {}
  const char *typeName() const override { return "FDim"; }
  std::array<Type, 3> types() const override { return {lowerBound.widen(), extent.widen(), stride.widen()}; }
};

struct FArrayDescMirror final : AggregateMirror<4> {
  Field<LLVM::LLVMPointerType, 0> addr;
  Field<IntegerType, 1> sizeInBytes;
  Field<IntegerType, 2> ranks;
  Field<LLVM::LLVMPointerType, 2> rankDims;
  explicit FArrayDescMirror(MLIRContext *C) : AggregateMirror(C), addr(C), sizeInBytes(C, 64), ranks(C, 64), rankDims(C) {}
  const char *typeName() const override { return "FArrayDesc"; }
  std::array<Type, 4> types() const override { return {addr.widen(), sizeInBytes.widen(), ranks.widen(), rankDims.widen()}; }
};

class PolyDCOMirror {
  OpBuilder TLB;
  LLVM::LLVMPointerType ptrTy;
  LLVM::LLVMVoidType voidTy;
  LLVM::LLVMFuncOp recordFn, releaseFn, debugArrayDescFn, debugLayoutFn, isPlatformKindFn, dispatchFn;

  Value valueOf(OpBuilder &B, const runtime::PlatformKind kind) { return intConst(B, TLB.getI8Type(), value_of(kind)); }

  Value convertIfNeeded(OpBuilder &B, Value value, Type required) {
    return value.getType() != required ? B.create<fir::ConvertOp>(TLB.getUnknownLoc(), required, value) : value;
  }

public:
  explicit PolyDCOMirror(ModuleOp &m)                                                                                   //
      : TLB(m), ptrTy(LLVM::LLVMPointerType::get(TLB.getContext())), voidTy(LLVM::LLVMVoidType::get(TLB.getContext())), //
        recordFn(defineFunc(m, "polydco_record", voidTy, {ptrTy, TLB.getI64Type()})),
        releaseFn(defineFunc(m, "polydco_release", voidTy, {ptrTy})),
        debugArrayDescFn(defineFunc(m, "polydco_debug_farraydesc", voidTy, {ptrTy})),
        debugLayoutFn(defineFunc(m, "polydco_debug_typelayout", voidTy, {ptrTy})),
        isPlatformKindFn(defineFunc(m, "polydco_is_platformkind", TLB.getI1Type(), {TLB.getI8Type()})),
        dispatchFn(defineFunc(m, "polydco_dispatch", TLB.getI1Type(),
                              {/* lowerBound   */ TLB.getI64Type(),
                               /* upperBound   */ TLB.getI64Type(),
                               /* step         */ TLB.getI64Type(),
                               /* platformKind */ TLB.getI8Type(),
                               /* bundle       */ ptrTy,
                               /* captures     */ ptrTy})) {}

  void record(OpBuilder &B, const Value ptr, const Value sizeInBytes) {
    B.create<LLVM::CallOp>(B.getUnknownLoc(), recordFn, ValueRange{ptr, convertIfNeeded(B, sizeInBytes, B.getI64Type())});
  }

  void release(OpBuilder &B, const Value ptr) { B.create<LLVM::CallOp>(B.getUnknownLoc(), releaseFn, ValueRange{ptr}); }

  Value isPlatformKind(OpBuilder &B, runtime::PlatformKind kind) {
    return B.create<LLVM::CallOp>(B.getUnknownLoc(), isPlatformKindFn, ValueRange{valueOf(B, kind)}).getResult();
  }

  Value dispatch(OpBuilder &B, const Value lowerBound, const Value upperBound, const Value step, const runtime::PlatformKind kind,
                 const Value bundle, const Value captures) {
    return B
        .create<LLVM::CallOp>(B.getUnknownLoc(), dispatchFn,
                              ValueRange{convertIfNeeded(B, lowerBound, B.getI64Type()), //
                                         convertIfNeeded(B, upperBound, B.getI64Type()), //
                                         convertIfNeeded(B, step, B.getI64Type()),       //
                                         valueOf(B, kind), bundle, captures})
        .getResult();
  }
};

class Binder {

  struct Field {
    struct Witness {
      Value ptr, sizeInBytes;
    };
    Type type;
    Value fieldPtr;
    std::vector<Witness> dependent, temporary;
  };

  PolyDCOMirror &dco;

  FDimMirror FDim;
  FArrayDescMirror FArrayDesc;
  LLVM::LLVMArrayType preludeTy;
  std::vector<Field> fields;
  DynamicAggregateMirror mirror;

  static Field bind(OpBuilder &B, const DataLayout &L, const FDimMirror &FDim, const FArrayDescMirror &FArrayDesc, Value ref) {
    const auto Loc = B.getUnknownLoc();
    const auto ptrTy = LLVM::LLVMPointerType::get(B.getContext());
    const auto i64Ty = B.getI64Type();

    if (const auto refTy = llvm::dyn_cast<fir::ReferenceType>(ref.getType())) { // T = fir.ref<E>
      const auto elemTy = refTy.getEleTy();
      if (auto seqTy = llvm::dyn_cast<fir::SequenceType>(elemTy)) { // E = fir.array<X>
        if (seqTy.hasDynamicExtents() || seqTy.hasUnknownShape())
          raise(fmt::format("Array has dynamic extent or unknown shape: {}", fir::mlirTypeToString(elemTy)));
        const auto maxExtent = intConst(B, i64Ty, seqTy.getConstantArraySize() * (seqTy.getEleTy().getIntOrFloatBitWidth() / 8));
        const auto i64Addr = B.create<fir::ConvertOp>(Loc, i64Ty, ref).getRes();
        const auto llvmPtr = B.create<LLVM::IntToPtrOp>(Loc, ptrTy, i64Addr).getRes();
        return Field{
            .type = refTy,
            .fieldPtr = llvmPtr,
            .dependent = {Field::Witness{llvmPtr, maxExtent}},
            .temporary = {},
        };
      } else if (elemTy.isIntOrFloat()) {
        const auto i64Addr = B.create<fir::ConvertOp>(Loc, i64Ty, ref).getRes();
        const auto llvmPtr = B.create<LLVM::IntToPtrOp>(Loc, ptrTy, i64Addr).getRes();
        const auto i64Size = intConst(B, i64Ty, elemTy.getIntOrFloatBitWidth() / 8);
        return Field{
            .type = refTy,
            .fieldPtr = llvmPtr,
            .dependent = {Field::Witness{llvmPtr, i64Size}},
            .temporary = {},
        };
      } else if (const auto boxTy = llvm::dyn_cast<fir::BoxType>(elemTy)) { // E = fir.box<X>
        const auto opaqueGEP = B.create<fir::BoxOffsetOp>(Loc, ref, fir::BoxFieldAttr::base_addr).getResult();
        const auto i64GEPAddr = B.create<fir::ConvertOp>(Loc, i64Ty, opaqueGEP).getRes();
        const auto llvmGEPPtr = B.create<LLVM::IntToPtrOp>(Loc, ptrTy, i64GEPAddr).getRes();
        const auto llvmPtr = B.create<LLVM::LoadOp>(Loc, ptrTy, llvmGEPPtr).getRes();
        const auto boxVal = B.create<fir::LoadOp>(Loc, ref).getRes();
        auto maxExtent = B.create<fir::BoxEleSizeOp>(Loc, i64Ty, boxVal).getResult();
        FDimMirror::Group rankDimValues{};
        for (size_t dim = 0; dim < fir::getBoxRank(boxTy); ++dim) {
          auto rank = intConst(B, i64Ty, dim);
          auto boxDims = B.create<fir::BoxDimsOp>(Loc, boxVal, rank);
          auto extent = B.create<fir::ConvertOp>(Loc, i64Ty, boxDims.getExtent());
          maxExtent = B.create<arith::MulIOp>(Loc, maxExtent, extent).getResult();
          rankDimValues.emplace_back(
              std::array{/* lowerBound */ B.create<fir::ConvertOp>(Loc, i64Ty, boxDims.getLowerBound()).getResult(),
                         /* extent     */ B.create<fir::ConvertOp>(Loc, i64Ty, boxDims.getExtent()).getResult(),
                         /* stride     */ B.create<fir::ConvertOp>(Loc, i64Ty, boxDims.getByteStride()).getResult()});
        }
        const auto rankDims = FDim.local(B, rankDimValues);
        const auto arrayDesc = FArrayDesc.local(B, std::vector{FArrayDescMirror::Init{
                                                       /* addr        */ llvmPtr,
                                                       /* sizeInBytes */ maxExtent,
                                                       /* ranks       */ intConst(B, i64Ty, rankDimValues.size()),
                                                       /* rankDims    */ rankDims,
                                                   }});
        return Field{
            .type = refTy,
            .fieldPtr = arrayDesc,
            .dependent = {Field::Witness{llvmPtr, maxExtent}},
            .temporary = {Field::Witness{arrayDesc, intConst(B, i64Ty, L.getTypeSize(FArrayDesc.structTy()))},
                          Field::Witness{rankDims, intConst(B, i64Ty, L.getTypeSize(FDim.structTy()) * rankDimValues.size())}},
        };
      } else raise(fmt::format("Unhandled binder type: {}", fir::mlirTypeToString(elemTy)));
    } else raise(fmt::format("Value is not a ref type: {}", show(ref)));
  }

public:
  Binder(OpBuilder &B, DataLayout &L, PolyDCOMirror &dco, const std::string &name, const size_t preludeSize,
         const std::vector<Value> &refs) //
      : dco(dco), FDim(B.getContext()), FArrayDesc(B.getContext()),
        preludeTy(LLVM::LLVMArrayType::get(B.getContext(), B.getI8Type(), preludeSize)),
        fields(refs ^ map([&](auto &ref) { return bind(B, L, FDim, FArrayDesc, ref); })),
        mirror(B.getContext(), "Binder_" + name,
               fields                                                  //
                   | map([](auto &f) { return f.fieldPtr.getType(); }) //
                   | prepend(preludeTy)                                //
                   | to_vector()) {}

  Type structType() const { return mirror.ty; }

  Value create(OpBuilder &B) const {

    // ;B.create<LLVM::ConstantOp>(uLoc(B), preludeTy, ArrayAttr::get(B.getContext(),zeros  ) )

    auto zeros = repeat<Attribute>(IntegerAttr::get(B.getI8Type(), 0)) | take(preludeTy.getNumElements()) | to_vector();
    return mirror.local(B, {fields                                                         //
                            | map([](auto &f) { return f.fieldPtr; })                      //
                            | prepend(B.create<LLVM::ZeroOp>(uLoc(B), preludeTy).getRes()) //
                            | to_vector()});
  }

  void recordTemporariesAndDependents(OpBuilder &B) {
    for (auto &field : fields) {
      field.dependent | concat(field.temporary) | for_each([&](auto &w) { dco.record(B, w.ptr, w.sizeInBytes); });
    }
  }

  void releaseTemporaries(OpBuilder &B) {
    for (auto &field : fields) {
      field.temporary | for_each([&](auto &w) { dco.release(B, w.ptr); });
    }
  }
};

class Rewriter {
  ModuleOp &m;
  PolyDCOMirror dco;
  CharStarMirror CharStar;
  KernelObjectMirror KernelObject;
  KernelBundleMirror KernelBundle;
  AggregateMemberMirror AggregateMember;
  TypeLayoutMirror TypeLayout;

  std::unordered_map<polyast::Type::Any, TypeLayoutMirror::Global> primitiveTypeLayouts;

public:
  explicit Rewriter(ModuleOp &m)
      : m(m), dco(m),                                                                                                            //
        CharStar(m.getContext()),                                                                                                //
        KernelObject(m.getContext()), KernelBundle(m.getContext()), AggregateMember(m.getContext()), TypeLayout(m.getContext()), //
        primitiveTypeLayouts(std::vector<polyast::Type::Any>{
                                 polyast::Type::Float16(), polyast::Type::Float32(), polyast::Type::Float64(),                      //
                                 polyast::Type::IntU8(), polyast::Type::IntU16(), polyast::Type::IntU32(), polyast::Type::IntU64(), //
                                 polyast::Type::IntS8(), polyast::Type::IntS16(), polyast::Type::IntS32(), polyast::Type::IntS64(), //
                                 polyast::Type::Unit0(), polyast::Type::Bool1(),                                                    //
                             } //
                             | collect([&](auto &t) {
                                 return polyast::primitiveSize(t) ^ map([&](auto sizeInBytes) {
                                          return std::pair{t, TypeLayout.global(m, [&](OpBuilder &B0) {
                                                             return std::vector{std::array{
                                                                 strConst(B0, m, polyast::repr(t)),    //
                                                                 intConst(B0, i64Ty(B0), sizeInBytes), //
                                                                 intConst(B0, i64Ty(B0), sizeInBytes), //
                                                                 intConst(B0, i64Ty(B0), 0),           //
                                                                 nullConst(B0)                         //
                                                             }};
                                                           })};
                                        }); //
                               })           //
                             | to<std::unordered_map>())

  {}

  void ifKindEq(OpBuilder &B, const polyfront::Options &opts, const runtime::PlatformKind kind,
                const std::function<void(OpBuilder &, runtime::PlatformKind)> &f) {
    if (!(opts.targets ^ exists([&](auto &t, auto) { return runtime::targetPlatformKind(t) == kind; }))) return;
    auto B0 = B.create<fir::IfOp>(uLoc(B), dco.isPlatformKind(B, kind), false).getThenBodyBuilder();
    f(B0, kind);
  }

  void invokeDispatch(OpBuilder &B, Value executeOriginal, const runtime::PlatformKind kind, clang::DiagnosticsEngine &diag,
                      const std::string &diagLoc, const polyfront::Options &opts, const std::string &moduleId, fir::DoLoopOp &doLoop) {
    DataLayout L(m);
    const auto region = Remapper::createRegion("_main", kind == runtime::PlatformKind::Managed, L, doLoop);
    const auto bundle = compileRegion(diag, diagLoc, opts, kind, moduleId, region);
    const auto table = bundle.layouts                                                                 //
                       | values()                                                                     //
                       | map([&](auto &sl) { return std::pair{polyast::Type::Struct(sl.name), sl}; }) //
                       | to<std::unordered_map>();

    auto structLayoutsArray = TypeLayout.global(m, [&](OpBuilder &B0) {
      return bundle.layouts ^ map([&](auto, auto &l) {
               return std::array{
                   strConst(B0, m, l.name),                   //
                   intConst(B0, i64Ty(B0), l.sizeInBytes),    //
                   intConst(B0, i64Ty(B0), l.alignment),      //
                   intConst(B0, i64Ty(B0), l.members.size()), //
                   nullConst(B0)                              //
               };
             });
    });

    const auto structNameToTypeLayoutIdx = bundle.layouts                          //
                                           | values()                              //
                                           | map([](auto &sl) { return sl.name; }) //
                                           | zip_with_index()                      //
                                           | to<std::unordered_map>();

    auto aggregateMembersArray =
        bundle.layouts     //
        | values()         //
        | zip_with_index() //
        | map([&](auto &l, auto idx) {
            auto g = AggregateMember.global(m, [&](OpBuilder &B0) {
              return l.members ^ map([&](auto &x) {
                       const auto [indirections, componentSize] = countIndirectionsAndComponentSize(x.name.tpe, table);
                       const auto ptrToTypeLayout =
                           polyast::extractComponent(x.name.tpe) ^ flat_map([&](auto &t) {
                             return primitiveTypeLayouts                                        //
                                    ^ get_maybe(t) ^ map([&](auto ptl) { return ptl.gep(B0); }) //
                                    ^ or_else(t.template get<polyast::Type::Struct>() ^ flat_map([&](auto &s) {
                                                return structNameToTypeLayoutIdx                                                     //
                                                       ^ get_maybe(s.name)                                                           //
                                                       ^ map([&](auto layoutIdx) { return structLayoutsArray.gep(B0, layoutIdx); }); //
                                              }));                                                                                   //
                           });                                                                                                       //
                       return std::array{strConst(B0, m, x.name.symbol),                                                             //
                                         intConst(B0, i64Ty(B0), x.offsetInBytes),                                                   //
                                         intConst(B0, i64Ty(B0), x.sizeInBytes),                                                     //
                                         intConst(B0, i64Ty(B0), indirections),                                                      //
                                         intConst(B0, i64Ty(B0), componentSize.value_or(x.sizeInBytes)),                             //
                                         ptrToTypeLayout ^ get_or_else(nullConst(B0))};
                     });
            });
            return std::pair{g, idx};
          }) //
        | to_vector();

    static size_t id = 0;
    defineGlobalCtor(m, fmt::format("dco_layoutInit_{}_{}", to_string(kind), ++id), [&](OpBuilder &FB) {
      aggregateMembersArray | for_each([&](auto &g, auto idx) {
        FB.create<LLVM::StoreOp>(uLoc(FB), g.gep(FB), structLayoutsArray.gep(FB, idx, TypeLayout.members));
      });
      FB.create<LLVM::ReturnOp>(uLoc(FB), ValueRange{});
    });

    auto globalKOs = KernelObject.global(m, [&](OpBuilder &B0) {
      return bundle.objects ^ map([&](auto &o) {
               fprintf(stderr, "!!!! %s\n", (o.features ^ mk_string("+")).c_str());
               auto features = CharStar.global(
                   m, [&](OpBuilder &B1) { return o.features ^ map([&](auto &f) { return std::array{strConst(B1, m, f)}; }); });
               return KernelObjectMirror::Init{intConst(B0, i8Ty(B0), value_of(o.kind)),      //
                                               intConst(B0, i8Ty(B0), value_of(o.format)),    //
                                               intConst(B0, i64Ty(B0), o.features.size()),    //
                                               features.gep(B0),                              //
                                               intConst(B0, i64Ty(B0), o.moduleImage.size()), //
                                               strConst(B0, m, o.moduleImage, false)};
             });
    });

    auto globalBundle = KernelBundle.global(m, [&](OpBuilder &B0) {
      return std::vector{std::array{strConst(B0, m, moduleId), //

                                    intConst(B0, i64Ty(B0), bundle.objects.size()), //
                                    globalKOs.gep(B0),                              //

                                    intConst(B0, i64Ty(B0), bundle.layouts.size()), //
                                    structLayoutsArray.gep(B0),
                                    intConst(B0, i64Ty(B0), bundle.layouts | index_where([](auto &iface, auto) { return iface; })),

                                    strConst(B0, m, bundle.metadata)}};
    });

    for (auto [x, v] : region.captures) {
      llvm::errs() << ">>>>>> " << repr(x) << " = " << v << "\n";
    }

    Binder binder(B, L, dco, moduleId, region.preludeLayout.sizeInBytes, region.captures | values() | to_vector());

    if (int64_t binderSize = L.getTypeSize(binder.structType()); binderSize != region.captureLayout.sizeInBytes) {
      raise(fmt::format("Capture and binder type size mismatch, expecting {} but binder gave {}", //
                        region.captureLayout.sizeInBytes, binderSize));
    }

    binder.recordTemporariesAndDependents(B);

    auto capture = binder.create(B);
    auto dispatch = dco.dispatch(B,
                                 doLoop.getLowerBound(), //
                                 doLoop.getUpperBound(), //
                                 doLoop.getStep(),       //
                                 kind,                   //
                                 globalBundle.gep(B),    //
                                 capture);
    auto noDispatch = B.create<arith::XOrIOp>(uLoc(B), dispatch, boolConst(B, true));
    auto noDispatchIf = B.create<fir::IfOp>(uLoc(B), noDispatch, true);
    {
      auto ifNoDispatchB = noDispatchIf.getThenBodyBuilder();
      ifNoDispatchB.create<LLVM::StoreOp>(uLoc(B), boolConst(ifNoDispatchB, true), executeOriginal);
    }
    {
      auto ifDispatchB = noDispatchIf.getElseBodyBuilder();
      binder.releaseTemporaries(ifDispatchB);
    }
  }
};

constexpr auto DoConcurrentAsWritten = "dco-as-written";

void doRewrite(ModuleOp op) {

  constexpr auto InductionStoreHoisted = "dco-induction-store-hoisted";
  constexpr auto HoistedStoreOp = "dco-hoisted-store-op";

  // this is OK because redefinition of DO variable is not legal (e.g. so we don't expect a store)
  struct HoistInductionStore : OpRewritePattern<fir::DoLoopOp> {
    using OpRewritePattern ::OpRewritePattern;
    LogicalResult matchAndRewrite(fir::DoLoopOp loopOp, PatternRewriter &R) const override {
      if (!loopOp->hasAttr(DoConcurrentAsWritten) || loopOp->hasAttr(InductionStoreHoisted)) return failure();
      const auto induction = loopOp.getInductionVar();
      if (!induction.hasOneUse()) return failure();
      auto convertOp = llvm::dyn_cast<fir::ConvertOp>(*induction.user_begin());
      if (!convertOp || !convertOp->hasOneUse()) return failure();
      auto storeOp = llvm::dyn_cast<fir::StoreOp>(*convertOp->user_begin());
      if (!storeOp) return failure();
      auto inductionRef = storeOp.getMemref(); // this is the outer scope induction capture
      for (auto &op : loopOp.getBody()->getOperations()) {
        if (auto loadOp = llvm::dyn_cast<fir::LoadOp>(op); loadOp && loadOp.getMemref() == inductionRef) {
          R.replaceAllUsesWith(loadOp.getRes(), convertOp.getRes());
        }
      }

      // hoist all the way out
      Operation *parentLoopOp = loopOp;
      while (const auto parentLoop = llvm::dyn_cast<fir::DoLoopOp>(parentLoopOp->getParentOp())) {
        parentLoopOp = parentLoop;
      }
      R.setInsertionPointAfter(parentLoopOp);
      auto cvt = R.create<fir::ConvertOp>(R.getUnknownLoc(), fir::unwrapRefType(inductionRef.getType()), loopOp.getUpperBound());
      R.create<fir::StoreOp>(R.getUnknownLoc(), cvt, inductionRef)->setAttr(HoistedStoreOp, R.getUnitAttr());
      loopOp->setAttr(InductionStoreHoisted, R.getUnitAttr());
      R.eraseOp(storeOp);
      return success();
    }
  };

  RewritePatternSet patterns(op.getContext());
  patterns.add<HoistInductionStore>(op.getContext());

#if LLVM_VERSION_MAJOR >= 20
  auto _ = applyPatternsGreedily(op, FrozenRewritePatternSet(std::move(patterns)));
#else
  auto _ = applyPatternsAndFoldGreedily(op, FrozenRewritePatternSet(std::move(patterns)));
#endif
}

} // namespace

void polyfc::rewriteHLFIR(clang::DiagnosticsEngine &, ModuleOp &m) {
  // XXX mark all written DoConcurrent loops first as elemental/forall are lowered to DoConcurrent too
  m.walk([&](Operation *op) {
    if (auto doLoop = llvm::dyn_cast<fir::DoLoopOp>(op)) {
      if (!doLoop.getUnordered()) return;
      doLoop->setAttr(DoConcurrentAsWritten, UnitAttr::get(m->getContext()));
    }
  });
}

void polyfc::rewriteFIR(clang::DiagnosticsEngine &diag, ModuleOp &m) {
  polyfront::Options opts;
  polyfront::Options::parseArgsFromEnv() //
      ^ foreach_total([&](const polyfront::Options &x) { opts = x; },
                      [&](const std::vector<std::string> &errors) {
                        for (auto error : errors)
                          diag.Report(diag.getCustomDiagID(clang::DiagnosticsEngine::Error, "%0")) << error;
                      });

  llvm::outs() << " ==== FIR  ====== \n";
  doRewrite(m);
  m.dump();
  m.walk([&](Operation *op) {
    if (auto doLoop = llvm::dyn_cast<fir::DoLoopOp>(op)) {
      if (!doLoop.getUnordered() || !doLoop->hasAttr(DoConcurrentAsWritten)) return;

      std::string moduleId = fmt::format("kernel_{:p}", fmt::ptr(&doLoop));
      std::string diagLoc = show(doLoop.getLoc());
      if (const auto flc = llvm::dyn_cast<FileLineColLoc>(doLoop.getLoc())) {
        const auto filename = llvm::sys::path::filename(flc.getFilename()).str();
        moduleId = fmt::format("kernel_{}:{}:{}", filename, flc.getLine(), flc.getColumn());
        diagLoc = fmt::format("{}:{}:{}", filename, flc.getLine(), flc.getColumn());
      }

      llvm::outs() << " === DCO: " << moduleId << " === ";
      auto captures = findCapturesInOrder(doLoop.getBody()) ^ map([&](auto &c) { return std::pair{c, resolveUniqueName(c)}; });

      llvm::outs() << "[Captures]\n";
      for (auto [k, v] : captures) {
        llvm::outs() << " - " << k
                     << (v ^ map([](auto &v) { return fmt::format("({}, orig={})", v, fir::NameUniquer::deconstruct(v).second.name); }) ^
                         get_or_else(""))
                     << "\n";
      }
      // The overall outlining logic is as follows:
      //   bool executeOriginal = false;
      //   if (isPlatformKind($Kind)) {
      //     Capture capture = <create $Kind captures>
      //     if(!dispatch(layout, capture, $Kind)) executeOriginal = true;
      //   } else if(isPlatformKind($Kind)) {
      //     (repeat)
      //   } else { assert("Unknown kind") }
      //   if (executeOriginal) (call original DoConcurrent)
      OpBuilder B(m);
      Rewriter rewriter(m);
      // At just before the doLoop
      B.setInsertionPoint(doLoop);
      // Create the guard variable
      auto executeOriginal = B.create<LLVM::AllocaOp>(uLoc(B), ptrTy(B), intConst(B, i64Ty(B), 1), B.getI64IntegerAttr(1), B.getI1Type());
      // Guard is false by default
      B.create<LLVM::StoreOp>(uLoc(B), boolConst(B, false), executeOriginal);
      const auto dispatchKind = [&](OpBuilder &B0, const runtime::PlatformKind kind) {
        rewriter.invokeDispatch(B0, executeOriginal, kind, diag, diagLoc, opts, moduleId, doLoop);
      };
      // Conditional dispatch
      rewriter.ifKindEq(B, opts, runtime::PlatformKind::HostThreaded, dispatchKind);
      rewriter.ifKindEq(B, opts, runtime::PlatformKind::Managed, dispatchKind);
      // Move and guard original doLoop
      auto ifOp = B.create<fir::IfOp>(uLoc(B), B.create<LLVM::LoadOp>(uLoc(B), B.getI1Type(), executeOriginal), false);
      auto &then = ifOp.getThenRegion().front();
      doLoop->moveBefore(&then, then.begin());
    }
  });
}
