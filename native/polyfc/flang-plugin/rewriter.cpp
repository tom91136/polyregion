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

struct AggregateMemberMirror final : AggregateMirror<7> {
  Field<LLVM::LLVMPointerType, 0> name;
  Field<IntegerType, 1> offsetInBytes;
  Field<IntegerType, 2> sizeInBytes;
  Field<IntegerType, 3> ptrIndirection;
  Field<IntegerType, 4> componentSize;
  Field<LLVM::LLVMPointerType, 5> type;
  Field<LLVM::LLVMPointerType, 6> resolvePtrSizeInBytes;

  explicit AggregateMemberMirror(MLIRContext *C)
      : AggregateMirror(C), name(C), offsetInBytes(C, 64), sizeInBytes(C, 64), ptrIndirection(C, 64), componentSize(C, 64), type(C),
        resolvePtrSizeInBytes(C) {}
  const char *typeName() const override { return "AggregateMember"; }
  std::array<Type, 7> types() const override {
    return {name.widen(),           //
            offsetInBytes.widen(),  //
            sizeInBytes.widen(),    //
            ptrIndirection.widen(), //
            componentSize.widen(),  //
            type.widen(),           //
            resolvePtrSizeInBytes.widen()};
  }
};

struct TypeLayoutMirror final : AggregateMirror<5> {
  Field<LLVM::LLVMPointerType, 0> name;
  Field<IntegerType, 1> sizeInBytes;
  Field<IntegerType, 2> alignmentInBytes;
  Field<IntegerType, 3> memberCount;
  Field<LLVM::LLVMPointerType, 4> members;
  explicit TypeLayoutMirror(MLIRContext *C)
      : AggregateMirror(C), //
        name(C), sizeInBytes(C, 64), alignmentInBytes(C, 64), memberCount(C, 64), members(C) {}
  const char *typeName() const override { return "TypeLayout"; }
  std::array<Type, 5> types() const override {
    return {name.widen(),             //
            sizeInBytes.widen(),      //
            alignmentInBytes.widen(), //
            memberCount.widen(),      //
            members.widen()};
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

  // FDimMirror FDim;
  // FArrayDescMirror FArrayDesc;
  LLVM::LLVMArrayType preludeTy;
  std::vector<Field> fields;
  DynamicAggregateMirror mirror;

  static Value getBoxPtr(OpBuilder &B, Value refToBox) {
    const auto opaqueGEP = B.create<fir::BoxOffsetOp>(uLoc(B), refToBox, fir::BoxFieldAttr::base_addr).getResult();
    const auto i64GEPAddr = B.create<fir::ConvertOp>(uLoc(B), i64Ty(B), opaqueGEP).getRes();
    return B.create<LLVM::IntToPtrOp>(uLoc(B), ptrTy(B), i64GEPAddr).getRes();
  }

  static std::vector<Field> bindRef(OpBuilder &B, Value val, const std::optional<polyast::StructLayout> &layout) {
    std::vector<Field> fields;
    if (const auto refTy = llvm::dyn_cast<fir::ReferenceType>(val.getType())) { // T = fir.ref<E>
      // XXX types mapped here should all be pointers because since they originate from a ref/memref
      const auto ty = fir::unwrapRefType(val.getType());
      if (ty.isIntOrFloat()) {
        const auto i64Addr = B.create<fir::ConvertOp>(uLoc(B), i64Ty(B), val).getRes();
        const auto llvmPtr = B.create<LLVM::IntToPtrOp>(uLoc(B), ptrTy(B), i64Addr).getRes();
        const auto i64Size = intConst(B, i64Ty(B), ty.getIntOrFloatBitWidth() / 8);
        fields.emplace_back(Field{
            .type = ty,
            .fieldPtr = llvmPtr,
            .dependent = {Field::Witness{llvmPtr, i64Size}},
            .temporary = {},
        });
      } else if (const auto recordTy = dyn_cast<fir::RecordType>(ty)) { // E = fir.type<X>
        const auto i64Addr = B.create<fir::ConvertOp>(uLoc(B), i64Ty(B), val).getRes();
        const auto llvmPtr = B.create<LLVM::IntToPtrOp>(uLoc(B), ptrTy(B), i64Addr).getRes();
        const auto i64Size = intConst(B, i64Ty(B), layout->sizeInBytes);
        fields.emplace_back(Field{
            .type = ty,
            .fieldPtr = llvmPtr,
            .dependent = {Field::Witness{llvmPtr, i64Size}},
            .temporary = {},
        });
      } else if (auto seqTy = dyn_cast<fir::SequenceType>(ty)) { // E = fir.array<X>
        if (seqTy.hasDynamicExtents() || seqTy.hasUnknownShape())
          raise(fmt::format("Array has dynamic extent or unknown shape: {}", fir::mlirTypeToString(ty)));
        const auto maxExtent = intConst(B, i64Ty(B), seqTy.getConstantArraySize() * (seqTy.getEleTy().getIntOrFloatBitWidth() / 8));
        const auto i64Addr = B.create<fir::ConvertOp>(uLoc(B), i64Ty(B), val).getRes();
        const auto llvmPtr = B.create<LLVM::IntToPtrOp>(uLoc(B), ptrTy(B), i64Addr).getRes();
        fields.emplace_back(Field{
            .type = ty,
            .fieldPtr = llvmPtr,
            .dependent = {Field::Witness{llvmPtr, maxExtent}},
            .temporary = {},
        });
      } else if (const auto boxTy = dyn_cast<fir::BoxType>(ty)) { // E = fir.box<X>
        if (!layout) raise(fmt::format("Binding box type {} but cannot find corresponding StructLayout!", show(ty)));
        const auto boxPtr = getBoxPtr(B, val);
        fields.emplace_back(Field{
            .type = ty,                                                                        //
            .fieldPtr = boxPtr,                                                                //
            .dependent = {Field::Witness{boxPtr, intConst(B, i64Ty(B), layout->sizeInBytes)}}, //
            .temporary = {}                                                                    //
        });

      } else raise(fmt::format("Unhandled binder type: {}", fir::mlirTypeToString(ty)));
    } else raise(fmt::format("Value is not a ref type: {}", show(val)));
    return fields;
  }

public:
  Binder(OpBuilder &B, DataLayout &L, PolyDCOMirror &dco, const std::string &name, const size_t preludeSize,
         const std::vector<std::pair<polyast::Named, Value>> &refs,
         const std::unordered_map<std::string, polyast::StructLayout> &layouts) //
      : dco(dco), preludeTy(LLVM::LLVMArrayType::get(B.getContext(), B.getI8Type(), preludeSize)),
        fields(refs ^ flat_map([&](auto &named, auto &ref) {
                 auto maybeLayout = named.tpe.template get<polyast::Type::Ptr>() ^
                                    flat_map([](auto &t) { return t.comp.template get<polyast::Type::Struct>(); }) ^
                                    flat_map([&](auto &s) { return layouts ^ get_maybe(s.name); });

                 return bindRef(B, ref, maybeLayout);
               })),
        mirror(B.getContext(), "Binder_" + name,
               fields                                                  //
                   | map([](auto &f) { return f.fieldPtr.getType(); }) //
                   | prepend(preludeTy)                                //
                   | to_vector()) {}

  Type structType() const { return mirror.ty; }

  Value create(OpBuilder &B) const {
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
  ModuleOp &M;
  PolyDCOMirror dco;
  CharStarMirror CharStar;
  KernelObjectMirror KernelObject;
  KernelBundleMirror KernelBundle;
  AggregateMemberMirror AggregateMember;
  TypeLayoutMirror TypeLayout;

  std::unordered_map<polyast::Type::Any, TypeLayoutMirror::Global> primitiveTypeLayouts;

public:
  explicit Rewriter(ModuleOp &m)
      : M(m), dco(m),                                                                                                            //
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
                                                             return std::vector{std::array{strConst(B0, m, polyast::repr(t)),    //
                                                                                           intConst(B0, i64Ty(B0), sizeInBytes), //
                                                                                           intConst(B0, i64Ty(B0), sizeInBytes), //
                                                                                           intConst(B0, i64Ty(B0), 0),           //
                                                                                           nullConst(B0)}};
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
    DataLayout L(M);
    const auto gpu = kind == runtime::PlatformKind::Managed;
    const auto region = Remapper::createRegion("_main", gpu, M, L, doLoop);
    const auto bundle = compileRegion(diag, diagLoc, opts, kind, moduleId, region);
    const auto table = bundle.layouts                                                                 //
                       | values()                                                                     //
                       | map([&](auto &sl) { return std::pair{polyast::Type::Struct(sl.name), sl}; }) //
                       | to<std::unordered_map>();

    auto structLayoutsArray = TypeLayout.global(M, [&](OpBuilder &B0) {
      return bundle.layouts ^ map([&](auto, auto &l) {
               return std::array{strConst(B0, M, l.name),                   //
                                 intConst(B0, i64Ty(B0), l.sizeInBytes),    //
                                 intConst(B0, i64Ty(B0), l.alignment),      //
                                 intConst(B0, i64Ty(B0), l.members.size()), //
                                 nullConst(B0)};
             });
    });

    const auto structNameToTypeLayoutIdx = bundle.layouts                          //
                                           | values()                              //
                                           | map([](auto &sl) { return sl.name; }) //
                                           | zip_with_index()                      //
                                           | to<std::unordered_map>();

    // region.boxes
    auto aggregateMembersArray =
        bundle.layouts | values() | map([&](const polyast::StructLayout &l) {
          return AggregateMember.global(M, [&](OpBuilder &B0) {
            const auto fbm =
                region.boxes ^ get_maybe(l.name) ^ map([&](const Remapper::FBoxedMirror &m) {
                  static size_t id = 0;
                  const auto fn =
                      defineFunc(M, fmt::format("box_size_resolver_{}", ++id), i64Ty(B0), {ptrTy(B0)}, LLVM::Linkage::Internal,
                                 [&](OpBuilder &B1, LLVM::LLVMFuncOp &f) {
                                   const auto loadField = [&](const polyast::Named &n, const Type &ty) {
                                     return l.members ^ find([&](auto &x) { return x.name == n; }) ^
                                            fold(
                                                [&](auto &slm) -> Value {
                                                  auto gep = B1.create<LLVM::GEPOp>(uLoc(B1), ptrTy(B1), B1.getI8Type(), f.getArgument(0),
                                                                                    ValueRange{intConst(B1, i64Ty(B1), slm.offsetInBytes)});
                                                  return B1.create<LLVM::LoadOp>(uLoc(B1), ty, gep.getRes());
                                                },
                                                [&]() -> Value { raise(fmt::format("Missing field {}", repr(n))); });
                                   };
                                   auto totalSizeInBytes = loadField(m.sizeInBytes, i64Ty(B1));
                                   if (m.ranks != 0) {
                                     const auto dimsTy =
                                         LLVM::LLVMStructType::getLiteral(M.getContext(), {i64Ty(B1), i64Ty(B1), i64Ty(B1)});
                                     const auto dims = loadField(m.dims, LLVM::LLVMArrayType::get(dimsTy, m.ranks));
                                     for (int64_t i = 0; i < static_cast<int64_t>(m.ranks); ++i) {
                                       // XXX Dim type fields are: { 0 = lowerBound, 1 = extent, 2 = stride }, we want the extent (1) at
                                       // rank
                                       const auto dim = B1.create<LLVM::ExtractValueOp>(uLoc(B1), dims, ArrayRef{i}).getResult();
                                       const auto extent = B1.create<LLVM::ExtractValueOp>(uLoc(B1), dim, ArrayRef<int64_t>{1}).getResult();
                                       totalSizeInBytes = B1.create<arith::MulIOp>(uLoc(B1), totalSizeInBytes, extent).getResult();
                                     }
                                   }
                                   B1.create<LLVM::ReturnOp>(uLoc(B1), totalSizeInBytes);
                                 });
                  return std::pair{m.addr, B0.create<LLVM::AddressOfOp>(uLoc(B0), fn).getRes()};
                });
            return l.members ^ map([&](auto &m) {
                     const auto [indirections, componentSize] = countIndirectionsAndComponentSize(m.name.tpe, table);
                     const auto compType = polyast::extractComponent(m.name.tpe);
                     const auto ptrToTypeLayout =
                         polyast::extractComponent(m.name.tpe) ^ flat_map([&](auto &t) {
                           return primitiveTypeLayouts                                        //
                                  ^ get_maybe(t) ^ map([&](auto ptl) { return ptl.gep(B0); }) //
                                  ^ or_else(t.template get<polyast::Type::Struct>() ^ flat_map([&](auto &s) {
                                              return structNameToTypeLayoutIdx                                                     //
                                                     ^ get_maybe(s.name)                                                           //
                                                     ^ map([&](auto layoutIdx) { return structLayoutsArray.gep(B0, layoutIdx); }); //
                                            }));                                                                                   //
                         });
                     return std::array{
                         strConst(B0, M, m.name.symbol),                                 //
                         intConst(B0, i64Ty(B0), m.offsetInBytes),                       //
                         intConst(B0, i64Ty(B0), m.sizeInBytes),                         //
                         intConst(B0, i64Ty(B0), indirections),                          //
                         intConst(B0, i64Ty(B0), componentSize.value_or(m.sizeInBytes)), //
                         ptrToTypeLayout ^ get_or_else(nullConst(B0)),
                         fbm                                                                                     //
                             ^ filter([&](auto &field, auto &) { return field == m.name; })                      //
                             ^ fold([](auto &, auto &v) -> Value { return v; }, [&]() { return nullConst(B0); }) //
                     };
                   });
          });
        })                 //
        | zip_with_index() //
        | to_vector();

    static size_t id = 0;
    defineGlobalCtor(M, fmt::format("dco_layoutInit_{}_{}", to_string(kind), ++id), [&](OpBuilder &FB, auto &) {
      aggregateMembersArray | for_each([&](auto &g, auto idx) {
        FB.create<LLVM::StoreOp>(uLoc(FB), g.gep(FB), structLayoutsArray.gep(FB, idx, TypeLayout.members));
      });
      FB.create<LLVM::ReturnOp>(uLoc(FB), ValueRange{});
    });

    auto globalKOs = KernelObject.global(M, [&](OpBuilder &B0) {
      return bundle.objects ^ map([&](auto &o) {
               auto features = CharStar.global(
                   M, [&](OpBuilder &B1) { return o.features ^ map([&](auto &f) { return std::array{strConst(B1, M, f)}; }); });
               return KernelObjectMirror::Init{intConst(B0, i8Ty(B0), value_of(o.kind)),      //
                                               intConst(B0, i8Ty(B0), value_of(o.format)),    //
                                               intConst(B0, i64Ty(B0), o.features.size()),    //
                                               features.gep(B0),                              //
                                               intConst(B0, i64Ty(B0), o.moduleImage.size()), //
                                               strConst(B0, M, o.moduleImage, false)};
             });
    });

    auto globalBundle = KernelBundle.global(M, [&](OpBuilder &B0) {
      return std::vector{std::array{strConst(B0, M, moduleId), //

                                    intConst(B0, i64Ty(B0), bundle.objects.size()), //
                                    globalKOs.gep(B0),                              //

                                    intConst(B0, i64Ty(B0), bundle.layouts.size()), //
                                    structLayoutsArray.gep(B0),
                                    intConst(B0, i64Ty(B0), bundle.layouts | index_where([](auto &iface, auto) { return iface; })),

                                    strConst(B0, M, bundle.metadata)}};
    });

    const auto layouts = region.layouts | map([](auto, auto &l) { return std::pair{l.name, l}; }) | to<std::unordered_map>();
    Binder binder(B, L, dco, moduleId, region.preludeLayout.sizeInBytes, region.captures, layouts);

    if (int64_t binderSize = L.getTypeSize(binder.structType()); binderSize != region.captureLayout.sizeInBytes) {
      raise(fmt::format(
          "Capture and binder type size mismatch, expecting {} but binder gave {}\n Binder layout is {}\nCapture layout is {}", //
          region.captureLayout.sizeInBytes, binderSize, show(binder.structType()), repr(region.captureLayout)));
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
      // doLoop.dump();
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
