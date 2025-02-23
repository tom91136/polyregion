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
#include "magic_enum.hpp"
#include "polyfront/options_backend.hpp"
#include "polyregion/types.h"

#include "codegen.h"
#include "ftypes.h"
#include "mirrors.h"
#include "mlir_utils.h"
#include "remapper.h"
#include "rewriter.h"
#include "utils.h"

namespace {

using namespace aspartame;
using namespace polyregion::polyfc;
using namespace polyregion;
using namespace mlir;

std::optional<runtime::Type> runtimeType(const polyast::Type::Any &tpe) {
  using R = std::optional<runtime::Type>;
  return tpe.match_total(                                                          //
      [&](const polyast::Type::Float16 &) -> R { return runtime::Type::Float16; }, //
      [&](const polyast::Type::Float32 &) -> R { return runtime::Type::Float32; }, //
      [&](const polyast::Type::Float64 &) -> R { return runtime::Type::Float64; }, //

      [&](const polyast::Type::IntU8 &) -> R { return runtime::Type::IntU8; },   //
      [&](const polyast::Type::IntU16 &) -> R { return runtime::Type::IntU16; }, //
      [&](const polyast::Type::IntU32 &) -> R { return runtime::Type::IntU32; }, //
      [&](const polyast::Type::IntU64 &) -> R { return runtime::Type::IntU64; }, //

      [&](const polyast::Type::IntS8 &) -> R { return runtime::Type::IntS8; },   //
      [&](const polyast::Type::IntS16 &) -> R { return runtime::Type::IntS16; }, //
      [&](const polyast::Type::IntS32 &) -> R { return runtime::Type::IntS32; }, //
      [&](const polyast::Type::IntS64 &) -> R { return runtime::Type::IntS64; }, //

      [&](const polyast::Type::Nothing &) -> R { return {}; },                 //
      [&](const polyast::Type::Unit0 &) -> R { return runtime::Type::Void; },  //
      [&](const polyast::Type::Bool1 &) -> R { return runtime::Type::Bool1; }, //

      [&](const polyast::Type::Struct &) -> R { return {}; },               //
      [&](const polyast::Type::Ptr &) -> R { return runtime::Type::Ptr; },  //
      [&](const polyast::Type::Annotated &t) { return runtimeType(t.tpe); } //
  );
}

class Binder {

  struct CaptureField {
    struct Witness {
      Value ptr, sizeInBytes;
    };
    Type type;
    Value fieldPtr;
    std::vector<Witness> dependent, temporary;
  };

  struct ReductionField {
    Remapper::DoConcurrentRegion::Reduction reduction;
    Value fieldPtr;
  };

  PolyDCOMirror &dco;
  FReductionMirror &reductionMirror;

  LLVM::LLVMArrayType preludeTy;
  std::vector<CaptureField> captureFields;
  DynamicAggregateMirror captureMirror;
  std::vector<ReductionField> reductionFields;

  static Value getBoxPtr(OpBuilder &B, Value refToBox) {
    const auto opaqueGEP = B.create<fir::BoxOffsetOp>(uLoc(B), refToBox, fir::BoxFieldAttr::base_addr).getResult();
    const auto i64GEPAddr = B.create<fir::ConvertOp>(uLoc(B), i64Ty(B), opaqueGEP).getRes();
    return B.create<LLVM::IntToPtrOp>(uLoc(B), ptrTy(B), i64GEPAddr).getRes();
  }

  static Value getRefPtr(OpBuilder &B, Value ref) {
    const auto i64Addr = B.create<fir::ConvertOp>(uLoc(B), i64Ty(B), ref).getRes();
    const auto llvmPtr = B.create<LLVM::IntToPtrOp>(uLoc(B), ptrTy(B), i64Addr).getRes();
    return llvmPtr;
  }

  static Value bindScalarRefOrBox(OpBuilder &B, const Value val) {
    if (const auto refTy = llvm::dyn_cast<fir::ReferenceType>(val.getType())) {
      if (const auto refElemTy = refTy.getEleTy(); refElemTy.isIntOrIndexOrFloat()) { //  fir.ref<X>
        return getRefPtr(B, val);
      } else if (const auto boxTy = dyn_cast<fir::BoxType>(refElemTy)) { // fir.ref<fir.box<X>>
        if (!fir::unwrapInnerType(boxTy.getEleTy()).isSignlessIntOrIndexOrFloat())
          raise(fmt::format("Binder value is not a boxed scalar type: {}", show(val)));
        // Load the box pointer first as it's the field zero pointer-to-scalar's value we want
        return B.create<LLVM::LoadOp>(uLoc(B), ptrTy(B), getBoxPtr(B, val));
      } else raise(fmt::format("Binder value is not a scalar type: {}", show(val)));
    } else raise(fmt::format("Binder value is not a ref type: {}", show(val)));
  }

  static void bindRef(OpBuilder &B, Value val, const std::optional<polyast::StructLayout> &layout, std::vector<CaptureField> &fields,
                      const bool temporary = false) {

    if (const auto refTy = llvm::dyn_cast<fir::ReferenceType>(val.getType())) { // T = fir.ref<E>
      // XXX types mapped here should all be pointers because since they originate from a ref/memref
      if (const auto refElemTy = refTy.getEleTy(); refElemTy.isIntOrIndexOrFloat()) {
        if (temporary) raise(fmt::format("Unsupported temporary type: {}", fir::mlirTypeToString(refElemTy)));
        const auto llvmPtr = getRefPtr(B, val);
        const auto i64Size = intConst(B, i64Ty(B), refElemTy.isIndex() ? 8 : refElemTy.getIntOrFloatBitWidth() / 8);
        fields.emplace_back(CaptureField{
            .type = refElemTy,
            .fieldPtr = llvmPtr,
            .dependent = {CaptureField::Witness{llvmPtr, i64Size}},
            .temporary = {},
        });
      } else if (const auto recordTy = dyn_cast<fir::RecordType>(refElemTy)) { //  fir.ref<fir.type<X>>
        if (temporary) raise(fmt::format("Unsupported temporary type: {}", fir::mlirTypeToString(refElemTy)));
        const auto llvmPtr = getRefPtr(B, val);
        const auto i64Size = intConst(B, i64Ty(B), layout->sizeInBytes);
        fields.emplace_back(CaptureField{
            .type = refElemTy,
            .fieldPtr = llvmPtr,
            .dependent = {CaptureField::Witness{llvmPtr, i64Size}},
            .temporary = {},
        });
      } else if (auto seqTy = dyn_cast<fir::SequenceType>(refElemTy)) { // fir.ref<fir.array<X>>
        if (temporary) raise(fmt::format("Unsupported temporary type: {}", fir::mlirTypeToString(refElemTy)));
        if (seqTy.hasDynamicExtents() || seqTy.hasUnknownShape())
          raise(fmt::format("Array has dynamic extent or unknown shape: {}", fir::mlirTypeToString(refElemTy)));
        const auto maxExtent = intConst(B, i64Ty(B), seqTy.getConstantArraySize() * (seqTy.getEleTy().getIntOrFloatBitWidth() / 8));
        const auto llvmPtr = getRefPtr(B, val);
        fields.emplace_back(CaptureField{
            .type = refElemTy,
            .fieldPtr = llvmPtr,
            .dependent = {CaptureField::Witness{llvmPtr, maxExtent}},
            .temporary = {},
        });
      } else if (const auto boxTy = dyn_cast<fir::BoxType>(refElemTy)) { // fir.ref<fir.box<X>>
        if (!layout) raise(fmt::format("Binding box type {} but cannot find corresponding StructLayout!", show(refElemTy)));
        const auto boxPtr = getBoxPtr(B, val);
        const std::vector witnesses{CaptureField::Witness{boxPtr, intConst(B, i64Ty(B), layout->sizeInBytes)}};
        fields.emplace_back(CaptureField{
            .type = refElemTy,                                                         //
            .fieldPtr = boxPtr,                                                        //
            .dependent = temporary ? std::vector<CaptureField::Witness>{} : witnesses, //
            .temporary = temporary ? witnesses : std::vector<CaptureField::Witness>{}  //
        });

      } else raise(fmt::format("Unhandled binder type: {}", fir::mlirTypeToString(refElemTy)));
    } else {
      if (const auto boxTy = llvm::dyn_cast<fir::BoxType>(val.getType())) {
        const auto ref = B.create<fir::AllocaOp>(uLoc(B), boxTy).getResult();
        B.create<fir::StoreOp>(uLoc(B), val, ref);
        bindRef(B, ref, layout, fields, true);
      } else {
        raise(fmt::format("Binder value is not a ref type: {}", show(val)));
      }
    }
  }

public:
  Binder(OpBuilder &B, PolyDCOMirror &dco, FReductionMirror &reductionMirror, const std::string &name,
         const Remapper::DoConcurrentRegion &region)                                                          //
      : dco(dco),                                                                                             //
        reductionMirror(reductionMirror),                                                                     //
        preludeTy(LLVM::LLVMArrayType::get(B.getContext(), B.getI8Type(), region.preludeLayout.sizeInBytes)), //
        captureFields(region.captures ^ flat_map([&](auto &c) {                                               //
                        const auto layouts = region.layouts                                                   //
                                             | map([](auto, auto &l) { return std::pair{l.name, l}; })        //
                                             | to<std::unordered_map>();
                        const auto maybeLayout = c.named.tpe.template get<polyast::Type::Ptr>()                                   //
                                                 ^ flat_map([](auto &t) { return t.comp.template get<polyast::Type::Struct>(); }) //
                                                 ^ flat_map([&](auto &s) { return layouts ^ get_maybe(s.name); });                //
                        std::vector<CaptureField> fields;                                                                         //
                        bindRef(B, c.value, maybeLayout, fields);                                                                 //
                        return fields;                                                                                            //
                      })),                                                                                                        //
        captureMirror(B.getContext(), "Capture_" + name,
                      captureFields | map([](auto &f) { return f.fieldPtr.getType(); }) | prepend(preludeTy) | to_vector()),
        reductionFields(region.reductions ^ map([&](auto &rd) { return ReductionField{rd, bindScalarRefOrBox(B, rd.value)}; })) {}

  Type structType() const { return captureMirror.ty; }

  Value createCapture(OpBuilder &B, ModuleOp &M) const {
    const auto annotationConst = strConst(B, M, "polyreflect-track");
    captureFields | for_each([&](auto &f) {
      B.create<LLVM::VarAnnotation>(uLoc(B), f.fieldPtr, annotationConst, nullConst(B), intConst(B, i32Ty(B), 0), nullConst(B));
    });
    return captureMirror.local(B, {captureFields                                                  //
                                   | map([](auto &f) { return f.fieldPtr; })                      //
                                   | prepend(B.create<LLVM::ZeroOp>(uLoc(B), preludeTy).getRes()) //
                                   | to_vector()});
  }

  Value createReductions(OpBuilder &B) const {
    if (reductionFields.empty()) return nullConst(B);
    return reductionMirror.local(
        B, reductionFields ^ map([&](auto &f) {
             const auto tpe = f.reduction.named.tpe;
             const auto rtTpe = runtimeType(tpe);
             if (!rtTpe) raise(fmt::format("Unsupported reduction type {}", polyast::repr(tpe)));
             return FReductionMirror::Init{
                 intConst(B, i8Ty(B), static_cast<std::underlying_type_t<polydco::FReduction::Kind>>(f.reduction.kind)), //
                 intConst(B, i8Ty(B), static_cast<std::underlying_type_t<runtime::Type>>(*rtTpe)),                       //
                 f.fieldPtr                                                                                              //
             };
           }));
  }

  Value createReductionsCount(OpBuilder &B) const { return intConst(B, i64Ty(B), reductionFields.size()); }

  void recordTemporariesAndDependents(OpBuilder &B) {
    for (auto &field : captureFields) {
      field.dependent | concat(field.temporary) | for_each([&](auto &w) { dco.record(B, w.ptr, w.sizeInBytes); });
    }
  }

  void releaseTemporaries(OpBuilder &B) {
    for (auto &field : captureFields) {
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
  FReductionMirror FReduction;

  std::unordered_map<polyast::Type::Any, TypeLayoutMirror::Global> primitiveTypeLayouts;

public:
  explicit Rewriter(ModuleOp &M)
      : M(M), dco(M),                                                                       //
        CharStar(M),                                                                        //
        KernelObject(M), KernelBundle(M), AggregateMember(M), TypeLayout(M), FReduction(M), //
        primitiveTypeLayouts(std::vector<polyast::Type::Any>{
                                 polyast::Type::Float16(), polyast::Type::Float32(), polyast::Type::Float64(),                      //
                                 polyast::Type::IntU8(), polyast::Type::IntU16(), polyast::Type::IntU32(), polyast::Type::IntU64(), //
                                 polyast::Type::IntS8(), polyast::Type::IntS16(), polyast::Type::IntS32(), polyast::Type::IntS64(), //
                                 polyast::Type::Unit0(), polyast::Type::Bool1(),                                                    //
                             } //
                             | collect([&](auto &t) {
                                 return polyast::primitiveSize(t) ^ map([&](auto sizeInBytes) {
                                          return std::pair{t, TypeLayout.global(M, [&](OpBuilder &B0) {
                                                             return std::vector{
                                                                 std::array{strConst(B0, M, polyast::repr(t)),    //
                                                                            intConst(B0, i64Ty(B0), sizeInBytes), //
                                                                            intConst(B0, i64Ty(B0), sizeInBytes), //
                                                                            intConst(B0, i64Ty(B0),
                                                                                     to_underlying(runtime::LayoutAttrs::Opaque |     //
                                                                                                   runtime::LayoutAttrs::SelfOpaque | //
                                                                                                   runtime::LayoutAttrs::Primitive)),
                                                                            intConst(B0, i64Ty(B0), 0), //
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
    const auto region = Remapper::createRegion(diagLoc, gpu, M, L, doLoop);
    const auto bundle = compileRegion(diag, diagLoc, opts, kind, moduleId, region);

    llvm::errs() << "[Captures]\n";
    for (auto &c : region.captures) {
      llvm::errs() << " - " << repr(c.named) << ": " << polyast::repr(c.named.tpe) << " = `" << c.value << "` ("
                   << magic_enum::enum_name(c.locality) << ")\n";
    }

    const auto table = bundle.layouts                                                                 //
                       | values()                                                                     //
                       | map([&](auto &sl) { return std::pair{polyast::Type::Struct(sl.name), sl}; }) //
                       | to<std::unordered_map>();

    auto structLayoutsArray = TypeLayout.global(M, [&](OpBuilder &B0) {
      return bundle.layouts ^ map([&](auto, auto &l) {
               auto attrs = runtime::LayoutAttrs::None;
               if (isSelfOpaque(l)) attrs |= runtime::LayoutAttrs::SelfOpaque;
               if (isOpaque(l, table)) attrs |= runtime::LayoutAttrs::Opaque;
               return std::array{strConst(B0, M, l.name),                //
                                 intConst(B0, i64Ty(B0), l.sizeInBytes), //
                                 intConst(B0, i64Ty(B0), l.alignment),   //
                                 intConst(B0, i64Ty(B0), to_underlying(attrs)),
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
                region.boxes ^ get_maybe(l.name) ^ map([&](const FBoxedMirror &m) {
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
                                  ^ or_else([&]() {
                                      return t.template get<polyast::Type::Struct>() ^ flat_map([&](auto &s) {
                                               return structNameToTypeLayoutIdx                                                     //
                                                      ^ get_maybe(s.name)                                                           //
                                                      ^ map([&](auto layoutIdx) { return structLayoutsArray.gep(B0, layoutIdx); }); //
                                             });
                                    }); //
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

    Binder binder(B, dco, FReduction, moduleId, region);

    if (int64_t binderSize = L.getTypeSize(binder.structType()); binderSize != region.captureLayout.sizeInBytes) {
      raise(fmt::format(
          "Capture and binder type size mismatch, expecting {} but binder gave {}\nBinder layout is:\n\t{}\nCapture layout is:\n\t{}", //
          region.captureLayout.sizeInBytes, binderSize, show(binder.structType()) ^ indent(2), repr(region.captureLayout) ^ indent(2)));
    }

    binder.recordTemporariesAndDependents(B);

    auto capture = binder.createCapture(B, M);
    auto reductionsCount = binder.createReductionsCount(B);
    auto reductions = binder.createReductions(B);
    auto dispatch = dco.dispatch(B,
                                 doLoop.getLowerBound(), //
                                 doLoop.getUpperBound(), //
                                 doLoop.getStep(),       //
                                 kind,                   //
                                 globalBundle.gep(B),    //
                                 reductionsCount,        //
                                 reductions,             //
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

  llvm::errs() << " ==== FIR  ====== \n";
  doRewrite(m);
  OpBuilder B(m);
  Rewriter rewriter(m);
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

      llvm::errs() << " === DCO: " << moduleId << " === ";
      doLoop->getParentOp()->dumpPretty();
      llvm::errs() << " === Rewrite: " << moduleId << " === ";

      // The overall outlining logic is as follows:
      //   bool executeOriginal = false;
      //   if (isPlatformKind($Kind)) {
      //     Capture capture = <create $Kind captures>
      //     if(!dispatch(layout, capture, $Kind)) executeOriginal = true;
      //   } else if(isPlatformKind($Kind)) {
      //     (repeat)
      //   } else { assert("Unknown kind") }
      //   if (executeOriginal) (call original DoConcurrent)

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
