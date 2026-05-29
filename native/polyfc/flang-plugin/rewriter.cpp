#include "rewriter.h"

#include <algorithm>
#include <cstdlib>
#include <optional>
#include <unordered_map>
#include <vector>

#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Optimizer/Support/InternalNames.h"
#include "flang/Optimizer/Support/Utils.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Path.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/InliningUtils.h"
#include "mlir/Transforms/Passes.h"

#include "aspartame/all.hpp"
#include "fmt/format.h"
#include "magic_enum/magic_enum.hpp"

#include "polyfront/options_backend.hpp"
#include "polyregion/types.h"

#include "codegen.h"
#include "ftypes.h"
#include "mirrors.h"
#include "mlir_utils.h"
#include "remapper.h"
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

      [&](const polyast::Type::Struct &) -> R { return {}; },              //
      [&](const polyast::Type::Ptr &) -> R { return runtime::Type::Ptr; }, //
      [&](const polyast::Type::Arr &) -> R { return runtime::Type::Ptr; }, //
      [&](const polyast::Type::Var &) -> R { return {}; },                 //
      [&](const polyast::Type::Exec &) -> R { return {}; }                 //
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
    const auto opaqueGEP = fir::BoxOffsetOp::create(B, uLoc(B), refToBox, fir::BoxFieldAttr::base_addr).getResult();
    const auto i64GEPAddr = fir::ConvertOp::create(B, uLoc(B), i64Ty(B), opaqueGEP).getRes();
    return LLVM::IntToPtrOp::create(B, uLoc(B), ptrTy(B), i64GEPAddr).getRes();
  }

  static Value getRefPtr(OpBuilder &B, Value ref) {
    const auto i64Addr = fir::ConvertOp::create(B, uLoc(B), i64Ty(B), ref).getRes();
    const auto llvmPtr = LLVM::IntToPtrOp::create(B, uLoc(B), ptrTy(B), i64Addr).getRes();
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
        return LLVM::LoadOp::create(B, uLoc(B), ptrTy(B), getBoxPtr(B, val));
      } else raise(fmt::format("Binder value is not a scalar type: {}", show(val)));
    } else raise(fmt::format("Binder value is not a ref type: {}", show(val)));
  }

  static void bindRef(OpBuilder &B, Value val, const std::optional<polyast::StructLayout> &layout, std::vector<CaptureField> &fields,
                      const bool temporary = false) {

    if (const auto refTy = llvm::dyn_cast<fir::ReferenceType>(val.getType())) { // T = fir.ref<E>
      // XXX types mapped here should all be pointers because since they originate from a ref/memref
      if (const auto refElemTy = refTy.getEleTy(); refElemTy.isIntOrIndexOrFloat()) {
        const auto llvmPtr = getRefPtr(B, val);
        const auto i64Size = intConst(B, i64Ty(B), refElemTy.isIndex() ? 8 : refElemTy.getIntOrFloatBitWidth() / 8);
        const std::vector witnesses{CaptureField::Witness{llvmPtr, i64Size}};
        fields.emplace_back(CaptureField{
            .type = refElemTy,
            .fieldPtr = llvmPtr,
            .dependent = temporary ? std::vector<CaptureField::Witness>{} : witnesses,
            .temporary = temporary ? witnesses : std::vector<CaptureField::Witness>{},
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
        if (seqTy.hasUnknownShape()) raise(fmt::format("Array has unknown shape: {}", fir::mlirTypeToString(refElemTy)));
        // XXX Dynamic extents omit the size witness; SMA still mirrors via polyreflect-rt.
        const bool dynamicExtent = seqTy.hasDynamicExtents();
        const auto llvmPtr = getRefPtr(B, val);
        std::vector<CaptureField::Witness> witnesses;
        if (!dynamicExtent) {
          const auto elemBits = seqTy.getEleTy().getIntOrFloatBitWidth();
          witnesses.push_back({llvmPtr, intConst(B, i64Ty(B), seqTy.getConstantArraySize() * (elemBits / 8))});
        }
        fields.emplace_back(CaptureField{
            .type = refElemTy,
            .fieldPtr = llvmPtr,
            .dependent = witnesses,
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
        const auto ref = fir::AllocaOp::create(B, uLoc(B), boxTy).getResult();
        fir::StoreOp::create(B, uLoc(B), val, ref);
        bindRef(B, ref, layout, fields, true);
      } else if (val.getType().isIntOrIndexOrFloat()) {
        // XXX O3 may inline shape/extent into the kernel body, leaving an unboxed scalar capture.
        // The polyast captures these as inline scalars (handleType returns IntS32/Float32/... not
        // Ptr), so store the value directly into the struct field; spilling to stack and binding
        // as a ref would leave the struct field shaped as a pointer while the kernel reads it as
        // an inline scalar - sizes and offsets mismatch and the kernel reads garbage at -O>0.
        auto ty = val.getType();
        if (ty.isIndex()) {
          ty = mlir::IntegerType::get(val.getContext(), 64);
          val = fir::ConvertOp::create(B, uLoc(B), ty, val).getResult();
        }
        fields.emplace_back(CaptureField{
            .type = ty,
            .fieldPtr = val,
            .dependent = {},
            .temporary = {},
        });
      } else if (auto heapTy = llvm::dyn_cast<fir::HeapType>(val.getType())) {
        const auto refTy = fir::ReferenceType::get(heapTy.getEleTy());
        const auto refVal = fir::ConvertOp::create(B, uLoc(B), refTy, val).getResult();
        bindRef(B, refVal, layout, fields, false);
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
                                                 ^ flat_map([&](auto &s) { return layouts ^ get_maybe(repr(s.name)); });          //
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
      // XXX skip non-pointer fields (inline scalar captures): VarAnnotation requires a ptr operand.
      if (!llvm::isa<LLVM::LLVMPointerType>(f.fieldPtr.getType())) return;
      LLVM::VarAnnotation::create(B, uLoc(B), f.fieldPtr, annotationConst, nullConst(B), intConst(B, i32Ty(B), 0), nullConst(B));
    });
    return captureMirror.local(B, {captureFields                                                   //
                                   | map([](auto &f) { return f.fieldPtr; })                       //
                                   | prepend(LLVM::ZeroOp::create(B, uLoc(B), preludeTy).getRes()) //
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
                                 polyast::Type::Float16(), polyast::Type::Float32(),
                                 polyast::Type::Float64(), //
                                 polyast::Type::IntU8(), polyast::Type::IntU16(), polyast::Type::IntU32(),
                                 polyast::Type::IntU64(), //
                                 polyast::Type::IntS8(), polyast::Type::IntS16(), polyast::Type::IntS32(),
                                 polyast::Type::IntS64(), //
                                 polyast::Type::Unit0(),
                                 polyast::Type::Bool1(), //
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
    auto B0 = fir::IfOp::create(B, uLoc(B), dco.isPlatformKind(B, kind), false).getThenBodyBuilder();
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

    const auto table = bundle.layouts                                                                //
                       | values()                                                                    //
                       | map([&](auto &sl) {                                                         //
                           return std::pair{polyast::Type::Struct(polyast::Sym({sl.name}), {}), sl}; //
                         })                                                                          //
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
                  const auto fn = defineFunc(
                      M, fmt::format("box_size_resolver_{}", ++id), i64Ty(B0), {ptrTy(B0)}, LLVM::Linkage::Internal,
                      [&](OpBuilder &B1, LLVM::LLVMFuncOp &f) {
                        const auto loadField = [&](const polyast::Named &n, const Type &ty) {
                          return l.members ^ find([&](auto &x) { return x.name == n; }) ^
                                 fold(
                                     [&](auto &slm) -> Value {
                                       auto gep = LLVM::GEPOp::create(B1, uLoc(B1), ptrTy(B1), B1.getI8Type(), f.getArgument(0),
                                                                      ValueRange{intConst(B1, i64Ty(B1), slm.offsetInBytes)});
                                       return LLVM::LoadOp::create(B1, uLoc(B1), ty, gep.getRes());
                                     },
                                     [&]() -> Value { raise(fmt::format("Missing field {}", repr(n))); });
                        };
                        auto totalSizeInBytes = loadField(m.sizeInBytes, i64Ty(B1));
                        if (m.ranks != 0) {
                          const auto dimsTy = LLVM::LLVMStructType::getLiteral(M.getContext(), {i64Ty(B1), i64Ty(B1), i64Ty(B1)});
                          const auto dims = loadField(m.dims, LLVM::LLVMArrayType::get(dimsTy, m.ranks));
                          for (int64_t i = 0; i < static_cast<int64_t>(m.ranks); ++i) {
                            // XXX Dim type fields are: { 0 = lowerBound, 1 = extent, 2 = stride }, we want the extent (1) at
                            // rank
                            const auto dim = LLVM::ExtractValueOp::create(B1, uLoc(B1), dims, ArrayRef{i}).getResult();
                            const auto extent = LLVM::ExtractValueOp::create(B1, uLoc(B1), dim, ArrayRef<int64_t>{1}).getResult();
                            totalSizeInBytes = arith::MulIOp::create(B1, uLoc(B1), totalSizeInBytes, extent).getResult();
                          }
                        }
                        LLVM::ReturnOp::create(B1, uLoc(B1), totalSizeInBytes);
                      });
                  return std::pair{m.addr, LLVM::AddressOfOp::create(B0, uLoc(B0), fn).getRes()};
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
                                                      ^ get_maybe(repr(s.name))                                                     //
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
    defineGlobalCtor(M, fmt::format("dco_layoutInit_{}_{}", magic_enum::enum_name(kind), ++id), [&](OpBuilder &FB, auto &) {
      aggregateMembersArray | for_each([&](auto &g, auto idx) {
        LLVM::StoreOp::create(FB, uLoc(FB), g.gep(FB), structLayoutsArray.gep(FB, idx, TypeLayout.members));
      });
      LLVM::ReturnOp::create(FB, uLoc(FB), ValueRange{});
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
    auto noDispatch = arith::XOrIOp::create(B, uLoc(B), dispatch, boolConst(B, true));
    auto noDispatchIf = fir::IfOp::create(B, uLoc(B), noDispatch, true);
    {
      auto ifNoDispatchB = noDispatchIf.getThenBodyBuilder();
      // Backend declined or no kernel loaded; fall through to the host doLoop.
      LLVM::StoreOp::create(ifNoDispatchB, uLoc(B), boolConst(ifNoDispatchB, true), executeOriginal);
    }
    {
      auto ifDispatchB = noDispatchIf.getElseBodyBuilder();
      LLVM::StoreOp::create(ifDispatchB, uLoc(B), boolConst(ifDispatchB, false), executeOriginal);
      binder.releaseTemporaries(ifDispatchB);
    }
  }
};

constexpr auto DoConcurrentAsWritten = "dco-as-written";
// XXX file-scope rather than function-scope so inner-class methods (HoistInductionStore) can
// access them; MSVC's strict conformance rejects nested-class access to enclosing function
// constexpr names (C2326).
constexpr auto InductionStoreHoisted = "dco-induction-store-hoisted";
constexpr auto HoistedStoreOp = "dco-hoisted-store-op";

void doRewrite(ModuleOp op) {

  // this is OK because redefinition of DO variable is not legal (e.g. so we don't expect a store)
  struct HoistInductionStore : OpRewritePattern<fir::DoLoopOp> {
    using OpRewritePattern ::OpRewritePattern;
    LogicalResult matchAndRewrite(fir::DoLoopOp loopOp, PatternRewriter &R) const override {
      if (!loopOp->hasAttr(DoConcurrentAsWritten) || loopOp->hasAttr(InductionStoreHoisted)) return failure();
      const auto induction = loopOp.getInductionVar();
      // XXX find the convert+store-to-IV-storage pattern among ALL uses of induction; reduce
      // lowering adds extra uses (ArrayCoor for the scratch slot) so a strict hasOneUse check
      // would skip the rewrite and leave the IV storage as a shared race across threads.
      fir::ConvertOp convertOp;
      fir::StoreOp storeOp;
      for (auto *user : induction.getUsers()) {
        auto cv = llvm::dyn_cast<fir::ConvertOp>(user);
        if (!cv) continue;
        for (auto *cvUser : cv->getUsers()) {
          if (auto st = llvm::dyn_cast<fir::StoreOp>(cvUser)) {
            convertOp = cv;
            storeOp = st;
            break;
          }
        }
        if (convertOp) break;
      }
      if (!convertOp || !storeOp) return failure();
      auto inductionRef = storeOp.getMemref(); // this is the outer scope induction capture
      // XXX walk the whole body (nested loops/regions too) and follow fir.declare wrappers.
      // Inlined callees may load the iv-alloca via a chain of declares emitted by hlfir-lowering;
      // a flat single-level check would leave those reads pointed at the stale alloca and the
      // per-thread iv value would never reach the inlined body.
      llvm::SmallVector<fir::LoadOp> loadsToReplace;
      loopOp.walk([&](fir::LoadOp loadOp) {
        mlir::Value mem = loadOp.getMemref();
        while (mem != inductionRef) {
          if (auto declareOp = mem.getDefiningOp<fir::DeclareOp>()) mem = declareOp.getMemref();
          else if (auto convOp = mem.getDefiningOp<fir::ConvertOp>()) mem = convOp.getValue();
          else return;
        }
        loadsToReplace.push_back(loadOp);
      });
      for (auto loadOp : loadsToReplace)
        R.replaceAllUsesWith(loadOp.getRes(), convertOp.getRes());

      // XXX Stop at the innermost loop holding the inductionRef alloca; hoisting past it would
      // place the store outside the alloca's scope (LLVM Translation: "operand does not dominate").
      auto *refDef = inductionRef.getDefiningOp();
      Operation *parentLoopOp = loopOp;
      while (const auto parentLoop = llvm::dyn_cast<fir::DoLoopOp>(parentLoopOp->getParentOp())) {
        if (refDef && parentLoop->isProperAncestor(refDef)) break;
        parentLoopOp = parentLoop;
      }
      R.setInsertionPointAfter(parentLoopOp);
      auto cvt = fir::ConvertOp::create(R, R.getUnknownLoc(), fir::unwrapRefType(inductionRef.getType()), loopOp.getUpperBound());
      fir::StoreOp::create(R, R.getUnknownLoc(), cvt, inductionRef)->setAttr(HoistedStoreOp, R.getUnitAttr());
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

void polyfc::rewriteHLFIR(clang::DiagnosticsEngine &diag, ModuleOp &m) {
  // XXX inline single-block callees inside do_concurrent before the rewrite. Surviving fir.call
  // becomes a no-op in the kernel since polypass can't lower function calls. MLIR's InlinerPass
  // needs SCC + symbol visibility that aren't available in-plugin; splice manually. Multi-block
  // callees fall through and stay un-inlined.
  {
    mlir::SymbolTableCollection symTabs;
    llvm::SmallVector<std::pair<mlir::CallOpInterface, mlir::func::FuncOp>> work;
    m.walk([&](fir::DoConcurrentOp doc) {
      doc.walk([&](mlir::CallOpInterface call) {
        auto callable = mlir::dyn_cast_or_null<mlir::func::FuncOp>(call.resolveCallableInTable(&symTabs));
        if (callable && !callable.isExternal()) work.emplace_back(call, callable);
      });
    });
    for (auto [call, callee] : work) {
      auto &calleeBody = callee.getBody();
      if (calleeBody.empty() || calleeBody.getBlocks().size() != 1) continue;
      OpBuilder builder(call);
      mlir::IRMapping mapping;
      for (auto [arg, blockArg] : llvm::zip(call.getArgOperands(), calleeBody.front().getArguments())) {
        mlir::Value v = arg;
        if (v.getType() != blockArg.getType()) v = fir::ConvertOp::create(builder, call.getLoc(), blockArg.getType(), v).getResult();
        mapping.map(blockArg, v);
      }
      auto &block = calleeBody.front();
      for (auto &op : llvm::make_early_inc_range(block.without_terminator()))
        builder.clone(op, mapping);
      if (auto ret = llvm::dyn_cast<mlir::func::ReturnOp>(block.getTerminator()))
        for (auto [r, v] : llvm::zip(call.getOperation()->getResults(), ret.getOperands()))
          r.replaceAllUsesWith(mapping.lookupOrDefault(v));
      call.erase();
    }
  }
  m.walk([&](fir::DoConcurrentOp doc) {
    auto &wrapperBlock = doc.getRegion().front();
    auto loopOp = mlir::cast<fir::DoConcurrentLoopOp>(wrapperBlock.getTerminator());
    if (loopOp.getLowerBound().size() != 1) {
      diag.Report(diag.getCustomDiagID(clang::DiagnosticsEngine::Error, "[PolyFC] multi-dimensional `do concurrent` is not yet supported"));
      return;
    }
    if (loopOp.getNumLocalOperands() != 0) {
      diag.Report(
          diag.getCustomDiagID(clang::DiagnosticsEngine::Error, "[PolyFC] `do concurrent ... local(...)` clause is not yet supported"));
      return;
    }
    OpBuilder B(doc);
    const auto loc = doc.getLoc();
    for (auto &op : llvm::make_early_inc_range(wrapperBlock.without_terminator()))
      op.moveBefore(doc);
    const auto lbV = loopOp.getLowerBound()[0];
    const auto ubV = loopOp.getUpperBound()[0];
    const auto stepV = loopOp.getStep()[0];
    const bool hasReduce = !loopOp.getReduceVars().empty();
    // XXX reduce lowers to a per-iteration scratch[N] (heap so polyreflect-rt tracks it for
    // SMA mirroring); each thread writes its slot, host folds after. Requires lb == 1 (else
    // we'd need to capture lb into the kernel). Supported ops: +, *, min, max.
    llvm::SmallVector<fir::ReduceOperationEnum> reduceOps;
    if (hasReduce) {
      if (auto attrs = loopOp.getReduceAttrsAttr()) {
        for (auto a : attrs.getValue()) {
          auto ra = mlir::dyn_cast<fir::ReduceAttr>(a);
          if (!ra) {
            diag.Report(diag.getCustomDiagID(clang::DiagnosticsEngine::Error,
                                             "[PolyFC] missing reduce attribute on `do concurrent ... reduce(...)`"));
            return;
          }
          const auto op = ra.getReduceOperation();
          if (op != fir::ReduceOperationEnum::Add && op != fir::ReduceOperationEnum::Multiply && op != fir::ReduceOperationEnum::MAX &&
              op != fir::ReduceOperationEnum::MIN) {
            diag.Report(diag.getCustomDiagID(
                clang::DiagnosticsEngine::Error,
                "[PolyFC] unsupported reduction operator in `do concurrent ... reduce(...)`; only +, *, min, max are supported"));
            return;
          }
          reduceOps.push_back(op);
        }
      }
      bool lbIsOne = false;
      if (auto v = mlir::getConstantIntValue(lbV)) lbIsOne = *v == 1;
      else if (auto cnv = lbV.getDefiningOp<fir::ConvertOp>())
        if (auto v = mlir::getConstantIntValue(cnv.getValue())) lbIsOne = *v == 1;
      if (!lbIsOne) {
        diag.Report(
            diag.getCustomDiagID(clang::DiagnosticsEngine::Error, "[PolyFC] `do concurrent ... reduce(...)` requires lower bound == 1"));
        return;
      }
    }
    // Identity value for `op` on `ty` (init each scratch slot to this so the per-thread fold
    // is correct even for threads that take no work). For min/max we pick the most-extreme
    // finite/infinity value of the type so any actual data wins on first comparison.
    auto identityConstFor = [&](OpBuilder &bb, mlir::Type ty, fir::ReduceOperationEnum op) -> mlir::Value {
      const auto loc1 = bb.getUnknownLoc();
      if (auto ft = mlir::dyn_cast<mlir::FloatType>(ty)) {
        switch (op) {
          case fir::ReduceOperationEnum::Add: return mlir::arith::ConstantOp::create(bb, loc1, bb.getFloatAttr(ft, 0.0));
          case fir::ReduceOperationEnum::Multiply: return mlir::arith::ConstantOp::create(bb, loc1, bb.getFloatAttr(ft, 1.0));
          case fir::ReduceOperationEnum::MAX: {
            llvm::APFloat v = llvm::APFloat::getInf(ft.getFloatSemantics(), /*Negative=*/true);
            return mlir::arith::ConstantOp::create(bb, loc1, bb.getFloatAttr(ft, v));
          }
          case fir::ReduceOperationEnum::MIN: {
            llvm::APFloat v = llvm::APFloat::getInf(ft.getFloatSemantics(), /*Negative=*/false);
            return mlir::arith::ConstantOp::create(bb, loc1, bb.getFloatAttr(ft, v));
          }
          default: break;
        }
      }
      auto it = mlir::cast<mlir::IntegerType>(ty);
      const auto bits = it.getWidth();
      switch (op) {
        case fir::ReduceOperationEnum::Add: return mlir::arith::ConstantOp::create(bb, loc1, bb.getIntegerAttr(it, 0));
        case fir::ReduceOperationEnum::Multiply: return mlir::arith::ConstantOp::create(bb, loc1, bb.getIntegerAttr(it, 1));
        case fir::ReduceOperationEnum::MAX: {
          auto v = llvm::APInt::getSignedMinValue(bits);
          return mlir::arith::ConstantOp::create(bb, loc1, bb.getIntegerAttr(it, v));
        }
        case fir::ReduceOperationEnum::MIN: {
          auto v = llvm::APInt::getSignedMaxValue(bits);
          return mlir::arith::ConstantOp::create(bb, loc1, bb.getIntegerAttr(it, v));
        }
        default: break;
      }
      return zeroConst(bb, ty);
    };
    // Combine `lhs op rhs` on `ty`. Matches the per-rv reduction operator.
    auto combineFor = [](OpBuilder &bb, mlir::Location l, mlir::Type ty, fir::ReduceOperationEnum op, mlir::Value lhs,
                         mlir::Value rhs) -> mlir::Value {
      const bool isFloat = mlir::isa<mlir::FloatType>(ty);
      switch (op) {
        case fir::ReduceOperationEnum::Add:
          return isFloat ? mlir::arith::AddFOp::create(bb, l, lhs, rhs).getResult()
                         : mlir::arith::AddIOp::create(bb, l, lhs, rhs).getResult();
        case fir::ReduceOperationEnum::Multiply:
          return isFloat ? mlir::arith::MulFOp::create(bb, l, lhs, rhs).getResult()
                         : mlir::arith::MulIOp::create(bb, l, lhs, rhs).getResult();
        case fir::ReduceOperationEnum::MAX:
          return isFloat ? mlir::arith::MaximumFOp::create(bb, l, lhs, rhs).getResult()
                         : mlir::arith::MaxSIOp::create(bb, l, lhs, rhs).getResult();
        case fir::ReduceOperationEnum::MIN:
          return isFloat ? mlir::arith::MinimumFOp::create(bb, l, lhs, rhs).getResult()
                         : mlir::arith::MinSIOp::create(bb, l, lhs, rhs).getResult();
        default: return lhs;
      }
    };
    llvm::SmallVector<mlir::Value> reduceVarsSnapshot(loopOp.getReduceVars().begin(), loopOp.getReduceVars().end());
    llvm::SmallVector<mlir::Value> scratchBoxes;
    llvm::SmallVector<mlir::Value> scratchHeaps;
    llvm::SmallVector<mlir::Type> scratchElemTys;
    // XXX per-rv flag for allocatable scalars; drives per-thread box synth + post-splice
    // assign rewrite (polypass can't lower `_FortranAAssign`).
    llvm::SmallVector<bool> rvIsAllocatable;
    mlir::Value numIters, idxZero, idxOne, shapeVal;
    if (hasReduce) {
      idxOne = idxConst(B, 1);
      idxZero = idxConst(B, 0);
      auto diff = mlir::arith::SubIOp::create(B, loc, ubV, lbV).getResult();
      auto div = mlir::arith::DivSIOp::create(B, loc, diff, stepV).getResult();
      numIters = mlir::arith::AddIOp::create(B, loc, div, idxOne).getResult();
      shapeVal = fir::ShapeOp::create(B, loc, mlir::ValueRange{numIters}).getResult();
      for (auto [rvIdx, rv] : llvm::enumerate(reduceVarsSnapshot)) {
        auto refTy = mlir::cast<fir::ReferenceType>(rv.getType());
        auto refEleTy = refTy.getEleTy();
        // XXX For allocatables, scratch holds scalar T (FIR rejects array<box<heap<T>>>) and we
        // rebox per-thread inside the loop body.
        mlir::Type scalarTy = refEleTy;
        bool isAllocatable = false;
        if (auto boxTy = mlir::dyn_cast<fir::BoxType>(refEleTy))
          if (auto heapTy = mlir::dyn_cast<fir::HeapType>(boxTy.getEleTy())) {
            scalarTy = heapTy.getEleTy();
            isAllocatable = true;
          }
        const auto op = rvIdx < reduceOps.size() ? reduceOps[rvIdx] : fir::ReduceOperationEnum::Add;
        auto arrTy = fir::SequenceType::get({fir::SequenceType::getUnknownExtent()}, scalarTy);
        auto boxArrTy = fir::BoxType::get(arrTy);
        auto scratchHeap =
            fir::AllocMemOp::create(B, loc, arrTy, llvm::StringRef{}, llvm::StringRef{}, mlir::ValueRange{}, mlir::ValueRange{numIters})
                .getResult();
        // XXX Convert fir.heap -> fir.ref so the Binder's ref-type path accepts it at O3.
        auto scratchRef = fir::ConvertOp::create(B, loc, fir::ReferenceType::get(arrTy), scratchHeap).getResult();
        auto scratchBox = fir::EmboxOp::create(B, loc, boxArrTy, scratchRef, shapeVal).getResult();
        // XXX identity-init happens in-kernel (see below); host-side init would race SMA's
        // mirror cache when the scratch host address gets reused across outer iterations.
        scratchBoxes.push_back(scratchBox);
        scratchHeaps.push_back(scratchHeap);
        scratchElemTys.push_back(scalarTy);
        rvIsAllocatable.push_back(isAllocatable);
      }
    }
    llvm::SmallVector<mlir::Attribute> reduceAttrs;
    if (auto attrs = loopOp.getReduceAttrsAttr()) reduceAttrs.assign(attrs.begin(), attrs.end());
    auto unordered =
        fir::DoLoopOp::create(B, loc, lbV, ubV, stepV, //
                              /*unordered=*/true, /*finalCountValue=*/false,
                              /*iterArgs=*/mlir::ValueRange{}, hasReduce ? mlir::ValueRange{} : mlir::ValueRange(reduceVarsSnapshot),
                              hasReduce ? llvm::ArrayRef<mlir::Attribute>{} : llvm::ArrayRef<mlir::Attribute>(reduceAttrs));
    unordered->setAttr(DoConcurrentAsWritten, UnitAttr::get(m->getContext()));
    auto &loopBlock = loopOp.getRegion().front();
    loopBlock.getArgument(0).replaceAllUsesWith(unordered.getInductionVar());
    auto srcReduceArgs = loopBlock.getArguments().drop_front(loopOp.getNumInductionVars() + loopOp.getNumLocalOperands());
    // Per-allocatable-rv: the synthesised box-storage op and the matching scratch slot. The
    // post-splice rewrite collapses `box_addr(load(declare(fakeBoxStorage)))` chains in the body
    // to the slot directly, so the shared fakeBoxStorage stops mattering and each thread reads
    // and writes its own slot without a race through the shared box descriptor.
    llvm::DenseMap<mlir::Operation *, mlir::Value> fakeBoxStorageToSlot;
    llvm::SmallVector<mlir::Operation *> fakeBoxSetupOps;
    if (hasReduce) {
      OpBuilder br(unordered.getBody(), unordered.getBody()->begin());
      for (auto [idx, src] : llvm::enumerate(srcReduceArgs)) {
        const auto scalarTy = scratchElemTys[idx];
        const auto rop = idx < reduceOps.size() ? reduceOps[idx] : fir::ReduceOperationEnum::Add;
        auto slot = fir::ArrayCoorOp::create(br, loc, fir::ReferenceType::get(scalarTy), scratchBoxes[idx], shapeVal,
                                             /*slice=*/mlir::Value{}, mlir::ValueRange{unordered.getInductionVar()},
                                             /*typeparams=*/mlir::ValueRange{})
                        .getResult();
        fir::StoreOp::create(br, loc, identityConstFor(br, scalarTy, rop), slot);
        if (rvIsAllocatable[idx]) {
          // XXX per-thread box<heap<T>> over the scratch slot; flang loads through the box,
          // and the post-splice rewrite turns hlfir.assign into a direct fir.store.
          auto refEleTy = mlir::cast<fir::ReferenceType>(src.getType()).getEleTy();
          auto boxTy = mlir::cast<fir::BoxType>(refEleTy);
          auto heapTy = mlir::cast<fir::HeapType>(boxTy.getEleTy());
          auto heapAddr = fir::ConvertOp::create(br, loc, heapTy, slot);
          auto fakeBox = fir::EmboxOp::create(br, loc, boxTy, heapAddr.getResult(), /*shape=*/mlir::Value{});
          auto fakeBoxStorage = fir::AllocaOp::create(br, loc, refEleTy);
          auto fakeBoxStore = fir::StoreOp::create(br, loc, fakeBox.getResult(), fakeBoxStorage.getResult());
          fakeBoxStorageToSlot[fakeBoxStorage.getOperation()] = slot;
          fakeBoxSetupOps.push_back(fakeBoxStore);
          fakeBoxSetupOps.push_back(fakeBoxStorage);
          fakeBoxSetupOps.push_back(fakeBox);
          fakeBoxSetupOps.push_back(heapAddr);
          src.replaceAllUsesWith(fakeBoxStorage.getResult());
        } else {
          src.replaceAllUsesWith(slot);
        }
      }
    } else {
      for (auto [src, ext] : llvm::zip_equal(srcReduceArgs, reduceVarsSnapshot))
        src.replaceAllUsesWith(ext);
    }
    auto *destBody = unordered.getBody();
    destBody->getOperations().splice(destBody->getTerminator()->getIterator(), loopBlock.getOperations());
    doc.erase();
    // XXX For each scratch slot, also bypass any `fir.declare`/`hlfir.declare` in the body that
    // ended up wrapping the slot. flang re-uses the original Fortran reduce-var symbol name on
    // these declares (e.g. `uniq_name = "_QMstreamFrunallEdotsum"`), which polypass and later
    // analyses can conflate with the outer-scope symbol of the same name - leading to dotSum's
    // host-side reset (`dotsum = 0` between outer-loop iterations) being lost in microstream.
    if (hasReduce) {
      auto chaseDef = [](mlir::Value v) -> mlir::Value {
        while (auto cv = v.getDefiningOp<fir::ConvertOp>())
          v = cv.getValue();
        return v;
      };
      llvm::SmallVector<mlir::Operation *> declsToErase;
      unordered.walk([&](mlir::Operation *op) {
        if (op->getName().getStringRef() != "fir.declare" && op->getName().getStringRef() != "hlfir.declare") return;
        if (op->getNumOperands() < 1) return;
        auto memref = chaseDef(op->getOperand(0));
        auto coor = memref.getDefiningOp<fir::ArrayCoorOp>();
        if (!coor) return;
        // Is this ArrayCoor on one of our scratchBoxes?
        bool isScratchSlot = false;
        for (auto sb : scratchBoxes)
          if (coor.getMemref() == sb) {
            isScratchSlot = true;
            break;
          }
        if (!isScratchSlot) return;
        // Replace declare uses with the underlying memref so the symbol name no longer aliases.
        for (auto r : op->getResults())
          r.replaceAllUsesWith(op->getOperand(0));
        declsToErase.push_back(op);
      });
      for (auto *d : declsToErase)
        d->erase();
    }
    // XXX rewrite allocatable-reduce body: box_addr(load(fakeBoxStorage)) -> per-thread slot,
    // hlfir.assign through fakeBoxStorage -> direct fir.store (polypass can't lower the runtime call).
    if (!fakeBoxStorageToSlot.empty()) {
      auto chaseRef = [](mlir::Value v) -> mlir::Value {
        while (true) {
          if (auto cv = v.getDefiningOp<fir::ConvertOp>()) {
            v = cv.getValue();
            continue;
          }
          if (auto dc = v.getDefiningOp<fir::DeclareOp>()) {
            v = dc.getMemref();
            continue;
          }
          if (auto dc = v.getDefiningOp<hlfir::DeclareOp>()) {
            v = dc.getMemref();
            continue;
          }
          return v;
        }
      };
      llvm::SmallVector<fir::BoxAddrOp> baToReplace;
      unordered.walk([&](fir::BoxAddrOp ba) {
        auto loadOp = ba.getVal().getDefiningOp<fir::LoadOp>();
        if (!loadOp) return;
        auto memref = chaseRef(loadOp.getMemref());
        if (fakeBoxStorageToSlot.find(memref.getDefiningOp()) == fakeBoxStorageToSlot.end()) return;
        baToReplace.push_back(ba);
      });
      for (auto ba : baToReplace) {
        auto loadOp = ba.getVal().getDefiningOp<fir::LoadOp>();
        auto memref = chaseRef(loadOp.getMemref());
        auto slot = fakeBoxStorageToSlot[memref.getDefiningOp()];
        for (auto &use : llvm::make_early_inc_range(ba.getResult().getUses())) {
          auto *user = use.getOwner();
          if (auto ld = llvm::dyn_cast<fir::LoadOp>(user)) {
            OpBuilder rb(ld);
            auto newLoad = fir::LoadOp::create(rb, ld.getLoc(), slot).getResult();
            ld.getResult().replaceAllUsesWith(newLoad);
            ld.erase();
          } else if (auto st = llvm::dyn_cast<fir::StoreOp>(user); st && st.getMemref() == ba.getResult()) {
            OpBuilder rb(st);
            fir::StoreOp::create(rb, st.getLoc(), st.getValue(), slot);
            st.erase();
          } else {
            // Fallback for unexpected users: keep the typed convert.
            OpBuilder rb(user);
            auto slotAsHeap = fir::ConvertOp::create(rb, ba.getLoc(), ba.getResult().getType(), slot).getResult();
            use.set(slotAsHeap);
          }
        }
        ba.erase();
        if (loadOp.getResult().use_empty()) loadOp.erase();
      }
      llvm::SmallVector<hlfir::AssignOp> assignToErase;
      unordered.walk([&](hlfir::AssignOp assign) {
        auto lhsRoot = chaseRef(assign.getLhs());
        auto it = fakeBoxStorageToSlot.find(lhsRoot.getDefiningOp());
        if (it == fakeBoxStorageToSlot.end()) return;
        OpBuilder rb(assign);
        auto rloc = assign.getLoc();
        auto rhs = assign.getRhs();
        if (mlir::isa<fir::ReferenceType, fir::HeapType, fir::PointerType>(rhs.getType()))
          rhs = fir::LoadOp::create(rb, rloc, rhs).getResult();
        else if (auto rhsBoxTy = mlir::dyn_cast<fir::BoxType>(rhs.getType())) {
          auto rhsAddr = fir::BoxAddrOp::create(rb, rloc, fir::ReferenceType::get(rhsBoxTy.getEleTy()), rhs).getResult();
          rhs = fir::LoadOp::create(rb, rloc, rhsAddr).getResult();
        }
        fir::StoreOp::create(rb, rloc, rhs, it->second);
        assignToErase.push_back(assign);
      });
      for (auto a : assignToErase)
        a.erase();
      // XXX alloca/store have side effects and won't self-DCE before polypass runs.
      llvm::SmallVector<mlir::Operation *> declaresToErase;
      unordered.walk([&](mlir::Operation *op) {
        if (op->getName().getStringRef() != "fir.declare" && op->getName().getStringRef() != "hlfir.declare") return;
        if (op->getNumOperands() < 1) return;
        if (fakeBoxStorageToSlot.find(op->getOperand(0).getDefiningOp()) == fakeBoxStorageToSlot.end()) return;
        declaresToErase.push_back(op);
      });
      for (auto *d : declaresToErase)
        d->erase();
      bool progress = true;
      while (progress) {
        progress = false;
        for (auto &op : fakeBoxSetupOps) {
          if (op && op->use_empty()) {
            op->erase();
            op = nullptr;
            progress = true;
          }
        }
      }
    }
    if (hasReduce) {
      B.setInsertionPointAfter(unordered);
      for (auto [idx, t] : llvm::enumerate(llvm::zip(reduceVarsSnapshot, scratchBoxes, scratchElemTys))) {
        auto [extRef, scratchBox, scalarTy] = t;
        const auto op = idx < reduceOps.size() ? reduceOps[idx] : fir::ReduceOperationEnum::Add;
        mlir::Value effectiveRef = extRef;
        if (rvIsAllocatable[idx]) {
          auto boxTy = mlir::cast<fir::BoxType>(mlir::cast<fir::ReferenceType>(extRef.getType()).getEleTy());
          auto heapTy = mlir::cast<fir::HeapType>(boxTy.getEleTy());
          auto boxVal = fir::LoadOp::create(B, loc, extRef).getResult();
          auto heapAddr = fir::BoxAddrOp::create(B, loc, heapTy, boxVal).getResult();
          effectiveRef = fir::ConvertOp::create(B, loc, fir::ReferenceType::get(scalarTy), heapAddr).getResult();
        }
        auto combineLoop = fir::DoLoopOp::create(B, loc, idxOne, numIters, idxOne, /*unordered=*/false, /*finalCountValue=*/false);
        OpBuilder bc(combineLoop.getBody(), combineLoop.getBody()->begin());
        auto slot = fir::ArrayCoorOp::create(bc, loc, fir::ReferenceType::get(scalarTy), scratchBox, shapeVal,
                                             /*slice=*/mlir::Value{}, mlir::ValueRange{combineLoop.getInductionVar()},
                                             /*typeparams=*/mlir::ValueRange{})
                        .getResult();
        auto delta = fir::LoadOp::create(bc, loc, slot).getResult();
        auto cur = fir::LoadOp::create(bc, loc, effectiveRef).getResult();
        auto folded = combineFor(bc, loc, scalarTy, op, cur, delta);
        fir::StoreOp::create(bc, loc, folded, effectiveRef);
      }
      // XXX disassociate from SMA before fir.freemem; libc recycles the host address and the
      // next iteration's mirrorToRemote would cache-hit the stale entry.
      const auto llvmPtrTy = mlir::LLVM::LLVMPointerType::get(B.getContext());
      const auto voidTy = mlir::LLVM::LLVMVoidType::get(B.getContext());
      auto releaseFn = m.lookupSymbol<mlir::LLVM::LLVMFuncOp>("polydco_release");
      if (!releaseFn) releaseFn = defineFunc(m, "polydco_release", voidTy, {llvmPtrTy});
      for (auto h : scratchHeaps) {
        auto i64Addr = fir::ConvertOp::create(B, loc, B.getI64Type(), h).getResult();
        auto llvmPtr = mlir::LLVM::IntToPtrOp::create(B, loc, llvmPtrTy, i64Addr).getResult();
        mlir::LLVM::CallOp::create(B, loc, releaseFn, mlir::ValueRange{llvmPtr});
        fir::FreeMemOp::create(B, loc, h);
      }
    }
  });

  // XXX distinguish synthesised + pre-Flang-22 unordered loops from elemental/forall artefacts.
  m.walk([&](fir::DoLoopOp doLoop) {
    if (!doLoop.getUnordered()) return;
    doLoop->setAttr(DoConcurrentAsWritten, UnitAttr::get(m->getContext()));
  });

  // XXX orphan declare_reduction trips LLVM Translation when body SSA refs no longer resolve.
  llvm::SmallVector<fir::DeclareReductionOp> deadReductions;
  m.walk([&](fir::DeclareReductionOp op) {
    if (mlir::SymbolTable::symbolKnownUseEmpty(op, m)) deadReductions.push_back(op);
  });
  for (auto op : deadReductions)
    op.erase();
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
      //   bool executeOriginal = true;
      //   if (isPlatformKind($Kind)) {
      //     Capture capture = <create $Kind captures>
      //     if (dispatch(layout, capture, $Kind)) executeOriginal = false;
      //   } else if(isPlatformKind($Kind)) {
      //     (repeat)
      //   }
      //   if (executeOriginal) (call original DoConcurrent)
      //
      // Default-true so unmatched platforms (no backend init succeeded, no compatible GPU)
      // still run the host doLoop instead of silently no-op'ing.

      B.setInsertionPoint(doLoop);
      auto executeOriginal = LLVM::AllocaOp::create(B, uLoc(B), ptrTy(B), intConst(B, i64Ty(B), 1), B.getI64IntegerAttr(1), B.getI1Type());
      LLVM::StoreOp::create(B, uLoc(B), boolConst(B, true), executeOriginal);
      const auto dispatchKind = [&](OpBuilder &B0, const runtime::PlatformKind kind) {
        rewriter.invokeDispatch(B0, executeOriginal, kind, diag, diagLoc, opts, moduleId, doLoop);
      };
      // Conditional dispatch
      rewriter.ifKindEq(B, opts, runtime::PlatformKind::HostThreaded, dispatchKind);
      rewriter.ifKindEq(B, opts, runtime::PlatformKind::Managed, dispatchKind);
      // Move and guard original doLoop
      auto ifOp = fir::IfOp::create(B, uLoc(B), LLVM::LoadOp::create(B, uLoc(B), B.getI1Type(), executeOriginal), false);
      auto &then = ifOp.getThenRegion().front();
      doLoop->moveBefore(&then, then.begin());
    }
  });
}
