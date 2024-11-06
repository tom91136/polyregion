#include "llvm_spirv_cl.h"
#include "aspartame/all.hpp"

using namespace polyregion::backend::details;
using namespace aspartame;

// Queue a memory fence to ensure correct ordering of memory operations to local memory
constexpr static uint32_t CLK_LOCAL_MEM_FENCE = 0x01;

// Queue a memory fence to ensure correct ordering of memory operations to global memory
constexpr static uint32_t CLK_GLOBAL_MEM_FENCE = 0x02;

void SPIRVOpenCLTargetSpecificHandler::witnessFn(CodeGen &cg, llvm::Function &fn, const Function &source) {
  fn.addFnAttr(llvm::Attribute::Convergent);
  fn.addFnAttr(llvm::Attribute::NoRecurse);
  fn.addFnAttr(llvm::Attribute::NoUnwind);

  // See logic defined in clang/lib/CodeGen/CodeGenModule.cpp @ CodeGenModule::GenKernelArgMetadata
  // We need to insert OpenCL metadata for clspv to pick up and identify the arg types
  llvm::SmallVector<llvm::Metadata *, 8> addressQuals;     // MDNode for the kernel argument address space qualifiers.
  llvm::SmallVector<llvm::Metadata *, 8> accessQuals;      // MDNode for the kernel argument access qualifiers (images only).
  llvm::SmallVector<llvm::Metadata *, 8> argTypeNames;     // MDNode for the kernel argument type names.
  llvm::SmallVector<llvm::Metadata *, 8> argBaseTypeNames; // MDNode for the kernel argument base type names.
  llvm::SmallVector<llvm::Metadata *, 8> argTypeQuals;     // MDNode for the kernel argument type qualifiers.
  llvm::SmallVector<llvm::Metadata *, 8> argNames;         // MDNode for the kernel argument names.

  for (auto arg : source.args) {
    const auto ty = cg.resolveType(arg.named.tpe, true);
    addressQuals.push_back(llvm::ConstantAsMetadata::get( //
        cg.B.getInt32(ty->isPointerTy() ? ty->getPointerAddressSpace() : 0)));
    accessQuals.push_back(llvm::MDString::get(cg.C.actual, "none")); // write_only | read_only | read_write | none

    auto typeName = [](const Type::Any &tpe) -> std::string {
      auto impl = [](const Type::Any &x, auto &thunk) -> std::string {
        return x.match_total(                                                              //
            [&](const Type::Float16 &) -> std::string { return "half"; },                  //
            [&](const Type::Float32 &) -> std::string { return "float"; },                 //
            [&](const Type::Float64 &) -> std::string { return "double"; },                //
            [&](const Type::IntU8 &) -> std::string { return "uchar"; },                   //
            [&](const Type::IntU16 &) -> std::string { return "ushort"; },                 //
            [&](const Type::IntU32 &) -> std::string { return "uint"; },                   //
            [&](const Type::IntU64 &) -> std::string { return "ulong"; },                  //
            [&](const Type::IntS8 &) -> std::string { return "char"; },                    //
            [&](const Type::IntS16 &) -> std::string { return "short"; },                  //
            [&](const Type::IntS32 &) -> std::string { return "int"; },                    //
            [&](const Type::IntS64 &) -> std::string { return "long"; },                   //
            [&](const Type::Bool1 &) -> std::string { return "char"; },                    //
            [&](const Type::Unit0 &) -> std::string { return "void"; },                    //
            [&](const Type::Nothing &) -> std::string { return "/*nothing*/"; },           //
            [&](const Type::Struct &s) -> std::string { return s.name; },                  //
            [&](const Type::Ptr &p) -> std::string { return thunk(p.comp, thunk) + "*"; }, //
            [&](const Type::Annotated &a) -> std::string { return thunk(a.tpe, thunk); }   //
        );
      };
      return impl(tpe, impl);
    };

    argTypeNames.push_back(llvm::MDString::get(cg.C.actual, typeName(arg.named.tpe)));
    argBaseTypeNames.push_back(llvm::MDString::get(cg.C.actual, typeName(arg.named.tpe)));
    argTypeQuals.push_back(llvm::MDString::get(cg.C.actual, "")); // const | restrict | volatile | pipe | ""
    argNames.push_back(llvm::MDString::get(cg.C.actual, arg.named.symbol));
  }

  fn.setMetadata("kernel_arg_addr_space", llvm::MDNode::get(cg.C.actual, addressQuals));
  fn.setMetadata("kernel_arg_access_qual", llvm::MDNode::get(cg.C.actual, accessQuals));
  fn.setMetadata("kernel_arg_type", llvm::MDNode::get(cg.C.actual, argTypeNames));
  fn.setMetadata("kernel_arg_base_type", llvm::MDNode::get(cg.C.actual, argBaseTypeNames));
  fn.setMetadata("kernel_arg_type_qual", llvm::MDNode::get(cg.C.actual, argTypeQuals));
  fn.setMetadata("kernel_arg_name", llvm::MDNode::get(cg.C.actual, argNames));
  fn.setCallingConv(llvm::CallingConv::SPIR_KERNEL);
}

// See https://github.com/KhronosGroup/SPIR-Tools/wiki/SPIR-1.2-built-in-functions
ValPtr SPIRVOpenCLTargetSpecificHandler::mkSpecVal(CodeGen &cg, const Expr::SpecOp &expr) {

  return expr.op.match_total( //
      [&](const Spec::Assert &) -> ValPtr { throw BackendException("unimplemented"); },
      [&](const Spec::GpuGlobalIdx &v) -> ValPtr { return cg.extFn1("_Z13get_global_idj", v.tpe, v.dim); },
      [&](const Spec::GpuGlobalSize &v) -> ValPtr { return cg.extFn1("_Z15get_global_sizej", v.tpe, v.dim); },
      [&](const Spec::GpuGroupIdx &v) -> ValPtr { return cg.extFn1("_Z12get_group_idj", v.tpe, v.dim); },
      [&](const Spec::GpuGroupSize &v) -> ValPtr { return cg.extFn1("_Z14get_num_groupsj", v.tpe, v.dim); },
      [&](const Spec::GpuLocalIdx &v) -> ValPtr { return cg.extFn1("_Z12get_local_idj", v.tpe, v.dim); },
      [&](const Spec::GpuLocalSize &v) -> ValPtr { return cg.extFn1("_Z14get_local_sizej", v.tpe, v.dim); },
      [&](const Spec::GpuBarrierGlobal &v) -> ValPtr { return cg.extFn1("_Z7barrierj", v.tpe, Expr::IntS32Const(CLK_GLOBAL_MEM_FENCE)); },
      [&](const Spec::GpuBarrierLocal &v) -> ValPtr { return cg.extFn1("_Z7barrierj", v.tpe, Expr::IntS32Const(CLK_LOCAL_MEM_FENCE)); },
      [&](const Spec::GpuBarrierAll &v) -> ValPtr {
        return cg.extFn1("_Z7barrierj", v.tpe, Expr::IntS32Const(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE));
      },
      [&](const Spec::GpuFenceGlobal &v) -> ValPtr { return cg.extFn1("_Z9mem_fencej", v.tpe, Expr::IntS32Const(CLK_GLOBAL_MEM_FENCE)); },
      [&](const Spec::GpuFenceLocal &v) -> ValPtr { return cg.extFn1("_Z9mem_fencej", v.tpe, Expr::IntS32Const(CLK_LOCAL_MEM_FENCE)); },
      [&](const Spec::GpuFenceAll &v) -> ValPtr {
        return cg.extFn1("_Z9mem_fencej", v.tpe, Expr::IntS32Const(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE));
      });
}
ValPtr SPIRVOpenCLTargetSpecificHandler::mkMathVal(CodeGen &cg, const Expr::MathOp &expr) {
  return expr.op.match_total( //                                                       //
      [&](const Math::Abs &v) -> ValPtr { throw BackendException("unimplemented"); },    //
      [&](const Math::Sin &v) -> ValPtr { throw BackendException("unimplemented"); },    //
      [&](const Math::Cos &v) -> ValPtr { throw BackendException("unimplemented"); },    //
      [&](const Math::Tan &v) -> ValPtr { throw BackendException("unimplemented"); },    //
      [&](const Math::Asin &v) -> ValPtr { throw BackendException("unimplemented"); },   //
      [&](const Math::Acos &v) -> ValPtr { throw BackendException("unimplemented"); },   //
      [&](const Math::Atan &v) -> ValPtr { throw BackendException("unimplemented"); },   //
      [&](const Math::Sinh &v) -> ValPtr { throw BackendException("unimplemented"); },   //
      [&](const Math::Cosh &v) -> ValPtr { throw BackendException("unimplemented"); },   //
      [&](const Math::Tanh &v) -> ValPtr { throw BackendException("unimplemented"); },   //
      [&](const Math::Signum &v) -> ValPtr { throw BackendException("unimplemented"); }, //
      [&](const Math::Round &v) -> ValPtr { throw BackendException("unimplemented"); },  //
      [&](const Math::Ceil &v) -> ValPtr { throw BackendException("unimplemented"); },   //
      [&](const Math::Floor &v) -> ValPtr { throw BackendException("unimplemented"); },  //
      [&](const Math::Rint &v) -> ValPtr { throw BackendException("unimplemented"); },   //
      [&](const Math::Sqrt &v) -> ValPtr { throw BackendException("unimplemented"); },   //
      [&](const Math::Cbrt &v) -> ValPtr { throw BackendException("unimplemented"); },   //
      [&](const Math::Exp &v) -> ValPtr { throw BackendException("unimplemented"); },    //
      [&](const Math::Expm1 &v) -> ValPtr { throw BackendException("unimplemented"); },  //
      [&](const Math::Log &v) -> ValPtr { throw BackendException("unimplemented"); },    //
      [&](const Math::Log1p &v) -> ValPtr { throw BackendException("unimplemented"); },  //
      [&](const Math::Log10 &v) -> ValPtr { throw BackendException("unimplemented"); },  //
      [&](const Math::Pow &v) -> ValPtr { throw BackendException("unimplemented"); },    //
      [&](const Math::Atan2 &v) -> ValPtr { throw BackendException("unimplemented"); },  //
      [&](const Math::Hypot &v) -> ValPtr { throw BackendException("unimplemented"); }   //
  );
}
