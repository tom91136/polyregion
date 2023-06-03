#pragma once

#include "json.hpp"
#include "polyast.h"
#include "export.h"

using json = nlohmann::json;

namespace polyregion::polyast { 
namespace Intr { 
[[nodiscard]] POLYREGION_EXPORT Intr::BNot bnot_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json bnot_to_json(const Intr::BNot &);
[[nodiscard]] POLYREGION_EXPORT Intr::LogicNot logicnot_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json logicnot_to_json(const Intr::LogicNot &);
[[nodiscard]] POLYREGION_EXPORT Intr::Pos pos_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json pos_to_json(const Intr::Pos &);
[[nodiscard]] POLYREGION_EXPORT Intr::Neg neg_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json neg_to_json(const Intr::Neg &);
[[nodiscard]] POLYREGION_EXPORT Intr::Add add_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json add_to_json(const Intr::Add &);
[[nodiscard]] POLYREGION_EXPORT Intr::Sub sub_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json sub_to_json(const Intr::Sub &);
[[nodiscard]] POLYREGION_EXPORT Intr::Mul mul_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json mul_to_json(const Intr::Mul &);
[[nodiscard]] POLYREGION_EXPORT Intr::Div div_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json div_to_json(const Intr::Div &);
[[nodiscard]] POLYREGION_EXPORT Intr::Rem rem_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json rem_to_json(const Intr::Rem &);
[[nodiscard]] POLYREGION_EXPORT Intr::Min min_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json min_to_json(const Intr::Min &);
[[nodiscard]] POLYREGION_EXPORT Intr::Max max_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json max_to_json(const Intr::Max &);
[[nodiscard]] POLYREGION_EXPORT Intr::BAnd band_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json band_to_json(const Intr::BAnd &);
[[nodiscard]] POLYREGION_EXPORT Intr::BOr bor_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json bor_to_json(const Intr::BOr &);
[[nodiscard]] POLYREGION_EXPORT Intr::BXor bxor_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json bxor_to_json(const Intr::BXor &);
[[nodiscard]] POLYREGION_EXPORT Intr::BSL bsl_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json bsl_to_json(const Intr::BSL &);
[[nodiscard]] POLYREGION_EXPORT Intr::BSR bsr_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json bsr_to_json(const Intr::BSR &);
[[nodiscard]] POLYREGION_EXPORT Intr::BZSR bzsr_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json bzsr_to_json(const Intr::BZSR &);
[[nodiscard]] POLYREGION_EXPORT Intr::LogicAnd logicand_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json logicand_to_json(const Intr::LogicAnd &);
[[nodiscard]] POLYREGION_EXPORT Intr::LogicOr logicor_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json logicor_to_json(const Intr::LogicOr &);
[[nodiscard]] POLYREGION_EXPORT Intr::LogicEq logiceq_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json logiceq_to_json(const Intr::LogicEq &);
[[nodiscard]] POLYREGION_EXPORT Intr::LogicNeq logicneq_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json logicneq_to_json(const Intr::LogicNeq &);
[[nodiscard]] POLYREGION_EXPORT Intr::LogicLte logiclte_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json logiclte_to_json(const Intr::LogicLte &);
[[nodiscard]] POLYREGION_EXPORT Intr::LogicGte logicgte_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json logicgte_to_json(const Intr::LogicGte &);
[[nodiscard]] POLYREGION_EXPORT Intr::LogicLt logiclt_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json logiclt_to_json(const Intr::LogicLt &);
[[nodiscard]] POLYREGION_EXPORT Intr::LogicGt logicgt_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json logicgt_to_json(const Intr::LogicGt &);
[[nodiscard]] POLYREGION_EXPORT Intr::Any any_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json any_to_json(const Intr::Any &);
} // namespace Intr
namespace Expr { 
[[nodiscard]] POLYREGION_EXPORT Expr::SpecOp specop_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json specop_to_json(const Expr::SpecOp &);
[[nodiscard]] POLYREGION_EXPORT Expr::MathOp mathop_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json mathop_to_json(const Expr::MathOp &);
[[nodiscard]] POLYREGION_EXPORT Expr::IntrOp introp_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json introp_to_json(const Expr::IntrOp &);
[[nodiscard]] POLYREGION_EXPORT Expr::Cast cast_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json cast_to_json(const Expr::Cast &);
[[nodiscard]] POLYREGION_EXPORT Expr::Alias alias_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json alias_to_json(const Expr::Alias &);
[[nodiscard]] POLYREGION_EXPORT Expr::Index index_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json index_to_json(const Expr::Index &);
[[nodiscard]] POLYREGION_EXPORT Expr::RefTo refto_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json refto_to_json(const Expr::RefTo &);
[[nodiscard]] POLYREGION_EXPORT Expr::Alloc alloc_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json alloc_to_json(const Expr::Alloc &);
[[nodiscard]] POLYREGION_EXPORT Expr::Invoke invoke_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json invoke_to_json(const Expr::Invoke &);
[[nodiscard]] POLYREGION_EXPORT Expr::Any any_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json any_to_json(const Expr::Any &);
} // namespace Expr
namespace FunctionAttr { 
[[nodiscard]] POLYREGION_EXPORT FunctionAttr::FPRelaxed fprelaxed_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json fprelaxed_to_json(const FunctionAttr::FPRelaxed &);
[[nodiscard]] POLYREGION_EXPORT FunctionAttr::FPStrict fpstrict_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json fpstrict_to_json(const FunctionAttr::FPStrict &);
[[nodiscard]] POLYREGION_EXPORT FunctionAttr::Any any_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json any_to_json(const FunctionAttr::Any &);
} // namespace FunctionAttr
namespace Math { 
[[nodiscard]] POLYREGION_EXPORT Math::Abs abs_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json abs_to_json(const Math::Abs &);
[[nodiscard]] POLYREGION_EXPORT Math::Sin sin_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json sin_to_json(const Math::Sin &);
[[nodiscard]] POLYREGION_EXPORT Math::Cos cos_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json cos_to_json(const Math::Cos &);
[[nodiscard]] POLYREGION_EXPORT Math::Tan tan_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json tan_to_json(const Math::Tan &);
[[nodiscard]] POLYREGION_EXPORT Math::Asin asin_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json asin_to_json(const Math::Asin &);
[[nodiscard]] POLYREGION_EXPORT Math::Acos acos_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json acos_to_json(const Math::Acos &);
[[nodiscard]] POLYREGION_EXPORT Math::Atan atan_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json atan_to_json(const Math::Atan &);
[[nodiscard]] POLYREGION_EXPORT Math::Sinh sinh_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json sinh_to_json(const Math::Sinh &);
[[nodiscard]] POLYREGION_EXPORT Math::Cosh cosh_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json cosh_to_json(const Math::Cosh &);
[[nodiscard]] POLYREGION_EXPORT Math::Tanh tanh_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json tanh_to_json(const Math::Tanh &);
[[nodiscard]] POLYREGION_EXPORT Math::Signum signum_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json signum_to_json(const Math::Signum &);
[[nodiscard]] POLYREGION_EXPORT Math::Round round_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json round_to_json(const Math::Round &);
[[nodiscard]] POLYREGION_EXPORT Math::Ceil ceil_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json ceil_to_json(const Math::Ceil &);
[[nodiscard]] POLYREGION_EXPORT Math::Floor floor_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json floor_to_json(const Math::Floor &);
[[nodiscard]] POLYREGION_EXPORT Math::Rint rint_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json rint_to_json(const Math::Rint &);
[[nodiscard]] POLYREGION_EXPORT Math::Sqrt sqrt_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json sqrt_to_json(const Math::Sqrt &);
[[nodiscard]] POLYREGION_EXPORT Math::Cbrt cbrt_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json cbrt_to_json(const Math::Cbrt &);
[[nodiscard]] POLYREGION_EXPORT Math::Exp exp_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json exp_to_json(const Math::Exp &);
[[nodiscard]] POLYREGION_EXPORT Math::Expm1 expm1_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json expm1_to_json(const Math::Expm1 &);
[[nodiscard]] POLYREGION_EXPORT Math::Log log_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json log_to_json(const Math::Log &);
[[nodiscard]] POLYREGION_EXPORT Math::Log1p log1p_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json log1p_to_json(const Math::Log1p &);
[[nodiscard]] POLYREGION_EXPORT Math::Log10 log10_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json log10_to_json(const Math::Log10 &);
[[nodiscard]] POLYREGION_EXPORT Math::Pow pow_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json pow_to_json(const Math::Pow &);
[[nodiscard]] POLYREGION_EXPORT Math::Atan2 atan2_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json atan2_to_json(const Math::Atan2 &);
[[nodiscard]] POLYREGION_EXPORT Math::Hypot hypot_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json hypot_to_json(const Math::Hypot &);
[[nodiscard]] POLYREGION_EXPORT Math::Any any_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json any_to_json(const Math::Any &);
} // namespace Math
namespace TypeSpace { 
[[nodiscard]] POLYREGION_EXPORT TypeSpace::Global global_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json global_to_json(const TypeSpace::Global &);
[[nodiscard]] POLYREGION_EXPORT TypeSpace::Local local_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json local_to_json(const TypeSpace::Local &);
[[nodiscard]] POLYREGION_EXPORT TypeSpace::Any any_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json any_to_json(const TypeSpace::Any &);
} // namespace TypeSpace
namespace Spec { 
[[nodiscard]] POLYREGION_EXPORT Spec::Assert assert_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json assert_to_json(const Spec::Assert &);
[[nodiscard]] POLYREGION_EXPORT Spec::GpuBarrierGlobal gpubarrierglobal_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json gpubarrierglobal_to_json(const Spec::GpuBarrierGlobal &);
[[nodiscard]] POLYREGION_EXPORT Spec::GpuBarrierLocal gpubarrierlocal_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json gpubarrierlocal_to_json(const Spec::GpuBarrierLocal &);
[[nodiscard]] POLYREGION_EXPORT Spec::GpuBarrierAll gpubarrierall_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json gpubarrierall_to_json(const Spec::GpuBarrierAll &);
[[nodiscard]] POLYREGION_EXPORT Spec::GpuFenceGlobal gpufenceglobal_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json gpufenceglobal_to_json(const Spec::GpuFenceGlobal &);
[[nodiscard]] POLYREGION_EXPORT Spec::GpuFenceLocal gpufencelocal_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json gpufencelocal_to_json(const Spec::GpuFenceLocal &);
[[nodiscard]] POLYREGION_EXPORT Spec::GpuFenceAll gpufenceall_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json gpufenceall_to_json(const Spec::GpuFenceAll &);
[[nodiscard]] POLYREGION_EXPORT Spec::GpuGlobalIdx gpuglobalidx_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json gpuglobalidx_to_json(const Spec::GpuGlobalIdx &);
[[nodiscard]] POLYREGION_EXPORT Spec::GpuGlobalSize gpuglobalsize_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json gpuglobalsize_to_json(const Spec::GpuGlobalSize &);
[[nodiscard]] POLYREGION_EXPORT Spec::GpuGroupIdx gpugroupidx_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json gpugroupidx_to_json(const Spec::GpuGroupIdx &);
[[nodiscard]] POLYREGION_EXPORT Spec::GpuGroupSize gpugroupsize_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json gpugroupsize_to_json(const Spec::GpuGroupSize &);
[[nodiscard]] POLYREGION_EXPORT Spec::GpuLocalIdx gpulocalidx_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json gpulocalidx_to_json(const Spec::GpuLocalIdx &);
[[nodiscard]] POLYREGION_EXPORT Spec::GpuLocalSize gpulocalsize_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json gpulocalsize_to_json(const Spec::GpuLocalSize &);
[[nodiscard]] POLYREGION_EXPORT Spec::Any any_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json any_to_json(const Spec::Any &);
} // namespace Spec
namespace TypeKind { 
[[nodiscard]] POLYREGION_EXPORT TypeKind::None none_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json none_to_json(const TypeKind::None &);
[[nodiscard]] POLYREGION_EXPORT TypeKind::Ref ref_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json ref_to_json(const TypeKind::Ref &);
[[nodiscard]] POLYREGION_EXPORT TypeKind::Integral integral_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json integral_to_json(const TypeKind::Integral &);
[[nodiscard]] POLYREGION_EXPORT TypeKind::Fractional fractional_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json fractional_to_json(const TypeKind::Fractional &);
[[nodiscard]] POLYREGION_EXPORT TypeKind::Any any_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json any_to_json(const TypeKind::Any &);
} // namespace TypeKind
namespace Type { 
[[nodiscard]] POLYREGION_EXPORT Type::Float16 float16_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json float16_to_json(const Type::Float16 &);
[[nodiscard]] POLYREGION_EXPORT Type::Float32 float32_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json float32_to_json(const Type::Float32 &);
[[nodiscard]] POLYREGION_EXPORT Type::Float64 float64_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json float64_to_json(const Type::Float64 &);
[[nodiscard]] POLYREGION_EXPORT Type::IntU8 intu8_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json intu8_to_json(const Type::IntU8 &);
[[nodiscard]] POLYREGION_EXPORT Type::IntU16 intu16_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json intu16_to_json(const Type::IntU16 &);
[[nodiscard]] POLYREGION_EXPORT Type::IntU32 intu32_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json intu32_to_json(const Type::IntU32 &);
[[nodiscard]] POLYREGION_EXPORT Type::IntU64 intu64_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json intu64_to_json(const Type::IntU64 &);
[[nodiscard]] POLYREGION_EXPORT Type::IntS8 ints8_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json ints8_to_json(const Type::IntS8 &);
[[nodiscard]] POLYREGION_EXPORT Type::IntS16 ints16_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json ints16_to_json(const Type::IntS16 &);
[[nodiscard]] POLYREGION_EXPORT Type::IntS32 ints32_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json ints32_to_json(const Type::IntS32 &);
[[nodiscard]] POLYREGION_EXPORT Type::IntS64 ints64_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json ints64_to_json(const Type::IntS64 &);
[[nodiscard]] POLYREGION_EXPORT Type::Nothing nothing_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json nothing_to_json(const Type::Nothing &);
[[nodiscard]] POLYREGION_EXPORT Type::Unit0 unit0_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json unit0_to_json(const Type::Unit0 &);
[[nodiscard]] POLYREGION_EXPORT Type::Bool1 bool1_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json bool1_to_json(const Type::Bool1 &);
[[nodiscard]] POLYREGION_EXPORT Type::Struct struct_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json struct_to_json(const Type::Struct &);
[[nodiscard]] POLYREGION_EXPORT Type::Array array_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json array_to_json(const Type::Array &);
[[nodiscard]] POLYREGION_EXPORT Type::Var var_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json var_to_json(const Type::Var &);
[[nodiscard]] POLYREGION_EXPORT Type::Exec exec_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json exec_to_json(const Type::Exec &);
[[nodiscard]] POLYREGION_EXPORT Type::Any any_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json any_to_json(const Type::Any &);
} // namespace Type
namespace Term { 
[[nodiscard]] POLYREGION_EXPORT Term::Select select_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json select_to_json(const Term::Select &);
[[nodiscard]] POLYREGION_EXPORT Term::Poison poison_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json poison_to_json(const Term::Poison &);
[[nodiscard]] POLYREGION_EXPORT Term::Float16Const float16const_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json float16const_to_json(const Term::Float16Const &);
[[nodiscard]] POLYREGION_EXPORT Term::Float32Const float32const_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json float32const_to_json(const Term::Float32Const &);
[[nodiscard]] POLYREGION_EXPORT Term::Float64Const float64const_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json float64const_to_json(const Term::Float64Const &);
[[nodiscard]] POLYREGION_EXPORT Term::IntU8Const intu8const_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json intu8const_to_json(const Term::IntU8Const &);
[[nodiscard]] POLYREGION_EXPORT Term::IntU16Const intu16const_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json intu16const_to_json(const Term::IntU16Const &);
[[nodiscard]] POLYREGION_EXPORT Term::IntU32Const intu32const_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json intu32const_to_json(const Term::IntU32Const &);
[[nodiscard]] POLYREGION_EXPORT Term::IntU64Const intu64const_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json intu64const_to_json(const Term::IntU64Const &);
[[nodiscard]] POLYREGION_EXPORT Term::IntS8Const ints8const_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json ints8const_to_json(const Term::IntS8Const &);
[[nodiscard]] POLYREGION_EXPORT Term::IntS16Const ints16const_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json ints16const_to_json(const Term::IntS16Const &);
[[nodiscard]] POLYREGION_EXPORT Term::IntS32Const ints32const_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json ints32const_to_json(const Term::IntS32Const &);
[[nodiscard]] POLYREGION_EXPORT Term::IntS64Const ints64const_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json ints64const_to_json(const Term::IntS64Const &);
[[nodiscard]] POLYREGION_EXPORT Term::Unit0Const unit0const_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json unit0const_to_json(const Term::Unit0Const &);
[[nodiscard]] POLYREGION_EXPORT Term::Bool1Const bool1const_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json bool1const_to_json(const Term::Bool1Const &);
[[nodiscard]] POLYREGION_EXPORT Term::Any any_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json any_to_json(const Term::Any &);
} // namespace Term
[[nodiscard]] POLYREGION_EXPORT Sym sym_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json sym_to_json(const Sym &);
[[nodiscard]] POLYREGION_EXPORT Named named_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json named_to_json(const Named &);
[[nodiscard]] POLYREGION_EXPORT SourcePosition sourceposition_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json sourceposition_to_json(const SourcePosition &);
[[nodiscard]] POLYREGION_EXPORT Overload overload_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json overload_to_json(const Overload &);
[[nodiscard]] POLYREGION_EXPORT StructMember structmember_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json structmember_to_json(const StructMember &);
[[nodiscard]] POLYREGION_EXPORT StructDef structdef_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json structdef_to_json(const StructDef &);
[[nodiscard]] POLYREGION_EXPORT Signature signature_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json signature_to_json(const Signature &);
[[nodiscard]] POLYREGION_EXPORT InvokeSignature invokesignature_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json invokesignature_to_json(const InvokeSignature &);
[[nodiscard]] POLYREGION_EXPORT Arg arg_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json arg_to_json(const Arg &);
[[nodiscard]] POLYREGION_EXPORT Function function_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json function_to_json(const Function &);
[[nodiscard]] POLYREGION_EXPORT Program program_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json program_to_json(const Program &);
namespace FunctionKind { 
[[nodiscard]] POLYREGION_EXPORT FunctionKind::Internal internal_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json internal_to_json(const FunctionKind::Internal &);
[[nodiscard]] POLYREGION_EXPORT FunctionKind::Exported exported_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json exported_to_json(const FunctionKind::Exported &);
[[nodiscard]] POLYREGION_EXPORT FunctionKind::Any any_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json any_to_json(const FunctionKind::Any &);
} // namespace FunctionKind
namespace Stmt { 
[[nodiscard]] POLYREGION_EXPORT Stmt::Block block_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json block_to_json(const Stmt::Block &);
[[nodiscard]] POLYREGION_EXPORT Stmt::Comment comment_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json comment_to_json(const Stmt::Comment &);
[[nodiscard]] POLYREGION_EXPORT Stmt::Var var_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json var_to_json(const Stmt::Var &);
[[nodiscard]] POLYREGION_EXPORT Stmt::Mut mut_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json mut_to_json(const Stmt::Mut &);
[[nodiscard]] POLYREGION_EXPORT Stmt::Update update_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json update_to_json(const Stmt::Update &);
[[nodiscard]] POLYREGION_EXPORT Stmt::While while_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json while_to_json(const Stmt::While &);
[[nodiscard]] POLYREGION_EXPORT Stmt::Break break_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json break_to_json(const Stmt::Break &);
[[nodiscard]] POLYREGION_EXPORT Stmt::Cont cont_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json cont_to_json(const Stmt::Cont &);
[[nodiscard]] POLYREGION_EXPORT Stmt::Cond cond_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json cond_to_json(const Stmt::Cond &);
[[nodiscard]] POLYREGION_EXPORT Stmt::Return return_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json return_to_json(const Stmt::Return &);
[[nodiscard]] POLYREGION_EXPORT Stmt::Any any_from_json(const json &);
[[nodiscard]] POLYREGION_EXPORT json any_to_json(const Stmt::Any &);
} // namespace Stmt
[[nodiscard]] POLYREGION_EXPORT json hashed_to_json(const json&);
[[nodiscard]] POLYREGION_EXPORT json hashed_from_json(const json&);
} // namespace polyregion::polyast

