#pragma once

#include "json.hpp"
#include "polyast.h"
#include "export.h"

using json = nlohmann::json;

namespace polyregion::polyast { 
namespace Intr { 
[[nodiscard]] EXPORT Intr::BNot bnot_from_json(const json &);
[[nodiscard]] EXPORT json bnot_to_json(const Intr::BNot &);
[[nodiscard]] EXPORT Intr::LogicNot logicnot_from_json(const json &);
[[nodiscard]] EXPORT json logicnot_to_json(const Intr::LogicNot &);
[[nodiscard]] EXPORT Intr::Pos pos_from_json(const json &);
[[nodiscard]] EXPORT json pos_to_json(const Intr::Pos &);
[[nodiscard]] EXPORT Intr::Neg neg_from_json(const json &);
[[nodiscard]] EXPORT json neg_to_json(const Intr::Neg &);
[[nodiscard]] EXPORT Intr::Add add_from_json(const json &);
[[nodiscard]] EXPORT json add_to_json(const Intr::Add &);
[[nodiscard]] EXPORT Intr::Sub sub_from_json(const json &);
[[nodiscard]] EXPORT json sub_to_json(const Intr::Sub &);
[[nodiscard]] EXPORT Intr::Mul mul_from_json(const json &);
[[nodiscard]] EXPORT json mul_to_json(const Intr::Mul &);
[[nodiscard]] EXPORT Intr::Div div_from_json(const json &);
[[nodiscard]] EXPORT json div_to_json(const Intr::Div &);
[[nodiscard]] EXPORT Intr::Rem rem_from_json(const json &);
[[nodiscard]] EXPORT json rem_to_json(const Intr::Rem &);
[[nodiscard]] EXPORT Intr::Min min_from_json(const json &);
[[nodiscard]] EXPORT json min_to_json(const Intr::Min &);
[[nodiscard]] EXPORT Intr::Max max_from_json(const json &);
[[nodiscard]] EXPORT json max_to_json(const Intr::Max &);
[[nodiscard]] EXPORT Intr::BAnd band_from_json(const json &);
[[nodiscard]] EXPORT json band_to_json(const Intr::BAnd &);
[[nodiscard]] EXPORT Intr::BOr bor_from_json(const json &);
[[nodiscard]] EXPORT json bor_to_json(const Intr::BOr &);
[[nodiscard]] EXPORT Intr::BXor bxor_from_json(const json &);
[[nodiscard]] EXPORT json bxor_to_json(const Intr::BXor &);
[[nodiscard]] EXPORT Intr::BSL bsl_from_json(const json &);
[[nodiscard]] EXPORT json bsl_to_json(const Intr::BSL &);
[[nodiscard]] EXPORT Intr::BSR bsr_from_json(const json &);
[[nodiscard]] EXPORT json bsr_to_json(const Intr::BSR &);
[[nodiscard]] EXPORT Intr::BZSR bzsr_from_json(const json &);
[[nodiscard]] EXPORT json bzsr_to_json(const Intr::BZSR &);
[[nodiscard]] EXPORT Intr::LogicAnd logicand_from_json(const json &);
[[nodiscard]] EXPORT json logicand_to_json(const Intr::LogicAnd &);
[[nodiscard]] EXPORT Intr::LogicOr logicor_from_json(const json &);
[[nodiscard]] EXPORT json logicor_to_json(const Intr::LogicOr &);
[[nodiscard]] EXPORT Intr::LogicEq logiceq_from_json(const json &);
[[nodiscard]] EXPORT json logiceq_to_json(const Intr::LogicEq &);
[[nodiscard]] EXPORT Intr::LogicNeq logicneq_from_json(const json &);
[[nodiscard]] EXPORT json logicneq_to_json(const Intr::LogicNeq &);
[[nodiscard]] EXPORT Intr::LogicLte logiclte_from_json(const json &);
[[nodiscard]] EXPORT json logiclte_to_json(const Intr::LogicLte &);
[[nodiscard]] EXPORT Intr::LogicGte logicgte_from_json(const json &);
[[nodiscard]] EXPORT json logicgte_to_json(const Intr::LogicGte &);
[[nodiscard]] EXPORT Intr::LogicLt logiclt_from_json(const json &);
[[nodiscard]] EXPORT json logiclt_to_json(const Intr::LogicLt &);
[[nodiscard]] EXPORT Intr::LogicGt logicgt_from_json(const json &);
[[nodiscard]] EXPORT json logicgt_to_json(const Intr::LogicGt &);
[[nodiscard]] EXPORT Intr::Any any_from_json(const json &);
[[nodiscard]] EXPORT json any_to_json(const Intr::Any &);
} // namespace Intr
namespace Expr { 
[[nodiscard]] EXPORT Expr::SpecOp specop_from_json(const json &);
[[nodiscard]] EXPORT json specop_to_json(const Expr::SpecOp &);
[[nodiscard]] EXPORT Expr::MathOp mathop_from_json(const json &);
[[nodiscard]] EXPORT json mathop_to_json(const Expr::MathOp &);
[[nodiscard]] EXPORT Expr::IntrOp introp_from_json(const json &);
[[nodiscard]] EXPORT json introp_to_json(const Expr::IntrOp &);
[[nodiscard]] EXPORT Expr::Cast cast_from_json(const json &);
[[nodiscard]] EXPORT json cast_to_json(const Expr::Cast &);
[[nodiscard]] EXPORT Expr::Alias alias_from_json(const json &);
[[nodiscard]] EXPORT json alias_to_json(const Expr::Alias &);
[[nodiscard]] EXPORT Expr::Index index_from_json(const json &);
[[nodiscard]] EXPORT json index_to_json(const Expr::Index &);
[[nodiscard]] EXPORT Expr::Alloc alloc_from_json(const json &);
[[nodiscard]] EXPORT json alloc_to_json(const Expr::Alloc &);
[[nodiscard]] EXPORT Expr::Invoke invoke_from_json(const json &);
[[nodiscard]] EXPORT json invoke_to_json(const Expr::Invoke &);
[[nodiscard]] EXPORT Expr::Any any_from_json(const json &);
[[nodiscard]] EXPORT json any_to_json(const Expr::Any &);
} // namespace Expr
namespace FunctionAttr { 
[[nodiscard]] EXPORT FunctionAttr::FPRelaxed fprelaxed_from_json(const json &);
[[nodiscard]] EXPORT json fprelaxed_to_json(const FunctionAttr::FPRelaxed &);
[[nodiscard]] EXPORT FunctionAttr::FPStrict fpstrict_from_json(const json &);
[[nodiscard]] EXPORT json fpstrict_to_json(const FunctionAttr::FPStrict &);
[[nodiscard]] EXPORT FunctionAttr::Any any_from_json(const json &);
[[nodiscard]] EXPORT json any_to_json(const FunctionAttr::Any &);
} // namespace FunctionAttr
namespace Math { 
[[nodiscard]] EXPORT Math::Abs abs_from_json(const json &);
[[nodiscard]] EXPORT json abs_to_json(const Math::Abs &);
[[nodiscard]] EXPORT Math::Sin sin_from_json(const json &);
[[nodiscard]] EXPORT json sin_to_json(const Math::Sin &);
[[nodiscard]] EXPORT Math::Cos cos_from_json(const json &);
[[nodiscard]] EXPORT json cos_to_json(const Math::Cos &);
[[nodiscard]] EXPORT Math::Tan tan_from_json(const json &);
[[nodiscard]] EXPORT json tan_to_json(const Math::Tan &);
[[nodiscard]] EXPORT Math::Asin asin_from_json(const json &);
[[nodiscard]] EXPORT json asin_to_json(const Math::Asin &);
[[nodiscard]] EXPORT Math::Acos acos_from_json(const json &);
[[nodiscard]] EXPORT json acos_to_json(const Math::Acos &);
[[nodiscard]] EXPORT Math::Atan atan_from_json(const json &);
[[nodiscard]] EXPORT json atan_to_json(const Math::Atan &);
[[nodiscard]] EXPORT Math::Sinh sinh_from_json(const json &);
[[nodiscard]] EXPORT json sinh_to_json(const Math::Sinh &);
[[nodiscard]] EXPORT Math::Cosh cosh_from_json(const json &);
[[nodiscard]] EXPORT json cosh_to_json(const Math::Cosh &);
[[nodiscard]] EXPORT Math::Tanh tanh_from_json(const json &);
[[nodiscard]] EXPORT json tanh_to_json(const Math::Tanh &);
[[nodiscard]] EXPORT Math::Signum signum_from_json(const json &);
[[nodiscard]] EXPORT json signum_to_json(const Math::Signum &);
[[nodiscard]] EXPORT Math::Round round_from_json(const json &);
[[nodiscard]] EXPORT json round_to_json(const Math::Round &);
[[nodiscard]] EXPORT Math::Ceil ceil_from_json(const json &);
[[nodiscard]] EXPORT json ceil_to_json(const Math::Ceil &);
[[nodiscard]] EXPORT Math::Floor floor_from_json(const json &);
[[nodiscard]] EXPORT json floor_to_json(const Math::Floor &);
[[nodiscard]] EXPORT Math::Rint rint_from_json(const json &);
[[nodiscard]] EXPORT json rint_to_json(const Math::Rint &);
[[nodiscard]] EXPORT Math::Sqrt sqrt_from_json(const json &);
[[nodiscard]] EXPORT json sqrt_to_json(const Math::Sqrt &);
[[nodiscard]] EXPORT Math::Cbrt cbrt_from_json(const json &);
[[nodiscard]] EXPORT json cbrt_to_json(const Math::Cbrt &);
[[nodiscard]] EXPORT Math::Exp exp_from_json(const json &);
[[nodiscard]] EXPORT json exp_to_json(const Math::Exp &);
[[nodiscard]] EXPORT Math::Expm1 expm1_from_json(const json &);
[[nodiscard]] EXPORT json expm1_to_json(const Math::Expm1 &);
[[nodiscard]] EXPORT Math::Log log_from_json(const json &);
[[nodiscard]] EXPORT json log_to_json(const Math::Log &);
[[nodiscard]] EXPORT Math::Log1p log1p_from_json(const json &);
[[nodiscard]] EXPORT json log1p_to_json(const Math::Log1p &);
[[nodiscard]] EXPORT Math::Log10 log10_from_json(const json &);
[[nodiscard]] EXPORT json log10_to_json(const Math::Log10 &);
[[nodiscard]] EXPORT Math::Pow pow_from_json(const json &);
[[nodiscard]] EXPORT json pow_to_json(const Math::Pow &);
[[nodiscard]] EXPORT Math::Atan2 atan2_from_json(const json &);
[[nodiscard]] EXPORT json atan2_to_json(const Math::Atan2 &);
[[nodiscard]] EXPORT Math::Hypot hypot_from_json(const json &);
[[nodiscard]] EXPORT json hypot_to_json(const Math::Hypot &);
[[nodiscard]] EXPORT Math::Any any_from_json(const json &);
[[nodiscard]] EXPORT json any_to_json(const Math::Any &);
} // namespace Math
namespace TypeSpace { 
[[nodiscard]] EXPORT TypeSpace::Global global_from_json(const json &);
[[nodiscard]] EXPORT json global_to_json(const TypeSpace::Global &);
[[nodiscard]] EXPORT TypeSpace::Local local_from_json(const json &);
[[nodiscard]] EXPORT json local_to_json(const TypeSpace::Local &);
[[nodiscard]] EXPORT TypeSpace::Any any_from_json(const json &);
[[nodiscard]] EXPORT json any_to_json(const TypeSpace::Any &);
} // namespace TypeSpace
namespace Spec { 
[[nodiscard]] EXPORT Spec::Assert assert_from_json(const json &);
[[nodiscard]] EXPORT json assert_to_json(const Spec::Assert &);
[[nodiscard]] EXPORT Spec::GpuBarrierGlobal gpubarrierglobal_from_json(const json &);
[[nodiscard]] EXPORT json gpubarrierglobal_to_json(const Spec::GpuBarrierGlobal &);
[[nodiscard]] EXPORT Spec::GpuBarrierLocal gpubarrierlocal_from_json(const json &);
[[nodiscard]] EXPORT json gpubarrierlocal_to_json(const Spec::GpuBarrierLocal &);
[[nodiscard]] EXPORT Spec::GpuBarrierAll gpubarrierall_from_json(const json &);
[[nodiscard]] EXPORT json gpubarrierall_to_json(const Spec::GpuBarrierAll &);
[[nodiscard]] EXPORT Spec::GpuFenceGlobal gpufenceglobal_from_json(const json &);
[[nodiscard]] EXPORT json gpufenceglobal_to_json(const Spec::GpuFenceGlobal &);
[[nodiscard]] EXPORT Spec::GpuFenceLocal gpufencelocal_from_json(const json &);
[[nodiscard]] EXPORT json gpufencelocal_to_json(const Spec::GpuFenceLocal &);
[[nodiscard]] EXPORT Spec::GpuFenceAll gpufenceall_from_json(const json &);
[[nodiscard]] EXPORT json gpufenceall_to_json(const Spec::GpuFenceAll &);
[[nodiscard]] EXPORT Spec::GpuGlobalIdx gpuglobalidx_from_json(const json &);
[[nodiscard]] EXPORT json gpuglobalidx_to_json(const Spec::GpuGlobalIdx &);
[[nodiscard]] EXPORT Spec::GpuGlobalSize gpuglobalsize_from_json(const json &);
[[nodiscard]] EXPORT json gpuglobalsize_to_json(const Spec::GpuGlobalSize &);
[[nodiscard]] EXPORT Spec::GpuGroupIdx gpugroupidx_from_json(const json &);
[[nodiscard]] EXPORT json gpugroupidx_to_json(const Spec::GpuGroupIdx &);
[[nodiscard]] EXPORT Spec::GpuGroupSize gpugroupsize_from_json(const json &);
[[nodiscard]] EXPORT json gpugroupsize_to_json(const Spec::GpuGroupSize &);
[[nodiscard]] EXPORT Spec::GpuLocalIdx gpulocalidx_from_json(const json &);
[[nodiscard]] EXPORT json gpulocalidx_to_json(const Spec::GpuLocalIdx &);
[[nodiscard]] EXPORT Spec::GpuLocalSize gpulocalsize_from_json(const json &);
[[nodiscard]] EXPORT json gpulocalsize_to_json(const Spec::GpuLocalSize &);
[[nodiscard]] EXPORT Spec::Any any_from_json(const json &);
[[nodiscard]] EXPORT json any_to_json(const Spec::Any &);
} // namespace Spec
namespace TypeKind { 
[[nodiscard]] EXPORT TypeKind::None none_from_json(const json &);
[[nodiscard]] EXPORT json none_to_json(const TypeKind::None &);
[[nodiscard]] EXPORT TypeKind::Ref ref_from_json(const json &);
[[nodiscard]] EXPORT json ref_to_json(const TypeKind::Ref &);
[[nodiscard]] EXPORT TypeKind::Integral integral_from_json(const json &);
[[nodiscard]] EXPORT json integral_to_json(const TypeKind::Integral &);
[[nodiscard]] EXPORT TypeKind::Fractional fractional_from_json(const json &);
[[nodiscard]] EXPORT json fractional_to_json(const TypeKind::Fractional &);
[[nodiscard]] EXPORT TypeKind::Any any_from_json(const json &);
[[nodiscard]] EXPORT json any_to_json(const TypeKind::Any &);
} // namespace TypeKind
namespace Type { 
[[nodiscard]] EXPORT Type::Float16 float16_from_json(const json &);
[[nodiscard]] EXPORT json float16_to_json(const Type::Float16 &);
[[nodiscard]] EXPORT Type::Float32 float32_from_json(const json &);
[[nodiscard]] EXPORT json float32_to_json(const Type::Float32 &);
[[nodiscard]] EXPORT Type::Float64 float64_from_json(const json &);
[[nodiscard]] EXPORT json float64_to_json(const Type::Float64 &);
[[nodiscard]] EXPORT Type::IntU8 intu8_from_json(const json &);
[[nodiscard]] EXPORT json intu8_to_json(const Type::IntU8 &);
[[nodiscard]] EXPORT Type::IntU16 intu16_from_json(const json &);
[[nodiscard]] EXPORT json intu16_to_json(const Type::IntU16 &);
[[nodiscard]] EXPORT Type::IntU32 intu32_from_json(const json &);
[[nodiscard]] EXPORT json intu32_to_json(const Type::IntU32 &);
[[nodiscard]] EXPORT Type::IntU64 intu64_from_json(const json &);
[[nodiscard]] EXPORT json intu64_to_json(const Type::IntU64 &);
[[nodiscard]] EXPORT Type::IntS8 ints8_from_json(const json &);
[[nodiscard]] EXPORT json ints8_to_json(const Type::IntS8 &);
[[nodiscard]] EXPORT Type::IntS16 ints16_from_json(const json &);
[[nodiscard]] EXPORT json ints16_to_json(const Type::IntS16 &);
[[nodiscard]] EXPORT Type::IntS32 ints32_from_json(const json &);
[[nodiscard]] EXPORT json ints32_to_json(const Type::IntS32 &);
[[nodiscard]] EXPORT Type::IntS64 ints64_from_json(const json &);
[[nodiscard]] EXPORT json ints64_to_json(const Type::IntS64 &);
[[nodiscard]] EXPORT Type::Nothing nothing_from_json(const json &);
[[nodiscard]] EXPORT json nothing_to_json(const Type::Nothing &);
[[nodiscard]] EXPORT Type::Unit0 unit0_from_json(const json &);
[[nodiscard]] EXPORT json unit0_to_json(const Type::Unit0 &);
[[nodiscard]] EXPORT Type::Bool1 bool1_from_json(const json &);
[[nodiscard]] EXPORT json bool1_to_json(const Type::Bool1 &);
[[nodiscard]] EXPORT Type::Struct struct_from_json(const json &);
[[nodiscard]] EXPORT json struct_to_json(const Type::Struct &);
[[nodiscard]] EXPORT Type::Array array_from_json(const json &);
[[nodiscard]] EXPORT json array_to_json(const Type::Array &);
[[nodiscard]] EXPORT Type::Var var_from_json(const json &);
[[nodiscard]] EXPORT json var_to_json(const Type::Var &);
[[nodiscard]] EXPORT Type::Exec exec_from_json(const json &);
[[nodiscard]] EXPORT json exec_to_json(const Type::Exec &);
[[nodiscard]] EXPORT Type::Any any_from_json(const json &);
[[nodiscard]] EXPORT json any_to_json(const Type::Any &);
} // namespace Type
namespace Term { 
[[nodiscard]] EXPORT Term::Select select_from_json(const json &);
[[nodiscard]] EXPORT json select_to_json(const Term::Select &);
[[nodiscard]] EXPORT Term::Poison poison_from_json(const json &);
[[nodiscard]] EXPORT json poison_to_json(const Term::Poison &);
[[nodiscard]] EXPORT Term::Float16Const float16const_from_json(const json &);
[[nodiscard]] EXPORT json float16const_to_json(const Term::Float16Const &);
[[nodiscard]] EXPORT Term::Float32Const float32const_from_json(const json &);
[[nodiscard]] EXPORT json float32const_to_json(const Term::Float32Const &);
[[nodiscard]] EXPORT Term::Float64Const float64const_from_json(const json &);
[[nodiscard]] EXPORT json float64const_to_json(const Term::Float64Const &);
[[nodiscard]] EXPORT Term::IntU8Const intu8const_from_json(const json &);
[[nodiscard]] EXPORT json intu8const_to_json(const Term::IntU8Const &);
[[nodiscard]] EXPORT Term::IntU16Const intu16const_from_json(const json &);
[[nodiscard]] EXPORT json intu16const_to_json(const Term::IntU16Const &);
[[nodiscard]] EXPORT Term::IntU32Const intu32const_from_json(const json &);
[[nodiscard]] EXPORT json intu32const_to_json(const Term::IntU32Const &);
[[nodiscard]] EXPORT Term::IntU64Const intu64const_from_json(const json &);
[[nodiscard]] EXPORT json intu64const_to_json(const Term::IntU64Const &);
[[nodiscard]] EXPORT Term::IntS8Const ints8const_from_json(const json &);
[[nodiscard]] EXPORT json ints8const_to_json(const Term::IntS8Const &);
[[nodiscard]] EXPORT Term::IntS16Const ints16const_from_json(const json &);
[[nodiscard]] EXPORT json ints16const_to_json(const Term::IntS16Const &);
[[nodiscard]] EXPORT Term::IntS32Const ints32const_from_json(const json &);
[[nodiscard]] EXPORT json ints32const_to_json(const Term::IntS32Const &);
[[nodiscard]] EXPORT Term::IntS64Const ints64const_from_json(const json &);
[[nodiscard]] EXPORT json ints64const_to_json(const Term::IntS64Const &);
[[nodiscard]] EXPORT Term::Unit0Const unit0const_from_json(const json &);
[[nodiscard]] EXPORT json unit0const_to_json(const Term::Unit0Const &);
[[nodiscard]] EXPORT Term::Bool1Const bool1const_from_json(const json &);
[[nodiscard]] EXPORT json bool1const_to_json(const Term::Bool1Const &);
[[nodiscard]] EXPORT Term::Any any_from_json(const json &);
[[nodiscard]] EXPORT json any_to_json(const Term::Any &);
} // namespace Term
[[nodiscard]] EXPORT Sym sym_from_json(const json &);
[[nodiscard]] EXPORT json sym_to_json(const Sym &);
[[nodiscard]] EXPORT Named named_from_json(const json &);
[[nodiscard]] EXPORT json named_to_json(const Named &);
[[nodiscard]] EXPORT SourcePosition sourceposition_from_json(const json &);
[[nodiscard]] EXPORT json sourceposition_to_json(const SourcePosition &);
[[nodiscard]] EXPORT Overload overload_from_json(const json &);
[[nodiscard]] EXPORT json overload_to_json(const Overload &);
[[nodiscard]] EXPORT StructMember structmember_from_json(const json &);
[[nodiscard]] EXPORT json structmember_to_json(const StructMember &);
[[nodiscard]] EXPORT StructDef structdef_from_json(const json &);
[[nodiscard]] EXPORT json structdef_to_json(const StructDef &);
[[nodiscard]] EXPORT Signature signature_from_json(const json &);
[[nodiscard]] EXPORT json signature_to_json(const Signature &);
[[nodiscard]] EXPORT InvokeSignature invokesignature_from_json(const json &);
[[nodiscard]] EXPORT json invokesignature_to_json(const InvokeSignature &);
[[nodiscard]] EXPORT Arg arg_from_json(const json &);
[[nodiscard]] EXPORT json arg_to_json(const Arg &);
[[nodiscard]] EXPORT Function function_from_json(const json &);
[[nodiscard]] EXPORT json function_to_json(const Function &);
[[nodiscard]] EXPORT Program program_from_json(const json &);
[[nodiscard]] EXPORT json program_to_json(const Program &);
namespace FunctionKind { 
[[nodiscard]] EXPORT FunctionKind::Internal internal_from_json(const json &);
[[nodiscard]] EXPORT json internal_to_json(const FunctionKind::Internal &);
[[nodiscard]] EXPORT FunctionKind::Exported exported_from_json(const json &);
[[nodiscard]] EXPORT json exported_to_json(const FunctionKind::Exported &);
[[nodiscard]] EXPORT FunctionKind::Any any_from_json(const json &);
[[nodiscard]] EXPORT json any_to_json(const FunctionKind::Any &);
} // namespace FunctionKind
namespace Stmt { 
[[nodiscard]] EXPORT Stmt::Block block_from_json(const json &);
[[nodiscard]] EXPORT json block_to_json(const Stmt::Block &);
[[nodiscard]] EXPORT Stmt::Comment comment_from_json(const json &);
[[nodiscard]] EXPORT json comment_to_json(const Stmt::Comment &);
[[nodiscard]] EXPORT Stmt::Var var_from_json(const json &);
[[nodiscard]] EXPORT json var_to_json(const Stmt::Var &);
[[nodiscard]] EXPORT Stmt::Mut mut_from_json(const json &);
[[nodiscard]] EXPORT json mut_to_json(const Stmt::Mut &);
[[nodiscard]] EXPORT Stmt::Update update_from_json(const json &);
[[nodiscard]] EXPORT json update_to_json(const Stmt::Update &);
[[nodiscard]] EXPORT Stmt::While while_from_json(const json &);
[[nodiscard]] EXPORT json while_to_json(const Stmt::While &);
[[nodiscard]] EXPORT Stmt::Break break_from_json(const json &);
[[nodiscard]] EXPORT json break_to_json(const Stmt::Break &);
[[nodiscard]] EXPORT Stmt::Cont cont_from_json(const json &);
[[nodiscard]] EXPORT json cont_to_json(const Stmt::Cont &);
[[nodiscard]] EXPORT Stmt::Cond cond_from_json(const json &);
[[nodiscard]] EXPORT json cond_to_json(const Stmt::Cond &);
[[nodiscard]] EXPORT Stmt::Return return_from_json(const json &);
[[nodiscard]] EXPORT json return_to_json(const Stmt::Return &);
[[nodiscard]] EXPORT Stmt::Any any_from_json(const json &);
[[nodiscard]] EXPORT json any_to_json(const Stmt::Any &);
} // namespace Stmt
[[nodiscard]] EXPORT json hashed_to_json(const json&);
[[nodiscard]] EXPORT json hashed_from_json(const json&);
} // namespace polyregion::polyast

