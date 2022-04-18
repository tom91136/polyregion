#pragma once

#include "json.hpp"
#include "polyast.h"
#include "export.h"

using json = nlohmann::json;

namespace polyregion::polyast { 
namespace Expr { 
[[nodiscard]] EXPORT Expr::UnaryIntrinsic unaryintrinsic_from_json(const json &);
[[nodiscard]] EXPORT json unaryintrinsic_to_json(const Expr::UnaryIntrinsic &);
[[nodiscard]] EXPORT Expr::BinaryIntrinsic binaryintrinsic_from_json(const json &);
[[nodiscard]] EXPORT json binaryintrinsic_to_json(const Expr::BinaryIntrinsic &);
[[nodiscard]] EXPORT Expr::UnaryLogicIntrinsic unarylogicintrinsic_from_json(const json &);
[[nodiscard]] EXPORT json unarylogicintrinsic_to_json(const Expr::UnaryLogicIntrinsic &);
[[nodiscard]] EXPORT Expr::BinaryLogicIntrinsic binarylogicintrinsic_from_json(const json &);
[[nodiscard]] EXPORT json binarylogicintrinsic_to_json(const Expr::BinaryLogicIntrinsic &);
[[nodiscard]] EXPORT Expr::Cast cast_from_json(const json &);
[[nodiscard]] EXPORT json cast_to_json(const Expr::Cast &);
[[nodiscard]] EXPORT Expr::Alias alias_from_json(const json &);
[[nodiscard]] EXPORT json alias_to_json(const Expr::Alias &);
[[nodiscard]] EXPORT Expr::Invoke invoke_from_json(const json &);
[[nodiscard]] EXPORT json invoke_to_json(const Expr::Invoke &);
[[nodiscard]] EXPORT Expr::Index index_from_json(const json &);
[[nodiscard]] EXPORT json index_to_json(const Expr::Index &);
[[nodiscard]] EXPORT Expr::Alloc alloc_from_json(const json &);
[[nodiscard]] EXPORT json alloc_to_json(const Expr::Alloc &);
[[nodiscard]] EXPORT Expr::Suspend suspend_from_json(const json &);
[[nodiscard]] EXPORT json suspend_to_json(const Expr::Suspend &);
[[nodiscard]] EXPORT Expr::Any any_from_json(const json &);
[[nodiscard]] EXPORT json any_to_json(const Expr::Any &);
} // namespace Expr
namespace BinaryLogicIntrinsicKind { 
[[nodiscard]] EXPORT BinaryLogicIntrinsicKind::Eq eq_from_json(const json &);
[[nodiscard]] EXPORT json eq_to_json(const BinaryLogicIntrinsicKind::Eq &);
[[nodiscard]] EXPORT BinaryLogicIntrinsicKind::Neq neq_from_json(const json &);
[[nodiscard]] EXPORT json neq_to_json(const BinaryLogicIntrinsicKind::Neq &);
[[nodiscard]] EXPORT BinaryLogicIntrinsicKind::And and_from_json(const json &);
[[nodiscard]] EXPORT json and_to_json(const BinaryLogicIntrinsicKind::And &);
[[nodiscard]] EXPORT BinaryLogicIntrinsicKind::Or or_from_json(const json &);
[[nodiscard]] EXPORT json or_to_json(const BinaryLogicIntrinsicKind::Or &);
[[nodiscard]] EXPORT BinaryLogicIntrinsicKind::Lte lte_from_json(const json &);
[[nodiscard]] EXPORT json lte_to_json(const BinaryLogicIntrinsicKind::Lte &);
[[nodiscard]] EXPORT BinaryLogicIntrinsicKind::Gte gte_from_json(const json &);
[[nodiscard]] EXPORT json gte_to_json(const BinaryLogicIntrinsicKind::Gte &);
[[nodiscard]] EXPORT BinaryLogicIntrinsicKind::Lt lt_from_json(const json &);
[[nodiscard]] EXPORT json lt_to_json(const BinaryLogicIntrinsicKind::Lt &);
[[nodiscard]] EXPORT BinaryLogicIntrinsicKind::Gt gt_from_json(const json &);
[[nodiscard]] EXPORT json gt_to_json(const BinaryLogicIntrinsicKind::Gt &);
[[nodiscard]] EXPORT BinaryLogicIntrinsicKind::Any any_from_json(const json &);
[[nodiscard]] EXPORT json any_to_json(const BinaryLogicIntrinsicKind::Any &);
} // namespace BinaryLogicIntrinsicKind
namespace Stmt { 
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
namespace UnaryIntrinsicKind { 
[[nodiscard]] EXPORT UnaryIntrinsicKind::Sin sin_from_json(const json &);
[[nodiscard]] EXPORT json sin_to_json(const UnaryIntrinsicKind::Sin &);
[[nodiscard]] EXPORT UnaryIntrinsicKind::Cos cos_from_json(const json &);
[[nodiscard]] EXPORT json cos_to_json(const UnaryIntrinsicKind::Cos &);
[[nodiscard]] EXPORT UnaryIntrinsicKind::Tan tan_from_json(const json &);
[[nodiscard]] EXPORT json tan_to_json(const UnaryIntrinsicKind::Tan &);
[[nodiscard]] EXPORT UnaryIntrinsicKind::Asin asin_from_json(const json &);
[[nodiscard]] EXPORT json asin_to_json(const UnaryIntrinsicKind::Asin &);
[[nodiscard]] EXPORT UnaryIntrinsicKind::Acos acos_from_json(const json &);
[[nodiscard]] EXPORT json acos_to_json(const UnaryIntrinsicKind::Acos &);
[[nodiscard]] EXPORT UnaryIntrinsicKind::Atan atan_from_json(const json &);
[[nodiscard]] EXPORT json atan_to_json(const UnaryIntrinsicKind::Atan &);
[[nodiscard]] EXPORT UnaryIntrinsicKind::Sinh sinh_from_json(const json &);
[[nodiscard]] EXPORT json sinh_to_json(const UnaryIntrinsicKind::Sinh &);
[[nodiscard]] EXPORT UnaryIntrinsicKind::Cosh cosh_from_json(const json &);
[[nodiscard]] EXPORT json cosh_to_json(const UnaryIntrinsicKind::Cosh &);
[[nodiscard]] EXPORT UnaryIntrinsicKind::Tanh tanh_from_json(const json &);
[[nodiscard]] EXPORT json tanh_to_json(const UnaryIntrinsicKind::Tanh &);
[[nodiscard]] EXPORT UnaryIntrinsicKind::Signum signum_from_json(const json &);
[[nodiscard]] EXPORT json signum_to_json(const UnaryIntrinsicKind::Signum &);
[[nodiscard]] EXPORT UnaryIntrinsicKind::Abs abs_from_json(const json &);
[[nodiscard]] EXPORT json abs_to_json(const UnaryIntrinsicKind::Abs &);
[[nodiscard]] EXPORT UnaryIntrinsicKind::Round round_from_json(const json &);
[[nodiscard]] EXPORT json round_to_json(const UnaryIntrinsicKind::Round &);
[[nodiscard]] EXPORT UnaryIntrinsicKind::Ceil ceil_from_json(const json &);
[[nodiscard]] EXPORT json ceil_to_json(const UnaryIntrinsicKind::Ceil &);
[[nodiscard]] EXPORT UnaryIntrinsicKind::Floor floor_from_json(const json &);
[[nodiscard]] EXPORT json floor_to_json(const UnaryIntrinsicKind::Floor &);
[[nodiscard]] EXPORT UnaryIntrinsicKind::Rint rint_from_json(const json &);
[[nodiscard]] EXPORT json rint_to_json(const UnaryIntrinsicKind::Rint &);
[[nodiscard]] EXPORT UnaryIntrinsicKind::Sqrt sqrt_from_json(const json &);
[[nodiscard]] EXPORT json sqrt_to_json(const UnaryIntrinsicKind::Sqrt &);
[[nodiscard]] EXPORT UnaryIntrinsicKind::Cbrt cbrt_from_json(const json &);
[[nodiscard]] EXPORT json cbrt_to_json(const UnaryIntrinsicKind::Cbrt &);
[[nodiscard]] EXPORT UnaryIntrinsicKind::Exp exp_from_json(const json &);
[[nodiscard]] EXPORT json exp_to_json(const UnaryIntrinsicKind::Exp &);
[[nodiscard]] EXPORT UnaryIntrinsicKind::Expm1 expm1_from_json(const json &);
[[nodiscard]] EXPORT json expm1_to_json(const UnaryIntrinsicKind::Expm1 &);
[[nodiscard]] EXPORT UnaryIntrinsicKind::Log log_from_json(const json &);
[[nodiscard]] EXPORT json log_to_json(const UnaryIntrinsicKind::Log &);
[[nodiscard]] EXPORT UnaryIntrinsicKind::Log1p log1p_from_json(const json &);
[[nodiscard]] EXPORT json log1p_to_json(const UnaryIntrinsicKind::Log1p &);
[[nodiscard]] EXPORT UnaryIntrinsicKind::Log10 log10_from_json(const json &);
[[nodiscard]] EXPORT json log10_to_json(const UnaryIntrinsicKind::Log10 &);
[[nodiscard]] EXPORT UnaryIntrinsicKind::BNot bnot_from_json(const json &);
[[nodiscard]] EXPORT json bnot_to_json(const UnaryIntrinsicKind::BNot &);
[[nodiscard]] EXPORT UnaryIntrinsicKind::Pos pos_from_json(const json &);
[[nodiscard]] EXPORT json pos_to_json(const UnaryIntrinsicKind::Pos &);
[[nodiscard]] EXPORT UnaryIntrinsicKind::Neg neg_from_json(const json &);
[[nodiscard]] EXPORT json neg_to_json(const UnaryIntrinsicKind::Neg &);
[[nodiscard]] EXPORT UnaryIntrinsicKind::Any any_from_json(const json &);
[[nodiscard]] EXPORT json any_to_json(const UnaryIntrinsicKind::Any &);
} // namespace UnaryIntrinsicKind
namespace UnaryLogicIntrinsicKind { 
[[nodiscard]] EXPORT UnaryLogicIntrinsicKind::Not not_from_json(const json &);
[[nodiscard]] EXPORT json not_to_json(const UnaryLogicIntrinsicKind::Not &);
[[nodiscard]] EXPORT UnaryLogicIntrinsicKind::Any any_from_json(const json &);
[[nodiscard]] EXPORT json any_to_json(const UnaryLogicIntrinsicKind::Any &);
} // namespace UnaryLogicIntrinsicKind
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
namespace Term { 
[[nodiscard]] EXPORT Term::Select select_from_json(const json &);
[[nodiscard]] EXPORT json select_to_json(const Term::Select &);
[[nodiscard]] EXPORT Term::UnitConst unitconst_from_json(const json &);
[[nodiscard]] EXPORT json unitconst_to_json(const Term::UnitConst &);
[[nodiscard]] EXPORT Term::BoolConst boolconst_from_json(const json &);
[[nodiscard]] EXPORT json boolconst_to_json(const Term::BoolConst &);
[[nodiscard]] EXPORT Term::ByteConst byteconst_from_json(const json &);
[[nodiscard]] EXPORT json byteconst_to_json(const Term::ByteConst &);
[[nodiscard]] EXPORT Term::CharConst charconst_from_json(const json &);
[[nodiscard]] EXPORT json charconst_to_json(const Term::CharConst &);
[[nodiscard]] EXPORT Term::ShortConst shortconst_from_json(const json &);
[[nodiscard]] EXPORT json shortconst_to_json(const Term::ShortConst &);
[[nodiscard]] EXPORT Term::IntConst intconst_from_json(const json &);
[[nodiscard]] EXPORT json intconst_to_json(const Term::IntConst &);
[[nodiscard]] EXPORT Term::LongConst longconst_from_json(const json &);
[[nodiscard]] EXPORT json longconst_to_json(const Term::LongConst &);
[[nodiscard]] EXPORT Term::FloatConst floatconst_from_json(const json &);
[[nodiscard]] EXPORT json floatconst_to_json(const Term::FloatConst &);
[[nodiscard]] EXPORT Term::DoubleConst doubleconst_from_json(const json &);
[[nodiscard]] EXPORT json doubleconst_to_json(const Term::DoubleConst &);
[[nodiscard]] EXPORT Term::StringConst stringconst_from_json(const json &);
[[nodiscard]] EXPORT json stringconst_to_json(const Term::StringConst &);
[[nodiscard]] EXPORT Term::Any any_from_json(const json &);
[[nodiscard]] EXPORT json any_to_json(const Term::Any &);
} // namespace Term
[[nodiscard]] EXPORT Sym sym_from_json(const json &);
[[nodiscard]] EXPORT json sym_to_json(const Sym &);
[[nodiscard]] EXPORT Named named_from_json(const json &);
[[nodiscard]] EXPORT json named_to_json(const Named &);
[[nodiscard]] EXPORT Position position_from_json(const json &);
[[nodiscard]] EXPORT json position_to_json(const Position &);
[[nodiscard]] EXPORT StructDef structdef_from_json(const json &);
[[nodiscard]] EXPORT json structdef_to_json(const StructDef &);
[[nodiscard]] EXPORT Signature signature_from_json(const json &);
[[nodiscard]] EXPORT json signature_to_json(const Signature &);
[[nodiscard]] EXPORT Function function_from_json(const json &);
[[nodiscard]] EXPORT json function_to_json(const Function &);
[[nodiscard]] EXPORT Program program_from_json(const json &);
[[nodiscard]] EXPORT json program_to_json(const Program &);
namespace Type { 
[[nodiscard]] EXPORT Type::Float float_from_json(const json &);
[[nodiscard]] EXPORT json float_to_json(const Type::Float &);
[[nodiscard]] EXPORT Type::Double double_from_json(const json &);
[[nodiscard]] EXPORT json double_to_json(const Type::Double &);
[[nodiscard]] EXPORT Type::Bool bool_from_json(const json &);
[[nodiscard]] EXPORT json bool_to_json(const Type::Bool &);
[[nodiscard]] EXPORT Type::Byte byte_from_json(const json &);
[[nodiscard]] EXPORT json byte_to_json(const Type::Byte &);
[[nodiscard]] EXPORT Type::Char char_from_json(const json &);
[[nodiscard]] EXPORT json char_to_json(const Type::Char &);
[[nodiscard]] EXPORT Type::Short short_from_json(const json &);
[[nodiscard]] EXPORT json short_to_json(const Type::Short &);
[[nodiscard]] EXPORT Type::Int int_from_json(const json &);
[[nodiscard]] EXPORT json int_to_json(const Type::Int &);
[[nodiscard]] EXPORT Type::Long long_from_json(const json &);
[[nodiscard]] EXPORT json long_to_json(const Type::Long &);
[[nodiscard]] EXPORT Type::Unit unit_from_json(const json &);
[[nodiscard]] EXPORT json unit_to_json(const Type::Unit &);
[[nodiscard]] EXPORT Type::String string_from_json(const json &);
[[nodiscard]] EXPORT json string_to_json(const Type::String &);
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
namespace BinaryIntrinsicKind { 
[[nodiscard]] EXPORT BinaryIntrinsicKind::Add add_from_json(const json &);
[[nodiscard]] EXPORT json add_to_json(const BinaryIntrinsicKind::Add &);
[[nodiscard]] EXPORT BinaryIntrinsicKind::Sub sub_from_json(const json &);
[[nodiscard]] EXPORT json sub_to_json(const BinaryIntrinsicKind::Sub &);
[[nodiscard]] EXPORT BinaryIntrinsicKind::Mul mul_from_json(const json &);
[[nodiscard]] EXPORT json mul_to_json(const BinaryIntrinsicKind::Mul &);
[[nodiscard]] EXPORT BinaryIntrinsicKind::Div div_from_json(const json &);
[[nodiscard]] EXPORT json div_to_json(const BinaryIntrinsicKind::Div &);
[[nodiscard]] EXPORT BinaryIntrinsicKind::Rem rem_from_json(const json &);
[[nodiscard]] EXPORT json rem_to_json(const BinaryIntrinsicKind::Rem &);
[[nodiscard]] EXPORT BinaryIntrinsicKind::Pow pow_from_json(const json &);
[[nodiscard]] EXPORT json pow_to_json(const BinaryIntrinsicKind::Pow &);
[[nodiscard]] EXPORT BinaryIntrinsicKind::Min min_from_json(const json &);
[[nodiscard]] EXPORT json min_to_json(const BinaryIntrinsicKind::Min &);
[[nodiscard]] EXPORT BinaryIntrinsicKind::Max max_from_json(const json &);
[[nodiscard]] EXPORT json max_to_json(const BinaryIntrinsicKind::Max &);
[[nodiscard]] EXPORT BinaryIntrinsicKind::Atan2 atan2_from_json(const json &);
[[nodiscard]] EXPORT json atan2_to_json(const BinaryIntrinsicKind::Atan2 &);
[[nodiscard]] EXPORT BinaryIntrinsicKind::Hypot hypot_from_json(const json &);
[[nodiscard]] EXPORT json hypot_to_json(const BinaryIntrinsicKind::Hypot &);
[[nodiscard]] EXPORT BinaryIntrinsicKind::BAnd band_from_json(const json &);
[[nodiscard]] EXPORT json band_to_json(const BinaryIntrinsicKind::BAnd &);
[[nodiscard]] EXPORT BinaryIntrinsicKind::BOr bor_from_json(const json &);
[[nodiscard]] EXPORT json bor_to_json(const BinaryIntrinsicKind::BOr &);
[[nodiscard]] EXPORT BinaryIntrinsicKind::BXor bxor_from_json(const json &);
[[nodiscard]] EXPORT json bxor_to_json(const BinaryIntrinsicKind::BXor &);
[[nodiscard]] EXPORT BinaryIntrinsicKind::BSL bsl_from_json(const json &);
[[nodiscard]] EXPORT json bsl_to_json(const BinaryIntrinsicKind::BSL &);
[[nodiscard]] EXPORT BinaryIntrinsicKind::BSR bsr_from_json(const json &);
[[nodiscard]] EXPORT json bsr_to_json(const BinaryIntrinsicKind::BSR &);
[[nodiscard]] EXPORT BinaryIntrinsicKind::BZSR bzsr_from_json(const json &);
[[nodiscard]] EXPORT json bzsr_to_json(const BinaryIntrinsicKind::BZSR &);
[[nodiscard]] EXPORT BinaryIntrinsicKind::Any any_from_json(const json &);
[[nodiscard]] EXPORT json any_to_json(const BinaryIntrinsicKind::Any &);
} // namespace BinaryIntrinsicKind
[[nodiscard]] EXPORT json hashed_to_json(const json&);
[[nodiscard]] EXPORT json hashed_from_json(const json&);
} // namespace polyregion::polyast

