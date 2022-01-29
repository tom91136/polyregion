#pragma once

#include "json.hpp"
#include "polyast.h"
#include "export.h"

using json = nlohmann::json;

namespace polyregion::polyast { 
namespace Expr { 
[[nodiscard]] EXPORT Expr::Sin sin_from_json(const json &);
[[nodiscard]] EXPORT json sin_to_json(const Expr::Sin &);
[[nodiscard]] EXPORT Expr::Cos cos_from_json(const json &);
[[nodiscard]] EXPORT json cos_to_json(const Expr::Cos &);
[[nodiscard]] EXPORT Expr::Tan tan_from_json(const json &);
[[nodiscard]] EXPORT json tan_to_json(const Expr::Tan &);
[[nodiscard]] EXPORT Expr::Abs abs_from_json(const json &);
[[nodiscard]] EXPORT json abs_to_json(const Expr::Abs &);
[[nodiscard]] EXPORT Expr::Add add_from_json(const json &);
[[nodiscard]] EXPORT json add_to_json(const Expr::Add &);
[[nodiscard]] EXPORT Expr::Sub sub_from_json(const json &);
[[nodiscard]] EXPORT json sub_to_json(const Expr::Sub &);
[[nodiscard]] EXPORT Expr::Mul mul_from_json(const json &);
[[nodiscard]] EXPORT json mul_to_json(const Expr::Mul &);
[[nodiscard]] EXPORT Expr::Div div_from_json(const json &);
[[nodiscard]] EXPORT json div_to_json(const Expr::Div &);
[[nodiscard]] EXPORT Expr::Rem rem_from_json(const json &);
[[nodiscard]] EXPORT json rem_to_json(const Expr::Rem &);
[[nodiscard]] EXPORT Expr::Pow pow_from_json(const json &);
[[nodiscard]] EXPORT json pow_to_json(const Expr::Pow &);
[[nodiscard]] EXPORT Expr::BNot bnot_from_json(const json &);
[[nodiscard]] EXPORT json bnot_to_json(const Expr::BNot &);
[[nodiscard]] EXPORT Expr::BAnd band_from_json(const json &);
[[nodiscard]] EXPORT json band_to_json(const Expr::BAnd &);
[[nodiscard]] EXPORT Expr::BOr bor_from_json(const json &);
[[nodiscard]] EXPORT json bor_to_json(const Expr::BOr &);
[[nodiscard]] EXPORT Expr::BXor bxor_from_json(const json &);
[[nodiscard]] EXPORT json bxor_to_json(const Expr::BXor &);
[[nodiscard]] EXPORT Expr::BSL bsl_from_json(const json &);
[[nodiscard]] EXPORT json bsl_to_json(const Expr::BSL &);
[[nodiscard]] EXPORT Expr::BSR bsr_from_json(const json &);
[[nodiscard]] EXPORT json bsr_to_json(const Expr::BSR &);
[[nodiscard]] EXPORT Expr::Not not_from_json(const json &);
[[nodiscard]] EXPORT json not_to_json(const Expr::Not &);
[[nodiscard]] EXPORT Expr::Eq eq_from_json(const json &);
[[nodiscard]] EXPORT json eq_to_json(const Expr::Eq &);
[[nodiscard]] EXPORT Expr::Neq neq_from_json(const json &);
[[nodiscard]] EXPORT json neq_to_json(const Expr::Neq &);
[[nodiscard]] EXPORT Expr::And and_from_json(const json &);
[[nodiscard]] EXPORT json and_to_json(const Expr::And &);
[[nodiscard]] EXPORT Expr::Or or_from_json(const json &);
[[nodiscard]] EXPORT json or_to_json(const Expr::Or &);
[[nodiscard]] EXPORT Expr::Lte lte_from_json(const json &);
[[nodiscard]] EXPORT json lte_to_json(const Expr::Lte &);
[[nodiscard]] EXPORT Expr::Gte gte_from_json(const json &);
[[nodiscard]] EXPORT json gte_to_json(const Expr::Gte &);
[[nodiscard]] EXPORT Expr::Lt lt_from_json(const json &);
[[nodiscard]] EXPORT json lt_to_json(const Expr::Lt &);
[[nodiscard]] EXPORT Expr::Gt gt_from_json(const json &);
[[nodiscard]] EXPORT json gt_to_json(const Expr::Gt &);
[[nodiscard]] EXPORT Expr::Alias alias_from_json(const json &);
[[nodiscard]] EXPORT json alias_to_json(const Expr::Alias &);
[[nodiscard]] EXPORT Expr::Invoke invoke_from_json(const json &);
[[nodiscard]] EXPORT json invoke_to_json(const Expr::Invoke &);
[[nodiscard]] EXPORT Expr::Index index_from_json(const json &);
[[nodiscard]] EXPORT json index_to_json(const Expr::Index &);
[[nodiscard]] EXPORT Expr::Any any_from_json(const json &);
[[nodiscard]] EXPORT json any_to_json(const Expr::Any &);
} // namespace Expr
namespace Stmt { 
[[nodiscard]] EXPORT Stmt::Comment comment_from_json(const json &);
[[nodiscard]] EXPORT json comment_to_json(const Stmt::Comment &);
[[nodiscard]] EXPORT Stmt::Var var_from_json(const json &);
[[nodiscard]] EXPORT json var_to_json(const Stmt::Var &);
[[nodiscard]] EXPORT Stmt::Mut mut_from_json(const json &);
[[nodiscard]] EXPORT json mut_to_json(const Stmt::Mut &);
[[nodiscard]] EXPORT Stmt::Update update_from_json(const json &);
[[nodiscard]] EXPORT json update_to_json(const Stmt::Update &);
[[nodiscard]] EXPORT Stmt::Effect effect_from_json(const json &);
[[nodiscard]] EXPORT json effect_to_json(const Stmt::Effect &);
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
namespace TypeKind { 
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
[[nodiscard]] EXPORT Type::String string_from_json(const json &);
[[nodiscard]] EXPORT json string_to_json(const Type::String &);
[[nodiscard]] EXPORT Type::Unit unit_from_json(const json &);
[[nodiscard]] EXPORT json unit_to_json(const Type::Unit &);
[[nodiscard]] EXPORT Type::Struct struct_from_json(const json &);
[[nodiscard]] EXPORT json struct_to_json(const Type::Struct &);
[[nodiscard]] EXPORT Type::Array array_from_json(const json &);
[[nodiscard]] EXPORT json array_to_json(const Type::Array &);
[[nodiscard]] EXPORT Type::Any any_from_json(const json &);
[[nodiscard]] EXPORT json any_to_json(const Type::Any &);
} // namespace Type
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
[[nodiscard]] EXPORT Function function_from_json(const json &);
[[nodiscard]] EXPORT json function_to_json(const Function &);
[[nodiscard]] EXPORT Program program_from_json(const json &);
[[nodiscard]] EXPORT json program_to_json(const Program &);
[[nodiscard]] EXPORT json hashed_to_json(const json&);
[[nodiscard]] EXPORT json hashed_from_json(const json&);
} // namespace polyregion::polyast

