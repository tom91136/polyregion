#pragma once

#include "json.hpp"
#include "polyast.h"

using json = nlohmann::json;

namespace polyregion::polyast { 
namespace Expr { 
[[nodiscard]] Expr::Sin sin_json(const json &);
[[nodiscard]] Expr::Cos cos_json(const json &);
[[nodiscard]] Expr::Tan tan_json(const json &);
[[nodiscard]] Expr::Add add_json(const json &);
[[nodiscard]] Expr::Sub sub_json(const json &);
[[nodiscard]] Expr::Mul mul_json(const json &);
[[nodiscard]] Expr::Div div_json(const json &);
[[nodiscard]] Expr::Mod mod_json(const json &);
[[nodiscard]] Expr::Pow pow_json(const json &);
[[nodiscard]] Expr::Inv inv_json(const json &);
[[nodiscard]] Expr::Eq eq_json(const json &);
[[nodiscard]] Expr::Lte lte_json(const json &);
[[nodiscard]] Expr::Gte gte_json(const json &);
[[nodiscard]] Expr::Lt lt_json(const json &);
[[nodiscard]] Expr::Gt gt_json(const json &);
[[nodiscard]] Expr::Alias alias_json(const json &);
[[nodiscard]] Expr::Invoke invoke_json(const json &);
[[nodiscard]] Expr::Index index_json(const json &);
[[nodiscard]] Expr::Any any_json(const json &);
} // namespace Expr
namespace Stmt { 
[[nodiscard]] Stmt::Comment comment_json(const json &);
[[nodiscard]] Stmt::Var var_json(const json &);
[[nodiscard]] Stmt::Mut mut_json(const json &);
[[nodiscard]] Stmt::Update update_json(const json &);
[[nodiscard]] Stmt::Effect effect_json(const json &);
[[nodiscard]] Stmt::While while_json(const json &);
[[nodiscard]] Stmt::Break break_json(const json &);
[[nodiscard]] Stmt::Cont cont_json(const json &);
[[nodiscard]] Stmt::Cond cond_json(const json &);
[[nodiscard]] Stmt::Return return_json(const json &);
[[nodiscard]] Stmt::Any any_json(const json &);
} // namespace Stmt
namespace TypeKind { 
[[nodiscard]] TypeKind::Ref ref_json(const json &);
[[nodiscard]] TypeKind::Integral integral_json(const json &);
[[nodiscard]] TypeKind::Fractional fractional_json(const json &);
[[nodiscard]] TypeKind::Any any_json(const json &);
} // namespace TypeKind
namespace Type { 
[[nodiscard]] Type::Float float_json(const json &);
[[nodiscard]] Type::Double double_json(const json &);
[[nodiscard]] Type::Bool bool_json(const json &);
[[nodiscard]] Type::Byte byte_json(const json &);
[[nodiscard]] Type::Char char_json(const json &);
[[nodiscard]] Type::Short short_json(const json &);
[[nodiscard]] Type::Int int_json(const json &);
[[nodiscard]] Type::Long long_json(const json &);
[[nodiscard]] Type::String string_json(const json &);
[[nodiscard]] Type::Unit unit_json(const json &);
[[nodiscard]] Type::Struct struct_json(const json &);
[[nodiscard]] Type::Array array_json(const json &);
[[nodiscard]] Type::Any any_json(const json &);
} // namespace Type
namespace Term { 
[[nodiscard]] Term::Select select_json(const json &);
[[nodiscard]] Term::BoolConst boolconst_json(const json &);
[[nodiscard]] Term::ByteConst byteconst_json(const json &);
[[nodiscard]] Term::CharConst charconst_json(const json &);
[[nodiscard]] Term::ShortConst shortconst_json(const json &);
[[nodiscard]] Term::IntConst intconst_json(const json &);
[[nodiscard]] Term::LongConst longconst_json(const json &);
[[nodiscard]] Term::FloatConst floatconst_json(const json &);
[[nodiscard]] Term::DoubleConst doubleconst_json(const json &);
[[nodiscard]] Term::StringConst stringconst_json(const json &);
[[nodiscard]] Term::Any any_json(const json &);
} // namespace Term
[[nodiscard]] Sym sym_json(const json &);
[[nodiscard]] Named named_json(const json &);
[[nodiscard]] Position position_json(const json &);
[[nodiscard]] Function function_json(const json &);
[[nodiscard]] StructDef structdef_json(const json &);
} // namespace polyregion::polyast

