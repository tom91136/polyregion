#include "polyast_codec.h"
namespace polyregion::polyast { 

Sym sym_json(const json& j) { 
  auto fqn = j.at(0).get<std::vector<std::string>>();
  return Sym(fqn);
}

TypeKind::Ref TypeKind::ref_json(const json& j) { 
  return {};
}

TypeKind::Integral TypeKind::integral_json(const json& j) { 
  return {};
}

TypeKind::Fractional TypeKind::fractional_json(const json& j) { 
  return {};
}

TypeKind::Any TypeKind::any_json(const json& j) { 
  size_t ord = j.at(0).get<size_t>();
  const auto t = j.at(1);
  switch (ord) {
  case 0: return TypeKind::ref_json(t);
  case 1: return TypeKind::integral_json(t);
  case 2: return TypeKind::fractional_json(t);
  default: throw std::out_of_range("Bad ordinal " + std::to_string(ord));
  }
}

Type::Float Type::float_json(const json& j) { 
  return {};
}

Type::Double Type::double_json(const json& j) { 
  return {};
}

Type::Bool Type::bool_json(const json& j) { 
  return {};
}

Type::Byte Type::byte_json(const json& j) { 
  return {};
}

Type::Char Type::char_json(const json& j) { 
  return {};
}

Type::Short Type::short_json(const json& j) { 
  return {};
}

Type::Int Type::int_json(const json& j) { 
  return {};
}

Type::Long Type::long_json(const json& j) { 
  return {};
}

Type::String Type::string_json(const json& j) { 
  return {};
}

Type::Unit Type::unit_json(const json& j) { 
  return {};
}

Type::Struct Type::struct_json(const json& j) { 
  auto name =  sym_json(j.at(0));
  std::vector<Type::Any> args;
  auto args_json = j.at(1);
  std::transform(args_json.begin(), args_json.end(), std::back_inserter(args), &Type::any_json);
  return {name, args};
}

Type::Array Type::array_json(const json& j) { 
  auto component =  Type::any_json(j.at(0));
  return Type::Array(component);
}

Type::Any Type::any_json(const json& j) { 
  size_t ord = j.at(0).get<size_t>();
  const auto t = j.at(1);
  switch (ord) {
  case 0: return Type::float_json(t);
  case 1: return Type::double_json(t);
  case 2: return Type::bool_json(t);
  case 3: return Type::byte_json(t);
  case 4: return Type::char_json(t);
  case 5: return Type::short_json(t);
  case 6: return Type::int_json(t);
  case 7: return Type::long_json(t);
  case 8: return Type::string_json(t);
  case 9: return Type::unit_json(t);
  case 10: return Type::struct_json(t);
  case 11: return Type::array_json(t);
  default: throw std::out_of_range("Bad ordinal " + std::to_string(ord));
  }
}

Named named_json(const json& j) { 
  auto symbol = j.at(0).get<std::string>();
  auto tpe =  Type::any_json(j.at(1));
  return {symbol, tpe};
}

Position position_json(const json& j) { 
  auto file = j.at(0).get<std::string>();
  auto line = j.at(1).get<int32_t>();
  auto col = j.at(2).get<int32_t>();
  return {file, line, col};
}

Term::Select Term::select_json(const json& j) { 
  std::vector<Named> init;
  auto init_json = j.at(0);
  std::transform(init_json.begin(), init_json.end(), std::back_inserter(init), &named_json);
  auto last =  named_json(j.at(1));
  return {init, last};
}

Term::BoolConst Term::boolconst_json(const json& j) { 
  auto value = j.at(0).get<bool>();
  return Term::BoolConst(value);
}

Term::ByteConst Term::byteconst_json(const json& j) { 
  auto value = j.at(0).get<int8_t>();
  return Term::ByteConst(value);
}

Term::CharConst Term::charconst_json(const json& j) { 
  auto value = j.at(0).get<uint16_t>();
  return Term::CharConst(value);
}

Term::ShortConst Term::shortconst_json(const json& j) { 
  auto value = j.at(0).get<int16_t>();
  return Term::ShortConst(value);
}

Term::IntConst Term::intconst_json(const json& j) { 
  auto value = j.at(0).get<int32_t>();
  return Term::IntConst(value);
}

Term::LongConst Term::longconst_json(const json& j) { 
  auto value = j.at(0).get<int64_t>();
  return Term::LongConst(value);
}

Term::FloatConst Term::floatconst_json(const json& j) { 
  auto value = j.at(0).get<float>();
  return Term::FloatConst(value);
}

Term::DoubleConst Term::doubleconst_json(const json& j) { 
  auto value = j.at(0).get<double>();
  return Term::DoubleConst(value);
}

Term::StringConst Term::stringconst_json(const json& j) { 
  auto value = j.at(0).get<std::string>();
  return Term::StringConst(value);
}

Term::Any Term::any_json(const json& j) { 
  size_t ord = j.at(0).get<size_t>();
  const auto t = j.at(1);
  switch (ord) {
  case 0: return Term::select_json(t);
  case 1: return Term::boolconst_json(t);
  case 2: return Term::byteconst_json(t);
  case 3: return Term::charconst_json(t);
  case 4: return Term::shortconst_json(t);
  case 5: return Term::intconst_json(t);
  case 6: return Term::longconst_json(t);
  case 7: return Term::floatconst_json(t);
  case 8: return Term::doubleconst_json(t);
  case 9: return Term::stringconst_json(t);
  default: throw std::out_of_range("Bad ordinal " + std::to_string(ord));
  }
}

Expr::Sin Expr::sin_json(const json& j) { 
  auto lhs =  Term::any_json(j.at(0));
  auto rtn =  Type::any_json(j.at(1));
  return {lhs, rtn};
}

Expr::Cos Expr::cos_json(const json& j) { 
  auto lhs =  Term::any_json(j.at(0));
  auto rtn =  Type::any_json(j.at(1));
  return {lhs, rtn};
}

Expr::Tan Expr::tan_json(const json& j) { 
  auto lhs =  Term::any_json(j.at(0));
  auto rtn =  Type::any_json(j.at(1));
  return {lhs, rtn};
}

Expr::Add Expr::add_json(const json& j) { 
  auto lhs =  Term::any_json(j.at(0));
  auto rhs =  Term::any_json(j.at(1));
  auto rtn =  Type::any_json(j.at(2));
  return {lhs, rhs, rtn};
}

Expr::Sub Expr::sub_json(const json& j) { 
  auto lhs =  Term::any_json(j.at(0));
  auto rhs =  Term::any_json(j.at(1));
  auto rtn =  Type::any_json(j.at(2));
  return {lhs, rhs, rtn};
}

Expr::Mul Expr::mul_json(const json& j) { 
  auto lhs =  Term::any_json(j.at(0));
  auto rhs =  Term::any_json(j.at(1));
  auto rtn =  Type::any_json(j.at(2));
  return {lhs, rhs, rtn};
}

Expr::Div Expr::div_json(const json& j) { 
  auto lhs =  Term::any_json(j.at(0));
  auto rhs =  Term::any_json(j.at(1));
  auto rtn =  Type::any_json(j.at(2));
  return {lhs, rhs, rtn};
}

Expr::Mod Expr::mod_json(const json& j) { 
  auto lhs =  Term::any_json(j.at(0));
  auto rhs =  Term::any_json(j.at(1));
  auto rtn =  Type::any_json(j.at(2));
  return {lhs, rhs, rtn};
}

Expr::Pow Expr::pow_json(const json& j) { 
  auto lhs =  Term::any_json(j.at(0));
  auto rhs =  Term::any_json(j.at(1));
  auto rtn =  Type::any_json(j.at(2));
  return {lhs, rhs, rtn};
}

Expr::Inv Expr::inv_json(const json& j) { 
  auto lhs =  Term::any_json(j.at(0));
  return Expr::Inv(lhs);
}

Expr::Eq Expr::eq_json(const json& j) { 
  auto lhs =  Term::any_json(j.at(0));
  auto rhs =  Term::any_json(j.at(1));
  return {lhs, rhs};
}

Expr::Lte Expr::lte_json(const json& j) { 
  auto lhs =  Term::any_json(j.at(0));
  auto rhs =  Term::any_json(j.at(1));
  return {lhs, rhs};
}

Expr::Gte Expr::gte_json(const json& j) { 
  auto lhs =  Term::any_json(j.at(0));
  auto rhs =  Term::any_json(j.at(1));
  return {lhs, rhs};
}

Expr::Lt Expr::lt_json(const json& j) { 
  auto lhs =  Term::any_json(j.at(0));
  auto rhs =  Term::any_json(j.at(1));
  return {lhs, rhs};
}

Expr::Gt Expr::gt_json(const json& j) { 
  auto lhs =  Term::any_json(j.at(0));
  auto rhs =  Term::any_json(j.at(1));
  return {lhs, rhs};
}

Expr::Alias Expr::alias_json(const json& j) { 
  auto ref =  Term::any_json(j.at(0));
  return Expr::Alias(ref);
}

Expr::Invoke Expr::invoke_json(const json& j) { 
  auto lhs =  Term::any_json(j.at(0));
  auto name = j.at(1).get<std::string>();
  std::vector<Term::Any> args;
  auto args_json = j.at(2);
  std::transform(args_json.begin(), args_json.end(), std::back_inserter(args), &Term::any_json);
  auto rtn =  Type::any_json(j.at(3));
  return {lhs, name, args, rtn};
}

Expr::Index Expr::index_json(const json& j) { 
  auto lhs =  Term::select_json(j.at(0));
  auto idx =  Term::any_json(j.at(1));
  auto component =  Type::any_json(j.at(2));
  return {lhs, idx, component};
}

Expr::Any Expr::any_json(const json& j) { 
  size_t ord = j.at(0).get<size_t>();
  const auto t = j.at(1);
  switch (ord) {
  case 0: return Expr::sin_json(t);
  case 1: return Expr::cos_json(t);
  case 2: return Expr::tan_json(t);
  case 3: return Expr::add_json(t);
  case 4: return Expr::sub_json(t);
  case 5: return Expr::mul_json(t);
  case 6: return Expr::div_json(t);
  case 7: return Expr::mod_json(t);
  case 8: return Expr::pow_json(t);
  case 9: return Expr::inv_json(t);
  case 10: return Expr::eq_json(t);
  case 11: return Expr::lte_json(t);
  case 12: return Expr::gte_json(t);
  case 13: return Expr::lt_json(t);
  case 14: return Expr::gt_json(t);
  case 15: return Expr::alias_json(t);
  case 16: return Expr::invoke_json(t);
  case 17: return Expr::index_json(t);
  default: throw std::out_of_range("Bad ordinal " + std::to_string(ord));
  }
}

Stmt::Comment Stmt::comment_json(const json& j) { 
  auto value = j.at(0).get<std::string>();
  return Stmt::Comment(value);
}

Stmt::Var Stmt::var_json(const json& j) { 
  auto name =  named_json(j.at(0));
  auto expr =  Expr::any_json(j.at(1));
  return {name, expr};
}

Stmt::Mut Stmt::mut_json(const json& j) { 
  auto name =  Term::select_json(j.at(0));
  auto expr =  Expr::any_json(j.at(1));
  return {name, expr};
}

Stmt::Update Stmt::update_json(const json& j) { 
  auto lhs =  Term::select_json(j.at(0));
  auto idx =  Term::any_json(j.at(1));
  auto value =  Term::any_json(j.at(2));
  return {lhs, idx, value};
}

Stmt::Effect Stmt::effect_json(const json& j) { 
  auto lhs =  Term::select_json(j.at(0));
  auto name = j.at(1).get<std::string>();
  std::vector<Term::Any> args;
  auto args_json = j.at(2);
  std::transform(args_json.begin(), args_json.end(), std::back_inserter(args), &Term::any_json);
  return {lhs, name, args};
}

Stmt::While Stmt::while_json(const json& j) { 
  auto cond =  Expr::any_json(j.at(0));
  std::vector<Stmt::Any> body;
  auto body_json = j.at(1);
  std::transform(body_json.begin(), body_json.end(), std::back_inserter(body), &Stmt::any_json);
  return {cond, body};
}

Stmt::Break Stmt::break_json(const json& j) { 
  return {};
}

Stmt::Cont Stmt::cont_json(const json& j) { 
  return {};
}

Stmt::Cond Stmt::cond_json(const json& j) { 
  auto cond =  Expr::any_json(j.at(0));
  std::vector<Stmt::Any> trueBr;
  auto trueBr_json = j.at(1);
  std::transform(trueBr_json.begin(), trueBr_json.end(), std::back_inserter(trueBr), &Stmt::any_json);
  std::vector<Stmt::Any> falseBr;
  auto falseBr_json = j.at(2);
  std::transform(falseBr_json.begin(), falseBr_json.end(), std::back_inserter(falseBr), &Stmt::any_json);
  return {cond, trueBr, falseBr};
}

Stmt::Return Stmt::return_json(const json& j) { 
  auto value =  Expr::any_json(j.at(0));
  return Stmt::Return(value);
}

Stmt::Any Stmt::any_json(const json& j) { 
  size_t ord = j.at(0).get<size_t>();
  const auto t = j.at(1);
  switch (ord) {
  case 0: return Stmt::comment_json(t);
  case 1: return Stmt::var_json(t);
  case 2: return Stmt::mut_json(t);
  case 3: return Stmt::update_json(t);
  case 4: return Stmt::effect_json(t);
  case 5: return Stmt::while_json(t);
  case 6: return Stmt::break_json(t);
  case 7: return Stmt::cont_json(t);
  case 8: return Stmt::cond_json(t);
  case 9: return Stmt::return_json(t);
  default: throw std::out_of_range("Bad ordinal " + std::to_string(ord));
  }
}

Function function_json(const json& j) { 
  auto name = j.at(0).get<std::string>();
  std::vector<Named> args;
  auto args_json = j.at(1);
  std::transform(args_json.begin(), args_json.end(), std::back_inserter(args), &named_json);
  auto rtn =  Type::any_json(j.at(2));
  std::vector<Stmt::Any> body;
  auto body_json = j.at(3);
  std::transform(body_json.begin(), body_json.end(), std::back_inserter(body), &Stmt::any_json);
  return {name, args, rtn, body};
}

StructDef structdef_json(const json& j) { 
  std::vector<Named> members;
  auto members_json = j.at(0);
  std::transform(members_json.begin(), members_json.end(), std::back_inserter(members), &named_json);
  return StructDef(members);
}
} // namespace polyregion::polyast
