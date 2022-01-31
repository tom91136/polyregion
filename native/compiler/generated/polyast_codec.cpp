#include "polyast_codec.h"

template<class... Ts> struct overloaded : Ts... { using Ts::operator()...; };
template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

namespace polyregion::polyast { 

Sym sym_from_json(const json& j) { 
  auto fqn = j.at(0).get<std::vector<std::string>>();
  return Sym(fqn);
}

json sym_to_json(const Sym& x) { 
  auto fqn = x.fqn;
  return json::array({fqn});
}

TypeKind::Ref TypeKind::ref_from_json(const json& j) { 
  return {};
}

json TypeKind::ref_to_json(const TypeKind::Ref& x) { 
  return json::array({});
}

TypeKind::Integral TypeKind::integral_from_json(const json& j) { 
  return {};
}

json TypeKind::integral_to_json(const TypeKind::Integral& x) { 
  return json::array({});
}

TypeKind::Fractional TypeKind::fractional_from_json(const json& j) { 
  return {};
}

json TypeKind::fractional_to_json(const TypeKind::Fractional& x) { 
  return json::array({});
}

TypeKind::Any TypeKind::any_from_json(const json& j) { 
  size_t ord = j.at(0).get<size_t>();
  const auto t = j.at(1);
  switch (ord) {
  case 0: return TypeKind::ref_from_json(t);
  case 1: return TypeKind::integral_from_json(t);
  case 2: return TypeKind::fractional_from_json(t);
  default: throw std::out_of_range("Bad ordinal " + std::to_string(ord));
  }
}

json TypeKind::any_to_json(const TypeKind::Any& x) { 
  return std::visit(overloaded{
  [](const TypeKind::Ref &y) -> json { return {0, TypeKind::ref_to_json(y)}; },
  [](const TypeKind::Integral &y) -> json { return {1, TypeKind::integral_to_json(y)}; },
  [](const TypeKind::Fractional &y) -> json { return {2, TypeKind::fractional_to_json(y)}; },
  [](const auto &x) -> json { throw std::out_of_range("Unimplemented type:" + to_string(x) ); }
  }, *x);
}

Type::Float Type::float_from_json(const json& j) { 
  return {};
}

json Type::float_to_json(const Type::Float& x) { 
  return json::array({});
}

Type::Double Type::double_from_json(const json& j) { 
  return {};
}

json Type::double_to_json(const Type::Double& x) { 
  return json::array({});
}

Type::Bool Type::bool_from_json(const json& j) { 
  return {};
}

json Type::bool_to_json(const Type::Bool& x) { 
  return json::array({});
}

Type::Byte Type::byte_from_json(const json& j) { 
  return {};
}

json Type::byte_to_json(const Type::Byte& x) { 
  return json::array({});
}

Type::Char Type::char_from_json(const json& j) { 
  return {};
}

json Type::char_to_json(const Type::Char& x) { 
  return json::array({});
}

Type::Short Type::short_from_json(const json& j) { 
  return {};
}

json Type::short_to_json(const Type::Short& x) { 
  return json::array({});
}

Type::Int Type::int_from_json(const json& j) { 
  return {};
}

json Type::int_to_json(const Type::Int& x) { 
  return json::array({});
}

Type::Long Type::long_from_json(const json& j) { 
  return {};
}

json Type::long_to_json(const Type::Long& x) { 
  return json::array({});
}

Type::String Type::string_from_json(const json& j) { 
  return {};
}

json Type::string_to_json(const Type::String& x) { 
  return json::array({});
}

Type::Unit Type::unit_from_json(const json& j) { 
  return {};
}

json Type::unit_to_json(const Type::Unit& x) { 
  return json::array({});
}

Type::Struct Type::struct_from_json(const json& j) { 
  auto name =  sym_from_json(j.at(0));
  return Type::Struct(name);
}

json Type::struct_to_json(const Type::Struct& x) { 
  auto name =  sym_to_json(x.name);
  return json::array({name});
}

Type::Array Type::array_from_json(const json& j) { 
  auto component =  Type::any_from_json(j.at(0));
  auto length = j.at(1).is_null() ? std::nullopt : std::make_optional(j.at(1).get<int32_t>());
  return {component, length};
}

json Type::array_to_json(const Type::Array& x) { 
  auto component =  Type::any_to_json(x.component);
  auto length = x.length ? json{*x.length} : json{};
  return json::array({component, length});
}

Type::Any Type::any_from_json(const json& j) { 
  size_t ord = j.at(0).get<size_t>();
  const auto t = j.at(1);
  switch (ord) {
  case 0: return Type::float_from_json(t);
  case 1: return Type::double_from_json(t);
  case 2: return Type::bool_from_json(t);
  case 3: return Type::byte_from_json(t);
  case 4: return Type::char_from_json(t);
  case 5: return Type::short_from_json(t);
  case 6: return Type::int_from_json(t);
  case 7: return Type::long_from_json(t);
  case 8: return Type::string_from_json(t);
  case 9: return Type::unit_from_json(t);
  case 10: return Type::struct_from_json(t);
  case 11: return Type::array_from_json(t);
  default: throw std::out_of_range("Bad ordinal " + std::to_string(ord));
  }
}

json Type::any_to_json(const Type::Any& x) { 
  return std::visit(overloaded{
  [](const Type::Float &y) -> json { return {0, Type::float_to_json(y)}; },
  [](const Type::Double &y) -> json { return {1, Type::double_to_json(y)}; },
  [](const Type::Bool &y) -> json { return {2, Type::bool_to_json(y)}; },
  [](const Type::Byte &y) -> json { return {3, Type::byte_to_json(y)}; },
  [](const Type::Char &y) -> json { return {4, Type::char_to_json(y)}; },
  [](const Type::Short &y) -> json { return {5, Type::short_to_json(y)}; },
  [](const Type::Int &y) -> json { return {6, Type::int_to_json(y)}; },
  [](const Type::Long &y) -> json { return {7, Type::long_to_json(y)}; },
  [](const Type::String &y) -> json { return {8, Type::string_to_json(y)}; },
  [](const Type::Unit &y) -> json { return {9, Type::unit_to_json(y)}; },
  [](const Type::Struct &y) -> json { return {10, Type::struct_to_json(y)}; },
  [](const Type::Array &y) -> json { return {11, Type::array_to_json(y)}; },
  [](const auto &x) -> json { throw std::out_of_range("Unimplemented type:" + to_string(x) ); }
  }, *x);
}

Named named_from_json(const json& j) { 
  auto symbol = j.at(0).get<std::string>();
  auto tpe =  Type::any_from_json(j.at(1));
  return {symbol, tpe};
}

json named_to_json(const Named& x) { 
  auto symbol = x.symbol;
  auto tpe =  Type::any_to_json(x.tpe);
  return json::array({symbol, tpe});
}

Position position_from_json(const json& j) { 
  auto file = j.at(0).get<std::string>();
  auto line = j.at(1).get<int32_t>();
  auto col = j.at(2).get<int32_t>();
  return {file, line, col};
}

json position_to_json(const Position& x) { 
  auto file = x.file;
  auto line = x.line;
  auto col = x.col;
  return json::array({file, line, col});
}

Term::Select Term::select_from_json(const json& j) { 
  std::vector<Named> init;
  auto init_json = j.at(0);
  std::transform(init_json.begin(), init_json.end(), std::back_inserter(init), &named_from_json);
  auto last =  named_from_json(j.at(1));
  return {init, last};
}

json Term::select_to_json(const Term::Select& x) { 
  std::vector<json> init;
  std::transform(x.init.begin(), x.init.end(), std::back_inserter(init), &named_to_json);
  auto last =  named_to_json(x.last);
  return json::array({init, last});
}

Term::UnitConst Term::unitconst_from_json(const json& j) { 
  return {};
}

json Term::unitconst_to_json(const Term::UnitConst& x) { 
  return json::array({});
}

Term::BoolConst Term::boolconst_from_json(const json& j) { 
  auto value = j.at(0).get<bool>();
  return Term::BoolConst(value);
}

json Term::boolconst_to_json(const Term::BoolConst& x) { 
  auto value = x.value;
  return json::array({value});
}

Term::ByteConst Term::byteconst_from_json(const json& j) { 
  auto value = j.at(0).get<int8_t>();
  return Term::ByteConst(value);
}

json Term::byteconst_to_json(const Term::ByteConst& x) { 
  auto value = x.value;
  return json::array({value});
}

Term::CharConst Term::charconst_from_json(const json& j) { 
  auto value = j.at(0).get<uint16_t>();
  return Term::CharConst(value);
}

json Term::charconst_to_json(const Term::CharConst& x) { 
  auto value = x.value;
  return json::array({value});
}

Term::ShortConst Term::shortconst_from_json(const json& j) { 
  auto value = j.at(0).get<int16_t>();
  return Term::ShortConst(value);
}

json Term::shortconst_to_json(const Term::ShortConst& x) { 
  auto value = x.value;
  return json::array({value});
}

Term::IntConst Term::intconst_from_json(const json& j) { 
  auto value = j.at(0).get<int32_t>();
  return Term::IntConst(value);
}

json Term::intconst_to_json(const Term::IntConst& x) { 
  auto value = x.value;
  return json::array({value});
}

Term::LongConst Term::longconst_from_json(const json& j) { 
  auto value = j.at(0).get<int64_t>();
  return Term::LongConst(value);
}

json Term::longconst_to_json(const Term::LongConst& x) { 
  auto value = x.value;
  return json::array({value});
}

Term::FloatConst Term::floatconst_from_json(const json& j) { 
  auto value = j.at(0).get<float>();
  return Term::FloatConst(value);
}

json Term::floatconst_to_json(const Term::FloatConst& x) { 
  auto value = x.value;
  return json::array({value});
}

Term::DoubleConst Term::doubleconst_from_json(const json& j) { 
  auto value = j.at(0).get<double>();
  return Term::DoubleConst(value);
}

json Term::doubleconst_to_json(const Term::DoubleConst& x) { 
  auto value = x.value;
  return json::array({value});
}

Term::StringConst Term::stringconst_from_json(const json& j) { 
  auto value = j.at(0).get<std::string>();
  return Term::StringConst(value);
}

json Term::stringconst_to_json(const Term::StringConst& x) { 
  auto value = x.value;
  return json::array({value});
}

Term::Any Term::any_from_json(const json& j) { 
  size_t ord = j.at(0).get<size_t>();
  const auto t = j.at(1);
  switch (ord) {
  case 0: return Term::select_from_json(t);
  case 1: return Term::unitconst_from_json(t);
  case 2: return Term::boolconst_from_json(t);
  case 3: return Term::byteconst_from_json(t);
  case 4: return Term::charconst_from_json(t);
  case 5: return Term::shortconst_from_json(t);
  case 6: return Term::intconst_from_json(t);
  case 7: return Term::longconst_from_json(t);
  case 8: return Term::floatconst_from_json(t);
  case 9: return Term::doubleconst_from_json(t);
  case 10: return Term::stringconst_from_json(t);
  default: throw std::out_of_range("Bad ordinal " + std::to_string(ord));
  }
}

json Term::any_to_json(const Term::Any& x) { 
  return std::visit(overloaded{
  [](const Term::Select &y) -> json { return {0, Term::select_to_json(y)}; },
  [](const Term::UnitConst &y) -> json { return {1, Term::unitconst_to_json(y)}; },
  [](const Term::BoolConst &y) -> json { return {2, Term::boolconst_to_json(y)}; },
  [](const Term::ByteConst &y) -> json { return {3, Term::byteconst_to_json(y)}; },
  [](const Term::CharConst &y) -> json { return {4, Term::charconst_to_json(y)}; },
  [](const Term::ShortConst &y) -> json { return {5, Term::shortconst_to_json(y)}; },
  [](const Term::IntConst &y) -> json { return {6, Term::intconst_to_json(y)}; },
  [](const Term::LongConst &y) -> json { return {7, Term::longconst_to_json(y)}; },
  [](const Term::FloatConst &y) -> json { return {8, Term::floatconst_to_json(y)}; },
  [](const Term::DoubleConst &y) -> json { return {9, Term::doubleconst_to_json(y)}; },
  [](const Term::StringConst &y) -> json { return {10, Term::stringconst_to_json(y)}; },
  [](const auto &x) -> json { throw std::out_of_range("Unimplemented type:" + to_string(x) ); }
  }, *x);
}

Expr::Sin Expr::sin_from_json(const json& j) { 
  auto lhs =  Term::any_from_json(j.at(0));
  auto rtn =  Type::any_from_json(j.at(1));
  return {lhs, rtn};
}

json Expr::sin_to_json(const Expr::Sin& x) { 
  auto lhs =  Term::any_to_json(x.lhs);
  auto rtn =  Type::any_to_json(x.rtn);
  return json::array({lhs, rtn});
}

Expr::Cos Expr::cos_from_json(const json& j) { 
  auto lhs =  Term::any_from_json(j.at(0));
  auto rtn =  Type::any_from_json(j.at(1));
  return {lhs, rtn};
}

json Expr::cos_to_json(const Expr::Cos& x) { 
  auto lhs =  Term::any_to_json(x.lhs);
  auto rtn =  Type::any_to_json(x.rtn);
  return json::array({lhs, rtn});
}

Expr::Tan Expr::tan_from_json(const json& j) { 
  auto lhs =  Term::any_from_json(j.at(0));
  auto rtn =  Type::any_from_json(j.at(1));
  return {lhs, rtn};
}

json Expr::tan_to_json(const Expr::Tan& x) { 
  auto lhs =  Term::any_to_json(x.lhs);
  auto rtn =  Type::any_to_json(x.rtn);
  return json::array({lhs, rtn});
}

Expr::Abs Expr::abs_from_json(const json& j) { 
  auto lhs =  Term::any_from_json(j.at(0));
  auto rtn =  Type::any_from_json(j.at(1));
  return {lhs, rtn};
}

json Expr::abs_to_json(const Expr::Abs& x) { 
  auto lhs =  Term::any_to_json(x.lhs);
  auto rtn =  Type::any_to_json(x.rtn);
  return json::array({lhs, rtn});
}

Expr::Add Expr::add_from_json(const json& j) { 
  auto lhs =  Term::any_from_json(j.at(0));
  auto rhs =  Term::any_from_json(j.at(1));
  auto rtn =  Type::any_from_json(j.at(2));
  return {lhs, rhs, rtn};
}

json Expr::add_to_json(const Expr::Add& x) { 
  auto lhs =  Term::any_to_json(x.lhs);
  auto rhs =  Term::any_to_json(x.rhs);
  auto rtn =  Type::any_to_json(x.rtn);
  return json::array({lhs, rhs, rtn});
}

Expr::Sub Expr::sub_from_json(const json& j) { 
  auto lhs =  Term::any_from_json(j.at(0));
  auto rhs =  Term::any_from_json(j.at(1));
  auto rtn =  Type::any_from_json(j.at(2));
  return {lhs, rhs, rtn};
}

json Expr::sub_to_json(const Expr::Sub& x) { 
  auto lhs =  Term::any_to_json(x.lhs);
  auto rhs =  Term::any_to_json(x.rhs);
  auto rtn =  Type::any_to_json(x.rtn);
  return json::array({lhs, rhs, rtn});
}

Expr::Mul Expr::mul_from_json(const json& j) { 
  auto lhs =  Term::any_from_json(j.at(0));
  auto rhs =  Term::any_from_json(j.at(1));
  auto rtn =  Type::any_from_json(j.at(2));
  return {lhs, rhs, rtn};
}

json Expr::mul_to_json(const Expr::Mul& x) { 
  auto lhs =  Term::any_to_json(x.lhs);
  auto rhs =  Term::any_to_json(x.rhs);
  auto rtn =  Type::any_to_json(x.rtn);
  return json::array({lhs, rhs, rtn});
}

Expr::Div Expr::div_from_json(const json& j) { 
  auto lhs =  Term::any_from_json(j.at(0));
  auto rhs =  Term::any_from_json(j.at(1));
  auto rtn =  Type::any_from_json(j.at(2));
  return {lhs, rhs, rtn};
}

json Expr::div_to_json(const Expr::Div& x) { 
  auto lhs =  Term::any_to_json(x.lhs);
  auto rhs =  Term::any_to_json(x.rhs);
  auto rtn =  Type::any_to_json(x.rtn);
  return json::array({lhs, rhs, rtn});
}

Expr::Rem Expr::rem_from_json(const json& j) { 
  auto lhs =  Term::any_from_json(j.at(0));
  auto rhs =  Term::any_from_json(j.at(1));
  auto rtn =  Type::any_from_json(j.at(2));
  return {lhs, rhs, rtn};
}

json Expr::rem_to_json(const Expr::Rem& x) { 
  auto lhs =  Term::any_to_json(x.lhs);
  auto rhs =  Term::any_to_json(x.rhs);
  auto rtn =  Type::any_to_json(x.rtn);
  return json::array({lhs, rhs, rtn});
}

Expr::Pow Expr::pow_from_json(const json& j) { 
  auto lhs =  Term::any_from_json(j.at(0));
  auto rhs =  Term::any_from_json(j.at(1));
  auto rtn =  Type::any_from_json(j.at(2));
  return {lhs, rhs, rtn};
}

json Expr::pow_to_json(const Expr::Pow& x) { 
  auto lhs =  Term::any_to_json(x.lhs);
  auto rhs =  Term::any_to_json(x.rhs);
  auto rtn =  Type::any_to_json(x.rtn);
  return json::array({lhs, rhs, rtn});
}

Expr::BNot Expr::bnot_from_json(const json& j) { 
  auto lhs =  Term::any_from_json(j.at(0));
  auto rtn =  Type::any_from_json(j.at(1));
  return {lhs, rtn};
}

json Expr::bnot_to_json(const Expr::BNot& x) { 
  auto lhs =  Term::any_to_json(x.lhs);
  auto rtn =  Type::any_to_json(x.rtn);
  return json::array({lhs, rtn});
}

Expr::BAnd Expr::band_from_json(const json& j) { 
  auto lhs =  Term::any_from_json(j.at(0));
  auto rhs =  Term::any_from_json(j.at(1));
  auto rtn =  Type::any_from_json(j.at(2));
  return {lhs, rhs, rtn};
}

json Expr::band_to_json(const Expr::BAnd& x) { 
  auto lhs =  Term::any_to_json(x.lhs);
  auto rhs =  Term::any_to_json(x.rhs);
  auto rtn =  Type::any_to_json(x.rtn);
  return json::array({lhs, rhs, rtn});
}

Expr::BOr Expr::bor_from_json(const json& j) { 
  auto lhs =  Term::any_from_json(j.at(0));
  auto rhs =  Term::any_from_json(j.at(1));
  auto rtn =  Type::any_from_json(j.at(2));
  return {lhs, rhs, rtn};
}

json Expr::bor_to_json(const Expr::BOr& x) { 
  auto lhs =  Term::any_to_json(x.lhs);
  auto rhs =  Term::any_to_json(x.rhs);
  auto rtn =  Type::any_to_json(x.rtn);
  return json::array({lhs, rhs, rtn});
}

Expr::BXor Expr::bxor_from_json(const json& j) { 
  auto lhs =  Term::any_from_json(j.at(0));
  auto rhs =  Term::any_from_json(j.at(1));
  auto rtn =  Type::any_from_json(j.at(2));
  return {lhs, rhs, rtn};
}

json Expr::bxor_to_json(const Expr::BXor& x) { 
  auto lhs =  Term::any_to_json(x.lhs);
  auto rhs =  Term::any_to_json(x.rhs);
  auto rtn =  Type::any_to_json(x.rtn);
  return json::array({lhs, rhs, rtn});
}

Expr::BSL Expr::bsl_from_json(const json& j) { 
  auto lhs =  Term::any_from_json(j.at(0));
  auto rhs =  Term::any_from_json(j.at(1));
  auto rtn =  Type::any_from_json(j.at(2));
  return {lhs, rhs, rtn};
}

json Expr::bsl_to_json(const Expr::BSL& x) { 
  auto lhs =  Term::any_to_json(x.lhs);
  auto rhs =  Term::any_to_json(x.rhs);
  auto rtn =  Type::any_to_json(x.rtn);
  return json::array({lhs, rhs, rtn});
}

Expr::BSR Expr::bsr_from_json(const json& j) { 
  auto lhs =  Term::any_from_json(j.at(0));
  auto rhs =  Term::any_from_json(j.at(1));
  auto rtn =  Type::any_from_json(j.at(2));
  return {lhs, rhs, rtn};
}

json Expr::bsr_to_json(const Expr::BSR& x) { 
  auto lhs =  Term::any_to_json(x.lhs);
  auto rhs =  Term::any_to_json(x.rhs);
  auto rtn =  Type::any_to_json(x.rtn);
  return json::array({lhs, rhs, rtn});
}

Expr::Not Expr::not_from_json(const json& j) { 
  auto lhs =  Term::any_from_json(j.at(0));
  return Expr::Not(lhs);
}

json Expr::not_to_json(const Expr::Not& x) { 
  auto lhs =  Term::any_to_json(x.lhs);
  return json::array({lhs});
}

Expr::Eq Expr::eq_from_json(const json& j) { 
  auto lhs =  Term::any_from_json(j.at(0));
  auto rhs =  Term::any_from_json(j.at(1));
  return {lhs, rhs};
}

json Expr::eq_to_json(const Expr::Eq& x) { 
  auto lhs =  Term::any_to_json(x.lhs);
  auto rhs =  Term::any_to_json(x.rhs);
  return json::array({lhs, rhs});
}

Expr::Neq Expr::neq_from_json(const json& j) { 
  auto lhs =  Term::any_from_json(j.at(0));
  auto rhs =  Term::any_from_json(j.at(1));
  return {lhs, rhs};
}

json Expr::neq_to_json(const Expr::Neq& x) { 
  auto lhs =  Term::any_to_json(x.lhs);
  auto rhs =  Term::any_to_json(x.rhs);
  return json::array({lhs, rhs});
}

Expr::And Expr::and_from_json(const json& j) { 
  auto lhs =  Term::any_from_json(j.at(0));
  auto rhs =  Term::any_from_json(j.at(1));
  return {lhs, rhs};
}

json Expr::and_to_json(const Expr::And& x) { 
  auto lhs =  Term::any_to_json(x.lhs);
  auto rhs =  Term::any_to_json(x.rhs);
  return json::array({lhs, rhs});
}

Expr::Or Expr::or_from_json(const json& j) { 
  auto lhs =  Term::any_from_json(j.at(0));
  auto rhs =  Term::any_from_json(j.at(1));
  return {lhs, rhs};
}

json Expr::or_to_json(const Expr::Or& x) { 
  auto lhs =  Term::any_to_json(x.lhs);
  auto rhs =  Term::any_to_json(x.rhs);
  return json::array({lhs, rhs});
}

Expr::Lte Expr::lte_from_json(const json& j) { 
  auto lhs =  Term::any_from_json(j.at(0));
  auto rhs =  Term::any_from_json(j.at(1));
  return {lhs, rhs};
}

json Expr::lte_to_json(const Expr::Lte& x) { 
  auto lhs =  Term::any_to_json(x.lhs);
  auto rhs =  Term::any_to_json(x.rhs);
  return json::array({lhs, rhs});
}

Expr::Gte Expr::gte_from_json(const json& j) { 
  auto lhs =  Term::any_from_json(j.at(0));
  auto rhs =  Term::any_from_json(j.at(1));
  return {lhs, rhs};
}

json Expr::gte_to_json(const Expr::Gte& x) { 
  auto lhs =  Term::any_to_json(x.lhs);
  auto rhs =  Term::any_to_json(x.rhs);
  return json::array({lhs, rhs});
}

Expr::Lt Expr::lt_from_json(const json& j) { 
  auto lhs =  Term::any_from_json(j.at(0));
  auto rhs =  Term::any_from_json(j.at(1));
  return {lhs, rhs};
}

json Expr::lt_to_json(const Expr::Lt& x) { 
  auto lhs =  Term::any_to_json(x.lhs);
  auto rhs =  Term::any_to_json(x.rhs);
  return json::array({lhs, rhs});
}

Expr::Gt Expr::gt_from_json(const json& j) { 
  auto lhs =  Term::any_from_json(j.at(0));
  auto rhs =  Term::any_from_json(j.at(1));
  return {lhs, rhs};
}

json Expr::gt_to_json(const Expr::Gt& x) { 
  auto lhs =  Term::any_to_json(x.lhs);
  auto rhs =  Term::any_to_json(x.rhs);
  return json::array({lhs, rhs});
}

Expr::Alias Expr::alias_from_json(const json& j) { 
  auto ref =  Term::any_from_json(j.at(0));
  return Expr::Alias(ref);
}

json Expr::alias_to_json(const Expr::Alias& x) { 
  auto ref =  Term::any_to_json(x.ref);
  return json::array({ref});
}

Expr::Invoke Expr::invoke_from_json(const json& j) { 
  auto name =  sym_from_json(j.at(0));
  auto receiver = j.at(1).is_null() ? std::nullopt : std::make_optional(Term::any_from_json(j.at(1)));
  std::vector<Term::Any> args;
  auto args_json = j.at(2);
  std::transform(args_json.begin(), args_json.end(), std::back_inserter(args), &Term::any_from_json);
  auto rtn =  Type::any_from_json(j.at(3));
  return {name, receiver, args, rtn};
}

json Expr::invoke_to_json(const Expr::Invoke& x) { 
  auto name =  sym_to_json(x.name);
  auto receiver = x.receiver ? Term::any_to_json(*x.receiver) : json{};
  std::vector<json> args;
  std::transform(x.args.begin(), x.args.end(), std::back_inserter(args), &Term::any_to_json);
  auto rtn =  Type::any_to_json(x.rtn);
  return json::array({name, receiver, args, rtn});
}

Expr::Index Expr::index_from_json(const json& j) { 
  auto lhs =  Term::select_from_json(j.at(0));
  auto idx =  Term::any_from_json(j.at(1));
  auto component =  Type::any_from_json(j.at(2));
  return {lhs, idx, component};
}

json Expr::index_to_json(const Expr::Index& x) { 
  auto lhs =  Term::select_to_json(x.lhs);
  auto idx =  Term::any_to_json(x.idx);
  auto component =  Type::any_to_json(x.component);
  return json::array({lhs, idx, component});
}

Expr::Any Expr::any_from_json(const json& j) { 
  size_t ord = j.at(0).get<size_t>();
  const auto t = j.at(1);
  switch (ord) {
  case 0: return Expr::sin_from_json(t);
  case 1: return Expr::cos_from_json(t);
  case 2: return Expr::tan_from_json(t);
  case 3: return Expr::abs_from_json(t);
  case 4: return Expr::add_from_json(t);
  case 5: return Expr::sub_from_json(t);
  case 6: return Expr::mul_from_json(t);
  case 7: return Expr::div_from_json(t);
  case 8: return Expr::rem_from_json(t);
  case 9: return Expr::pow_from_json(t);
  case 10: return Expr::bnot_from_json(t);
  case 11: return Expr::band_from_json(t);
  case 12: return Expr::bor_from_json(t);
  case 13: return Expr::bxor_from_json(t);
  case 14: return Expr::bsl_from_json(t);
  case 15: return Expr::bsr_from_json(t);
  case 16: return Expr::not_from_json(t);
  case 17: return Expr::eq_from_json(t);
  case 18: return Expr::neq_from_json(t);
  case 19: return Expr::and_from_json(t);
  case 20: return Expr::or_from_json(t);
  case 21: return Expr::lte_from_json(t);
  case 22: return Expr::gte_from_json(t);
  case 23: return Expr::lt_from_json(t);
  case 24: return Expr::gt_from_json(t);
  case 25: return Expr::alias_from_json(t);
  case 26: return Expr::invoke_from_json(t);
  case 27: return Expr::index_from_json(t);
  default: throw std::out_of_range("Bad ordinal " + std::to_string(ord));
  }
}

json Expr::any_to_json(const Expr::Any& x) { 
  return std::visit(overloaded{
  [](const Expr::Sin &y) -> json { return {0, Expr::sin_to_json(y)}; },
  [](const Expr::Cos &y) -> json { return {1, Expr::cos_to_json(y)}; },
  [](const Expr::Tan &y) -> json { return {2, Expr::tan_to_json(y)}; },
  [](const Expr::Abs &y) -> json { return {3, Expr::abs_to_json(y)}; },
  [](const Expr::Add &y) -> json { return {4, Expr::add_to_json(y)}; },
  [](const Expr::Sub &y) -> json { return {5, Expr::sub_to_json(y)}; },
  [](const Expr::Mul &y) -> json { return {6, Expr::mul_to_json(y)}; },
  [](const Expr::Div &y) -> json { return {7, Expr::div_to_json(y)}; },
  [](const Expr::Rem &y) -> json { return {8, Expr::rem_to_json(y)}; },
  [](const Expr::Pow &y) -> json { return {9, Expr::pow_to_json(y)}; },
  [](const Expr::BNot &y) -> json { return {10, Expr::bnot_to_json(y)}; },
  [](const Expr::BAnd &y) -> json { return {11, Expr::band_to_json(y)}; },
  [](const Expr::BOr &y) -> json { return {12, Expr::bor_to_json(y)}; },
  [](const Expr::BXor &y) -> json { return {13, Expr::bxor_to_json(y)}; },
  [](const Expr::BSL &y) -> json { return {14, Expr::bsl_to_json(y)}; },
  [](const Expr::BSR &y) -> json { return {15, Expr::bsr_to_json(y)}; },
  [](const Expr::Not &y) -> json { return {16, Expr::not_to_json(y)}; },
  [](const Expr::Eq &y) -> json { return {17, Expr::eq_to_json(y)}; },
  [](const Expr::Neq &y) -> json { return {18, Expr::neq_to_json(y)}; },
  [](const Expr::And &y) -> json { return {19, Expr::and_to_json(y)}; },
  [](const Expr::Or &y) -> json { return {20, Expr::or_to_json(y)}; },
  [](const Expr::Lte &y) -> json { return {21, Expr::lte_to_json(y)}; },
  [](const Expr::Gte &y) -> json { return {22, Expr::gte_to_json(y)}; },
  [](const Expr::Lt &y) -> json { return {23, Expr::lt_to_json(y)}; },
  [](const Expr::Gt &y) -> json { return {24, Expr::gt_to_json(y)}; },
  [](const Expr::Alias &y) -> json { return {25, Expr::alias_to_json(y)}; },
  [](const Expr::Invoke &y) -> json { return {26, Expr::invoke_to_json(y)}; },
  [](const Expr::Index &y) -> json { return {27, Expr::index_to_json(y)}; },
  [](const auto &x) -> json { throw std::out_of_range("Unimplemented type:" + to_string(x) ); }
  }, *x);
}

Stmt::Comment Stmt::comment_from_json(const json& j) { 
  auto value = j.at(0).get<std::string>();
  return Stmt::Comment(value);
}

json Stmt::comment_to_json(const Stmt::Comment& x) { 
  auto value = x.value;
  return json::array({value});
}

Stmt::Var Stmt::var_from_json(const json& j) { 
  auto name =  named_from_json(j.at(0));
  auto expr = j.at(1).is_null() ? std::nullopt : std::make_optional(Expr::any_from_json(j.at(1)));
  return {name, expr};
}

json Stmt::var_to_json(const Stmt::Var& x) { 
  auto name =  named_to_json(x.name);
  auto expr = x.expr ? Expr::any_to_json(*x.expr) : json{};
  return json::array({name, expr});
}

Stmt::Mut Stmt::mut_from_json(const json& j) { 
  auto name =  Term::select_from_json(j.at(0));
  auto expr =  Expr::any_from_json(j.at(1));
  return {name, expr};
}

json Stmt::mut_to_json(const Stmt::Mut& x) { 
  auto name =  Term::select_to_json(x.name);
  auto expr =  Expr::any_to_json(x.expr);
  return json::array({name, expr});
}

Stmt::Update Stmt::update_from_json(const json& j) { 
  auto lhs =  Term::select_from_json(j.at(0));
  auto idx =  Term::any_from_json(j.at(1));
  auto value =  Term::any_from_json(j.at(2));
  return {lhs, idx, value};
}

json Stmt::update_to_json(const Stmt::Update& x) { 
  auto lhs =  Term::select_to_json(x.lhs);
  auto idx =  Term::any_to_json(x.idx);
  auto value =  Term::any_to_json(x.value);
  return json::array({lhs, idx, value});
}

Stmt::While Stmt::while_from_json(const json& j) { 
  auto cond =  Expr::any_from_json(j.at(0));
  std::vector<Stmt::Any> body;
  auto body_json = j.at(1);
  std::transform(body_json.begin(), body_json.end(), std::back_inserter(body), &Stmt::any_from_json);
  return {cond, body};
}

json Stmt::while_to_json(const Stmt::While& x) { 
  auto cond =  Expr::any_to_json(x.cond);
  std::vector<json> body;
  std::transform(x.body.begin(), x.body.end(), std::back_inserter(body), &Stmt::any_to_json);
  return json::array({cond, body});
}

Stmt::Break Stmt::break_from_json(const json& j) { 
  return {};
}

json Stmt::break_to_json(const Stmt::Break& x) { 
  return json::array({});
}

Stmt::Cont Stmt::cont_from_json(const json& j) { 
  return {};
}

json Stmt::cont_to_json(const Stmt::Cont& x) { 
  return json::array({});
}

Stmt::Cond Stmt::cond_from_json(const json& j) { 
  auto cond =  Expr::any_from_json(j.at(0));
  std::vector<Stmt::Any> trueBr;
  auto trueBr_json = j.at(1);
  std::transform(trueBr_json.begin(), trueBr_json.end(), std::back_inserter(trueBr), &Stmt::any_from_json);
  std::vector<Stmt::Any> falseBr;
  auto falseBr_json = j.at(2);
  std::transform(falseBr_json.begin(), falseBr_json.end(), std::back_inserter(falseBr), &Stmt::any_from_json);
  return {cond, trueBr, falseBr};
}

json Stmt::cond_to_json(const Stmt::Cond& x) { 
  auto cond =  Expr::any_to_json(x.cond);
  std::vector<json> trueBr;
  std::transform(x.trueBr.begin(), x.trueBr.end(), std::back_inserter(trueBr), &Stmt::any_to_json);
  std::vector<json> falseBr;
  std::transform(x.falseBr.begin(), x.falseBr.end(), std::back_inserter(falseBr), &Stmt::any_to_json);
  return json::array({cond, trueBr, falseBr});
}

Stmt::Return Stmt::return_from_json(const json& j) { 
  auto value =  Expr::any_from_json(j.at(0));
  return Stmt::Return(value);
}

json Stmt::return_to_json(const Stmt::Return& x) { 
  auto value =  Expr::any_to_json(x.value);
  return json::array({value});
}

Stmt::Any Stmt::any_from_json(const json& j) { 
  size_t ord = j.at(0).get<size_t>();
  const auto t = j.at(1);
  switch (ord) {
  case 0: return Stmt::comment_from_json(t);
  case 1: return Stmt::var_from_json(t);
  case 2: return Stmt::mut_from_json(t);
  case 3: return Stmt::update_from_json(t);
  case 4: return Stmt::while_from_json(t);
  case 5: return Stmt::break_from_json(t);
  case 6: return Stmt::cont_from_json(t);
  case 7: return Stmt::cond_from_json(t);
  case 8: return Stmt::return_from_json(t);
  default: throw std::out_of_range("Bad ordinal " + std::to_string(ord));
  }
}

json Stmt::any_to_json(const Stmt::Any& x) { 
  return std::visit(overloaded{
  [](const Stmt::Comment &y) -> json { return {0, Stmt::comment_to_json(y)}; },
  [](const Stmt::Var &y) -> json { return {1, Stmt::var_to_json(y)}; },
  [](const Stmt::Mut &y) -> json { return {2, Stmt::mut_to_json(y)}; },
  [](const Stmt::Update &y) -> json { return {3, Stmt::update_to_json(y)}; },
  [](const Stmt::While &y) -> json { return {4, Stmt::while_to_json(y)}; },
  [](const Stmt::Break &y) -> json { return {5, Stmt::break_to_json(y)}; },
  [](const Stmt::Cont &y) -> json { return {6, Stmt::cont_to_json(y)}; },
  [](const Stmt::Cond &y) -> json { return {7, Stmt::cond_to_json(y)}; },
  [](const Stmt::Return &y) -> json { return {8, Stmt::return_to_json(y)}; },
  [](const auto &x) -> json { throw std::out_of_range("Unimplemented type:" + to_string(x) ); }
  }, *x);
}

StructDef structdef_from_json(const json& j) { 
  auto name =  sym_from_json(j.at(0));
  std::vector<Named> members;
  auto members_json = j.at(1);
  std::transform(members_json.begin(), members_json.end(), std::back_inserter(members), &named_from_json);
  return {name, members};
}

json structdef_to_json(const StructDef& x) { 
  auto name =  sym_to_json(x.name);
  std::vector<json> members;
  std::transform(x.members.begin(), x.members.end(), std::back_inserter(members), &named_to_json);
  return json::array({name, members});
}

Function function_from_json(const json& j) { 
  auto name = j.at(0).get<std::string>();
  std::vector<Named> args;
  auto args_json = j.at(1);
  std::transform(args_json.begin(), args_json.end(), std::back_inserter(args), &named_from_json);
  auto rtn =  Type::any_from_json(j.at(2));
  std::vector<Stmt::Any> body;
  auto body_json = j.at(3);
  std::transform(body_json.begin(), body_json.end(), std::back_inserter(body), &Stmt::any_from_json);
  return {name, args, rtn, body};
}

json function_to_json(const Function& x) { 
  auto name = x.name;
  std::vector<json> args;
  std::transform(x.args.begin(), x.args.end(), std::back_inserter(args), &named_to_json);
  auto rtn =  Type::any_to_json(x.rtn);
  std::vector<json> body;
  std::transform(x.body.begin(), x.body.end(), std::back_inserter(body), &Stmt::any_to_json);
  return json::array({name, args, rtn, body});
}

Program program_from_json(const json& j) { 
  auto entry =  function_from_json(j.at(0));
  std::vector<Function> functions;
  auto functions_json = j.at(1);
  std::transform(functions_json.begin(), functions_json.end(), std::back_inserter(functions), &function_from_json);
  std::vector<StructDef> defs;
  auto defs_json = j.at(2);
  std::transform(defs_json.begin(), defs_json.end(), std::back_inserter(defs), &structdef_from_json);
  return {entry, functions, defs};
}

json program_to_json(const Program& x) { 
  auto entry =  function_to_json(x.entry);
  std::vector<json> functions;
  std::transform(x.functions.begin(), x.functions.end(), std::back_inserter(functions), &function_to_json);
  std::vector<json> defs;
  std::transform(x.defs.begin(), x.defs.end(), std::back_inserter(defs), &structdef_to_json);
  return json::array({entry, functions, defs});
}
json hashed_from_json(const json& j) { 
  auto hash = j.at(0).get<std::string>();
  auto data = j.at(1);
  if(hash != "02bd4aae1ce661d6c7d9c7438185313c") {
   throw std::runtime_error("Expecting ADT hash to be 02bd4aae1ce661d6c7d9c7438185313c, but was " + hash);
  }
  return data;
}

json hashed_to_json(const json& x) { 
  return json::array({"02bd4aae1ce661d6c7d9c7438185313c", x});
}
} // namespace polyregion::polyast
