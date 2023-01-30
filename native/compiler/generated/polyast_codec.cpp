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

TypeKind::None TypeKind::none_from_json(const json& j) { 
  return {};
}

json TypeKind::none_to_json(const TypeKind::None& x) { 
  return json::array({});
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
  case 0: return TypeKind::none_from_json(t);
  case 1: return TypeKind::ref_from_json(t);
  case 2: return TypeKind::integral_from_json(t);
  case 3: return TypeKind::fractional_from_json(t);
  default: throw std::out_of_range("Bad ordinal " + std::to_string(ord));
  }
}

json TypeKind::any_to_json(const TypeKind::Any& x) { 
  return std::visit(overloaded{
  [](const TypeKind::None &y) -> json { return {0, TypeKind::none_to_json(y)}; },
  [](const TypeKind::Ref &y) -> json { return {1, TypeKind::ref_to_json(y)}; },
  [](const TypeKind::Integral &y) -> json { return {2, TypeKind::integral_to_json(y)}; },
  [](const TypeKind::Fractional &y) -> json { return {3, TypeKind::fractional_to_json(y)}; },
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

Type::Unit Type::unit_from_json(const json& j) { 
  return {};
}

json Type::unit_to_json(const Type::Unit& x) { 
  return json::array({});
}

Type::Nothing Type::nothing_from_json(const json& j) { 
  return {};
}

json Type::nothing_to_json(const Type::Nothing& x) { 
  return json::array({});
}

Type::Struct Type::struct_from_json(const json& j) { 
  auto name =  sym_from_json(j.at(0));
  auto tpeVars = j.at(1).get<std::vector<std::string>>();
  std::vector<Type::Any> args;
  auto args_json = j.at(2);
  std::transform(args_json.begin(), args_json.end(), std::back_inserter(args), &Type::any_from_json);
  std::vector<Sym> parents;
  auto parents_json = j.at(3);
  std::transform(parents_json.begin(), parents_json.end(), std::back_inserter(parents), &sym_from_json);
  return {name, tpeVars, args, parents};
}

json Type::struct_to_json(const Type::Struct& x) { 
  auto name =  sym_to_json(x.name);
  auto tpeVars = x.tpeVars;
  std::vector<json> args;
  std::transform(x.args.begin(), x.args.end(), std::back_inserter(args), &Type::any_to_json);
  std::vector<json> parents;
  std::transform(x.parents.begin(), x.parents.end(), std::back_inserter(parents), &sym_to_json);
  return json::array({name, tpeVars, args, parents});
}

Type::Array Type::array_from_json(const json& j) { 
  auto component =  Type::any_from_json(j.at(0));
  return Type::Array(component);
}

json Type::array_to_json(const Type::Array& x) { 
  auto component =  Type::any_to_json(x.component);
  return json::array({component});
}

Type::Var Type::var_from_json(const json& j) { 
  auto name = j.at(0).get<std::string>();
  return Type::Var(name);
}

json Type::var_to_json(const Type::Var& x) { 
  auto name = x.name;
  return json::array({name});
}

Type::Exec Type::exec_from_json(const json& j) { 
  auto tpeVars = j.at(0).get<std::vector<std::string>>();
  std::vector<Type::Any> args;
  auto args_json = j.at(1);
  std::transform(args_json.begin(), args_json.end(), std::back_inserter(args), &Type::any_from_json);
  auto rtn =  Type::any_from_json(j.at(2));
  return {tpeVars, args, rtn};
}

json Type::exec_to_json(const Type::Exec& x) { 
  auto tpeVars = x.tpeVars;
  std::vector<json> args;
  std::transform(x.args.begin(), x.args.end(), std::back_inserter(args), &Type::any_to_json);
  auto rtn =  Type::any_to_json(x.rtn);
  return json::array({tpeVars, args, rtn});
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
  case 8: return Type::unit_from_json(t);
  case 9: return Type::nothing_from_json(t);
  case 10: return Type::struct_from_json(t);
  case 11: return Type::array_from_json(t);
  case 12: return Type::var_from_json(t);
  case 13: return Type::exec_from_json(t);
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
  [](const Type::Unit &y) -> json { return {8, Type::unit_to_json(y)}; },
  [](const Type::Nothing &y) -> json { return {9, Type::nothing_to_json(y)}; },
  [](const Type::Struct &y) -> json { return {10, Type::struct_to_json(y)}; },
  [](const Type::Array &y) -> json { return {11, Type::array_to_json(y)}; },
  [](const Type::Var &y) -> json { return {12, Type::var_to_json(y)}; },
  [](const Type::Exec &y) -> json { return {13, Type::exec_to_json(y)}; },
  [](const auto &x) -> json { throw std::out_of_range("Unimplemented type:" + to_string(x) ); }
  }, *x);
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

Term::Poison Term::poison_from_json(const json& j) { 
  auto t =  Type::any_from_json(j.at(0));
  return Term::Poison(t);
}

json Term::poison_to_json(const Term::Poison& x) { 
  auto t =  Type::any_to_json(x.t);
  return json::array({t});
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

Term::Any Term::any_from_json(const json& j) { 
  size_t ord = j.at(0).get<size_t>();
  const auto t = j.at(1);
  switch (ord) {
  case 0: return Term::select_from_json(t);
  case 1: return Term::poison_from_json(t);
  case 2: return Term::unitconst_from_json(t);
  case 3: return Term::boolconst_from_json(t);
  case 4: return Term::byteconst_from_json(t);
  case 5: return Term::charconst_from_json(t);
  case 6: return Term::shortconst_from_json(t);
  case 7: return Term::intconst_from_json(t);
  case 8: return Term::longconst_from_json(t);
  case 9: return Term::floatconst_from_json(t);
  case 10: return Term::doubleconst_from_json(t);
  default: throw std::out_of_range("Bad ordinal " + std::to_string(ord));
  }
}

json Term::any_to_json(const Term::Any& x) { 
  return std::visit(overloaded{
  [](const Term::Select &y) -> json { return {0, Term::select_to_json(y)}; },
  [](const Term::Poison &y) -> json { return {1, Term::poison_to_json(y)}; },
  [](const Term::UnitConst &y) -> json { return {2, Term::unitconst_to_json(y)}; },
  [](const Term::BoolConst &y) -> json { return {3, Term::boolconst_to_json(y)}; },
  [](const Term::ByteConst &y) -> json { return {4, Term::byteconst_to_json(y)}; },
  [](const Term::CharConst &y) -> json { return {5, Term::charconst_to_json(y)}; },
  [](const Term::ShortConst &y) -> json { return {6, Term::shortconst_to_json(y)}; },
  [](const Term::IntConst &y) -> json { return {7, Term::intconst_to_json(y)}; },
  [](const Term::LongConst &y) -> json { return {8, Term::longconst_to_json(y)}; },
  [](const Term::FloatConst &y) -> json { return {9, Term::floatconst_to_json(y)}; },
  [](const Term::DoubleConst &y) -> json { return {10, Term::doubleconst_to_json(y)}; },
  [](const auto &x) -> json { throw std::out_of_range("Unimplemented type:" + to_string(x) ); }
  }, *x);
}

NullaryIntrinsicKind::GpuGlobalIdxX NullaryIntrinsicKind::gpuglobalidxx_from_json(const json& j) { 
  return {};
}

json NullaryIntrinsicKind::gpuglobalidxx_to_json(const NullaryIntrinsicKind::GpuGlobalIdxX& x) { 
  return json::array({});
}

NullaryIntrinsicKind::GpuGlobalIdxY NullaryIntrinsicKind::gpuglobalidxy_from_json(const json& j) { 
  return {};
}

json NullaryIntrinsicKind::gpuglobalidxy_to_json(const NullaryIntrinsicKind::GpuGlobalIdxY& x) { 
  return json::array({});
}

NullaryIntrinsicKind::GpuGlobalIdxZ NullaryIntrinsicKind::gpuglobalidxz_from_json(const json& j) { 
  return {};
}

json NullaryIntrinsicKind::gpuglobalidxz_to_json(const NullaryIntrinsicKind::GpuGlobalIdxZ& x) { 
  return json::array({});
}

NullaryIntrinsicKind::GpuGlobalSizeX NullaryIntrinsicKind::gpuglobalsizex_from_json(const json& j) { 
  return {};
}

json NullaryIntrinsicKind::gpuglobalsizex_to_json(const NullaryIntrinsicKind::GpuGlobalSizeX& x) { 
  return json::array({});
}

NullaryIntrinsicKind::GpuGlobalSizeY NullaryIntrinsicKind::gpuglobalsizey_from_json(const json& j) { 
  return {};
}

json NullaryIntrinsicKind::gpuglobalsizey_to_json(const NullaryIntrinsicKind::GpuGlobalSizeY& x) { 
  return json::array({});
}

NullaryIntrinsicKind::GpuGlobalSizeZ NullaryIntrinsicKind::gpuglobalsizez_from_json(const json& j) { 
  return {};
}

json NullaryIntrinsicKind::gpuglobalsizez_to_json(const NullaryIntrinsicKind::GpuGlobalSizeZ& x) { 
  return json::array({});
}

NullaryIntrinsicKind::GpuGroupIdxX NullaryIntrinsicKind::gpugroupidxx_from_json(const json& j) { 
  return {};
}

json NullaryIntrinsicKind::gpugroupidxx_to_json(const NullaryIntrinsicKind::GpuGroupIdxX& x) { 
  return json::array({});
}

NullaryIntrinsicKind::GpuGroupIdxY NullaryIntrinsicKind::gpugroupidxy_from_json(const json& j) { 
  return {};
}

json NullaryIntrinsicKind::gpugroupidxy_to_json(const NullaryIntrinsicKind::GpuGroupIdxY& x) { 
  return json::array({});
}

NullaryIntrinsicKind::GpuGroupIdxZ NullaryIntrinsicKind::gpugroupidxz_from_json(const json& j) { 
  return {};
}

json NullaryIntrinsicKind::gpugroupidxz_to_json(const NullaryIntrinsicKind::GpuGroupIdxZ& x) { 
  return json::array({});
}

NullaryIntrinsicKind::GpuGroupSizeX NullaryIntrinsicKind::gpugroupsizex_from_json(const json& j) { 
  return {};
}

json NullaryIntrinsicKind::gpugroupsizex_to_json(const NullaryIntrinsicKind::GpuGroupSizeX& x) { 
  return json::array({});
}

NullaryIntrinsicKind::GpuGroupSizeY NullaryIntrinsicKind::gpugroupsizey_from_json(const json& j) { 
  return {};
}

json NullaryIntrinsicKind::gpugroupsizey_to_json(const NullaryIntrinsicKind::GpuGroupSizeY& x) { 
  return json::array({});
}

NullaryIntrinsicKind::GpuGroupSizeZ NullaryIntrinsicKind::gpugroupsizez_from_json(const json& j) { 
  return {};
}

json NullaryIntrinsicKind::gpugroupsizez_to_json(const NullaryIntrinsicKind::GpuGroupSizeZ& x) { 
  return json::array({});
}

NullaryIntrinsicKind::GpuLocalIdxX NullaryIntrinsicKind::gpulocalidxx_from_json(const json& j) { 
  return {};
}

json NullaryIntrinsicKind::gpulocalidxx_to_json(const NullaryIntrinsicKind::GpuLocalIdxX& x) { 
  return json::array({});
}

NullaryIntrinsicKind::GpuLocalIdxY NullaryIntrinsicKind::gpulocalidxy_from_json(const json& j) { 
  return {};
}

json NullaryIntrinsicKind::gpulocalidxy_to_json(const NullaryIntrinsicKind::GpuLocalIdxY& x) { 
  return json::array({});
}

NullaryIntrinsicKind::GpuLocalIdxZ NullaryIntrinsicKind::gpulocalidxz_from_json(const json& j) { 
  return {};
}

json NullaryIntrinsicKind::gpulocalidxz_to_json(const NullaryIntrinsicKind::GpuLocalIdxZ& x) { 
  return json::array({});
}

NullaryIntrinsicKind::GpuLocalSizeX NullaryIntrinsicKind::gpulocalsizex_from_json(const json& j) { 
  return {};
}

json NullaryIntrinsicKind::gpulocalsizex_to_json(const NullaryIntrinsicKind::GpuLocalSizeX& x) { 
  return json::array({});
}

NullaryIntrinsicKind::GpuLocalSizeY NullaryIntrinsicKind::gpulocalsizey_from_json(const json& j) { 
  return {};
}

json NullaryIntrinsicKind::gpulocalsizey_to_json(const NullaryIntrinsicKind::GpuLocalSizeY& x) { 
  return json::array({});
}

NullaryIntrinsicKind::GpuLocalSizeZ NullaryIntrinsicKind::gpulocalsizez_from_json(const json& j) { 
  return {};
}

json NullaryIntrinsicKind::gpulocalsizez_to_json(const NullaryIntrinsicKind::GpuLocalSizeZ& x) { 
  return json::array({});
}

NullaryIntrinsicKind::GpuGroupBarrier NullaryIntrinsicKind::gpugroupbarrier_from_json(const json& j) { 
  return {};
}

json NullaryIntrinsicKind::gpugroupbarrier_to_json(const NullaryIntrinsicKind::GpuGroupBarrier& x) { 
  return json::array({});
}

NullaryIntrinsicKind::GpuGroupFence NullaryIntrinsicKind::gpugroupfence_from_json(const json& j) { 
  return {};
}

json NullaryIntrinsicKind::gpugroupfence_to_json(const NullaryIntrinsicKind::GpuGroupFence& x) { 
  return json::array({});
}

NullaryIntrinsicKind::Any NullaryIntrinsicKind::any_from_json(const json& j) { 
  size_t ord = j.at(0).get<size_t>();
  const auto t = j.at(1);
  switch (ord) {
  case 0: return NullaryIntrinsicKind::gpuglobalidxx_from_json(t);
  case 1: return NullaryIntrinsicKind::gpuglobalidxy_from_json(t);
  case 2: return NullaryIntrinsicKind::gpuglobalidxz_from_json(t);
  case 3: return NullaryIntrinsicKind::gpuglobalsizex_from_json(t);
  case 4: return NullaryIntrinsicKind::gpuglobalsizey_from_json(t);
  case 5: return NullaryIntrinsicKind::gpuglobalsizez_from_json(t);
  case 6: return NullaryIntrinsicKind::gpugroupidxx_from_json(t);
  case 7: return NullaryIntrinsicKind::gpugroupidxy_from_json(t);
  case 8: return NullaryIntrinsicKind::gpugroupidxz_from_json(t);
  case 9: return NullaryIntrinsicKind::gpugroupsizex_from_json(t);
  case 10: return NullaryIntrinsicKind::gpugroupsizey_from_json(t);
  case 11: return NullaryIntrinsicKind::gpugroupsizez_from_json(t);
  case 12: return NullaryIntrinsicKind::gpulocalidxx_from_json(t);
  case 13: return NullaryIntrinsicKind::gpulocalidxy_from_json(t);
  case 14: return NullaryIntrinsicKind::gpulocalidxz_from_json(t);
  case 15: return NullaryIntrinsicKind::gpulocalsizex_from_json(t);
  case 16: return NullaryIntrinsicKind::gpulocalsizey_from_json(t);
  case 17: return NullaryIntrinsicKind::gpulocalsizez_from_json(t);
  case 18: return NullaryIntrinsicKind::gpugroupbarrier_from_json(t);
  case 19: return NullaryIntrinsicKind::gpugroupfence_from_json(t);
  default: throw std::out_of_range("Bad ordinal " + std::to_string(ord));
  }
}

json NullaryIntrinsicKind::any_to_json(const NullaryIntrinsicKind::Any& x) { 
  return std::visit(overloaded{
  [](const NullaryIntrinsicKind::GpuGlobalIdxX &y) -> json { return {0, NullaryIntrinsicKind::gpuglobalidxx_to_json(y)}; },
  [](const NullaryIntrinsicKind::GpuGlobalIdxY &y) -> json { return {1, NullaryIntrinsicKind::gpuglobalidxy_to_json(y)}; },
  [](const NullaryIntrinsicKind::GpuGlobalIdxZ &y) -> json { return {2, NullaryIntrinsicKind::gpuglobalidxz_to_json(y)}; },
  [](const NullaryIntrinsicKind::GpuGlobalSizeX &y) -> json { return {3, NullaryIntrinsicKind::gpuglobalsizex_to_json(y)}; },
  [](const NullaryIntrinsicKind::GpuGlobalSizeY &y) -> json { return {4, NullaryIntrinsicKind::gpuglobalsizey_to_json(y)}; },
  [](const NullaryIntrinsicKind::GpuGlobalSizeZ &y) -> json { return {5, NullaryIntrinsicKind::gpuglobalsizez_to_json(y)}; },
  [](const NullaryIntrinsicKind::GpuGroupIdxX &y) -> json { return {6, NullaryIntrinsicKind::gpugroupidxx_to_json(y)}; },
  [](const NullaryIntrinsicKind::GpuGroupIdxY &y) -> json { return {7, NullaryIntrinsicKind::gpugroupidxy_to_json(y)}; },
  [](const NullaryIntrinsicKind::GpuGroupIdxZ &y) -> json { return {8, NullaryIntrinsicKind::gpugroupidxz_to_json(y)}; },
  [](const NullaryIntrinsicKind::GpuGroupSizeX &y) -> json { return {9, NullaryIntrinsicKind::gpugroupsizex_to_json(y)}; },
  [](const NullaryIntrinsicKind::GpuGroupSizeY &y) -> json { return {10, NullaryIntrinsicKind::gpugroupsizey_to_json(y)}; },
  [](const NullaryIntrinsicKind::GpuGroupSizeZ &y) -> json { return {11, NullaryIntrinsicKind::gpugroupsizez_to_json(y)}; },
  [](const NullaryIntrinsicKind::GpuLocalIdxX &y) -> json { return {12, NullaryIntrinsicKind::gpulocalidxx_to_json(y)}; },
  [](const NullaryIntrinsicKind::GpuLocalIdxY &y) -> json { return {13, NullaryIntrinsicKind::gpulocalidxy_to_json(y)}; },
  [](const NullaryIntrinsicKind::GpuLocalIdxZ &y) -> json { return {14, NullaryIntrinsicKind::gpulocalidxz_to_json(y)}; },
  [](const NullaryIntrinsicKind::GpuLocalSizeX &y) -> json { return {15, NullaryIntrinsicKind::gpulocalsizex_to_json(y)}; },
  [](const NullaryIntrinsicKind::GpuLocalSizeY &y) -> json { return {16, NullaryIntrinsicKind::gpulocalsizey_to_json(y)}; },
  [](const NullaryIntrinsicKind::GpuLocalSizeZ &y) -> json { return {17, NullaryIntrinsicKind::gpulocalsizez_to_json(y)}; },
  [](const NullaryIntrinsicKind::GpuGroupBarrier &y) -> json { return {18, NullaryIntrinsicKind::gpugroupbarrier_to_json(y)}; },
  [](const NullaryIntrinsicKind::GpuGroupFence &y) -> json { return {19, NullaryIntrinsicKind::gpugroupfence_to_json(y)}; },
  [](const auto &x) -> json { throw std::out_of_range("Unimplemented type:" + to_string(x) ); }
  }, *x);
}

UnaryIntrinsicKind::Sin UnaryIntrinsicKind::sin_from_json(const json& j) { 
  return {};
}

json UnaryIntrinsicKind::sin_to_json(const UnaryIntrinsicKind::Sin& x) { 
  return json::array({});
}

UnaryIntrinsicKind::Cos UnaryIntrinsicKind::cos_from_json(const json& j) { 
  return {};
}

json UnaryIntrinsicKind::cos_to_json(const UnaryIntrinsicKind::Cos& x) { 
  return json::array({});
}

UnaryIntrinsicKind::Tan UnaryIntrinsicKind::tan_from_json(const json& j) { 
  return {};
}

json UnaryIntrinsicKind::tan_to_json(const UnaryIntrinsicKind::Tan& x) { 
  return json::array({});
}

UnaryIntrinsicKind::Asin UnaryIntrinsicKind::asin_from_json(const json& j) { 
  return {};
}

json UnaryIntrinsicKind::asin_to_json(const UnaryIntrinsicKind::Asin& x) { 
  return json::array({});
}

UnaryIntrinsicKind::Acos UnaryIntrinsicKind::acos_from_json(const json& j) { 
  return {};
}

json UnaryIntrinsicKind::acos_to_json(const UnaryIntrinsicKind::Acos& x) { 
  return json::array({});
}

UnaryIntrinsicKind::Atan UnaryIntrinsicKind::atan_from_json(const json& j) { 
  return {};
}

json UnaryIntrinsicKind::atan_to_json(const UnaryIntrinsicKind::Atan& x) { 
  return json::array({});
}

UnaryIntrinsicKind::Sinh UnaryIntrinsicKind::sinh_from_json(const json& j) { 
  return {};
}

json UnaryIntrinsicKind::sinh_to_json(const UnaryIntrinsicKind::Sinh& x) { 
  return json::array({});
}

UnaryIntrinsicKind::Cosh UnaryIntrinsicKind::cosh_from_json(const json& j) { 
  return {};
}

json UnaryIntrinsicKind::cosh_to_json(const UnaryIntrinsicKind::Cosh& x) { 
  return json::array({});
}

UnaryIntrinsicKind::Tanh UnaryIntrinsicKind::tanh_from_json(const json& j) { 
  return {};
}

json UnaryIntrinsicKind::tanh_to_json(const UnaryIntrinsicKind::Tanh& x) { 
  return json::array({});
}

UnaryIntrinsicKind::Signum UnaryIntrinsicKind::signum_from_json(const json& j) { 
  return {};
}

json UnaryIntrinsicKind::signum_to_json(const UnaryIntrinsicKind::Signum& x) { 
  return json::array({});
}

UnaryIntrinsicKind::Abs UnaryIntrinsicKind::abs_from_json(const json& j) { 
  return {};
}

json UnaryIntrinsicKind::abs_to_json(const UnaryIntrinsicKind::Abs& x) { 
  return json::array({});
}

UnaryIntrinsicKind::Round UnaryIntrinsicKind::round_from_json(const json& j) { 
  return {};
}

json UnaryIntrinsicKind::round_to_json(const UnaryIntrinsicKind::Round& x) { 
  return json::array({});
}

UnaryIntrinsicKind::Ceil UnaryIntrinsicKind::ceil_from_json(const json& j) { 
  return {};
}

json UnaryIntrinsicKind::ceil_to_json(const UnaryIntrinsicKind::Ceil& x) { 
  return json::array({});
}

UnaryIntrinsicKind::Floor UnaryIntrinsicKind::floor_from_json(const json& j) { 
  return {};
}

json UnaryIntrinsicKind::floor_to_json(const UnaryIntrinsicKind::Floor& x) { 
  return json::array({});
}

UnaryIntrinsicKind::Rint UnaryIntrinsicKind::rint_from_json(const json& j) { 
  return {};
}

json UnaryIntrinsicKind::rint_to_json(const UnaryIntrinsicKind::Rint& x) { 
  return json::array({});
}

UnaryIntrinsicKind::Sqrt UnaryIntrinsicKind::sqrt_from_json(const json& j) { 
  return {};
}

json UnaryIntrinsicKind::sqrt_to_json(const UnaryIntrinsicKind::Sqrt& x) { 
  return json::array({});
}

UnaryIntrinsicKind::Cbrt UnaryIntrinsicKind::cbrt_from_json(const json& j) { 
  return {};
}

json UnaryIntrinsicKind::cbrt_to_json(const UnaryIntrinsicKind::Cbrt& x) { 
  return json::array({});
}

UnaryIntrinsicKind::Exp UnaryIntrinsicKind::exp_from_json(const json& j) { 
  return {};
}

json UnaryIntrinsicKind::exp_to_json(const UnaryIntrinsicKind::Exp& x) { 
  return json::array({});
}

UnaryIntrinsicKind::Expm1 UnaryIntrinsicKind::expm1_from_json(const json& j) { 
  return {};
}

json UnaryIntrinsicKind::expm1_to_json(const UnaryIntrinsicKind::Expm1& x) { 
  return json::array({});
}

UnaryIntrinsicKind::Log UnaryIntrinsicKind::log_from_json(const json& j) { 
  return {};
}

json UnaryIntrinsicKind::log_to_json(const UnaryIntrinsicKind::Log& x) { 
  return json::array({});
}

UnaryIntrinsicKind::Log1p UnaryIntrinsicKind::log1p_from_json(const json& j) { 
  return {};
}

json UnaryIntrinsicKind::log1p_to_json(const UnaryIntrinsicKind::Log1p& x) { 
  return json::array({});
}

UnaryIntrinsicKind::Log10 UnaryIntrinsicKind::log10_from_json(const json& j) { 
  return {};
}

json UnaryIntrinsicKind::log10_to_json(const UnaryIntrinsicKind::Log10& x) { 
  return json::array({});
}

UnaryIntrinsicKind::BNot UnaryIntrinsicKind::bnot_from_json(const json& j) { 
  return {};
}

json UnaryIntrinsicKind::bnot_to_json(const UnaryIntrinsicKind::BNot& x) { 
  return json::array({});
}

UnaryIntrinsicKind::Pos UnaryIntrinsicKind::pos_from_json(const json& j) { 
  return {};
}

json UnaryIntrinsicKind::pos_to_json(const UnaryIntrinsicKind::Pos& x) { 
  return json::array({});
}

UnaryIntrinsicKind::Neg UnaryIntrinsicKind::neg_from_json(const json& j) { 
  return {};
}

json UnaryIntrinsicKind::neg_to_json(const UnaryIntrinsicKind::Neg& x) { 
  return json::array({});
}

UnaryIntrinsicKind::LogicNot UnaryIntrinsicKind::logicnot_from_json(const json& j) { 
  return {};
}

json UnaryIntrinsicKind::logicnot_to_json(const UnaryIntrinsicKind::LogicNot& x) { 
  return json::array({});
}

UnaryIntrinsicKind::Any UnaryIntrinsicKind::any_from_json(const json& j) { 
  size_t ord = j.at(0).get<size_t>();
  const auto t = j.at(1);
  switch (ord) {
  case 0: return UnaryIntrinsicKind::sin_from_json(t);
  case 1: return UnaryIntrinsicKind::cos_from_json(t);
  case 2: return UnaryIntrinsicKind::tan_from_json(t);
  case 3: return UnaryIntrinsicKind::asin_from_json(t);
  case 4: return UnaryIntrinsicKind::acos_from_json(t);
  case 5: return UnaryIntrinsicKind::atan_from_json(t);
  case 6: return UnaryIntrinsicKind::sinh_from_json(t);
  case 7: return UnaryIntrinsicKind::cosh_from_json(t);
  case 8: return UnaryIntrinsicKind::tanh_from_json(t);
  case 9: return UnaryIntrinsicKind::signum_from_json(t);
  case 10: return UnaryIntrinsicKind::abs_from_json(t);
  case 11: return UnaryIntrinsicKind::round_from_json(t);
  case 12: return UnaryIntrinsicKind::ceil_from_json(t);
  case 13: return UnaryIntrinsicKind::floor_from_json(t);
  case 14: return UnaryIntrinsicKind::rint_from_json(t);
  case 15: return UnaryIntrinsicKind::sqrt_from_json(t);
  case 16: return UnaryIntrinsicKind::cbrt_from_json(t);
  case 17: return UnaryIntrinsicKind::exp_from_json(t);
  case 18: return UnaryIntrinsicKind::expm1_from_json(t);
  case 19: return UnaryIntrinsicKind::log_from_json(t);
  case 20: return UnaryIntrinsicKind::log1p_from_json(t);
  case 21: return UnaryIntrinsicKind::log10_from_json(t);
  case 22: return UnaryIntrinsicKind::bnot_from_json(t);
  case 23: return UnaryIntrinsicKind::pos_from_json(t);
  case 24: return UnaryIntrinsicKind::neg_from_json(t);
  case 25: return UnaryIntrinsicKind::logicnot_from_json(t);
  default: throw std::out_of_range("Bad ordinal " + std::to_string(ord));
  }
}

json UnaryIntrinsicKind::any_to_json(const UnaryIntrinsicKind::Any& x) { 
  return std::visit(overloaded{
  [](const UnaryIntrinsicKind::Sin &y) -> json { return {0, UnaryIntrinsicKind::sin_to_json(y)}; },
  [](const UnaryIntrinsicKind::Cos &y) -> json { return {1, UnaryIntrinsicKind::cos_to_json(y)}; },
  [](const UnaryIntrinsicKind::Tan &y) -> json { return {2, UnaryIntrinsicKind::tan_to_json(y)}; },
  [](const UnaryIntrinsicKind::Asin &y) -> json { return {3, UnaryIntrinsicKind::asin_to_json(y)}; },
  [](const UnaryIntrinsicKind::Acos &y) -> json { return {4, UnaryIntrinsicKind::acos_to_json(y)}; },
  [](const UnaryIntrinsicKind::Atan &y) -> json { return {5, UnaryIntrinsicKind::atan_to_json(y)}; },
  [](const UnaryIntrinsicKind::Sinh &y) -> json { return {6, UnaryIntrinsicKind::sinh_to_json(y)}; },
  [](const UnaryIntrinsicKind::Cosh &y) -> json { return {7, UnaryIntrinsicKind::cosh_to_json(y)}; },
  [](const UnaryIntrinsicKind::Tanh &y) -> json { return {8, UnaryIntrinsicKind::tanh_to_json(y)}; },
  [](const UnaryIntrinsicKind::Signum &y) -> json { return {9, UnaryIntrinsicKind::signum_to_json(y)}; },
  [](const UnaryIntrinsicKind::Abs &y) -> json { return {10, UnaryIntrinsicKind::abs_to_json(y)}; },
  [](const UnaryIntrinsicKind::Round &y) -> json { return {11, UnaryIntrinsicKind::round_to_json(y)}; },
  [](const UnaryIntrinsicKind::Ceil &y) -> json { return {12, UnaryIntrinsicKind::ceil_to_json(y)}; },
  [](const UnaryIntrinsicKind::Floor &y) -> json { return {13, UnaryIntrinsicKind::floor_to_json(y)}; },
  [](const UnaryIntrinsicKind::Rint &y) -> json { return {14, UnaryIntrinsicKind::rint_to_json(y)}; },
  [](const UnaryIntrinsicKind::Sqrt &y) -> json { return {15, UnaryIntrinsicKind::sqrt_to_json(y)}; },
  [](const UnaryIntrinsicKind::Cbrt &y) -> json { return {16, UnaryIntrinsicKind::cbrt_to_json(y)}; },
  [](const UnaryIntrinsicKind::Exp &y) -> json { return {17, UnaryIntrinsicKind::exp_to_json(y)}; },
  [](const UnaryIntrinsicKind::Expm1 &y) -> json { return {18, UnaryIntrinsicKind::expm1_to_json(y)}; },
  [](const UnaryIntrinsicKind::Log &y) -> json { return {19, UnaryIntrinsicKind::log_to_json(y)}; },
  [](const UnaryIntrinsicKind::Log1p &y) -> json { return {20, UnaryIntrinsicKind::log1p_to_json(y)}; },
  [](const UnaryIntrinsicKind::Log10 &y) -> json { return {21, UnaryIntrinsicKind::log10_to_json(y)}; },
  [](const UnaryIntrinsicKind::BNot &y) -> json { return {22, UnaryIntrinsicKind::bnot_to_json(y)}; },
  [](const UnaryIntrinsicKind::Pos &y) -> json { return {23, UnaryIntrinsicKind::pos_to_json(y)}; },
  [](const UnaryIntrinsicKind::Neg &y) -> json { return {24, UnaryIntrinsicKind::neg_to_json(y)}; },
  [](const UnaryIntrinsicKind::LogicNot &y) -> json { return {25, UnaryIntrinsicKind::logicnot_to_json(y)}; },
  [](const auto &x) -> json { throw std::out_of_range("Unimplemented type:" + to_string(x) ); }
  }, *x);
}

BinaryIntrinsicKind::Add BinaryIntrinsicKind::add_from_json(const json& j) { 
  return {};
}

json BinaryIntrinsicKind::add_to_json(const BinaryIntrinsicKind::Add& x) { 
  return json::array({});
}

BinaryIntrinsicKind::Sub BinaryIntrinsicKind::sub_from_json(const json& j) { 
  return {};
}

json BinaryIntrinsicKind::sub_to_json(const BinaryIntrinsicKind::Sub& x) { 
  return json::array({});
}

BinaryIntrinsicKind::Mul BinaryIntrinsicKind::mul_from_json(const json& j) { 
  return {};
}

json BinaryIntrinsicKind::mul_to_json(const BinaryIntrinsicKind::Mul& x) { 
  return json::array({});
}

BinaryIntrinsicKind::Div BinaryIntrinsicKind::div_from_json(const json& j) { 
  return {};
}

json BinaryIntrinsicKind::div_to_json(const BinaryIntrinsicKind::Div& x) { 
  return json::array({});
}

BinaryIntrinsicKind::Rem BinaryIntrinsicKind::rem_from_json(const json& j) { 
  return {};
}

json BinaryIntrinsicKind::rem_to_json(const BinaryIntrinsicKind::Rem& x) { 
  return json::array({});
}

BinaryIntrinsicKind::Pow BinaryIntrinsicKind::pow_from_json(const json& j) { 
  return {};
}

json BinaryIntrinsicKind::pow_to_json(const BinaryIntrinsicKind::Pow& x) { 
  return json::array({});
}

BinaryIntrinsicKind::Min BinaryIntrinsicKind::min_from_json(const json& j) { 
  return {};
}

json BinaryIntrinsicKind::min_to_json(const BinaryIntrinsicKind::Min& x) { 
  return json::array({});
}

BinaryIntrinsicKind::Max BinaryIntrinsicKind::max_from_json(const json& j) { 
  return {};
}

json BinaryIntrinsicKind::max_to_json(const BinaryIntrinsicKind::Max& x) { 
  return json::array({});
}

BinaryIntrinsicKind::Atan2 BinaryIntrinsicKind::atan2_from_json(const json& j) { 
  return {};
}

json BinaryIntrinsicKind::atan2_to_json(const BinaryIntrinsicKind::Atan2& x) { 
  return json::array({});
}

BinaryIntrinsicKind::Hypot BinaryIntrinsicKind::hypot_from_json(const json& j) { 
  return {};
}

json BinaryIntrinsicKind::hypot_to_json(const BinaryIntrinsicKind::Hypot& x) { 
  return json::array({});
}

BinaryIntrinsicKind::BAnd BinaryIntrinsicKind::band_from_json(const json& j) { 
  return {};
}

json BinaryIntrinsicKind::band_to_json(const BinaryIntrinsicKind::BAnd& x) { 
  return json::array({});
}

BinaryIntrinsicKind::BOr BinaryIntrinsicKind::bor_from_json(const json& j) { 
  return {};
}

json BinaryIntrinsicKind::bor_to_json(const BinaryIntrinsicKind::BOr& x) { 
  return json::array({});
}

BinaryIntrinsicKind::BXor BinaryIntrinsicKind::bxor_from_json(const json& j) { 
  return {};
}

json BinaryIntrinsicKind::bxor_to_json(const BinaryIntrinsicKind::BXor& x) { 
  return json::array({});
}

BinaryIntrinsicKind::BSL BinaryIntrinsicKind::bsl_from_json(const json& j) { 
  return {};
}

json BinaryIntrinsicKind::bsl_to_json(const BinaryIntrinsicKind::BSL& x) { 
  return json::array({});
}

BinaryIntrinsicKind::BSR BinaryIntrinsicKind::bsr_from_json(const json& j) { 
  return {};
}

json BinaryIntrinsicKind::bsr_to_json(const BinaryIntrinsicKind::BSR& x) { 
  return json::array({});
}

BinaryIntrinsicKind::BZSR BinaryIntrinsicKind::bzsr_from_json(const json& j) { 
  return {};
}

json BinaryIntrinsicKind::bzsr_to_json(const BinaryIntrinsicKind::BZSR& x) { 
  return json::array({});
}

BinaryIntrinsicKind::LogicEq BinaryIntrinsicKind::logiceq_from_json(const json& j) { 
  return {};
}

json BinaryIntrinsicKind::logiceq_to_json(const BinaryIntrinsicKind::LogicEq& x) { 
  return json::array({});
}

BinaryIntrinsicKind::LogicNeq BinaryIntrinsicKind::logicneq_from_json(const json& j) { 
  return {};
}

json BinaryIntrinsicKind::logicneq_to_json(const BinaryIntrinsicKind::LogicNeq& x) { 
  return json::array({});
}

BinaryIntrinsicKind::LogicAnd BinaryIntrinsicKind::logicand_from_json(const json& j) { 
  return {};
}

json BinaryIntrinsicKind::logicand_to_json(const BinaryIntrinsicKind::LogicAnd& x) { 
  return json::array({});
}

BinaryIntrinsicKind::LogicOr BinaryIntrinsicKind::logicor_from_json(const json& j) { 
  return {};
}

json BinaryIntrinsicKind::logicor_to_json(const BinaryIntrinsicKind::LogicOr& x) { 
  return json::array({});
}

BinaryIntrinsicKind::LogicLte BinaryIntrinsicKind::logiclte_from_json(const json& j) { 
  return {};
}

json BinaryIntrinsicKind::logiclte_to_json(const BinaryIntrinsicKind::LogicLte& x) { 
  return json::array({});
}

BinaryIntrinsicKind::LogicGte BinaryIntrinsicKind::logicgte_from_json(const json& j) { 
  return {};
}

json BinaryIntrinsicKind::logicgte_to_json(const BinaryIntrinsicKind::LogicGte& x) { 
  return json::array({});
}

BinaryIntrinsicKind::LogicLt BinaryIntrinsicKind::logiclt_from_json(const json& j) { 
  return {};
}

json BinaryIntrinsicKind::logiclt_to_json(const BinaryIntrinsicKind::LogicLt& x) { 
  return json::array({});
}

BinaryIntrinsicKind::LogicGt BinaryIntrinsicKind::logicgt_from_json(const json& j) { 
  return {};
}

json BinaryIntrinsicKind::logicgt_to_json(const BinaryIntrinsicKind::LogicGt& x) { 
  return json::array({});
}

BinaryIntrinsicKind::Any BinaryIntrinsicKind::any_from_json(const json& j) { 
  size_t ord = j.at(0).get<size_t>();
  const auto t = j.at(1);
  switch (ord) {
  case 0: return BinaryIntrinsicKind::add_from_json(t);
  case 1: return BinaryIntrinsicKind::sub_from_json(t);
  case 2: return BinaryIntrinsicKind::mul_from_json(t);
  case 3: return BinaryIntrinsicKind::div_from_json(t);
  case 4: return BinaryIntrinsicKind::rem_from_json(t);
  case 5: return BinaryIntrinsicKind::pow_from_json(t);
  case 6: return BinaryIntrinsicKind::min_from_json(t);
  case 7: return BinaryIntrinsicKind::max_from_json(t);
  case 8: return BinaryIntrinsicKind::atan2_from_json(t);
  case 9: return BinaryIntrinsicKind::hypot_from_json(t);
  case 10: return BinaryIntrinsicKind::band_from_json(t);
  case 11: return BinaryIntrinsicKind::bor_from_json(t);
  case 12: return BinaryIntrinsicKind::bxor_from_json(t);
  case 13: return BinaryIntrinsicKind::bsl_from_json(t);
  case 14: return BinaryIntrinsicKind::bsr_from_json(t);
  case 15: return BinaryIntrinsicKind::bzsr_from_json(t);
  case 16: return BinaryIntrinsicKind::logiceq_from_json(t);
  case 17: return BinaryIntrinsicKind::logicneq_from_json(t);
  case 18: return BinaryIntrinsicKind::logicand_from_json(t);
  case 19: return BinaryIntrinsicKind::logicor_from_json(t);
  case 20: return BinaryIntrinsicKind::logiclte_from_json(t);
  case 21: return BinaryIntrinsicKind::logicgte_from_json(t);
  case 22: return BinaryIntrinsicKind::logiclt_from_json(t);
  case 23: return BinaryIntrinsicKind::logicgt_from_json(t);
  default: throw std::out_of_range("Bad ordinal " + std::to_string(ord));
  }
}

json BinaryIntrinsicKind::any_to_json(const BinaryIntrinsicKind::Any& x) { 
  return std::visit(overloaded{
  [](const BinaryIntrinsicKind::Add &y) -> json { return {0, BinaryIntrinsicKind::add_to_json(y)}; },
  [](const BinaryIntrinsicKind::Sub &y) -> json { return {1, BinaryIntrinsicKind::sub_to_json(y)}; },
  [](const BinaryIntrinsicKind::Mul &y) -> json { return {2, BinaryIntrinsicKind::mul_to_json(y)}; },
  [](const BinaryIntrinsicKind::Div &y) -> json { return {3, BinaryIntrinsicKind::div_to_json(y)}; },
  [](const BinaryIntrinsicKind::Rem &y) -> json { return {4, BinaryIntrinsicKind::rem_to_json(y)}; },
  [](const BinaryIntrinsicKind::Pow &y) -> json { return {5, BinaryIntrinsicKind::pow_to_json(y)}; },
  [](const BinaryIntrinsicKind::Min &y) -> json { return {6, BinaryIntrinsicKind::min_to_json(y)}; },
  [](const BinaryIntrinsicKind::Max &y) -> json { return {7, BinaryIntrinsicKind::max_to_json(y)}; },
  [](const BinaryIntrinsicKind::Atan2 &y) -> json { return {8, BinaryIntrinsicKind::atan2_to_json(y)}; },
  [](const BinaryIntrinsicKind::Hypot &y) -> json { return {9, BinaryIntrinsicKind::hypot_to_json(y)}; },
  [](const BinaryIntrinsicKind::BAnd &y) -> json { return {10, BinaryIntrinsicKind::band_to_json(y)}; },
  [](const BinaryIntrinsicKind::BOr &y) -> json { return {11, BinaryIntrinsicKind::bor_to_json(y)}; },
  [](const BinaryIntrinsicKind::BXor &y) -> json { return {12, BinaryIntrinsicKind::bxor_to_json(y)}; },
  [](const BinaryIntrinsicKind::BSL &y) -> json { return {13, BinaryIntrinsicKind::bsl_to_json(y)}; },
  [](const BinaryIntrinsicKind::BSR &y) -> json { return {14, BinaryIntrinsicKind::bsr_to_json(y)}; },
  [](const BinaryIntrinsicKind::BZSR &y) -> json { return {15, BinaryIntrinsicKind::bzsr_to_json(y)}; },
  [](const BinaryIntrinsicKind::LogicEq &y) -> json { return {16, BinaryIntrinsicKind::logiceq_to_json(y)}; },
  [](const BinaryIntrinsicKind::LogicNeq &y) -> json { return {17, BinaryIntrinsicKind::logicneq_to_json(y)}; },
  [](const BinaryIntrinsicKind::LogicAnd &y) -> json { return {18, BinaryIntrinsicKind::logicand_to_json(y)}; },
  [](const BinaryIntrinsicKind::LogicOr &y) -> json { return {19, BinaryIntrinsicKind::logicor_to_json(y)}; },
  [](const BinaryIntrinsicKind::LogicLte &y) -> json { return {20, BinaryIntrinsicKind::logiclte_to_json(y)}; },
  [](const BinaryIntrinsicKind::LogicGte &y) -> json { return {21, BinaryIntrinsicKind::logicgte_to_json(y)}; },
  [](const BinaryIntrinsicKind::LogicLt &y) -> json { return {22, BinaryIntrinsicKind::logiclt_to_json(y)}; },
  [](const BinaryIntrinsicKind::LogicGt &y) -> json { return {23, BinaryIntrinsicKind::logicgt_to_json(y)}; },
  [](const auto &x) -> json { throw std::out_of_range("Unimplemented type:" + to_string(x) ); }
  }, *x);
}

Expr::NullaryIntrinsic Expr::nullaryintrinsic_from_json(const json& j) { 
  auto kind =  NullaryIntrinsicKind::any_from_json(j.at(0));
  auto rtn =  Type::any_from_json(j.at(1));
  return {kind, rtn};
}

json Expr::nullaryintrinsic_to_json(const Expr::NullaryIntrinsic& x) { 
  auto kind =  NullaryIntrinsicKind::any_to_json(x.kind);
  auto rtn =  Type::any_to_json(x.rtn);
  return json::array({kind, rtn});
}

Expr::UnaryIntrinsic Expr::unaryintrinsic_from_json(const json& j) { 
  auto lhs =  Term::any_from_json(j.at(0));
  auto kind =  UnaryIntrinsicKind::any_from_json(j.at(1));
  auto rtn =  Type::any_from_json(j.at(2));
  return {lhs, kind, rtn};
}

json Expr::unaryintrinsic_to_json(const Expr::UnaryIntrinsic& x) { 
  auto lhs =  Term::any_to_json(x.lhs);
  auto kind =  UnaryIntrinsicKind::any_to_json(x.kind);
  auto rtn =  Type::any_to_json(x.rtn);
  return json::array({lhs, kind, rtn});
}

Expr::BinaryIntrinsic Expr::binaryintrinsic_from_json(const json& j) { 
  auto lhs =  Term::any_from_json(j.at(0));
  auto rhs =  Term::any_from_json(j.at(1));
  auto kind =  BinaryIntrinsicKind::any_from_json(j.at(2));
  auto rtn =  Type::any_from_json(j.at(3));
  return {lhs, rhs, kind, rtn};
}

json Expr::binaryintrinsic_to_json(const Expr::BinaryIntrinsic& x) { 
  auto lhs =  Term::any_to_json(x.lhs);
  auto rhs =  Term::any_to_json(x.rhs);
  auto kind =  BinaryIntrinsicKind::any_to_json(x.kind);
  auto rtn =  Type::any_to_json(x.rtn);
  return json::array({lhs, rhs, kind, rtn});
}

Expr::Cast Expr::cast_from_json(const json& j) { 
  auto from =  Term::any_from_json(j.at(0));
  auto as =  Type::any_from_json(j.at(1));
  return {from, as};
}

json Expr::cast_to_json(const Expr::Cast& x) { 
  auto from =  Term::any_to_json(x.from);
  auto as =  Type::any_to_json(x.as);
  return json::array({from, as});
}

Expr::Alias Expr::alias_from_json(const json& j) { 
  auto ref =  Term::any_from_json(j.at(0));
  return Expr::Alias(ref);
}

json Expr::alias_to_json(const Expr::Alias& x) { 
  auto ref =  Term::any_to_json(x.ref);
  return json::array({ref});
}

Expr::Index Expr::index_from_json(const json& j) { 
  auto lhs =  Term::any_from_json(j.at(0));
  auto idx =  Term::any_from_json(j.at(1));
  auto component =  Type::any_from_json(j.at(2));
  return {lhs, idx, component};
}

json Expr::index_to_json(const Expr::Index& x) { 
  auto lhs =  Term::any_to_json(x.lhs);
  auto idx =  Term::any_to_json(x.idx);
  auto component =  Type::any_to_json(x.component);
  return json::array({lhs, idx, component});
}

Expr::Alloc Expr::alloc_from_json(const json& j) { 
  auto component =  Type::any_from_json(j.at(0));
  auto size =  Term::any_from_json(j.at(1));
  return {component, size};
}

json Expr::alloc_to_json(const Expr::Alloc& x) { 
  auto component =  Type::any_to_json(x.component);
  auto size =  Term::any_to_json(x.size);
  return json::array({component, size});
}

Expr::Invoke Expr::invoke_from_json(const json& j) { 
  auto name =  sym_from_json(j.at(0));
  std::vector<Type::Any> tpeArgs;
  auto tpeArgs_json = j.at(1);
  std::transform(tpeArgs_json.begin(), tpeArgs_json.end(), std::back_inserter(tpeArgs), &Type::any_from_json);
  auto receiver = j.at(2).is_null() ? std::nullopt : std::make_optional(Term::any_from_json(j.at(2)));
  std::vector<Term::Any> args;
  auto args_json = j.at(3);
  std::transform(args_json.begin(), args_json.end(), std::back_inserter(args), &Term::any_from_json);
  std::vector<Term::Any> captures;
  auto captures_json = j.at(4);
  std::transform(captures_json.begin(), captures_json.end(), std::back_inserter(captures), &Term::any_from_json);
  auto rtn =  Type::any_from_json(j.at(5));
  return {name, tpeArgs, receiver, args, captures, rtn};
}

json Expr::invoke_to_json(const Expr::Invoke& x) { 
  auto name =  sym_to_json(x.name);
  std::vector<json> tpeArgs;
  std::transform(x.tpeArgs.begin(), x.tpeArgs.end(), std::back_inserter(tpeArgs), &Type::any_to_json);
  auto receiver = x.receiver ? Term::any_to_json(*x.receiver) : json{};
  std::vector<json> args;
  std::transform(x.args.begin(), x.args.end(), std::back_inserter(args), &Term::any_to_json);
  std::vector<json> captures;
  std::transform(x.captures.begin(), x.captures.end(), std::back_inserter(captures), &Term::any_to_json);
  auto rtn =  Type::any_to_json(x.rtn);
  return json::array({name, tpeArgs, receiver, args, captures, rtn});
}

Expr::Any Expr::any_from_json(const json& j) { 
  size_t ord = j.at(0).get<size_t>();
  const auto t = j.at(1);
  switch (ord) {
  case 0: return Expr::nullaryintrinsic_from_json(t);
  case 1: return Expr::unaryintrinsic_from_json(t);
  case 2: return Expr::binaryintrinsic_from_json(t);
  case 3: return Expr::cast_from_json(t);
  case 4: return Expr::alias_from_json(t);
  case 5: return Expr::index_from_json(t);
  case 6: return Expr::alloc_from_json(t);
  case 7: return Expr::invoke_from_json(t);
  default: throw std::out_of_range("Bad ordinal " + std::to_string(ord));
  }
}

json Expr::any_to_json(const Expr::Any& x) { 
  return std::visit(overloaded{
  [](const Expr::NullaryIntrinsic &y) -> json { return {0, Expr::nullaryintrinsic_to_json(y)}; },
  [](const Expr::UnaryIntrinsic &y) -> json { return {1, Expr::unaryintrinsic_to_json(y)}; },
  [](const Expr::BinaryIntrinsic &y) -> json { return {2, Expr::binaryintrinsic_to_json(y)}; },
  [](const Expr::Cast &y) -> json { return {3, Expr::cast_to_json(y)}; },
  [](const Expr::Alias &y) -> json { return {4, Expr::alias_to_json(y)}; },
  [](const Expr::Index &y) -> json { return {5, Expr::index_to_json(y)}; },
  [](const Expr::Alloc &y) -> json { return {6, Expr::alloc_to_json(y)}; },
  [](const Expr::Invoke &y) -> json { return {7, Expr::invoke_to_json(y)}; },
  [](const auto &x) -> json { throw std::out_of_range("Unimplemented type:" + to_string(x) ); }
  }, *x);
}

Stmt::Block Stmt::block_from_json(const json& j) { 
  std::vector<Stmt::Any> stmts;
  auto stmts_json = j.at(0);
  std::transform(stmts_json.begin(), stmts_json.end(), std::back_inserter(stmts), &Stmt::any_from_json);
  return Stmt::Block(stmts);
}

json Stmt::block_to_json(const Stmt::Block& x) { 
  std::vector<json> stmts;
  std::transform(x.stmts.begin(), x.stmts.end(), std::back_inserter(stmts), &Stmt::any_to_json);
  return json::array({stmts});
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
  auto name =  Term::any_from_json(j.at(0));
  auto expr =  Expr::any_from_json(j.at(1));
  auto copy = j.at(2).get<bool>();
  return {name, expr, copy};
}

json Stmt::mut_to_json(const Stmt::Mut& x) { 
  auto name =  Term::any_to_json(x.name);
  auto expr =  Expr::any_to_json(x.expr);
  auto copy = x.copy;
  return json::array({name, expr, copy});
}

Stmt::Update Stmt::update_from_json(const json& j) { 
  auto lhs =  Term::any_from_json(j.at(0));
  auto idx =  Term::any_from_json(j.at(1));
  auto value =  Term::any_from_json(j.at(2));
  return {lhs, idx, value};
}

json Stmt::update_to_json(const Stmt::Update& x) { 
  auto lhs =  Term::any_to_json(x.lhs);
  auto idx =  Term::any_to_json(x.idx);
  auto value =  Term::any_to_json(x.value);
  return json::array({lhs, idx, value});
}

Stmt::While Stmt::while_from_json(const json& j) { 
  std::vector<Stmt::Any> tests;
  auto tests_json = j.at(0);
  std::transform(tests_json.begin(), tests_json.end(), std::back_inserter(tests), &Stmt::any_from_json);
  auto cond =  Term::any_from_json(j.at(1));
  std::vector<Stmt::Any> body;
  auto body_json = j.at(2);
  std::transform(body_json.begin(), body_json.end(), std::back_inserter(body), &Stmt::any_from_json);
  return {tests, cond, body};
}

json Stmt::while_to_json(const Stmt::While& x) { 
  std::vector<json> tests;
  std::transform(x.tests.begin(), x.tests.end(), std::back_inserter(tests), &Stmt::any_to_json);
  auto cond =  Term::any_to_json(x.cond);
  std::vector<json> body;
  std::transform(x.body.begin(), x.body.end(), std::back_inserter(body), &Stmt::any_to_json);
  return json::array({tests, cond, body});
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
  case 0: return Stmt::block_from_json(t);
  case 1: return Stmt::comment_from_json(t);
  case 2: return Stmt::var_from_json(t);
  case 3: return Stmt::mut_from_json(t);
  case 4: return Stmt::update_from_json(t);
  case 5: return Stmt::while_from_json(t);
  case 6: return Stmt::break_from_json(t);
  case 7: return Stmt::cont_from_json(t);
  case 8: return Stmt::cond_from_json(t);
  case 9: return Stmt::return_from_json(t);
  default: throw std::out_of_range("Bad ordinal " + std::to_string(ord));
  }
}

json Stmt::any_to_json(const Stmt::Any& x) { 
  return std::visit(overloaded{
  [](const Stmt::Block &y) -> json { return {0, Stmt::block_to_json(y)}; },
  [](const Stmt::Comment &y) -> json { return {1, Stmt::comment_to_json(y)}; },
  [](const Stmt::Var &y) -> json { return {2, Stmt::var_to_json(y)}; },
  [](const Stmt::Mut &y) -> json { return {3, Stmt::mut_to_json(y)}; },
  [](const Stmt::Update &y) -> json { return {4, Stmt::update_to_json(y)}; },
  [](const Stmt::While &y) -> json { return {5, Stmt::while_to_json(y)}; },
  [](const Stmt::Break &y) -> json { return {6, Stmt::break_to_json(y)}; },
  [](const Stmt::Cont &y) -> json { return {7, Stmt::cont_to_json(y)}; },
  [](const Stmt::Cond &y) -> json { return {8, Stmt::cond_to_json(y)}; },
  [](const Stmt::Return &y) -> json { return {9, Stmt::return_to_json(y)}; },
  [](const auto &x) -> json { throw std::out_of_range("Unimplemented type:" + to_string(x) ); }
  }, *x);
}

StructMember structmember_from_json(const json& j) { 
  auto named =  named_from_json(j.at(0));
  auto isMutable = j.at(1).get<bool>();
  return {named, isMutable};
}

json structmember_to_json(const StructMember& x) { 
  auto named =  named_to_json(x.named);
  auto isMutable = x.isMutable;
  return json::array({named, isMutable});
}

StructDef structdef_from_json(const json& j) { 
  auto name =  sym_from_json(j.at(0));
  auto isReference = j.at(1).get<bool>();
  auto tpeVars = j.at(2).get<std::vector<std::string>>();
  std::vector<StructMember> members;
  auto members_json = j.at(3);
  std::transform(members_json.begin(), members_json.end(), std::back_inserter(members), &structmember_from_json);
  std::vector<Sym> parents;
  auto parents_json = j.at(4);
  std::transform(parents_json.begin(), parents_json.end(), std::back_inserter(parents), &sym_from_json);
  return {name, isReference, tpeVars, members, parents};
}

json structdef_to_json(const StructDef& x) { 
  auto name =  sym_to_json(x.name);
  auto isReference = x.isReference;
  auto tpeVars = x.tpeVars;
  std::vector<json> members;
  std::transform(x.members.begin(), x.members.end(), std::back_inserter(members), &structmember_to_json);
  std::vector<json> parents;
  std::transform(x.parents.begin(), x.parents.end(), std::back_inserter(parents), &sym_to_json);
  return json::array({name, isReference, tpeVars, members, parents});
}

Signature signature_from_json(const json& j) { 
  auto name =  sym_from_json(j.at(0));
  auto tpeVars = j.at(1).get<std::vector<std::string>>();
  auto receiver = j.at(2).is_null() ? std::nullopt : std::make_optional(Type::any_from_json(j.at(2)));
  std::vector<Type::Any> args;
  auto args_json = j.at(3);
  std::transform(args_json.begin(), args_json.end(), std::back_inserter(args), &Type::any_from_json);
  std::vector<Type::Any> moduleCaptures;
  auto moduleCaptures_json = j.at(4);
  std::transform(moduleCaptures_json.begin(), moduleCaptures_json.end(), std::back_inserter(moduleCaptures), &Type::any_from_json);
  std::vector<Type::Any> termCaptures;
  auto termCaptures_json = j.at(5);
  std::transform(termCaptures_json.begin(), termCaptures_json.end(), std::back_inserter(termCaptures), &Type::any_from_json);
  auto rtn =  Type::any_from_json(j.at(6));
  return {name, tpeVars, receiver, args, moduleCaptures, termCaptures, rtn};
}

json signature_to_json(const Signature& x) { 
  auto name =  sym_to_json(x.name);
  auto tpeVars = x.tpeVars;
  auto receiver = x.receiver ? Type::any_to_json(*x.receiver) : json{};
  std::vector<json> args;
  std::transform(x.args.begin(), x.args.end(), std::back_inserter(args), &Type::any_to_json);
  std::vector<json> moduleCaptures;
  std::transform(x.moduleCaptures.begin(), x.moduleCaptures.end(), std::back_inserter(moduleCaptures), &Type::any_to_json);
  std::vector<json> termCaptures;
  std::transform(x.termCaptures.begin(), x.termCaptures.end(), std::back_inserter(termCaptures), &Type::any_to_json);
  auto rtn =  Type::any_to_json(x.rtn);
  return json::array({name, tpeVars, receiver, args, moduleCaptures, termCaptures, rtn});
}

InvokeSignature invokesignature_from_json(const json& j) { 
  auto name =  sym_from_json(j.at(0));
  std::vector<Type::Any> tpeVars;
  auto tpeVars_json = j.at(1);
  std::transform(tpeVars_json.begin(), tpeVars_json.end(), std::back_inserter(tpeVars), &Type::any_from_json);
  auto receiver = j.at(2).is_null() ? std::nullopt : std::make_optional(Type::any_from_json(j.at(2)));
  std::vector<Type::Any> args;
  auto args_json = j.at(3);
  std::transform(args_json.begin(), args_json.end(), std::back_inserter(args), &Type::any_from_json);
  std::vector<Type::Any> captures;
  auto captures_json = j.at(4);
  std::transform(captures_json.begin(), captures_json.end(), std::back_inserter(captures), &Type::any_from_json);
  auto rtn =  Type::any_from_json(j.at(5));
  return {name, tpeVars, receiver, args, captures, rtn};
}

json invokesignature_to_json(const InvokeSignature& x) { 
  auto name =  sym_to_json(x.name);
  std::vector<json> tpeVars;
  std::transform(x.tpeVars.begin(), x.tpeVars.end(), std::back_inserter(tpeVars), &Type::any_to_json);
  auto receiver = x.receiver ? Type::any_to_json(*x.receiver) : json{};
  std::vector<json> args;
  std::transform(x.args.begin(), x.args.end(), std::back_inserter(args), &Type::any_to_json);
  std::vector<json> captures;
  std::transform(x.captures.begin(), x.captures.end(), std::back_inserter(captures), &Type::any_to_json);
  auto rtn =  Type::any_to_json(x.rtn);
  return json::array({name, tpeVars, receiver, args, captures, rtn});
}

Function function_from_json(const json& j) { 
  auto name =  sym_from_json(j.at(0));
  auto tpeVars = j.at(1).get<std::vector<std::string>>();
  auto receiver = j.at(2).is_null() ? std::nullopt : std::make_optional(named_from_json(j.at(2)));
  std::vector<Named> args;
  auto args_json = j.at(3);
  std::transform(args_json.begin(), args_json.end(), std::back_inserter(args), &named_from_json);
  std::vector<Named> moduleCaptures;
  auto moduleCaptures_json = j.at(4);
  std::transform(moduleCaptures_json.begin(), moduleCaptures_json.end(), std::back_inserter(moduleCaptures), &named_from_json);
  std::vector<Named> termCaptures;
  auto termCaptures_json = j.at(5);
  std::transform(termCaptures_json.begin(), termCaptures_json.end(), std::back_inserter(termCaptures), &named_from_json);
  auto rtn =  Type::any_from_json(j.at(6));
  std::vector<Stmt::Any> body;
  auto body_json = j.at(7);
  std::transform(body_json.begin(), body_json.end(), std::back_inserter(body), &Stmt::any_from_json);
  return {name, tpeVars, receiver, args, moduleCaptures, termCaptures, rtn, body};
}

json function_to_json(const Function& x) { 
  auto name =  sym_to_json(x.name);
  auto tpeVars = x.tpeVars;
  auto receiver = x.receiver ? named_to_json(*x.receiver) : json{};
  std::vector<json> args;
  std::transform(x.args.begin(), x.args.end(), std::back_inserter(args), &named_to_json);
  std::vector<json> moduleCaptures;
  std::transform(x.moduleCaptures.begin(), x.moduleCaptures.end(), std::back_inserter(moduleCaptures), &named_to_json);
  std::vector<json> termCaptures;
  std::transform(x.termCaptures.begin(), x.termCaptures.end(), std::back_inserter(termCaptures), &named_to_json);
  auto rtn =  Type::any_to_json(x.rtn);
  std::vector<json> body;
  std::transform(x.body.begin(), x.body.end(), std::back_inserter(body), &Stmt::any_to_json);
  return json::array({name, tpeVars, receiver, args, moduleCaptures, termCaptures, rtn, body});
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
  if(hash != "075c7cbef64ab48bdf60398ceb0b5abf") {
   throw std::runtime_error("Expecting ADT hash to be 075c7cbef64ab48bdf60398ceb0b5abf, but was " + hash);
  }
  return data;
}

json hashed_to_json(const json& x) { 
  return json::array({"075c7cbef64ab48bdf60398ceb0b5abf", x});
}
} // namespace polyregion::polyast
