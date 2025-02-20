#include "polyast_codec.h"

template<class... Ts> struct overloaded : Ts... { using Ts::operator()...; };
template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

namespace polyregion::polyast { 

SourcePosition sourceposition_from_json(const json& j_) { 
  auto file = j_.at(0).get<std::string>();
  auto line = j_.at(1).get<int32_t>();
  auto col = j_.at(2).is_null() ? std::nullopt : std::make_optional(j_.at(2).get<int32_t>());
  return {file, line, col};
}

json sourceposition_to_json(const SourcePosition& x_) { 
  auto file = x_.file;
  auto line = x_.line;
  auto col = x_.col ? json(*x_.col) : json();
  return json::array({file, line, col});
}

Named named_from_json(const json& j_) { 
  auto symbol = j_.at(0).get<std::string>();
  auto tpe = Type::any_from_json(j_.at(1));
  return {symbol, tpe};
}

json named_to_json(const Named& x_) { 
  auto symbol = x_.symbol;
  auto tpe = Type::any_to_json(x_.tpe);
  return json::array({symbol, tpe});
}

TypeKind::None TypeKind::none_from_json(const json& j_) { 
  return {};
}

json TypeKind::none_to_json(const TypeKind::None& x_) { 
  return json::array({});
}

TypeKind::Ref TypeKind::ref_from_json(const json& j_) { 
  return {};
}

json TypeKind::ref_to_json(const TypeKind::Ref& x_) { 
  return json::array({});
}

TypeKind::Integral TypeKind::integral_from_json(const json& j_) { 
  return {};
}

json TypeKind::integral_to_json(const TypeKind::Integral& x_) { 
  return json::array({});
}

TypeKind::Fractional TypeKind::fractional_from_json(const json& j_) { 
  return {};
}

json TypeKind::fractional_to_json(const TypeKind::Fractional& x_) { 
  return json::array({});
}

TypeKind::Any TypeKind::any_from_json(const json& j_) { 
  size_t ord_ = j_.at(0).get<size_t>();
  const auto &t_ = j_.at(1);
  switch (ord_) {
  case 0: return TypeKind::none_from_json(t_);
  case 1: return TypeKind::ref_from_json(t_);
  case 2: return TypeKind::integral_from_json(t_);
  case 3: return TypeKind::fractional_from_json(t_);
  default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
  }
}

json TypeKind::any_to_json(const TypeKind::Any& x_) { 
  return x_.match_total(
  [](const TypeKind::None &y_) -> json { return {0, TypeKind::none_to_json(y_)}; }
  ,
  [](const TypeKind::Ref &y_) -> json { return {1, TypeKind::ref_to_json(y_)}; }
  ,
  [](const TypeKind::Integral &y_) -> json { return {2, TypeKind::integral_to_json(y_)}; }
  ,
  [](const TypeKind::Fractional &y_) -> json { return {3, TypeKind::fractional_to_json(y_)}; }
  );
}

TypeSpace::Global TypeSpace::global_from_json(const json& j_) { 
  return {};
}

json TypeSpace::global_to_json(const TypeSpace::Global& x_) { 
  return json::array({});
}

TypeSpace::Local TypeSpace::local_from_json(const json& j_) { 
  return {};
}

json TypeSpace::local_to_json(const TypeSpace::Local& x_) { 
  return json::array({});
}

TypeSpace::Private TypeSpace::private_from_json(const json& j_) { 
  return {};
}

json TypeSpace::private_to_json(const TypeSpace::Private& x_) { 
  return json::array({});
}

TypeSpace::Any TypeSpace::any_from_json(const json& j_) { 
  size_t ord_ = j_.at(0).get<size_t>();
  const auto &t_ = j_.at(1);
  switch (ord_) {
  case 0: return TypeSpace::global_from_json(t_);
  case 1: return TypeSpace::local_from_json(t_);
  case 2: return TypeSpace::private_from_json(t_);
  default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
  }
}

json TypeSpace::any_to_json(const TypeSpace::Any& x_) { 
  return x_.match_total(
  [](const TypeSpace::Global &y_) -> json { return {0, TypeSpace::global_to_json(y_)}; }
  ,
  [](const TypeSpace::Local &y_) -> json { return {1, TypeSpace::local_to_json(y_)}; }
  ,
  [](const TypeSpace::Private &y_) -> json { return {2, TypeSpace::private_to_json(y_)}; }
  );
}

Type::Float16 Type::float16_from_json(const json& j_) { 
  return {};
}

json Type::float16_to_json(const Type::Float16& x_) { 
  return json::array({});
}

Type::Float32 Type::float32_from_json(const json& j_) { 
  return {};
}

json Type::float32_to_json(const Type::Float32& x_) { 
  return json::array({});
}

Type::Float64 Type::float64_from_json(const json& j_) { 
  return {};
}

json Type::float64_to_json(const Type::Float64& x_) { 
  return json::array({});
}

Type::IntU8 Type::intu8_from_json(const json& j_) { 
  return {};
}

json Type::intu8_to_json(const Type::IntU8& x_) { 
  return json::array({});
}

Type::IntU16 Type::intu16_from_json(const json& j_) { 
  return {};
}

json Type::intu16_to_json(const Type::IntU16& x_) { 
  return json::array({});
}

Type::IntU32 Type::intu32_from_json(const json& j_) { 
  return {};
}

json Type::intu32_to_json(const Type::IntU32& x_) { 
  return json::array({});
}

Type::IntU64 Type::intu64_from_json(const json& j_) { 
  return {};
}

json Type::intu64_to_json(const Type::IntU64& x_) { 
  return json::array({});
}

Type::IntS8 Type::ints8_from_json(const json& j_) { 
  return {};
}

json Type::ints8_to_json(const Type::IntS8& x_) { 
  return json::array({});
}

Type::IntS16 Type::ints16_from_json(const json& j_) { 
  return {};
}

json Type::ints16_to_json(const Type::IntS16& x_) { 
  return json::array({});
}

Type::IntS32 Type::ints32_from_json(const json& j_) { 
  return {};
}

json Type::ints32_to_json(const Type::IntS32& x_) { 
  return json::array({});
}

Type::IntS64 Type::ints64_from_json(const json& j_) { 
  return {};
}

json Type::ints64_to_json(const Type::IntS64& x_) { 
  return json::array({});
}

Type::Nothing Type::nothing_from_json(const json& j_) { 
  return {};
}

json Type::nothing_to_json(const Type::Nothing& x_) { 
  return json::array({});
}

Type::Unit0 Type::unit0_from_json(const json& j_) { 
  return {};
}

json Type::unit0_to_json(const Type::Unit0& x_) { 
  return json::array({});
}

Type::Bool1 Type::bool1_from_json(const json& j_) { 
  return {};
}

json Type::bool1_to_json(const Type::Bool1& x_) { 
  return json::array({});
}

Type::Struct Type::struct_from_json(const json& j_) { 
  auto name = j_.at(0).get<std::string>();
  return Type::Struct(name);
}

json Type::struct_to_json(const Type::Struct& x_) { 
  auto name = x_.name;
  return json::array({name});
}

Type::Ptr Type::ptr_from_json(const json& j_) { 
  auto comp = Type::any_from_json(j_.at(0));
  auto length = j_.at(1).is_null() ? std::nullopt : std::make_optional(j_.at(1).get<int32_t>());
  auto space = TypeSpace::any_from_json(j_.at(2));
  return {comp, length, space};
}

json Type::ptr_to_json(const Type::Ptr& x_) { 
  auto comp = Type::any_to_json(x_.comp);
  auto length = x_.length ? json(*x_.length) : json();
  auto space = TypeSpace::any_to_json(x_.space);
  return json::array({comp, length, space});
}

Type::Annotated Type::annotated_from_json(const json& j_) { 
  auto tpe = Type::any_from_json(j_.at(0));
  auto pos = j_.at(1).is_null() ? std::nullopt : std::make_optional(sourceposition_from_json(j_.at(1)));
  auto comment = j_.at(2).is_null() ? std::nullopt : std::make_optional(j_.at(2).get<std::string>());
  return {tpe, pos, comment};
}

json Type::annotated_to_json(const Type::Annotated& x_) { 
  auto tpe = Type::any_to_json(x_.tpe);
  auto pos = x_.pos ? sourceposition_to_json(*x_.pos) : json();
  auto comment = x_.comment ? json(*x_.comment) : json();
  return json::array({tpe, pos, comment});
}

Type::Any Type::any_from_json(const json& j_) { 
  size_t ord_ = j_.at(0).get<size_t>();
  const auto &t_ = j_.at(1);
  switch (ord_) {
  case 0: return Type::float16_from_json(t_);
  case 1: return Type::float32_from_json(t_);
  case 2: return Type::float64_from_json(t_);
  case 3: return Type::intu8_from_json(t_);
  case 4: return Type::intu16_from_json(t_);
  case 5: return Type::intu32_from_json(t_);
  case 6: return Type::intu64_from_json(t_);
  case 7: return Type::ints8_from_json(t_);
  case 8: return Type::ints16_from_json(t_);
  case 9: return Type::ints32_from_json(t_);
  case 10: return Type::ints64_from_json(t_);
  case 11: return Type::nothing_from_json(t_);
  case 12: return Type::unit0_from_json(t_);
  case 13: return Type::bool1_from_json(t_);
  case 14: return Type::struct_from_json(t_);
  case 15: return Type::ptr_from_json(t_);
  case 16: return Type::annotated_from_json(t_);
  default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
  }
}

json Type::any_to_json(const Type::Any& x_) { 
  return x_.match_total(
  [](const Type::Float16 &y_) -> json { return {0, Type::float16_to_json(y_)}; }
  ,
  [](const Type::Float32 &y_) -> json { return {1, Type::float32_to_json(y_)}; }
  ,
  [](const Type::Float64 &y_) -> json { return {2, Type::float64_to_json(y_)}; }
  ,
  [](const Type::IntU8 &y_) -> json { return {3, Type::intu8_to_json(y_)}; }
  ,
  [](const Type::IntU16 &y_) -> json { return {4, Type::intu16_to_json(y_)}; }
  ,
  [](const Type::IntU32 &y_) -> json { return {5, Type::intu32_to_json(y_)}; }
  ,
  [](const Type::IntU64 &y_) -> json { return {6, Type::intu64_to_json(y_)}; }
  ,
  [](const Type::IntS8 &y_) -> json { return {7, Type::ints8_to_json(y_)}; }
  ,
  [](const Type::IntS16 &y_) -> json { return {8, Type::ints16_to_json(y_)}; }
  ,
  [](const Type::IntS32 &y_) -> json { return {9, Type::ints32_to_json(y_)}; }
  ,
  [](const Type::IntS64 &y_) -> json { return {10, Type::ints64_to_json(y_)}; }
  ,
  [](const Type::Nothing &y_) -> json { return {11, Type::nothing_to_json(y_)}; }
  ,
  [](const Type::Unit0 &y_) -> json { return {12, Type::unit0_to_json(y_)}; }
  ,
  [](const Type::Bool1 &y_) -> json { return {13, Type::bool1_to_json(y_)}; }
  ,
  [](const Type::Struct &y_) -> json { return {14, Type::struct_to_json(y_)}; }
  ,
  [](const Type::Ptr &y_) -> json { return {15, Type::ptr_to_json(y_)}; }
  ,
  [](const Type::Annotated &y_) -> json { return {16, Type::annotated_to_json(y_)}; }
  );
}

Expr::Float16Const Expr::float16const_from_json(const json& j_) { 
  auto value = j_.at(0).get<float>();
  return Expr::Float16Const(value);
}

json Expr::float16const_to_json(const Expr::Float16Const& x_) { 
  auto value = x_.value;
  return json::array({value});
}

Expr::Float32Const Expr::float32const_from_json(const json& j_) { 
  auto value = j_.at(0).get<float>();
  return Expr::Float32Const(value);
}

json Expr::float32const_to_json(const Expr::Float32Const& x_) { 
  auto value = x_.value;
  return json::array({value});
}

Expr::Float64Const Expr::float64const_from_json(const json& j_) { 
  auto value = j_.at(0).get<double>();
  return Expr::Float64Const(value);
}

json Expr::float64const_to_json(const Expr::Float64Const& x_) { 
  auto value = x_.value;
  return json::array({value});
}

Expr::IntU8Const Expr::intu8const_from_json(const json& j_) { 
  auto value = j_.at(0).get<int8_t>();
  return Expr::IntU8Const(value);
}

json Expr::intu8const_to_json(const Expr::IntU8Const& x_) { 
  auto value = x_.value;
  return json::array({value});
}

Expr::IntU16Const Expr::intu16const_from_json(const json& j_) { 
  auto value = j_.at(0).get<uint16_t>();
  return Expr::IntU16Const(value);
}

json Expr::intu16const_to_json(const Expr::IntU16Const& x_) { 
  auto value = x_.value;
  return json::array({value});
}

Expr::IntU32Const Expr::intu32const_from_json(const json& j_) { 
  auto value = j_.at(0).get<int32_t>();
  return Expr::IntU32Const(value);
}

json Expr::intu32const_to_json(const Expr::IntU32Const& x_) { 
  auto value = x_.value;
  return json::array({value});
}

Expr::IntU64Const Expr::intu64const_from_json(const json& j_) { 
  auto value = j_.at(0).get<int64_t>();
  return Expr::IntU64Const(value);
}

json Expr::intu64const_to_json(const Expr::IntU64Const& x_) { 
  auto value = x_.value;
  return json::array({value});
}

Expr::IntS8Const Expr::ints8const_from_json(const json& j_) { 
  auto value = j_.at(0).get<int8_t>();
  return Expr::IntS8Const(value);
}

json Expr::ints8const_to_json(const Expr::IntS8Const& x_) { 
  auto value = x_.value;
  return json::array({value});
}

Expr::IntS16Const Expr::ints16const_from_json(const json& j_) { 
  auto value = j_.at(0).get<int16_t>();
  return Expr::IntS16Const(value);
}

json Expr::ints16const_to_json(const Expr::IntS16Const& x_) { 
  auto value = x_.value;
  return json::array({value});
}

Expr::IntS32Const Expr::ints32const_from_json(const json& j_) { 
  auto value = j_.at(0).get<int32_t>();
  return Expr::IntS32Const(value);
}

json Expr::ints32const_to_json(const Expr::IntS32Const& x_) { 
  auto value = x_.value;
  return json::array({value});
}

Expr::IntS64Const Expr::ints64const_from_json(const json& j_) { 
  auto value = j_.at(0).get<int64_t>();
  return Expr::IntS64Const(value);
}

json Expr::ints64const_to_json(const Expr::IntS64Const& x_) { 
  auto value = x_.value;
  return json::array({value});
}

Expr::Unit0Const Expr::unit0const_from_json(const json& j_) { 
  return {};
}

json Expr::unit0const_to_json(const Expr::Unit0Const& x_) { 
  return json::array({});
}

Expr::Bool1Const Expr::bool1const_from_json(const json& j_) { 
  auto value = j_.at(0).get<bool>();
  return Expr::Bool1Const(value);
}

json Expr::bool1const_to_json(const Expr::Bool1Const& x_) { 
  auto value = x_.value;
  return json::array({value});
}

Expr::NullPtrConst Expr::nullptrconst_from_json(const json& j_) { 
  auto comp = Type::any_from_json(j_.at(0));
  auto space = TypeSpace::any_from_json(j_.at(1));
  return {comp, space};
}

json Expr::nullptrconst_to_json(const Expr::NullPtrConst& x_) { 
  auto comp = Type::any_to_json(x_.comp);
  auto space = TypeSpace::any_to_json(x_.space);
  return json::array({comp, space});
}

Expr::SpecOp Expr::specop_from_json(const json& j_) { 
  auto op = Spec::any_from_json(j_.at(0));
  return Expr::SpecOp(op);
}

json Expr::specop_to_json(const Expr::SpecOp& x_) { 
  auto op = Spec::any_to_json(x_.op);
  return json::array({op});
}

Expr::MathOp Expr::mathop_from_json(const json& j_) { 
  auto op = Math::any_from_json(j_.at(0));
  return Expr::MathOp(op);
}

json Expr::mathop_to_json(const Expr::MathOp& x_) { 
  auto op = Math::any_to_json(x_.op);
  return json::array({op});
}

Expr::IntrOp Expr::introp_from_json(const json& j_) { 
  auto op = Intr::any_from_json(j_.at(0));
  return Expr::IntrOp(op);
}

json Expr::introp_to_json(const Expr::IntrOp& x_) { 
  auto op = Intr::any_to_json(x_.op);
  return json::array({op});
}

Expr::Select Expr::select_from_json(const json& j_) { 
  std::vector<Named> init;
  for(const auto &v_ : j_.at(0)) { init.emplace_back(named_from_json(v_)); }
  auto last = named_from_json(j_.at(1));
  return {init, last};
}

json Expr::select_to_json(const Expr::Select& x_) { 
  std::vector<json> init;
  for(const auto &v_ : x_.init) { init.emplace_back(named_to_json(v_)); }
  auto last = named_to_json(x_.last);
  return json::array({init, last});
}

Expr::Poison Expr::poison_from_json(const json& j_) { 
  auto t = Type::any_from_json(j_.at(0));
  return Expr::Poison(t);
}

json Expr::poison_to_json(const Expr::Poison& x_) { 
  auto t = Type::any_to_json(x_.t);
  return json::array({t});
}

Expr::Cast Expr::cast_from_json(const json& j_) { 
  auto from = Expr::any_from_json(j_.at(0));
  auto as = Type::any_from_json(j_.at(1));
  return {from, as};
}

json Expr::cast_to_json(const Expr::Cast& x_) { 
  auto from = Expr::any_to_json(x_.from);
  auto as = Type::any_to_json(x_.as);
  return json::array({from, as});
}

Expr::Index Expr::index_from_json(const json& j_) { 
  auto lhs = Expr::any_from_json(j_.at(0));
  auto idx = Expr::any_from_json(j_.at(1));
  auto comp = Type::any_from_json(j_.at(2));
  return {lhs, idx, comp};
}

json Expr::index_to_json(const Expr::Index& x_) { 
  auto lhs = Expr::any_to_json(x_.lhs);
  auto idx = Expr::any_to_json(x_.idx);
  auto comp = Type::any_to_json(x_.comp);
  return json::array({lhs, idx, comp});
}

Expr::RefTo Expr::refto_from_json(const json& j_) { 
  auto lhs = Expr::any_from_json(j_.at(0));
  auto idx = j_.at(1).is_null() ? std::nullopt : std::make_optional(Expr::any_from_json(j_.at(1)));
  auto comp = Type::any_from_json(j_.at(2));
  auto space = TypeSpace::any_from_json(j_.at(3));
  return {lhs, idx, comp, space};
}

json Expr::refto_to_json(const Expr::RefTo& x_) { 
  auto lhs = Expr::any_to_json(x_.lhs);
  auto idx = x_.idx ? Expr::any_to_json(*x_.idx) : json();
  auto comp = Type::any_to_json(x_.comp);
  auto space = TypeSpace::any_to_json(x_.space);
  return json::array({lhs, idx, comp, space});
}

Expr::Alloc Expr::alloc_from_json(const json& j_) { 
  auto comp = Type::any_from_json(j_.at(0));
  auto size = Expr::any_from_json(j_.at(1));
  auto space = TypeSpace::any_from_json(j_.at(2));
  return {comp, size, space};
}

json Expr::alloc_to_json(const Expr::Alloc& x_) { 
  auto comp = Type::any_to_json(x_.comp);
  auto size = Expr::any_to_json(x_.size);
  auto space = TypeSpace::any_to_json(x_.space);
  return json::array({comp, size, space});
}

Expr::Invoke Expr::invoke_from_json(const json& j_) { 
  auto name = j_.at(0).get<std::string>();
  std::vector<Expr::Any> args;
  for(const auto &v_ : j_.at(1)) { args.emplace_back(Expr::any_from_json(v_)); }
  auto rtn = Type::any_from_json(j_.at(2));
  return {name, args, rtn};
}

json Expr::invoke_to_json(const Expr::Invoke& x_) { 
  auto name = x_.name;
  std::vector<json> args;
  for(const auto &v_ : x_.args) { args.emplace_back(Expr::any_to_json(v_)); }
  auto rtn = Type::any_to_json(x_.rtn);
  return json::array({name, args, rtn});
}

Expr::Annotated Expr::annotated_from_json(const json& j_) { 
  auto expr = Expr::any_from_json(j_.at(0));
  auto pos = j_.at(1).is_null() ? std::nullopt : std::make_optional(sourceposition_from_json(j_.at(1)));
  auto comment = j_.at(2).is_null() ? std::nullopt : std::make_optional(j_.at(2).get<std::string>());
  return {expr, pos, comment};
}

json Expr::annotated_to_json(const Expr::Annotated& x_) { 
  auto expr = Expr::any_to_json(x_.expr);
  auto pos = x_.pos ? sourceposition_to_json(*x_.pos) : json();
  auto comment = x_.comment ? json(*x_.comment) : json();
  return json::array({expr, pos, comment});
}

Expr::Any Expr::any_from_json(const json& j_) { 
  size_t ord_ = j_.at(0).get<size_t>();
  const auto &t_ = j_.at(1);
  switch (ord_) {
  case 0: return Expr::float16const_from_json(t_);
  case 1: return Expr::float32const_from_json(t_);
  case 2: return Expr::float64const_from_json(t_);
  case 3: return Expr::intu8const_from_json(t_);
  case 4: return Expr::intu16const_from_json(t_);
  case 5: return Expr::intu32const_from_json(t_);
  case 6: return Expr::intu64const_from_json(t_);
  case 7: return Expr::ints8const_from_json(t_);
  case 8: return Expr::ints16const_from_json(t_);
  case 9: return Expr::ints32const_from_json(t_);
  case 10: return Expr::ints64const_from_json(t_);
  case 11: return Expr::unit0const_from_json(t_);
  case 12: return Expr::bool1const_from_json(t_);
  case 13: return Expr::nullptrconst_from_json(t_);
  case 14: return Expr::specop_from_json(t_);
  case 15: return Expr::mathop_from_json(t_);
  case 16: return Expr::introp_from_json(t_);
  case 17: return Expr::select_from_json(t_);
  case 18: return Expr::poison_from_json(t_);
  case 19: return Expr::cast_from_json(t_);
  case 20: return Expr::index_from_json(t_);
  case 21: return Expr::refto_from_json(t_);
  case 22: return Expr::alloc_from_json(t_);
  case 23: return Expr::invoke_from_json(t_);
  case 24: return Expr::annotated_from_json(t_);
  default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
  }
}

json Expr::any_to_json(const Expr::Any& x_) { 
  return x_.match_total(
  [](const Expr::Float16Const &y_) -> json { return {0, Expr::float16const_to_json(y_)}; }
  ,
  [](const Expr::Float32Const &y_) -> json { return {1, Expr::float32const_to_json(y_)}; }
  ,
  [](const Expr::Float64Const &y_) -> json { return {2, Expr::float64const_to_json(y_)}; }
  ,
  [](const Expr::IntU8Const &y_) -> json { return {3, Expr::intu8const_to_json(y_)}; }
  ,
  [](const Expr::IntU16Const &y_) -> json { return {4, Expr::intu16const_to_json(y_)}; }
  ,
  [](const Expr::IntU32Const &y_) -> json { return {5, Expr::intu32const_to_json(y_)}; }
  ,
  [](const Expr::IntU64Const &y_) -> json { return {6, Expr::intu64const_to_json(y_)}; }
  ,
  [](const Expr::IntS8Const &y_) -> json { return {7, Expr::ints8const_to_json(y_)}; }
  ,
  [](const Expr::IntS16Const &y_) -> json { return {8, Expr::ints16const_to_json(y_)}; }
  ,
  [](const Expr::IntS32Const &y_) -> json { return {9, Expr::ints32const_to_json(y_)}; }
  ,
  [](const Expr::IntS64Const &y_) -> json { return {10, Expr::ints64const_to_json(y_)}; }
  ,
  [](const Expr::Unit0Const &y_) -> json { return {11, Expr::unit0const_to_json(y_)}; }
  ,
  [](const Expr::Bool1Const &y_) -> json { return {12, Expr::bool1const_to_json(y_)}; }
  ,
  [](const Expr::NullPtrConst &y_) -> json { return {13, Expr::nullptrconst_to_json(y_)}; }
  ,
  [](const Expr::SpecOp &y_) -> json { return {14, Expr::specop_to_json(y_)}; }
  ,
  [](const Expr::MathOp &y_) -> json { return {15, Expr::mathop_to_json(y_)}; }
  ,
  [](const Expr::IntrOp &y_) -> json { return {16, Expr::introp_to_json(y_)}; }
  ,
  [](const Expr::Select &y_) -> json { return {17, Expr::select_to_json(y_)}; }
  ,
  [](const Expr::Poison &y_) -> json { return {18, Expr::poison_to_json(y_)}; }
  ,
  [](const Expr::Cast &y_) -> json { return {19, Expr::cast_to_json(y_)}; }
  ,
  [](const Expr::Index &y_) -> json { return {20, Expr::index_to_json(y_)}; }
  ,
  [](const Expr::RefTo &y_) -> json { return {21, Expr::refto_to_json(y_)}; }
  ,
  [](const Expr::Alloc &y_) -> json { return {22, Expr::alloc_to_json(y_)}; }
  ,
  [](const Expr::Invoke &y_) -> json { return {23, Expr::invoke_to_json(y_)}; }
  ,
  [](const Expr::Annotated &y_) -> json { return {24, Expr::annotated_to_json(y_)}; }
  );
}

Overload overload_from_json(const json& j_) { 
  std::vector<Type::Any> args;
  for(const auto &v_ : j_.at(0)) { args.emplace_back(Type::any_from_json(v_)); }
  auto rtn = Type::any_from_json(j_.at(1));
  return {args, rtn};
}

json overload_to_json(const Overload& x_) { 
  std::vector<json> args;
  for(const auto &v_ : x_.args) { args.emplace_back(Type::any_to_json(v_)); }
  auto rtn = Type::any_to_json(x_.rtn);
  return json::array({args, rtn});
}

Spec::Assert Spec::assert_from_json(const json& j_) { 
  return {};
}

json Spec::assert_to_json(const Spec::Assert& x_) { 
  return json::array({});
}

Spec::GpuBarrierGlobal Spec::gpubarrierglobal_from_json(const json& j_) { 
  return {};
}

json Spec::gpubarrierglobal_to_json(const Spec::GpuBarrierGlobal& x_) { 
  return json::array({});
}

Spec::GpuBarrierLocal Spec::gpubarrierlocal_from_json(const json& j_) { 
  return {};
}

json Spec::gpubarrierlocal_to_json(const Spec::GpuBarrierLocal& x_) { 
  return json::array({});
}

Spec::GpuBarrierAll Spec::gpubarrierall_from_json(const json& j_) { 
  return {};
}

json Spec::gpubarrierall_to_json(const Spec::GpuBarrierAll& x_) { 
  return json::array({});
}

Spec::GpuFenceGlobal Spec::gpufenceglobal_from_json(const json& j_) { 
  return {};
}

json Spec::gpufenceglobal_to_json(const Spec::GpuFenceGlobal& x_) { 
  return json::array({});
}

Spec::GpuFenceLocal Spec::gpufencelocal_from_json(const json& j_) { 
  return {};
}

json Spec::gpufencelocal_to_json(const Spec::GpuFenceLocal& x_) { 
  return json::array({});
}

Spec::GpuFenceAll Spec::gpufenceall_from_json(const json& j_) { 
  return {};
}

json Spec::gpufenceall_to_json(const Spec::GpuFenceAll& x_) { 
  return json::array({});
}

Spec::GpuGlobalIdx Spec::gpuglobalidx_from_json(const json& j_) { 
  auto dim = Expr::any_from_json(j_.at(0));
  return Spec::GpuGlobalIdx(dim);
}

json Spec::gpuglobalidx_to_json(const Spec::GpuGlobalIdx& x_) { 
  auto dim = Expr::any_to_json(x_.dim);
  return json::array({dim});
}

Spec::GpuGlobalSize Spec::gpuglobalsize_from_json(const json& j_) { 
  auto dim = Expr::any_from_json(j_.at(0));
  return Spec::GpuGlobalSize(dim);
}

json Spec::gpuglobalsize_to_json(const Spec::GpuGlobalSize& x_) { 
  auto dim = Expr::any_to_json(x_.dim);
  return json::array({dim});
}

Spec::GpuGroupIdx Spec::gpugroupidx_from_json(const json& j_) { 
  auto dim = Expr::any_from_json(j_.at(0));
  return Spec::GpuGroupIdx(dim);
}

json Spec::gpugroupidx_to_json(const Spec::GpuGroupIdx& x_) { 
  auto dim = Expr::any_to_json(x_.dim);
  return json::array({dim});
}

Spec::GpuGroupSize Spec::gpugroupsize_from_json(const json& j_) { 
  auto dim = Expr::any_from_json(j_.at(0));
  return Spec::GpuGroupSize(dim);
}

json Spec::gpugroupsize_to_json(const Spec::GpuGroupSize& x_) { 
  auto dim = Expr::any_to_json(x_.dim);
  return json::array({dim});
}

Spec::GpuLocalIdx Spec::gpulocalidx_from_json(const json& j_) { 
  auto dim = Expr::any_from_json(j_.at(0));
  return Spec::GpuLocalIdx(dim);
}

json Spec::gpulocalidx_to_json(const Spec::GpuLocalIdx& x_) { 
  auto dim = Expr::any_to_json(x_.dim);
  return json::array({dim});
}

Spec::GpuLocalSize Spec::gpulocalsize_from_json(const json& j_) { 
  auto dim = Expr::any_from_json(j_.at(0));
  return Spec::GpuLocalSize(dim);
}

json Spec::gpulocalsize_to_json(const Spec::GpuLocalSize& x_) { 
  auto dim = Expr::any_to_json(x_.dim);
  return json::array({dim});
}

Spec::Any Spec::any_from_json(const json& j_) { 
  size_t ord_ = j_.at(0).get<size_t>();
  const auto &t_ = j_.at(1);
  switch (ord_) {
  case 0: return Spec::assert_from_json(t_);
  case 1: return Spec::gpubarrierglobal_from_json(t_);
  case 2: return Spec::gpubarrierlocal_from_json(t_);
  case 3: return Spec::gpubarrierall_from_json(t_);
  case 4: return Spec::gpufenceglobal_from_json(t_);
  case 5: return Spec::gpufencelocal_from_json(t_);
  case 6: return Spec::gpufenceall_from_json(t_);
  case 7: return Spec::gpuglobalidx_from_json(t_);
  case 8: return Spec::gpuglobalsize_from_json(t_);
  case 9: return Spec::gpugroupidx_from_json(t_);
  case 10: return Spec::gpugroupsize_from_json(t_);
  case 11: return Spec::gpulocalidx_from_json(t_);
  case 12: return Spec::gpulocalsize_from_json(t_);
  default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
  }
}

json Spec::any_to_json(const Spec::Any& x_) { 
  return x_.match_total(
  [](const Spec::Assert &y_) -> json { return {0, Spec::assert_to_json(y_)}; }
  ,
  [](const Spec::GpuBarrierGlobal &y_) -> json { return {1, Spec::gpubarrierglobal_to_json(y_)}; }
  ,
  [](const Spec::GpuBarrierLocal &y_) -> json { return {2, Spec::gpubarrierlocal_to_json(y_)}; }
  ,
  [](const Spec::GpuBarrierAll &y_) -> json { return {3, Spec::gpubarrierall_to_json(y_)}; }
  ,
  [](const Spec::GpuFenceGlobal &y_) -> json { return {4, Spec::gpufenceglobal_to_json(y_)}; }
  ,
  [](const Spec::GpuFenceLocal &y_) -> json { return {5, Spec::gpufencelocal_to_json(y_)}; }
  ,
  [](const Spec::GpuFenceAll &y_) -> json { return {6, Spec::gpufenceall_to_json(y_)}; }
  ,
  [](const Spec::GpuGlobalIdx &y_) -> json { return {7, Spec::gpuglobalidx_to_json(y_)}; }
  ,
  [](const Spec::GpuGlobalSize &y_) -> json { return {8, Spec::gpuglobalsize_to_json(y_)}; }
  ,
  [](const Spec::GpuGroupIdx &y_) -> json { return {9, Spec::gpugroupidx_to_json(y_)}; }
  ,
  [](const Spec::GpuGroupSize &y_) -> json { return {10, Spec::gpugroupsize_to_json(y_)}; }
  ,
  [](const Spec::GpuLocalIdx &y_) -> json { return {11, Spec::gpulocalidx_to_json(y_)}; }
  ,
  [](const Spec::GpuLocalSize &y_) -> json { return {12, Spec::gpulocalsize_to_json(y_)}; }
  );
}

Intr::BNot Intr::bnot_from_json(const json& j_) { 
  auto x = Expr::any_from_json(j_.at(0));
  auto rtn = Type::any_from_json(j_.at(1));
  return {x, rtn};
}

json Intr::bnot_to_json(const Intr::BNot& x_) { 
  auto x = Expr::any_to_json(x_.x);
  auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, rtn});
}

Intr::LogicNot Intr::logicnot_from_json(const json& j_) { 
  auto x = Expr::any_from_json(j_.at(0));
  return Intr::LogicNot(x);
}

json Intr::logicnot_to_json(const Intr::LogicNot& x_) { 
  auto x = Expr::any_to_json(x_.x);
  return json::array({x});
}

Intr::Pos Intr::pos_from_json(const json& j_) { 
  auto x = Expr::any_from_json(j_.at(0));
  auto rtn = Type::any_from_json(j_.at(1));
  return {x, rtn};
}

json Intr::pos_to_json(const Intr::Pos& x_) { 
  auto x = Expr::any_to_json(x_.x);
  auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, rtn});
}

Intr::Neg Intr::neg_from_json(const json& j_) { 
  auto x = Expr::any_from_json(j_.at(0));
  auto rtn = Type::any_from_json(j_.at(1));
  return {x, rtn};
}

json Intr::neg_to_json(const Intr::Neg& x_) { 
  auto x = Expr::any_to_json(x_.x);
  auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, rtn});
}

Intr::Add Intr::add_from_json(const json& j_) { 
  auto x = Expr::any_from_json(j_.at(0));
  auto y = Expr::any_from_json(j_.at(1));
  auto rtn = Type::any_from_json(j_.at(2));
  return {x, y, rtn};
}

json Intr::add_to_json(const Intr::Add& x_) { 
  auto x = Expr::any_to_json(x_.x);
  auto y = Expr::any_to_json(x_.y);
  auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, y, rtn});
}

Intr::Sub Intr::sub_from_json(const json& j_) { 
  auto x = Expr::any_from_json(j_.at(0));
  auto y = Expr::any_from_json(j_.at(1));
  auto rtn = Type::any_from_json(j_.at(2));
  return {x, y, rtn};
}

json Intr::sub_to_json(const Intr::Sub& x_) { 
  auto x = Expr::any_to_json(x_.x);
  auto y = Expr::any_to_json(x_.y);
  auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, y, rtn});
}

Intr::Mul Intr::mul_from_json(const json& j_) { 
  auto x = Expr::any_from_json(j_.at(0));
  auto y = Expr::any_from_json(j_.at(1));
  auto rtn = Type::any_from_json(j_.at(2));
  return {x, y, rtn};
}

json Intr::mul_to_json(const Intr::Mul& x_) { 
  auto x = Expr::any_to_json(x_.x);
  auto y = Expr::any_to_json(x_.y);
  auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, y, rtn});
}

Intr::Div Intr::div_from_json(const json& j_) { 
  auto x = Expr::any_from_json(j_.at(0));
  auto y = Expr::any_from_json(j_.at(1));
  auto rtn = Type::any_from_json(j_.at(2));
  return {x, y, rtn};
}

json Intr::div_to_json(const Intr::Div& x_) { 
  auto x = Expr::any_to_json(x_.x);
  auto y = Expr::any_to_json(x_.y);
  auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, y, rtn});
}

Intr::Rem Intr::rem_from_json(const json& j_) { 
  auto x = Expr::any_from_json(j_.at(0));
  auto y = Expr::any_from_json(j_.at(1));
  auto rtn = Type::any_from_json(j_.at(2));
  return {x, y, rtn};
}

json Intr::rem_to_json(const Intr::Rem& x_) { 
  auto x = Expr::any_to_json(x_.x);
  auto y = Expr::any_to_json(x_.y);
  auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, y, rtn});
}

Intr::Min Intr::min_from_json(const json& j_) { 
  auto x = Expr::any_from_json(j_.at(0));
  auto y = Expr::any_from_json(j_.at(1));
  auto rtn = Type::any_from_json(j_.at(2));
  return {x, y, rtn};
}

json Intr::min_to_json(const Intr::Min& x_) { 
  auto x = Expr::any_to_json(x_.x);
  auto y = Expr::any_to_json(x_.y);
  auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, y, rtn});
}

Intr::Max Intr::max_from_json(const json& j_) { 
  auto x = Expr::any_from_json(j_.at(0));
  auto y = Expr::any_from_json(j_.at(1));
  auto rtn = Type::any_from_json(j_.at(2));
  return {x, y, rtn};
}

json Intr::max_to_json(const Intr::Max& x_) { 
  auto x = Expr::any_to_json(x_.x);
  auto y = Expr::any_to_json(x_.y);
  auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, y, rtn});
}

Intr::BAnd Intr::band_from_json(const json& j_) { 
  auto x = Expr::any_from_json(j_.at(0));
  auto y = Expr::any_from_json(j_.at(1));
  auto rtn = Type::any_from_json(j_.at(2));
  return {x, y, rtn};
}

json Intr::band_to_json(const Intr::BAnd& x_) { 
  auto x = Expr::any_to_json(x_.x);
  auto y = Expr::any_to_json(x_.y);
  auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, y, rtn});
}

Intr::BOr Intr::bor_from_json(const json& j_) { 
  auto x = Expr::any_from_json(j_.at(0));
  auto y = Expr::any_from_json(j_.at(1));
  auto rtn = Type::any_from_json(j_.at(2));
  return {x, y, rtn};
}

json Intr::bor_to_json(const Intr::BOr& x_) { 
  auto x = Expr::any_to_json(x_.x);
  auto y = Expr::any_to_json(x_.y);
  auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, y, rtn});
}

Intr::BXor Intr::bxor_from_json(const json& j_) { 
  auto x = Expr::any_from_json(j_.at(0));
  auto y = Expr::any_from_json(j_.at(1));
  auto rtn = Type::any_from_json(j_.at(2));
  return {x, y, rtn};
}

json Intr::bxor_to_json(const Intr::BXor& x_) { 
  auto x = Expr::any_to_json(x_.x);
  auto y = Expr::any_to_json(x_.y);
  auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, y, rtn});
}

Intr::BSL Intr::bsl_from_json(const json& j_) { 
  auto x = Expr::any_from_json(j_.at(0));
  auto y = Expr::any_from_json(j_.at(1));
  auto rtn = Type::any_from_json(j_.at(2));
  return {x, y, rtn};
}

json Intr::bsl_to_json(const Intr::BSL& x_) { 
  auto x = Expr::any_to_json(x_.x);
  auto y = Expr::any_to_json(x_.y);
  auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, y, rtn});
}

Intr::BSR Intr::bsr_from_json(const json& j_) { 
  auto x = Expr::any_from_json(j_.at(0));
  auto y = Expr::any_from_json(j_.at(1));
  auto rtn = Type::any_from_json(j_.at(2));
  return {x, y, rtn};
}

json Intr::bsr_to_json(const Intr::BSR& x_) { 
  auto x = Expr::any_to_json(x_.x);
  auto y = Expr::any_to_json(x_.y);
  auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, y, rtn});
}

Intr::BZSR Intr::bzsr_from_json(const json& j_) { 
  auto x = Expr::any_from_json(j_.at(0));
  auto y = Expr::any_from_json(j_.at(1));
  auto rtn = Type::any_from_json(j_.at(2));
  return {x, y, rtn};
}

json Intr::bzsr_to_json(const Intr::BZSR& x_) { 
  auto x = Expr::any_to_json(x_.x);
  auto y = Expr::any_to_json(x_.y);
  auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, y, rtn});
}

Intr::LogicAnd Intr::logicand_from_json(const json& j_) { 
  auto x = Expr::any_from_json(j_.at(0));
  auto y = Expr::any_from_json(j_.at(1));
  return {x, y};
}

json Intr::logicand_to_json(const Intr::LogicAnd& x_) { 
  auto x = Expr::any_to_json(x_.x);
  auto y = Expr::any_to_json(x_.y);
  return json::array({x, y});
}

Intr::LogicOr Intr::logicor_from_json(const json& j_) { 
  auto x = Expr::any_from_json(j_.at(0));
  auto y = Expr::any_from_json(j_.at(1));
  return {x, y};
}

json Intr::logicor_to_json(const Intr::LogicOr& x_) { 
  auto x = Expr::any_to_json(x_.x);
  auto y = Expr::any_to_json(x_.y);
  return json::array({x, y});
}

Intr::LogicEq Intr::logiceq_from_json(const json& j_) { 
  auto x = Expr::any_from_json(j_.at(0));
  auto y = Expr::any_from_json(j_.at(1));
  return {x, y};
}

json Intr::logiceq_to_json(const Intr::LogicEq& x_) { 
  auto x = Expr::any_to_json(x_.x);
  auto y = Expr::any_to_json(x_.y);
  return json::array({x, y});
}

Intr::LogicNeq Intr::logicneq_from_json(const json& j_) { 
  auto x = Expr::any_from_json(j_.at(0));
  auto y = Expr::any_from_json(j_.at(1));
  return {x, y};
}

json Intr::logicneq_to_json(const Intr::LogicNeq& x_) { 
  auto x = Expr::any_to_json(x_.x);
  auto y = Expr::any_to_json(x_.y);
  return json::array({x, y});
}

Intr::LogicLte Intr::logiclte_from_json(const json& j_) { 
  auto x = Expr::any_from_json(j_.at(0));
  auto y = Expr::any_from_json(j_.at(1));
  return {x, y};
}

json Intr::logiclte_to_json(const Intr::LogicLte& x_) { 
  auto x = Expr::any_to_json(x_.x);
  auto y = Expr::any_to_json(x_.y);
  return json::array({x, y});
}

Intr::LogicGte Intr::logicgte_from_json(const json& j_) { 
  auto x = Expr::any_from_json(j_.at(0));
  auto y = Expr::any_from_json(j_.at(1));
  return {x, y};
}

json Intr::logicgte_to_json(const Intr::LogicGte& x_) { 
  auto x = Expr::any_to_json(x_.x);
  auto y = Expr::any_to_json(x_.y);
  return json::array({x, y});
}

Intr::LogicLt Intr::logiclt_from_json(const json& j_) { 
  auto x = Expr::any_from_json(j_.at(0));
  auto y = Expr::any_from_json(j_.at(1));
  return {x, y};
}

json Intr::logiclt_to_json(const Intr::LogicLt& x_) { 
  auto x = Expr::any_to_json(x_.x);
  auto y = Expr::any_to_json(x_.y);
  return json::array({x, y});
}

Intr::LogicGt Intr::logicgt_from_json(const json& j_) { 
  auto x = Expr::any_from_json(j_.at(0));
  auto y = Expr::any_from_json(j_.at(1));
  return {x, y};
}

json Intr::logicgt_to_json(const Intr::LogicGt& x_) { 
  auto x = Expr::any_to_json(x_.x);
  auto y = Expr::any_to_json(x_.y);
  return json::array({x, y});
}

Intr::Any Intr::any_from_json(const json& j_) { 
  size_t ord_ = j_.at(0).get<size_t>();
  const auto &t_ = j_.at(1);
  switch (ord_) {
  case 0: return Intr::bnot_from_json(t_);
  case 1: return Intr::logicnot_from_json(t_);
  case 2: return Intr::pos_from_json(t_);
  case 3: return Intr::neg_from_json(t_);
  case 4: return Intr::add_from_json(t_);
  case 5: return Intr::sub_from_json(t_);
  case 6: return Intr::mul_from_json(t_);
  case 7: return Intr::div_from_json(t_);
  case 8: return Intr::rem_from_json(t_);
  case 9: return Intr::min_from_json(t_);
  case 10: return Intr::max_from_json(t_);
  case 11: return Intr::band_from_json(t_);
  case 12: return Intr::bor_from_json(t_);
  case 13: return Intr::bxor_from_json(t_);
  case 14: return Intr::bsl_from_json(t_);
  case 15: return Intr::bsr_from_json(t_);
  case 16: return Intr::bzsr_from_json(t_);
  case 17: return Intr::logicand_from_json(t_);
  case 18: return Intr::logicor_from_json(t_);
  case 19: return Intr::logiceq_from_json(t_);
  case 20: return Intr::logicneq_from_json(t_);
  case 21: return Intr::logiclte_from_json(t_);
  case 22: return Intr::logicgte_from_json(t_);
  case 23: return Intr::logiclt_from_json(t_);
  case 24: return Intr::logicgt_from_json(t_);
  default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
  }
}

json Intr::any_to_json(const Intr::Any& x_) { 
  return x_.match_total(
  [](const Intr::BNot &y_) -> json { return {0, Intr::bnot_to_json(y_)}; }
  ,
  [](const Intr::LogicNot &y_) -> json { return {1, Intr::logicnot_to_json(y_)}; }
  ,
  [](const Intr::Pos &y_) -> json { return {2, Intr::pos_to_json(y_)}; }
  ,
  [](const Intr::Neg &y_) -> json { return {3, Intr::neg_to_json(y_)}; }
  ,
  [](const Intr::Add &y_) -> json { return {4, Intr::add_to_json(y_)}; }
  ,
  [](const Intr::Sub &y_) -> json { return {5, Intr::sub_to_json(y_)}; }
  ,
  [](const Intr::Mul &y_) -> json { return {6, Intr::mul_to_json(y_)}; }
  ,
  [](const Intr::Div &y_) -> json { return {7, Intr::div_to_json(y_)}; }
  ,
  [](const Intr::Rem &y_) -> json { return {8, Intr::rem_to_json(y_)}; }
  ,
  [](const Intr::Min &y_) -> json { return {9, Intr::min_to_json(y_)}; }
  ,
  [](const Intr::Max &y_) -> json { return {10, Intr::max_to_json(y_)}; }
  ,
  [](const Intr::BAnd &y_) -> json { return {11, Intr::band_to_json(y_)}; }
  ,
  [](const Intr::BOr &y_) -> json { return {12, Intr::bor_to_json(y_)}; }
  ,
  [](const Intr::BXor &y_) -> json { return {13, Intr::bxor_to_json(y_)}; }
  ,
  [](const Intr::BSL &y_) -> json { return {14, Intr::bsl_to_json(y_)}; }
  ,
  [](const Intr::BSR &y_) -> json { return {15, Intr::bsr_to_json(y_)}; }
  ,
  [](const Intr::BZSR &y_) -> json { return {16, Intr::bzsr_to_json(y_)}; }
  ,
  [](const Intr::LogicAnd &y_) -> json { return {17, Intr::logicand_to_json(y_)}; }
  ,
  [](const Intr::LogicOr &y_) -> json { return {18, Intr::logicor_to_json(y_)}; }
  ,
  [](const Intr::LogicEq &y_) -> json { return {19, Intr::logiceq_to_json(y_)}; }
  ,
  [](const Intr::LogicNeq &y_) -> json { return {20, Intr::logicneq_to_json(y_)}; }
  ,
  [](const Intr::LogicLte &y_) -> json { return {21, Intr::logiclte_to_json(y_)}; }
  ,
  [](const Intr::LogicGte &y_) -> json { return {22, Intr::logicgte_to_json(y_)}; }
  ,
  [](const Intr::LogicLt &y_) -> json { return {23, Intr::logiclt_to_json(y_)}; }
  ,
  [](const Intr::LogicGt &y_) -> json { return {24, Intr::logicgt_to_json(y_)}; }
  );
}

Math::Abs Math::abs_from_json(const json& j_) { 
  auto x = Expr::any_from_json(j_.at(0));
  auto rtn = Type::any_from_json(j_.at(1));
  return {x, rtn};
}

json Math::abs_to_json(const Math::Abs& x_) { 
  auto x = Expr::any_to_json(x_.x);
  auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, rtn});
}

Math::Sin Math::sin_from_json(const json& j_) { 
  auto x = Expr::any_from_json(j_.at(0));
  auto rtn = Type::any_from_json(j_.at(1));
  return {x, rtn};
}

json Math::sin_to_json(const Math::Sin& x_) { 
  auto x = Expr::any_to_json(x_.x);
  auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, rtn});
}

Math::Cos Math::cos_from_json(const json& j_) { 
  auto x = Expr::any_from_json(j_.at(0));
  auto rtn = Type::any_from_json(j_.at(1));
  return {x, rtn};
}

json Math::cos_to_json(const Math::Cos& x_) { 
  auto x = Expr::any_to_json(x_.x);
  auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, rtn});
}

Math::Tan Math::tan_from_json(const json& j_) { 
  auto x = Expr::any_from_json(j_.at(0));
  auto rtn = Type::any_from_json(j_.at(1));
  return {x, rtn};
}

json Math::tan_to_json(const Math::Tan& x_) { 
  auto x = Expr::any_to_json(x_.x);
  auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, rtn});
}

Math::Asin Math::asin_from_json(const json& j_) { 
  auto x = Expr::any_from_json(j_.at(0));
  auto rtn = Type::any_from_json(j_.at(1));
  return {x, rtn};
}

json Math::asin_to_json(const Math::Asin& x_) { 
  auto x = Expr::any_to_json(x_.x);
  auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, rtn});
}

Math::Acos Math::acos_from_json(const json& j_) { 
  auto x = Expr::any_from_json(j_.at(0));
  auto rtn = Type::any_from_json(j_.at(1));
  return {x, rtn};
}

json Math::acos_to_json(const Math::Acos& x_) { 
  auto x = Expr::any_to_json(x_.x);
  auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, rtn});
}

Math::Atan Math::atan_from_json(const json& j_) { 
  auto x = Expr::any_from_json(j_.at(0));
  auto rtn = Type::any_from_json(j_.at(1));
  return {x, rtn};
}

json Math::atan_to_json(const Math::Atan& x_) { 
  auto x = Expr::any_to_json(x_.x);
  auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, rtn});
}

Math::Sinh Math::sinh_from_json(const json& j_) { 
  auto x = Expr::any_from_json(j_.at(0));
  auto rtn = Type::any_from_json(j_.at(1));
  return {x, rtn};
}

json Math::sinh_to_json(const Math::Sinh& x_) { 
  auto x = Expr::any_to_json(x_.x);
  auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, rtn});
}

Math::Cosh Math::cosh_from_json(const json& j_) { 
  auto x = Expr::any_from_json(j_.at(0));
  auto rtn = Type::any_from_json(j_.at(1));
  return {x, rtn};
}

json Math::cosh_to_json(const Math::Cosh& x_) { 
  auto x = Expr::any_to_json(x_.x);
  auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, rtn});
}

Math::Tanh Math::tanh_from_json(const json& j_) { 
  auto x = Expr::any_from_json(j_.at(0));
  auto rtn = Type::any_from_json(j_.at(1));
  return {x, rtn};
}

json Math::tanh_to_json(const Math::Tanh& x_) { 
  auto x = Expr::any_to_json(x_.x);
  auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, rtn});
}

Math::Signum Math::signum_from_json(const json& j_) { 
  auto x = Expr::any_from_json(j_.at(0));
  auto rtn = Type::any_from_json(j_.at(1));
  return {x, rtn};
}

json Math::signum_to_json(const Math::Signum& x_) { 
  auto x = Expr::any_to_json(x_.x);
  auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, rtn});
}

Math::Round Math::round_from_json(const json& j_) { 
  auto x = Expr::any_from_json(j_.at(0));
  auto rtn = Type::any_from_json(j_.at(1));
  return {x, rtn};
}

json Math::round_to_json(const Math::Round& x_) { 
  auto x = Expr::any_to_json(x_.x);
  auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, rtn});
}

Math::Ceil Math::ceil_from_json(const json& j_) { 
  auto x = Expr::any_from_json(j_.at(0));
  auto rtn = Type::any_from_json(j_.at(1));
  return {x, rtn};
}

json Math::ceil_to_json(const Math::Ceil& x_) { 
  auto x = Expr::any_to_json(x_.x);
  auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, rtn});
}

Math::Floor Math::floor_from_json(const json& j_) { 
  auto x = Expr::any_from_json(j_.at(0));
  auto rtn = Type::any_from_json(j_.at(1));
  return {x, rtn};
}

json Math::floor_to_json(const Math::Floor& x_) { 
  auto x = Expr::any_to_json(x_.x);
  auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, rtn});
}

Math::Rint Math::rint_from_json(const json& j_) { 
  auto x = Expr::any_from_json(j_.at(0));
  auto rtn = Type::any_from_json(j_.at(1));
  return {x, rtn};
}

json Math::rint_to_json(const Math::Rint& x_) { 
  auto x = Expr::any_to_json(x_.x);
  auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, rtn});
}

Math::Sqrt Math::sqrt_from_json(const json& j_) { 
  auto x = Expr::any_from_json(j_.at(0));
  auto rtn = Type::any_from_json(j_.at(1));
  return {x, rtn};
}

json Math::sqrt_to_json(const Math::Sqrt& x_) { 
  auto x = Expr::any_to_json(x_.x);
  auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, rtn});
}

Math::Cbrt Math::cbrt_from_json(const json& j_) { 
  auto x = Expr::any_from_json(j_.at(0));
  auto rtn = Type::any_from_json(j_.at(1));
  return {x, rtn};
}

json Math::cbrt_to_json(const Math::Cbrt& x_) { 
  auto x = Expr::any_to_json(x_.x);
  auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, rtn});
}

Math::Exp Math::exp_from_json(const json& j_) { 
  auto x = Expr::any_from_json(j_.at(0));
  auto rtn = Type::any_from_json(j_.at(1));
  return {x, rtn};
}

json Math::exp_to_json(const Math::Exp& x_) { 
  auto x = Expr::any_to_json(x_.x);
  auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, rtn});
}

Math::Expm1 Math::expm1_from_json(const json& j_) { 
  auto x = Expr::any_from_json(j_.at(0));
  auto rtn = Type::any_from_json(j_.at(1));
  return {x, rtn};
}

json Math::expm1_to_json(const Math::Expm1& x_) { 
  auto x = Expr::any_to_json(x_.x);
  auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, rtn});
}

Math::Log Math::log_from_json(const json& j_) { 
  auto x = Expr::any_from_json(j_.at(0));
  auto rtn = Type::any_from_json(j_.at(1));
  return {x, rtn};
}

json Math::log_to_json(const Math::Log& x_) { 
  auto x = Expr::any_to_json(x_.x);
  auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, rtn});
}

Math::Log1p Math::log1p_from_json(const json& j_) { 
  auto x = Expr::any_from_json(j_.at(0));
  auto rtn = Type::any_from_json(j_.at(1));
  return {x, rtn};
}

json Math::log1p_to_json(const Math::Log1p& x_) { 
  auto x = Expr::any_to_json(x_.x);
  auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, rtn});
}

Math::Log10 Math::log10_from_json(const json& j_) { 
  auto x = Expr::any_from_json(j_.at(0));
  auto rtn = Type::any_from_json(j_.at(1));
  return {x, rtn};
}

json Math::log10_to_json(const Math::Log10& x_) { 
  auto x = Expr::any_to_json(x_.x);
  auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, rtn});
}

Math::Pow Math::pow_from_json(const json& j_) { 
  auto x = Expr::any_from_json(j_.at(0));
  auto y = Expr::any_from_json(j_.at(1));
  auto rtn = Type::any_from_json(j_.at(2));
  return {x, y, rtn};
}

json Math::pow_to_json(const Math::Pow& x_) { 
  auto x = Expr::any_to_json(x_.x);
  auto y = Expr::any_to_json(x_.y);
  auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, y, rtn});
}

Math::Atan2 Math::atan2_from_json(const json& j_) { 
  auto x = Expr::any_from_json(j_.at(0));
  auto y = Expr::any_from_json(j_.at(1));
  auto rtn = Type::any_from_json(j_.at(2));
  return {x, y, rtn};
}

json Math::atan2_to_json(const Math::Atan2& x_) { 
  auto x = Expr::any_to_json(x_.x);
  auto y = Expr::any_to_json(x_.y);
  auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, y, rtn});
}

Math::Hypot Math::hypot_from_json(const json& j_) { 
  auto x = Expr::any_from_json(j_.at(0));
  auto y = Expr::any_from_json(j_.at(1));
  auto rtn = Type::any_from_json(j_.at(2));
  return {x, y, rtn};
}

json Math::hypot_to_json(const Math::Hypot& x_) { 
  auto x = Expr::any_to_json(x_.x);
  auto y = Expr::any_to_json(x_.y);
  auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, y, rtn});
}

Math::Any Math::any_from_json(const json& j_) { 
  size_t ord_ = j_.at(0).get<size_t>();
  const auto &t_ = j_.at(1);
  switch (ord_) {
  case 0: return Math::abs_from_json(t_);
  case 1: return Math::sin_from_json(t_);
  case 2: return Math::cos_from_json(t_);
  case 3: return Math::tan_from_json(t_);
  case 4: return Math::asin_from_json(t_);
  case 5: return Math::acos_from_json(t_);
  case 6: return Math::atan_from_json(t_);
  case 7: return Math::sinh_from_json(t_);
  case 8: return Math::cosh_from_json(t_);
  case 9: return Math::tanh_from_json(t_);
  case 10: return Math::signum_from_json(t_);
  case 11: return Math::round_from_json(t_);
  case 12: return Math::ceil_from_json(t_);
  case 13: return Math::floor_from_json(t_);
  case 14: return Math::rint_from_json(t_);
  case 15: return Math::sqrt_from_json(t_);
  case 16: return Math::cbrt_from_json(t_);
  case 17: return Math::exp_from_json(t_);
  case 18: return Math::expm1_from_json(t_);
  case 19: return Math::log_from_json(t_);
  case 20: return Math::log1p_from_json(t_);
  case 21: return Math::log10_from_json(t_);
  case 22: return Math::pow_from_json(t_);
  case 23: return Math::atan2_from_json(t_);
  case 24: return Math::hypot_from_json(t_);
  default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
  }
}

json Math::any_to_json(const Math::Any& x_) { 
  return x_.match_total(
  [](const Math::Abs &y_) -> json { return {0, Math::abs_to_json(y_)}; }
  ,
  [](const Math::Sin &y_) -> json { return {1, Math::sin_to_json(y_)}; }
  ,
  [](const Math::Cos &y_) -> json { return {2, Math::cos_to_json(y_)}; }
  ,
  [](const Math::Tan &y_) -> json { return {3, Math::tan_to_json(y_)}; }
  ,
  [](const Math::Asin &y_) -> json { return {4, Math::asin_to_json(y_)}; }
  ,
  [](const Math::Acos &y_) -> json { return {5, Math::acos_to_json(y_)}; }
  ,
  [](const Math::Atan &y_) -> json { return {6, Math::atan_to_json(y_)}; }
  ,
  [](const Math::Sinh &y_) -> json { return {7, Math::sinh_to_json(y_)}; }
  ,
  [](const Math::Cosh &y_) -> json { return {8, Math::cosh_to_json(y_)}; }
  ,
  [](const Math::Tanh &y_) -> json { return {9, Math::tanh_to_json(y_)}; }
  ,
  [](const Math::Signum &y_) -> json { return {10, Math::signum_to_json(y_)}; }
  ,
  [](const Math::Round &y_) -> json { return {11, Math::round_to_json(y_)}; }
  ,
  [](const Math::Ceil &y_) -> json { return {12, Math::ceil_to_json(y_)}; }
  ,
  [](const Math::Floor &y_) -> json { return {13, Math::floor_to_json(y_)}; }
  ,
  [](const Math::Rint &y_) -> json { return {14, Math::rint_to_json(y_)}; }
  ,
  [](const Math::Sqrt &y_) -> json { return {15, Math::sqrt_to_json(y_)}; }
  ,
  [](const Math::Cbrt &y_) -> json { return {16, Math::cbrt_to_json(y_)}; }
  ,
  [](const Math::Exp &y_) -> json { return {17, Math::exp_to_json(y_)}; }
  ,
  [](const Math::Expm1 &y_) -> json { return {18, Math::expm1_to_json(y_)}; }
  ,
  [](const Math::Log &y_) -> json { return {19, Math::log_to_json(y_)}; }
  ,
  [](const Math::Log1p &y_) -> json { return {20, Math::log1p_to_json(y_)}; }
  ,
  [](const Math::Log10 &y_) -> json { return {21, Math::log10_to_json(y_)}; }
  ,
  [](const Math::Pow &y_) -> json { return {22, Math::pow_to_json(y_)}; }
  ,
  [](const Math::Atan2 &y_) -> json { return {23, Math::atan2_to_json(y_)}; }
  ,
  [](const Math::Hypot &y_) -> json { return {24, Math::hypot_to_json(y_)}; }
  );
}

Stmt::Block Stmt::block_from_json(const json& j_) { 
  std::vector<Stmt::Any> stmts;
  for(const auto &v_ : j_.at(0)) { stmts.emplace_back(Stmt::any_from_json(v_)); }
  return Stmt::Block(stmts);
}

json Stmt::block_to_json(const Stmt::Block& x_) { 
  std::vector<json> stmts;
  for(const auto &v_ : x_.stmts) { stmts.emplace_back(Stmt::any_to_json(v_)); }
  return json::array({stmts});
}

Stmt::Comment Stmt::comment_from_json(const json& j_) { 
  auto value = j_.at(0).get<std::string>();
  return Stmt::Comment(value);
}

json Stmt::comment_to_json(const Stmt::Comment& x_) { 
  auto value = x_.value;
  return json::array({value});
}

Stmt::Var Stmt::var_from_json(const json& j_) { 
  auto name = named_from_json(j_.at(0));
  auto expr = j_.at(1).is_null() ? std::nullopt : std::make_optional(Expr::any_from_json(j_.at(1)));
  return {name, expr};
}

json Stmt::var_to_json(const Stmt::Var& x_) { 
  auto name = named_to_json(x_.name);
  auto expr = x_.expr ? Expr::any_to_json(*x_.expr) : json();
  return json::array({name, expr});
}

Stmt::Mut Stmt::mut_from_json(const json& j_) { 
  auto name = Expr::any_from_json(j_.at(0));
  auto expr = Expr::any_from_json(j_.at(1));
  return {name, expr};
}

json Stmt::mut_to_json(const Stmt::Mut& x_) { 
  auto name = Expr::any_to_json(x_.name);
  auto expr = Expr::any_to_json(x_.expr);
  return json::array({name, expr});
}

Stmt::Update Stmt::update_from_json(const json& j_) { 
  auto lhs = Expr::any_from_json(j_.at(0));
  auto idx = Expr::any_from_json(j_.at(1));
  auto value = Expr::any_from_json(j_.at(2));
  return {lhs, idx, value};
}

json Stmt::update_to_json(const Stmt::Update& x_) { 
  auto lhs = Expr::any_to_json(x_.lhs);
  auto idx = Expr::any_to_json(x_.idx);
  auto value = Expr::any_to_json(x_.value);
  return json::array({lhs, idx, value});
}

Stmt::While Stmt::while_from_json(const json& j_) { 
  std::vector<Stmt::Any> tests;
  for(const auto &v_ : j_.at(0)) { tests.emplace_back(Stmt::any_from_json(v_)); }
  auto cond = Expr::any_from_json(j_.at(1));
  std::vector<Stmt::Any> body;
  for(const auto &v_ : j_.at(2)) { body.emplace_back(Stmt::any_from_json(v_)); }
  return {tests, cond, body};
}

json Stmt::while_to_json(const Stmt::While& x_) { 
  std::vector<json> tests;
  for(const auto &v_ : x_.tests) { tests.emplace_back(Stmt::any_to_json(v_)); }
  auto cond = Expr::any_to_json(x_.cond);
  std::vector<json> body;
  for(const auto &v_ : x_.body) { body.emplace_back(Stmt::any_to_json(v_)); }
  return json::array({tests, cond, body});
}

Stmt::ForRange Stmt::forrange_from_json(const json& j_) { 
  auto induction = Expr::select_from_json(j_.at(0));
  auto lbIncl = Expr::any_from_json(j_.at(1));
  auto ubExcl = Expr::any_from_json(j_.at(2));
  auto step = Expr::any_from_json(j_.at(3));
  std::vector<Stmt::Any> body;
  for(const auto &v_ : j_.at(4)) { body.emplace_back(Stmt::any_from_json(v_)); }
  return {induction, lbIncl, ubExcl, step, body};
}

json Stmt::forrange_to_json(const Stmt::ForRange& x_) { 
  auto induction = Expr::select_to_json(x_.induction);
  auto lbIncl = Expr::any_to_json(x_.lbIncl);
  auto ubExcl = Expr::any_to_json(x_.ubExcl);
  auto step = Expr::any_to_json(x_.step);
  std::vector<json> body;
  for(const auto &v_ : x_.body) { body.emplace_back(Stmt::any_to_json(v_)); }
  return json::array({induction, lbIncl, ubExcl, step, body});
}

Stmt::Break Stmt::break_from_json(const json& j_) { 
  return {};
}

json Stmt::break_to_json(const Stmt::Break& x_) { 
  return json::array({});
}

Stmt::Cont Stmt::cont_from_json(const json& j_) { 
  return {};
}

json Stmt::cont_to_json(const Stmt::Cont& x_) { 
  return json::array({});
}

Stmt::Cond Stmt::cond_from_json(const json& j_) { 
  auto cond = Expr::any_from_json(j_.at(0));
  std::vector<Stmt::Any> trueBr;
  for(const auto &v_ : j_.at(1)) { trueBr.emplace_back(Stmt::any_from_json(v_)); }
  std::vector<Stmt::Any> falseBr;
  for(const auto &v_ : j_.at(2)) { falseBr.emplace_back(Stmt::any_from_json(v_)); }
  return {cond, trueBr, falseBr};
}

json Stmt::cond_to_json(const Stmt::Cond& x_) { 
  auto cond = Expr::any_to_json(x_.cond);
  std::vector<json> trueBr;
  for(const auto &v_ : x_.trueBr) { trueBr.emplace_back(Stmt::any_to_json(v_)); }
  std::vector<json> falseBr;
  for(const auto &v_ : x_.falseBr) { falseBr.emplace_back(Stmt::any_to_json(v_)); }
  return json::array({cond, trueBr, falseBr});
}

Stmt::Return Stmt::return_from_json(const json& j_) { 
  auto value = Expr::any_from_json(j_.at(0));
  return Stmt::Return(value);
}

json Stmt::return_to_json(const Stmt::Return& x_) { 
  auto value = Expr::any_to_json(x_.value);
  return json::array({value});
}

Stmt::Annotated Stmt::annotated_from_json(const json& j_) { 
  auto stmt = Stmt::any_from_json(j_.at(0));
  auto pos = j_.at(1).is_null() ? std::nullopt : std::make_optional(sourceposition_from_json(j_.at(1)));
  auto comment = j_.at(2).is_null() ? std::nullopt : std::make_optional(j_.at(2).get<std::string>());
  return {stmt, pos, comment};
}

json Stmt::annotated_to_json(const Stmt::Annotated& x_) { 
  auto stmt = Stmt::any_to_json(x_.stmt);
  auto pos = x_.pos ? sourceposition_to_json(*x_.pos) : json();
  auto comment = x_.comment ? json(*x_.comment) : json();
  return json::array({stmt, pos, comment});
}

Stmt::Any Stmt::any_from_json(const json& j_) { 
  size_t ord_ = j_.at(0).get<size_t>();
  const auto &t_ = j_.at(1);
  switch (ord_) {
  case 0: return Stmt::block_from_json(t_);
  case 1: return Stmt::comment_from_json(t_);
  case 2: return Stmt::var_from_json(t_);
  case 3: return Stmt::mut_from_json(t_);
  case 4: return Stmt::update_from_json(t_);
  case 5: return Stmt::while_from_json(t_);
  case 6: return Stmt::forrange_from_json(t_);
  case 7: return Stmt::break_from_json(t_);
  case 8: return Stmt::cont_from_json(t_);
  case 9: return Stmt::cond_from_json(t_);
  case 10: return Stmt::return_from_json(t_);
  case 11: return Stmt::annotated_from_json(t_);
  default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
  }
}

json Stmt::any_to_json(const Stmt::Any& x_) { 
  return x_.match_total(
  [](const Stmt::Block &y_) -> json { return {0, Stmt::block_to_json(y_)}; }
  ,
  [](const Stmt::Comment &y_) -> json { return {1, Stmt::comment_to_json(y_)}; }
  ,
  [](const Stmt::Var &y_) -> json { return {2, Stmt::var_to_json(y_)}; }
  ,
  [](const Stmt::Mut &y_) -> json { return {3, Stmt::mut_to_json(y_)}; }
  ,
  [](const Stmt::Update &y_) -> json { return {4, Stmt::update_to_json(y_)}; }
  ,
  [](const Stmt::While &y_) -> json { return {5, Stmt::while_to_json(y_)}; }
  ,
  [](const Stmt::ForRange &y_) -> json { return {6, Stmt::forrange_to_json(y_)}; }
  ,
  [](const Stmt::Break &y_) -> json { return {7, Stmt::break_to_json(y_)}; }
  ,
  [](const Stmt::Cont &y_) -> json { return {8, Stmt::cont_to_json(y_)}; }
  ,
  [](const Stmt::Cond &y_) -> json { return {9, Stmt::cond_to_json(y_)}; }
  ,
  [](const Stmt::Return &y_) -> json { return {10, Stmt::return_to_json(y_)}; }
  ,
  [](const Stmt::Annotated &y_) -> json { return {11, Stmt::annotated_to_json(y_)}; }
  );
}

Signature signature_from_json(const json& j_) { 
  auto name = j_.at(0).get<std::string>();
  std::vector<Type::Any> args;
  for(const auto &v_ : j_.at(1)) { args.emplace_back(Type::any_from_json(v_)); }
  auto rtn = Type::any_from_json(j_.at(2));
  return {name, args, rtn};
}

json signature_to_json(const Signature& x_) { 
  auto name = x_.name;
  std::vector<json> args;
  for(const auto &v_ : x_.args) { args.emplace_back(Type::any_to_json(v_)); }
  auto rtn = Type::any_to_json(x_.rtn);
  return json::array({name, args, rtn});
}

FunctionAttr::Internal FunctionAttr::internal_from_json(const json& j_) { 
  return {};
}

json FunctionAttr::internal_to_json(const FunctionAttr::Internal& x_) { 
  return json::array({});
}

FunctionAttr::Exported FunctionAttr::exported_from_json(const json& j_) { 
  return {};
}

json FunctionAttr::exported_to_json(const FunctionAttr::Exported& x_) { 
  return json::array({});
}

FunctionAttr::FPRelaxed FunctionAttr::fprelaxed_from_json(const json& j_) { 
  return {};
}

json FunctionAttr::fprelaxed_to_json(const FunctionAttr::FPRelaxed& x_) { 
  return json::array({});
}

FunctionAttr::FPStrict FunctionAttr::fpstrict_from_json(const json& j_) { 
  return {};
}

json FunctionAttr::fpstrict_to_json(const FunctionAttr::FPStrict& x_) { 
  return json::array({});
}

FunctionAttr::Entry FunctionAttr::entry_from_json(const json& j_) { 
  return {};
}

json FunctionAttr::entry_to_json(const FunctionAttr::Entry& x_) { 
  return json::array({});
}

FunctionAttr::Any FunctionAttr::any_from_json(const json& j_) { 
  size_t ord_ = j_.at(0).get<size_t>();
  const auto &t_ = j_.at(1);
  switch (ord_) {
  case 0: return FunctionAttr::internal_from_json(t_);
  case 1: return FunctionAttr::exported_from_json(t_);
  case 2: return FunctionAttr::fprelaxed_from_json(t_);
  case 3: return FunctionAttr::fpstrict_from_json(t_);
  case 4: return FunctionAttr::entry_from_json(t_);
  default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
  }
}

json FunctionAttr::any_to_json(const FunctionAttr::Any& x_) { 
  return x_.match_total(
  [](const FunctionAttr::Internal &y_) -> json { return {0, FunctionAttr::internal_to_json(y_)}; }
  ,
  [](const FunctionAttr::Exported &y_) -> json { return {1, FunctionAttr::exported_to_json(y_)}; }
  ,
  [](const FunctionAttr::FPRelaxed &y_) -> json { return {2, FunctionAttr::fprelaxed_to_json(y_)}; }
  ,
  [](const FunctionAttr::FPStrict &y_) -> json { return {3, FunctionAttr::fpstrict_to_json(y_)}; }
  ,
  [](const FunctionAttr::Entry &y_) -> json { return {4, FunctionAttr::entry_to_json(y_)}; }
  );
}

Arg arg_from_json(const json& j_) { 
  auto named = named_from_json(j_.at(0));
  auto pos = j_.at(1).is_null() ? std::nullopt : std::make_optional(sourceposition_from_json(j_.at(1)));
  return {named, pos};
}

json arg_to_json(const Arg& x_) { 
  auto named = named_to_json(x_.named);
  auto pos = x_.pos ? sourceposition_to_json(*x_.pos) : json();
  return json::array({named, pos});
}

Function function_from_json(const json& j_) { 
  auto name = j_.at(0).get<std::string>();
  std::vector<Arg> args;
  for(const auto &v_ : j_.at(1)) { args.emplace_back(arg_from_json(v_)); }
  auto rtn = Type::any_from_json(j_.at(2));
  std::vector<Stmt::Any> body;
  for(const auto &v_ : j_.at(3)) { body.emplace_back(Stmt::any_from_json(v_)); }
  std::set<FunctionAttr::Any> attrs;
  for(const auto &v_ : j_.at(4)) { attrs.emplace(FunctionAttr::any_from_json(v_)); }
  return {name, args, rtn, body, attrs};
}

json function_to_json(const Function& x_) { 
  auto name = x_.name;
  std::vector<json> args;
  for(const auto &v_ : x_.args) { args.emplace_back(arg_to_json(v_)); }
  auto rtn = Type::any_to_json(x_.rtn);
  std::vector<json> body;
  for(const auto &v_ : x_.body) { body.emplace_back(Stmt::any_to_json(v_)); }
  std::vector<json> attrs;
  for(const auto &v_ : x_.attrs) { attrs.emplace_back(FunctionAttr::any_to_json(v_)); }
  return json::array({name, args, rtn, body, attrs});
}

StructDef structdef_from_json(const json& j_) { 
  auto name = j_.at(0).get<std::string>();
  std::vector<Named> members;
  for(const auto &v_ : j_.at(1)) { members.emplace_back(named_from_json(v_)); }
  return {name, members};
}

json structdef_to_json(const StructDef& x_) { 
  auto name = x_.name;
  std::vector<json> members;
  for(const auto &v_ : x_.members) { members.emplace_back(named_to_json(v_)); }
  return json::array({name, members});
}

Program program_from_json(const json& j_) { 
  std::vector<StructDef> structs;
  for(const auto &v_ : j_.at(0)) { structs.emplace_back(structdef_from_json(v_)); }
  std::vector<Function> functions;
  for(const auto &v_ : j_.at(1)) { functions.emplace_back(function_from_json(v_)); }
  return {structs, functions};
}

json program_to_json(const Program& x_) { 
  std::vector<json> structs;
  for(const auto &v_ : x_.structs) { structs.emplace_back(structdef_to_json(v_)); }
  std::vector<json> functions;
  for(const auto &v_ : x_.functions) { functions.emplace_back(function_to_json(v_)); }
  return json::array({structs, functions});
}

StructLayoutMember structlayoutmember_from_json(const json& j_) { 
  auto name = named_from_json(j_.at(0));
  auto offsetInBytes = j_.at(1).get<int64_t>();
  auto sizeInBytes = j_.at(2).get<int64_t>();
  return {name, offsetInBytes, sizeInBytes};
}

json structlayoutmember_to_json(const StructLayoutMember& x_) { 
  auto name = named_to_json(x_.name);
  auto offsetInBytes = x_.offsetInBytes;
  auto sizeInBytes = x_.sizeInBytes;
  return json::array({name, offsetInBytes, sizeInBytes});
}

StructLayout structlayout_from_json(const json& j_) { 
  auto name = j_.at(0).get<std::string>();
  auto sizeInBytes = j_.at(1).get<int64_t>();
  auto alignment = j_.at(2).get<int64_t>();
  std::vector<StructLayoutMember> members;
  for(const auto &v_ : j_.at(3)) { members.emplace_back(structlayoutmember_from_json(v_)); }
  return {name, sizeInBytes, alignment, members};
}

json structlayout_to_json(const StructLayout& x_) { 
  auto name = x_.name;
  auto sizeInBytes = x_.sizeInBytes;
  auto alignment = x_.alignment;
  std::vector<json> members;
  for(const auto &v_ : x_.members) { members.emplace_back(structlayoutmember_to_json(v_)); }
  return json::array({name, sizeInBytes, alignment, members});
}

CompileEvent compileevent_from_json(const json& j_) { 
  auto epochMillis = j_.at(0).get<int64_t>();
  auto elapsedNanos = j_.at(1).get<int64_t>();
  auto name = j_.at(2).get<std::string>();
  auto data = j_.at(3).get<std::string>();
  return {epochMillis, elapsedNanos, name, data};
}

json compileevent_to_json(const CompileEvent& x_) { 
  auto epochMillis = x_.epochMillis;
  auto elapsedNanos = x_.elapsedNanos;
  auto name = x_.name;
  auto data = x_.data;
  return json::array({epochMillis, elapsedNanos, name, data});
}

CompileResult compileresult_from_json(const json& j_) { 
  auto binary = j_.at(0).is_null() ? std::nullopt : std::make_optional(j_.at(0).get<std::vector<int8_t>>());
  auto features = j_.at(1).get<std::vector<std::string>>();
  std::vector<CompileEvent> events;
  for(const auto &v_ : j_.at(2)) { events.emplace_back(compileevent_from_json(v_)); }
  std::vector<StructLayout> layouts;
  for(const auto &v_ : j_.at(3)) { layouts.emplace_back(structlayout_from_json(v_)); }
  auto messages = j_.at(4).get<std::string>();
  return {binary, features, events, layouts, messages};
}

json compileresult_to_json(const CompileResult& x_) { 
  auto binary = x_.binary ? json(*x_.binary) : json();
  auto features = x_.features;
  std::vector<json> events;
  for(const auto &v_ : x_.events) { events.emplace_back(compileevent_to_json(v_)); }
  std::vector<json> layouts;
  for(const auto &v_ : x_.layouts) { layouts.emplace_back(structlayout_to_json(v_)); }
  auto messages = x_.messages;
  return json::array({binary, features, events, layouts, messages});
}
json hashed_from_json(const json& j_) { 
  auto hash_ = j_.at(0).get<std::string>();
  auto data_ = j_.at(1);
  if(hash_ != "4e302bec836097cb30370f46e9a85c5b") {
   throw std::runtime_error("Expecting ADT hash to be 4e302bec836097cb30370f46e9a85c5b, but was " + hash_);
  }
  return data_;
}

json hashed_to_json(const json& x_) { 
  return json::array({"4e302bec836097cb30370f46e9a85c5b", x_});
}
} // namespace polyregion::polyast
