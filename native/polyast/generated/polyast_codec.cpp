#include "polyast_codec.h"

#include "msgpack.hpp"

namespace polyregion::polyast {
constexpr auto AdtHash = "2ed29ea38d8b2713f746993b52e0949c";
constexpr auto ProgramHash = "f62204b887e77b014700a7b3a718f079";
constexpr auto PackageHash = "cea6ab38193bdb45ffe1d89f3bb37215";
constexpr auto PackageWireHash = "51add37508362a9652dd5fee3f607e4e";
constexpr auto CompileWireHash = "761bad2d8b6004abf355f5063d0116a9";
using msgpack::decodeMaybeInterned;
using msgpack::encodeInterned;
using msgpack::MsgpackReader;
using msgpack::MsgpackWriter;

Sym sym_from_json(const json &j_) {
  auto fqn = j_.at(0).get<std::vector<std::string>>();
  return Sym(std::move(fqn));
}

json sym_to_json(const Sym &x_) {
  const auto fqn = x_.fqn;
  return json::array({fqn});
}

SourcePosition sourceposition_from_json(const json &j_) {
  auto file = j_.at(0).get<std::string>();
  const auto line = j_.at(1).get<int32_t>();
  const auto col = j_.at(2).is_null() ? std::nullopt : std::make_optional(j_.at(2).get<int32_t>());
  return {std::move(file), line, col};
}

json sourceposition_to_json(const SourcePosition &x_) {
  const auto file = x_.file;
  const auto line = x_.line;
  const auto col = x_.col ? json(*x_.col) : json();
  return json::array({file, line, col});
}

Origin origin_from_json(const json &j_) {
  auto pos = j_.at(0).is_null() ? std::nullopt : std::make_optional(sourceposition_from_json(j_.at(0)));
  auto source = j_.at(1).is_null() ? std::nullopt : std::make_optional(j_.at(1).get<std::string>());
  auto inlinedFrom = j_.at(2).is_null() ? std::nullopt : std::make_optional(sym_from_json(j_.at(2)));
  return {std::move(pos), std::move(source), std::move(inlinedFrom)};
}

json origin_to_json(const Origin &x_) {
  const auto pos = x_.pos ? sourceposition_to_json(*x_.pos) : json();
  const auto source = x_.source ? json(*x_.source) : json();
  const auto inlinedFrom = x_.inlinedFrom ? sym_to_json(*x_.inlinedFrom) : json();
  return json::array({pos, source, inlinedFrom});
}

Named named_from_json(const json &j_) {
  auto symbol = j_.at(0).get<std::string>();
  auto tpe = Type::any_from_json(j_.at(1));
  auto origin = origin_from_json(j_.at(2));
  return {std::move(symbol), std::move(tpe), std::move(origin)};
}

json named_to_json(const Named &x_) {
  const auto symbol = x_.symbol;
  const auto tpe = Type::any_to_json(x_.tpe);
  const auto origin = origin_to_json(x_.origin);
  return json::array({symbol, tpe, origin});
}

TypeKind::None TypeKind::none_from_json(const json &j_) { return {}; }

json TypeKind::none_to_json(const TypeKind::None &x_) { return json::array({}); }

TypeKind::Ref TypeKind::ref_from_json(const json &j_) { return {}; }

json TypeKind::ref_to_json(const TypeKind::Ref &x_) { return json::array({}); }

TypeKind::Integral TypeKind::integral_from_json(const json &j_) { return {}; }

json TypeKind::integral_to_json(const TypeKind::Integral &x_) { return json::array({}); }

TypeKind::Fractional TypeKind::fractional_from_json(const json &j_) { return {}; }

json TypeKind::fractional_to_json(const TypeKind::Fractional &x_) { return json::array({}); }

TypeKind::Any TypeKind::any_from_json(const json &j_) {
  const auto ord_ = j_.at(0).get<size_t>();
  const auto &t_ = j_.at(1);
  switch (ord_) {
    case 0: return TypeKind::none_from_json(t_);
    case 1: return TypeKind::ref_from_json(t_);
    case 2: return TypeKind::integral_from_json(t_);
    case 3: return TypeKind::fractional_from_json(t_);
    default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
  }
}

json TypeKind::any_to_json(const TypeKind::Any &x_) {
  return x_.match_total([](const TypeKind::None &y_) -> json { return {0, TypeKind::none_to_json(y_)}; },
                        [](const TypeKind::Ref &y_) -> json { return {1, TypeKind::ref_to_json(y_)}; },
                        [](const TypeKind::Integral &y_) -> json { return {2, TypeKind::integral_to_json(y_)}; },
                        [](const TypeKind::Fractional &y_) -> json { return {3, TypeKind::fractional_to_json(y_)}; });
}

TypeSpace::Global TypeSpace::global_from_json(const json &j_) { return {}; }

json TypeSpace::global_to_json(const TypeSpace::Global &x_) { return json::array({}); }

TypeSpace::Local TypeSpace::local_from_json(const json &j_) { return {}; }

json TypeSpace::local_to_json(const TypeSpace::Local &x_) { return json::array({}); }

TypeSpace::Private TypeSpace::private_from_json(const json &j_) { return {}; }

json TypeSpace::private_to_json(const TypeSpace::Private &x_) { return json::array({}); }

TypeSpace::Constant TypeSpace::constant_from_json(const json &j_) { return {}; }

json TypeSpace::constant_to_json(const TypeSpace::Constant &x_) { return json::array({}); }

TypeSpace::Any TypeSpace::any_from_json(const json &j_) {
  const auto ord_ = j_.at(0).get<size_t>();
  const auto &t_ = j_.at(1);
  switch (ord_) {
    case 0: return TypeSpace::global_from_json(t_);
    case 1: return TypeSpace::local_from_json(t_);
    case 2: return TypeSpace::private_from_json(t_);
    case 3: return TypeSpace::constant_from_json(t_);
    default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
  }
}

json TypeSpace::any_to_json(const TypeSpace::Any &x_) {
  return x_.match_total([](const TypeSpace::Global &y_) -> json { return {0, TypeSpace::global_to_json(y_)}; },
                        [](const TypeSpace::Local &y_) -> json { return {1, TypeSpace::local_to_json(y_)}; },
                        [](const TypeSpace::Private &y_) -> json { return {2, TypeSpace::private_to_json(y_)}; },
                        [](const TypeSpace::Constant &y_) -> json { return {3, TypeSpace::constant_to_json(y_)}; });
}

Type::Float16 Type::float16_from_json(const json &j_) { return {}; }

json Type::float16_to_json(const Type::Float16 &x_) { return json::array({}); }

Type::Float32 Type::float32_from_json(const json &j_) { return {}; }

json Type::float32_to_json(const Type::Float32 &x_) { return json::array({}); }

Type::Float64 Type::float64_from_json(const json &j_) { return {}; }

json Type::float64_to_json(const Type::Float64 &x_) { return json::array({}); }

Type::IntU8 Type::intu8_from_json(const json &j_) { return {}; }

json Type::intu8_to_json(const Type::IntU8 &x_) { return json::array({}); }

Type::IntU16 Type::intu16_from_json(const json &j_) { return {}; }

json Type::intu16_to_json(const Type::IntU16 &x_) { return json::array({}); }

Type::IntU32 Type::intu32_from_json(const json &j_) { return {}; }

json Type::intu32_to_json(const Type::IntU32 &x_) { return json::array({}); }

Type::IntU64 Type::intu64_from_json(const json &j_) { return {}; }

json Type::intu64_to_json(const Type::IntU64 &x_) { return json::array({}); }

Type::IntS8 Type::ints8_from_json(const json &j_) { return {}; }

json Type::ints8_to_json(const Type::IntS8 &x_) { return json::array({}); }

Type::IntS16 Type::ints16_from_json(const json &j_) { return {}; }

json Type::ints16_to_json(const Type::IntS16 &x_) { return json::array({}); }

Type::IntS32 Type::ints32_from_json(const json &j_) { return {}; }

json Type::ints32_to_json(const Type::IntS32 &x_) { return json::array({}); }

Type::IntS64 Type::ints64_from_json(const json &j_) { return {}; }

json Type::ints64_to_json(const Type::IntS64 &x_) { return json::array({}); }

Type::Nothing Type::nothing_from_json(const json &j_) { return {}; }

json Type::nothing_to_json(const Type::Nothing &x_) { return json::array({}); }

Type::Unit0 Type::unit0_from_json(const json &j_) { return {}; }

json Type::unit0_to_json(const Type::Unit0 &x_) { return json::array({}); }

Type::Bool1 Type::bool1_from_json(const json &j_) { return {}; }

json Type::bool1_to_json(const Type::Bool1 &x_) { return json::array({}); }

Type::Struct Type::struct_from_json(const json &j_) {
  auto name = sym_from_json(j_.at(0));
  std::vector<Type::Any> args;
  args.reserve(j_.at(1).size());
  for (const auto &v_ : j_.at(1)) {
    args.emplace_back(Type::any_from_json(v_));
  }
  return {std::move(name), std::move(args)};
}

json Type::struct_to_json(const Type::Struct &x_) {
  const auto name = sym_to_json(x_.name);
  std::vector<json> args;
  args.reserve(x_.args.size());
  for (const auto &v_ : x_.args) {
    args.emplace_back(Type::any_to_json(v_));
  }
  return json::array({name, args});
}

Type::Ptr Type::ptr_from_json(const json &j_) {
  auto comp = Type::any_from_json(j_.at(0));
  auto space = TypeSpace::any_from_json(j_.at(1));
  return {std::move(comp), std::move(space)};
}

json Type::ptr_to_json(const Type::Ptr &x_) {
  const auto comp = Type::any_to_json(x_.comp);
  const auto space = TypeSpace::any_to_json(x_.space);
  return json::array({comp, space});
}

Type::Arr Type::arr_from_json(const json &j_) {
  auto comp = Type::any_from_json(j_.at(0));
  const auto length = j_.at(1).get<int32_t>();
  auto space = TypeSpace::any_from_json(j_.at(2));
  return {std::move(comp), length, std::move(space)};
}

json Type::arr_to_json(const Type::Arr &x_) {
  const auto comp = Type::any_to_json(x_.comp);
  const auto length = x_.length;
  const auto space = TypeSpace::any_to_json(x_.space);
  return json::array({comp, length, space});
}

Type::Var Type::var_from_json(const json &j_) {
  auto name = j_.at(0).get<std::string>();
  const auto exactSizeInBytes = j_.at(1).is_null() ? std::nullopt : std::make_optional(j_.at(1).get<int32_t>());
  return {std::move(name), exactSizeInBytes};
}

json Type::var_to_json(const Type::Var &x_) {
  const auto name = x_.name;
  const auto exactSizeInBytes = x_.exactSizeInBytes ? json(*x_.exactSizeInBytes) : json();
  return json::array({name, exactSizeInBytes});
}

Type::Exec Type::exec_from_json(const json &j_) {
  std::vector<Type::Var> tpeVars;
  tpeVars.reserve(j_.at(0).size());
  for (const auto &v_ : j_.at(0)) {
    tpeVars.emplace_back(Type::var_from_json(v_));
  }
  std::vector<Type::Any> args;
  args.reserve(j_.at(1).size());
  for (const auto &v_ : j_.at(1)) {
    args.emplace_back(Type::any_from_json(v_));
  }
  auto rtn = Type::any_from_json(j_.at(2));
  return {std::move(tpeVars), std::move(args), std::move(rtn)};
}

json Type::exec_to_json(const Type::Exec &x_) {
  std::vector<json> tpeVars;
  tpeVars.reserve(x_.tpeVars.size());
  for (const auto &v_ : x_.tpeVars) {
    tpeVars.emplace_back(Type::var_to_json(v_));
  }
  std::vector<json> args;
  args.reserve(x_.args.size());
  for (const auto &v_ : x_.args) {
    args.emplace_back(Type::any_to_json(v_));
  }
  const auto rtn = Type::any_to_json(x_.rtn);
  return json::array({tpeVars, args, rtn});
}

Type::FnRef Type::fnref_from_json(const json &j_) {
  auto name = sym_from_json(j_.at(0));
  return Type::FnRef(std::move(name));
}

json Type::fnref_to_json(const Type::FnRef &x_) {
  const auto name = sym_to_json(x_.name);
  return json::array({name});
}

Type::Any Type::any_from_json(const json &j_) {
  const auto ord_ = j_.at(0).get<size_t>();
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
    case 16: return Type::arr_from_json(t_);
    case 17: return Type::var_from_json(t_);
    case 18: return Type::exec_from_json(t_);
    case 19: return Type::fnref_from_json(t_);
    default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
  }
}

json Type::any_to_json(const Type::Any &x_) {
  return x_.match_total([](const Type::Float16 &y_) -> json { return {0, Type::float16_to_json(y_)}; },
                        [](const Type::Float32 &y_) -> json { return {1, Type::float32_to_json(y_)}; },
                        [](const Type::Float64 &y_) -> json { return {2, Type::float64_to_json(y_)}; },
                        [](const Type::IntU8 &y_) -> json { return {3, Type::intu8_to_json(y_)}; },
                        [](const Type::IntU16 &y_) -> json { return {4, Type::intu16_to_json(y_)}; },
                        [](const Type::IntU32 &y_) -> json { return {5, Type::intu32_to_json(y_)}; },
                        [](const Type::IntU64 &y_) -> json { return {6, Type::intu64_to_json(y_)}; },
                        [](const Type::IntS8 &y_) -> json { return {7, Type::ints8_to_json(y_)}; },
                        [](const Type::IntS16 &y_) -> json { return {8, Type::ints16_to_json(y_)}; },
                        [](const Type::IntS32 &y_) -> json { return {9, Type::ints32_to_json(y_)}; },
                        [](const Type::IntS64 &y_) -> json { return {10, Type::ints64_to_json(y_)}; },
                        [](const Type::Nothing &y_) -> json { return {11, Type::nothing_to_json(y_)}; },
                        [](const Type::Unit0 &y_) -> json { return {12, Type::unit0_to_json(y_)}; },
                        [](const Type::Bool1 &y_) -> json { return {13, Type::bool1_to_json(y_)}; },
                        [](const Type::Struct &y_) -> json { return {14, Type::struct_to_json(y_)}; },
                        [](const Type::Ptr &y_) -> json { return {15, Type::ptr_to_json(y_)}; },
                        [](const Type::Arr &y_) -> json { return {16, Type::arr_to_json(y_)}; },
                        [](const Type::Var &y_) -> json { return {17, Type::var_to_json(y_)}; },
                        [](const Type::Exec &y_) -> json { return {18, Type::exec_to_json(y_)}; },
                        [](const Type::FnRef &y_) -> json { return {19, Type::fnref_to_json(y_)}; });
}

PathStep::Field PathStep::field_from_json(const json &j_) {
  auto name = j_.at(0).get<std::string>();
  return PathStep::Field(std::move(name));
}

json PathStep::field_to_json(const PathStep::Field &x_) {
  const auto name = x_.name;
  return json::array({name});
}

PathStep::Deref PathStep::deref_from_json(const json &j_) { return {}; }

json PathStep::deref_to_json(const PathStep::Deref &x_) { return json::array({}); }

PathStep::Index PathStep::index_from_json(const json &j_) {
  const auto idx = j_.at(0).get<int32_t>();
  return PathStep::Index(idx);
}

json PathStep::index_to_json(const PathStep::Index &x_) {
  const auto idx = x_.idx;
  return json::array({idx});
}

PathStep::IndexDyn PathStep::indexdyn_from_json(const json &j_) {
  auto idx = Term::any_from_json(j_.at(0));
  return PathStep::IndexDyn(std::move(idx));
}

json PathStep::indexdyn_to_json(const PathStep::IndexDyn &x_) {
  const auto idx = Term::any_to_json(x_.idx);
  return json::array({idx});
}

PathStep::Any PathStep::any_from_json(const json &j_) {
  const auto ord_ = j_.at(0).get<size_t>();
  const auto &t_ = j_.at(1);
  switch (ord_) {
    case 0: return PathStep::field_from_json(t_);
    case 1: return PathStep::deref_from_json(t_);
    case 2: return PathStep::index_from_json(t_);
    case 3: return PathStep::indexdyn_from_json(t_);
    default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
  }
}

json PathStep::any_to_json(const PathStep::Any &x_) {
  return x_.match_total([](const PathStep::Field &y_) -> json { return {0, PathStep::field_to_json(y_)}; },
                        [](const PathStep::Deref &y_) -> json { return {1, PathStep::deref_to_json(y_)}; },
                        [](const PathStep::Index &y_) -> json { return {2, PathStep::index_to_json(y_)}; },
                        [](const PathStep::IndexDyn &y_) -> json { return {3, PathStep::indexdyn_to_json(y_)}; });
}

Region::Rooted Region::rooted_from_json(const json &j_) {
  auto root = named_from_json(j_.at(0));
  return Region::Rooted(std::move(root));
}

json Region::rooted_to_json(const Region::Rooted &x_) {
  const auto root = named_to_json(x_.root);
  return json::array({root});
}

Region::Opaque Region::opaque_from_json(const json &j_) { return {}; }

json Region::opaque_to_json(const Region::Opaque &x_) { return json::array({}); }

Region::Any Region::any_from_json(const json &j_) {
  const auto ord_ = j_.at(0).get<size_t>();
  const auto &t_ = j_.at(1);
  switch (ord_) {
    case 0: return Region::rooted_from_json(t_);
    case 1: return Region::opaque_from_json(t_);
    default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
  }
}

json Region::any_to_json(const Region::Any &x_) {
  return x_.match_total([](const Region::Rooted &y_) -> json { return {0, Region::rooted_to_json(y_)}; },
                        [](const Region::Opaque &y_) -> json { return {1, Region::opaque_to_json(y_)}; });
}

Term::Float16Const Term::float16const_from_json(const json &j_) {
  const auto value = j_.at(0).get<float>();
  return Term::Float16Const(value);
}

json Term::float16const_to_json(const Term::Float16Const &x_) {
  const auto value = x_.value;
  return json::array({value});
}

Term::Float32Const Term::float32const_from_json(const json &j_) {
  const auto value = j_.at(0).get<float>();
  return Term::Float32Const(value);
}

json Term::float32const_to_json(const Term::Float32Const &x_) {
  const auto value = x_.value;
  return json::array({value});
}

Term::Float64Const Term::float64const_from_json(const json &j_) {
  const auto value = j_.at(0).get<double>();
  return Term::Float64Const(value);
}

json Term::float64const_to_json(const Term::Float64Const &x_) {
  const auto value = x_.value;
  return json::array({value});
}

Term::IntU8Const Term::intu8const_from_json(const json &j_) {
  const auto value = j_.at(0).get<int8_t>();
  return Term::IntU8Const(value);
}

json Term::intu8const_to_json(const Term::IntU8Const &x_) {
  const auto value = x_.value;
  return json::array({value});
}

Term::IntU16Const Term::intu16const_from_json(const json &j_) {
  const auto value = j_.at(0).get<uint16_t>();
  return Term::IntU16Const(value);
}

json Term::intu16const_to_json(const Term::IntU16Const &x_) {
  const auto value = x_.value;
  return json::array({value});
}

Term::IntU32Const Term::intu32const_from_json(const json &j_) {
  const auto value = j_.at(0).get<int32_t>();
  return Term::IntU32Const(value);
}

json Term::intu32const_to_json(const Term::IntU32Const &x_) {
  const auto value = x_.value;
  return json::array({value});
}

Term::IntU64Const Term::intu64const_from_json(const json &j_) {
  const auto value = j_.at(0).get<int64_t>();
  return Term::IntU64Const(value);
}

json Term::intu64const_to_json(const Term::IntU64Const &x_) {
  const auto value = x_.value;
  return json::array({value});
}

Term::IntS8Const Term::ints8const_from_json(const json &j_) {
  const auto value = j_.at(0).get<int8_t>();
  return Term::IntS8Const(value);
}

json Term::ints8const_to_json(const Term::IntS8Const &x_) {
  const auto value = x_.value;
  return json::array({value});
}

Term::IntS16Const Term::ints16const_from_json(const json &j_) {
  const auto value = j_.at(0).get<int16_t>();
  return Term::IntS16Const(value);
}

json Term::ints16const_to_json(const Term::IntS16Const &x_) {
  const auto value = x_.value;
  return json::array({value});
}

Term::IntS32Const Term::ints32const_from_json(const json &j_) {
  const auto value = j_.at(0).get<int32_t>();
  return Term::IntS32Const(value);
}

json Term::ints32const_to_json(const Term::IntS32Const &x_) {
  const auto value = x_.value;
  return json::array({value});
}

Term::IntS64Const Term::ints64const_from_json(const json &j_) {
  const auto value = j_.at(0).get<int64_t>();
  return Term::IntS64Const(value);
}

json Term::ints64const_to_json(const Term::IntS64Const &x_) {
  const auto value = x_.value;
  return json::array({value});
}

Term::Unit0Const Term::unit0const_from_json(const json &j_) { return {}; }

json Term::unit0const_to_json(const Term::Unit0Const &x_) { return json::array({}); }

Term::Bool1Const Term::bool1const_from_json(const json &j_) {
  const auto value = j_.at(0).get<bool>();
  return Term::Bool1Const(value);
}

json Term::bool1const_to_json(const Term::Bool1Const &x_) {
  const auto value = x_.value;
  return json::array({value});
}

Term::NullPtrConst Term::nullptrconst_from_json(const json &j_) {
  auto comp = Type::any_from_json(j_.at(0));
  auto space = TypeSpace::any_from_json(j_.at(1));
  auto region = Region::any_from_json(j_.at(2));
  return {std::move(comp), std::move(space), std::move(region)};
}

json Term::nullptrconst_to_json(const Term::NullPtrConst &x_) {
  const auto comp = Type::any_to_json(x_.comp);
  const auto space = TypeSpace::any_to_json(x_.space);
  const auto region = Region::any_to_json(x_.region);
  return json::array({comp, space, region});
}

Term::StringConst Term::stringconst_from_json(const json &j_) {
  auto value = j_.at(0).get<std::string>();
  return Term::StringConst(std::move(value));
}

json Term::stringconst_to_json(const Term::StringConst &x_) {
  const auto value = x_.value;
  return json::array({value});
}

Term::Poison Term::poison_from_json(const json &j_) {
  auto t = Type::any_from_json(j_.at(0));
  return Term::Poison(std::move(t));
}

json Term::poison_to_json(const Term::Poison &x_) {
  const auto t = Type::any_to_json(x_.t);
  return json::array({t});
}

Term::Defer Term::defer_from_json(const json &j_) {
  auto t = Type::any_from_json(j_.at(0));
  return Term::Defer(std::move(t));
}

json Term::defer_to_json(const Term::Defer &x_) {
  const auto t = Type::any_to_json(x_.t);
  return json::array({t});
}

Term::Select Term::select_from_json(const json &j_) {
  auto root = named_from_json(j_.at(0));
  std::vector<PathStep::Any> steps;
  steps.reserve(j_.at(1).size());
  for (const auto &v_ : j_.at(1)) {
    steps.emplace_back(PathStep::any_from_json(v_));
  }
  auto tpe = Type::any_from_json(j_.at(2));
  return {std::move(root), std::move(steps), std::move(tpe)};
}

json Term::select_to_json(const Term::Select &x_) {
  const auto root = named_to_json(x_.root);
  std::vector<json> steps;
  steps.reserve(x_.steps.size());
  for (const auto &v_ : x_.steps) {
    steps.emplace_back(PathStep::any_to_json(v_));
  }
  const auto tpe = Type::any_to_json(x_.tpe);
  return json::array({root, steps, tpe});
}

Term::Any Term::any_from_json(const json &j_) {
  const auto ord_ = j_.at(0).get<size_t>();
  const auto &t_ = j_.at(1);
  switch (ord_) {
    case 0: return Term::float16const_from_json(t_);
    case 1: return Term::float32const_from_json(t_);
    case 2: return Term::float64const_from_json(t_);
    case 3: return Term::intu8const_from_json(t_);
    case 4: return Term::intu16const_from_json(t_);
    case 5: return Term::intu32const_from_json(t_);
    case 6: return Term::intu64const_from_json(t_);
    case 7: return Term::ints8const_from_json(t_);
    case 8: return Term::ints16const_from_json(t_);
    case 9: return Term::ints32const_from_json(t_);
    case 10: return Term::ints64const_from_json(t_);
    case 11: return Term::unit0const_from_json(t_);
    case 12: return Term::bool1const_from_json(t_);
    case 13: return Term::nullptrconst_from_json(t_);
    case 14: return Term::stringconst_from_json(t_);
    case 15: return Term::poison_from_json(t_);
    case 16: return Term::defer_from_json(t_);
    case 17: return Term::select_from_json(t_);
    default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
  }
}

json Term::any_to_json(const Term::Any &x_) {
  return x_.match_total([](const Term::Float16Const &y_) -> json { return {0, Term::float16const_to_json(y_)}; },
                        [](const Term::Float32Const &y_) -> json { return {1, Term::float32const_to_json(y_)}; },
                        [](const Term::Float64Const &y_) -> json { return {2, Term::float64const_to_json(y_)}; },
                        [](const Term::IntU8Const &y_) -> json { return {3, Term::intu8const_to_json(y_)}; },
                        [](const Term::IntU16Const &y_) -> json { return {4, Term::intu16const_to_json(y_)}; },
                        [](const Term::IntU32Const &y_) -> json { return {5, Term::intu32const_to_json(y_)}; },
                        [](const Term::IntU64Const &y_) -> json { return {6, Term::intu64const_to_json(y_)}; },
                        [](const Term::IntS8Const &y_) -> json { return {7, Term::ints8const_to_json(y_)}; },
                        [](const Term::IntS16Const &y_) -> json { return {8, Term::ints16const_to_json(y_)}; },
                        [](const Term::IntS32Const &y_) -> json { return {9, Term::ints32const_to_json(y_)}; },
                        [](const Term::IntS64Const &y_) -> json { return {10, Term::ints64const_to_json(y_)}; },
                        [](const Term::Unit0Const &y_) -> json { return {11, Term::unit0const_to_json(y_)}; },
                        [](const Term::Bool1Const &y_) -> json { return {12, Term::bool1const_to_json(y_)}; },
                        [](const Term::NullPtrConst &y_) -> json { return {13, Term::nullptrconst_to_json(y_)}; },
                        [](const Term::StringConst &y_) -> json { return {14, Term::stringconst_to_json(y_)}; },
                        [](const Term::Poison &y_) -> json { return {15, Term::poison_to_json(y_)}; },
                        [](const Term::Defer &y_) -> json { return {16, Term::defer_to_json(y_)}; },
                        [](const Term::Select &y_) -> json { return {17, Term::select_to_json(y_)}; });
}

Expr::Alias Expr::alias_from_json(const json &j_) {
  auto ref = Term::any_from_json(j_.at(0));
  return Expr::Alias(std::move(ref));
}

json Expr::alias_to_json(const Expr::Alias &x_) {
  const auto ref = Term::any_to_json(x_.ref);
  return json::array({ref});
}

Expr::SpecOp Expr::specop_from_json(const json &j_) {
  auto op = Spec::any_from_json(j_.at(0));
  return Expr::SpecOp(std::move(op));
}

json Expr::specop_to_json(const Expr::SpecOp &x_) {
  const auto op = Spec::any_to_json(x_.op);
  return json::array({op});
}

Expr::MathOp Expr::mathop_from_json(const json &j_) {
  auto op = Math::any_from_json(j_.at(0));
  return Expr::MathOp(std::move(op));
}

json Expr::mathop_to_json(const Expr::MathOp &x_) {
  const auto op = Math::any_to_json(x_.op);
  return json::array({op});
}

Expr::IntrOp Expr::introp_from_json(const json &j_) {
  auto op = Intr::any_from_json(j_.at(0));
  return Expr::IntrOp(std::move(op));
}

json Expr::introp_to_json(const Expr::IntrOp &x_) {
  const auto op = Intr::any_to_json(x_.op);
  return json::array({op});
}

Expr::Cast Expr::cast_from_json(const json &j_) {
  auto from = Term::any_from_json(j_.at(0));
  auto as = Type::any_from_json(j_.at(1));
  return {std::move(from), std::move(as)};
}

json Expr::cast_to_json(const Expr::Cast &x_) {
  const auto from = Term::any_to_json(x_.from);
  const auto as = Type::any_to_json(x_.as);
  return json::array({from, as});
}

Expr::BitCast Expr::bitcast_from_json(const json &j_) {
  auto from = Term::any_from_json(j_.at(0));
  auto as = Type::any_from_json(j_.at(1));
  return {std::move(from), std::move(as)};
}

json Expr::bitcast_to_json(const Expr::BitCast &x_) {
  const auto from = Term::any_to_json(x_.from);
  const auto as = Type::any_to_json(x_.as);
  return json::array({from, as});
}

Expr::Index Expr::index_from_json(const json &j_) {
  auto lhs = Term::any_from_json(j_.at(0));
  auto idx = Term::any_from_json(j_.at(1));
  auto comp = Type::any_from_json(j_.at(2));
  return {std::move(lhs), std::move(idx), std::move(comp)};
}

json Expr::index_to_json(const Expr::Index &x_) {
  const auto lhs = Term::any_to_json(x_.lhs);
  const auto idx = Term::any_to_json(x_.idx);
  const auto comp = Type::any_to_json(x_.comp);
  return json::array({lhs, idx, comp});
}

Expr::RefTo Expr::refto_from_json(const json &j_) {
  auto lhs = Term::any_from_json(j_.at(0));
  auto idx = j_.at(1).is_null() ? std::nullopt : std::make_optional(Term::any_from_json(j_.at(1)));
  auto comp = Type::any_from_json(j_.at(2));
  auto space = TypeSpace::any_from_json(j_.at(3));
  auto region = Region::any_from_json(j_.at(4));
  return {std::move(lhs), std::move(idx), std::move(comp), std::move(space), std::move(region)};
}

json Expr::refto_to_json(const Expr::RefTo &x_) {
  const auto lhs = Term::any_to_json(x_.lhs);
  const auto idx = x_.idx ? Term::any_to_json(*x_.idx) : json();
  const auto comp = Type::any_to_json(x_.comp);
  const auto space = TypeSpace::any_to_json(x_.space);
  const auto region = Region::any_to_json(x_.region);
  return json::array({lhs, idx, comp, space, region});
}

Expr::Alloc Expr::alloc_from_json(const json &j_) {
  auto comp = Type::any_from_json(j_.at(0));
  auto size = Term::any_from_json(j_.at(1));
  auto space = TypeSpace::any_from_json(j_.at(2));
  auto region = Region::any_from_json(j_.at(3));
  return {std::move(comp), std::move(size), std::move(space), std::move(region)};
}

json Expr::alloc_to_json(const Expr::Alloc &x_) {
  const auto comp = Type::any_to_json(x_.comp);
  const auto size = Term::any_to_json(x_.size);
  const auto space = TypeSpace::any_to_json(x_.space);
  const auto region = Region::any_to_json(x_.region);
  return json::array({comp, size, space, region});
}

Expr::Invoke Expr::invoke_from_json(const json &j_) {
  auto callee = Type::any_from_json(j_.at(0));
  std::vector<Type::Any> tpeArgs;
  tpeArgs.reserve(j_.at(1).size());
  for (const auto &v_ : j_.at(1)) {
    tpeArgs.emplace_back(Type::any_from_json(v_));
  }
  auto receiver = j_.at(2).is_null() ? std::nullopt : std::make_optional(Term::any_from_json(j_.at(2)));
  std::vector<Term::Any> args;
  args.reserve(j_.at(3).size());
  for (const auto &v_ : j_.at(3)) {
    args.emplace_back(Term::any_from_json(v_));
  }
  auto rtn = Type::any_from_json(j_.at(4));
  return {std::move(callee), std::move(tpeArgs), std::move(receiver), std::move(args), std::move(rtn)};
}

json Expr::invoke_to_json(const Expr::Invoke &x_) {
  const auto callee = Type::any_to_json(x_.callee);
  std::vector<json> tpeArgs;
  tpeArgs.reserve(x_.tpeArgs.size());
  for (const auto &v_ : x_.tpeArgs) {
    tpeArgs.emplace_back(Type::any_to_json(v_));
  }
  const auto receiver = x_.receiver ? Term::any_to_json(*x_.receiver) : json();
  std::vector<json> args;
  args.reserve(x_.args.size());
  for (const auto &v_ : x_.args) {
    args.emplace_back(Term::any_to_json(v_));
  }
  const auto rtn = Type::any_to_json(x_.rtn);
  return json::array({callee, tpeArgs, receiver, args, rtn});
}

Expr::ForeignCall Expr::foreigncall_from_json(const json &j_) {
  auto name = j_.at(0).get<std::string>();
  std::vector<Term::Any> args;
  args.reserve(j_.at(1).size());
  for (const auto &v_ : j_.at(1)) {
    args.emplace_back(Term::any_from_json(v_));
  }
  auto rtn = Type::any_from_json(j_.at(2));
  return {std::move(name), std::move(args), std::move(rtn)};
}

json Expr::foreigncall_to_json(const Expr::ForeignCall &x_) {
  const auto name = x_.name;
  std::vector<json> args;
  args.reserve(x_.args.size());
  for (const auto &v_ : x_.args) {
    args.emplace_back(Term::any_to_json(v_));
  }
  const auto rtn = Type::any_to_json(x_.rtn);
  return json::array({name, args, rtn});
}

Expr::OffsetOf Expr::offsetof_from_json(const json &j_) {
  auto structTpe = Type::any_from_json(j_.at(0));
  auto field = j_.at(1).get<std::string>();
  return {std::move(structTpe), std::move(field)};
}

json Expr::offsetof_to_json(const Expr::OffsetOf &x_) {
  const auto structTpe = Type::any_to_json(x_.structTpe);
  const auto field = x_.field;
  return json::array({structTpe, field});
}

Expr::SizeOf Expr::sizeof_from_json(const json &j_) {
  auto forTpe = Type::any_from_json(j_.at(0));
  return Expr::SizeOf(std::move(forTpe));
}

json Expr::sizeof_to_json(const Expr::SizeOf &x_) {
  const auto forTpe = Type::any_to_json(x_.forTpe);
  return json::array({forTpe});
}

Expr::Any Expr::any_from_json(const json &j_) {
  const auto ord_ = j_.at(0).get<size_t>();
  const auto &t_ = j_.at(1);
  switch (ord_) {
    case 0: return Expr::alias_from_json(t_);
    case 1: return Expr::specop_from_json(t_);
    case 2: return Expr::mathop_from_json(t_);
    case 3: return Expr::introp_from_json(t_);
    case 4: return Expr::cast_from_json(t_);
    case 5: return Expr::bitcast_from_json(t_);
    case 6: return Expr::index_from_json(t_);
    case 7: return Expr::refto_from_json(t_);
    case 8: return Expr::alloc_from_json(t_);
    case 9: return Expr::invoke_from_json(t_);
    case 10: return Expr::foreigncall_from_json(t_);
    case 11: return Expr::offsetof_from_json(t_);
    case 12: return Expr::sizeof_from_json(t_);
    default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
  }
}

json Expr::any_to_json(const Expr::Any &x_) {
  return x_.match_total([](const Expr::Alias &y_) -> json { return {0, Expr::alias_to_json(y_)}; },
                        [](const Expr::SpecOp &y_) -> json { return {1, Expr::specop_to_json(y_)}; },
                        [](const Expr::MathOp &y_) -> json { return {2, Expr::mathop_to_json(y_)}; },
                        [](const Expr::IntrOp &y_) -> json { return {3, Expr::introp_to_json(y_)}; },
                        [](const Expr::Cast &y_) -> json { return {4, Expr::cast_to_json(y_)}; },
                        [](const Expr::BitCast &y_) -> json { return {5, Expr::bitcast_to_json(y_)}; },
                        [](const Expr::Index &y_) -> json { return {6, Expr::index_to_json(y_)}; },
                        [](const Expr::RefTo &y_) -> json { return {7, Expr::refto_to_json(y_)}; },
                        [](const Expr::Alloc &y_) -> json { return {8, Expr::alloc_to_json(y_)}; },
                        [](const Expr::Invoke &y_) -> json { return {9, Expr::invoke_to_json(y_)}; },
                        [](const Expr::ForeignCall &y_) -> json { return {10, Expr::foreigncall_to_json(y_)}; },
                        [](const Expr::OffsetOf &y_) -> json { return {11, Expr::offsetof_to_json(y_)}; },
                        [](const Expr::SizeOf &y_) -> json { return {12, Expr::sizeof_to_json(y_)}; });
}

Overload overload_from_json(const json &j_) {
  std::vector<Type::Any> args;
  args.reserve(j_.at(0).size());
  for (const auto &v_ : j_.at(0)) {
    args.emplace_back(Type::any_from_json(v_));
  }
  auto rtn = Type::any_from_json(j_.at(1));
  return {std::move(args), std::move(rtn)};
}

json overload_to_json(const Overload &x_) {
  std::vector<json> args;
  args.reserve(x_.args.size());
  for (const auto &v_ : x_.args) {
    args.emplace_back(Type::any_to_json(v_));
  }
  const auto rtn = Type::any_to_json(x_.rtn);
  return json::array({args, rtn});
}

AtomicOp::Xchg AtomicOp::xchg_from_json(const json &j_) { return {}; }

json AtomicOp::xchg_to_json(const AtomicOp::Xchg &x_) { return json::array({}); }

AtomicOp::Add AtomicOp::add_from_json(const json &j_) { return {}; }

json AtomicOp::add_to_json(const AtomicOp::Add &x_) { return json::array({}); }

AtomicOp::Sub AtomicOp::sub_from_json(const json &j_) { return {}; }

json AtomicOp::sub_to_json(const AtomicOp::Sub &x_) { return json::array({}); }

AtomicOp::And AtomicOp::and_from_json(const json &j_) { return {}; }

json AtomicOp::and_to_json(const AtomicOp::And &x_) { return json::array({}); }

AtomicOp::Or AtomicOp::or_from_json(const json &j_) { return {}; }

json AtomicOp::or_to_json(const AtomicOp::Or &x_) { return json::array({}); }

AtomicOp::Xor AtomicOp::xor_from_json(const json &j_) { return {}; }

json AtomicOp::xor_to_json(const AtomicOp::Xor &x_) { return json::array({}); }

AtomicOp::Min AtomicOp::min_from_json(const json &j_) { return {}; }

json AtomicOp::min_to_json(const AtomicOp::Min &x_) { return json::array({}); }

AtomicOp::Max AtomicOp::max_from_json(const json &j_) { return {}; }

json AtomicOp::max_to_json(const AtomicOp::Max &x_) { return json::array({}); }

AtomicOp::Any AtomicOp::any_from_json(const json &j_) {
  const auto ord_ = j_.at(0).get<size_t>();
  const auto &t_ = j_.at(1);
  switch (ord_) {
    case 0: return AtomicOp::xchg_from_json(t_);
    case 1: return AtomicOp::add_from_json(t_);
    case 2: return AtomicOp::sub_from_json(t_);
    case 3: return AtomicOp::and_from_json(t_);
    case 4: return AtomicOp::or_from_json(t_);
    case 5: return AtomicOp::xor_from_json(t_);
    case 6: return AtomicOp::min_from_json(t_);
    case 7: return AtomicOp::max_from_json(t_);
    default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
  }
}

json AtomicOp::any_to_json(const AtomicOp::Any &x_) {
  return x_.match_total([](const AtomicOp::Xchg &y_) -> json { return {0, AtomicOp::xchg_to_json(y_)}; },
                        [](const AtomicOp::Add &y_) -> json { return {1, AtomicOp::add_to_json(y_)}; },
                        [](const AtomicOp::Sub &y_) -> json { return {2, AtomicOp::sub_to_json(y_)}; },
                        [](const AtomicOp::And &y_) -> json { return {3, AtomicOp::and_to_json(y_)}; },
                        [](const AtomicOp::Or &y_) -> json { return {4, AtomicOp::or_to_json(y_)}; },
                        [](const AtomicOp::Xor &y_) -> json { return {5, AtomicOp::xor_to_json(y_)}; },
                        [](const AtomicOp::Min &y_) -> json { return {6, AtomicOp::min_to_json(y_)}; },
                        [](const AtomicOp::Max &y_) -> json { return {7, AtomicOp::max_to_json(y_)}; });
}

MemScope::Subgroup MemScope::subgroup_from_json(const json &j_) { return {}; }

json MemScope::subgroup_to_json(const MemScope::Subgroup &x_) { return json::array({}); }

MemScope::Workgroup MemScope::workgroup_from_json(const json &j_) { return {}; }

json MemScope::workgroup_to_json(const MemScope::Workgroup &x_) { return json::array({}); }

MemScope::Device MemScope::device_from_json(const json &j_) { return {}; }

json MemScope::device_to_json(const MemScope::Device &x_) { return json::array({}); }

MemScope::System MemScope::system_from_json(const json &j_) { return {}; }

json MemScope::system_to_json(const MemScope::System &x_) { return json::array({}); }

MemScope::Any MemScope::any_from_json(const json &j_) {
  const auto ord_ = j_.at(0).get<size_t>();
  const auto &t_ = j_.at(1);
  switch (ord_) {
    case 0: return MemScope::subgroup_from_json(t_);
    case 1: return MemScope::workgroup_from_json(t_);
    case 2: return MemScope::device_from_json(t_);
    case 3: return MemScope::system_from_json(t_);
    default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
  }
}

json MemScope::any_to_json(const MemScope::Any &x_) {
  return x_.match_total([](const MemScope::Subgroup &y_) -> json { return {0, MemScope::subgroup_to_json(y_)}; },
                        [](const MemScope::Workgroup &y_) -> json { return {1, MemScope::workgroup_to_json(y_)}; },
                        [](const MemScope::Device &y_) -> json { return {2, MemScope::device_to_json(y_)}; },
                        [](const MemScope::System &y_) -> json { return {3, MemScope::system_to_json(y_)}; });
}

MemOrder::Relaxed MemOrder::relaxed_from_json(const json &j_) { return {}; }

json MemOrder::relaxed_to_json(const MemOrder::Relaxed &x_) { return json::array({}); }

MemOrder::Acquire MemOrder::acquire_from_json(const json &j_) { return {}; }

json MemOrder::acquire_to_json(const MemOrder::Acquire &x_) { return json::array({}); }

MemOrder::Release MemOrder::release_from_json(const json &j_) { return {}; }

json MemOrder::release_to_json(const MemOrder::Release &x_) { return json::array({}); }

MemOrder::AcqRel MemOrder::acqrel_from_json(const json &j_) { return {}; }

json MemOrder::acqrel_to_json(const MemOrder::AcqRel &x_) { return json::array({}); }

MemOrder::SeqCst MemOrder::seqcst_from_json(const json &j_) { return {}; }

json MemOrder::seqcst_to_json(const MemOrder::SeqCst &x_) { return json::array({}); }

MemOrder::Any MemOrder::any_from_json(const json &j_) {
  const auto ord_ = j_.at(0).get<size_t>();
  const auto &t_ = j_.at(1);
  switch (ord_) {
    case 0: return MemOrder::relaxed_from_json(t_);
    case 1: return MemOrder::acquire_from_json(t_);
    case 2: return MemOrder::release_from_json(t_);
    case 3: return MemOrder::acqrel_from_json(t_);
    case 4: return MemOrder::seqcst_from_json(t_);
    default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
  }
}

json MemOrder::any_to_json(const MemOrder::Any &x_) {
  return x_.match_total([](const MemOrder::Relaxed &y_) -> json { return {0, MemOrder::relaxed_to_json(y_)}; },
                        [](const MemOrder::Acquire &y_) -> json { return {1, MemOrder::acquire_to_json(y_)}; },
                        [](const MemOrder::Release &y_) -> json { return {2, MemOrder::release_to_json(y_)}; },
                        [](const MemOrder::AcqRel &y_) -> json { return {3, MemOrder::acqrel_to_json(y_)}; },
                        [](const MemOrder::SeqCst &y_) -> json { return {4, MemOrder::seqcst_to_json(y_)}; });
}

Direction::LocalToRemote Direction::localtoremote_from_json(const json &j_) { return {}; }

json Direction::localtoremote_to_json(const Direction::LocalToRemote &x_) { return json::array({}); }

Direction::RemoteToLocal Direction::remotetolocal_from_json(const json &j_) { return {}; }

json Direction::remotetolocal_to_json(const Direction::RemoteToLocal &x_) { return json::array({}); }

Direction::RemoteToRemote Direction::remotetoremote_from_json(const json &j_) { return {}; }

json Direction::remotetoremote_to_json(const Direction::RemoteToRemote &x_) { return json::array({}); }

Direction::Any Direction::any_from_json(const json &j_) {
  const auto ord_ = j_.at(0).get<size_t>();
  const auto &t_ = j_.at(1);
  switch (ord_) {
    case 0: return Direction::localtoremote_from_json(t_);
    case 1: return Direction::remotetolocal_from_json(t_);
    case 2: return Direction::remotetoremote_from_json(t_);
    default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
  }
}

json Direction::any_to_json(const Direction::Any &x_) {
  return x_.match_total([](const Direction::LocalToRemote &y_) -> json { return {0, Direction::localtoremote_to_json(y_)}; },
                        [](const Direction::RemoteToLocal &y_) -> json { return {1, Direction::remotetolocal_to_json(y_)}; },
                        [](const Direction::RemoteToRemote &y_) -> json { return {2, Direction::remotetoremote_to_json(y_)}; });
}

Spec::Assert Spec::assert_from_json(const json &j_) {
  auto code = Term::any_from_json(j_.at(0));
  auto message = Term::any_from_json(j_.at(1));
  return {std::move(code), std::move(message)};
}

json Spec::assert_to_json(const Spec::Assert &x_) {
  const auto code = Term::any_to_json(x_.code);
  const auto message = Term::any_to_json(x_.message);
  return json::array({code, message});
}

Spec::GpuBarrierGlobal Spec::gpubarrierglobal_from_json(const json &j_) { return {}; }

json Spec::gpubarrierglobal_to_json(const Spec::GpuBarrierGlobal &x_) { return json::array({}); }

Spec::GpuBarrierLocal Spec::gpubarrierlocal_from_json(const json &j_) { return {}; }

json Spec::gpubarrierlocal_to_json(const Spec::GpuBarrierLocal &x_) { return json::array({}); }

Spec::GpuBarrierAll Spec::gpubarrierall_from_json(const json &j_) { return {}; }

json Spec::gpubarrierall_to_json(const Spec::GpuBarrierAll &x_) { return json::array({}); }

Spec::GpuFenceGlobal Spec::gpufenceglobal_from_json(const json &j_) { return {}; }

json Spec::gpufenceglobal_to_json(const Spec::GpuFenceGlobal &x_) { return json::array({}); }

Spec::GpuFenceLocal Spec::gpufencelocal_from_json(const json &j_) { return {}; }

json Spec::gpufencelocal_to_json(const Spec::GpuFenceLocal &x_) { return json::array({}); }

Spec::GpuFenceAll Spec::gpufenceall_from_json(const json &j_) { return {}; }

json Spec::gpufenceall_to_json(const Spec::GpuFenceAll &x_) { return json::array({}); }

Spec::GpuGlobalIdx Spec::gpuglobalidx_from_json(const json &j_) {
  auto dim = Term::any_from_json(j_.at(0));
  return Spec::GpuGlobalIdx(std::move(dim));
}

json Spec::gpuglobalidx_to_json(const Spec::GpuGlobalIdx &x_) {
  const auto dim = Term::any_to_json(x_.dim);
  return json::array({dim});
}

Spec::GpuGlobalSize Spec::gpuglobalsize_from_json(const json &j_) {
  auto dim = Term::any_from_json(j_.at(0));
  return Spec::GpuGlobalSize(std::move(dim));
}

json Spec::gpuglobalsize_to_json(const Spec::GpuGlobalSize &x_) {
  const auto dim = Term::any_to_json(x_.dim);
  return json::array({dim});
}

Spec::GpuGroupIdx Spec::gpugroupidx_from_json(const json &j_) {
  auto dim = Term::any_from_json(j_.at(0));
  return Spec::GpuGroupIdx(std::move(dim));
}

json Spec::gpugroupidx_to_json(const Spec::GpuGroupIdx &x_) {
  const auto dim = Term::any_to_json(x_.dim);
  return json::array({dim});
}

Spec::GpuGroupSize Spec::gpugroupsize_from_json(const json &j_) {
  auto dim = Term::any_from_json(j_.at(0));
  return Spec::GpuGroupSize(std::move(dim));
}

json Spec::gpugroupsize_to_json(const Spec::GpuGroupSize &x_) {
  const auto dim = Term::any_to_json(x_.dim);
  return json::array({dim});
}

Spec::GpuLocalIdx Spec::gpulocalidx_from_json(const json &j_) {
  auto dim = Term::any_from_json(j_.at(0));
  return Spec::GpuLocalIdx(std::move(dim));
}

json Spec::gpulocalidx_to_json(const Spec::GpuLocalIdx &x_) {
  const auto dim = Term::any_to_json(x_.dim);
  return json::array({dim});
}

Spec::GpuLocalSize Spec::gpulocalsize_from_json(const json &j_) {
  auto dim = Term::any_from_json(j_.at(0));
  return Spec::GpuLocalSize(std::move(dim));
}

json Spec::gpulocalsize_to_json(const Spec::GpuLocalSize &x_) {
  const auto dim = Term::any_to_json(x_.dim);
  return json::array({dim});
}

Spec::GpuLaneIdx Spec::gpulaneidx_from_json(const json &j_) { return {}; }

json Spec::gpulaneidx_to_json(const Spec::GpuLaneIdx &x_) { return json::array({}); }

Spec::GpuSubgroupSize Spec::gpusubgroupsize_from_json(const json &j_) { return {}; }

json Spec::gpusubgroupsize_to_json(const Spec::GpuSubgroupSize &x_) { return json::array({}); }

Spec::GpuShuffleDown Spec::gpushuffledown_from_json(const json &j_) {
  auto value = Term::any_from_json(j_.at(0));
  auto delta = Term::any_from_json(j_.at(1));
  auto width = Term::any_from_json(j_.at(2));
  auto mask = Term::any_from_json(j_.at(3));
  auto rtn = Type::any_from_json(j_.at(4));
  return {std::move(value), std::move(delta), std::move(width), std::move(mask), std::move(rtn)};
}

json Spec::gpushuffledown_to_json(const Spec::GpuShuffleDown &x_) {
  const auto value = Term::any_to_json(x_.value);
  const auto delta = Term::any_to_json(x_.delta);
  const auto width = Term::any_to_json(x_.width);
  const auto mask = Term::any_to_json(x_.mask);
  const auto rtn = Type::any_to_json(x_.rtn);
  return json::array({value, delta, width, mask, rtn});
}

Spec::GpuShuffleUp Spec::gpushuffleup_from_json(const json &j_) {
  auto value = Term::any_from_json(j_.at(0));
  auto delta = Term::any_from_json(j_.at(1));
  auto width = Term::any_from_json(j_.at(2));
  auto mask = Term::any_from_json(j_.at(3));
  auto rtn = Type::any_from_json(j_.at(4));
  return {std::move(value), std::move(delta), std::move(width), std::move(mask), std::move(rtn)};
}

json Spec::gpushuffleup_to_json(const Spec::GpuShuffleUp &x_) {
  const auto value = Term::any_to_json(x_.value);
  const auto delta = Term::any_to_json(x_.delta);
  const auto width = Term::any_to_json(x_.width);
  const auto mask = Term::any_to_json(x_.mask);
  const auto rtn = Type::any_to_json(x_.rtn);
  return json::array({value, delta, width, mask, rtn});
}

Spec::GpuShuffleIdx Spec::gpushuffleidx_from_json(const json &j_) {
  auto value = Term::any_from_json(j_.at(0));
  auto srcLane = Term::any_from_json(j_.at(1));
  auto width = Term::any_from_json(j_.at(2));
  auto mask = Term::any_from_json(j_.at(3));
  auto rtn = Type::any_from_json(j_.at(4));
  return {std::move(value), std::move(srcLane), std::move(width), std::move(mask), std::move(rtn)};
}

json Spec::gpushuffleidx_to_json(const Spec::GpuShuffleIdx &x_) {
  const auto value = Term::any_to_json(x_.value);
  const auto srcLane = Term::any_to_json(x_.srcLane);
  const auto width = Term::any_to_json(x_.width);
  const auto mask = Term::any_to_json(x_.mask);
  const auto rtn = Type::any_to_json(x_.rtn);
  return json::array({value, srcLane, width, mask, rtn});
}

Spec::GpuShuffleXor Spec::gpushufflexor_from_json(const json &j_) {
  auto value = Term::any_from_json(j_.at(0));
  auto laneMask = Term::any_from_json(j_.at(1));
  auto width = Term::any_from_json(j_.at(2));
  auto mask = Term::any_from_json(j_.at(3));
  auto rtn = Type::any_from_json(j_.at(4));
  return {std::move(value), std::move(laneMask), std::move(width), std::move(mask), std::move(rtn)};
}

json Spec::gpushufflexor_to_json(const Spec::GpuShuffleXor &x_) {
  const auto value = Term::any_to_json(x_.value);
  const auto laneMask = Term::any_to_json(x_.laneMask);
  const auto width = Term::any_to_json(x_.width);
  const auto mask = Term::any_to_json(x_.mask);
  const auto rtn = Type::any_to_json(x_.rtn);
  return json::array({value, laneMask, width, mask, rtn});
}

Spec::GpuSubgroupBarrier Spec::gpusubgroupbarrier_from_json(const json &j_) {
  auto mask = Term::any_from_json(j_.at(0));
  return Spec::GpuSubgroupBarrier(std::move(mask));
}

json Spec::gpusubgroupbarrier_to_json(const Spec::GpuSubgroupBarrier &x_) {
  const auto mask = Term::any_to_json(x_.mask);
  return json::array({mask});
}

Spec::GpuBallot Spec::gpuballot_from_json(const json &j_) {
  auto mask = Term::any_from_json(j_.at(0));
  auto pred = Term::any_from_json(j_.at(1));
  return {std::move(mask), std::move(pred)};
}

json Spec::gpuballot_to_json(const Spec::GpuBallot &x_) {
  const auto mask = Term::any_to_json(x_.mask);
  const auto pred = Term::any_to_json(x_.pred);
  return json::array({mask, pred});
}

Spec::GpuVoteAny Spec::gpuvoteany_from_json(const json &j_) {
  auto mask = Term::any_from_json(j_.at(0));
  auto pred = Term::any_from_json(j_.at(1));
  return {std::move(mask), std::move(pred)};
}

json Spec::gpuvoteany_to_json(const Spec::GpuVoteAny &x_) {
  const auto mask = Term::any_to_json(x_.mask);
  const auto pred = Term::any_to_json(x_.pred);
  return json::array({mask, pred});
}

Spec::GpuVoteAll Spec::gpuvoteall_from_json(const json &j_) {
  auto mask = Term::any_from_json(j_.at(0));
  auto pred = Term::any_from_json(j_.at(1));
  return {std::move(mask), std::move(pred)};
}

json Spec::gpuvoteall_to_json(const Spec::GpuVoteAll &x_) {
  const auto mask = Term::any_to_json(x_.mask);
  const auto pred = Term::any_to_json(x_.pred);
  return json::array({mask, pred});
}

Spec::GpuAtomicRMW Spec::gpuatomicrmw_from_json(const json &j_) {
  auto op = AtomicOp::any_from_json(j_.at(0));
  auto ptr = Term::any_from_json(j_.at(1));
  auto value = Term::any_from_json(j_.at(2));
  auto scope = MemScope::any_from_json(j_.at(3));
  auto order = MemOrder::any_from_json(j_.at(4));
  auto rtn = Type::any_from_json(j_.at(5));
  return {std::move(op), std::move(ptr), std::move(value), std::move(scope), std::move(order), std::move(rtn)};
}

json Spec::gpuatomicrmw_to_json(const Spec::GpuAtomicRMW &x_) {
  const auto op = AtomicOp::any_to_json(x_.op);
  const auto ptr = Term::any_to_json(x_.ptr);
  const auto value = Term::any_to_json(x_.value);
  const auto scope = MemScope::any_to_json(x_.scope);
  const auto order = MemOrder::any_to_json(x_.order);
  const auto rtn = Type::any_to_json(x_.rtn);
  return json::array({op, ptr, value, scope, order, rtn});
}

Spec::GpuAtomicCAS Spec::gpuatomiccas_from_json(const json &j_) {
  auto ptr = Term::any_from_json(j_.at(0));
  auto expected = Term::any_from_json(j_.at(1));
  auto desired = Term::any_from_json(j_.at(2));
  auto scope = MemScope::any_from_json(j_.at(3));
  auto order = MemOrder::any_from_json(j_.at(4));
  auto rtn = Type::any_from_json(j_.at(5));
  return {std::move(ptr), std::move(expected), std::move(desired), std::move(scope), std::move(order), std::move(rtn)};
}

json Spec::gpuatomiccas_to_json(const Spec::GpuAtomicCAS &x_) {
  const auto ptr = Term::any_to_json(x_.ptr);
  const auto expected = Term::any_to_json(x_.expected);
  const auto desired = Term::any_to_json(x_.desired);
  const auto scope = MemScope::any_to_json(x_.scope);
  const auto order = MemOrder::any_to_json(x_.order);
  const auto rtn = Type::any_to_json(x_.rtn);
  return json::array({ptr, expected, desired, scope, order, rtn});
}

Spec::GpuGroupReduce Spec::gpugroupreduce_from_json(const json &j_) {
  auto op = AtomicOp::any_from_json(j_.at(0));
  auto value = Term::any_from_json(j_.at(1));
  auto rtn = Type::any_from_json(j_.at(2));
  return {std::move(op), std::move(value), std::move(rtn)};
}

json Spec::gpugroupreduce_to_json(const Spec::GpuGroupReduce &x_) {
  const auto op = AtomicOp::any_to_json(x_.op);
  const auto value = Term::any_to_json(x_.value);
  const auto rtn = Type::any_to_json(x_.rtn);
  return json::array({op, value, rtn});
}

Spec::GpuGroupInclusiveScan Spec::gpugroupinclusivescan_from_json(const json &j_) {
  auto op = AtomicOp::any_from_json(j_.at(0));
  auto value = Term::any_from_json(j_.at(1));
  auto rtn = Type::any_from_json(j_.at(2));
  return {std::move(op), std::move(value), std::move(rtn)};
}

json Spec::gpugroupinclusivescan_to_json(const Spec::GpuGroupInclusiveScan &x_) {
  const auto op = AtomicOp::any_to_json(x_.op);
  const auto value = Term::any_to_json(x_.value);
  const auto rtn = Type::any_to_json(x_.rtn);
  return json::array({op, value, rtn});
}

Spec::GpuGroupExclusiveScan Spec::gpugroupexclusivescan_from_json(const json &j_) {
  auto op = AtomicOp::any_from_json(j_.at(0));
  auto value = Term::any_from_json(j_.at(1));
  auto rtn = Type::any_from_json(j_.at(2));
  return {std::move(op), std::move(value), std::move(rtn)};
}

json Spec::gpugroupexclusivescan_to_json(const Spec::GpuGroupExclusiveScan &x_) {
  const auto op = AtomicOp::any_to_json(x_.op);
  const auto value = Term::any_to_json(x_.value);
  const auto rtn = Type::any_to_json(x_.rtn);
  return json::array({op, value, rtn});
}

Spec::GpuVolatileLoad Spec::gpuvolatileload_from_json(const json &j_) {
  auto ptr = Term::any_from_json(j_.at(0));
  auto rtn = Type::any_from_json(j_.at(1));
  return {std::move(ptr), std::move(rtn)};
}

json Spec::gpuvolatileload_to_json(const Spec::GpuVolatileLoad &x_) {
  const auto ptr = Term::any_to_json(x_.ptr);
  const auto rtn = Type::any_to_json(x_.rtn);
  return json::array({ptr, rtn});
}

Spec::GpuVolatileStore Spec::gpuvolatilestore_from_json(const json &j_) {
  auto ptr = Term::any_from_json(j_.at(0));
  auto value = Term::any_from_json(j_.at(1));
  return {std::move(ptr), std::move(value)};
}

json Spec::gpuvolatilestore_to_json(const Spec::GpuVolatileStore &x_) {
  const auto ptr = Term::any_to_json(x_.ptr);
  const auto value = Term::any_to_json(x_.value);
  return json::array({ptr, value});
}

Spec::RemoteLaunch Spec::remotelaunch_from_json(const json &j_) {
  auto context = Term::any_from_json(j_.at(0));
  auto kernel = Term::any_from_json(j_.at(1));
  std::vector<Type::Any> tpeArgs;
  tpeArgs.reserve(j_.at(2).size());
  for (const auto &v_ : j_.at(2)) {
    tpeArgs.emplace_back(Type::any_from_json(v_));
  }
  auto gridX = Term::any_from_json(j_.at(3));
  auto gridY = Term::any_from_json(j_.at(4));
  auto gridZ = Term::any_from_json(j_.at(5));
  auto blockX = Term::any_from_json(j_.at(6));
  auto blockY = Term::any_from_json(j_.at(7));
  auto blockZ = Term::any_from_json(j_.at(8));
  auto shmem = Term::any_from_json(j_.at(9));
  std::vector<Term::Any> args;
  args.reserve(j_.at(10).size());
  for (const auto &v_ : j_.at(10)) {
    args.emplace_back(Term::any_from_json(v_));
  }
  return {std::move(context), std::move(kernel), std::move(tpeArgs), std::move(gridX), std::move(gridY), std::move(gridZ),
          std::move(blockX),  std::move(blockY), std::move(blockZ),  std::move(shmem), std::move(args)};
}

json Spec::remotelaunch_to_json(const Spec::RemoteLaunch &x_) {
  const auto context = Term::any_to_json(x_.context);
  const auto kernel = Term::any_to_json(x_.kernel);
  std::vector<json> tpeArgs;
  tpeArgs.reserve(x_.tpeArgs.size());
  for (const auto &v_ : x_.tpeArgs) {
    tpeArgs.emplace_back(Type::any_to_json(v_));
  }
  const auto gridX = Term::any_to_json(x_.gridX);
  const auto gridY = Term::any_to_json(x_.gridY);
  const auto gridZ = Term::any_to_json(x_.gridZ);
  const auto blockX = Term::any_to_json(x_.blockX);
  const auto blockY = Term::any_to_json(x_.blockY);
  const auto blockZ = Term::any_to_json(x_.blockZ);
  const auto shmem = Term::any_to_json(x_.shmem);
  std::vector<json> args;
  args.reserve(x_.args.size());
  for (const auto &v_ : x_.args) {
    args.emplace_back(Term::any_to_json(v_));
  }
  return json::array({context, kernel, tpeArgs, gridX, gridY, gridZ, blockX, blockY, blockZ, shmem, args});
}

Spec::RemoteAlloc Spec::remotealloc_from_json(const json &j_) {
  auto context = Term::any_from_json(j_.at(0));
  auto bytes = Term::any_from_json(j_.at(1));
  return {std::move(context), std::move(bytes)};
}

json Spec::remotealloc_to_json(const Spec::RemoteAlloc &x_) {
  const auto context = Term::any_to_json(x_.context);
  const auto bytes = Term::any_to_json(x_.bytes);
  return json::array({context, bytes});
}

Spec::RemoteTempAlloc Spec::remotetempalloc_from_json(const json &j_) {
  auto context = Term::any_from_json(j_.at(0));
  auto bytes = Term::any_from_json(j_.at(1));
  return {std::move(context), std::move(bytes)};
}

json Spec::remotetempalloc_to_json(const Spec::RemoteTempAlloc &x_) {
  const auto context = Term::any_to_json(x_.context);
  const auto bytes = Term::any_to_json(x_.bytes);
  return json::array({context, bytes});
}

Spec::RemoteFree Spec::remotefree_from_json(const json &j_) {
  auto context = Term::any_from_json(j_.at(0));
  auto ptr = Term::any_from_json(j_.at(1));
  return {std::move(context), std::move(ptr)};
}

json Spec::remotefree_to_json(const Spec::RemoteFree &x_) {
  const auto context = Term::any_to_json(x_.context);
  const auto ptr = Term::any_to_json(x_.ptr);
  return json::array({context, ptr});
}

Spec::RemoteMemcpy Spec::remotememcpy_from_json(const json &j_) {
  auto context = Term::any_from_json(j_.at(0));
  auto dst = Term::any_from_json(j_.at(1));
  auto src = Term::any_from_json(j_.at(2));
  auto bytes = Term::any_from_json(j_.at(3));
  auto direction = Direction::any_from_json(j_.at(4));
  return {std::move(context), std::move(dst), std::move(src), std::move(bytes), std::move(direction)};
}

json Spec::remotememcpy_to_json(const Spec::RemoteMemcpy &x_) {
  const auto context = Term::any_to_json(x_.context);
  const auto dst = Term::any_to_json(x_.dst);
  const auto src = Term::any_to_json(x_.src);
  const auto bytes = Term::any_to_json(x_.bytes);
  const auto direction = Direction::any_to_json(x_.direction);
  return json::array({context, dst, src, bytes, direction});
}

Spec::RemoteSync Spec::remotesync_from_json(const json &j_) {
  auto context = Term::any_from_json(j_.at(0));
  return Spec::RemoteSync(std::move(context));
}

json Spec::remotesync_to_json(const Spec::RemoteSync &x_) {
  const auto context = Term::any_to_json(x_.context);
  return json::array({context});
}

Spec::Any Spec::any_from_json(const json &j_) {
  const auto ord_ = j_.at(0).get<size_t>();
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
    case 13: return Spec::gpulaneidx_from_json(t_);
    case 14: return Spec::gpusubgroupsize_from_json(t_);
    case 15: return Spec::gpushuffledown_from_json(t_);
    case 16: return Spec::gpushuffleup_from_json(t_);
    case 17: return Spec::gpushuffleidx_from_json(t_);
    case 18: return Spec::gpushufflexor_from_json(t_);
    case 19: return Spec::gpusubgroupbarrier_from_json(t_);
    case 20: return Spec::gpuballot_from_json(t_);
    case 21: return Spec::gpuvoteany_from_json(t_);
    case 22: return Spec::gpuvoteall_from_json(t_);
    case 23: return Spec::gpuatomicrmw_from_json(t_);
    case 24: return Spec::gpuatomiccas_from_json(t_);
    case 25: return Spec::gpugroupreduce_from_json(t_);
    case 26: return Spec::gpugroupinclusivescan_from_json(t_);
    case 27: return Spec::gpugroupexclusivescan_from_json(t_);
    case 28: return Spec::gpuvolatileload_from_json(t_);
    case 29: return Spec::gpuvolatilestore_from_json(t_);
    case 30: return Spec::remotelaunch_from_json(t_);
    case 31: return Spec::remotealloc_from_json(t_);
    case 32: return Spec::remotetempalloc_from_json(t_);
    case 33: return Spec::remotefree_from_json(t_);
    case 34: return Spec::remotememcpy_from_json(t_);
    case 35: return Spec::remotesync_from_json(t_);
    default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
  }
}

json Spec::any_to_json(const Spec::Any &x_) {
  return x_.match_total([](const Spec::Assert &y_) -> json { return {0, Spec::assert_to_json(y_)}; },
                        [](const Spec::GpuBarrierGlobal &y_) -> json { return {1, Spec::gpubarrierglobal_to_json(y_)}; },
                        [](const Spec::GpuBarrierLocal &y_) -> json { return {2, Spec::gpubarrierlocal_to_json(y_)}; },
                        [](const Spec::GpuBarrierAll &y_) -> json { return {3, Spec::gpubarrierall_to_json(y_)}; },
                        [](const Spec::GpuFenceGlobal &y_) -> json { return {4, Spec::gpufenceglobal_to_json(y_)}; },
                        [](const Spec::GpuFenceLocal &y_) -> json { return {5, Spec::gpufencelocal_to_json(y_)}; },
                        [](const Spec::GpuFenceAll &y_) -> json { return {6, Spec::gpufenceall_to_json(y_)}; },
                        [](const Spec::GpuGlobalIdx &y_) -> json { return {7, Spec::gpuglobalidx_to_json(y_)}; },
                        [](const Spec::GpuGlobalSize &y_) -> json { return {8, Spec::gpuglobalsize_to_json(y_)}; },
                        [](const Spec::GpuGroupIdx &y_) -> json { return {9, Spec::gpugroupidx_to_json(y_)}; },
                        [](const Spec::GpuGroupSize &y_) -> json { return {10, Spec::gpugroupsize_to_json(y_)}; },
                        [](const Spec::GpuLocalIdx &y_) -> json { return {11, Spec::gpulocalidx_to_json(y_)}; },
                        [](const Spec::GpuLocalSize &y_) -> json { return {12, Spec::gpulocalsize_to_json(y_)}; },
                        [](const Spec::GpuLaneIdx &y_) -> json { return {13, Spec::gpulaneidx_to_json(y_)}; },
                        [](const Spec::GpuSubgroupSize &y_) -> json { return {14, Spec::gpusubgroupsize_to_json(y_)}; },
                        [](const Spec::GpuShuffleDown &y_) -> json { return {15, Spec::gpushuffledown_to_json(y_)}; },
                        [](const Spec::GpuShuffleUp &y_) -> json { return {16, Spec::gpushuffleup_to_json(y_)}; },
                        [](const Spec::GpuShuffleIdx &y_) -> json { return {17, Spec::gpushuffleidx_to_json(y_)}; },
                        [](const Spec::GpuShuffleXor &y_) -> json { return {18, Spec::gpushufflexor_to_json(y_)}; },
                        [](const Spec::GpuSubgroupBarrier &y_) -> json { return {19, Spec::gpusubgroupbarrier_to_json(y_)}; },
                        [](const Spec::GpuBallot &y_) -> json { return {20, Spec::gpuballot_to_json(y_)}; },
                        [](const Spec::GpuVoteAny &y_) -> json { return {21, Spec::gpuvoteany_to_json(y_)}; },
                        [](const Spec::GpuVoteAll &y_) -> json { return {22, Spec::gpuvoteall_to_json(y_)}; },
                        [](const Spec::GpuAtomicRMW &y_) -> json { return {23, Spec::gpuatomicrmw_to_json(y_)}; },
                        [](const Spec::GpuAtomicCAS &y_) -> json { return {24, Spec::gpuatomiccas_to_json(y_)}; },
                        [](const Spec::GpuGroupReduce &y_) -> json { return {25, Spec::gpugroupreduce_to_json(y_)}; },
                        [](const Spec::GpuGroupInclusiveScan &y_) -> json { return {26, Spec::gpugroupinclusivescan_to_json(y_)}; },
                        [](const Spec::GpuGroupExclusiveScan &y_) -> json { return {27, Spec::gpugroupexclusivescan_to_json(y_)}; },
                        [](const Spec::GpuVolatileLoad &y_) -> json { return {28, Spec::gpuvolatileload_to_json(y_)}; },
                        [](const Spec::GpuVolatileStore &y_) -> json { return {29, Spec::gpuvolatilestore_to_json(y_)}; },
                        [](const Spec::RemoteLaunch &y_) -> json { return {30, Spec::remotelaunch_to_json(y_)}; },
                        [](const Spec::RemoteAlloc &y_) -> json { return {31, Spec::remotealloc_to_json(y_)}; },
                        [](const Spec::RemoteTempAlloc &y_) -> json { return {32, Spec::remotetempalloc_to_json(y_)}; },
                        [](const Spec::RemoteFree &y_) -> json { return {33, Spec::remotefree_to_json(y_)}; },
                        [](const Spec::RemoteMemcpy &y_) -> json { return {34, Spec::remotememcpy_to_json(y_)}; },
                        [](const Spec::RemoteSync &y_) -> json { return {35, Spec::remotesync_to_json(y_)}; });
}

Intr::BNot Intr::bnot_from_json(const json &j_) {
  auto x = Term::any_from_json(j_.at(0));
  auto rtn = Type::any_from_json(j_.at(1));
  return {std::move(x), std::move(rtn)};
}

json Intr::bnot_to_json(const Intr::BNot &x_) {
  const auto x = Term::any_to_json(x_.x);
  const auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, rtn});
}

Intr::LogicNot Intr::logicnot_from_json(const json &j_) {
  auto x = Term::any_from_json(j_.at(0));
  return Intr::LogicNot(std::move(x));
}

json Intr::logicnot_to_json(const Intr::LogicNot &x_) {
  const auto x = Term::any_to_json(x_.x);
  return json::array({x});
}

Intr::Pos Intr::pos_from_json(const json &j_) {
  auto x = Term::any_from_json(j_.at(0));
  auto rtn = Type::any_from_json(j_.at(1));
  return {std::move(x), std::move(rtn)};
}

json Intr::pos_to_json(const Intr::Pos &x_) {
  const auto x = Term::any_to_json(x_.x);
  const auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, rtn});
}

Intr::Neg Intr::neg_from_json(const json &j_) {
  auto x = Term::any_from_json(j_.at(0));
  auto rtn = Type::any_from_json(j_.at(1));
  return {std::move(x), std::move(rtn)};
}

json Intr::neg_to_json(const Intr::Neg &x_) {
  const auto x = Term::any_to_json(x_.x);
  const auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, rtn});
}

Intr::Add Intr::add_from_json(const json &j_) {
  auto x = Term::any_from_json(j_.at(0));
  auto y = Term::any_from_json(j_.at(1));
  auto rtn = Type::any_from_json(j_.at(2));
  return {std::move(x), std::move(y), std::move(rtn)};
}

json Intr::add_to_json(const Intr::Add &x_) {
  const auto x = Term::any_to_json(x_.x);
  const auto y = Term::any_to_json(x_.y);
  const auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, y, rtn});
}

Intr::Sub Intr::sub_from_json(const json &j_) {
  auto x = Term::any_from_json(j_.at(0));
  auto y = Term::any_from_json(j_.at(1));
  auto rtn = Type::any_from_json(j_.at(2));
  return {std::move(x), std::move(y), std::move(rtn)};
}

json Intr::sub_to_json(const Intr::Sub &x_) {
  const auto x = Term::any_to_json(x_.x);
  const auto y = Term::any_to_json(x_.y);
  const auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, y, rtn});
}

Intr::Mul Intr::mul_from_json(const json &j_) {
  auto x = Term::any_from_json(j_.at(0));
  auto y = Term::any_from_json(j_.at(1));
  auto rtn = Type::any_from_json(j_.at(2));
  return {std::move(x), std::move(y), std::move(rtn)};
}

json Intr::mul_to_json(const Intr::Mul &x_) {
  const auto x = Term::any_to_json(x_.x);
  const auto y = Term::any_to_json(x_.y);
  const auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, y, rtn});
}

Intr::Div Intr::div_from_json(const json &j_) {
  auto x = Term::any_from_json(j_.at(0));
  auto y = Term::any_from_json(j_.at(1));
  auto rtn = Type::any_from_json(j_.at(2));
  return {std::move(x), std::move(y), std::move(rtn)};
}

json Intr::div_to_json(const Intr::Div &x_) {
  const auto x = Term::any_to_json(x_.x);
  const auto y = Term::any_to_json(x_.y);
  const auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, y, rtn});
}

Intr::Rem Intr::rem_from_json(const json &j_) {
  auto x = Term::any_from_json(j_.at(0));
  auto y = Term::any_from_json(j_.at(1));
  auto rtn = Type::any_from_json(j_.at(2));
  return {std::move(x), std::move(y), std::move(rtn)};
}

json Intr::rem_to_json(const Intr::Rem &x_) {
  const auto x = Term::any_to_json(x_.x);
  const auto y = Term::any_to_json(x_.y);
  const auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, y, rtn});
}

Intr::Min Intr::min_from_json(const json &j_) {
  auto x = Term::any_from_json(j_.at(0));
  auto y = Term::any_from_json(j_.at(1));
  auto rtn = Type::any_from_json(j_.at(2));
  return {std::move(x), std::move(y), std::move(rtn)};
}

json Intr::min_to_json(const Intr::Min &x_) {
  const auto x = Term::any_to_json(x_.x);
  const auto y = Term::any_to_json(x_.y);
  const auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, y, rtn});
}

Intr::Max Intr::max_from_json(const json &j_) {
  auto x = Term::any_from_json(j_.at(0));
  auto y = Term::any_from_json(j_.at(1));
  auto rtn = Type::any_from_json(j_.at(2));
  return {std::move(x), std::move(y), std::move(rtn)};
}

json Intr::max_to_json(const Intr::Max &x_) {
  const auto x = Term::any_to_json(x_.x);
  const auto y = Term::any_to_json(x_.y);
  const auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, y, rtn});
}

Intr::BAnd Intr::band_from_json(const json &j_) {
  auto x = Term::any_from_json(j_.at(0));
  auto y = Term::any_from_json(j_.at(1));
  auto rtn = Type::any_from_json(j_.at(2));
  return {std::move(x), std::move(y), std::move(rtn)};
}

json Intr::band_to_json(const Intr::BAnd &x_) {
  const auto x = Term::any_to_json(x_.x);
  const auto y = Term::any_to_json(x_.y);
  const auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, y, rtn});
}

Intr::BOr Intr::bor_from_json(const json &j_) {
  auto x = Term::any_from_json(j_.at(0));
  auto y = Term::any_from_json(j_.at(1));
  auto rtn = Type::any_from_json(j_.at(2));
  return {std::move(x), std::move(y), std::move(rtn)};
}

json Intr::bor_to_json(const Intr::BOr &x_) {
  const auto x = Term::any_to_json(x_.x);
  const auto y = Term::any_to_json(x_.y);
  const auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, y, rtn});
}

Intr::BXor Intr::bxor_from_json(const json &j_) {
  auto x = Term::any_from_json(j_.at(0));
  auto y = Term::any_from_json(j_.at(1));
  auto rtn = Type::any_from_json(j_.at(2));
  return {std::move(x), std::move(y), std::move(rtn)};
}

json Intr::bxor_to_json(const Intr::BXor &x_) {
  const auto x = Term::any_to_json(x_.x);
  const auto y = Term::any_to_json(x_.y);
  const auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, y, rtn});
}

Intr::BSL Intr::bsl_from_json(const json &j_) {
  auto x = Term::any_from_json(j_.at(0));
  auto y = Term::any_from_json(j_.at(1));
  auto rtn = Type::any_from_json(j_.at(2));
  return {std::move(x), std::move(y), std::move(rtn)};
}

json Intr::bsl_to_json(const Intr::BSL &x_) {
  const auto x = Term::any_to_json(x_.x);
  const auto y = Term::any_to_json(x_.y);
  const auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, y, rtn});
}

Intr::BSR Intr::bsr_from_json(const json &j_) {
  auto x = Term::any_from_json(j_.at(0));
  auto y = Term::any_from_json(j_.at(1));
  auto rtn = Type::any_from_json(j_.at(2));
  return {std::move(x), std::move(y), std::move(rtn)};
}

json Intr::bsr_to_json(const Intr::BSR &x_) {
  const auto x = Term::any_to_json(x_.x);
  const auto y = Term::any_to_json(x_.y);
  const auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, y, rtn});
}

Intr::BZSR Intr::bzsr_from_json(const json &j_) {
  auto x = Term::any_from_json(j_.at(0));
  auto y = Term::any_from_json(j_.at(1));
  auto rtn = Type::any_from_json(j_.at(2));
  return {std::move(x), std::move(y), std::move(rtn)};
}

json Intr::bzsr_to_json(const Intr::BZSR &x_) {
  const auto x = Term::any_to_json(x_.x);
  const auto y = Term::any_to_json(x_.y);
  const auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, y, rtn});
}

Intr::PopCount Intr::popcount_from_json(const json &j_) {
  auto x = Term::any_from_json(j_.at(0));
  auto rtn = Type::any_from_json(j_.at(1));
  return {std::move(x), std::move(rtn)};
}

json Intr::popcount_to_json(const Intr::PopCount &x_) {
  const auto x = Term::any_to_json(x_.x);
  const auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, rtn});
}

Intr::LogicAnd Intr::logicand_from_json(const json &j_) {
  auto x = Term::any_from_json(j_.at(0));
  auto y = Term::any_from_json(j_.at(1));
  return {std::move(x), std::move(y)};
}

json Intr::logicand_to_json(const Intr::LogicAnd &x_) {
  const auto x = Term::any_to_json(x_.x);
  const auto y = Term::any_to_json(x_.y);
  return json::array({x, y});
}

Intr::LogicOr Intr::logicor_from_json(const json &j_) {
  auto x = Term::any_from_json(j_.at(0));
  auto y = Term::any_from_json(j_.at(1));
  return {std::move(x), std::move(y)};
}

json Intr::logicor_to_json(const Intr::LogicOr &x_) {
  const auto x = Term::any_to_json(x_.x);
  const auto y = Term::any_to_json(x_.y);
  return json::array({x, y});
}

Intr::LogicEq Intr::logiceq_from_json(const json &j_) {
  auto x = Term::any_from_json(j_.at(0));
  auto y = Term::any_from_json(j_.at(1));
  return {std::move(x), std::move(y)};
}

json Intr::logiceq_to_json(const Intr::LogicEq &x_) {
  const auto x = Term::any_to_json(x_.x);
  const auto y = Term::any_to_json(x_.y);
  return json::array({x, y});
}

Intr::LogicNeq Intr::logicneq_from_json(const json &j_) {
  auto x = Term::any_from_json(j_.at(0));
  auto y = Term::any_from_json(j_.at(1));
  return {std::move(x), std::move(y)};
}

json Intr::logicneq_to_json(const Intr::LogicNeq &x_) {
  const auto x = Term::any_to_json(x_.x);
  const auto y = Term::any_to_json(x_.y);
  return json::array({x, y});
}

Intr::LogicLte Intr::logiclte_from_json(const json &j_) {
  auto x = Term::any_from_json(j_.at(0));
  auto y = Term::any_from_json(j_.at(1));
  return {std::move(x), std::move(y)};
}

json Intr::logiclte_to_json(const Intr::LogicLte &x_) {
  const auto x = Term::any_to_json(x_.x);
  const auto y = Term::any_to_json(x_.y);
  return json::array({x, y});
}

Intr::LogicGte Intr::logicgte_from_json(const json &j_) {
  auto x = Term::any_from_json(j_.at(0));
  auto y = Term::any_from_json(j_.at(1));
  return {std::move(x), std::move(y)};
}

json Intr::logicgte_to_json(const Intr::LogicGte &x_) {
  const auto x = Term::any_to_json(x_.x);
  const auto y = Term::any_to_json(x_.y);
  return json::array({x, y});
}

Intr::LogicLt Intr::logiclt_from_json(const json &j_) {
  auto x = Term::any_from_json(j_.at(0));
  auto y = Term::any_from_json(j_.at(1));
  return {std::move(x), std::move(y)};
}

json Intr::logiclt_to_json(const Intr::LogicLt &x_) {
  const auto x = Term::any_to_json(x_.x);
  const auto y = Term::any_to_json(x_.y);
  return json::array({x, y});
}

Intr::LogicGt Intr::logicgt_from_json(const json &j_) {
  auto x = Term::any_from_json(j_.at(0));
  auto y = Term::any_from_json(j_.at(1));
  return {std::move(x), std::move(y)};
}

json Intr::logicgt_to_json(const Intr::LogicGt &x_) {
  const auto x = Term::any_to_json(x_.x);
  const auto y = Term::any_to_json(x_.y);
  return json::array({x, y});
}

Intr::Any Intr::any_from_json(const json &j_) {
  const auto ord_ = j_.at(0).get<size_t>();
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
    case 17: return Intr::popcount_from_json(t_);
    case 18: return Intr::logicand_from_json(t_);
    case 19: return Intr::logicor_from_json(t_);
    case 20: return Intr::logiceq_from_json(t_);
    case 21: return Intr::logicneq_from_json(t_);
    case 22: return Intr::logiclte_from_json(t_);
    case 23: return Intr::logicgte_from_json(t_);
    case 24: return Intr::logiclt_from_json(t_);
    case 25: return Intr::logicgt_from_json(t_);
    default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
  }
}

json Intr::any_to_json(const Intr::Any &x_) {
  return x_.match_total([](const Intr::BNot &y_) -> json { return {0, Intr::bnot_to_json(y_)}; },
                        [](const Intr::LogicNot &y_) -> json { return {1, Intr::logicnot_to_json(y_)}; },
                        [](const Intr::Pos &y_) -> json { return {2, Intr::pos_to_json(y_)}; },
                        [](const Intr::Neg &y_) -> json { return {3, Intr::neg_to_json(y_)}; },
                        [](const Intr::Add &y_) -> json { return {4, Intr::add_to_json(y_)}; },
                        [](const Intr::Sub &y_) -> json { return {5, Intr::sub_to_json(y_)}; },
                        [](const Intr::Mul &y_) -> json { return {6, Intr::mul_to_json(y_)}; },
                        [](const Intr::Div &y_) -> json { return {7, Intr::div_to_json(y_)}; },
                        [](const Intr::Rem &y_) -> json { return {8, Intr::rem_to_json(y_)}; },
                        [](const Intr::Min &y_) -> json { return {9, Intr::min_to_json(y_)}; },
                        [](const Intr::Max &y_) -> json { return {10, Intr::max_to_json(y_)}; },
                        [](const Intr::BAnd &y_) -> json { return {11, Intr::band_to_json(y_)}; },
                        [](const Intr::BOr &y_) -> json { return {12, Intr::bor_to_json(y_)}; },
                        [](const Intr::BXor &y_) -> json { return {13, Intr::bxor_to_json(y_)}; },
                        [](const Intr::BSL &y_) -> json { return {14, Intr::bsl_to_json(y_)}; },
                        [](const Intr::BSR &y_) -> json { return {15, Intr::bsr_to_json(y_)}; },
                        [](const Intr::BZSR &y_) -> json { return {16, Intr::bzsr_to_json(y_)}; },
                        [](const Intr::PopCount &y_) -> json { return {17, Intr::popcount_to_json(y_)}; },
                        [](const Intr::LogicAnd &y_) -> json { return {18, Intr::logicand_to_json(y_)}; },
                        [](const Intr::LogicOr &y_) -> json { return {19, Intr::logicor_to_json(y_)}; },
                        [](const Intr::LogicEq &y_) -> json { return {20, Intr::logiceq_to_json(y_)}; },
                        [](const Intr::LogicNeq &y_) -> json { return {21, Intr::logicneq_to_json(y_)}; },
                        [](const Intr::LogicLte &y_) -> json { return {22, Intr::logiclte_to_json(y_)}; },
                        [](const Intr::LogicGte &y_) -> json { return {23, Intr::logicgte_to_json(y_)}; },
                        [](const Intr::LogicLt &y_) -> json { return {24, Intr::logiclt_to_json(y_)}; },
                        [](const Intr::LogicGt &y_) -> json { return {25, Intr::logicgt_to_json(y_)}; });
}

Math::Abs Math::abs_from_json(const json &j_) {
  auto x = Term::any_from_json(j_.at(0));
  auto rtn = Type::any_from_json(j_.at(1));
  return {std::move(x), std::move(rtn)};
}

json Math::abs_to_json(const Math::Abs &x_) {
  const auto x = Term::any_to_json(x_.x);
  const auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, rtn});
}

Math::Sin Math::sin_from_json(const json &j_) {
  auto x = Term::any_from_json(j_.at(0));
  auto rtn = Type::any_from_json(j_.at(1));
  return {std::move(x), std::move(rtn)};
}

json Math::sin_to_json(const Math::Sin &x_) {
  const auto x = Term::any_to_json(x_.x);
  const auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, rtn});
}

Math::Cos Math::cos_from_json(const json &j_) {
  auto x = Term::any_from_json(j_.at(0));
  auto rtn = Type::any_from_json(j_.at(1));
  return {std::move(x), std::move(rtn)};
}

json Math::cos_to_json(const Math::Cos &x_) {
  const auto x = Term::any_to_json(x_.x);
  const auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, rtn});
}

Math::Tan Math::tan_from_json(const json &j_) {
  auto x = Term::any_from_json(j_.at(0));
  auto rtn = Type::any_from_json(j_.at(1));
  return {std::move(x), std::move(rtn)};
}

json Math::tan_to_json(const Math::Tan &x_) {
  const auto x = Term::any_to_json(x_.x);
  const auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, rtn});
}

Math::Asin Math::asin_from_json(const json &j_) {
  auto x = Term::any_from_json(j_.at(0));
  auto rtn = Type::any_from_json(j_.at(1));
  return {std::move(x), std::move(rtn)};
}

json Math::asin_to_json(const Math::Asin &x_) {
  const auto x = Term::any_to_json(x_.x);
  const auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, rtn});
}

Math::Acos Math::acos_from_json(const json &j_) {
  auto x = Term::any_from_json(j_.at(0));
  auto rtn = Type::any_from_json(j_.at(1));
  return {std::move(x), std::move(rtn)};
}

json Math::acos_to_json(const Math::Acos &x_) {
  const auto x = Term::any_to_json(x_.x);
  const auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, rtn});
}

Math::Atan Math::atan_from_json(const json &j_) {
  auto x = Term::any_from_json(j_.at(0));
  auto rtn = Type::any_from_json(j_.at(1));
  return {std::move(x), std::move(rtn)};
}

json Math::atan_to_json(const Math::Atan &x_) {
  const auto x = Term::any_to_json(x_.x);
  const auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, rtn});
}

Math::Sinh Math::sinh_from_json(const json &j_) {
  auto x = Term::any_from_json(j_.at(0));
  auto rtn = Type::any_from_json(j_.at(1));
  return {std::move(x), std::move(rtn)};
}

json Math::sinh_to_json(const Math::Sinh &x_) {
  const auto x = Term::any_to_json(x_.x);
  const auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, rtn});
}

Math::Cosh Math::cosh_from_json(const json &j_) {
  auto x = Term::any_from_json(j_.at(0));
  auto rtn = Type::any_from_json(j_.at(1));
  return {std::move(x), std::move(rtn)};
}

json Math::cosh_to_json(const Math::Cosh &x_) {
  const auto x = Term::any_to_json(x_.x);
  const auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, rtn});
}

Math::Tanh Math::tanh_from_json(const json &j_) {
  auto x = Term::any_from_json(j_.at(0));
  auto rtn = Type::any_from_json(j_.at(1));
  return {std::move(x), std::move(rtn)};
}

json Math::tanh_to_json(const Math::Tanh &x_) {
  const auto x = Term::any_to_json(x_.x);
  const auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, rtn});
}

Math::Signum Math::signum_from_json(const json &j_) {
  auto x = Term::any_from_json(j_.at(0));
  auto rtn = Type::any_from_json(j_.at(1));
  return {std::move(x), std::move(rtn)};
}

json Math::signum_to_json(const Math::Signum &x_) {
  const auto x = Term::any_to_json(x_.x);
  const auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, rtn});
}

Math::Round Math::round_from_json(const json &j_) {
  auto x = Term::any_from_json(j_.at(0));
  auto rtn = Type::any_from_json(j_.at(1));
  return {std::move(x), std::move(rtn)};
}

json Math::round_to_json(const Math::Round &x_) {
  const auto x = Term::any_to_json(x_.x);
  const auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, rtn});
}

Math::Ceil Math::ceil_from_json(const json &j_) {
  auto x = Term::any_from_json(j_.at(0));
  auto rtn = Type::any_from_json(j_.at(1));
  return {std::move(x), std::move(rtn)};
}

json Math::ceil_to_json(const Math::Ceil &x_) {
  const auto x = Term::any_to_json(x_.x);
  const auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, rtn});
}

Math::Floor Math::floor_from_json(const json &j_) {
  auto x = Term::any_from_json(j_.at(0));
  auto rtn = Type::any_from_json(j_.at(1));
  return {std::move(x), std::move(rtn)};
}

json Math::floor_to_json(const Math::Floor &x_) {
  const auto x = Term::any_to_json(x_.x);
  const auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, rtn});
}

Math::Rint Math::rint_from_json(const json &j_) {
  auto x = Term::any_from_json(j_.at(0));
  auto rtn = Type::any_from_json(j_.at(1));
  return {std::move(x), std::move(rtn)};
}

json Math::rint_to_json(const Math::Rint &x_) {
  const auto x = Term::any_to_json(x_.x);
  const auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, rtn});
}

Math::Sqrt Math::sqrt_from_json(const json &j_) {
  auto x = Term::any_from_json(j_.at(0));
  auto rtn = Type::any_from_json(j_.at(1));
  return {std::move(x), std::move(rtn)};
}

json Math::sqrt_to_json(const Math::Sqrt &x_) {
  const auto x = Term::any_to_json(x_.x);
  const auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, rtn});
}

Math::Cbrt Math::cbrt_from_json(const json &j_) {
  auto x = Term::any_from_json(j_.at(0));
  auto rtn = Type::any_from_json(j_.at(1));
  return {std::move(x), std::move(rtn)};
}

json Math::cbrt_to_json(const Math::Cbrt &x_) {
  const auto x = Term::any_to_json(x_.x);
  const auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, rtn});
}

Math::Exp Math::exp_from_json(const json &j_) {
  auto x = Term::any_from_json(j_.at(0));
  auto rtn = Type::any_from_json(j_.at(1));
  return {std::move(x), std::move(rtn)};
}

json Math::exp_to_json(const Math::Exp &x_) {
  const auto x = Term::any_to_json(x_.x);
  const auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, rtn});
}

Math::Expm1 Math::expm1_from_json(const json &j_) {
  auto x = Term::any_from_json(j_.at(0));
  auto rtn = Type::any_from_json(j_.at(1));
  return {std::move(x), std::move(rtn)};
}

json Math::expm1_to_json(const Math::Expm1 &x_) {
  const auto x = Term::any_to_json(x_.x);
  const auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, rtn});
}

Math::Log Math::log_from_json(const json &j_) {
  auto x = Term::any_from_json(j_.at(0));
  auto rtn = Type::any_from_json(j_.at(1));
  return {std::move(x), std::move(rtn)};
}

json Math::log_to_json(const Math::Log &x_) {
  const auto x = Term::any_to_json(x_.x);
  const auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, rtn});
}

Math::Log1p Math::log1p_from_json(const json &j_) {
  auto x = Term::any_from_json(j_.at(0));
  auto rtn = Type::any_from_json(j_.at(1));
  return {std::move(x), std::move(rtn)};
}

json Math::log1p_to_json(const Math::Log1p &x_) {
  const auto x = Term::any_to_json(x_.x);
  const auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, rtn});
}

Math::Log10 Math::log10_from_json(const json &j_) {
  auto x = Term::any_from_json(j_.at(0));
  auto rtn = Type::any_from_json(j_.at(1));
  return {std::move(x), std::move(rtn)};
}

json Math::log10_to_json(const Math::Log10 &x_) {
  const auto x = Term::any_to_json(x_.x);
  const auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, rtn});
}

Math::Pow Math::pow_from_json(const json &j_) {
  auto x = Term::any_from_json(j_.at(0));
  auto y = Term::any_from_json(j_.at(1));
  auto rtn = Type::any_from_json(j_.at(2));
  return {std::move(x), std::move(y), std::move(rtn)};
}

json Math::pow_to_json(const Math::Pow &x_) {
  const auto x = Term::any_to_json(x_.x);
  const auto y = Term::any_to_json(x_.y);
  const auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, y, rtn});
}

Math::Atan2 Math::atan2_from_json(const json &j_) {
  auto x = Term::any_from_json(j_.at(0));
  auto y = Term::any_from_json(j_.at(1));
  auto rtn = Type::any_from_json(j_.at(2));
  return {std::move(x), std::move(y), std::move(rtn)};
}

json Math::atan2_to_json(const Math::Atan2 &x_) {
  const auto x = Term::any_to_json(x_.x);
  const auto y = Term::any_to_json(x_.y);
  const auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, y, rtn});
}

Math::Hypot Math::hypot_from_json(const json &j_) {
  auto x = Term::any_from_json(j_.at(0));
  auto y = Term::any_from_json(j_.at(1));
  auto rtn = Type::any_from_json(j_.at(2));
  return {std::move(x), std::move(y), std::move(rtn)};
}

json Math::hypot_to_json(const Math::Hypot &x_) {
  const auto x = Term::any_to_json(x_.x);
  const auto y = Term::any_to_json(x_.y);
  const auto rtn = Type::any_to_json(x_.rtn);
  return json::array({x, y, rtn});
}

Math::Any Math::any_from_json(const json &j_) {
  const auto ord_ = j_.at(0).get<size_t>();
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

json Math::any_to_json(const Math::Any &x_) {
  return x_.match_total([](const Math::Abs &y_) -> json { return {0, Math::abs_to_json(y_)}; },
                        [](const Math::Sin &y_) -> json { return {1, Math::sin_to_json(y_)}; },
                        [](const Math::Cos &y_) -> json { return {2, Math::cos_to_json(y_)}; },
                        [](const Math::Tan &y_) -> json { return {3, Math::tan_to_json(y_)}; },
                        [](const Math::Asin &y_) -> json { return {4, Math::asin_to_json(y_)}; },
                        [](const Math::Acos &y_) -> json { return {5, Math::acos_to_json(y_)}; },
                        [](const Math::Atan &y_) -> json { return {6, Math::atan_to_json(y_)}; },
                        [](const Math::Sinh &y_) -> json { return {7, Math::sinh_to_json(y_)}; },
                        [](const Math::Cosh &y_) -> json { return {8, Math::cosh_to_json(y_)}; },
                        [](const Math::Tanh &y_) -> json { return {9, Math::tanh_to_json(y_)}; },
                        [](const Math::Signum &y_) -> json { return {10, Math::signum_to_json(y_)}; },
                        [](const Math::Round &y_) -> json { return {11, Math::round_to_json(y_)}; },
                        [](const Math::Ceil &y_) -> json { return {12, Math::ceil_to_json(y_)}; },
                        [](const Math::Floor &y_) -> json { return {13, Math::floor_to_json(y_)}; },
                        [](const Math::Rint &y_) -> json { return {14, Math::rint_to_json(y_)}; },
                        [](const Math::Sqrt &y_) -> json { return {15, Math::sqrt_to_json(y_)}; },
                        [](const Math::Cbrt &y_) -> json { return {16, Math::cbrt_to_json(y_)}; },
                        [](const Math::Exp &y_) -> json { return {17, Math::exp_to_json(y_)}; },
                        [](const Math::Expm1 &y_) -> json { return {18, Math::expm1_to_json(y_)}; },
                        [](const Math::Log &y_) -> json { return {19, Math::log_to_json(y_)}; },
                        [](const Math::Log1p &y_) -> json { return {20, Math::log1p_to_json(y_)}; },
                        [](const Math::Log10 &y_) -> json { return {21, Math::log10_to_json(y_)}; },
                        [](const Math::Pow &y_) -> json { return {22, Math::pow_to_json(y_)}; },
                        [](const Math::Atan2 &y_) -> json { return {23, Math::atan2_to_json(y_)}; },
                        [](const Math::Hypot &y_) -> json { return {24, Math::hypot_to_json(y_)}; });
}

ExceptionKind exceptionkind_from_json(const json &j_) {
  auto tpe = Type::any_from_json(j_.at(0));
  auto sourceName = j_.at(1).get<std::string>();
  return {std::move(tpe), std::move(sourceName)};
}

json exceptionkind_to_json(const ExceptionKind &x_) {
  const auto tpe = Type::any_to_json(x_.tpe);
  const auto sourceName = x_.sourceName;
  return json::array({tpe, sourceName});
}

Handler handler_from_json(const json &j_) {
  auto caught = j_.at(0).is_null() ? std::nullopt : std::make_optional(exceptionkind_from_json(j_.at(0)));
  auto binder = j_.at(1).is_null() ? std::nullopt : std::make_optional(named_from_json(j_.at(1)));
  std::vector<Stmt::Any> body;
  body.reserve(j_.at(2).size());
  for (const auto &v_ : j_.at(2)) {
    body.emplace_back(Stmt::any_from_json(v_));
  }
  return {std::move(caught), std::move(binder), std::move(body)};
}

json handler_to_json(const Handler &x_) {
  const auto caught = x_.caught ? exceptionkind_to_json(*x_.caught) : json();
  const auto binder = x_.binder ? named_to_json(*x_.binder) : json();
  std::vector<json> body;
  body.reserve(x_.body.size());
  for (const auto &v_ : x_.body) {
    body.emplace_back(Stmt::any_to_json(v_));
  }
  return json::array({caught, binder, body});
}

Stmt::Var Stmt::var_from_json(const json &j_) {
  auto name = named_from_json(j_.at(0));
  auto expr = j_.at(1).is_null() ? std::nullopt : std::make_optional(Expr::any_from_json(j_.at(1)));
  const auto isMutable = j_.at(2).get<bool>();
  return {std::move(name), std::move(expr), isMutable};
}

json Stmt::var_to_json(const Stmt::Var &x_) {
  const auto name = named_to_json(x_.name);
  const auto expr = x_.expr ? Expr::any_to_json(*x_.expr) : json();
  const auto isMutable = x_.isMutable;
  return json::array({name, expr, isMutable});
}

Stmt::Mut Stmt::mut_from_json(const json &j_) {
  auto name = Term::select_from_json(j_.at(0));
  auto expr = Expr::any_from_json(j_.at(1));
  return {std::move(name), std::move(expr)};
}

json Stmt::mut_to_json(const Stmt::Mut &x_) {
  const auto name = Term::select_to_json(x_.name);
  const auto expr = Expr::any_to_json(x_.expr);
  return json::array({name, expr});
}

Stmt::Update Stmt::update_from_json(const json &j_) {
  auto lhs = Term::select_from_json(j_.at(0));
  auto idx = Term::any_from_json(j_.at(1));
  auto value = Term::any_from_json(j_.at(2));
  return {std::move(lhs), std::move(idx), std::move(value)};
}

json Stmt::update_to_json(const Stmt::Update &x_) {
  const auto lhs = Term::select_to_json(x_.lhs);
  const auto idx = Term::any_to_json(x_.idx);
  const auto value = Term::any_to_json(x_.value);
  return json::array({lhs, idx, value});
}

Stmt::While Stmt::while_from_json(const json &j_) {
  auto cond = Term::any_from_json(j_.at(0));
  std::vector<Stmt::Any> body;
  body.reserve(j_.at(1).size());
  for (const auto &v_ : j_.at(1)) {
    body.emplace_back(Stmt::any_from_json(v_));
  }
  return {std::move(cond), std::move(body)};
}

json Stmt::while_to_json(const Stmt::While &x_) {
  const auto cond = Term::any_to_json(x_.cond);
  std::vector<json> body;
  body.reserve(x_.body.size());
  for (const auto &v_ : x_.body) {
    body.emplace_back(Stmt::any_to_json(v_));
  }
  return json::array({cond, body});
}

Stmt::ForRange Stmt::forrange_from_json(const json &j_) {
  auto induction = named_from_json(j_.at(0));
  auto lbIncl = Term::any_from_json(j_.at(1));
  auto ubExcl = Term::any_from_json(j_.at(2));
  auto step = Term::any_from_json(j_.at(3));
  std::vector<Stmt::Any> body;
  body.reserve(j_.at(4).size());
  for (const auto &v_ : j_.at(4)) {
    body.emplace_back(Stmt::any_from_json(v_));
  }
  return {std::move(induction), std::move(lbIncl), std::move(ubExcl), std::move(step), std::move(body)};
}

json Stmt::forrange_to_json(const Stmt::ForRange &x_) {
  const auto induction = named_to_json(x_.induction);
  const auto lbIncl = Term::any_to_json(x_.lbIncl);
  const auto ubExcl = Term::any_to_json(x_.ubExcl);
  const auto step = Term::any_to_json(x_.step);
  std::vector<json> body;
  body.reserve(x_.body.size());
  for (const auto &v_ : x_.body) {
    body.emplace_back(Stmt::any_to_json(v_));
  }
  return json::array({induction, lbIncl, ubExcl, step, body});
}

Stmt::Break Stmt::break_from_json(const json &j_) { return {}; }

json Stmt::break_to_json(const Stmt::Break &x_) { return json::array({}); }

Stmt::Cont Stmt::cont_from_json(const json &j_) { return {}; }

json Stmt::cont_to_json(const Stmt::Cont &x_) { return json::array({}); }

Stmt::Cond Stmt::cond_from_json(const json &j_) {
  auto cond = Term::any_from_json(j_.at(0));
  std::vector<Stmt::Any> trueBr;
  trueBr.reserve(j_.at(1).size());
  for (const auto &v_ : j_.at(1)) {
    trueBr.emplace_back(Stmt::any_from_json(v_));
  }
  std::vector<Stmt::Any> falseBr;
  falseBr.reserve(j_.at(2).size());
  for (const auto &v_ : j_.at(2)) {
    falseBr.emplace_back(Stmt::any_from_json(v_));
  }
  return {std::move(cond), std::move(trueBr), std::move(falseBr)};
}

json Stmt::cond_to_json(const Stmt::Cond &x_) {
  const auto cond = Term::any_to_json(x_.cond);
  std::vector<json> trueBr;
  trueBr.reserve(x_.trueBr.size());
  for (const auto &v_ : x_.trueBr) {
    trueBr.emplace_back(Stmt::any_to_json(v_));
  }
  std::vector<json> falseBr;
  falseBr.reserve(x_.falseBr.size());
  for (const auto &v_ : x_.falseBr) {
    falseBr.emplace_back(Stmt::any_to_json(v_));
  }
  return json::array({cond, trueBr, falseBr});
}

Stmt::Return Stmt::return_from_json(const json &j_) {
  auto value = Expr::any_from_json(j_.at(0));
  return Stmt::Return(std::move(value));
}

json Stmt::return_to_json(const Stmt::Return &x_) {
  const auto value = Expr::any_to_json(x_.value);
  return json::array({value});
}

Stmt::Annotated Stmt::annotated_from_json(const json &j_) {
  auto inner = Stmt::any_from_json(j_.at(0));
  auto pos = j_.at(1).is_null() ? std::nullopt : std::make_optional(sourceposition_from_json(j_.at(1)));
  auto comment = j_.at(2).is_null() ? std::nullopt : std::make_optional(j_.at(2).get<std::string>());
  return {std::move(inner), std::move(pos), std::move(comment)};
}

json Stmt::annotated_to_json(const Stmt::Annotated &x_) {
  const auto inner = Stmt::any_to_json(x_.inner);
  const auto pos = x_.pos ? sourceposition_to_json(*x_.pos) : json();
  const auto comment = x_.comment ? json(*x_.comment) : json();
  return json::array({inner, pos, comment});
}

Stmt::Try Stmt::try_from_json(const json &j_) {
  std::vector<Stmt::Any> body;
  body.reserve(j_.at(0).size());
  for (const auto &v_ : j_.at(0)) {
    body.emplace_back(Stmt::any_from_json(v_));
  }
  std::vector<Handler> handlers;
  handlers.reserve(j_.at(1).size());
  for (const auto &v_ : j_.at(1)) {
    handlers.emplace_back(handler_from_json(v_));
  }
  std::vector<Stmt::Any> fin;
  fin.reserve(j_.at(2).size());
  for (const auto &v_ : j_.at(2)) {
    fin.emplace_back(Stmt::any_from_json(v_));
  }
  return {std::move(body), std::move(handlers), std::move(fin)};
}

json Stmt::try_to_json(const Stmt::Try &x_) {
  std::vector<json> body;
  body.reserve(x_.body.size());
  for (const auto &v_ : x_.body) {
    body.emplace_back(Stmt::any_to_json(v_));
  }
  std::vector<json> handlers;
  handlers.reserve(x_.handlers.size());
  for (const auto &v_ : x_.handlers) {
    handlers.emplace_back(handler_to_json(v_));
  }
  std::vector<json> fin;
  fin.reserve(x_.fin.size());
  for (const auto &v_ : x_.fin) {
    fin.emplace_back(Stmt::any_to_json(v_));
  }
  return json::array({body, handlers, fin});
}

Stmt::Raise Stmt::raise_from_json(const json &j_) {
  auto value = Term::any_from_json(j_.at(0));
  auto exceptionKind = exceptionkind_from_json(j_.at(1));
  std::vector<Stmt::Any> cleanup;
  cleanup.reserve(j_.at(2).size());
  for (const auto &v_ : j_.at(2)) {
    cleanup.emplace_back(Stmt::any_from_json(v_));
  }
  return {std::move(value), std::move(exceptionKind), std::move(cleanup)};
}

json Stmt::raise_to_json(const Stmt::Raise &x_) {
  const auto value = Term::any_to_json(x_.value);
  const auto exceptionKind = exceptionkind_to_json(x_.exceptionKind);
  std::vector<json> cleanup;
  cleanup.reserve(x_.cleanup.size());
  for (const auto &v_ : x_.cleanup) {
    cleanup.emplace_back(Stmt::any_to_json(v_));
  }
  return json::array({value, exceptionKind, cleanup});
}

Stmt::Rethrow Stmt::rethrow_from_json(const json &j_) { return {}; }

json Stmt::rethrow_to_json(const Stmt::Rethrow &x_) { return json::array({}); }

Stmt::Any Stmt::any_from_json(const json &j_) {
  const auto ord_ = j_.at(0).get<size_t>();
  const auto &t_ = j_.at(1);
  switch (ord_) {
    case 0: return Stmt::var_from_json(t_);
    case 1: return Stmt::mut_from_json(t_);
    case 2: return Stmt::update_from_json(t_);
    case 3: return Stmt::while_from_json(t_);
    case 4: return Stmt::forrange_from_json(t_);
    case 5: return Stmt::break_from_json(t_);
    case 6: return Stmt::cont_from_json(t_);
    case 7: return Stmt::cond_from_json(t_);
    case 8: return Stmt::return_from_json(t_);
    case 9: return Stmt::annotated_from_json(t_);
    case 10: return Stmt::try_from_json(t_);
    case 11: return Stmt::raise_from_json(t_);
    case 12: return Stmt::rethrow_from_json(t_);
    default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
  }
}

json Stmt::any_to_json(const Stmt::Any &x_) {
  return x_.match_total([](const Stmt::Var &y_) -> json { return {0, Stmt::var_to_json(y_)}; },
                        [](const Stmt::Mut &y_) -> json { return {1, Stmt::mut_to_json(y_)}; },
                        [](const Stmt::Update &y_) -> json { return {2, Stmt::update_to_json(y_)}; },
                        [](const Stmt::While &y_) -> json { return {3, Stmt::while_to_json(y_)}; },
                        [](const Stmt::ForRange &y_) -> json { return {4, Stmt::forrange_to_json(y_)}; },
                        [](const Stmt::Break &y_) -> json { return {5, Stmt::break_to_json(y_)}; },
                        [](const Stmt::Cont &y_) -> json { return {6, Stmt::cont_to_json(y_)}; },
                        [](const Stmt::Cond &y_) -> json { return {7, Stmt::cond_to_json(y_)}; },
                        [](const Stmt::Return &y_) -> json { return {8, Stmt::return_to_json(y_)}; },
                        [](const Stmt::Annotated &y_) -> json { return {9, Stmt::annotated_to_json(y_)}; },
                        [](const Stmt::Try &y_) -> json { return {10, Stmt::try_to_json(y_)}; },
                        [](const Stmt::Raise &y_) -> json { return {11, Stmt::raise_to_json(y_)}; },
                        [](const Stmt::Rethrow &y_) -> json { return {12, Stmt::rethrow_to_json(y_)}; });
}

Signature signature_from_json(const json &j_) {
  auto name = sym_from_json(j_.at(0));
  std::vector<Type::Var> tpeVars;
  tpeVars.reserve(j_.at(1).size());
  for (const auto &v_ : j_.at(1)) {
    tpeVars.emplace_back(Type::var_from_json(v_));
  }
  auto receiver = j_.at(2).is_null() ? std::nullopt : std::make_optional(Type::any_from_json(j_.at(2)));
  std::vector<Type::Any> args;
  args.reserve(j_.at(3).size());
  for (const auto &v_ : j_.at(3)) {
    args.emplace_back(Type::any_from_json(v_));
  }
  std::vector<Type::Any> moduleCaptures;
  moduleCaptures.reserve(j_.at(4).size());
  for (const auto &v_ : j_.at(4)) {
    moduleCaptures.emplace_back(Type::any_from_json(v_));
  }
  std::vector<Type::Any> termCaptures;
  termCaptures.reserve(j_.at(5).size());
  for (const auto &v_ : j_.at(5)) {
    termCaptures.emplace_back(Type::any_from_json(v_));
  }
  auto rtn = Type::any_from_json(j_.at(6));
  return {std::move(name),           std::move(tpeVars),      std::move(receiver), std::move(args),
          std::move(moduleCaptures), std::move(termCaptures), std::move(rtn)};
}

json signature_to_json(const Signature &x_) {
  const auto name = sym_to_json(x_.name);
  std::vector<json> tpeVars;
  tpeVars.reserve(x_.tpeVars.size());
  for (const auto &v_ : x_.tpeVars) {
    tpeVars.emplace_back(Type::var_to_json(v_));
  }
  const auto receiver = x_.receiver ? Type::any_to_json(*x_.receiver) : json();
  std::vector<json> args;
  args.reserve(x_.args.size());
  for (const auto &v_ : x_.args) {
    args.emplace_back(Type::any_to_json(v_));
  }
  std::vector<json> moduleCaptures;
  moduleCaptures.reserve(x_.moduleCaptures.size());
  for (const auto &v_ : x_.moduleCaptures) {
    moduleCaptures.emplace_back(Type::any_to_json(v_));
  }
  std::vector<json> termCaptures;
  termCaptures.reserve(x_.termCaptures.size());
  for (const auto &v_ : x_.termCaptures) {
    termCaptures.emplace_back(Type::any_to_json(v_));
  }
  const auto rtn = Type::any_to_json(x_.rtn);
  return json::array({name, tpeVars, receiver, args, moduleCaptures, termCaptures, rtn});
}

FunctionVisibility::Internal FunctionVisibility::internal_from_json(const json &j_) { return {}; }

json FunctionVisibility::internal_to_json(const FunctionVisibility::Internal &x_) { return json::array({}); }

FunctionVisibility::Exported FunctionVisibility::exported_from_json(const json &j_) { return {}; }

json FunctionVisibility::exported_to_json(const FunctionVisibility::Exported &x_) { return json::array({}); }

FunctionVisibility::Any FunctionVisibility::any_from_json(const json &j_) {
  const auto ord_ = j_.at(0).get<size_t>();
  const auto &t_ = j_.at(1);
  switch (ord_) {
    case 0: return FunctionVisibility::internal_from_json(t_);
    case 1: return FunctionVisibility::exported_from_json(t_);
    default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
  }
}

json FunctionVisibility::any_to_json(const FunctionVisibility::Any &x_) {
  return x_.match_total([](const FunctionVisibility::Internal &y_) -> json { return {0, FunctionVisibility::internal_to_json(y_)}; },
                        [](const FunctionVisibility::Exported &y_) -> json { return {1, FunctionVisibility::exported_to_json(y_)}; });
}

FunctionFpMode::Relaxed FunctionFpMode::relaxed_from_json(const json &j_) { return {}; }

json FunctionFpMode::relaxed_to_json(const FunctionFpMode::Relaxed &x_) { return json::array({}); }

FunctionFpMode::Strict FunctionFpMode::strict_from_json(const json &j_) { return {}; }

json FunctionFpMode::strict_to_json(const FunctionFpMode::Strict &x_) { return json::array({}); }

FunctionFpMode::Any FunctionFpMode::any_from_json(const json &j_) {
  const auto ord_ = j_.at(0).get<size_t>();
  const auto &t_ = j_.at(1);
  switch (ord_) {
    case 0: return FunctionFpMode::relaxed_from_json(t_);
    case 1: return FunctionFpMode::strict_from_json(t_);
    default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
  }
}

json FunctionFpMode::any_to_json(const FunctionFpMode::Any &x_) {
  return x_.match_total([](const FunctionFpMode::Relaxed &y_) -> json { return {0, FunctionFpMode::relaxed_to_json(y_)}; },
                        [](const FunctionFpMode::Strict &y_) -> json { return {1, FunctionFpMode::strict_to_json(y_)}; });
}

FunctionAffinity::Offload FunctionAffinity::offload_from_json(const json &j_) { return {}; }

json FunctionAffinity::offload_to_json(const FunctionAffinity::Offload &x_) { return json::array({}); }

FunctionAffinity::Host FunctionAffinity::host_from_json(const json &j_) { return {}; }

json FunctionAffinity::host_to_json(const FunctionAffinity::Host &x_) { return json::array({}); }

FunctionAffinity::Any FunctionAffinity::any_from_json(const json &j_) {
  const auto ord_ = j_.at(0).get<size_t>();
  const auto &t_ = j_.at(1);
  switch (ord_) {
    case 0: return FunctionAffinity::offload_from_json(t_);
    case 1: return FunctionAffinity::host_from_json(t_);
    default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
  }
}

json FunctionAffinity::any_to_json(const FunctionAffinity::Any &x_) {
  return x_.match_total([](const FunctionAffinity::Offload &y_) -> json { return {0, FunctionAffinity::offload_to_json(y_)}; },
                        [](const FunctionAffinity::Host &y_) -> json { return {1, FunctionAffinity::host_to_json(y_)}; });
}

ArgAccess::Read ArgAccess::read_from_json(const json &j_) { return {}; }

json ArgAccess::read_to_json(const ArgAccess::Read &x_) { return json::array({}); }

ArgAccess::Write ArgAccess::write_from_json(const json &j_) { return {}; }

json ArgAccess::write_to_json(const ArgAccess::Write &x_) { return json::array({}); }

ArgAccess::ReadWrite ArgAccess::readwrite_from_json(const json &j_) { return {}; }

json ArgAccess::readwrite_to_json(const ArgAccess::ReadWrite &x_) { return json::array({}); }

ArgAccess::Any ArgAccess::any_from_json(const json &j_) {
  const auto ord_ = j_.at(0).get<size_t>();
  const auto &t_ = j_.at(1);
  switch (ord_) {
    case 0: return ArgAccess::read_from_json(t_);
    case 1: return ArgAccess::write_from_json(t_);
    case 2: return ArgAccess::readwrite_from_json(t_);
    default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
  }
}

json ArgAccess::any_to_json(const ArgAccess::Any &x_) {
  return x_.match_total([](const ArgAccess::Read &y_) -> json { return {0, ArgAccess::read_to_json(y_)}; },
                        [](const ArgAccess::Write &y_) -> json { return {1, ArgAccess::write_to_json(y_)}; },
                        [](const ArgAccess::ReadWrite &y_) -> json { return {2, ArgAccess::readwrite_to_json(y_)}; });
}

ArgSizeExpr::Param ArgSizeExpr::param_from_json(const json &j_) {
  const auto index = j_.at(0).get<int32_t>();
  return ArgSizeExpr::Param(index);
}

json ArgSizeExpr::param_to_json(const ArgSizeExpr::Param &x_) {
  const auto index = x_.index;
  return json::array({index});
}

ArgSizeExpr::Const ArgSizeExpr::const_from_json(const json &j_) {
  const auto value = j_.at(0).get<int64_t>();
  return ArgSizeExpr::Const(value);
}

json ArgSizeExpr::const_to_json(const ArgSizeExpr::Const &x_) {
  const auto value = x_.value;
  return json::array({value});
}

ArgSizeExpr::Add ArgSizeExpr::add_from_json(const json &j_) {
  auto lhs = ArgSizeExpr::any_from_json(j_.at(0));
  auto rhs = ArgSizeExpr::any_from_json(j_.at(1));
  return {std::move(lhs), std::move(rhs)};
}

json ArgSizeExpr::add_to_json(const ArgSizeExpr::Add &x_) {
  const auto lhs = ArgSizeExpr::any_to_json(x_.lhs);
  const auto rhs = ArgSizeExpr::any_to_json(x_.rhs);
  return json::array({lhs, rhs});
}

ArgSizeExpr::Mul ArgSizeExpr::mul_from_json(const json &j_) {
  auto lhs = ArgSizeExpr::any_from_json(j_.at(0));
  auto rhs = ArgSizeExpr::any_from_json(j_.at(1));
  return {std::move(lhs), std::move(rhs)};
}

json ArgSizeExpr::mul_to_json(const ArgSizeExpr::Mul &x_) {
  const auto lhs = ArgSizeExpr::any_to_json(x_.lhs);
  const auto rhs = ArgSizeExpr::any_to_json(x_.rhs);
  return json::array({lhs, rhs});
}

ArgSizeExpr::Min ArgSizeExpr::min_from_json(const json &j_) {
  auto lhs = ArgSizeExpr::any_from_json(j_.at(0));
  auto rhs = ArgSizeExpr::any_from_json(j_.at(1));
  return {std::move(lhs), std::move(rhs)};
}

json ArgSizeExpr::min_to_json(const ArgSizeExpr::Min &x_) {
  const auto lhs = ArgSizeExpr::any_to_json(x_.lhs);
  const auto rhs = ArgSizeExpr::any_to_json(x_.rhs);
  return json::array({lhs, rhs});
}

ArgSizeExpr::Result ArgSizeExpr::result_from_json(const json &j_) {
  auto bound = ArgSizeExpr::any_from_json(j_.at(0));
  return ArgSizeExpr::Result(std::move(bound));
}

json ArgSizeExpr::result_to_json(const ArgSizeExpr::Result &x_) {
  const auto bound = ArgSizeExpr::any_to_json(x_.bound);
  return json::array({bound});
}

ArgSizeExpr::Any ArgSizeExpr::any_from_json(const json &j_) {
  const auto ord_ = j_.at(0).get<size_t>();
  const auto &t_ = j_.at(1);
  switch (ord_) {
    case 0: return ArgSizeExpr::param_from_json(t_);
    case 1: return ArgSizeExpr::const_from_json(t_);
    case 2: return ArgSizeExpr::add_from_json(t_);
    case 3: return ArgSizeExpr::mul_from_json(t_);
    case 4: return ArgSizeExpr::min_from_json(t_);
    case 5: return ArgSizeExpr::result_from_json(t_);
    default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
  }
}

json ArgSizeExpr::any_to_json(const ArgSizeExpr::Any &x_) {
  return x_.match_total([](const ArgSizeExpr::Param &y_) -> json { return {0, ArgSizeExpr::param_to_json(y_)}; },
                        [](const ArgSizeExpr::Const &y_) -> json { return {1, ArgSizeExpr::const_to_json(y_)}; },
                        [](const ArgSizeExpr::Add &y_) -> json { return {2, ArgSizeExpr::add_to_json(y_)}; },
                        [](const ArgSizeExpr::Mul &y_) -> json { return {3, ArgSizeExpr::mul_to_json(y_)}; },
                        [](const ArgSizeExpr::Min &y_) -> json { return {4, ArgSizeExpr::min_to_json(y_)}; },
                        [](const ArgSizeExpr::Result &y_) -> json { return {5, ArgSizeExpr::result_to_json(y_)}; });
}

ArgExtent::Elements ArgExtent::elements_from_json(const json &j_) {
  auto size = ArgSizeExpr::any_from_json(j_.at(0));
  return ArgExtent::Elements(std::move(size));
}

json ArgExtent::elements_to_json(const ArgExtent::Elements &x_) {
  const auto size = ArgSizeExpr::any_to_json(x_.size);
  return json::array({size});
}

ArgExtent::Bytes ArgExtent::bytes_from_json(const json &j_) {
  auto size = ArgSizeExpr::any_from_json(j_.at(0));
  return ArgExtent::Bytes(std::move(size));
}

json ArgExtent::bytes_to_json(const ArgExtent::Bytes &x_) {
  const auto size = ArgSizeExpr::any_to_json(x_.size);
  return json::array({size});
}

ArgExtent::Any ArgExtent::any_from_json(const json &j_) {
  const auto ord_ = j_.at(0).get<size_t>();
  const auto &t_ = j_.at(1);
  switch (ord_) {
    case 0: return ArgExtent::elements_from_json(t_);
    case 1: return ArgExtent::bytes_from_json(t_);
    default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
  }
}

json ArgExtent::any_to_json(const ArgExtent::Any &x_) {
  return x_.match_total([](const ArgExtent::Elements &y_) -> json { return {0, ArgExtent::elements_to_json(y_)}; },
                        [](const ArgExtent::Bytes &y_) -> json { return {1, ArgExtent::bytes_to_json(y_)}; });
}

ArgBoundary argboundary_from_json(const json &j_) {
  auto access = ArgAccess::any_from_json(j_.at(0));
  auto extent = ArgExtent::any_from_json(j_.at(1));
  return {std::move(access), std::move(extent)};
}

json argboundary_to_json(const ArgBoundary &x_) {
  const auto access = ArgAccess::any_to_json(x_.access);
  const auto extent = ArgExtent::any_to_json(x_.extent);
  return json::array({access, extent});
}

Arg arg_from_json(const json &j_) {
  auto named = named_from_json(j_.at(0));
  auto pos = j_.at(1).is_null() ? std::nullopt : std::make_optional(sourceposition_from_json(j_.at(1)));
  auto boundary = j_.at(2).is_null() ? std::nullopt : std::make_optional(argboundary_from_json(j_.at(2)));
  return {std::move(named), std::move(pos), std::move(boundary)};
}

json arg_to_json(const Arg &x_) {
  const auto named = named_to_json(x_.named);
  const auto pos = x_.pos ? sourceposition_to_json(*x_.pos) : json();
  const auto boundary = x_.boundary ? argboundary_to_json(*x_.boundary) : json();
  return json::array({named, pos, boundary});
}

FunctionDecl functiondecl_from_json(const json &j_) {
  auto name = sym_from_json(j_.at(0));
  std::vector<Type::Var> tpeVars;
  tpeVars.reserve(j_.at(1).size());
  for (const auto &v_ : j_.at(1)) {
    tpeVars.emplace_back(Type::var_from_json(v_));
  }
  auto receiver = j_.at(2).is_null() ? std::nullopt : std::make_optional(arg_from_json(j_.at(2)));
  std::vector<Arg> args;
  args.reserve(j_.at(3).size());
  for (const auto &v_ : j_.at(3)) {
    args.emplace_back(arg_from_json(v_));
  }
  std::vector<Arg> moduleCaptures;
  moduleCaptures.reserve(j_.at(4).size());
  for (const auto &v_ : j_.at(4)) {
    moduleCaptures.emplace_back(arg_from_json(v_));
  }
  std::vector<Arg> termCaptures;
  termCaptures.reserve(j_.at(5).size());
  for (const auto &v_ : j_.at(5)) {
    termCaptures.emplace_back(arg_from_json(v_));
  }
  auto rtn = Type::any_from_json(j_.at(6));
  auto affinity = FunctionAffinity::any_from_json(j_.at(7));
  return {std::move(name),           std::move(tpeVars),      std::move(receiver), std::move(args),
          std::move(moduleCaptures), std::move(termCaptures), std::move(rtn),      std::move(affinity)};
}

json functiondecl_to_json(const FunctionDecl &x_) {
  const auto name = sym_to_json(x_.name);
  std::vector<json> tpeVars;
  tpeVars.reserve(x_.tpeVars.size());
  for (const auto &v_ : x_.tpeVars) {
    tpeVars.emplace_back(Type::var_to_json(v_));
  }
  const auto receiver = x_.receiver ? arg_to_json(*x_.receiver) : json();
  std::vector<json> args;
  args.reserve(x_.args.size());
  for (const auto &v_ : x_.args) {
    args.emplace_back(arg_to_json(v_));
  }
  std::vector<json> moduleCaptures;
  moduleCaptures.reserve(x_.moduleCaptures.size());
  for (const auto &v_ : x_.moduleCaptures) {
    moduleCaptures.emplace_back(arg_to_json(v_));
  }
  std::vector<json> termCaptures;
  termCaptures.reserve(x_.termCaptures.size());
  for (const auto &v_ : x_.termCaptures) {
    termCaptures.emplace_back(arg_to_json(v_));
  }
  const auto rtn = Type::any_to_json(x_.rtn);
  const auto affinity = FunctionAffinity::any_to_json(x_.affinity);
  return json::array({name, tpeVars, receiver, args, moduleCaptures, termCaptures, rtn, affinity});
}

CallConvention::RegularCall CallConvention::regularcall_from_json(const json &j_) { return {}; }

json CallConvention::regularcall_to_json(const CallConvention::RegularCall &x_) { return json::array({}); }

CallConvention::OffloadEntry CallConvention::offloadentry_from_json(const json &j_) { return {}; }

json CallConvention::offloadentry_to_json(const CallConvention::OffloadEntry &x_) { return json::array({}); }

CallConvention::Any CallConvention::any_from_json(const json &j_) {
  const auto ord_ = j_.at(0).get<size_t>();
  const auto &t_ = j_.at(1);
  switch (ord_) {
    case 0: return CallConvention::regularcall_from_json(t_);
    case 1: return CallConvention::offloadentry_from_json(t_);
    default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
  }
}

json CallConvention::any_to_json(const CallConvention::Any &x_) {
  return x_.match_total([](const CallConvention::RegularCall &y_) -> json { return {0, CallConvention::regularcall_to_json(y_)}; },
                        [](const CallConvention::OffloadEntry &y_) -> json { return {1, CallConvention::offloadentry_to_json(y_)}; });
}

Function function_from_json(const json &j_) {
  auto decl = functiondecl_from_json(j_.at(0));
  std::vector<Stmt::Any> body;
  body.reserve(j_.at(1).size());
  for (const auto &v_ : j_.at(1)) {
    body.emplace_back(Stmt::any_from_json(v_));
  }
  auto visibility = FunctionVisibility::any_from_json(j_.at(2));
  auto fpMode = FunctionFpMode::any_from_json(j_.at(3));
  auto convention = CallConvention::any_from_json(j_.at(4));
  auto implements = j_.at(5).is_null() ? std::nullopt : std::make_optional(sym_from_json(j_.at(5)));
  auto requiredCapabilities = j_.at(6).get<std::vector<std::string>>();
  return {std::move(decl),       std::move(body),       std::move(visibility),          std::move(fpMode),
          std::move(convention), std::move(implements), std::move(requiredCapabilities)};
}

json function_to_json(const Function &x_) {
  const auto decl = functiondecl_to_json(x_.decl);
  std::vector<json> body;
  body.reserve(x_.body.size());
  for (const auto &v_ : x_.body) {
    body.emplace_back(Stmt::any_to_json(v_));
  }
  const auto visibility = FunctionVisibility::any_to_json(x_.visibility);
  const auto fpMode = FunctionFpMode::any_to_json(x_.fpMode);
  const auto convention = CallConvention::any_to_json(x_.convention);
  const auto implements = x_.implements ? sym_to_json(*x_.implements) : json();
  const auto requiredCapabilities = x_.requiredCapabilities;
  return json::array({decl, body, visibility, fpMode, convention, implements, requiredCapabilities});
}

StructDef structdef_from_json(const json &j_) {
  auto name = sym_from_json(j_.at(0));
  std::vector<Type::Var> tpeVars;
  tpeVars.reserve(j_.at(1).size());
  for (const auto &v_ : j_.at(1)) {
    tpeVars.emplace_back(Type::var_from_json(v_));
  }
  std::vector<Named> members;
  members.reserve(j_.at(2).size());
  for (const auto &v_ : j_.at(2)) {
    members.emplace_back(named_from_json(v_));
  }
  std::vector<Type::Struct> parents;
  parents.reserve(j_.at(3).size());
  for (const auto &v_ : j_.at(3)) {
    parents.emplace_back(Type::struct_from_json(v_));
  }
  const auto isUnion = j_.at(4).get<bool>();
  return {std::move(name), std::move(tpeVars), std::move(members), std::move(parents), isUnion};
}

json structdef_to_json(const StructDef &x_) {
  const auto name = sym_to_json(x_.name);
  std::vector<json> tpeVars;
  tpeVars.reserve(x_.tpeVars.size());
  for (const auto &v_ : x_.tpeVars) {
    tpeVars.emplace_back(Type::var_to_json(v_));
  }
  std::vector<json> members;
  members.reserve(x_.members.size());
  for (const auto &v_ : x_.members) {
    members.emplace_back(named_to_json(v_));
  }
  std::vector<json> parents;
  parents.reserve(x_.parents.size());
  for (const auto &v_ : x_.parents) {
    parents.emplace_back(Type::struct_to_json(v_));
  }
  const auto isUnion = x_.isUnion;
  return json::array({name, tpeVars, members, parents, isUnion});
}

Mirror mirror_from_json(const json &j_) {
  auto source = sym_from_json(j_.at(0));
  std::vector<Sym> sourceParents;
  sourceParents.reserve(j_.at(1).size());
  for (const auto &v_ : j_.at(1)) {
    sourceParents.emplace_back(sym_from_json(v_));
  }
  auto structDef = structdef_from_json(j_.at(2));
  std::vector<Function> functions;
  functions.reserve(j_.at(3).size());
  for (const auto &v_ : j_.at(3)) {
    functions.emplace_back(function_from_json(v_));
  }
  std::vector<StructDef> dependencies;
  dependencies.reserve(j_.at(4).size());
  for (const auto &v_ : j_.at(4)) {
    dependencies.emplace_back(structdef_from_json(v_));
  }
  return {std::move(source), std::move(sourceParents), std::move(structDef), std::move(functions), std::move(dependencies)};
}

json mirror_to_json(const Mirror &x_) {
  const auto source = sym_to_json(x_.source);
  std::vector<json> sourceParents;
  sourceParents.reserve(x_.sourceParents.size());
  for (const auto &v_ : x_.sourceParents) {
    sourceParents.emplace_back(sym_to_json(v_));
  }
  const auto structDef = structdef_to_json(x_.structDef);
  std::vector<json> functions;
  functions.reserve(x_.functions.size());
  for (const auto &v_ : x_.functions) {
    functions.emplace_back(function_to_json(v_));
  }
  std::vector<json> dependencies;
  dependencies.reserve(x_.dependencies.size());
  for (const auto &v_ : x_.dependencies) {
    dependencies.emplace_back(structdef_to_json(v_));
  }
  return json::array({source, sourceParents, structDef, functions, dependencies});
}

PassPhase::Initial PassPhase::initial_from_json(const json &j_) { return {}; }

json PassPhase::initial_to_json(const PassPhase::Initial &x_) { return json::array({}); }

PassPhase::PostMono PassPhase::postmono_from_json(const json &j_) { return {}; }

json PassPhase::postmono_to_json(const PassPhase::PostMono &x_) { return json::array({}); }

PassPhase::Any PassPhase::any_from_json(const json &j_) {
  const auto ord_ = j_.at(0).get<size_t>();
  const auto &t_ = j_.at(1);
  switch (ord_) {
    case 0: return PassPhase::initial_from_json(t_);
    case 1: return PassPhase::postmono_from_json(t_);
    default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
  }
}

json PassPhase::any_to_json(const PassPhase::Any &x_) {
  return x_.match_total([](const PassPhase::Initial &y_) -> json { return {0, PassPhase::initial_to_json(y_)}; },
                        [](const PassPhase::PostMono &y_) -> json { return {1, PassPhase::postmono_to_json(y_)}; });
}

MetaEntry metaentry_from_json(const json &j_) {
  auto key = j_.at(0).get<std::string>();
  auto value = j_.at(1).get<std::string>();
  return {std::move(key), std::move(value)};
}

json metaentry_to_json(const MetaEntry &x_) {
  const auto key = x_.key;
  const auto value = x_.value;
  return json::array({key, value});
}

Program program_from_json(const json &j_) {
  auto entry = j_.at(0).is_null() ? std::nullopt : std::make_optional(function_from_json(j_.at(0)));
  std::vector<Function> functions;
  functions.reserve(j_.at(1).size());
  for (const auto &v_ : j_.at(1)) {
    functions.emplace_back(function_from_json(v_));
  }
  std::vector<StructDef> defs;
  defs.reserve(j_.at(2).size());
  for (const auto &v_ : j_.at(2)) {
    defs.emplace_back(structdef_from_json(v_));
  }
  auto phase = PassPhase::any_from_json(j_.at(3));
  std::vector<MetaEntry> metadata;
  metadata.reserve(j_.at(4).size());
  for (const auto &v_ : j_.at(4)) {
    metadata.emplace_back(metaentry_from_json(v_));
  }
  return {std::move(entry), std::move(functions), std::move(defs), std::move(phase), std::move(metadata)};
}

json program_to_json(const Program &x_) {
  const auto entry = x_.entry ? function_to_json(*x_.entry) : json();
  std::vector<json> functions;
  functions.reserve(x_.functions.size());
  for (const auto &v_ : x_.functions) {
    functions.emplace_back(function_to_json(v_));
  }
  std::vector<json> defs;
  defs.reserve(x_.defs.size());
  for (const auto &v_ : x_.defs) {
    defs.emplace_back(structdef_to_json(v_));
  }
  const auto phase = PassPhase::any_to_json(x_.phase);
  std::vector<json> metadata;
  metadata.reserve(x_.metadata.size());
  for (const auto &v_ : x_.metadata) {
    metadata.emplace_back(metaentry_to_json(v_));
  }
  return json::array({entry, functions, defs, phase, metadata});
}

StructLayoutMember structlayoutmember_from_json(const json &j_) {
  auto name = named_from_json(j_.at(0));
  const auto offsetInBytes = j_.at(1).get<int64_t>();
  const auto sizeInBytes = j_.at(2).get<int64_t>();
  return {std::move(name), offsetInBytes, sizeInBytes};
}

json structlayoutmember_to_json(const StructLayoutMember &x_) {
  const auto name = named_to_json(x_.name);
  const auto offsetInBytes = x_.offsetInBytes;
  const auto sizeInBytes = x_.sizeInBytes;
  return json::array({name, offsetInBytes, sizeInBytes});
}

StructLayout structlayout_from_json(const json &j_) {
  auto name = j_.at(0).get<std::string>();
  const auto sizeInBytes = j_.at(1).get<int64_t>();
  const auto alignment = j_.at(2).get<int64_t>();
  std::vector<StructLayoutMember> members;
  members.reserve(j_.at(3).size());
  for (const auto &v_ : j_.at(3)) {
    members.emplace_back(structlayoutmember_from_json(v_));
  }
  return {std::move(name), sizeInBytes, alignment, std::move(members)};
}

json structlayout_to_json(const StructLayout &x_) {
  const auto name = x_.name;
  const auto sizeInBytes = x_.sizeInBytes;
  const auto alignment = x_.alignment;
  std::vector<json> members;
  members.reserve(x_.members.size());
  for (const auto &v_ : x_.members) {
    members.emplace_back(structlayoutmember_to_json(v_));
  }
  return json::array({name, sizeInBytes, alignment, members});
}

CompileEvent compileevent_from_json(const json &j_) {
  const auto epochMillis = j_.at(0).get<int64_t>();
  const auto elapsedNanos = j_.at(1).get<int64_t>();
  auto name = j_.at(2).get<std::string>();
  auto data = j_.at(3).get<std::string>();
  std::vector<CompileEvent> items;
  items.reserve(j_.at(4).size());
  for (const auto &v_ : j_.at(4)) {
    items.emplace_back(compileevent_from_json(v_));
  }
  return {epochMillis, elapsedNanos, std::move(name), std::move(data), std::move(items)};
}

json compileevent_to_json(const CompileEvent &x_) {
  const auto epochMillis = x_.epochMillis;
  const auto elapsedNanos = x_.elapsedNanos;
  const auto name = x_.name;
  const auto data = x_.data;
  std::vector<json> items;
  items.reserve(x_.items.size());
  for (const auto &v_ : x_.items) {
    items.emplace_back(compileevent_to_json(v_));
  }
  return json::array({epochMillis, elapsedNanos, name, data, items});
}

PassArg passarg_from_json(const json &j_) {
  auto name = j_.at(0).get<std::string>();
  auto value = j_.at(1).get<std::string>();
  return {std::move(name), std::move(value)};
}

json passarg_to_json(const PassArg &x_) {
  const auto name = x_.name;
  const auto value = x_.value;
  return json::array({name, value});
}

PassSpec passspec_from_json(const json &j_) {
  auto name = j_.at(0).get<std::string>();
  std::vector<PassArg> args;
  args.reserve(j_.at(1).size());
  for (const auto &v_ : j_.at(1)) {
    args.emplace_back(passarg_from_json(v_));
  }
  return {std::move(name), std::move(args)};
}

json passspec_to_json(const PassSpec &x_) {
  const auto name = x_.name;
  std::vector<json> args;
  args.reserve(x_.args.size());
  for (const auto &v_ : x_.args) {
    args.emplace_back(passarg_to_json(v_));
  }
  return json::array({name, args});
}

PassPipeline passpipeline_from_json(const json &j_) {
  std::vector<PassSpec> steps;
  steps.reserve(j_.at(0).size());
  for (const auto &v_ : j_.at(0)) {
    steps.emplace_back(passspec_from_json(v_));
  }
  return PassPipeline(std::move(steps));
}

json passpipeline_to_json(const PassPipeline &x_) {
  std::vector<json> steps;
  steps.reserve(x_.steps.size());
  for (const auto &v_ : x_.steps) {
    steps.emplace_back(passspec_to_json(v_));
  }
  return json::array({steps});
}

PassRunResult passrunresult_from_json(const json &j_) {
  auto program = program_from_json(j_.at(0));
  auto event = compileevent_from_json(j_.at(1));
  return {std::move(program), std::move(event)};
}

json passrunresult_to_json(const PassRunResult &x_) {
  const auto program = program_to_json(x_.program);
  const auto event = compileevent_to_json(x_.event);
  return json::array({program, event});
}

CompileResult compileresult_from_json(const json &j_) {
  auto binary = j_.at(0).is_null() ? std::nullopt : std::make_optional(j_.at(0).get<std::vector<int8_t>>());
  auto features = j_.at(1).get<std::vector<std::string>>();
  std::vector<CompileEvent> events;
  events.reserve(j_.at(2).size());
  for (const auto &v_ : j_.at(2)) {
    events.emplace_back(compileevent_from_json(v_));
  }
  std::vector<StructLayout> layouts;
  layouts.reserve(j_.at(3).size());
  for (const auto &v_ : j_.at(3)) {
    layouts.emplace_back(structlayout_from_json(v_));
  }
  auto messages = j_.at(4).get<std::string>();
  std::vector<Named> entryArgs;
  entryArgs.reserve(j_.at(5).size());
  for (const auto &v_ : j_.at(5)) {
    entryArgs.emplace_back(named_from_json(v_));
  }
  return {std::move(binary), std::move(features), std::move(events), std::move(layouts), std::move(messages), std::move(entryArgs)};
}

json compileresult_to_json(const CompileResult &x_) {
  const auto binary = x_.binary ? json(*x_.binary) : json();
  const auto features = x_.features;
  std::vector<json> events;
  events.reserve(x_.events.size());
  for (const auto &v_ : x_.events) {
    events.emplace_back(compileevent_to_json(v_));
  }
  std::vector<json> layouts;
  layouts.reserve(x_.layouts.size());
  for (const auto &v_ : x_.layouts) {
    layouts.emplace_back(structlayout_to_json(v_));
  }
  const auto messages = x_.messages;
  std::vector<json> entryArgs;
  entryArgs.reserve(x_.entryArgs.size());
  for (const auto &v_ : x_.entryArgs) {
    entryArgs.emplace_back(named_to_json(v_));
  }
  return json::array({binary, features, events, layouts, messages, entryArgs});
}

Interface interface_from_json(const json &j_) {
  auto name = sym_from_json(j_.at(0));
  std::vector<FunctionDecl> declarations;
  declarations.reserve(j_.at(1).size());
  for (const auto &v_ : j_.at(1)) {
    declarations.emplace_back(functiondecl_from_json(v_));
  }
  std::vector<MetaEntry> metadata;
  metadata.reserve(j_.at(2).size());
  for (const auto &v_ : j_.at(2)) {
    metadata.emplace_back(metaentry_from_json(v_));
  }
  return {std::move(name), std::move(declarations), std::move(metadata)};
}

json interface_to_json(const Interface &x_) {
  const auto name = sym_to_json(x_.name);
  std::vector<json> declarations;
  declarations.reserve(x_.declarations.size());
  for (const auto &v_ : x_.declarations) {
    declarations.emplace_back(functiondecl_to_json(v_));
  }
  std::vector<json> metadata;
  metadata.reserve(x_.metadata.size());
  for (const auto &v_ : x_.metadata) {
    metadata.emplace_back(metaentry_to_json(v_));
  }
  return json::array({name, declarations, metadata});
}

Package package_from_json(const json &j_) {
  auto interface = interface_from_json(j_.at(0));
  auto program = program_from_json(j_.at(1));
  return {std::move(interface), std::move(program)};
}

json package_to_json(const Package &x_) {
  const auto interface = interface_to_json(x_.interface);
  const auto program = program_to_json(x_.program);
  return json::array({interface, program});
}

ProgramTypeSize programtypesize_from_json(const json &j_) {
  auto tpe = Type::any_from_json(j_.at(0));
  const auto sizeInBytes = j_.at(1).get<int32_t>();
  return {std::move(tpe), sizeInBytes};
}

json programtypesize_to_json(const ProgramTypeSize &x_) {
  const auto tpe = Type::any_to_json(x_.tpe);
  const auto sizeInBytes = x_.sizeInBytes;
  return json::array({tpe, sizeInBytes});
}

ProgramLinkRequest programlinkrequest_from_json(const json &j_) {
  std::vector<Package> packages;
  packages.reserve(j_.at(0).size());
  for (const auto &v_ : j_.at(0)) {
    packages.emplace_back(package_from_json(v_));
  }
  auto consumer = program_from_json(j_.at(1));
  auto capabilities = j_.at(2).get<std::vector<std::string>>();
  std::vector<ProgramTypeSize> typeSizes;
  typeSizes.reserve(j_.at(3).size());
  for (const auto &v_ : j_.at(3)) {
    typeSizes.emplace_back(programtypesize_from_json(v_));
  }
  const auto stackDepth = j_.at(4).is_null() ? std::nullopt : std::make_optional(j_.at(4).get<int32_t>());
  const auto logicalAddressing = j_.at(5).get<bool>();
  return {std::move(packages), std::move(consumer), std::move(capabilities), std::move(typeSizes), stackDepth, logicalAddressing};
}

json programlinkrequest_to_json(const ProgramLinkRequest &x_) {
  std::vector<json> packages;
  packages.reserve(x_.packages.size());
  for (const auto &v_ : x_.packages) {
    packages.emplace_back(package_to_json(v_));
  }
  const auto consumer = program_to_json(x_.consumer);
  const auto capabilities = x_.capabilities;
  std::vector<json> typeSizes;
  typeSizes.reserve(x_.typeSizes.size());
  for (const auto &v_ : x_.typeSizes) {
    typeSizes.emplace_back(programtypesize_to_json(v_));
  }
  const auto stackDepth = x_.stackDepth ? json(*x_.stackDepth) : json();
  const auto logicalAddressing = x_.logicalAddressing;
  return json::array({packages, consumer, capabilities, typeSizes, stackDepth, logicalAddressing});
}

PackageFragment packagefragment_from_json(const json &j_) {
  auto identity = j_.at(0).get<std::string>();
  auto program = program_from_json(j_.at(1));
  return {std::move(identity), std::move(program)};
}

json packagefragment_to_json(const PackageFragment &x_) {
  const auto identity = x_.identity;
  const auto program = program_to_json(x_.program);
  return json::array({identity, program});
}

PackageLinkRequest packagelinkrequest_from_json(const json &j_) {
  auto interface = interface_from_json(j_.at(0));
  std::vector<PackageFragment> fragments;
  fragments.reserve(j_.at(1).size());
  for (const auto &v_ : j_.at(1)) {
    fragments.emplace_back(packagefragment_from_json(v_));
  }
  auto capabilities = j_.at(2).get<std::vector<std::string>>();
  const auto pruneUnimplementedDeclarations = j_.at(3).get<bool>();
  return {std::move(interface), std::move(fragments), std::move(capabilities), pruneUnimplementedDeclarations};
}

json packagelinkrequest_to_json(const PackageLinkRequest &x_) {
  const auto interface = interface_to_json(x_.interface);
  std::vector<json> fragments;
  fragments.reserve(x_.fragments.size());
  for (const auto &v_ : x_.fragments) {
    fragments.emplace_back(packagefragment_to_json(v_));
  }
  const auto capabilities = x_.capabilities;
  const auto pruneUnimplementedDeclarations = x_.pruneUnimplementedDeclarations;
  return json::array({interface, fragments, capabilities, pruneUnimplementedDeclarations});
}

CompileModule compilemodule_from_json(const json &j_) {
  auto moduleName = j_.at(0).get<std::string>();
  const auto format = j_.at(1).get<int32_t>();
  const auto kind = j_.at(2).get<int32_t>();
  auto features = j_.at(3).get<std::vector<std::string>>();
  auto image = j_.at(4).get<std::vector<int8_t>>();
  return {std::move(moduleName), format, kind, std::move(features), std::move(image)};
}

json compilemodule_to_json(const CompileModule &x_) {
  const auto moduleName = x_.moduleName;
  const auto format = x_.format;
  const auto kind = x_.kind;
  const auto features = x_.features;
  const auto image = x_.image;
  return json::array({moduleName, format, kind, features, image});
}

CompileRoot compileroot_from_json(const json &j_) {
  auto name = j_.at(0).get<std::string>();
  auto modules = j_.at(1).get<std::vector<std::string>>();
  auto implementations = j_.at(2).get<std::vector<std::string>>();
  auto unfit = j_.at(3).get<std::vector<std::string>>();
  return {std::move(name), std::move(modules), std::move(implementations), std::move(unfit)};
}

json compileroot_to_json(const CompileRoot &x_) {
  const auto name = x_.name;
  const auto modules = x_.modules;
  const auto implementations = x_.implementations;
  const auto unfit = x_.unfit;
  return json::array({name, modules, implementations, unfit});
}

CompileBundle compilebundle_from_json(const json &j_) {
  auto hostObject = j_.at(0).get<std::vector<int8_t>>();
  std::vector<CompileModule> remoteModules;
  remoteModules.reserve(j_.at(1).size());
  for (const auto &v_ : j_.at(1)) {
    remoteModules.emplace_back(compilemodule_from_json(v_));
  }
  auto implementations = j_.at(2).get<std::vector<std::string>>();
  auto unfit = j_.at(3).get<std::vector<std::string>>();
  std::vector<CompileRoot> roots;
  roots.reserve(j_.at(4).size());
  for (const auto &v_ : j_.at(4)) {
    roots.emplace_back(compileroot_from_json(v_));
  }
  return {std::move(hostObject), std::move(remoteModules), std::move(implementations), std::move(unfit), std::move(roots)};
}

json compilebundle_to_json(const CompileBundle &x_) {
  const auto hostObject = x_.hostObject;
  std::vector<json> remoteModules;
  remoteModules.reserve(x_.remoteModules.size());
  for (const auto &v_ : x_.remoteModules) {
    remoteModules.emplace_back(compilemodule_to_json(v_));
  }
  const auto implementations = x_.implementations;
  const auto unfit = x_.unfit;
  std::vector<json> roots;
  roots.reserve(x_.roots.size());
  for (const auto &v_ : x_.roots) {
    roots.emplace_back(compileroot_to_json(v_));
  }
  return json::array({hostObject, remoteModules, implementations, unfit, roots});
}
json hashed_from_json(const json &j_) {
  const auto hash_ = j_.at(0).get<std::string>();
  auto data_ = j_.at(1);
  if (hash_ != AdtHash) {
    throw std::runtime_error("Expecting ADT hash to be " + std::string(AdtHash) + ", but was " + hash_);
  }
  return data_;
}

json hashed_to_json(const json &x_) { return json::array({AdtHash, x_}); }

namespace ArgExtent { namespace {
ArgExtent::Elements elements_fields_from_msgpack(MsgpackReader &, size_t);
void elements_fields_to_msgpack(MsgpackWriter &, const ArgExtent::Elements &);
ArgExtent::Bytes bytes_fields_from_msgpack(MsgpackReader &, size_t);
void bytes_fields_to_msgpack(MsgpackWriter &, const ArgExtent::Bytes &);
ArgExtent::Any any_from_msgpack(MsgpackReader &);
void any_to_msgpack(MsgpackWriter &, const ArgExtent::Any &);
}} // namespace ArgExtent
namespace Expr { namespace {
Expr::Alias alias_fields_from_msgpack(MsgpackReader &, size_t);
void alias_fields_to_msgpack(MsgpackWriter &, const Expr::Alias &);
Expr::SpecOp specop_fields_from_msgpack(MsgpackReader &, size_t);
void specop_fields_to_msgpack(MsgpackWriter &, const Expr::SpecOp &);
Expr::MathOp mathop_fields_from_msgpack(MsgpackReader &, size_t);
void mathop_fields_to_msgpack(MsgpackWriter &, const Expr::MathOp &);
Expr::IntrOp introp_fields_from_msgpack(MsgpackReader &, size_t);
void introp_fields_to_msgpack(MsgpackWriter &, const Expr::IntrOp &);
Expr::Cast cast_fields_from_msgpack(MsgpackReader &, size_t);
void cast_fields_to_msgpack(MsgpackWriter &, const Expr::Cast &);
Expr::BitCast bitcast_fields_from_msgpack(MsgpackReader &, size_t);
void bitcast_fields_to_msgpack(MsgpackWriter &, const Expr::BitCast &);
Expr::Index index_fields_from_msgpack(MsgpackReader &, size_t);
void index_fields_to_msgpack(MsgpackWriter &, const Expr::Index &);
Expr::RefTo refto_fields_from_msgpack(MsgpackReader &, size_t);
void refto_fields_to_msgpack(MsgpackWriter &, const Expr::RefTo &);
Expr::Alloc alloc_fields_from_msgpack(MsgpackReader &, size_t);
void alloc_fields_to_msgpack(MsgpackWriter &, const Expr::Alloc &);
Expr::Invoke invoke_fields_from_msgpack(MsgpackReader &, size_t);
void invoke_fields_to_msgpack(MsgpackWriter &, const Expr::Invoke &);
Expr::ForeignCall foreigncall_fields_from_msgpack(MsgpackReader &, size_t);
void foreigncall_fields_to_msgpack(MsgpackWriter &, const Expr::ForeignCall &);
Expr::OffsetOf offsetof_fields_from_msgpack(MsgpackReader &, size_t);
void offsetof_fields_to_msgpack(MsgpackWriter &, const Expr::OffsetOf &);
Expr::SizeOf sizeof_fields_from_msgpack(MsgpackReader &, size_t);
void sizeof_fields_to_msgpack(MsgpackWriter &, const Expr::SizeOf &);
Expr::Any any_from_msgpack(MsgpackReader &);
void any_to_msgpack(MsgpackWriter &, const Expr::Any &);
}} // namespace Expr
namespace Region { namespace {
Region::Rooted rooted_fields_from_msgpack(MsgpackReader &, size_t);
void rooted_fields_to_msgpack(MsgpackWriter &, const Region::Rooted &);
Region::Opaque opaque_fields_from_msgpack(MsgpackReader &, size_t);
Region::Any any_from_msgpack(MsgpackReader &);
void any_to_msgpack(MsgpackWriter &, const Region::Any &);
}} // namespace Region
namespace Stmt { namespace {
Stmt::Var var_fields_from_msgpack(MsgpackReader &, size_t);
void var_fields_to_msgpack(MsgpackWriter &, const Stmt::Var &);
Stmt::Mut mut_fields_from_msgpack(MsgpackReader &, size_t);
void mut_fields_to_msgpack(MsgpackWriter &, const Stmt::Mut &);
Stmt::Update update_fields_from_msgpack(MsgpackReader &, size_t);
void update_fields_to_msgpack(MsgpackWriter &, const Stmt::Update &);
Stmt::While while_fields_from_msgpack(MsgpackReader &, size_t);
void while_fields_to_msgpack(MsgpackWriter &, const Stmt::While &);
Stmt::ForRange forrange_fields_from_msgpack(MsgpackReader &, size_t);
void forrange_fields_to_msgpack(MsgpackWriter &, const Stmt::ForRange &);
Stmt::Break break_fields_from_msgpack(MsgpackReader &, size_t);
Stmt::Cont cont_fields_from_msgpack(MsgpackReader &, size_t);
Stmt::Cond cond_fields_from_msgpack(MsgpackReader &, size_t);
void cond_fields_to_msgpack(MsgpackWriter &, const Stmt::Cond &);
Stmt::Return return_fields_from_msgpack(MsgpackReader &, size_t);
void return_fields_to_msgpack(MsgpackWriter &, const Stmt::Return &);
Stmt::Annotated annotated_fields_from_msgpack(MsgpackReader &, size_t);
void annotated_fields_to_msgpack(MsgpackWriter &, const Stmt::Annotated &);
Stmt::Try try_fields_from_msgpack(MsgpackReader &, size_t);
void try_fields_to_msgpack(MsgpackWriter &, const Stmt::Try &);
Stmt::Raise raise_fields_from_msgpack(MsgpackReader &, size_t);
void raise_fields_to_msgpack(MsgpackWriter &, const Stmt::Raise &);
Stmt::Rethrow rethrow_fields_from_msgpack(MsgpackReader &, size_t);
Stmt::Any any_from_msgpack(MsgpackReader &);
void any_to_msgpack(MsgpackWriter &, const Stmt::Any &);
}} // namespace Stmt
namespace Math { namespace {
Math::Abs abs_fields_from_msgpack(MsgpackReader &, size_t);
void abs_fields_to_msgpack(MsgpackWriter &, const Math::Abs &);
Math::Sin sin_fields_from_msgpack(MsgpackReader &, size_t);
void sin_fields_to_msgpack(MsgpackWriter &, const Math::Sin &);
Math::Cos cos_fields_from_msgpack(MsgpackReader &, size_t);
void cos_fields_to_msgpack(MsgpackWriter &, const Math::Cos &);
Math::Tan tan_fields_from_msgpack(MsgpackReader &, size_t);
void tan_fields_to_msgpack(MsgpackWriter &, const Math::Tan &);
Math::Asin asin_fields_from_msgpack(MsgpackReader &, size_t);
void asin_fields_to_msgpack(MsgpackWriter &, const Math::Asin &);
Math::Acos acos_fields_from_msgpack(MsgpackReader &, size_t);
void acos_fields_to_msgpack(MsgpackWriter &, const Math::Acos &);
Math::Atan atan_fields_from_msgpack(MsgpackReader &, size_t);
void atan_fields_to_msgpack(MsgpackWriter &, const Math::Atan &);
Math::Sinh sinh_fields_from_msgpack(MsgpackReader &, size_t);
void sinh_fields_to_msgpack(MsgpackWriter &, const Math::Sinh &);
Math::Cosh cosh_fields_from_msgpack(MsgpackReader &, size_t);
void cosh_fields_to_msgpack(MsgpackWriter &, const Math::Cosh &);
Math::Tanh tanh_fields_from_msgpack(MsgpackReader &, size_t);
void tanh_fields_to_msgpack(MsgpackWriter &, const Math::Tanh &);
Math::Signum signum_fields_from_msgpack(MsgpackReader &, size_t);
void signum_fields_to_msgpack(MsgpackWriter &, const Math::Signum &);
Math::Round round_fields_from_msgpack(MsgpackReader &, size_t);
void round_fields_to_msgpack(MsgpackWriter &, const Math::Round &);
Math::Ceil ceil_fields_from_msgpack(MsgpackReader &, size_t);
void ceil_fields_to_msgpack(MsgpackWriter &, const Math::Ceil &);
Math::Floor floor_fields_from_msgpack(MsgpackReader &, size_t);
void floor_fields_to_msgpack(MsgpackWriter &, const Math::Floor &);
Math::Rint rint_fields_from_msgpack(MsgpackReader &, size_t);
void rint_fields_to_msgpack(MsgpackWriter &, const Math::Rint &);
Math::Sqrt sqrt_fields_from_msgpack(MsgpackReader &, size_t);
void sqrt_fields_to_msgpack(MsgpackWriter &, const Math::Sqrt &);
Math::Cbrt cbrt_fields_from_msgpack(MsgpackReader &, size_t);
void cbrt_fields_to_msgpack(MsgpackWriter &, const Math::Cbrt &);
Math::Exp exp_fields_from_msgpack(MsgpackReader &, size_t);
void exp_fields_to_msgpack(MsgpackWriter &, const Math::Exp &);
Math::Expm1 expm1_fields_from_msgpack(MsgpackReader &, size_t);
void expm1_fields_to_msgpack(MsgpackWriter &, const Math::Expm1 &);
Math::Log log_fields_from_msgpack(MsgpackReader &, size_t);
void log_fields_to_msgpack(MsgpackWriter &, const Math::Log &);
Math::Log1p log1p_fields_from_msgpack(MsgpackReader &, size_t);
void log1p_fields_to_msgpack(MsgpackWriter &, const Math::Log1p &);
Math::Log10 log10_fields_from_msgpack(MsgpackReader &, size_t);
void log10_fields_to_msgpack(MsgpackWriter &, const Math::Log10 &);
Math::Pow pow_fields_from_msgpack(MsgpackReader &, size_t);
void pow_fields_to_msgpack(MsgpackWriter &, const Math::Pow &);
Math::Atan2 atan2_fields_from_msgpack(MsgpackReader &, size_t);
void atan2_fields_to_msgpack(MsgpackWriter &, const Math::Atan2 &);
Math::Hypot hypot_fields_from_msgpack(MsgpackReader &, size_t);
void hypot_fields_to_msgpack(MsgpackWriter &, const Math::Hypot &);
Math::Any any_from_msgpack(MsgpackReader &);
void any_to_msgpack(MsgpackWriter &, const Math::Any &);
}} // namespace Math
namespace CallConvention { namespace {
CallConvention::RegularCall regularcall_fields_from_msgpack(MsgpackReader &, size_t);
CallConvention::OffloadEntry offloadentry_fields_from_msgpack(MsgpackReader &, size_t);
CallConvention::Any any_from_msgpack(MsgpackReader &);
void any_to_msgpack(MsgpackWriter &, const CallConvention::Any &);
}} // namespace CallConvention
namespace MemOrder { namespace {
MemOrder::Relaxed relaxed_fields_from_msgpack(MsgpackReader &, size_t);
MemOrder::Acquire acquire_fields_from_msgpack(MsgpackReader &, size_t);
MemOrder::Release release_fields_from_msgpack(MsgpackReader &, size_t);
MemOrder::AcqRel acqrel_fields_from_msgpack(MsgpackReader &, size_t);
MemOrder::SeqCst seqcst_fields_from_msgpack(MsgpackReader &, size_t);
MemOrder::Any any_from_msgpack(MsgpackReader &);
void any_to_msgpack(MsgpackWriter &, const MemOrder::Any &);
}} // namespace MemOrder
namespace TypeSpace { namespace {
TypeSpace::Global global_fields_from_msgpack(MsgpackReader &, size_t);
TypeSpace::Local local_fields_from_msgpack(MsgpackReader &, size_t);
TypeSpace::Private private_fields_from_msgpack(MsgpackReader &, size_t);
TypeSpace::Constant constant_fields_from_msgpack(MsgpackReader &, size_t);
TypeSpace::Any any_from_msgpack(MsgpackReader &);
void any_to_msgpack(MsgpackWriter &, const TypeSpace::Any &);
}} // namespace TypeSpace
namespace PassPhase { namespace {
PassPhase::Initial initial_fields_from_msgpack(MsgpackReader &, size_t);
PassPhase::PostMono postmono_fields_from_msgpack(MsgpackReader &, size_t);
PassPhase::Any any_from_msgpack(MsgpackReader &);
void any_to_msgpack(MsgpackWriter &, const PassPhase::Any &);
}} // namespace PassPhase
namespace Spec { namespace {
Spec::Assert assert_fields_from_msgpack(MsgpackReader &, size_t);
void assert_fields_to_msgpack(MsgpackWriter &, const Spec::Assert &);
Spec::GpuBarrierGlobal gpubarrierglobal_fields_from_msgpack(MsgpackReader &, size_t);
Spec::GpuBarrierLocal gpubarrierlocal_fields_from_msgpack(MsgpackReader &, size_t);
Spec::GpuBarrierAll gpubarrierall_fields_from_msgpack(MsgpackReader &, size_t);
Spec::GpuFenceGlobal gpufenceglobal_fields_from_msgpack(MsgpackReader &, size_t);
Spec::GpuFenceLocal gpufencelocal_fields_from_msgpack(MsgpackReader &, size_t);
Spec::GpuFenceAll gpufenceall_fields_from_msgpack(MsgpackReader &, size_t);
Spec::GpuGlobalIdx gpuglobalidx_fields_from_msgpack(MsgpackReader &, size_t);
void gpuglobalidx_fields_to_msgpack(MsgpackWriter &, const Spec::GpuGlobalIdx &);
Spec::GpuGlobalSize gpuglobalsize_fields_from_msgpack(MsgpackReader &, size_t);
void gpuglobalsize_fields_to_msgpack(MsgpackWriter &, const Spec::GpuGlobalSize &);
Spec::GpuGroupIdx gpugroupidx_fields_from_msgpack(MsgpackReader &, size_t);
void gpugroupidx_fields_to_msgpack(MsgpackWriter &, const Spec::GpuGroupIdx &);
Spec::GpuGroupSize gpugroupsize_fields_from_msgpack(MsgpackReader &, size_t);
void gpugroupsize_fields_to_msgpack(MsgpackWriter &, const Spec::GpuGroupSize &);
Spec::GpuLocalIdx gpulocalidx_fields_from_msgpack(MsgpackReader &, size_t);
void gpulocalidx_fields_to_msgpack(MsgpackWriter &, const Spec::GpuLocalIdx &);
Spec::GpuLocalSize gpulocalsize_fields_from_msgpack(MsgpackReader &, size_t);
void gpulocalsize_fields_to_msgpack(MsgpackWriter &, const Spec::GpuLocalSize &);
Spec::GpuLaneIdx gpulaneidx_fields_from_msgpack(MsgpackReader &, size_t);
Spec::GpuSubgroupSize gpusubgroupsize_fields_from_msgpack(MsgpackReader &, size_t);
Spec::GpuShuffleDown gpushuffledown_fields_from_msgpack(MsgpackReader &, size_t);
void gpushuffledown_fields_to_msgpack(MsgpackWriter &, const Spec::GpuShuffleDown &);
Spec::GpuShuffleUp gpushuffleup_fields_from_msgpack(MsgpackReader &, size_t);
void gpushuffleup_fields_to_msgpack(MsgpackWriter &, const Spec::GpuShuffleUp &);
Spec::GpuShuffleIdx gpushuffleidx_fields_from_msgpack(MsgpackReader &, size_t);
void gpushuffleidx_fields_to_msgpack(MsgpackWriter &, const Spec::GpuShuffleIdx &);
Spec::GpuShuffleXor gpushufflexor_fields_from_msgpack(MsgpackReader &, size_t);
void gpushufflexor_fields_to_msgpack(MsgpackWriter &, const Spec::GpuShuffleXor &);
Spec::GpuSubgroupBarrier gpusubgroupbarrier_fields_from_msgpack(MsgpackReader &, size_t);
void gpusubgroupbarrier_fields_to_msgpack(MsgpackWriter &, const Spec::GpuSubgroupBarrier &);
Spec::GpuBallot gpuballot_fields_from_msgpack(MsgpackReader &, size_t);
void gpuballot_fields_to_msgpack(MsgpackWriter &, const Spec::GpuBallot &);
Spec::GpuVoteAny gpuvoteany_fields_from_msgpack(MsgpackReader &, size_t);
void gpuvoteany_fields_to_msgpack(MsgpackWriter &, const Spec::GpuVoteAny &);
Spec::GpuVoteAll gpuvoteall_fields_from_msgpack(MsgpackReader &, size_t);
void gpuvoteall_fields_to_msgpack(MsgpackWriter &, const Spec::GpuVoteAll &);
Spec::GpuAtomicRMW gpuatomicrmw_fields_from_msgpack(MsgpackReader &, size_t);
void gpuatomicrmw_fields_to_msgpack(MsgpackWriter &, const Spec::GpuAtomicRMW &);
Spec::GpuAtomicCAS gpuatomiccas_fields_from_msgpack(MsgpackReader &, size_t);
void gpuatomiccas_fields_to_msgpack(MsgpackWriter &, const Spec::GpuAtomicCAS &);
Spec::GpuGroupReduce gpugroupreduce_fields_from_msgpack(MsgpackReader &, size_t);
void gpugroupreduce_fields_to_msgpack(MsgpackWriter &, const Spec::GpuGroupReduce &);
Spec::GpuGroupInclusiveScan gpugroupinclusivescan_fields_from_msgpack(MsgpackReader &, size_t);
void gpugroupinclusivescan_fields_to_msgpack(MsgpackWriter &, const Spec::GpuGroupInclusiveScan &);
Spec::GpuGroupExclusiveScan gpugroupexclusivescan_fields_from_msgpack(MsgpackReader &, size_t);
void gpugroupexclusivescan_fields_to_msgpack(MsgpackWriter &, const Spec::GpuGroupExclusiveScan &);
Spec::GpuVolatileLoad gpuvolatileload_fields_from_msgpack(MsgpackReader &, size_t);
void gpuvolatileload_fields_to_msgpack(MsgpackWriter &, const Spec::GpuVolatileLoad &);
Spec::GpuVolatileStore gpuvolatilestore_fields_from_msgpack(MsgpackReader &, size_t);
void gpuvolatilestore_fields_to_msgpack(MsgpackWriter &, const Spec::GpuVolatileStore &);
Spec::RemoteLaunch remotelaunch_fields_from_msgpack(MsgpackReader &, size_t);
void remotelaunch_fields_to_msgpack(MsgpackWriter &, const Spec::RemoteLaunch &);
Spec::RemoteAlloc remotealloc_fields_from_msgpack(MsgpackReader &, size_t);
void remotealloc_fields_to_msgpack(MsgpackWriter &, const Spec::RemoteAlloc &);
Spec::RemoteTempAlloc remotetempalloc_fields_from_msgpack(MsgpackReader &, size_t);
void remotetempalloc_fields_to_msgpack(MsgpackWriter &, const Spec::RemoteTempAlloc &);
Spec::RemoteFree remotefree_fields_from_msgpack(MsgpackReader &, size_t);
void remotefree_fields_to_msgpack(MsgpackWriter &, const Spec::RemoteFree &);
Spec::RemoteMemcpy remotememcpy_fields_from_msgpack(MsgpackReader &, size_t);
void remotememcpy_fields_to_msgpack(MsgpackWriter &, const Spec::RemoteMemcpy &);
Spec::RemoteSync remotesync_fields_from_msgpack(MsgpackReader &, size_t);
void remotesync_fields_to_msgpack(MsgpackWriter &, const Spec::RemoteSync &);
Spec::Any any_from_msgpack(MsgpackReader &);
void any_to_msgpack(MsgpackWriter &, const Spec::Any &);
}} // namespace Spec
namespace FunctionAffinity { namespace {
FunctionAffinity::Offload offload_fields_from_msgpack(MsgpackReader &, size_t);
FunctionAffinity::Host host_fields_from_msgpack(MsgpackReader &, size_t);
FunctionAffinity::Any any_from_msgpack(MsgpackReader &);
void any_to_msgpack(MsgpackWriter &, const FunctionAffinity::Any &);
}} // namespace FunctionAffinity
namespace Term { namespace {
Term::Float16Const float16const_fields_from_msgpack(MsgpackReader &, size_t);
void float16const_fields_to_msgpack(MsgpackWriter &, const Term::Float16Const &);
Term::Float32Const float32const_fields_from_msgpack(MsgpackReader &, size_t);
void float32const_fields_to_msgpack(MsgpackWriter &, const Term::Float32Const &);
Term::Float64Const float64const_fields_from_msgpack(MsgpackReader &, size_t);
void float64const_fields_to_msgpack(MsgpackWriter &, const Term::Float64Const &);
Term::IntU8Const intu8const_fields_from_msgpack(MsgpackReader &, size_t);
void intu8const_fields_to_msgpack(MsgpackWriter &, const Term::IntU8Const &);
Term::IntU16Const intu16const_fields_from_msgpack(MsgpackReader &, size_t);
void intu16const_fields_to_msgpack(MsgpackWriter &, const Term::IntU16Const &);
Term::IntU32Const intu32const_fields_from_msgpack(MsgpackReader &, size_t);
void intu32const_fields_to_msgpack(MsgpackWriter &, const Term::IntU32Const &);
Term::IntU64Const intu64const_fields_from_msgpack(MsgpackReader &, size_t);
void intu64const_fields_to_msgpack(MsgpackWriter &, const Term::IntU64Const &);
Term::IntS8Const ints8const_fields_from_msgpack(MsgpackReader &, size_t);
void ints8const_fields_to_msgpack(MsgpackWriter &, const Term::IntS8Const &);
Term::IntS16Const ints16const_fields_from_msgpack(MsgpackReader &, size_t);
void ints16const_fields_to_msgpack(MsgpackWriter &, const Term::IntS16Const &);
Term::IntS32Const ints32const_fields_from_msgpack(MsgpackReader &, size_t);
void ints32const_fields_to_msgpack(MsgpackWriter &, const Term::IntS32Const &);
Term::IntS64Const ints64const_fields_from_msgpack(MsgpackReader &, size_t);
void ints64const_fields_to_msgpack(MsgpackWriter &, const Term::IntS64Const &);
Term::Unit0Const unit0const_fields_from_msgpack(MsgpackReader &, size_t);
Term::Bool1Const bool1const_fields_from_msgpack(MsgpackReader &, size_t);
void bool1const_fields_to_msgpack(MsgpackWriter &, const Term::Bool1Const &);
Term::NullPtrConst nullptrconst_fields_from_msgpack(MsgpackReader &, size_t);
void nullptrconst_fields_to_msgpack(MsgpackWriter &, const Term::NullPtrConst &);
Term::StringConst stringconst_fields_from_msgpack(MsgpackReader &, size_t);
void stringconst_fields_to_msgpack(MsgpackWriter &, const Term::StringConst &);
Term::Poison poison_fields_from_msgpack(MsgpackReader &, size_t);
void poison_fields_to_msgpack(MsgpackWriter &, const Term::Poison &);
Term::Defer defer_fields_from_msgpack(MsgpackReader &, size_t);
void defer_fields_to_msgpack(MsgpackWriter &, const Term::Defer &);
Term::Select select_fields_from_msgpack(MsgpackReader &, size_t);
void select_fields_to_msgpack(MsgpackWriter &, const Term::Select &);
Term::Select select_from_msgpack(MsgpackReader &);
void select_to_msgpack(MsgpackWriter &, const Term::Select &);
Term::Any any_from_msgpack(MsgpackReader &);
void any_to_msgpack(MsgpackWriter &, const Term::Any &);
}} // namespace Term
namespace {
Sym sym_fields_from_msgpack(MsgpackReader &, size_t);
void sym_fields_to_msgpack(MsgpackWriter &, const Sym &);
Sym sym_from_msgpack(MsgpackReader &);
void sym_to_msgpack(MsgpackWriter &, const Sym &);
SourcePosition sourceposition_fields_from_msgpack(MsgpackReader &, size_t);
void sourceposition_fields_to_msgpack(MsgpackWriter &, const SourcePosition &);
SourcePosition sourceposition_from_msgpack(MsgpackReader &);
void sourceposition_to_msgpack(MsgpackWriter &, const SourcePosition &);
Origin origin_fields_from_msgpack(MsgpackReader &, size_t);
void origin_fields_to_msgpack(MsgpackWriter &, const Origin &);
Origin origin_from_msgpack(MsgpackReader &);
void origin_to_msgpack(MsgpackWriter &, const Origin &);
Named named_fields_from_msgpack(MsgpackReader &, size_t);
void named_fields_to_msgpack(MsgpackWriter &, const Named &);
Named named_from_msgpack(MsgpackReader &);
void named_to_msgpack(MsgpackWriter &, const Named &);
ExceptionKind exceptionkind_fields_from_msgpack(MsgpackReader &, size_t);
void exceptionkind_fields_to_msgpack(MsgpackWriter &, const ExceptionKind &);
ExceptionKind exceptionkind_from_msgpack(MsgpackReader &);
void exceptionkind_to_msgpack(MsgpackWriter &, const ExceptionKind &);
Handler handler_fields_from_msgpack(MsgpackReader &, size_t);
void handler_fields_to_msgpack(MsgpackWriter &, const Handler &);
Handler handler_from_msgpack(MsgpackReader &);
void handler_to_msgpack(MsgpackWriter &, const Handler &);
ArgBoundary argboundary_fields_from_msgpack(MsgpackReader &, size_t);
void argboundary_fields_to_msgpack(MsgpackWriter &, const ArgBoundary &);
ArgBoundary argboundary_from_msgpack(MsgpackReader &);
void argboundary_to_msgpack(MsgpackWriter &, const ArgBoundary &);
Arg arg_fields_from_msgpack(MsgpackReader &, size_t);
void arg_fields_to_msgpack(MsgpackWriter &, const Arg &);
Arg arg_from_msgpack(MsgpackReader &);
void arg_to_msgpack(MsgpackWriter &, const Arg &);
FunctionDecl functiondecl_fields_from_msgpack(MsgpackReader &, size_t);
void functiondecl_fields_to_msgpack(MsgpackWriter &, const FunctionDecl &);
FunctionDecl functiondecl_from_msgpack(MsgpackReader &);
void functiondecl_to_msgpack(MsgpackWriter &, const FunctionDecl &);
Function function_fields_from_msgpack(MsgpackReader &, size_t);
void function_fields_to_msgpack(MsgpackWriter &, const Function &);
Function function_from_msgpack(MsgpackReader &);
void function_to_msgpack(MsgpackWriter &, const Function &);
StructDef structdef_fields_from_msgpack(MsgpackReader &, size_t);
void structdef_fields_to_msgpack(MsgpackWriter &, const StructDef &);
StructDef structdef_from_msgpack(MsgpackReader &);
void structdef_to_msgpack(MsgpackWriter &, const StructDef &);
MetaEntry metaentry_fields_from_msgpack(MsgpackReader &, size_t);
void metaentry_fields_to_msgpack(MsgpackWriter &, const MetaEntry &);
MetaEntry metaentry_from_msgpack(MsgpackReader &);
void metaentry_to_msgpack(MsgpackWriter &, const MetaEntry &);
Program program_fields_from_msgpack(MsgpackReader &, size_t);
void program_fields_to_msgpack(MsgpackWriter &, const Program &);
Program program_from_msgpack(MsgpackReader &);
void program_to_msgpack(MsgpackWriter &, const Program &);
StructLayoutMember structlayoutmember_fields_from_msgpack(MsgpackReader &, size_t);
void structlayoutmember_fields_to_msgpack(MsgpackWriter &, const StructLayoutMember &);
StructLayoutMember structlayoutmember_from_msgpack(MsgpackReader &);
void structlayoutmember_to_msgpack(MsgpackWriter &, const StructLayoutMember &);
StructLayout structlayout_fields_from_msgpack(MsgpackReader &, size_t);
void structlayout_fields_to_msgpack(MsgpackWriter &, const StructLayout &);
StructLayout structlayout_from_msgpack(MsgpackReader &);
void structlayout_to_msgpack(MsgpackWriter &, const StructLayout &);
CompileEvent compileevent_fields_from_msgpack(MsgpackReader &, size_t);
void compileevent_fields_to_msgpack(MsgpackWriter &, const CompileEvent &);
CompileEvent compileevent_from_msgpack(MsgpackReader &);
void compileevent_to_msgpack(MsgpackWriter &, const CompileEvent &);
PassRunResult passrunresult_fields_from_msgpack(MsgpackReader &, size_t);
void passrunresult_fields_to_msgpack(MsgpackWriter &, const PassRunResult &);
PassRunResult passrunresult_from_msgpack(MsgpackReader &);
void passrunresult_to_msgpack(MsgpackWriter &, const PassRunResult &);
CompileResult compileresult_fields_from_msgpack(MsgpackReader &, size_t);
void compileresult_fields_to_msgpack(MsgpackWriter &, const CompileResult &);
CompileResult compileresult_from_msgpack(MsgpackReader &);
void compileresult_to_msgpack(MsgpackWriter &, const CompileResult &);
Interface interface_fields_from_msgpack(MsgpackReader &, size_t);
void interface_fields_to_msgpack(MsgpackWriter &, const Interface &);
Interface interface_from_msgpack(MsgpackReader &);
void interface_to_msgpack(MsgpackWriter &, const Interface &);
Package package_fields_from_msgpack(MsgpackReader &, size_t);
void package_fields_to_msgpack(MsgpackWriter &, const Package &);
Package package_from_msgpack(MsgpackReader &);
void package_to_msgpack(MsgpackWriter &, const Package &);
ProgramTypeSize programtypesize_fields_from_msgpack(MsgpackReader &, size_t);
void programtypesize_fields_to_msgpack(MsgpackWriter &, const ProgramTypeSize &);
ProgramTypeSize programtypesize_from_msgpack(MsgpackReader &);
void programtypesize_to_msgpack(MsgpackWriter &, const ProgramTypeSize &);
ProgramLinkRequest programlinkrequest_fields_from_msgpack(MsgpackReader &, size_t);
void programlinkrequest_fields_to_msgpack(MsgpackWriter &, const ProgramLinkRequest &);
ProgramLinkRequest programlinkrequest_from_msgpack(MsgpackReader &);
void programlinkrequest_to_msgpack(MsgpackWriter &, const ProgramLinkRequest &);
PackageFragment packagefragment_fields_from_msgpack(MsgpackReader &, size_t);
void packagefragment_fields_to_msgpack(MsgpackWriter &, const PackageFragment &);
PackageFragment packagefragment_from_msgpack(MsgpackReader &);
void packagefragment_to_msgpack(MsgpackWriter &, const PackageFragment &);
PackageLinkRequest packagelinkrequest_fields_from_msgpack(MsgpackReader &, size_t);
void packagelinkrequest_fields_to_msgpack(MsgpackWriter &, const PackageLinkRequest &);
PackageLinkRequest packagelinkrequest_from_msgpack(MsgpackReader &);
void packagelinkrequest_to_msgpack(MsgpackWriter &, const PackageLinkRequest &);
CompileModule compilemodule_fields_from_msgpack(MsgpackReader &, size_t);
void compilemodule_fields_to_msgpack(MsgpackWriter &, const CompileModule &);
CompileModule compilemodule_from_msgpack(MsgpackReader &);
void compilemodule_to_msgpack(MsgpackWriter &, const CompileModule &);
CompileRoot compileroot_fields_from_msgpack(MsgpackReader &, size_t);
void compileroot_fields_to_msgpack(MsgpackWriter &, const CompileRoot &);
CompileRoot compileroot_from_msgpack(MsgpackReader &);
void compileroot_to_msgpack(MsgpackWriter &, const CompileRoot &);
CompileBundle compilebundle_fields_from_msgpack(MsgpackReader &, size_t);
void compilebundle_fields_to_msgpack(MsgpackWriter &, const CompileBundle &);
CompileBundle compilebundle_from_msgpack(MsgpackReader &);
void compilebundle_to_msgpack(MsgpackWriter &, const CompileBundle &);
} // namespace
namespace Intr { namespace {
Intr::BNot bnot_fields_from_msgpack(MsgpackReader &, size_t);
void bnot_fields_to_msgpack(MsgpackWriter &, const Intr::BNot &);
Intr::LogicNot logicnot_fields_from_msgpack(MsgpackReader &, size_t);
void logicnot_fields_to_msgpack(MsgpackWriter &, const Intr::LogicNot &);
Intr::Pos pos_fields_from_msgpack(MsgpackReader &, size_t);
void pos_fields_to_msgpack(MsgpackWriter &, const Intr::Pos &);
Intr::Neg neg_fields_from_msgpack(MsgpackReader &, size_t);
void neg_fields_to_msgpack(MsgpackWriter &, const Intr::Neg &);
Intr::Add add_fields_from_msgpack(MsgpackReader &, size_t);
void add_fields_to_msgpack(MsgpackWriter &, const Intr::Add &);
Intr::Sub sub_fields_from_msgpack(MsgpackReader &, size_t);
void sub_fields_to_msgpack(MsgpackWriter &, const Intr::Sub &);
Intr::Mul mul_fields_from_msgpack(MsgpackReader &, size_t);
void mul_fields_to_msgpack(MsgpackWriter &, const Intr::Mul &);
Intr::Div div_fields_from_msgpack(MsgpackReader &, size_t);
void div_fields_to_msgpack(MsgpackWriter &, const Intr::Div &);
Intr::Rem rem_fields_from_msgpack(MsgpackReader &, size_t);
void rem_fields_to_msgpack(MsgpackWriter &, const Intr::Rem &);
Intr::Min min_fields_from_msgpack(MsgpackReader &, size_t);
void min_fields_to_msgpack(MsgpackWriter &, const Intr::Min &);
Intr::Max max_fields_from_msgpack(MsgpackReader &, size_t);
void max_fields_to_msgpack(MsgpackWriter &, const Intr::Max &);
Intr::BAnd band_fields_from_msgpack(MsgpackReader &, size_t);
void band_fields_to_msgpack(MsgpackWriter &, const Intr::BAnd &);
Intr::BOr bor_fields_from_msgpack(MsgpackReader &, size_t);
void bor_fields_to_msgpack(MsgpackWriter &, const Intr::BOr &);
Intr::BXor bxor_fields_from_msgpack(MsgpackReader &, size_t);
void bxor_fields_to_msgpack(MsgpackWriter &, const Intr::BXor &);
Intr::BSL bsl_fields_from_msgpack(MsgpackReader &, size_t);
void bsl_fields_to_msgpack(MsgpackWriter &, const Intr::BSL &);
Intr::BSR bsr_fields_from_msgpack(MsgpackReader &, size_t);
void bsr_fields_to_msgpack(MsgpackWriter &, const Intr::BSR &);
Intr::BZSR bzsr_fields_from_msgpack(MsgpackReader &, size_t);
void bzsr_fields_to_msgpack(MsgpackWriter &, const Intr::BZSR &);
Intr::PopCount popcount_fields_from_msgpack(MsgpackReader &, size_t);
void popcount_fields_to_msgpack(MsgpackWriter &, const Intr::PopCount &);
Intr::LogicAnd logicand_fields_from_msgpack(MsgpackReader &, size_t);
void logicand_fields_to_msgpack(MsgpackWriter &, const Intr::LogicAnd &);
Intr::LogicOr logicor_fields_from_msgpack(MsgpackReader &, size_t);
void logicor_fields_to_msgpack(MsgpackWriter &, const Intr::LogicOr &);
Intr::LogicEq logiceq_fields_from_msgpack(MsgpackReader &, size_t);
void logiceq_fields_to_msgpack(MsgpackWriter &, const Intr::LogicEq &);
Intr::LogicNeq logicneq_fields_from_msgpack(MsgpackReader &, size_t);
void logicneq_fields_to_msgpack(MsgpackWriter &, const Intr::LogicNeq &);
Intr::LogicLte logiclte_fields_from_msgpack(MsgpackReader &, size_t);
void logiclte_fields_to_msgpack(MsgpackWriter &, const Intr::LogicLte &);
Intr::LogicGte logicgte_fields_from_msgpack(MsgpackReader &, size_t);
void logicgte_fields_to_msgpack(MsgpackWriter &, const Intr::LogicGte &);
Intr::LogicLt logiclt_fields_from_msgpack(MsgpackReader &, size_t);
void logiclt_fields_to_msgpack(MsgpackWriter &, const Intr::LogicLt &);
Intr::LogicGt logicgt_fields_from_msgpack(MsgpackReader &, size_t);
void logicgt_fields_to_msgpack(MsgpackWriter &, const Intr::LogicGt &);
Intr::Any any_from_msgpack(MsgpackReader &);
void any_to_msgpack(MsgpackWriter &, const Intr::Any &);
}} // namespace Intr
namespace AtomicOp { namespace {
AtomicOp::Xchg xchg_fields_from_msgpack(MsgpackReader &, size_t);
AtomicOp::Add add_fields_from_msgpack(MsgpackReader &, size_t);
AtomicOp::Sub sub_fields_from_msgpack(MsgpackReader &, size_t);
AtomicOp::And and_fields_from_msgpack(MsgpackReader &, size_t);
AtomicOp::Or or_fields_from_msgpack(MsgpackReader &, size_t);
AtomicOp::Xor xor_fields_from_msgpack(MsgpackReader &, size_t);
AtomicOp::Min min_fields_from_msgpack(MsgpackReader &, size_t);
AtomicOp::Max max_fields_from_msgpack(MsgpackReader &, size_t);
AtomicOp::Any any_from_msgpack(MsgpackReader &);
void any_to_msgpack(MsgpackWriter &, const AtomicOp::Any &);
}} // namespace AtomicOp
namespace MemScope { namespace {
MemScope::Subgroup subgroup_fields_from_msgpack(MsgpackReader &, size_t);
MemScope::Workgroup workgroup_fields_from_msgpack(MsgpackReader &, size_t);
MemScope::Device device_fields_from_msgpack(MsgpackReader &, size_t);
MemScope::System system_fields_from_msgpack(MsgpackReader &, size_t);
MemScope::Any any_from_msgpack(MsgpackReader &);
void any_to_msgpack(MsgpackWriter &, const MemScope::Any &);
}} // namespace MemScope
namespace FunctionFpMode { namespace {
FunctionFpMode::Relaxed relaxed_fields_from_msgpack(MsgpackReader &, size_t);
FunctionFpMode::Strict strict_fields_from_msgpack(MsgpackReader &, size_t);
FunctionFpMode::Any any_from_msgpack(MsgpackReader &);
void any_to_msgpack(MsgpackWriter &, const FunctionFpMode::Any &);
}} // namespace FunctionFpMode
namespace ArgAccess { namespace {
ArgAccess::Read read_fields_from_msgpack(MsgpackReader &, size_t);
ArgAccess::Write write_fields_from_msgpack(MsgpackReader &, size_t);
ArgAccess::ReadWrite readwrite_fields_from_msgpack(MsgpackReader &, size_t);
ArgAccess::Any any_from_msgpack(MsgpackReader &);
void any_to_msgpack(MsgpackWriter &, const ArgAccess::Any &);
}} // namespace ArgAccess
namespace ArgSizeExpr { namespace {
ArgSizeExpr::Param param_fields_from_msgpack(MsgpackReader &, size_t);
void param_fields_to_msgpack(MsgpackWriter &, const ArgSizeExpr::Param &);
ArgSizeExpr::Const const_fields_from_msgpack(MsgpackReader &, size_t);
void const_fields_to_msgpack(MsgpackWriter &, const ArgSizeExpr::Const &);
ArgSizeExpr::Add add_fields_from_msgpack(MsgpackReader &, size_t);
void add_fields_to_msgpack(MsgpackWriter &, const ArgSizeExpr::Add &);
ArgSizeExpr::Mul mul_fields_from_msgpack(MsgpackReader &, size_t);
void mul_fields_to_msgpack(MsgpackWriter &, const ArgSizeExpr::Mul &);
ArgSizeExpr::Min min_fields_from_msgpack(MsgpackReader &, size_t);
void min_fields_to_msgpack(MsgpackWriter &, const ArgSizeExpr::Min &);
ArgSizeExpr::Result result_fields_from_msgpack(MsgpackReader &, size_t);
void result_fields_to_msgpack(MsgpackWriter &, const ArgSizeExpr::Result &);
ArgSizeExpr::Any any_from_msgpack(MsgpackReader &);
void any_to_msgpack(MsgpackWriter &, const ArgSizeExpr::Any &);
}} // namespace ArgSizeExpr
namespace PathStep { namespace {
PathStep::Field field_fields_from_msgpack(MsgpackReader &, size_t);
void field_fields_to_msgpack(MsgpackWriter &, const PathStep::Field &);
PathStep::Deref deref_fields_from_msgpack(MsgpackReader &, size_t);
PathStep::Index index_fields_from_msgpack(MsgpackReader &, size_t);
void index_fields_to_msgpack(MsgpackWriter &, const PathStep::Index &);
PathStep::IndexDyn indexdyn_fields_from_msgpack(MsgpackReader &, size_t);
void indexdyn_fields_to_msgpack(MsgpackWriter &, const PathStep::IndexDyn &);
PathStep::Any any_from_msgpack(MsgpackReader &);
void any_to_msgpack(MsgpackWriter &, const PathStep::Any &);
}} // namespace PathStep
namespace TypeKind { namespace {
}} // namespace TypeKind
namespace FunctionVisibility { namespace {
FunctionVisibility::Internal internal_fields_from_msgpack(MsgpackReader &, size_t);
FunctionVisibility::Exported exported_fields_from_msgpack(MsgpackReader &, size_t);
FunctionVisibility::Any any_from_msgpack(MsgpackReader &);
void any_to_msgpack(MsgpackWriter &, const FunctionVisibility::Any &);
}} // namespace FunctionVisibility
namespace Type { namespace {
Type::Float16 float16_fields_from_msgpack(MsgpackReader &, size_t);
Type::Float32 float32_fields_from_msgpack(MsgpackReader &, size_t);
Type::Float64 float64_fields_from_msgpack(MsgpackReader &, size_t);
Type::IntU8 intu8_fields_from_msgpack(MsgpackReader &, size_t);
Type::IntU16 intu16_fields_from_msgpack(MsgpackReader &, size_t);
Type::IntU32 intu32_fields_from_msgpack(MsgpackReader &, size_t);
Type::IntU64 intu64_fields_from_msgpack(MsgpackReader &, size_t);
Type::IntS8 ints8_fields_from_msgpack(MsgpackReader &, size_t);
Type::IntS16 ints16_fields_from_msgpack(MsgpackReader &, size_t);
Type::IntS32 ints32_fields_from_msgpack(MsgpackReader &, size_t);
Type::IntS64 ints64_fields_from_msgpack(MsgpackReader &, size_t);
Type::Nothing nothing_fields_from_msgpack(MsgpackReader &, size_t);
Type::Unit0 unit0_fields_from_msgpack(MsgpackReader &, size_t);
Type::Bool1 bool1_fields_from_msgpack(MsgpackReader &, size_t);
Type::Struct struct_fields_from_msgpack(MsgpackReader &, size_t);
void struct_fields_to_msgpack(MsgpackWriter &, const Type::Struct &);
Type::Struct struct_from_msgpack(MsgpackReader &);
void struct_to_msgpack(MsgpackWriter &, const Type::Struct &);
Type::Ptr ptr_fields_from_msgpack(MsgpackReader &, size_t);
void ptr_fields_to_msgpack(MsgpackWriter &, const Type::Ptr &);
Type::Arr arr_fields_from_msgpack(MsgpackReader &, size_t);
void arr_fields_to_msgpack(MsgpackWriter &, const Type::Arr &);
Type::Var var_fields_from_msgpack(MsgpackReader &, size_t);
void var_fields_to_msgpack(MsgpackWriter &, const Type::Var &);
Type::Var var_from_msgpack(MsgpackReader &);
void var_to_msgpack(MsgpackWriter &, const Type::Var &);
Type::Exec exec_fields_from_msgpack(MsgpackReader &, size_t);
void exec_fields_to_msgpack(MsgpackWriter &, const Type::Exec &);
Type::FnRef fnref_fields_from_msgpack(MsgpackReader &, size_t);
void fnref_fields_to_msgpack(MsgpackWriter &, const Type::FnRef &);
Type::Any any_from_msgpack(MsgpackReader &);
void any_to_msgpack(MsgpackWriter &, const Type::Any &);
}} // namespace Type
namespace Direction { namespace {
Direction::LocalToRemote localtoremote_fields_from_msgpack(MsgpackReader &, size_t);
Direction::RemoteToLocal remotetolocal_fields_from_msgpack(MsgpackReader &, size_t);
Direction::RemoteToRemote remotetoremote_fields_from_msgpack(MsgpackReader &, size_t);
Direction::Any any_from_msgpack(MsgpackReader &);
void any_to_msgpack(MsgpackWriter &, const Direction::Any &);
}} // namespace Direction
namespace ArgExtent { namespace {

ArgExtent::Elements elements_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 1) throw std::runtime_error("Expected ArgExtent::Elements with 1 field(s)");
  auto size = ArgSizeExpr::any_from_msgpack(r_);
  return ArgExtent::Elements(std::move(size));
}

void elements_fields_to_msgpack(MsgpackWriter &w_, const ArgExtent::Elements &x_) { ArgSizeExpr::any_to_msgpack(w_, x_.size); }

ArgExtent::Bytes bytes_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 1) throw std::runtime_error("Expected ArgExtent::Bytes with 1 field(s)");
  auto size = ArgSizeExpr::any_from_msgpack(r_);
  return ArgExtent::Bytes(std::move(size));
}

void bytes_fields_to_msgpack(MsgpackWriter &w_, const ArgExtent::Bytes &x_) { ArgSizeExpr::any_to_msgpack(w_, x_.size); }

ArgExtent::Any any_from_msgpack(MsgpackReader &r_) {
  if (r_.nextIsArray()) {
    const auto n_ = r_.readArrayHeader();
    if (n_ == 0) throw std::runtime_error("Expected non-empty sum payload");
    const auto ord_ = r_.readInt32();
    switch (ord_) {
      case 0: return ArgExtent::elements_fields_from_msgpack(r_, n_ - 1);
      case 1: return ArgExtent::bytes_fields_from_msgpack(r_, n_ - 1);
      default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
    }
  } else {
    const auto ord_ = r_.readInt32();
    switch (ord_) {
      case 0:
      case 1: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
    }
  }
}

void any_to_msgpack(MsgpackWriter &w_, const ArgExtent::Any &x_) {
  x_.match_total(
      [&](const ArgExtent::Elements &y_) -> void {
        w_.writeArrayHeader(2);
        w_.writeInt32(0);
        ArgExtent::elements_fields_to_msgpack(w_, y_);
      },
      [&](const ArgExtent::Bytes &y_) -> void {
        w_.writeArrayHeader(2);
        w_.writeInt32(1);
        ArgExtent::bytes_fields_to_msgpack(w_, y_);
      });
}
}} // namespace ArgExtent
namespace Expr { namespace {

Expr::Alias alias_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 1) throw std::runtime_error("Expected Expr::Alias with 1 field(s)");
  auto ref = Term::any_from_msgpack(r_);
  return Expr::Alias(std::move(ref));
}

void alias_fields_to_msgpack(MsgpackWriter &w_, const Expr::Alias &x_) { Term::any_to_msgpack(w_, x_.ref); }

Expr::SpecOp specop_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 1) throw std::runtime_error("Expected Expr::SpecOp with 1 field(s)");
  auto op = Spec::any_from_msgpack(r_);
  return Expr::SpecOp(std::move(op));
}

void specop_fields_to_msgpack(MsgpackWriter &w_, const Expr::SpecOp &x_) { Spec::any_to_msgpack(w_, x_.op); }

Expr::MathOp mathop_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 1) throw std::runtime_error("Expected Expr::MathOp with 1 field(s)");
  auto op = Math::any_from_msgpack(r_);
  return Expr::MathOp(std::move(op));
}

void mathop_fields_to_msgpack(MsgpackWriter &w_, const Expr::MathOp &x_) { Math::any_to_msgpack(w_, x_.op); }

Expr::IntrOp introp_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 1) throw std::runtime_error("Expected Expr::IntrOp with 1 field(s)");
  auto op = Intr::any_from_msgpack(r_);
  return Expr::IntrOp(std::move(op));
}

void introp_fields_to_msgpack(MsgpackWriter &w_, const Expr::IntrOp &x_) { Intr::any_to_msgpack(w_, x_.op); }

Expr::Cast cast_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected Expr::Cast with 2 field(s)");
  auto from = Term::any_from_msgpack(r_);
  auto as = Type::any_from_msgpack(r_);
  return {std::move(from), std::move(as)};
}

void cast_fields_to_msgpack(MsgpackWriter &w_, const Expr::Cast &x_) {
  Term::any_to_msgpack(w_, x_.from);
  Type::any_to_msgpack(w_, x_.as);
}

Expr::BitCast bitcast_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected Expr::BitCast with 2 field(s)");
  auto from = Term::any_from_msgpack(r_);
  auto as = Type::any_from_msgpack(r_);
  return {std::move(from), std::move(as)};
}

void bitcast_fields_to_msgpack(MsgpackWriter &w_, const Expr::BitCast &x_) {
  Term::any_to_msgpack(w_, x_.from);
  Type::any_to_msgpack(w_, x_.as);
}

Expr::Index index_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 3) throw std::runtime_error("Expected Expr::Index with 3 field(s)");
  auto lhs = Term::any_from_msgpack(r_);
  auto idx = Term::any_from_msgpack(r_);
  auto comp = Type::any_from_msgpack(r_);
  return {std::move(lhs), std::move(idx), std::move(comp)};
}

void index_fields_to_msgpack(MsgpackWriter &w_, const Expr::Index &x_) {
  Term::any_to_msgpack(w_, x_.lhs);
  Term::any_to_msgpack(w_, x_.idx);
  Type::any_to_msgpack(w_, x_.comp);
}

Expr::RefTo refto_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 5) throw std::runtime_error("Expected Expr::RefTo with 5 field(s)");
  auto lhs = Term::any_from_msgpack(r_);
  std::optional<Term::Any> idx;
  if (!r_.tryReadNil()) {
    auto idx_value = Term::any_from_msgpack(r_);
    idx = std::move(idx_value);
  }
  auto comp = Type::any_from_msgpack(r_);
  auto space = TypeSpace::any_from_msgpack(r_);
  auto region = Region::any_from_msgpack(r_);
  return {std::move(lhs), std::move(idx), std::move(comp), std::move(space), std::move(region)};
}

void refto_fields_to_msgpack(MsgpackWriter &w_, const Expr::RefTo &x_) {
  Term::any_to_msgpack(w_, x_.lhs);
  if (x_.idx) {
    Term::any_to_msgpack(w_, (*x_.idx));
  } else {
    w_.writeNil();
  }
  Type::any_to_msgpack(w_, x_.comp);
  TypeSpace::any_to_msgpack(w_, x_.space);
  Region::any_to_msgpack(w_, x_.region);
}

Expr::Alloc alloc_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 4) throw std::runtime_error("Expected Expr::Alloc with 4 field(s)");
  auto comp = Type::any_from_msgpack(r_);
  auto size = Term::any_from_msgpack(r_);
  auto space = TypeSpace::any_from_msgpack(r_);
  auto region = Region::any_from_msgpack(r_);
  return {std::move(comp), std::move(size), std::move(space), std::move(region)};
}

void alloc_fields_to_msgpack(MsgpackWriter &w_, const Expr::Alloc &x_) {
  Type::any_to_msgpack(w_, x_.comp);
  Term::any_to_msgpack(w_, x_.size);
  TypeSpace::any_to_msgpack(w_, x_.space);
  Region::any_to_msgpack(w_, x_.region);
}

Expr::Invoke invoke_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 5) throw std::runtime_error("Expected Expr::Invoke with 5 field(s)");
  auto callee = Type::any_from_msgpack(r_);
  std::vector<Type::Any> tpeArgs;
  {
    const auto tpeArgs_size = r_.readArrayHeader();
    tpeArgs.reserve(tpeArgs_size);
    for (size_t tpeArgs_idx = 0; tpeArgs_idx < tpeArgs_size; ++tpeArgs_idx) {
      auto tpeArgs_elem = Type::any_from_msgpack(r_);
      tpeArgs.emplace_back(std::move(tpeArgs_elem));
    }
  }
  std::optional<Term::Any> receiver;
  if (!r_.tryReadNil()) {
    auto receiver_value = Term::any_from_msgpack(r_);
    receiver = std::move(receiver_value);
  }
  std::vector<Term::Any> args;
  {
    const auto args_size = r_.readArrayHeader();
    args.reserve(args_size);
    for (size_t args_idx = 0; args_idx < args_size; ++args_idx) {
      auto args_elem = Term::any_from_msgpack(r_);
      args.emplace_back(std::move(args_elem));
    }
  }
  auto rtn = Type::any_from_msgpack(r_);
  return {std::move(callee), std::move(tpeArgs), std::move(receiver), std::move(args), std::move(rtn)};
}

void invoke_fields_to_msgpack(MsgpackWriter &w_, const Expr::Invoke &x_) {
  Type::any_to_msgpack(w_, x_.callee);
  w_.writeArrayHeader(x_.tpeArgs.size());
  for (const auto &v0_ : x_.tpeArgs) {
    Type::any_to_msgpack(w_, v0_);
  }
  if (x_.receiver) {
    Term::any_to_msgpack(w_, (*x_.receiver));
  } else {
    w_.writeNil();
  }
  w_.writeArrayHeader(x_.args.size());
  for (const auto &v0_ : x_.args) {
    Term::any_to_msgpack(w_, v0_);
  }
  Type::any_to_msgpack(w_, x_.rtn);
}

Expr::ForeignCall foreigncall_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 3) throw std::runtime_error("Expected Expr::ForeignCall with 3 field(s)");
  auto name = r_.readString();
  std::vector<Term::Any> args;
  {
    const auto args_size = r_.readArrayHeader();
    args.reserve(args_size);
    for (size_t args_idx = 0; args_idx < args_size; ++args_idx) {
      auto args_elem = Term::any_from_msgpack(r_);
      args.emplace_back(std::move(args_elem));
    }
  }
  auto rtn = Type::any_from_msgpack(r_);
  return {std::move(name), std::move(args), std::move(rtn)};
}

void foreigncall_fields_to_msgpack(MsgpackWriter &w_, const Expr::ForeignCall &x_) {
  w_.writeString(x_.name);
  w_.writeArrayHeader(x_.args.size());
  for (const auto &v0_ : x_.args) {
    Term::any_to_msgpack(w_, v0_);
  }
  Type::any_to_msgpack(w_, x_.rtn);
}

Expr::OffsetOf offsetof_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected Expr::OffsetOf with 2 field(s)");
  auto structTpe = Type::any_from_msgpack(r_);
  auto field = r_.readString();
  return {std::move(structTpe), std::move(field)};
}

void offsetof_fields_to_msgpack(MsgpackWriter &w_, const Expr::OffsetOf &x_) {
  Type::any_to_msgpack(w_, x_.structTpe);
  w_.writeString(x_.field);
}

Expr::SizeOf sizeof_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 1) throw std::runtime_error("Expected Expr::SizeOf with 1 field(s)");
  auto forTpe = Type::any_from_msgpack(r_);
  return Expr::SizeOf(std::move(forTpe));
}

void sizeof_fields_to_msgpack(MsgpackWriter &w_, const Expr::SizeOf &x_) { Type::any_to_msgpack(w_, x_.forTpe); }

Expr::Any any_from_msgpack(MsgpackReader &r_) {
  if (r_.nextIsArray()) {
    const auto n_ = r_.readArrayHeader();
    if (n_ == 0) throw std::runtime_error("Expected non-empty sum payload");
    const auto ord_ = r_.readInt32();
    switch (ord_) {
      case 0: return Expr::alias_fields_from_msgpack(r_, n_ - 1);
      case 1: return Expr::specop_fields_from_msgpack(r_, n_ - 1);
      case 2: return Expr::mathop_fields_from_msgpack(r_, n_ - 1);
      case 3: return Expr::introp_fields_from_msgpack(r_, n_ - 1);
      case 4: return Expr::cast_fields_from_msgpack(r_, n_ - 1);
      case 5: return Expr::bitcast_fields_from_msgpack(r_, n_ - 1);
      case 6: return Expr::index_fields_from_msgpack(r_, n_ - 1);
      case 7: return Expr::refto_fields_from_msgpack(r_, n_ - 1);
      case 8: return Expr::alloc_fields_from_msgpack(r_, n_ - 1);
      case 9: return Expr::invoke_fields_from_msgpack(r_, n_ - 1);
      case 10: return Expr::foreigncall_fields_from_msgpack(r_, n_ - 1);
      case 11: return Expr::offsetof_fields_from_msgpack(r_, n_ - 1);
      case 12: return Expr::sizeof_fields_from_msgpack(r_, n_ - 1);
      default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
    }
  } else {
    const auto ord_ = r_.readInt32();
    switch (ord_) {
      case 0:
      case 1:
      case 2:
      case 3:
      case 4:
      case 5:
      case 6:
      case 7:
      case 8:
      case 9:
      case 10:
      case 11:
      case 12: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
    }
  }
}

void any_to_msgpack(MsgpackWriter &w_, const Expr::Any &x_) {
  x_.match_total(
      [&](const Expr::Alias &y_) -> void {
        w_.writeArrayHeader(2);
        w_.writeInt32(0);
        Expr::alias_fields_to_msgpack(w_, y_);
      },
      [&](const Expr::SpecOp &y_) -> void {
        w_.writeArrayHeader(2);
        w_.writeInt32(1);
        Expr::specop_fields_to_msgpack(w_, y_);
      },
      [&](const Expr::MathOp &y_) -> void {
        w_.writeArrayHeader(2);
        w_.writeInt32(2);
        Expr::mathop_fields_to_msgpack(w_, y_);
      },
      [&](const Expr::IntrOp &y_) -> void {
        w_.writeArrayHeader(2);
        w_.writeInt32(3);
        Expr::introp_fields_to_msgpack(w_, y_);
      },
      [&](const Expr::Cast &y_) -> void {
        w_.writeArrayHeader(3);
        w_.writeInt32(4);
        Expr::cast_fields_to_msgpack(w_, y_);
      },
      [&](const Expr::BitCast &y_) -> void {
        w_.writeArrayHeader(3);
        w_.writeInt32(5);
        Expr::bitcast_fields_to_msgpack(w_, y_);
      },
      [&](const Expr::Index &y_) -> void {
        w_.writeArrayHeader(4);
        w_.writeInt32(6);
        Expr::index_fields_to_msgpack(w_, y_);
      },
      [&](const Expr::RefTo &y_) -> void {
        w_.writeArrayHeader(6);
        w_.writeInt32(7);
        Expr::refto_fields_to_msgpack(w_, y_);
      },
      [&](const Expr::Alloc &y_) -> void {
        w_.writeArrayHeader(5);
        w_.writeInt32(8);
        Expr::alloc_fields_to_msgpack(w_, y_);
      },
      [&](const Expr::Invoke &y_) -> void {
        w_.writeArrayHeader(6);
        w_.writeInt32(9);
        Expr::invoke_fields_to_msgpack(w_, y_);
      },
      [&](const Expr::ForeignCall &y_) -> void {
        w_.writeArrayHeader(4);
        w_.writeInt32(10);
        Expr::foreigncall_fields_to_msgpack(w_, y_);
      },
      [&](const Expr::OffsetOf &y_) -> void {
        w_.writeArrayHeader(3);
        w_.writeInt32(11);
        Expr::offsetof_fields_to_msgpack(w_, y_);
      },
      [&](const Expr::SizeOf &y_) -> void {
        w_.writeArrayHeader(2);
        w_.writeInt32(12);
        Expr::sizeof_fields_to_msgpack(w_, y_);
      });
}
}} // namespace Expr
namespace Region { namespace {

Region::Rooted rooted_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 1) throw std::runtime_error("Expected Region::Rooted with 1 field(s)");
  auto root = named_from_msgpack(r_);
  return Region::Rooted(std::move(root));
}

void rooted_fields_to_msgpack(MsgpackWriter &w_, const Region::Rooted &x_) { named_to_msgpack(w_, x_.root); }

Region::Opaque opaque_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected Region::Opaque with 0 field(s)");
  return {};
}

Region::Any any_from_msgpack(MsgpackReader &r_) {
  if (r_.nextIsArray()) {
    const auto n_ = r_.readArrayHeader();
    if (n_ == 0) throw std::runtime_error("Expected non-empty sum payload");
    const auto ord_ = r_.readInt32();
    switch (ord_) {
      case 0: return Region::rooted_fields_from_msgpack(r_, n_ - 1);
      case 1: return Region::opaque_fields_from_msgpack(r_, n_ - 1);
      default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
    }
  } else {
    const auto ord_ = r_.readInt32();
    switch (ord_) {
      case 0: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      case 1: return Region::opaque_fields_from_msgpack(r_, 0);
      default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
    }
  }
}

void any_to_msgpack(MsgpackWriter &w_, const Region::Any &x_) {
  x_.match_total(
      [&](const Region::Rooted &y_) -> void {
        w_.writeArrayHeader(2);
        w_.writeInt32(0);
        Region::rooted_fields_to_msgpack(w_, y_);
      },
      [&](const Region::Opaque &y_) -> void { w_.writeInt32(1); });
}
}} // namespace Region
namespace Stmt { namespace {

Stmt::Var var_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 3) throw std::runtime_error("Expected Stmt::Var with 3 field(s)");
  auto name = named_from_msgpack(r_);
  std::optional<Expr::Any> expr;
  if (!r_.tryReadNil()) {
    auto expr_value = Expr::any_from_msgpack(r_);
    expr = std::move(expr_value);
  }
  const auto isMutable = r_.readBoolean();
  return {std::move(name), std::move(expr), isMutable};
}

void var_fields_to_msgpack(MsgpackWriter &w_, const Stmt::Var &x_) {
  named_to_msgpack(w_, x_.name);
  if (x_.expr) {
    Expr::any_to_msgpack(w_, (*x_.expr));
  } else {
    w_.writeNil();
  }
  w_.writeBoolean(x_.isMutable);
}

Stmt::Mut mut_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected Stmt::Mut with 2 field(s)");
  auto name = Term::select_from_msgpack(r_);
  auto expr = Expr::any_from_msgpack(r_);
  return {std::move(name), std::move(expr)};
}

void mut_fields_to_msgpack(MsgpackWriter &w_, const Stmt::Mut &x_) {
  Term::select_to_msgpack(w_, x_.name);
  Expr::any_to_msgpack(w_, x_.expr);
}

Stmt::Update update_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 3) throw std::runtime_error("Expected Stmt::Update with 3 field(s)");
  auto lhs = Term::select_from_msgpack(r_);
  auto idx = Term::any_from_msgpack(r_);
  auto value = Term::any_from_msgpack(r_);
  return {std::move(lhs), std::move(idx), std::move(value)};
}

void update_fields_to_msgpack(MsgpackWriter &w_, const Stmt::Update &x_) {
  Term::select_to_msgpack(w_, x_.lhs);
  Term::any_to_msgpack(w_, x_.idx);
  Term::any_to_msgpack(w_, x_.value);
}

Stmt::While while_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected Stmt::While with 2 field(s)");
  auto cond = Term::any_from_msgpack(r_);
  std::vector<Stmt::Any> body;
  {
    const auto body_size = r_.readArrayHeader();
    body.reserve(body_size);
    for (size_t body_idx = 0; body_idx < body_size; ++body_idx) {
      auto body_elem = Stmt::any_from_msgpack(r_);
      body.emplace_back(std::move(body_elem));
    }
  }
  return {std::move(cond), std::move(body)};
}

void while_fields_to_msgpack(MsgpackWriter &w_, const Stmt::While &x_) {
  Term::any_to_msgpack(w_, x_.cond);
  w_.writeArrayHeader(x_.body.size());
  for (const auto &v0_ : x_.body) {
    Stmt::any_to_msgpack(w_, v0_);
  }
}

Stmt::ForRange forrange_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 5) throw std::runtime_error("Expected Stmt::ForRange with 5 field(s)");
  auto induction = named_from_msgpack(r_);
  auto lbIncl = Term::any_from_msgpack(r_);
  auto ubExcl = Term::any_from_msgpack(r_);
  auto step = Term::any_from_msgpack(r_);
  std::vector<Stmt::Any> body;
  {
    const auto body_size = r_.readArrayHeader();
    body.reserve(body_size);
    for (size_t body_idx = 0; body_idx < body_size; ++body_idx) {
      auto body_elem = Stmt::any_from_msgpack(r_);
      body.emplace_back(std::move(body_elem));
    }
  }
  return {std::move(induction), std::move(lbIncl), std::move(ubExcl), std::move(step), std::move(body)};
}

void forrange_fields_to_msgpack(MsgpackWriter &w_, const Stmt::ForRange &x_) {
  named_to_msgpack(w_, x_.induction);
  Term::any_to_msgpack(w_, x_.lbIncl);
  Term::any_to_msgpack(w_, x_.ubExcl);
  Term::any_to_msgpack(w_, x_.step);
  w_.writeArrayHeader(x_.body.size());
  for (const auto &v0_ : x_.body) {
    Stmt::any_to_msgpack(w_, v0_);
  }
}

Stmt::Break break_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected Stmt::Break with 0 field(s)");
  return {};
}

Stmt::Cont cont_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected Stmt::Cont with 0 field(s)");
  return {};
}

Stmt::Cond cond_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 3) throw std::runtime_error("Expected Stmt::Cond with 3 field(s)");
  auto cond = Term::any_from_msgpack(r_);
  std::vector<Stmt::Any> trueBr;
  {
    const auto trueBr_size = r_.readArrayHeader();
    trueBr.reserve(trueBr_size);
    for (size_t trueBr_idx = 0; trueBr_idx < trueBr_size; ++trueBr_idx) {
      auto trueBr_elem = Stmt::any_from_msgpack(r_);
      trueBr.emplace_back(std::move(trueBr_elem));
    }
  }
  std::vector<Stmt::Any> falseBr;
  {
    const auto falseBr_size = r_.readArrayHeader();
    falseBr.reserve(falseBr_size);
    for (size_t falseBr_idx = 0; falseBr_idx < falseBr_size; ++falseBr_idx) {
      auto falseBr_elem = Stmt::any_from_msgpack(r_);
      falseBr.emplace_back(std::move(falseBr_elem));
    }
  }
  return {std::move(cond), std::move(trueBr), std::move(falseBr)};
}

void cond_fields_to_msgpack(MsgpackWriter &w_, const Stmt::Cond &x_) {
  Term::any_to_msgpack(w_, x_.cond);
  w_.writeArrayHeader(x_.trueBr.size());
  for (const auto &v0_ : x_.trueBr) {
    Stmt::any_to_msgpack(w_, v0_);
  }
  w_.writeArrayHeader(x_.falseBr.size());
  for (const auto &v0_ : x_.falseBr) {
    Stmt::any_to_msgpack(w_, v0_);
  }
}

Stmt::Return return_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 1) throw std::runtime_error("Expected Stmt::Return with 1 field(s)");
  auto value = Expr::any_from_msgpack(r_);
  return Stmt::Return(std::move(value));
}

void return_fields_to_msgpack(MsgpackWriter &w_, const Stmt::Return &x_) { Expr::any_to_msgpack(w_, x_.value); }

Stmt::Annotated annotated_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 3) throw std::runtime_error("Expected Stmt::Annotated with 3 field(s)");
  auto inner = Stmt::any_from_msgpack(r_);
  std::optional<SourcePosition> pos;
  if (!r_.tryReadNil()) {
    auto pos_value = sourceposition_from_msgpack(r_);
    pos = std::move(pos_value);
  }
  std::optional<std::string> comment;
  if (!r_.tryReadNil()) {
    auto comment_value = r_.readString();
    comment = std::move(comment_value);
  }
  return {std::move(inner), std::move(pos), std::move(comment)};
}

void annotated_fields_to_msgpack(MsgpackWriter &w_, const Stmt::Annotated &x_) {
  Stmt::any_to_msgpack(w_, x_.inner);
  if (x_.pos) {
    sourceposition_to_msgpack(w_, (*x_.pos));
  } else {
    w_.writeNil();
  }
  if (x_.comment) {
    w_.writeString((*x_.comment));
  } else {
    w_.writeNil();
  }
}

Stmt::Try try_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 3) throw std::runtime_error("Expected Stmt::Try with 3 field(s)");
  std::vector<Stmt::Any> body;
  {
    const auto body_size = r_.readArrayHeader();
    body.reserve(body_size);
    for (size_t body_idx = 0; body_idx < body_size; ++body_idx) {
      auto body_elem = Stmt::any_from_msgpack(r_);
      body.emplace_back(std::move(body_elem));
    }
  }
  std::vector<Handler> handlers;
  {
    const auto handlers_size = r_.readArrayHeader();
    handlers.reserve(handlers_size);
    for (size_t handlers_idx = 0; handlers_idx < handlers_size; ++handlers_idx) {
      auto handlers_elem = handler_from_msgpack(r_);
      handlers.emplace_back(std::move(handlers_elem));
    }
  }
  std::vector<Stmt::Any> fin;
  {
    const auto fin_size = r_.readArrayHeader();
    fin.reserve(fin_size);
    for (size_t fin_idx = 0; fin_idx < fin_size; ++fin_idx) {
      auto fin_elem = Stmt::any_from_msgpack(r_);
      fin.emplace_back(std::move(fin_elem));
    }
  }
  return {std::move(body), std::move(handlers), std::move(fin)};
}

void try_fields_to_msgpack(MsgpackWriter &w_, const Stmt::Try &x_) {
  w_.writeArrayHeader(x_.body.size());
  for (const auto &v0_ : x_.body) {
    Stmt::any_to_msgpack(w_, v0_);
  }
  w_.writeArrayHeader(x_.handlers.size());
  for (const auto &v0_ : x_.handlers) {
    handler_to_msgpack(w_, v0_);
  }
  w_.writeArrayHeader(x_.fin.size());
  for (const auto &v0_ : x_.fin) {
    Stmt::any_to_msgpack(w_, v0_);
  }
}

Stmt::Raise raise_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 3) throw std::runtime_error("Expected Stmt::Raise with 3 field(s)");
  auto value = Term::any_from_msgpack(r_);
  auto exceptionKind = exceptionkind_from_msgpack(r_);
  std::vector<Stmt::Any> cleanup;
  {
    const auto cleanup_size = r_.readArrayHeader();
    cleanup.reserve(cleanup_size);
    for (size_t cleanup_idx = 0; cleanup_idx < cleanup_size; ++cleanup_idx) {
      auto cleanup_elem = Stmt::any_from_msgpack(r_);
      cleanup.emplace_back(std::move(cleanup_elem));
    }
  }
  return {std::move(value), std::move(exceptionKind), std::move(cleanup)};
}

void raise_fields_to_msgpack(MsgpackWriter &w_, const Stmt::Raise &x_) {
  Term::any_to_msgpack(w_, x_.value);
  exceptionkind_to_msgpack(w_, x_.exceptionKind);
  w_.writeArrayHeader(x_.cleanup.size());
  for (const auto &v0_ : x_.cleanup) {
    Stmt::any_to_msgpack(w_, v0_);
  }
}

Stmt::Rethrow rethrow_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected Stmt::Rethrow with 0 field(s)");
  return {};
}

Stmt::Any any_from_msgpack(MsgpackReader &r_) {
  if (r_.nextIsArray()) {
    const auto n_ = r_.readArrayHeader();
    if (n_ == 0) throw std::runtime_error("Expected non-empty sum payload");
    const auto ord_ = r_.readInt32();
    switch (ord_) {
      case 0: return Stmt::var_fields_from_msgpack(r_, n_ - 1);
      case 1: return Stmt::mut_fields_from_msgpack(r_, n_ - 1);
      case 2: return Stmt::update_fields_from_msgpack(r_, n_ - 1);
      case 3: return Stmt::while_fields_from_msgpack(r_, n_ - 1);
      case 4: return Stmt::forrange_fields_from_msgpack(r_, n_ - 1);
      case 5: return Stmt::break_fields_from_msgpack(r_, n_ - 1);
      case 6: return Stmt::cont_fields_from_msgpack(r_, n_ - 1);
      case 7: return Stmt::cond_fields_from_msgpack(r_, n_ - 1);
      case 8: return Stmt::return_fields_from_msgpack(r_, n_ - 1);
      case 9: return Stmt::annotated_fields_from_msgpack(r_, n_ - 1);
      case 10: return Stmt::try_fields_from_msgpack(r_, n_ - 1);
      case 11: return Stmt::raise_fields_from_msgpack(r_, n_ - 1);
      case 12: return Stmt::rethrow_fields_from_msgpack(r_, n_ - 1);
      default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
    }
  } else {
    const auto ord_ = r_.readInt32();
    switch (ord_) {
      case 0:
      case 1:
      case 2:
      case 3:
      case 4:
      case 7:
      case 8:
      case 9:
      case 10:
      case 11: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      case 5: return Stmt::break_fields_from_msgpack(r_, 0);
      case 6: return Stmt::cont_fields_from_msgpack(r_, 0);
      case 12: return Stmt::rethrow_fields_from_msgpack(r_, 0);
      default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
    }
  }
}

void any_to_msgpack(MsgpackWriter &w_, const Stmt::Any &x_) {
  x_.match_total(
      [&](const Stmt::Var &y_) -> void {
        w_.writeArrayHeader(4);
        w_.writeInt32(0);
        Stmt::var_fields_to_msgpack(w_, y_);
      },
      [&](const Stmt::Mut &y_) -> void {
        w_.writeArrayHeader(3);
        w_.writeInt32(1);
        Stmt::mut_fields_to_msgpack(w_, y_);
      },
      [&](const Stmt::Update &y_) -> void {
        w_.writeArrayHeader(4);
        w_.writeInt32(2);
        Stmt::update_fields_to_msgpack(w_, y_);
      },
      [&](const Stmt::While &y_) -> void {
        w_.writeArrayHeader(3);
        w_.writeInt32(3);
        Stmt::while_fields_to_msgpack(w_, y_);
      },
      [&](const Stmt::ForRange &y_) -> void {
        w_.writeArrayHeader(6);
        w_.writeInt32(4);
        Stmt::forrange_fields_to_msgpack(w_, y_);
      },
      [&](const Stmt::Break &y_) -> void { w_.writeInt32(5); }, [&](const Stmt::Cont &y_) -> void { w_.writeInt32(6); },
      [&](const Stmt::Cond &y_) -> void {
        w_.writeArrayHeader(4);
        w_.writeInt32(7);
        Stmt::cond_fields_to_msgpack(w_, y_);
      },
      [&](const Stmt::Return &y_) -> void {
        w_.writeArrayHeader(2);
        w_.writeInt32(8);
        Stmt::return_fields_to_msgpack(w_, y_);
      },
      [&](const Stmt::Annotated &y_) -> void {
        w_.writeArrayHeader(4);
        w_.writeInt32(9);
        Stmt::annotated_fields_to_msgpack(w_, y_);
      },
      [&](const Stmt::Try &y_) -> void {
        w_.writeArrayHeader(4);
        w_.writeInt32(10);
        Stmt::try_fields_to_msgpack(w_, y_);
      },
      [&](const Stmt::Raise &y_) -> void {
        w_.writeArrayHeader(4);
        w_.writeInt32(11);
        Stmt::raise_fields_to_msgpack(w_, y_);
      },
      [&](const Stmt::Rethrow &y_) -> void { w_.writeInt32(12); });
}
}} // namespace Stmt
namespace Math { namespace {

Math::Abs abs_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected Math::Abs with 2 field(s)");
  auto x = Term::any_from_msgpack(r_);
  auto rtn = Type::any_from_msgpack(r_);
  return {std::move(x), std::move(rtn)};
}

void abs_fields_to_msgpack(MsgpackWriter &w_, const Math::Abs &x_) {
  Term::any_to_msgpack(w_, x_.x);
  Type::any_to_msgpack(w_, x_.rtn);
}

Math::Sin sin_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected Math::Sin with 2 field(s)");
  auto x = Term::any_from_msgpack(r_);
  auto rtn = Type::any_from_msgpack(r_);
  return {std::move(x), std::move(rtn)};
}

void sin_fields_to_msgpack(MsgpackWriter &w_, const Math::Sin &x_) {
  Term::any_to_msgpack(w_, x_.x);
  Type::any_to_msgpack(w_, x_.rtn);
}

Math::Cos cos_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected Math::Cos with 2 field(s)");
  auto x = Term::any_from_msgpack(r_);
  auto rtn = Type::any_from_msgpack(r_);
  return {std::move(x), std::move(rtn)};
}

void cos_fields_to_msgpack(MsgpackWriter &w_, const Math::Cos &x_) {
  Term::any_to_msgpack(w_, x_.x);
  Type::any_to_msgpack(w_, x_.rtn);
}

Math::Tan tan_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected Math::Tan with 2 field(s)");
  auto x = Term::any_from_msgpack(r_);
  auto rtn = Type::any_from_msgpack(r_);
  return {std::move(x), std::move(rtn)};
}

void tan_fields_to_msgpack(MsgpackWriter &w_, const Math::Tan &x_) {
  Term::any_to_msgpack(w_, x_.x);
  Type::any_to_msgpack(w_, x_.rtn);
}

Math::Asin asin_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected Math::Asin with 2 field(s)");
  auto x = Term::any_from_msgpack(r_);
  auto rtn = Type::any_from_msgpack(r_);
  return {std::move(x), std::move(rtn)};
}

void asin_fields_to_msgpack(MsgpackWriter &w_, const Math::Asin &x_) {
  Term::any_to_msgpack(w_, x_.x);
  Type::any_to_msgpack(w_, x_.rtn);
}

Math::Acos acos_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected Math::Acos with 2 field(s)");
  auto x = Term::any_from_msgpack(r_);
  auto rtn = Type::any_from_msgpack(r_);
  return {std::move(x), std::move(rtn)};
}

void acos_fields_to_msgpack(MsgpackWriter &w_, const Math::Acos &x_) {
  Term::any_to_msgpack(w_, x_.x);
  Type::any_to_msgpack(w_, x_.rtn);
}

Math::Atan atan_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected Math::Atan with 2 field(s)");
  auto x = Term::any_from_msgpack(r_);
  auto rtn = Type::any_from_msgpack(r_);
  return {std::move(x), std::move(rtn)};
}

void atan_fields_to_msgpack(MsgpackWriter &w_, const Math::Atan &x_) {
  Term::any_to_msgpack(w_, x_.x);
  Type::any_to_msgpack(w_, x_.rtn);
}

Math::Sinh sinh_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected Math::Sinh with 2 field(s)");
  auto x = Term::any_from_msgpack(r_);
  auto rtn = Type::any_from_msgpack(r_);
  return {std::move(x), std::move(rtn)};
}

void sinh_fields_to_msgpack(MsgpackWriter &w_, const Math::Sinh &x_) {
  Term::any_to_msgpack(w_, x_.x);
  Type::any_to_msgpack(w_, x_.rtn);
}

Math::Cosh cosh_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected Math::Cosh with 2 field(s)");
  auto x = Term::any_from_msgpack(r_);
  auto rtn = Type::any_from_msgpack(r_);
  return {std::move(x), std::move(rtn)};
}

void cosh_fields_to_msgpack(MsgpackWriter &w_, const Math::Cosh &x_) {
  Term::any_to_msgpack(w_, x_.x);
  Type::any_to_msgpack(w_, x_.rtn);
}

Math::Tanh tanh_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected Math::Tanh with 2 field(s)");
  auto x = Term::any_from_msgpack(r_);
  auto rtn = Type::any_from_msgpack(r_);
  return {std::move(x), std::move(rtn)};
}

void tanh_fields_to_msgpack(MsgpackWriter &w_, const Math::Tanh &x_) {
  Term::any_to_msgpack(w_, x_.x);
  Type::any_to_msgpack(w_, x_.rtn);
}

Math::Signum signum_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected Math::Signum with 2 field(s)");
  auto x = Term::any_from_msgpack(r_);
  auto rtn = Type::any_from_msgpack(r_);
  return {std::move(x), std::move(rtn)};
}

void signum_fields_to_msgpack(MsgpackWriter &w_, const Math::Signum &x_) {
  Term::any_to_msgpack(w_, x_.x);
  Type::any_to_msgpack(w_, x_.rtn);
}

Math::Round round_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected Math::Round with 2 field(s)");
  auto x = Term::any_from_msgpack(r_);
  auto rtn = Type::any_from_msgpack(r_);
  return {std::move(x), std::move(rtn)};
}

void round_fields_to_msgpack(MsgpackWriter &w_, const Math::Round &x_) {
  Term::any_to_msgpack(w_, x_.x);
  Type::any_to_msgpack(w_, x_.rtn);
}

Math::Ceil ceil_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected Math::Ceil with 2 field(s)");
  auto x = Term::any_from_msgpack(r_);
  auto rtn = Type::any_from_msgpack(r_);
  return {std::move(x), std::move(rtn)};
}

void ceil_fields_to_msgpack(MsgpackWriter &w_, const Math::Ceil &x_) {
  Term::any_to_msgpack(w_, x_.x);
  Type::any_to_msgpack(w_, x_.rtn);
}

Math::Floor floor_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected Math::Floor with 2 field(s)");
  auto x = Term::any_from_msgpack(r_);
  auto rtn = Type::any_from_msgpack(r_);
  return {std::move(x), std::move(rtn)};
}

void floor_fields_to_msgpack(MsgpackWriter &w_, const Math::Floor &x_) {
  Term::any_to_msgpack(w_, x_.x);
  Type::any_to_msgpack(w_, x_.rtn);
}

Math::Rint rint_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected Math::Rint with 2 field(s)");
  auto x = Term::any_from_msgpack(r_);
  auto rtn = Type::any_from_msgpack(r_);
  return {std::move(x), std::move(rtn)};
}

void rint_fields_to_msgpack(MsgpackWriter &w_, const Math::Rint &x_) {
  Term::any_to_msgpack(w_, x_.x);
  Type::any_to_msgpack(w_, x_.rtn);
}

Math::Sqrt sqrt_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected Math::Sqrt with 2 field(s)");
  auto x = Term::any_from_msgpack(r_);
  auto rtn = Type::any_from_msgpack(r_);
  return {std::move(x), std::move(rtn)};
}

void sqrt_fields_to_msgpack(MsgpackWriter &w_, const Math::Sqrt &x_) {
  Term::any_to_msgpack(w_, x_.x);
  Type::any_to_msgpack(w_, x_.rtn);
}

Math::Cbrt cbrt_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected Math::Cbrt with 2 field(s)");
  auto x = Term::any_from_msgpack(r_);
  auto rtn = Type::any_from_msgpack(r_);
  return {std::move(x), std::move(rtn)};
}

void cbrt_fields_to_msgpack(MsgpackWriter &w_, const Math::Cbrt &x_) {
  Term::any_to_msgpack(w_, x_.x);
  Type::any_to_msgpack(w_, x_.rtn);
}

Math::Exp exp_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected Math::Exp with 2 field(s)");
  auto x = Term::any_from_msgpack(r_);
  auto rtn = Type::any_from_msgpack(r_);
  return {std::move(x), std::move(rtn)};
}

void exp_fields_to_msgpack(MsgpackWriter &w_, const Math::Exp &x_) {
  Term::any_to_msgpack(w_, x_.x);
  Type::any_to_msgpack(w_, x_.rtn);
}

Math::Expm1 expm1_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected Math::Expm1 with 2 field(s)");
  auto x = Term::any_from_msgpack(r_);
  auto rtn = Type::any_from_msgpack(r_);
  return {std::move(x), std::move(rtn)};
}

void expm1_fields_to_msgpack(MsgpackWriter &w_, const Math::Expm1 &x_) {
  Term::any_to_msgpack(w_, x_.x);
  Type::any_to_msgpack(w_, x_.rtn);
}

Math::Log log_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected Math::Log with 2 field(s)");
  auto x = Term::any_from_msgpack(r_);
  auto rtn = Type::any_from_msgpack(r_);
  return {std::move(x), std::move(rtn)};
}

void log_fields_to_msgpack(MsgpackWriter &w_, const Math::Log &x_) {
  Term::any_to_msgpack(w_, x_.x);
  Type::any_to_msgpack(w_, x_.rtn);
}

Math::Log1p log1p_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected Math::Log1p with 2 field(s)");
  auto x = Term::any_from_msgpack(r_);
  auto rtn = Type::any_from_msgpack(r_);
  return {std::move(x), std::move(rtn)};
}

void log1p_fields_to_msgpack(MsgpackWriter &w_, const Math::Log1p &x_) {
  Term::any_to_msgpack(w_, x_.x);
  Type::any_to_msgpack(w_, x_.rtn);
}

Math::Log10 log10_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected Math::Log10 with 2 field(s)");
  auto x = Term::any_from_msgpack(r_);
  auto rtn = Type::any_from_msgpack(r_);
  return {std::move(x), std::move(rtn)};
}

void log10_fields_to_msgpack(MsgpackWriter &w_, const Math::Log10 &x_) {
  Term::any_to_msgpack(w_, x_.x);
  Type::any_to_msgpack(w_, x_.rtn);
}

Math::Pow pow_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 3) throw std::runtime_error("Expected Math::Pow with 3 field(s)");
  auto x = Term::any_from_msgpack(r_);
  auto y = Term::any_from_msgpack(r_);
  auto rtn = Type::any_from_msgpack(r_);
  return {std::move(x), std::move(y), std::move(rtn)};
}

void pow_fields_to_msgpack(MsgpackWriter &w_, const Math::Pow &x_) {
  Term::any_to_msgpack(w_, x_.x);
  Term::any_to_msgpack(w_, x_.y);
  Type::any_to_msgpack(w_, x_.rtn);
}

Math::Atan2 atan2_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 3) throw std::runtime_error("Expected Math::Atan2 with 3 field(s)");
  auto x = Term::any_from_msgpack(r_);
  auto y = Term::any_from_msgpack(r_);
  auto rtn = Type::any_from_msgpack(r_);
  return {std::move(x), std::move(y), std::move(rtn)};
}

void atan2_fields_to_msgpack(MsgpackWriter &w_, const Math::Atan2 &x_) {
  Term::any_to_msgpack(w_, x_.x);
  Term::any_to_msgpack(w_, x_.y);
  Type::any_to_msgpack(w_, x_.rtn);
}

Math::Hypot hypot_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 3) throw std::runtime_error("Expected Math::Hypot with 3 field(s)");
  auto x = Term::any_from_msgpack(r_);
  auto y = Term::any_from_msgpack(r_);
  auto rtn = Type::any_from_msgpack(r_);
  return {std::move(x), std::move(y), std::move(rtn)};
}

void hypot_fields_to_msgpack(MsgpackWriter &w_, const Math::Hypot &x_) {
  Term::any_to_msgpack(w_, x_.x);
  Term::any_to_msgpack(w_, x_.y);
  Type::any_to_msgpack(w_, x_.rtn);
}

Math::Any any_from_msgpack(MsgpackReader &r_) {
  if (r_.nextIsArray()) {
    const auto n_ = r_.readArrayHeader();
    if (n_ == 0) throw std::runtime_error("Expected non-empty sum payload");
    const auto ord_ = r_.readInt32();
    switch (ord_) {
      case 0: return Math::abs_fields_from_msgpack(r_, n_ - 1);
      case 1: return Math::sin_fields_from_msgpack(r_, n_ - 1);
      case 2: return Math::cos_fields_from_msgpack(r_, n_ - 1);
      case 3: return Math::tan_fields_from_msgpack(r_, n_ - 1);
      case 4: return Math::asin_fields_from_msgpack(r_, n_ - 1);
      case 5: return Math::acos_fields_from_msgpack(r_, n_ - 1);
      case 6: return Math::atan_fields_from_msgpack(r_, n_ - 1);
      case 7: return Math::sinh_fields_from_msgpack(r_, n_ - 1);
      case 8: return Math::cosh_fields_from_msgpack(r_, n_ - 1);
      case 9: return Math::tanh_fields_from_msgpack(r_, n_ - 1);
      case 10: return Math::signum_fields_from_msgpack(r_, n_ - 1);
      case 11: return Math::round_fields_from_msgpack(r_, n_ - 1);
      case 12: return Math::ceil_fields_from_msgpack(r_, n_ - 1);
      case 13: return Math::floor_fields_from_msgpack(r_, n_ - 1);
      case 14: return Math::rint_fields_from_msgpack(r_, n_ - 1);
      case 15: return Math::sqrt_fields_from_msgpack(r_, n_ - 1);
      case 16: return Math::cbrt_fields_from_msgpack(r_, n_ - 1);
      case 17: return Math::exp_fields_from_msgpack(r_, n_ - 1);
      case 18: return Math::expm1_fields_from_msgpack(r_, n_ - 1);
      case 19: return Math::log_fields_from_msgpack(r_, n_ - 1);
      case 20: return Math::log1p_fields_from_msgpack(r_, n_ - 1);
      case 21: return Math::log10_fields_from_msgpack(r_, n_ - 1);
      case 22: return Math::pow_fields_from_msgpack(r_, n_ - 1);
      case 23: return Math::atan2_fields_from_msgpack(r_, n_ - 1);
      case 24: return Math::hypot_fields_from_msgpack(r_, n_ - 1);
      default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
    }
  } else {
    const auto ord_ = r_.readInt32();
    switch (ord_) {
      case 0:
      case 1:
      case 2:
      case 3:
      case 4:
      case 5:
      case 6:
      case 7:
      case 8:
      case 9:
      case 10:
      case 11:
      case 12:
      case 13:
      case 14:
      case 15:
      case 16:
      case 17:
      case 18:
      case 19:
      case 20:
      case 21:
      case 22:
      case 23:
      case 24: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
    }
  }
}

void any_to_msgpack(MsgpackWriter &w_, const Math::Any &x_) {
  x_.match_total(
      [&](const Math::Abs &y_) -> void {
        w_.writeArrayHeader(3);
        w_.writeInt32(0);
        Math::abs_fields_to_msgpack(w_, y_);
      },
      [&](const Math::Sin &y_) -> void {
        w_.writeArrayHeader(3);
        w_.writeInt32(1);
        Math::sin_fields_to_msgpack(w_, y_);
      },
      [&](const Math::Cos &y_) -> void {
        w_.writeArrayHeader(3);
        w_.writeInt32(2);
        Math::cos_fields_to_msgpack(w_, y_);
      },
      [&](const Math::Tan &y_) -> void {
        w_.writeArrayHeader(3);
        w_.writeInt32(3);
        Math::tan_fields_to_msgpack(w_, y_);
      },
      [&](const Math::Asin &y_) -> void {
        w_.writeArrayHeader(3);
        w_.writeInt32(4);
        Math::asin_fields_to_msgpack(w_, y_);
      },
      [&](const Math::Acos &y_) -> void {
        w_.writeArrayHeader(3);
        w_.writeInt32(5);
        Math::acos_fields_to_msgpack(w_, y_);
      },
      [&](const Math::Atan &y_) -> void {
        w_.writeArrayHeader(3);
        w_.writeInt32(6);
        Math::atan_fields_to_msgpack(w_, y_);
      },
      [&](const Math::Sinh &y_) -> void {
        w_.writeArrayHeader(3);
        w_.writeInt32(7);
        Math::sinh_fields_to_msgpack(w_, y_);
      },
      [&](const Math::Cosh &y_) -> void {
        w_.writeArrayHeader(3);
        w_.writeInt32(8);
        Math::cosh_fields_to_msgpack(w_, y_);
      },
      [&](const Math::Tanh &y_) -> void {
        w_.writeArrayHeader(3);
        w_.writeInt32(9);
        Math::tanh_fields_to_msgpack(w_, y_);
      },
      [&](const Math::Signum &y_) -> void {
        w_.writeArrayHeader(3);
        w_.writeInt32(10);
        Math::signum_fields_to_msgpack(w_, y_);
      },
      [&](const Math::Round &y_) -> void {
        w_.writeArrayHeader(3);
        w_.writeInt32(11);
        Math::round_fields_to_msgpack(w_, y_);
      },
      [&](const Math::Ceil &y_) -> void {
        w_.writeArrayHeader(3);
        w_.writeInt32(12);
        Math::ceil_fields_to_msgpack(w_, y_);
      },
      [&](const Math::Floor &y_) -> void {
        w_.writeArrayHeader(3);
        w_.writeInt32(13);
        Math::floor_fields_to_msgpack(w_, y_);
      },
      [&](const Math::Rint &y_) -> void {
        w_.writeArrayHeader(3);
        w_.writeInt32(14);
        Math::rint_fields_to_msgpack(w_, y_);
      },
      [&](const Math::Sqrt &y_) -> void {
        w_.writeArrayHeader(3);
        w_.writeInt32(15);
        Math::sqrt_fields_to_msgpack(w_, y_);
      },
      [&](const Math::Cbrt &y_) -> void {
        w_.writeArrayHeader(3);
        w_.writeInt32(16);
        Math::cbrt_fields_to_msgpack(w_, y_);
      },
      [&](const Math::Exp &y_) -> void {
        w_.writeArrayHeader(3);
        w_.writeInt32(17);
        Math::exp_fields_to_msgpack(w_, y_);
      },
      [&](const Math::Expm1 &y_) -> void {
        w_.writeArrayHeader(3);
        w_.writeInt32(18);
        Math::expm1_fields_to_msgpack(w_, y_);
      },
      [&](const Math::Log &y_) -> void {
        w_.writeArrayHeader(3);
        w_.writeInt32(19);
        Math::log_fields_to_msgpack(w_, y_);
      },
      [&](const Math::Log1p &y_) -> void {
        w_.writeArrayHeader(3);
        w_.writeInt32(20);
        Math::log1p_fields_to_msgpack(w_, y_);
      },
      [&](const Math::Log10 &y_) -> void {
        w_.writeArrayHeader(3);
        w_.writeInt32(21);
        Math::log10_fields_to_msgpack(w_, y_);
      },
      [&](const Math::Pow &y_) -> void {
        w_.writeArrayHeader(4);
        w_.writeInt32(22);
        Math::pow_fields_to_msgpack(w_, y_);
      },
      [&](const Math::Atan2 &y_) -> void {
        w_.writeArrayHeader(4);
        w_.writeInt32(23);
        Math::atan2_fields_to_msgpack(w_, y_);
      },
      [&](const Math::Hypot &y_) -> void {
        w_.writeArrayHeader(4);
        w_.writeInt32(24);
        Math::hypot_fields_to_msgpack(w_, y_);
      });
}
}} // namespace Math
namespace CallConvention { namespace {

CallConvention::RegularCall regularcall_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected CallConvention::RegularCall with 0 field(s)");
  return {};
}

CallConvention::OffloadEntry offloadentry_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected CallConvention::OffloadEntry with 0 field(s)");
  return {};
}

CallConvention::Any any_from_msgpack(MsgpackReader &r_) {
  if (r_.nextIsArray()) {
    const auto n_ = r_.readArrayHeader();
    if (n_ == 0) throw std::runtime_error("Expected non-empty sum payload");
    const auto ord_ = r_.readInt32();
    switch (ord_) {
      case 0: return CallConvention::regularcall_fields_from_msgpack(r_, n_ - 1);
      case 1: return CallConvention::offloadentry_fields_from_msgpack(r_, n_ - 1);
      default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
    }
  } else {
    const auto ord_ = r_.readInt32();
    switch (ord_) {
      case 0: return CallConvention::regularcall_fields_from_msgpack(r_, 0);
      case 1: return CallConvention::offloadentry_fields_from_msgpack(r_, 0);
      default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
    }
  }
}

void any_to_msgpack(MsgpackWriter &w_, const CallConvention::Any &x_) {
  x_.match_total([&](const CallConvention::RegularCall &y_) -> void { w_.writeInt32(0); },
                 [&](const CallConvention::OffloadEntry &y_) -> void { w_.writeInt32(1); });
}
}} // namespace CallConvention
namespace MemOrder { namespace {

MemOrder::Relaxed relaxed_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected MemOrder::Relaxed with 0 field(s)");
  return {};
}

MemOrder::Acquire acquire_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected MemOrder::Acquire with 0 field(s)");
  return {};
}

MemOrder::Release release_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected MemOrder::Release with 0 field(s)");
  return {};
}

MemOrder::AcqRel acqrel_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected MemOrder::AcqRel with 0 field(s)");
  return {};
}

MemOrder::SeqCst seqcst_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected MemOrder::SeqCst with 0 field(s)");
  return {};
}

MemOrder::Any any_from_msgpack(MsgpackReader &r_) {
  if (r_.nextIsArray()) {
    const auto n_ = r_.readArrayHeader();
    if (n_ == 0) throw std::runtime_error("Expected non-empty sum payload");
    const auto ord_ = r_.readInt32();
    switch (ord_) {
      case 0: return MemOrder::relaxed_fields_from_msgpack(r_, n_ - 1);
      case 1: return MemOrder::acquire_fields_from_msgpack(r_, n_ - 1);
      case 2: return MemOrder::release_fields_from_msgpack(r_, n_ - 1);
      case 3: return MemOrder::acqrel_fields_from_msgpack(r_, n_ - 1);
      case 4: return MemOrder::seqcst_fields_from_msgpack(r_, n_ - 1);
      default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
    }
  } else {
    const auto ord_ = r_.readInt32();
    switch (ord_) {
      case 0: return MemOrder::relaxed_fields_from_msgpack(r_, 0);
      case 1: return MemOrder::acquire_fields_from_msgpack(r_, 0);
      case 2: return MemOrder::release_fields_from_msgpack(r_, 0);
      case 3: return MemOrder::acqrel_fields_from_msgpack(r_, 0);
      case 4: return MemOrder::seqcst_fields_from_msgpack(r_, 0);
      default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
    }
  }
}

void any_to_msgpack(MsgpackWriter &w_, const MemOrder::Any &x_) {
  x_.match_total(
      [&](const MemOrder::Relaxed &y_) -> void { w_.writeInt32(0); }, [&](const MemOrder::Acquire &y_) -> void { w_.writeInt32(1); },
      [&](const MemOrder::Release &y_) -> void { w_.writeInt32(2); }, [&](const MemOrder::AcqRel &y_) -> void { w_.writeInt32(3); },
      [&](const MemOrder::SeqCst &y_) -> void { w_.writeInt32(4); });
}
}} // namespace MemOrder
namespace TypeSpace { namespace {

TypeSpace::Global global_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected TypeSpace::Global with 0 field(s)");
  return {};
}

TypeSpace::Local local_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected TypeSpace::Local with 0 field(s)");
  return {};
}

TypeSpace::Private private_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected TypeSpace::Private with 0 field(s)");
  return {};
}

TypeSpace::Constant constant_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected TypeSpace::Constant with 0 field(s)");
  return {};
}

TypeSpace::Any any_from_msgpack(MsgpackReader &r_) {
  if (r_.nextIsArray()) {
    const auto n_ = r_.readArrayHeader();
    if (n_ == 0) throw std::runtime_error("Expected non-empty sum payload");
    const auto ord_ = r_.readInt32();
    switch (ord_) {
      case 0: return TypeSpace::global_fields_from_msgpack(r_, n_ - 1);
      case 1: return TypeSpace::local_fields_from_msgpack(r_, n_ - 1);
      case 2: return TypeSpace::private_fields_from_msgpack(r_, n_ - 1);
      case 3: return TypeSpace::constant_fields_from_msgpack(r_, n_ - 1);
      default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
    }
  } else {
    const auto ord_ = r_.readInt32();
    switch (ord_) {
      case 0: return TypeSpace::global_fields_from_msgpack(r_, 0);
      case 1: return TypeSpace::local_fields_from_msgpack(r_, 0);
      case 2: return TypeSpace::private_fields_from_msgpack(r_, 0);
      case 3: return TypeSpace::constant_fields_from_msgpack(r_, 0);
      default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
    }
  }
}

void any_to_msgpack(MsgpackWriter &w_, const TypeSpace::Any &x_) {
  x_.match_total(
      [&](const TypeSpace::Global &y_) -> void { w_.writeInt32(0); }, [&](const TypeSpace::Local &y_) -> void { w_.writeInt32(1); },
      [&](const TypeSpace::Private &y_) -> void { w_.writeInt32(2); }, [&](const TypeSpace::Constant &y_) -> void { w_.writeInt32(3); });
}
}} // namespace TypeSpace
namespace PassPhase { namespace {

PassPhase::Initial initial_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected PassPhase::Initial with 0 field(s)");
  return {};
}

PassPhase::PostMono postmono_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected PassPhase::PostMono with 0 field(s)");
  return {};
}

PassPhase::Any any_from_msgpack(MsgpackReader &r_) {
  if (r_.nextIsArray()) {
    const auto n_ = r_.readArrayHeader();
    if (n_ == 0) throw std::runtime_error("Expected non-empty sum payload");
    const auto ord_ = r_.readInt32();
    switch (ord_) {
      case 0: return PassPhase::initial_fields_from_msgpack(r_, n_ - 1);
      case 1: return PassPhase::postmono_fields_from_msgpack(r_, n_ - 1);
      default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
    }
  } else {
    const auto ord_ = r_.readInt32();
    switch (ord_) {
      case 0: return PassPhase::initial_fields_from_msgpack(r_, 0);
      case 1: return PassPhase::postmono_fields_from_msgpack(r_, 0);
      default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
    }
  }
}

void any_to_msgpack(MsgpackWriter &w_, const PassPhase::Any &x_) {
  x_.match_total([&](const PassPhase::Initial &y_) -> void { w_.writeInt32(0); },
                 [&](const PassPhase::PostMono &y_) -> void { w_.writeInt32(1); });
}
}} // namespace PassPhase
namespace Spec { namespace {

Spec::Assert assert_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected Spec::Assert with 2 field(s)");
  auto code = Term::any_from_msgpack(r_);
  auto message = Term::any_from_msgpack(r_);
  return {std::move(code), std::move(message)};
}

void assert_fields_to_msgpack(MsgpackWriter &w_, const Spec::Assert &x_) {
  Term::any_to_msgpack(w_, x_.code);
  Term::any_to_msgpack(w_, x_.message);
}

Spec::GpuBarrierGlobal gpubarrierglobal_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected Spec::GpuBarrierGlobal with 0 field(s)");
  return {};
}

Spec::GpuBarrierLocal gpubarrierlocal_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected Spec::GpuBarrierLocal with 0 field(s)");
  return {};
}

Spec::GpuBarrierAll gpubarrierall_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected Spec::GpuBarrierAll with 0 field(s)");
  return {};
}

Spec::GpuFenceGlobal gpufenceglobal_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected Spec::GpuFenceGlobal with 0 field(s)");
  return {};
}

Spec::GpuFenceLocal gpufencelocal_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected Spec::GpuFenceLocal with 0 field(s)");
  return {};
}

Spec::GpuFenceAll gpufenceall_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected Spec::GpuFenceAll with 0 field(s)");
  return {};
}

Spec::GpuGlobalIdx gpuglobalidx_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 1) throw std::runtime_error("Expected Spec::GpuGlobalIdx with 1 field(s)");
  auto dim = Term::any_from_msgpack(r_);
  return Spec::GpuGlobalIdx(std::move(dim));
}

void gpuglobalidx_fields_to_msgpack(MsgpackWriter &w_, const Spec::GpuGlobalIdx &x_) { Term::any_to_msgpack(w_, x_.dim); }

Spec::GpuGlobalSize gpuglobalsize_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 1) throw std::runtime_error("Expected Spec::GpuGlobalSize with 1 field(s)");
  auto dim = Term::any_from_msgpack(r_);
  return Spec::GpuGlobalSize(std::move(dim));
}

void gpuglobalsize_fields_to_msgpack(MsgpackWriter &w_, const Spec::GpuGlobalSize &x_) { Term::any_to_msgpack(w_, x_.dim); }

Spec::GpuGroupIdx gpugroupidx_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 1) throw std::runtime_error("Expected Spec::GpuGroupIdx with 1 field(s)");
  auto dim = Term::any_from_msgpack(r_);
  return Spec::GpuGroupIdx(std::move(dim));
}

void gpugroupidx_fields_to_msgpack(MsgpackWriter &w_, const Spec::GpuGroupIdx &x_) { Term::any_to_msgpack(w_, x_.dim); }

Spec::GpuGroupSize gpugroupsize_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 1) throw std::runtime_error("Expected Spec::GpuGroupSize with 1 field(s)");
  auto dim = Term::any_from_msgpack(r_);
  return Spec::GpuGroupSize(std::move(dim));
}

void gpugroupsize_fields_to_msgpack(MsgpackWriter &w_, const Spec::GpuGroupSize &x_) { Term::any_to_msgpack(w_, x_.dim); }

Spec::GpuLocalIdx gpulocalidx_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 1) throw std::runtime_error("Expected Spec::GpuLocalIdx with 1 field(s)");
  auto dim = Term::any_from_msgpack(r_);
  return Spec::GpuLocalIdx(std::move(dim));
}

void gpulocalidx_fields_to_msgpack(MsgpackWriter &w_, const Spec::GpuLocalIdx &x_) { Term::any_to_msgpack(w_, x_.dim); }

Spec::GpuLocalSize gpulocalsize_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 1) throw std::runtime_error("Expected Spec::GpuLocalSize with 1 field(s)");
  auto dim = Term::any_from_msgpack(r_);
  return Spec::GpuLocalSize(std::move(dim));
}

void gpulocalsize_fields_to_msgpack(MsgpackWriter &w_, const Spec::GpuLocalSize &x_) { Term::any_to_msgpack(w_, x_.dim); }

Spec::GpuLaneIdx gpulaneidx_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected Spec::GpuLaneIdx with 0 field(s)");
  return {};
}

Spec::GpuSubgroupSize gpusubgroupsize_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected Spec::GpuSubgroupSize with 0 field(s)");
  return {};
}

Spec::GpuShuffleDown gpushuffledown_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 5) throw std::runtime_error("Expected Spec::GpuShuffleDown with 5 field(s)");
  auto value = Term::any_from_msgpack(r_);
  auto delta = Term::any_from_msgpack(r_);
  auto width = Term::any_from_msgpack(r_);
  auto mask = Term::any_from_msgpack(r_);
  auto rtn = Type::any_from_msgpack(r_);
  return {std::move(value), std::move(delta), std::move(width), std::move(mask), std::move(rtn)};
}

void gpushuffledown_fields_to_msgpack(MsgpackWriter &w_, const Spec::GpuShuffleDown &x_) {
  Term::any_to_msgpack(w_, x_.value);
  Term::any_to_msgpack(w_, x_.delta);
  Term::any_to_msgpack(w_, x_.width);
  Term::any_to_msgpack(w_, x_.mask);
  Type::any_to_msgpack(w_, x_.rtn);
}

Spec::GpuShuffleUp gpushuffleup_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 5) throw std::runtime_error("Expected Spec::GpuShuffleUp with 5 field(s)");
  auto value = Term::any_from_msgpack(r_);
  auto delta = Term::any_from_msgpack(r_);
  auto width = Term::any_from_msgpack(r_);
  auto mask = Term::any_from_msgpack(r_);
  auto rtn = Type::any_from_msgpack(r_);
  return {std::move(value), std::move(delta), std::move(width), std::move(mask), std::move(rtn)};
}

void gpushuffleup_fields_to_msgpack(MsgpackWriter &w_, const Spec::GpuShuffleUp &x_) {
  Term::any_to_msgpack(w_, x_.value);
  Term::any_to_msgpack(w_, x_.delta);
  Term::any_to_msgpack(w_, x_.width);
  Term::any_to_msgpack(w_, x_.mask);
  Type::any_to_msgpack(w_, x_.rtn);
}

Spec::GpuShuffleIdx gpushuffleidx_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 5) throw std::runtime_error("Expected Spec::GpuShuffleIdx with 5 field(s)");
  auto value = Term::any_from_msgpack(r_);
  auto srcLane = Term::any_from_msgpack(r_);
  auto width = Term::any_from_msgpack(r_);
  auto mask = Term::any_from_msgpack(r_);
  auto rtn = Type::any_from_msgpack(r_);
  return {std::move(value), std::move(srcLane), std::move(width), std::move(mask), std::move(rtn)};
}

void gpushuffleidx_fields_to_msgpack(MsgpackWriter &w_, const Spec::GpuShuffleIdx &x_) {
  Term::any_to_msgpack(w_, x_.value);
  Term::any_to_msgpack(w_, x_.srcLane);
  Term::any_to_msgpack(w_, x_.width);
  Term::any_to_msgpack(w_, x_.mask);
  Type::any_to_msgpack(w_, x_.rtn);
}

Spec::GpuShuffleXor gpushufflexor_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 5) throw std::runtime_error("Expected Spec::GpuShuffleXor with 5 field(s)");
  auto value = Term::any_from_msgpack(r_);
  auto laneMask = Term::any_from_msgpack(r_);
  auto width = Term::any_from_msgpack(r_);
  auto mask = Term::any_from_msgpack(r_);
  auto rtn = Type::any_from_msgpack(r_);
  return {std::move(value), std::move(laneMask), std::move(width), std::move(mask), std::move(rtn)};
}

void gpushufflexor_fields_to_msgpack(MsgpackWriter &w_, const Spec::GpuShuffleXor &x_) {
  Term::any_to_msgpack(w_, x_.value);
  Term::any_to_msgpack(w_, x_.laneMask);
  Term::any_to_msgpack(w_, x_.width);
  Term::any_to_msgpack(w_, x_.mask);
  Type::any_to_msgpack(w_, x_.rtn);
}

Spec::GpuSubgroupBarrier gpusubgroupbarrier_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 1) throw std::runtime_error("Expected Spec::GpuSubgroupBarrier with 1 field(s)");
  auto mask = Term::any_from_msgpack(r_);
  return Spec::GpuSubgroupBarrier(std::move(mask));
}

void gpusubgroupbarrier_fields_to_msgpack(MsgpackWriter &w_, const Spec::GpuSubgroupBarrier &x_) { Term::any_to_msgpack(w_, x_.mask); }

Spec::GpuBallot gpuballot_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected Spec::GpuBallot with 2 field(s)");
  auto mask = Term::any_from_msgpack(r_);
  auto pred = Term::any_from_msgpack(r_);
  return {std::move(mask), std::move(pred)};
}

void gpuballot_fields_to_msgpack(MsgpackWriter &w_, const Spec::GpuBallot &x_) {
  Term::any_to_msgpack(w_, x_.mask);
  Term::any_to_msgpack(w_, x_.pred);
}

Spec::GpuVoteAny gpuvoteany_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected Spec::GpuVoteAny with 2 field(s)");
  auto mask = Term::any_from_msgpack(r_);
  auto pred = Term::any_from_msgpack(r_);
  return {std::move(mask), std::move(pred)};
}

void gpuvoteany_fields_to_msgpack(MsgpackWriter &w_, const Spec::GpuVoteAny &x_) {
  Term::any_to_msgpack(w_, x_.mask);
  Term::any_to_msgpack(w_, x_.pred);
}

Spec::GpuVoteAll gpuvoteall_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected Spec::GpuVoteAll with 2 field(s)");
  auto mask = Term::any_from_msgpack(r_);
  auto pred = Term::any_from_msgpack(r_);
  return {std::move(mask), std::move(pred)};
}

void gpuvoteall_fields_to_msgpack(MsgpackWriter &w_, const Spec::GpuVoteAll &x_) {
  Term::any_to_msgpack(w_, x_.mask);
  Term::any_to_msgpack(w_, x_.pred);
}

Spec::GpuAtomicRMW gpuatomicrmw_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 6) throw std::runtime_error("Expected Spec::GpuAtomicRMW with 6 field(s)");
  auto op = AtomicOp::any_from_msgpack(r_);
  auto ptr = Term::any_from_msgpack(r_);
  auto value = Term::any_from_msgpack(r_);
  auto scope = MemScope::any_from_msgpack(r_);
  auto order = MemOrder::any_from_msgpack(r_);
  auto rtn = Type::any_from_msgpack(r_);
  return {std::move(op), std::move(ptr), std::move(value), std::move(scope), std::move(order), std::move(rtn)};
}

void gpuatomicrmw_fields_to_msgpack(MsgpackWriter &w_, const Spec::GpuAtomicRMW &x_) {
  AtomicOp::any_to_msgpack(w_, x_.op);
  Term::any_to_msgpack(w_, x_.ptr);
  Term::any_to_msgpack(w_, x_.value);
  MemScope::any_to_msgpack(w_, x_.scope);
  MemOrder::any_to_msgpack(w_, x_.order);
  Type::any_to_msgpack(w_, x_.rtn);
}

Spec::GpuAtomicCAS gpuatomiccas_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 6) throw std::runtime_error("Expected Spec::GpuAtomicCAS with 6 field(s)");
  auto ptr = Term::any_from_msgpack(r_);
  auto expected = Term::any_from_msgpack(r_);
  auto desired = Term::any_from_msgpack(r_);
  auto scope = MemScope::any_from_msgpack(r_);
  auto order = MemOrder::any_from_msgpack(r_);
  auto rtn = Type::any_from_msgpack(r_);
  return {std::move(ptr), std::move(expected), std::move(desired), std::move(scope), std::move(order), std::move(rtn)};
}

void gpuatomiccas_fields_to_msgpack(MsgpackWriter &w_, const Spec::GpuAtomicCAS &x_) {
  Term::any_to_msgpack(w_, x_.ptr);
  Term::any_to_msgpack(w_, x_.expected);
  Term::any_to_msgpack(w_, x_.desired);
  MemScope::any_to_msgpack(w_, x_.scope);
  MemOrder::any_to_msgpack(w_, x_.order);
  Type::any_to_msgpack(w_, x_.rtn);
}

Spec::GpuGroupReduce gpugroupreduce_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 3) throw std::runtime_error("Expected Spec::GpuGroupReduce with 3 field(s)");
  auto op = AtomicOp::any_from_msgpack(r_);
  auto value = Term::any_from_msgpack(r_);
  auto rtn = Type::any_from_msgpack(r_);
  return {std::move(op), std::move(value), std::move(rtn)};
}

void gpugroupreduce_fields_to_msgpack(MsgpackWriter &w_, const Spec::GpuGroupReduce &x_) {
  AtomicOp::any_to_msgpack(w_, x_.op);
  Term::any_to_msgpack(w_, x_.value);
  Type::any_to_msgpack(w_, x_.rtn);
}

Spec::GpuGroupInclusiveScan gpugroupinclusivescan_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 3) throw std::runtime_error("Expected Spec::GpuGroupInclusiveScan with 3 field(s)");
  auto op = AtomicOp::any_from_msgpack(r_);
  auto value = Term::any_from_msgpack(r_);
  auto rtn = Type::any_from_msgpack(r_);
  return {std::move(op), std::move(value), std::move(rtn)};
}

void gpugroupinclusivescan_fields_to_msgpack(MsgpackWriter &w_, const Spec::GpuGroupInclusiveScan &x_) {
  AtomicOp::any_to_msgpack(w_, x_.op);
  Term::any_to_msgpack(w_, x_.value);
  Type::any_to_msgpack(w_, x_.rtn);
}

Spec::GpuGroupExclusiveScan gpugroupexclusivescan_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 3) throw std::runtime_error("Expected Spec::GpuGroupExclusiveScan with 3 field(s)");
  auto op = AtomicOp::any_from_msgpack(r_);
  auto value = Term::any_from_msgpack(r_);
  auto rtn = Type::any_from_msgpack(r_);
  return {std::move(op), std::move(value), std::move(rtn)};
}

void gpugroupexclusivescan_fields_to_msgpack(MsgpackWriter &w_, const Spec::GpuGroupExclusiveScan &x_) {
  AtomicOp::any_to_msgpack(w_, x_.op);
  Term::any_to_msgpack(w_, x_.value);
  Type::any_to_msgpack(w_, x_.rtn);
}

Spec::GpuVolatileLoad gpuvolatileload_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected Spec::GpuVolatileLoad with 2 field(s)");
  auto ptr = Term::any_from_msgpack(r_);
  auto rtn = Type::any_from_msgpack(r_);
  return {std::move(ptr), std::move(rtn)};
}

void gpuvolatileload_fields_to_msgpack(MsgpackWriter &w_, const Spec::GpuVolatileLoad &x_) {
  Term::any_to_msgpack(w_, x_.ptr);
  Type::any_to_msgpack(w_, x_.rtn);
}

Spec::GpuVolatileStore gpuvolatilestore_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected Spec::GpuVolatileStore with 2 field(s)");
  auto ptr = Term::any_from_msgpack(r_);
  auto value = Term::any_from_msgpack(r_);
  return {std::move(ptr), std::move(value)};
}

void gpuvolatilestore_fields_to_msgpack(MsgpackWriter &w_, const Spec::GpuVolatileStore &x_) {
  Term::any_to_msgpack(w_, x_.ptr);
  Term::any_to_msgpack(w_, x_.value);
}

Spec::RemoteLaunch remotelaunch_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 11) throw std::runtime_error("Expected Spec::RemoteLaunch with 11 field(s)");
  auto context = Term::any_from_msgpack(r_);
  auto kernel = Term::any_from_msgpack(r_);
  std::vector<Type::Any> tpeArgs;
  {
    const auto tpeArgs_size = r_.readArrayHeader();
    tpeArgs.reserve(tpeArgs_size);
    for (size_t tpeArgs_idx = 0; tpeArgs_idx < tpeArgs_size; ++tpeArgs_idx) {
      auto tpeArgs_elem = Type::any_from_msgpack(r_);
      tpeArgs.emplace_back(std::move(tpeArgs_elem));
    }
  }
  auto gridX = Term::any_from_msgpack(r_);
  auto gridY = Term::any_from_msgpack(r_);
  auto gridZ = Term::any_from_msgpack(r_);
  auto blockX = Term::any_from_msgpack(r_);
  auto blockY = Term::any_from_msgpack(r_);
  auto blockZ = Term::any_from_msgpack(r_);
  auto shmem = Term::any_from_msgpack(r_);
  std::vector<Term::Any> args;
  {
    const auto args_size = r_.readArrayHeader();
    args.reserve(args_size);
    for (size_t args_idx = 0; args_idx < args_size; ++args_idx) {
      auto args_elem = Term::any_from_msgpack(r_);
      args.emplace_back(std::move(args_elem));
    }
  }
  return {std::move(context), std::move(kernel), std::move(tpeArgs), std::move(gridX), std::move(gridY), std::move(gridZ),
          std::move(blockX),  std::move(blockY), std::move(blockZ),  std::move(shmem), std::move(args)};
}

void remotelaunch_fields_to_msgpack(MsgpackWriter &w_, const Spec::RemoteLaunch &x_) {
  Term::any_to_msgpack(w_, x_.context);
  Term::any_to_msgpack(w_, x_.kernel);
  w_.writeArrayHeader(x_.tpeArgs.size());
  for (const auto &v0_ : x_.tpeArgs) {
    Type::any_to_msgpack(w_, v0_);
  }
  Term::any_to_msgpack(w_, x_.gridX);
  Term::any_to_msgpack(w_, x_.gridY);
  Term::any_to_msgpack(w_, x_.gridZ);
  Term::any_to_msgpack(w_, x_.blockX);
  Term::any_to_msgpack(w_, x_.blockY);
  Term::any_to_msgpack(w_, x_.blockZ);
  Term::any_to_msgpack(w_, x_.shmem);
  w_.writeArrayHeader(x_.args.size());
  for (const auto &v0_ : x_.args) {
    Term::any_to_msgpack(w_, v0_);
  }
}

Spec::RemoteAlloc remotealloc_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected Spec::RemoteAlloc with 2 field(s)");
  auto context = Term::any_from_msgpack(r_);
  auto bytes = Term::any_from_msgpack(r_);
  return {std::move(context), std::move(bytes)};
}

void remotealloc_fields_to_msgpack(MsgpackWriter &w_, const Spec::RemoteAlloc &x_) {
  Term::any_to_msgpack(w_, x_.context);
  Term::any_to_msgpack(w_, x_.bytes);
}

Spec::RemoteTempAlloc remotetempalloc_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected Spec::RemoteTempAlloc with 2 field(s)");
  auto context = Term::any_from_msgpack(r_);
  auto bytes = Term::any_from_msgpack(r_);
  return {std::move(context), std::move(bytes)};
}

void remotetempalloc_fields_to_msgpack(MsgpackWriter &w_, const Spec::RemoteTempAlloc &x_) {
  Term::any_to_msgpack(w_, x_.context);
  Term::any_to_msgpack(w_, x_.bytes);
}

Spec::RemoteFree remotefree_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected Spec::RemoteFree with 2 field(s)");
  auto context = Term::any_from_msgpack(r_);
  auto ptr = Term::any_from_msgpack(r_);
  return {std::move(context), std::move(ptr)};
}

void remotefree_fields_to_msgpack(MsgpackWriter &w_, const Spec::RemoteFree &x_) {
  Term::any_to_msgpack(w_, x_.context);
  Term::any_to_msgpack(w_, x_.ptr);
}

Spec::RemoteMemcpy remotememcpy_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 5) throw std::runtime_error("Expected Spec::RemoteMemcpy with 5 field(s)");
  auto context = Term::any_from_msgpack(r_);
  auto dst = Term::any_from_msgpack(r_);
  auto src = Term::any_from_msgpack(r_);
  auto bytes = Term::any_from_msgpack(r_);
  auto direction = Direction::any_from_msgpack(r_);
  return {std::move(context), std::move(dst), std::move(src), std::move(bytes), std::move(direction)};
}

void remotememcpy_fields_to_msgpack(MsgpackWriter &w_, const Spec::RemoteMemcpy &x_) {
  Term::any_to_msgpack(w_, x_.context);
  Term::any_to_msgpack(w_, x_.dst);
  Term::any_to_msgpack(w_, x_.src);
  Term::any_to_msgpack(w_, x_.bytes);
  Direction::any_to_msgpack(w_, x_.direction);
}

Spec::RemoteSync remotesync_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 1) throw std::runtime_error("Expected Spec::RemoteSync with 1 field(s)");
  auto context = Term::any_from_msgpack(r_);
  return Spec::RemoteSync(std::move(context));
}

void remotesync_fields_to_msgpack(MsgpackWriter &w_, const Spec::RemoteSync &x_) { Term::any_to_msgpack(w_, x_.context); }

Spec::Any any_from_msgpack(MsgpackReader &r_) {
  if (r_.nextIsArray()) {
    const auto n_ = r_.readArrayHeader();
    if (n_ == 0) throw std::runtime_error("Expected non-empty sum payload");
    const auto ord_ = r_.readInt32();
    switch (ord_) {
      case 0: return Spec::assert_fields_from_msgpack(r_, n_ - 1);
      case 1: return Spec::gpubarrierglobal_fields_from_msgpack(r_, n_ - 1);
      case 2: return Spec::gpubarrierlocal_fields_from_msgpack(r_, n_ - 1);
      case 3: return Spec::gpubarrierall_fields_from_msgpack(r_, n_ - 1);
      case 4: return Spec::gpufenceglobal_fields_from_msgpack(r_, n_ - 1);
      case 5: return Spec::gpufencelocal_fields_from_msgpack(r_, n_ - 1);
      case 6: return Spec::gpufenceall_fields_from_msgpack(r_, n_ - 1);
      case 7: return Spec::gpuglobalidx_fields_from_msgpack(r_, n_ - 1);
      case 8: return Spec::gpuglobalsize_fields_from_msgpack(r_, n_ - 1);
      case 9: return Spec::gpugroupidx_fields_from_msgpack(r_, n_ - 1);
      case 10: return Spec::gpugroupsize_fields_from_msgpack(r_, n_ - 1);
      case 11: return Spec::gpulocalidx_fields_from_msgpack(r_, n_ - 1);
      case 12: return Spec::gpulocalsize_fields_from_msgpack(r_, n_ - 1);
      case 13: return Spec::gpulaneidx_fields_from_msgpack(r_, n_ - 1);
      case 14: return Spec::gpusubgroupsize_fields_from_msgpack(r_, n_ - 1);
      case 15: return Spec::gpushuffledown_fields_from_msgpack(r_, n_ - 1);
      case 16: return Spec::gpushuffleup_fields_from_msgpack(r_, n_ - 1);
      case 17: return Spec::gpushuffleidx_fields_from_msgpack(r_, n_ - 1);
      case 18: return Spec::gpushufflexor_fields_from_msgpack(r_, n_ - 1);
      case 19: return Spec::gpusubgroupbarrier_fields_from_msgpack(r_, n_ - 1);
      case 20: return Spec::gpuballot_fields_from_msgpack(r_, n_ - 1);
      case 21: return Spec::gpuvoteany_fields_from_msgpack(r_, n_ - 1);
      case 22: return Spec::gpuvoteall_fields_from_msgpack(r_, n_ - 1);
      case 23: return Spec::gpuatomicrmw_fields_from_msgpack(r_, n_ - 1);
      case 24: return Spec::gpuatomiccas_fields_from_msgpack(r_, n_ - 1);
      case 25: return Spec::gpugroupreduce_fields_from_msgpack(r_, n_ - 1);
      case 26: return Spec::gpugroupinclusivescan_fields_from_msgpack(r_, n_ - 1);
      case 27: return Spec::gpugroupexclusivescan_fields_from_msgpack(r_, n_ - 1);
      case 28: return Spec::gpuvolatileload_fields_from_msgpack(r_, n_ - 1);
      case 29: return Spec::gpuvolatilestore_fields_from_msgpack(r_, n_ - 1);
      case 30: return Spec::remotelaunch_fields_from_msgpack(r_, n_ - 1);
      case 31: return Spec::remotealloc_fields_from_msgpack(r_, n_ - 1);
      case 32: return Spec::remotetempalloc_fields_from_msgpack(r_, n_ - 1);
      case 33: return Spec::remotefree_fields_from_msgpack(r_, n_ - 1);
      case 34: return Spec::remotememcpy_fields_from_msgpack(r_, n_ - 1);
      case 35: return Spec::remotesync_fields_from_msgpack(r_, n_ - 1);
      default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
    }
  } else {
    const auto ord_ = r_.readInt32();
    switch (ord_) {
      case 0:
      case 7:
      case 8:
      case 9:
      case 10:
      case 11:
      case 12:
      case 15:
      case 16:
      case 17:
      case 18:
      case 19:
      case 20:
      case 21:
      case 22:
      case 23:
      case 24:
      case 25:
      case 26:
      case 27:
      case 28:
      case 29:
      case 30:
      case 31:
      case 32:
      case 33:
      case 34:
      case 35: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      case 1: return Spec::gpubarrierglobal_fields_from_msgpack(r_, 0);
      case 2: return Spec::gpubarrierlocal_fields_from_msgpack(r_, 0);
      case 3: return Spec::gpubarrierall_fields_from_msgpack(r_, 0);
      case 4: return Spec::gpufenceglobal_fields_from_msgpack(r_, 0);
      case 5: return Spec::gpufencelocal_fields_from_msgpack(r_, 0);
      case 6: return Spec::gpufenceall_fields_from_msgpack(r_, 0);
      case 13: return Spec::gpulaneidx_fields_from_msgpack(r_, 0);
      case 14: return Spec::gpusubgroupsize_fields_from_msgpack(r_, 0);
      default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
    }
  }
}

void any_to_msgpack(MsgpackWriter &w_, const Spec::Any &x_) {
  x_.match_total(
      [&](const Spec::Assert &y_) -> void {
        w_.writeArrayHeader(3);
        w_.writeInt32(0);
        Spec::assert_fields_to_msgpack(w_, y_);
      },
      [&](const Spec::GpuBarrierGlobal &y_) -> void { w_.writeInt32(1); },
      [&](const Spec::GpuBarrierLocal &y_) -> void { w_.writeInt32(2); }, [&](const Spec::GpuBarrierAll &y_) -> void { w_.writeInt32(3); },
      [&](const Spec::GpuFenceGlobal &y_) -> void { w_.writeInt32(4); }, [&](const Spec::GpuFenceLocal &y_) -> void { w_.writeInt32(5); },
      [&](const Spec::GpuFenceAll &y_) -> void { w_.writeInt32(6); },
      [&](const Spec::GpuGlobalIdx &y_) -> void {
        w_.writeArrayHeader(2);
        w_.writeInt32(7);
        Spec::gpuglobalidx_fields_to_msgpack(w_, y_);
      },
      [&](const Spec::GpuGlobalSize &y_) -> void {
        w_.writeArrayHeader(2);
        w_.writeInt32(8);
        Spec::gpuglobalsize_fields_to_msgpack(w_, y_);
      },
      [&](const Spec::GpuGroupIdx &y_) -> void {
        w_.writeArrayHeader(2);
        w_.writeInt32(9);
        Spec::gpugroupidx_fields_to_msgpack(w_, y_);
      },
      [&](const Spec::GpuGroupSize &y_) -> void {
        w_.writeArrayHeader(2);
        w_.writeInt32(10);
        Spec::gpugroupsize_fields_to_msgpack(w_, y_);
      },
      [&](const Spec::GpuLocalIdx &y_) -> void {
        w_.writeArrayHeader(2);
        w_.writeInt32(11);
        Spec::gpulocalidx_fields_to_msgpack(w_, y_);
      },
      [&](const Spec::GpuLocalSize &y_) -> void {
        w_.writeArrayHeader(2);
        w_.writeInt32(12);
        Spec::gpulocalsize_fields_to_msgpack(w_, y_);
      },
      [&](const Spec::GpuLaneIdx &y_) -> void { w_.writeInt32(13); }, [&](const Spec::GpuSubgroupSize &y_) -> void { w_.writeInt32(14); },
      [&](const Spec::GpuShuffleDown &y_) -> void {
        w_.writeArrayHeader(6);
        w_.writeInt32(15);
        Spec::gpushuffledown_fields_to_msgpack(w_, y_);
      },
      [&](const Spec::GpuShuffleUp &y_) -> void {
        w_.writeArrayHeader(6);
        w_.writeInt32(16);
        Spec::gpushuffleup_fields_to_msgpack(w_, y_);
      },
      [&](const Spec::GpuShuffleIdx &y_) -> void {
        w_.writeArrayHeader(6);
        w_.writeInt32(17);
        Spec::gpushuffleidx_fields_to_msgpack(w_, y_);
      },
      [&](const Spec::GpuShuffleXor &y_) -> void {
        w_.writeArrayHeader(6);
        w_.writeInt32(18);
        Spec::gpushufflexor_fields_to_msgpack(w_, y_);
      },
      [&](const Spec::GpuSubgroupBarrier &y_) -> void {
        w_.writeArrayHeader(2);
        w_.writeInt32(19);
        Spec::gpusubgroupbarrier_fields_to_msgpack(w_, y_);
      },
      [&](const Spec::GpuBallot &y_) -> void {
        w_.writeArrayHeader(3);
        w_.writeInt32(20);
        Spec::gpuballot_fields_to_msgpack(w_, y_);
      },
      [&](const Spec::GpuVoteAny &y_) -> void {
        w_.writeArrayHeader(3);
        w_.writeInt32(21);
        Spec::gpuvoteany_fields_to_msgpack(w_, y_);
      },
      [&](const Spec::GpuVoteAll &y_) -> void {
        w_.writeArrayHeader(3);
        w_.writeInt32(22);
        Spec::gpuvoteall_fields_to_msgpack(w_, y_);
      },
      [&](const Spec::GpuAtomicRMW &y_) -> void {
        w_.writeArrayHeader(7);
        w_.writeInt32(23);
        Spec::gpuatomicrmw_fields_to_msgpack(w_, y_);
      },
      [&](const Spec::GpuAtomicCAS &y_) -> void {
        w_.writeArrayHeader(7);
        w_.writeInt32(24);
        Spec::gpuatomiccas_fields_to_msgpack(w_, y_);
      },
      [&](const Spec::GpuGroupReduce &y_) -> void {
        w_.writeArrayHeader(4);
        w_.writeInt32(25);
        Spec::gpugroupreduce_fields_to_msgpack(w_, y_);
      },
      [&](const Spec::GpuGroupInclusiveScan &y_) -> void {
        w_.writeArrayHeader(4);
        w_.writeInt32(26);
        Spec::gpugroupinclusivescan_fields_to_msgpack(w_, y_);
      },
      [&](const Spec::GpuGroupExclusiveScan &y_) -> void {
        w_.writeArrayHeader(4);
        w_.writeInt32(27);
        Spec::gpugroupexclusivescan_fields_to_msgpack(w_, y_);
      },
      [&](const Spec::GpuVolatileLoad &y_) -> void {
        w_.writeArrayHeader(3);
        w_.writeInt32(28);
        Spec::gpuvolatileload_fields_to_msgpack(w_, y_);
      },
      [&](const Spec::GpuVolatileStore &y_) -> void {
        w_.writeArrayHeader(3);
        w_.writeInt32(29);
        Spec::gpuvolatilestore_fields_to_msgpack(w_, y_);
      },
      [&](const Spec::RemoteLaunch &y_) -> void {
        w_.writeArrayHeader(12);
        w_.writeInt32(30);
        Spec::remotelaunch_fields_to_msgpack(w_, y_);
      },
      [&](const Spec::RemoteAlloc &y_) -> void {
        w_.writeArrayHeader(3);
        w_.writeInt32(31);
        Spec::remotealloc_fields_to_msgpack(w_, y_);
      },
      [&](const Spec::RemoteTempAlloc &y_) -> void {
        w_.writeArrayHeader(3);
        w_.writeInt32(32);
        Spec::remotetempalloc_fields_to_msgpack(w_, y_);
      },
      [&](const Spec::RemoteFree &y_) -> void {
        w_.writeArrayHeader(3);
        w_.writeInt32(33);
        Spec::remotefree_fields_to_msgpack(w_, y_);
      },
      [&](const Spec::RemoteMemcpy &y_) -> void {
        w_.writeArrayHeader(6);
        w_.writeInt32(34);
        Spec::remotememcpy_fields_to_msgpack(w_, y_);
      },
      [&](const Spec::RemoteSync &y_) -> void {
        w_.writeArrayHeader(2);
        w_.writeInt32(35);
        Spec::remotesync_fields_to_msgpack(w_, y_);
      });
}
}} // namespace Spec
namespace FunctionAffinity { namespace {

FunctionAffinity::Offload offload_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected FunctionAffinity::Offload with 0 field(s)");
  return {};
}

FunctionAffinity::Host host_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected FunctionAffinity::Host with 0 field(s)");
  return {};
}

FunctionAffinity::Any any_from_msgpack(MsgpackReader &r_) {
  if (r_.nextIsArray()) {
    const auto n_ = r_.readArrayHeader();
    if (n_ == 0) throw std::runtime_error("Expected non-empty sum payload");
    const auto ord_ = r_.readInt32();
    switch (ord_) {
      case 0: return FunctionAffinity::offload_fields_from_msgpack(r_, n_ - 1);
      case 1: return FunctionAffinity::host_fields_from_msgpack(r_, n_ - 1);
      default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
    }
  } else {
    const auto ord_ = r_.readInt32();
    switch (ord_) {
      case 0: return FunctionAffinity::offload_fields_from_msgpack(r_, 0);
      case 1: return FunctionAffinity::host_fields_from_msgpack(r_, 0);
      default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
    }
  }
}

void any_to_msgpack(MsgpackWriter &w_, const FunctionAffinity::Any &x_) {
  x_.match_total([&](const FunctionAffinity::Offload &y_) -> void { w_.writeInt32(0); },
                 [&](const FunctionAffinity::Host &y_) -> void { w_.writeInt32(1); });
}
}} // namespace FunctionAffinity
namespace Term { namespace {

Term::Float16Const float16const_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 1) throw std::runtime_error("Expected Term::Float16Const with 1 field(s)");
  const auto value = r_.readFloat32();
  return Term::Float16Const(value);
}

void float16const_fields_to_msgpack(MsgpackWriter &w_, const Term::Float16Const &x_) { w_.writeFloat32(x_.value); }

Term::Float32Const float32const_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 1) throw std::runtime_error("Expected Term::Float32Const with 1 field(s)");
  const auto value = r_.readFloat32();
  return Term::Float32Const(value);
}

void float32const_fields_to_msgpack(MsgpackWriter &w_, const Term::Float32Const &x_) { w_.writeFloat32(x_.value); }

Term::Float64Const float64const_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 1) throw std::runtime_error("Expected Term::Float64Const with 1 field(s)");
  const auto value = r_.readFloat64();
  return Term::Float64Const(value);
}

void float64const_fields_to_msgpack(MsgpackWriter &w_, const Term::Float64Const &x_) { w_.writeFloat64(x_.value); }

Term::IntU8Const intu8const_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 1) throw std::runtime_error("Expected Term::IntU8Const with 1 field(s)");
  const auto value = static_cast<int8_t>(r_.readInt32());
  return Term::IntU8Const(value);
}

void intu8const_fields_to_msgpack(MsgpackWriter &w_, const Term::IntU8Const &x_) { w_.writeInt32(static_cast<int32_t>(x_.value)); }

Term::IntU16Const intu16const_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 1) throw std::runtime_error("Expected Term::IntU16Const with 1 field(s)");
  const auto value = static_cast<uint16_t>(r_.readInt32());
  return Term::IntU16Const(value);
}

void intu16const_fields_to_msgpack(MsgpackWriter &w_, const Term::IntU16Const &x_) { w_.writeInt32(static_cast<int32_t>(x_.value)); }

Term::IntU32Const intu32const_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 1) throw std::runtime_error("Expected Term::IntU32Const with 1 field(s)");
  const auto value = r_.readInt32();
  return Term::IntU32Const(value);
}

void intu32const_fields_to_msgpack(MsgpackWriter &w_, const Term::IntU32Const &x_) { w_.writeInt32(x_.value); }

Term::IntU64Const intu64const_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 1) throw std::runtime_error("Expected Term::IntU64Const with 1 field(s)");
  const auto value = r_.readInt64();
  return Term::IntU64Const(value);
}

void intu64const_fields_to_msgpack(MsgpackWriter &w_, const Term::IntU64Const &x_) { w_.writeInt64(x_.value); }

Term::IntS8Const ints8const_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 1) throw std::runtime_error("Expected Term::IntS8Const with 1 field(s)");
  const auto value = static_cast<int8_t>(r_.readInt32());
  return Term::IntS8Const(value);
}

void ints8const_fields_to_msgpack(MsgpackWriter &w_, const Term::IntS8Const &x_) { w_.writeInt32(static_cast<int32_t>(x_.value)); }

Term::IntS16Const ints16const_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 1) throw std::runtime_error("Expected Term::IntS16Const with 1 field(s)");
  const auto value = static_cast<int16_t>(r_.readInt32());
  return Term::IntS16Const(value);
}

void ints16const_fields_to_msgpack(MsgpackWriter &w_, const Term::IntS16Const &x_) { w_.writeInt32(static_cast<int32_t>(x_.value)); }

Term::IntS32Const ints32const_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 1) throw std::runtime_error("Expected Term::IntS32Const with 1 field(s)");
  const auto value = r_.readInt32();
  return Term::IntS32Const(value);
}

void ints32const_fields_to_msgpack(MsgpackWriter &w_, const Term::IntS32Const &x_) { w_.writeInt32(x_.value); }

Term::IntS64Const ints64const_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 1) throw std::runtime_error("Expected Term::IntS64Const with 1 field(s)");
  const auto value = r_.readInt64();
  return Term::IntS64Const(value);
}

void ints64const_fields_to_msgpack(MsgpackWriter &w_, const Term::IntS64Const &x_) { w_.writeInt64(x_.value); }

Term::Unit0Const unit0const_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected Term::Unit0Const with 0 field(s)");
  return {};
}

Term::Bool1Const bool1const_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 1) throw std::runtime_error("Expected Term::Bool1Const with 1 field(s)");
  const auto value = r_.readBoolean();
  return Term::Bool1Const(value);
}

void bool1const_fields_to_msgpack(MsgpackWriter &w_, const Term::Bool1Const &x_) { w_.writeBoolean(x_.value); }

Term::NullPtrConst nullptrconst_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 3) throw std::runtime_error("Expected Term::NullPtrConst with 3 field(s)");
  auto comp = Type::any_from_msgpack(r_);
  auto space = TypeSpace::any_from_msgpack(r_);
  auto region = Region::any_from_msgpack(r_);
  return {std::move(comp), std::move(space), std::move(region)};
}

void nullptrconst_fields_to_msgpack(MsgpackWriter &w_, const Term::NullPtrConst &x_) {
  Type::any_to_msgpack(w_, x_.comp);
  TypeSpace::any_to_msgpack(w_, x_.space);
  Region::any_to_msgpack(w_, x_.region);
}

Term::StringConst stringconst_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 1) throw std::runtime_error("Expected Term::StringConst with 1 field(s)");
  auto value = r_.readString();
  return Term::StringConst(std::move(value));
}

void stringconst_fields_to_msgpack(MsgpackWriter &w_, const Term::StringConst &x_) { w_.writeString(x_.value); }

Term::Poison poison_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 1) throw std::runtime_error("Expected Term::Poison with 1 field(s)");
  auto t = Type::any_from_msgpack(r_);
  return Term::Poison(std::move(t));
}

void poison_fields_to_msgpack(MsgpackWriter &w_, const Term::Poison &x_) { Type::any_to_msgpack(w_, x_.t); }

Term::Defer defer_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 1) throw std::runtime_error("Expected Term::Defer with 1 field(s)");
  auto t = Type::any_from_msgpack(r_);
  return Term::Defer(std::move(t));
}

void defer_fields_to_msgpack(MsgpackWriter &w_, const Term::Defer &x_) { Type::any_to_msgpack(w_, x_.t); }

Term::Select select_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 3) throw std::runtime_error("Expected Term::Select with 3 field(s)");
  auto root = named_from_msgpack(r_);
  std::vector<PathStep::Any> steps;
  {
    const auto steps_size = r_.readArrayHeader();
    steps.reserve(steps_size);
    for (size_t steps_idx = 0; steps_idx < steps_size; ++steps_idx) {
      auto steps_elem = PathStep::any_from_msgpack(r_);
      steps.emplace_back(std::move(steps_elem));
    }
  }
  auto tpe = Type::any_from_msgpack(r_);
  return {std::move(root), std::move(steps), std::move(tpe)};
}

void select_fields_to_msgpack(MsgpackWriter &w_, const Term::Select &x_) {
  named_to_msgpack(w_, x_.root);
  w_.writeArrayHeader(x_.steps.size());
  for (const auto &v0_ : x_.steps) {
    PathStep::any_to_msgpack(w_, v0_);
  }
  Type::any_to_msgpack(w_, x_.tpe);
}

Term::Select select_from_msgpack(MsgpackReader &r_) {
  const auto n_ = r_.readArrayHeader();
  return Term::select_fields_from_msgpack(r_, n_);
}

void select_to_msgpack(MsgpackWriter &w_, const Term::Select &x_) {
  w_.writeArrayHeader(3);
  Term::select_fields_to_msgpack(w_, x_);
}

Term::Any any_from_msgpack(MsgpackReader &r_) {
  if (r_.nextIsArray()) {
    const auto n_ = r_.readArrayHeader();
    if (n_ == 0) throw std::runtime_error("Expected non-empty sum payload");
    const auto ord_ = r_.readInt32();
    switch (ord_) {
      case 0: return Term::float16const_fields_from_msgpack(r_, n_ - 1);
      case 1: return Term::float32const_fields_from_msgpack(r_, n_ - 1);
      case 2: return Term::float64const_fields_from_msgpack(r_, n_ - 1);
      case 3: return Term::intu8const_fields_from_msgpack(r_, n_ - 1);
      case 4: return Term::intu16const_fields_from_msgpack(r_, n_ - 1);
      case 5: return Term::intu32const_fields_from_msgpack(r_, n_ - 1);
      case 6: return Term::intu64const_fields_from_msgpack(r_, n_ - 1);
      case 7: return Term::ints8const_fields_from_msgpack(r_, n_ - 1);
      case 8: return Term::ints16const_fields_from_msgpack(r_, n_ - 1);
      case 9: return Term::ints32const_fields_from_msgpack(r_, n_ - 1);
      case 10: return Term::ints64const_fields_from_msgpack(r_, n_ - 1);
      case 11: return Term::unit0const_fields_from_msgpack(r_, n_ - 1);
      case 12: return Term::bool1const_fields_from_msgpack(r_, n_ - 1);
      case 13: return Term::nullptrconst_fields_from_msgpack(r_, n_ - 1);
      case 14: return Term::stringconst_fields_from_msgpack(r_, n_ - 1);
      case 15: return Term::poison_fields_from_msgpack(r_, n_ - 1);
      case 16: return Term::defer_fields_from_msgpack(r_, n_ - 1);
      case 17: return Term::select_fields_from_msgpack(r_, n_ - 1);
      default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
    }
  } else {
    const auto ord_ = r_.readInt32();
    switch (ord_) {
      case 0:
      case 1:
      case 2:
      case 3:
      case 4:
      case 5:
      case 6:
      case 7:
      case 8:
      case 9:
      case 10:
      case 12:
      case 13:
      case 14:
      case 15:
      case 16:
      case 17: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      case 11: return Term::unit0const_fields_from_msgpack(r_, 0);
      default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
    }
  }
}

void any_to_msgpack(MsgpackWriter &w_, const Term::Any &x_) {
  x_.match_total(
      [&](const Term::Float16Const &y_) -> void {
        w_.writeArrayHeader(2);
        w_.writeInt32(0);
        Term::float16const_fields_to_msgpack(w_, y_);
      },
      [&](const Term::Float32Const &y_) -> void {
        w_.writeArrayHeader(2);
        w_.writeInt32(1);
        Term::float32const_fields_to_msgpack(w_, y_);
      },
      [&](const Term::Float64Const &y_) -> void {
        w_.writeArrayHeader(2);
        w_.writeInt32(2);
        Term::float64const_fields_to_msgpack(w_, y_);
      },
      [&](const Term::IntU8Const &y_) -> void {
        w_.writeArrayHeader(2);
        w_.writeInt32(3);
        Term::intu8const_fields_to_msgpack(w_, y_);
      },
      [&](const Term::IntU16Const &y_) -> void {
        w_.writeArrayHeader(2);
        w_.writeInt32(4);
        Term::intu16const_fields_to_msgpack(w_, y_);
      },
      [&](const Term::IntU32Const &y_) -> void {
        w_.writeArrayHeader(2);
        w_.writeInt32(5);
        Term::intu32const_fields_to_msgpack(w_, y_);
      },
      [&](const Term::IntU64Const &y_) -> void {
        w_.writeArrayHeader(2);
        w_.writeInt32(6);
        Term::intu64const_fields_to_msgpack(w_, y_);
      },
      [&](const Term::IntS8Const &y_) -> void {
        w_.writeArrayHeader(2);
        w_.writeInt32(7);
        Term::ints8const_fields_to_msgpack(w_, y_);
      },
      [&](const Term::IntS16Const &y_) -> void {
        w_.writeArrayHeader(2);
        w_.writeInt32(8);
        Term::ints16const_fields_to_msgpack(w_, y_);
      },
      [&](const Term::IntS32Const &y_) -> void {
        w_.writeArrayHeader(2);
        w_.writeInt32(9);
        Term::ints32const_fields_to_msgpack(w_, y_);
      },
      [&](const Term::IntS64Const &y_) -> void {
        w_.writeArrayHeader(2);
        w_.writeInt32(10);
        Term::ints64const_fields_to_msgpack(w_, y_);
      },
      [&](const Term::Unit0Const &y_) -> void { w_.writeInt32(11); },
      [&](const Term::Bool1Const &y_) -> void {
        w_.writeArrayHeader(2);
        w_.writeInt32(12);
        Term::bool1const_fields_to_msgpack(w_, y_);
      },
      [&](const Term::NullPtrConst &y_) -> void {
        w_.writeArrayHeader(4);
        w_.writeInt32(13);
        Term::nullptrconst_fields_to_msgpack(w_, y_);
      },
      [&](const Term::StringConst &y_) -> void {
        w_.writeArrayHeader(2);
        w_.writeInt32(14);
        Term::stringconst_fields_to_msgpack(w_, y_);
      },
      [&](const Term::Poison &y_) -> void {
        w_.writeArrayHeader(2);
        w_.writeInt32(15);
        Term::poison_fields_to_msgpack(w_, y_);
      },
      [&](const Term::Defer &y_) -> void {
        w_.writeArrayHeader(2);
        w_.writeInt32(16);
        Term::defer_fields_to_msgpack(w_, y_);
      },
      [&](const Term::Select &y_) -> void {
        w_.writeArrayHeader(4);
        w_.writeInt32(17);
        Term::select_fields_to_msgpack(w_, y_);
      });
}
}} // namespace Term
namespace {

Sym sym_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 1) throw std::runtime_error("Expected Sym with 1 field(s)");
  std::vector<std::string> fqn;
  {
    const auto fqn_size = r_.readArrayHeader();
    fqn.reserve(fqn_size);
    for (size_t fqn_idx = 0; fqn_idx < fqn_size; ++fqn_idx) {
      auto fqn_elem = r_.readString();
      fqn.emplace_back(std::move(fqn_elem));
    }
  }
  return Sym(std::move(fqn));
}

void sym_fields_to_msgpack(MsgpackWriter &w_, const Sym &x_) {
  w_.writeArrayHeader(x_.fqn.size());
  for (const auto &v0_ : x_.fqn) {
    w_.writeString(v0_);
  }
}

Sym sym_from_msgpack(MsgpackReader &r_) {
  const auto n_ = r_.readArrayHeader();
  return sym_fields_from_msgpack(r_, n_);
}

void sym_to_msgpack(MsgpackWriter &w_, const Sym &x_) {
  w_.writeArrayHeader(1);
  sym_fields_to_msgpack(w_, x_);
}

SourcePosition sourceposition_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 3) throw std::runtime_error("Expected SourcePosition with 3 field(s)");
  auto file = r_.readString();
  const auto line = r_.readInt32();
  std::optional<int32_t> col;
  if (!r_.tryReadNil()) {
    const auto col_value = r_.readInt32();
    col = col_value;
  }
  return {std::move(file), line, col};
}

void sourceposition_fields_to_msgpack(MsgpackWriter &w_, const SourcePosition &x_) {
  w_.writeString(x_.file);
  w_.writeInt32(x_.line);
  if (x_.col) {
    w_.writeInt32((*x_.col));
  } else {
    w_.writeNil();
  }
}

SourcePosition sourceposition_from_msgpack(MsgpackReader &r_) {
  const auto n_ = r_.readArrayHeader();
  return sourceposition_fields_from_msgpack(r_, n_);
}

void sourceposition_to_msgpack(MsgpackWriter &w_, const SourcePosition &x_) {
  w_.writeArrayHeader(3);
  sourceposition_fields_to_msgpack(w_, x_);
}

Origin origin_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 3) throw std::runtime_error("Expected Origin with 3 field(s)");
  std::optional<SourcePosition> pos;
  if (!r_.tryReadNil()) {
    auto pos_value = sourceposition_from_msgpack(r_);
    pos = std::move(pos_value);
  }
  std::optional<std::string> source;
  if (!r_.tryReadNil()) {
    auto source_value = r_.readString();
    source = std::move(source_value);
  }
  std::optional<Sym> inlinedFrom;
  if (!r_.tryReadNil()) {
    auto inlinedFrom_value = sym_from_msgpack(r_);
    inlinedFrom = std::move(inlinedFrom_value);
  }
  return {std::move(pos), std::move(source), std::move(inlinedFrom)};
}

void origin_fields_to_msgpack(MsgpackWriter &w_, const Origin &x_) {
  if (x_.pos) {
    sourceposition_to_msgpack(w_, (*x_.pos));
  } else {
    w_.writeNil();
  }
  if (x_.source) {
    w_.writeString((*x_.source));
  } else {
    w_.writeNil();
  }
  if (x_.inlinedFrom) {
    sym_to_msgpack(w_, (*x_.inlinedFrom));
  } else {
    w_.writeNil();
  }
}

Origin origin_from_msgpack(MsgpackReader &r_) {
  const auto n_ = r_.readArrayHeader();
  return origin_fields_from_msgpack(r_, n_);
}

void origin_to_msgpack(MsgpackWriter &w_, const Origin &x_) {
  w_.writeArrayHeader(3);
  origin_fields_to_msgpack(w_, x_);
}

Named named_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 3) throw std::runtime_error("Expected Named with 3 field(s)");
  auto symbol = r_.readString();
  auto tpe = Type::any_from_msgpack(r_);
  auto origin = origin_from_msgpack(r_);
  return {std::move(symbol), std::move(tpe), std::move(origin)};
}

void named_fields_to_msgpack(MsgpackWriter &w_, const Named &x_) {
  w_.writeString(x_.symbol);
  Type::any_to_msgpack(w_, x_.tpe);
  origin_to_msgpack(w_, x_.origin);
}

Named named_from_msgpack(MsgpackReader &r_) {
  const auto n_ = r_.readArrayHeader();
  return named_fields_from_msgpack(r_, n_);
}

void named_to_msgpack(MsgpackWriter &w_, const Named &x_) {
  w_.writeArrayHeader(3);
  named_fields_to_msgpack(w_, x_);
}

ExceptionKind exceptionkind_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected ExceptionKind with 2 field(s)");
  auto tpe = Type::any_from_msgpack(r_);
  auto sourceName = r_.readString();
  return {std::move(tpe), std::move(sourceName)};
}

void exceptionkind_fields_to_msgpack(MsgpackWriter &w_, const ExceptionKind &x_) {
  Type::any_to_msgpack(w_, x_.tpe);
  w_.writeString(x_.sourceName);
}

ExceptionKind exceptionkind_from_msgpack(MsgpackReader &r_) {
  const auto n_ = r_.readArrayHeader();
  return exceptionkind_fields_from_msgpack(r_, n_);
}

void exceptionkind_to_msgpack(MsgpackWriter &w_, const ExceptionKind &x_) {
  w_.writeArrayHeader(2);
  exceptionkind_fields_to_msgpack(w_, x_);
}

Handler handler_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 3) throw std::runtime_error("Expected Handler with 3 field(s)");
  std::optional<ExceptionKind> caught;
  if (!r_.tryReadNil()) {
    auto caught_value = exceptionkind_from_msgpack(r_);
    caught = std::move(caught_value);
  }
  std::optional<Named> binder;
  if (!r_.tryReadNil()) {
    auto binder_value = named_from_msgpack(r_);
    binder = std::move(binder_value);
  }
  std::vector<Stmt::Any> body;
  {
    const auto body_size = r_.readArrayHeader();
    body.reserve(body_size);
    for (size_t body_idx = 0; body_idx < body_size; ++body_idx) {
      auto body_elem = Stmt::any_from_msgpack(r_);
      body.emplace_back(std::move(body_elem));
    }
  }
  return {std::move(caught), std::move(binder), std::move(body)};
}

void handler_fields_to_msgpack(MsgpackWriter &w_, const Handler &x_) {
  if (x_.caught) {
    exceptionkind_to_msgpack(w_, (*x_.caught));
  } else {
    w_.writeNil();
  }
  if (x_.binder) {
    named_to_msgpack(w_, (*x_.binder));
  } else {
    w_.writeNil();
  }
  w_.writeArrayHeader(x_.body.size());
  for (const auto &v0_ : x_.body) {
    Stmt::any_to_msgpack(w_, v0_);
  }
}

Handler handler_from_msgpack(MsgpackReader &r_) {
  const auto n_ = r_.readArrayHeader();
  return handler_fields_from_msgpack(r_, n_);
}

void handler_to_msgpack(MsgpackWriter &w_, const Handler &x_) {
  w_.writeArrayHeader(3);
  handler_fields_to_msgpack(w_, x_);
}

ArgBoundary argboundary_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected ArgBoundary with 2 field(s)");
  auto access = ArgAccess::any_from_msgpack(r_);
  auto extent = ArgExtent::any_from_msgpack(r_);
  return {std::move(access), std::move(extent)};
}

void argboundary_fields_to_msgpack(MsgpackWriter &w_, const ArgBoundary &x_) {
  ArgAccess::any_to_msgpack(w_, x_.access);
  ArgExtent::any_to_msgpack(w_, x_.extent);
}

ArgBoundary argboundary_from_msgpack(MsgpackReader &r_) {
  const auto n_ = r_.readArrayHeader();
  return argboundary_fields_from_msgpack(r_, n_);
}

void argboundary_to_msgpack(MsgpackWriter &w_, const ArgBoundary &x_) {
  w_.writeArrayHeader(2);
  argboundary_fields_to_msgpack(w_, x_);
}

Arg arg_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 3) throw std::runtime_error("Expected Arg with 3 field(s)");
  auto named = named_from_msgpack(r_);
  std::optional<SourcePosition> pos;
  if (!r_.tryReadNil()) {
    auto pos_value = sourceposition_from_msgpack(r_);
    pos = std::move(pos_value);
  }
  std::optional<ArgBoundary> boundary;
  if (!r_.tryReadNil()) {
    auto boundary_value = argboundary_from_msgpack(r_);
    boundary = std::move(boundary_value);
  }
  return {std::move(named), std::move(pos), std::move(boundary)};
}

void arg_fields_to_msgpack(MsgpackWriter &w_, const Arg &x_) {
  named_to_msgpack(w_, x_.named);
  if (x_.pos) {
    sourceposition_to_msgpack(w_, (*x_.pos));
  } else {
    w_.writeNil();
  }
  if (x_.boundary) {
    argboundary_to_msgpack(w_, (*x_.boundary));
  } else {
    w_.writeNil();
  }
}

Arg arg_from_msgpack(MsgpackReader &r_) {
  const auto n_ = r_.readArrayHeader();
  return arg_fields_from_msgpack(r_, n_);
}

void arg_to_msgpack(MsgpackWriter &w_, const Arg &x_) {
  w_.writeArrayHeader(3);
  arg_fields_to_msgpack(w_, x_);
}

FunctionDecl functiondecl_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 8) throw std::runtime_error("Expected FunctionDecl with 8 field(s)");
  auto name = sym_from_msgpack(r_);
  std::vector<Type::Var> tpeVars;
  {
    const auto tpeVars_size = r_.readArrayHeader();
    tpeVars.reserve(tpeVars_size);
    for (size_t tpeVars_idx = 0; tpeVars_idx < tpeVars_size; ++tpeVars_idx) {
      auto tpeVars_elem = Type::var_from_msgpack(r_);
      tpeVars.emplace_back(std::move(tpeVars_elem));
    }
  }
  std::optional<Arg> receiver;
  if (!r_.tryReadNil()) {
    auto receiver_value = arg_from_msgpack(r_);
    receiver = std::move(receiver_value);
  }
  std::vector<Arg> args;
  {
    const auto args_size = r_.readArrayHeader();
    args.reserve(args_size);
    for (size_t args_idx = 0; args_idx < args_size; ++args_idx) {
      auto args_elem = arg_from_msgpack(r_);
      args.emplace_back(std::move(args_elem));
    }
  }
  std::vector<Arg> moduleCaptures;
  {
    const auto moduleCaptures_size = r_.readArrayHeader();
    moduleCaptures.reserve(moduleCaptures_size);
    for (size_t moduleCaptures_idx = 0; moduleCaptures_idx < moduleCaptures_size; ++moduleCaptures_idx) {
      auto moduleCaptures_elem = arg_from_msgpack(r_);
      moduleCaptures.emplace_back(std::move(moduleCaptures_elem));
    }
  }
  std::vector<Arg> termCaptures;
  {
    const auto termCaptures_size = r_.readArrayHeader();
    termCaptures.reserve(termCaptures_size);
    for (size_t termCaptures_idx = 0; termCaptures_idx < termCaptures_size; ++termCaptures_idx) {
      auto termCaptures_elem = arg_from_msgpack(r_);
      termCaptures.emplace_back(std::move(termCaptures_elem));
    }
  }
  auto rtn = Type::any_from_msgpack(r_);
  auto affinity = FunctionAffinity::any_from_msgpack(r_);
  return {std::move(name),           std::move(tpeVars),      std::move(receiver), std::move(args),
          std::move(moduleCaptures), std::move(termCaptures), std::move(rtn),      std::move(affinity)};
}

void functiondecl_fields_to_msgpack(MsgpackWriter &w_, const FunctionDecl &x_) {
  sym_to_msgpack(w_, x_.name);
  w_.writeArrayHeader(x_.tpeVars.size());
  for (const auto &v0_ : x_.tpeVars) {
    Type::var_to_msgpack(w_, v0_);
  }
  if (x_.receiver) {
    arg_to_msgpack(w_, (*x_.receiver));
  } else {
    w_.writeNil();
  }
  w_.writeArrayHeader(x_.args.size());
  for (const auto &v0_ : x_.args) {
    arg_to_msgpack(w_, v0_);
  }
  w_.writeArrayHeader(x_.moduleCaptures.size());
  for (const auto &v0_ : x_.moduleCaptures) {
    arg_to_msgpack(w_, v0_);
  }
  w_.writeArrayHeader(x_.termCaptures.size());
  for (const auto &v0_ : x_.termCaptures) {
    arg_to_msgpack(w_, v0_);
  }
  Type::any_to_msgpack(w_, x_.rtn);
  FunctionAffinity::any_to_msgpack(w_, x_.affinity);
}

FunctionDecl functiondecl_from_msgpack(MsgpackReader &r_) {
  const auto n_ = r_.readArrayHeader();
  return functiondecl_fields_from_msgpack(r_, n_);
}

void functiondecl_to_msgpack(MsgpackWriter &w_, const FunctionDecl &x_) {
  w_.writeArrayHeader(8);
  functiondecl_fields_to_msgpack(w_, x_);
}

Function function_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 7) throw std::runtime_error("Expected Function with 7 field(s)");
  auto decl = functiondecl_from_msgpack(r_);
  std::vector<Stmt::Any> body;
  {
    const auto body_size = r_.readArrayHeader();
    body.reserve(body_size);
    for (size_t body_idx = 0; body_idx < body_size; ++body_idx) {
      auto body_elem = Stmt::any_from_msgpack(r_);
      body.emplace_back(std::move(body_elem));
    }
  }
  auto visibility = FunctionVisibility::any_from_msgpack(r_);
  auto fpMode = FunctionFpMode::any_from_msgpack(r_);
  auto convention = CallConvention::any_from_msgpack(r_);
  std::optional<Sym> implements;
  if (!r_.tryReadNil()) {
    auto implements_value = sym_from_msgpack(r_);
    implements = std::move(implements_value);
  }
  std::vector<std::string> requiredCapabilities;
  {
    const auto requiredCapabilities_size = r_.readArrayHeader();
    requiredCapabilities.reserve(requiredCapabilities_size);
    for (size_t requiredCapabilities_idx = 0; requiredCapabilities_idx < requiredCapabilities_size; ++requiredCapabilities_idx) {
      auto requiredCapabilities_elem = r_.readString();
      requiredCapabilities.emplace_back(std::move(requiredCapabilities_elem));
    }
  }
  return {std::move(decl),       std::move(body),       std::move(visibility),          std::move(fpMode),
          std::move(convention), std::move(implements), std::move(requiredCapabilities)};
}

void function_fields_to_msgpack(MsgpackWriter &w_, const Function &x_) {
  functiondecl_to_msgpack(w_, x_.decl);
  w_.writeArrayHeader(x_.body.size());
  for (const auto &v0_ : x_.body) {
    Stmt::any_to_msgpack(w_, v0_);
  }
  FunctionVisibility::any_to_msgpack(w_, x_.visibility);
  FunctionFpMode::any_to_msgpack(w_, x_.fpMode);
  CallConvention::any_to_msgpack(w_, x_.convention);
  if (x_.implements) {
    sym_to_msgpack(w_, (*x_.implements));
  } else {
    w_.writeNil();
  }
  w_.writeArrayHeader(x_.requiredCapabilities.size());
  for (const auto &v0_ : x_.requiredCapabilities) {
    w_.writeString(v0_);
  }
}

Function function_from_msgpack(MsgpackReader &r_) {
  const auto n_ = r_.readArrayHeader();
  return function_fields_from_msgpack(r_, n_);
}

void function_to_msgpack(MsgpackWriter &w_, const Function &x_) {
  w_.writeArrayHeader(7);
  function_fields_to_msgpack(w_, x_);
}

StructDef structdef_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 5) throw std::runtime_error("Expected StructDef with 5 field(s)");
  auto name = sym_from_msgpack(r_);
  std::vector<Type::Var> tpeVars;
  {
    const auto tpeVars_size = r_.readArrayHeader();
    tpeVars.reserve(tpeVars_size);
    for (size_t tpeVars_idx = 0; tpeVars_idx < tpeVars_size; ++tpeVars_idx) {
      auto tpeVars_elem = Type::var_from_msgpack(r_);
      tpeVars.emplace_back(std::move(tpeVars_elem));
    }
  }
  std::vector<Named> members;
  {
    const auto members_size = r_.readArrayHeader();
    members.reserve(members_size);
    for (size_t members_idx = 0; members_idx < members_size; ++members_idx) {
      auto members_elem = named_from_msgpack(r_);
      members.emplace_back(std::move(members_elem));
    }
  }
  std::vector<Type::Struct> parents;
  {
    const auto parents_size = r_.readArrayHeader();
    parents.reserve(parents_size);
    for (size_t parents_idx = 0; parents_idx < parents_size; ++parents_idx) {
      auto parents_elem = Type::struct_from_msgpack(r_);
      parents.emplace_back(std::move(parents_elem));
    }
  }
  const auto isUnion = r_.readBoolean();
  return {std::move(name), std::move(tpeVars), std::move(members), std::move(parents), isUnion};
}

void structdef_fields_to_msgpack(MsgpackWriter &w_, const StructDef &x_) {
  sym_to_msgpack(w_, x_.name);
  w_.writeArrayHeader(x_.tpeVars.size());
  for (const auto &v0_ : x_.tpeVars) {
    Type::var_to_msgpack(w_, v0_);
  }
  w_.writeArrayHeader(x_.members.size());
  for (const auto &v0_ : x_.members) {
    named_to_msgpack(w_, v0_);
  }
  w_.writeArrayHeader(x_.parents.size());
  for (const auto &v0_ : x_.parents) {
    Type::struct_to_msgpack(w_, v0_);
  }
  w_.writeBoolean(x_.isUnion);
}

StructDef structdef_from_msgpack(MsgpackReader &r_) {
  const auto n_ = r_.readArrayHeader();
  return structdef_fields_from_msgpack(r_, n_);
}

void structdef_to_msgpack(MsgpackWriter &w_, const StructDef &x_) {
  w_.writeArrayHeader(5);
  structdef_fields_to_msgpack(w_, x_);
}

MetaEntry metaentry_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected MetaEntry with 2 field(s)");
  auto key = r_.readString();
  auto value = r_.readString();
  return {std::move(key), std::move(value)};
}

void metaentry_fields_to_msgpack(MsgpackWriter &w_, const MetaEntry &x_) {
  w_.writeString(x_.key);
  w_.writeString(x_.value);
}

MetaEntry metaentry_from_msgpack(MsgpackReader &r_) {
  const auto n_ = r_.readArrayHeader();
  return metaentry_fields_from_msgpack(r_, n_);
}

void metaentry_to_msgpack(MsgpackWriter &w_, const MetaEntry &x_) {
  w_.writeArrayHeader(2);
  metaentry_fields_to_msgpack(w_, x_);
}

Program program_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 5) throw std::runtime_error("Expected Program with 5 field(s)");
  std::optional<Function> entry;
  if (!r_.tryReadNil()) {
    auto entry_value = function_from_msgpack(r_);
    entry = std::move(entry_value);
  }
  std::vector<Function> functions;
  {
    const auto functions_size = r_.readArrayHeader();
    functions.reserve(functions_size);
    for (size_t functions_idx = 0; functions_idx < functions_size; ++functions_idx) {
      auto functions_elem = function_from_msgpack(r_);
      functions.emplace_back(std::move(functions_elem));
    }
  }
  std::vector<StructDef> defs;
  {
    const auto defs_size = r_.readArrayHeader();
    defs.reserve(defs_size);
    for (size_t defs_idx = 0; defs_idx < defs_size; ++defs_idx) {
      auto defs_elem = structdef_from_msgpack(r_);
      defs.emplace_back(std::move(defs_elem));
    }
  }
  auto phase = PassPhase::any_from_msgpack(r_);
  std::vector<MetaEntry> metadata;
  {
    const auto metadata_size = r_.readArrayHeader();
    metadata.reserve(metadata_size);
    for (size_t metadata_idx = 0; metadata_idx < metadata_size; ++metadata_idx) {
      auto metadata_elem = metaentry_from_msgpack(r_);
      metadata.emplace_back(std::move(metadata_elem));
    }
  }
  return {std::move(entry), std::move(functions), std::move(defs), std::move(phase), std::move(metadata)};
}

void program_fields_to_msgpack(MsgpackWriter &w_, const Program &x_) {
  if (x_.entry) {
    function_to_msgpack(w_, (*x_.entry));
  } else {
    w_.writeNil();
  }
  w_.writeArrayHeader(x_.functions.size());
  for (const auto &v0_ : x_.functions) {
    function_to_msgpack(w_, v0_);
  }
  w_.writeArrayHeader(x_.defs.size());
  for (const auto &v0_ : x_.defs) {
    structdef_to_msgpack(w_, v0_);
  }
  PassPhase::any_to_msgpack(w_, x_.phase);
  w_.writeArrayHeader(x_.metadata.size());
  for (const auto &v0_ : x_.metadata) {
    metaentry_to_msgpack(w_, v0_);
  }
}

Program program_from_msgpack(MsgpackReader &r_) {
  const auto n_ = r_.readArrayHeader();
  return program_fields_from_msgpack(r_, n_);
}

void program_to_msgpack(MsgpackWriter &w_, const Program &x_) {
  w_.writeArrayHeader(5);
  program_fields_to_msgpack(w_, x_);
}

StructLayoutMember structlayoutmember_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 3) throw std::runtime_error("Expected StructLayoutMember with 3 field(s)");
  auto name = named_from_msgpack(r_);
  const auto offsetInBytes = r_.readInt64();
  const auto sizeInBytes = r_.readInt64();
  return {std::move(name), offsetInBytes, sizeInBytes};
}

void structlayoutmember_fields_to_msgpack(MsgpackWriter &w_, const StructLayoutMember &x_) {
  named_to_msgpack(w_, x_.name);
  w_.writeInt64(x_.offsetInBytes);
  w_.writeInt64(x_.sizeInBytes);
}

StructLayoutMember structlayoutmember_from_msgpack(MsgpackReader &r_) {
  const auto n_ = r_.readArrayHeader();
  return structlayoutmember_fields_from_msgpack(r_, n_);
}

void structlayoutmember_to_msgpack(MsgpackWriter &w_, const StructLayoutMember &x_) {
  w_.writeArrayHeader(3);
  structlayoutmember_fields_to_msgpack(w_, x_);
}

StructLayout structlayout_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 4) throw std::runtime_error("Expected StructLayout with 4 field(s)");
  auto name = r_.readString();
  const auto sizeInBytes = r_.readInt64();
  const auto alignment = r_.readInt64();
  std::vector<StructLayoutMember> members;
  {
    const auto members_size = r_.readArrayHeader();
    members.reserve(members_size);
    for (size_t members_idx = 0; members_idx < members_size; ++members_idx) {
      auto members_elem = structlayoutmember_from_msgpack(r_);
      members.emplace_back(std::move(members_elem));
    }
  }
  return {std::move(name), sizeInBytes, alignment, std::move(members)};
}

void structlayout_fields_to_msgpack(MsgpackWriter &w_, const StructLayout &x_) {
  w_.writeString(x_.name);
  w_.writeInt64(x_.sizeInBytes);
  w_.writeInt64(x_.alignment);
  w_.writeArrayHeader(x_.members.size());
  for (const auto &v0_ : x_.members) {
    structlayoutmember_to_msgpack(w_, v0_);
  }
}

StructLayout structlayout_from_msgpack(MsgpackReader &r_) {
  const auto n_ = r_.readArrayHeader();
  return structlayout_fields_from_msgpack(r_, n_);
}

void structlayout_to_msgpack(MsgpackWriter &w_, const StructLayout &x_) {
  w_.writeArrayHeader(4);
  structlayout_fields_to_msgpack(w_, x_);
}

CompileEvent compileevent_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 5) throw std::runtime_error("Expected CompileEvent with 5 field(s)");
  const auto epochMillis = r_.readInt64();
  const auto elapsedNanos = r_.readInt64();
  auto name = r_.readString();
  auto data = r_.readString();
  std::vector<CompileEvent> items;
  {
    const auto items_size = r_.readArrayHeader();
    items.reserve(items_size);
    for (size_t items_idx = 0; items_idx < items_size; ++items_idx) {
      auto items_elem = compileevent_from_msgpack(r_);
      items.emplace_back(std::move(items_elem));
    }
  }
  return {epochMillis, elapsedNanos, std::move(name), std::move(data), std::move(items)};
}

void compileevent_fields_to_msgpack(MsgpackWriter &w_, const CompileEvent &x_) {
  w_.writeInt64(x_.epochMillis);
  w_.writeInt64(x_.elapsedNanos);
  w_.writeString(x_.name);
  w_.writeString(x_.data);
  w_.writeArrayHeader(x_.items.size());
  for (const auto &v0_ : x_.items) {
    compileevent_to_msgpack(w_, v0_);
  }
}

CompileEvent compileevent_from_msgpack(MsgpackReader &r_) {
  const auto n_ = r_.readArrayHeader();
  return compileevent_fields_from_msgpack(r_, n_);
}

void compileevent_to_msgpack(MsgpackWriter &w_, const CompileEvent &x_) {
  w_.writeArrayHeader(5);
  compileevent_fields_to_msgpack(w_, x_);
}

PassRunResult passrunresult_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected PassRunResult with 2 field(s)");
  auto program = program_from_msgpack(r_);
  auto event = compileevent_from_msgpack(r_);
  return {std::move(program), std::move(event)};
}

void passrunresult_fields_to_msgpack(MsgpackWriter &w_, const PassRunResult &x_) {
  program_to_msgpack(w_, x_.program);
  compileevent_to_msgpack(w_, x_.event);
}

PassRunResult passrunresult_from_msgpack(MsgpackReader &r_) {
  const auto n_ = r_.readArrayHeader();
  return passrunresult_fields_from_msgpack(r_, n_);
}

void passrunresult_to_msgpack(MsgpackWriter &w_, const PassRunResult &x_) {
  w_.writeArrayHeader(2);
  passrunresult_fields_to_msgpack(w_, x_);
}

CompileResult compileresult_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 6) throw std::runtime_error("Expected CompileResult with 6 field(s)");
  std::optional<std::vector<int8_t>> binary;
  if (!r_.tryReadNil()) {
    std::vector<int8_t> binary_value;
    {
      const auto binary_value_size = r_.readArrayHeader();
      binary_value.reserve(binary_value_size);
      for (size_t binary_value_idx = 0; binary_value_idx < binary_value_size; ++binary_value_idx) {
        const auto binary_value_elem = static_cast<int8_t>(r_.readInt32());
        binary_value.emplace_back(binary_value_elem);
      }
    }
    binary = std::move(binary_value);
  }
  std::vector<std::string> features;
  {
    const auto features_size = r_.readArrayHeader();
    features.reserve(features_size);
    for (size_t features_idx = 0; features_idx < features_size; ++features_idx) {
      auto features_elem = r_.readString();
      features.emplace_back(std::move(features_elem));
    }
  }
  std::vector<CompileEvent> events;
  {
    const auto events_size = r_.readArrayHeader();
    events.reserve(events_size);
    for (size_t events_idx = 0; events_idx < events_size; ++events_idx) {
      auto events_elem = compileevent_from_msgpack(r_);
      events.emplace_back(std::move(events_elem));
    }
  }
  std::vector<StructLayout> layouts;
  {
    const auto layouts_size = r_.readArrayHeader();
    layouts.reserve(layouts_size);
    for (size_t layouts_idx = 0; layouts_idx < layouts_size; ++layouts_idx) {
      auto layouts_elem = structlayout_from_msgpack(r_);
      layouts.emplace_back(std::move(layouts_elem));
    }
  }
  auto messages = r_.readString();
  std::vector<Named> entryArgs;
  {
    const auto entryArgs_size = r_.readArrayHeader();
    entryArgs.reserve(entryArgs_size);
    for (size_t entryArgs_idx = 0; entryArgs_idx < entryArgs_size; ++entryArgs_idx) {
      auto entryArgs_elem = named_from_msgpack(r_);
      entryArgs.emplace_back(std::move(entryArgs_elem));
    }
  }
  return {std::move(binary), std::move(features), std::move(events), std::move(layouts), std::move(messages), std::move(entryArgs)};
}

void compileresult_fields_to_msgpack(MsgpackWriter &w_, const CompileResult &x_) {
  if (x_.binary) {
    w_.writeArrayHeader((*x_.binary).size());
    for (const auto &v1_ : (*x_.binary)) {
      w_.writeInt32(static_cast<int32_t>(v1_));
    }
  } else {
    w_.writeNil();
  }
  w_.writeArrayHeader(x_.features.size());
  for (const auto &v0_ : x_.features) {
    w_.writeString(v0_);
  }
  w_.writeArrayHeader(x_.events.size());
  for (const auto &v0_ : x_.events) {
    compileevent_to_msgpack(w_, v0_);
  }
  w_.writeArrayHeader(x_.layouts.size());
  for (const auto &v0_ : x_.layouts) {
    structlayout_to_msgpack(w_, v0_);
  }
  w_.writeString(x_.messages);
  w_.writeArrayHeader(x_.entryArgs.size());
  for (const auto &v0_ : x_.entryArgs) {
    named_to_msgpack(w_, v0_);
  }
}

CompileResult compileresult_from_msgpack(MsgpackReader &r_) {
  const auto n_ = r_.readArrayHeader();
  return compileresult_fields_from_msgpack(r_, n_);
}

void compileresult_to_msgpack(MsgpackWriter &w_, const CompileResult &x_) {
  w_.writeArrayHeader(6);
  compileresult_fields_to_msgpack(w_, x_);
}

Interface interface_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 3) throw std::runtime_error("Expected Interface with 3 field(s)");
  auto name = sym_from_msgpack(r_);
  std::vector<FunctionDecl> declarations;
  {
    const auto declarations_size = r_.readArrayHeader();
    declarations.reserve(declarations_size);
    for (size_t declarations_idx = 0; declarations_idx < declarations_size; ++declarations_idx) {
      auto declarations_elem = functiondecl_from_msgpack(r_);
      declarations.emplace_back(std::move(declarations_elem));
    }
  }
  std::vector<MetaEntry> metadata;
  {
    const auto metadata_size = r_.readArrayHeader();
    metadata.reserve(metadata_size);
    for (size_t metadata_idx = 0; metadata_idx < metadata_size; ++metadata_idx) {
      auto metadata_elem = metaentry_from_msgpack(r_);
      metadata.emplace_back(std::move(metadata_elem));
    }
  }
  return {std::move(name), std::move(declarations), std::move(metadata)};
}

void interface_fields_to_msgpack(MsgpackWriter &w_, const Interface &x_) {
  sym_to_msgpack(w_, x_.name);
  w_.writeArrayHeader(x_.declarations.size());
  for (const auto &v0_ : x_.declarations) {
    functiondecl_to_msgpack(w_, v0_);
  }
  w_.writeArrayHeader(x_.metadata.size());
  for (const auto &v0_ : x_.metadata) {
    metaentry_to_msgpack(w_, v0_);
  }
}

Interface interface_from_msgpack(MsgpackReader &r_) {
  const auto n_ = r_.readArrayHeader();
  return interface_fields_from_msgpack(r_, n_);
}

void interface_to_msgpack(MsgpackWriter &w_, const Interface &x_) {
  w_.writeArrayHeader(3);
  interface_fields_to_msgpack(w_, x_);
}

Package package_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected Package with 2 field(s)");
  auto interface = interface_from_msgpack(r_);
  auto program = program_from_msgpack(r_);
  return {std::move(interface), std::move(program)};
}

void package_fields_to_msgpack(MsgpackWriter &w_, const Package &x_) {
  interface_to_msgpack(w_, x_.interface);
  program_to_msgpack(w_, x_.program);
}

Package package_from_msgpack(MsgpackReader &r_) {
  const auto n_ = r_.readArrayHeader();
  return package_fields_from_msgpack(r_, n_);
}

void package_to_msgpack(MsgpackWriter &w_, const Package &x_) {
  w_.writeArrayHeader(2);
  package_fields_to_msgpack(w_, x_);
}

ProgramTypeSize programtypesize_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected ProgramTypeSize with 2 field(s)");
  auto tpe = Type::any_from_msgpack(r_);
  const auto sizeInBytes = r_.readInt32();
  return {std::move(tpe), sizeInBytes};
}

void programtypesize_fields_to_msgpack(MsgpackWriter &w_, const ProgramTypeSize &x_) {
  Type::any_to_msgpack(w_, x_.tpe);
  w_.writeInt32(x_.sizeInBytes);
}

ProgramTypeSize programtypesize_from_msgpack(MsgpackReader &r_) {
  const auto n_ = r_.readArrayHeader();
  return programtypesize_fields_from_msgpack(r_, n_);
}

void programtypesize_to_msgpack(MsgpackWriter &w_, const ProgramTypeSize &x_) {
  w_.writeArrayHeader(2);
  programtypesize_fields_to_msgpack(w_, x_);
}

ProgramLinkRequest programlinkrequest_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 6) throw std::runtime_error("Expected ProgramLinkRequest with 6 field(s)");
  std::vector<Package> packages;
  {
    const auto packages_size = r_.readArrayHeader();
    packages.reserve(packages_size);
    for (size_t packages_idx = 0; packages_idx < packages_size; ++packages_idx) {
      auto packages_elem = package_from_msgpack(r_);
      packages.emplace_back(std::move(packages_elem));
    }
  }
  auto consumer = program_from_msgpack(r_);
  std::vector<std::string> capabilities;
  {
    const auto capabilities_size = r_.readArrayHeader();
    capabilities.reserve(capabilities_size);
    for (size_t capabilities_idx = 0; capabilities_idx < capabilities_size; ++capabilities_idx) {
      auto capabilities_elem = r_.readString();
      capabilities.emplace_back(std::move(capabilities_elem));
    }
  }
  std::vector<ProgramTypeSize> typeSizes;
  {
    const auto typeSizes_size = r_.readArrayHeader();
    typeSizes.reserve(typeSizes_size);
    for (size_t typeSizes_idx = 0; typeSizes_idx < typeSizes_size; ++typeSizes_idx) {
      auto typeSizes_elem = programtypesize_from_msgpack(r_);
      typeSizes.emplace_back(std::move(typeSizes_elem));
    }
  }
  std::optional<int32_t> stackDepth;
  if (!r_.tryReadNil()) {
    const auto stackDepth_value = r_.readInt32();
    stackDepth = stackDepth_value;
  }
  const auto logicalAddressing = r_.readBoolean();
  return {std::move(packages), std::move(consumer), std::move(capabilities), std::move(typeSizes), stackDepth, logicalAddressing};
}

void programlinkrequest_fields_to_msgpack(MsgpackWriter &w_, const ProgramLinkRequest &x_) {
  w_.writeArrayHeader(x_.packages.size());
  for (const auto &v0_ : x_.packages) {
    package_to_msgpack(w_, v0_);
  }
  program_to_msgpack(w_, x_.consumer);
  w_.writeArrayHeader(x_.capabilities.size());
  for (const auto &v0_ : x_.capabilities) {
    w_.writeString(v0_);
  }
  w_.writeArrayHeader(x_.typeSizes.size());
  for (const auto &v0_ : x_.typeSizes) {
    programtypesize_to_msgpack(w_, v0_);
  }
  if (x_.stackDepth) {
    w_.writeInt32((*x_.stackDepth));
  } else {
    w_.writeNil();
  }
  w_.writeBoolean(x_.logicalAddressing);
}

ProgramLinkRequest programlinkrequest_from_msgpack(MsgpackReader &r_) {
  const auto n_ = r_.readArrayHeader();
  return programlinkrequest_fields_from_msgpack(r_, n_);
}

void programlinkrequest_to_msgpack(MsgpackWriter &w_, const ProgramLinkRequest &x_) {
  w_.writeArrayHeader(6);
  programlinkrequest_fields_to_msgpack(w_, x_);
}

PackageFragment packagefragment_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected PackageFragment with 2 field(s)");
  auto identity = r_.readString();
  auto program = program_from_msgpack(r_);
  return {std::move(identity), std::move(program)};
}

void packagefragment_fields_to_msgpack(MsgpackWriter &w_, const PackageFragment &x_) {
  w_.writeString(x_.identity);
  program_to_msgpack(w_, x_.program);
}

PackageFragment packagefragment_from_msgpack(MsgpackReader &r_) {
  const auto n_ = r_.readArrayHeader();
  return packagefragment_fields_from_msgpack(r_, n_);
}

void packagefragment_to_msgpack(MsgpackWriter &w_, const PackageFragment &x_) {
  w_.writeArrayHeader(2);
  packagefragment_fields_to_msgpack(w_, x_);
}

PackageLinkRequest packagelinkrequest_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 4) throw std::runtime_error("Expected PackageLinkRequest with 4 field(s)");
  auto interface = interface_from_msgpack(r_);
  std::vector<PackageFragment> fragments;
  {
    const auto fragments_size = r_.readArrayHeader();
    fragments.reserve(fragments_size);
    for (size_t fragments_idx = 0; fragments_idx < fragments_size; ++fragments_idx) {
      auto fragments_elem = packagefragment_from_msgpack(r_);
      fragments.emplace_back(std::move(fragments_elem));
    }
  }
  std::vector<std::string> capabilities;
  {
    const auto capabilities_size = r_.readArrayHeader();
    capabilities.reserve(capabilities_size);
    for (size_t capabilities_idx = 0; capabilities_idx < capabilities_size; ++capabilities_idx) {
      auto capabilities_elem = r_.readString();
      capabilities.emplace_back(std::move(capabilities_elem));
    }
  }
  const auto pruneUnimplementedDeclarations = r_.readBoolean();
  return {std::move(interface), std::move(fragments), std::move(capabilities), pruneUnimplementedDeclarations};
}

void packagelinkrequest_fields_to_msgpack(MsgpackWriter &w_, const PackageLinkRequest &x_) {
  interface_to_msgpack(w_, x_.interface);
  w_.writeArrayHeader(x_.fragments.size());
  for (const auto &v0_ : x_.fragments) {
    packagefragment_to_msgpack(w_, v0_);
  }
  w_.writeArrayHeader(x_.capabilities.size());
  for (const auto &v0_ : x_.capabilities) {
    w_.writeString(v0_);
  }
  w_.writeBoolean(x_.pruneUnimplementedDeclarations);
}

PackageLinkRequest packagelinkrequest_from_msgpack(MsgpackReader &r_) {
  const auto n_ = r_.readArrayHeader();
  return packagelinkrequest_fields_from_msgpack(r_, n_);
}

void packagelinkrequest_to_msgpack(MsgpackWriter &w_, const PackageLinkRequest &x_) {
  w_.writeArrayHeader(4);
  packagelinkrequest_fields_to_msgpack(w_, x_);
}

CompileModule compilemodule_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 5) throw std::runtime_error("Expected CompileModule with 5 field(s)");
  auto moduleName = r_.readString();
  const auto format = r_.readInt32();
  const auto kind = r_.readInt32();
  std::vector<std::string> features;
  {
    const auto features_size = r_.readArrayHeader();
    features.reserve(features_size);
    for (size_t features_idx = 0; features_idx < features_size; ++features_idx) {
      auto features_elem = r_.readString();
      features.emplace_back(std::move(features_elem));
    }
  }
  std::vector<int8_t> image;
  {
    const auto image_size = r_.readArrayHeader();
    image.reserve(image_size);
    for (size_t image_idx = 0; image_idx < image_size; ++image_idx) {
      const auto image_elem = static_cast<int8_t>(r_.readInt32());
      image.emplace_back(image_elem);
    }
  }
  return {std::move(moduleName), format, kind, std::move(features), std::move(image)};
}

void compilemodule_fields_to_msgpack(MsgpackWriter &w_, const CompileModule &x_) {
  w_.writeString(x_.moduleName);
  w_.writeInt32(x_.format);
  w_.writeInt32(x_.kind);
  w_.writeArrayHeader(x_.features.size());
  for (const auto &v0_ : x_.features) {
    w_.writeString(v0_);
  }
  w_.writeArrayHeader(x_.image.size());
  for (const auto &v0_ : x_.image) {
    w_.writeInt32(static_cast<int32_t>(v0_));
  }
}

CompileModule compilemodule_from_msgpack(MsgpackReader &r_) {
  const auto n_ = r_.readArrayHeader();
  return compilemodule_fields_from_msgpack(r_, n_);
}

void compilemodule_to_msgpack(MsgpackWriter &w_, const CompileModule &x_) {
  w_.writeArrayHeader(5);
  compilemodule_fields_to_msgpack(w_, x_);
}

CompileRoot compileroot_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 4) throw std::runtime_error("Expected CompileRoot with 4 field(s)");
  auto name = r_.readString();
  std::vector<std::string> modules;
  {
    const auto modules_size = r_.readArrayHeader();
    modules.reserve(modules_size);
    for (size_t modules_idx = 0; modules_idx < modules_size; ++modules_idx) {
      auto modules_elem = r_.readString();
      modules.emplace_back(std::move(modules_elem));
    }
  }
  std::vector<std::string> implementations;
  {
    const auto implementations_size = r_.readArrayHeader();
    implementations.reserve(implementations_size);
    for (size_t implementations_idx = 0; implementations_idx < implementations_size; ++implementations_idx) {
      auto implementations_elem = r_.readString();
      implementations.emplace_back(std::move(implementations_elem));
    }
  }
  std::vector<std::string> unfit;
  {
    const auto unfit_size = r_.readArrayHeader();
    unfit.reserve(unfit_size);
    for (size_t unfit_idx = 0; unfit_idx < unfit_size; ++unfit_idx) {
      auto unfit_elem = r_.readString();
      unfit.emplace_back(std::move(unfit_elem));
    }
  }
  return {std::move(name), std::move(modules), std::move(implementations), std::move(unfit)};
}

void compileroot_fields_to_msgpack(MsgpackWriter &w_, const CompileRoot &x_) {
  w_.writeString(x_.name);
  w_.writeArrayHeader(x_.modules.size());
  for (const auto &v0_ : x_.modules) {
    w_.writeString(v0_);
  }
  w_.writeArrayHeader(x_.implementations.size());
  for (const auto &v0_ : x_.implementations) {
    w_.writeString(v0_);
  }
  w_.writeArrayHeader(x_.unfit.size());
  for (const auto &v0_ : x_.unfit) {
    w_.writeString(v0_);
  }
}

CompileRoot compileroot_from_msgpack(MsgpackReader &r_) {
  const auto n_ = r_.readArrayHeader();
  return compileroot_fields_from_msgpack(r_, n_);
}

void compileroot_to_msgpack(MsgpackWriter &w_, const CompileRoot &x_) {
  w_.writeArrayHeader(4);
  compileroot_fields_to_msgpack(w_, x_);
}

CompileBundle compilebundle_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 5) throw std::runtime_error("Expected CompileBundle with 5 field(s)");
  std::vector<int8_t> hostObject;
  {
    const auto hostObject_size = r_.readArrayHeader();
    hostObject.reserve(hostObject_size);
    for (size_t hostObject_idx = 0; hostObject_idx < hostObject_size; ++hostObject_idx) {
      const auto hostObject_elem = static_cast<int8_t>(r_.readInt32());
      hostObject.emplace_back(hostObject_elem);
    }
  }
  std::vector<CompileModule> remoteModules;
  {
    const auto remoteModules_size = r_.readArrayHeader();
    remoteModules.reserve(remoteModules_size);
    for (size_t remoteModules_idx = 0; remoteModules_idx < remoteModules_size; ++remoteModules_idx) {
      auto remoteModules_elem = compilemodule_from_msgpack(r_);
      remoteModules.emplace_back(std::move(remoteModules_elem));
    }
  }
  std::vector<std::string> implementations;
  {
    const auto implementations_size = r_.readArrayHeader();
    implementations.reserve(implementations_size);
    for (size_t implementations_idx = 0; implementations_idx < implementations_size; ++implementations_idx) {
      auto implementations_elem = r_.readString();
      implementations.emplace_back(std::move(implementations_elem));
    }
  }
  std::vector<std::string> unfit;
  {
    const auto unfit_size = r_.readArrayHeader();
    unfit.reserve(unfit_size);
    for (size_t unfit_idx = 0; unfit_idx < unfit_size; ++unfit_idx) {
      auto unfit_elem = r_.readString();
      unfit.emplace_back(std::move(unfit_elem));
    }
  }
  std::vector<CompileRoot> roots;
  {
    const auto roots_size = r_.readArrayHeader();
    roots.reserve(roots_size);
    for (size_t roots_idx = 0; roots_idx < roots_size; ++roots_idx) {
      auto roots_elem = compileroot_from_msgpack(r_);
      roots.emplace_back(std::move(roots_elem));
    }
  }
  return {std::move(hostObject), std::move(remoteModules), std::move(implementations), std::move(unfit), std::move(roots)};
}

void compilebundle_fields_to_msgpack(MsgpackWriter &w_, const CompileBundle &x_) {
  w_.writeArrayHeader(x_.hostObject.size());
  for (const auto &v0_ : x_.hostObject) {
    w_.writeInt32(static_cast<int32_t>(v0_));
  }
  w_.writeArrayHeader(x_.remoteModules.size());
  for (const auto &v0_ : x_.remoteModules) {
    compilemodule_to_msgpack(w_, v0_);
  }
  w_.writeArrayHeader(x_.implementations.size());
  for (const auto &v0_ : x_.implementations) {
    w_.writeString(v0_);
  }
  w_.writeArrayHeader(x_.unfit.size());
  for (const auto &v0_ : x_.unfit) {
    w_.writeString(v0_);
  }
  w_.writeArrayHeader(x_.roots.size());
  for (const auto &v0_ : x_.roots) {
    compileroot_to_msgpack(w_, v0_);
  }
}

CompileBundle compilebundle_from_msgpack(MsgpackReader &r_) {
  const auto n_ = r_.readArrayHeader();
  return compilebundle_fields_from_msgpack(r_, n_);
}

void compilebundle_to_msgpack(MsgpackWriter &w_, const CompileBundle &x_) {
  w_.writeArrayHeader(5);
  compilebundle_fields_to_msgpack(w_, x_);
}
} // namespace
namespace Intr { namespace {

Intr::BNot bnot_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected Intr::BNot with 2 field(s)");
  auto x = Term::any_from_msgpack(r_);
  auto rtn = Type::any_from_msgpack(r_);
  return {std::move(x), std::move(rtn)};
}

void bnot_fields_to_msgpack(MsgpackWriter &w_, const Intr::BNot &x_) {
  Term::any_to_msgpack(w_, x_.x);
  Type::any_to_msgpack(w_, x_.rtn);
}

Intr::LogicNot logicnot_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 1) throw std::runtime_error("Expected Intr::LogicNot with 1 field(s)");
  auto x = Term::any_from_msgpack(r_);
  return Intr::LogicNot(std::move(x));
}

void logicnot_fields_to_msgpack(MsgpackWriter &w_, const Intr::LogicNot &x_) { Term::any_to_msgpack(w_, x_.x); }

Intr::Pos pos_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected Intr::Pos with 2 field(s)");
  auto x = Term::any_from_msgpack(r_);
  auto rtn = Type::any_from_msgpack(r_);
  return {std::move(x), std::move(rtn)};
}

void pos_fields_to_msgpack(MsgpackWriter &w_, const Intr::Pos &x_) {
  Term::any_to_msgpack(w_, x_.x);
  Type::any_to_msgpack(w_, x_.rtn);
}

Intr::Neg neg_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected Intr::Neg with 2 field(s)");
  auto x = Term::any_from_msgpack(r_);
  auto rtn = Type::any_from_msgpack(r_);
  return {std::move(x), std::move(rtn)};
}

void neg_fields_to_msgpack(MsgpackWriter &w_, const Intr::Neg &x_) {
  Term::any_to_msgpack(w_, x_.x);
  Type::any_to_msgpack(w_, x_.rtn);
}

Intr::Add add_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 3) throw std::runtime_error("Expected Intr::Add with 3 field(s)");
  auto x = Term::any_from_msgpack(r_);
  auto y = Term::any_from_msgpack(r_);
  auto rtn = Type::any_from_msgpack(r_);
  return {std::move(x), std::move(y), std::move(rtn)};
}

void add_fields_to_msgpack(MsgpackWriter &w_, const Intr::Add &x_) {
  Term::any_to_msgpack(w_, x_.x);
  Term::any_to_msgpack(w_, x_.y);
  Type::any_to_msgpack(w_, x_.rtn);
}

Intr::Sub sub_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 3) throw std::runtime_error("Expected Intr::Sub with 3 field(s)");
  auto x = Term::any_from_msgpack(r_);
  auto y = Term::any_from_msgpack(r_);
  auto rtn = Type::any_from_msgpack(r_);
  return {std::move(x), std::move(y), std::move(rtn)};
}

void sub_fields_to_msgpack(MsgpackWriter &w_, const Intr::Sub &x_) {
  Term::any_to_msgpack(w_, x_.x);
  Term::any_to_msgpack(w_, x_.y);
  Type::any_to_msgpack(w_, x_.rtn);
}

Intr::Mul mul_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 3) throw std::runtime_error("Expected Intr::Mul with 3 field(s)");
  auto x = Term::any_from_msgpack(r_);
  auto y = Term::any_from_msgpack(r_);
  auto rtn = Type::any_from_msgpack(r_);
  return {std::move(x), std::move(y), std::move(rtn)};
}

void mul_fields_to_msgpack(MsgpackWriter &w_, const Intr::Mul &x_) {
  Term::any_to_msgpack(w_, x_.x);
  Term::any_to_msgpack(w_, x_.y);
  Type::any_to_msgpack(w_, x_.rtn);
}

Intr::Div div_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 3) throw std::runtime_error("Expected Intr::Div with 3 field(s)");
  auto x = Term::any_from_msgpack(r_);
  auto y = Term::any_from_msgpack(r_);
  auto rtn = Type::any_from_msgpack(r_);
  return {std::move(x), std::move(y), std::move(rtn)};
}

void div_fields_to_msgpack(MsgpackWriter &w_, const Intr::Div &x_) {
  Term::any_to_msgpack(w_, x_.x);
  Term::any_to_msgpack(w_, x_.y);
  Type::any_to_msgpack(w_, x_.rtn);
}

Intr::Rem rem_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 3) throw std::runtime_error("Expected Intr::Rem with 3 field(s)");
  auto x = Term::any_from_msgpack(r_);
  auto y = Term::any_from_msgpack(r_);
  auto rtn = Type::any_from_msgpack(r_);
  return {std::move(x), std::move(y), std::move(rtn)};
}

void rem_fields_to_msgpack(MsgpackWriter &w_, const Intr::Rem &x_) {
  Term::any_to_msgpack(w_, x_.x);
  Term::any_to_msgpack(w_, x_.y);
  Type::any_to_msgpack(w_, x_.rtn);
}

Intr::Min min_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 3) throw std::runtime_error("Expected Intr::Min with 3 field(s)");
  auto x = Term::any_from_msgpack(r_);
  auto y = Term::any_from_msgpack(r_);
  auto rtn = Type::any_from_msgpack(r_);
  return {std::move(x), std::move(y), std::move(rtn)};
}

void min_fields_to_msgpack(MsgpackWriter &w_, const Intr::Min &x_) {
  Term::any_to_msgpack(w_, x_.x);
  Term::any_to_msgpack(w_, x_.y);
  Type::any_to_msgpack(w_, x_.rtn);
}

Intr::Max max_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 3) throw std::runtime_error("Expected Intr::Max with 3 field(s)");
  auto x = Term::any_from_msgpack(r_);
  auto y = Term::any_from_msgpack(r_);
  auto rtn = Type::any_from_msgpack(r_);
  return {std::move(x), std::move(y), std::move(rtn)};
}

void max_fields_to_msgpack(MsgpackWriter &w_, const Intr::Max &x_) {
  Term::any_to_msgpack(w_, x_.x);
  Term::any_to_msgpack(w_, x_.y);
  Type::any_to_msgpack(w_, x_.rtn);
}

Intr::BAnd band_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 3) throw std::runtime_error("Expected Intr::BAnd with 3 field(s)");
  auto x = Term::any_from_msgpack(r_);
  auto y = Term::any_from_msgpack(r_);
  auto rtn = Type::any_from_msgpack(r_);
  return {std::move(x), std::move(y), std::move(rtn)};
}

void band_fields_to_msgpack(MsgpackWriter &w_, const Intr::BAnd &x_) {
  Term::any_to_msgpack(w_, x_.x);
  Term::any_to_msgpack(w_, x_.y);
  Type::any_to_msgpack(w_, x_.rtn);
}

Intr::BOr bor_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 3) throw std::runtime_error("Expected Intr::BOr with 3 field(s)");
  auto x = Term::any_from_msgpack(r_);
  auto y = Term::any_from_msgpack(r_);
  auto rtn = Type::any_from_msgpack(r_);
  return {std::move(x), std::move(y), std::move(rtn)};
}

void bor_fields_to_msgpack(MsgpackWriter &w_, const Intr::BOr &x_) {
  Term::any_to_msgpack(w_, x_.x);
  Term::any_to_msgpack(w_, x_.y);
  Type::any_to_msgpack(w_, x_.rtn);
}

Intr::BXor bxor_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 3) throw std::runtime_error("Expected Intr::BXor with 3 field(s)");
  auto x = Term::any_from_msgpack(r_);
  auto y = Term::any_from_msgpack(r_);
  auto rtn = Type::any_from_msgpack(r_);
  return {std::move(x), std::move(y), std::move(rtn)};
}

void bxor_fields_to_msgpack(MsgpackWriter &w_, const Intr::BXor &x_) {
  Term::any_to_msgpack(w_, x_.x);
  Term::any_to_msgpack(w_, x_.y);
  Type::any_to_msgpack(w_, x_.rtn);
}

Intr::BSL bsl_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 3) throw std::runtime_error("Expected Intr::BSL with 3 field(s)");
  auto x = Term::any_from_msgpack(r_);
  auto y = Term::any_from_msgpack(r_);
  auto rtn = Type::any_from_msgpack(r_);
  return {std::move(x), std::move(y), std::move(rtn)};
}

void bsl_fields_to_msgpack(MsgpackWriter &w_, const Intr::BSL &x_) {
  Term::any_to_msgpack(w_, x_.x);
  Term::any_to_msgpack(w_, x_.y);
  Type::any_to_msgpack(w_, x_.rtn);
}

Intr::BSR bsr_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 3) throw std::runtime_error("Expected Intr::BSR with 3 field(s)");
  auto x = Term::any_from_msgpack(r_);
  auto y = Term::any_from_msgpack(r_);
  auto rtn = Type::any_from_msgpack(r_);
  return {std::move(x), std::move(y), std::move(rtn)};
}

void bsr_fields_to_msgpack(MsgpackWriter &w_, const Intr::BSR &x_) {
  Term::any_to_msgpack(w_, x_.x);
  Term::any_to_msgpack(w_, x_.y);
  Type::any_to_msgpack(w_, x_.rtn);
}

Intr::BZSR bzsr_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 3) throw std::runtime_error("Expected Intr::BZSR with 3 field(s)");
  auto x = Term::any_from_msgpack(r_);
  auto y = Term::any_from_msgpack(r_);
  auto rtn = Type::any_from_msgpack(r_);
  return {std::move(x), std::move(y), std::move(rtn)};
}

void bzsr_fields_to_msgpack(MsgpackWriter &w_, const Intr::BZSR &x_) {
  Term::any_to_msgpack(w_, x_.x);
  Term::any_to_msgpack(w_, x_.y);
  Type::any_to_msgpack(w_, x_.rtn);
}

Intr::PopCount popcount_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected Intr::PopCount with 2 field(s)");
  auto x = Term::any_from_msgpack(r_);
  auto rtn = Type::any_from_msgpack(r_);
  return {std::move(x), std::move(rtn)};
}

void popcount_fields_to_msgpack(MsgpackWriter &w_, const Intr::PopCount &x_) {
  Term::any_to_msgpack(w_, x_.x);
  Type::any_to_msgpack(w_, x_.rtn);
}

Intr::LogicAnd logicand_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected Intr::LogicAnd with 2 field(s)");
  auto x = Term::any_from_msgpack(r_);
  auto y = Term::any_from_msgpack(r_);
  return {std::move(x), std::move(y)};
}

void logicand_fields_to_msgpack(MsgpackWriter &w_, const Intr::LogicAnd &x_) {
  Term::any_to_msgpack(w_, x_.x);
  Term::any_to_msgpack(w_, x_.y);
}

Intr::LogicOr logicor_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected Intr::LogicOr with 2 field(s)");
  auto x = Term::any_from_msgpack(r_);
  auto y = Term::any_from_msgpack(r_);
  return {std::move(x), std::move(y)};
}

void logicor_fields_to_msgpack(MsgpackWriter &w_, const Intr::LogicOr &x_) {
  Term::any_to_msgpack(w_, x_.x);
  Term::any_to_msgpack(w_, x_.y);
}

Intr::LogicEq logiceq_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected Intr::LogicEq with 2 field(s)");
  auto x = Term::any_from_msgpack(r_);
  auto y = Term::any_from_msgpack(r_);
  return {std::move(x), std::move(y)};
}

void logiceq_fields_to_msgpack(MsgpackWriter &w_, const Intr::LogicEq &x_) {
  Term::any_to_msgpack(w_, x_.x);
  Term::any_to_msgpack(w_, x_.y);
}

Intr::LogicNeq logicneq_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected Intr::LogicNeq with 2 field(s)");
  auto x = Term::any_from_msgpack(r_);
  auto y = Term::any_from_msgpack(r_);
  return {std::move(x), std::move(y)};
}

void logicneq_fields_to_msgpack(MsgpackWriter &w_, const Intr::LogicNeq &x_) {
  Term::any_to_msgpack(w_, x_.x);
  Term::any_to_msgpack(w_, x_.y);
}

Intr::LogicLte logiclte_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected Intr::LogicLte with 2 field(s)");
  auto x = Term::any_from_msgpack(r_);
  auto y = Term::any_from_msgpack(r_);
  return {std::move(x), std::move(y)};
}

void logiclte_fields_to_msgpack(MsgpackWriter &w_, const Intr::LogicLte &x_) {
  Term::any_to_msgpack(w_, x_.x);
  Term::any_to_msgpack(w_, x_.y);
}

Intr::LogicGte logicgte_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected Intr::LogicGte with 2 field(s)");
  auto x = Term::any_from_msgpack(r_);
  auto y = Term::any_from_msgpack(r_);
  return {std::move(x), std::move(y)};
}

void logicgte_fields_to_msgpack(MsgpackWriter &w_, const Intr::LogicGte &x_) {
  Term::any_to_msgpack(w_, x_.x);
  Term::any_to_msgpack(w_, x_.y);
}

Intr::LogicLt logiclt_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected Intr::LogicLt with 2 field(s)");
  auto x = Term::any_from_msgpack(r_);
  auto y = Term::any_from_msgpack(r_);
  return {std::move(x), std::move(y)};
}

void logiclt_fields_to_msgpack(MsgpackWriter &w_, const Intr::LogicLt &x_) {
  Term::any_to_msgpack(w_, x_.x);
  Term::any_to_msgpack(w_, x_.y);
}

Intr::LogicGt logicgt_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected Intr::LogicGt with 2 field(s)");
  auto x = Term::any_from_msgpack(r_);
  auto y = Term::any_from_msgpack(r_);
  return {std::move(x), std::move(y)};
}

void logicgt_fields_to_msgpack(MsgpackWriter &w_, const Intr::LogicGt &x_) {
  Term::any_to_msgpack(w_, x_.x);
  Term::any_to_msgpack(w_, x_.y);
}

Intr::Any any_from_msgpack(MsgpackReader &r_) {
  if (r_.nextIsArray()) {
    const auto n_ = r_.readArrayHeader();
    if (n_ == 0) throw std::runtime_error("Expected non-empty sum payload");
    const auto ord_ = r_.readInt32();
    switch (ord_) {
      case 0: return Intr::bnot_fields_from_msgpack(r_, n_ - 1);
      case 1: return Intr::logicnot_fields_from_msgpack(r_, n_ - 1);
      case 2: return Intr::pos_fields_from_msgpack(r_, n_ - 1);
      case 3: return Intr::neg_fields_from_msgpack(r_, n_ - 1);
      case 4: return Intr::add_fields_from_msgpack(r_, n_ - 1);
      case 5: return Intr::sub_fields_from_msgpack(r_, n_ - 1);
      case 6: return Intr::mul_fields_from_msgpack(r_, n_ - 1);
      case 7: return Intr::div_fields_from_msgpack(r_, n_ - 1);
      case 8: return Intr::rem_fields_from_msgpack(r_, n_ - 1);
      case 9: return Intr::min_fields_from_msgpack(r_, n_ - 1);
      case 10: return Intr::max_fields_from_msgpack(r_, n_ - 1);
      case 11: return Intr::band_fields_from_msgpack(r_, n_ - 1);
      case 12: return Intr::bor_fields_from_msgpack(r_, n_ - 1);
      case 13: return Intr::bxor_fields_from_msgpack(r_, n_ - 1);
      case 14: return Intr::bsl_fields_from_msgpack(r_, n_ - 1);
      case 15: return Intr::bsr_fields_from_msgpack(r_, n_ - 1);
      case 16: return Intr::bzsr_fields_from_msgpack(r_, n_ - 1);
      case 17: return Intr::popcount_fields_from_msgpack(r_, n_ - 1);
      case 18: return Intr::logicand_fields_from_msgpack(r_, n_ - 1);
      case 19: return Intr::logicor_fields_from_msgpack(r_, n_ - 1);
      case 20: return Intr::logiceq_fields_from_msgpack(r_, n_ - 1);
      case 21: return Intr::logicneq_fields_from_msgpack(r_, n_ - 1);
      case 22: return Intr::logiclte_fields_from_msgpack(r_, n_ - 1);
      case 23: return Intr::logicgte_fields_from_msgpack(r_, n_ - 1);
      case 24: return Intr::logiclt_fields_from_msgpack(r_, n_ - 1);
      case 25: return Intr::logicgt_fields_from_msgpack(r_, n_ - 1);
      default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
    }
  } else {
    const auto ord_ = r_.readInt32();
    switch (ord_) {
      case 0:
      case 1:
      case 2:
      case 3:
      case 4:
      case 5:
      case 6:
      case 7:
      case 8:
      case 9:
      case 10:
      case 11:
      case 12:
      case 13:
      case 14:
      case 15:
      case 16:
      case 17:
      case 18:
      case 19:
      case 20:
      case 21:
      case 22:
      case 23:
      case 24:
      case 25: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
    }
  }
}

void any_to_msgpack(MsgpackWriter &w_, const Intr::Any &x_) {
  x_.match_total(
      [&](const Intr::BNot &y_) -> void {
        w_.writeArrayHeader(3);
        w_.writeInt32(0);
        Intr::bnot_fields_to_msgpack(w_, y_);
      },
      [&](const Intr::LogicNot &y_) -> void {
        w_.writeArrayHeader(2);
        w_.writeInt32(1);
        Intr::logicnot_fields_to_msgpack(w_, y_);
      },
      [&](const Intr::Pos &y_) -> void {
        w_.writeArrayHeader(3);
        w_.writeInt32(2);
        Intr::pos_fields_to_msgpack(w_, y_);
      },
      [&](const Intr::Neg &y_) -> void {
        w_.writeArrayHeader(3);
        w_.writeInt32(3);
        Intr::neg_fields_to_msgpack(w_, y_);
      },
      [&](const Intr::Add &y_) -> void {
        w_.writeArrayHeader(4);
        w_.writeInt32(4);
        Intr::add_fields_to_msgpack(w_, y_);
      },
      [&](const Intr::Sub &y_) -> void {
        w_.writeArrayHeader(4);
        w_.writeInt32(5);
        Intr::sub_fields_to_msgpack(w_, y_);
      },
      [&](const Intr::Mul &y_) -> void {
        w_.writeArrayHeader(4);
        w_.writeInt32(6);
        Intr::mul_fields_to_msgpack(w_, y_);
      },
      [&](const Intr::Div &y_) -> void {
        w_.writeArrayHeader(4);
        w_.writeInt32(7);
        Intr::div_fields_to_msgpack(w_, y_);
      },
      [&](const Intr::Rem &y_) -> void {
        w_.writeArrayHeader(4);
        w_.writeInt32(8);
        Intr::rem_fields_to_msgpack(w_, y_);
      },
      [&](const Intr::Min &y_) -> void {
        w_.writeArrayHeader(4);
        w_.writeInt32(9);
        Intr::min_fields_to_msgpack(w_, y_);
      },
      [&](const Intr::Max &y_) -> void {
        w_.writeArrayHeader(4);
        w_.writeInt32(10);
        Intr::max_fields_to_msgpack(w_, y_);
      },
      [&](const Intr::BAnd &y_) -> void {
        w_.writeArrayHeader(4);
        w_.writeInt32(11);
        Intr::band_fields_to_msgpack(w_, y_);
      },
      [&](const Intr::BOr &y_) -> void {
        w_.writeArrayHeader(4);
        w_.writeInt32(12);
        Intr::bor_fields_to_msgpack(w_, y_);
      },
      [&](const Intr::BXor &y_) -> void {
        w_.writeArrayHeader(4);
        w_.writeInt32(13);
        Intr::bxor_fields_to_msgpack(w_, y_);
      },
      [&](const Intr::BSL &y_) -> void {
        w_.writeArrayHeader(4);
        w_.writeInt32(14);
        Intr::bsl_fields_to_msgpack(w_, y_);
      },
      [&](const Intr::BSR &y_) -> void {
        w_.writeArrayHeader(4);
        w_.writeInt32(15);
        Intr::bsr_fields_to_msgpack(w_, y_);
      },
      [&](const Intr::BZSR &y_) -> void {
        w_.writeArrayHeader(4);
        w_.writeInt32(16);
        Intr::bzsr_fields_to_msgpack(w_, y_);
      },
      [&](const Intr::PopCount &y_) -> void {
        w_.writeArrayHeader(3);
        w_.writeInt32(17);
        Intr::popcount_fields_to_msgpack(w_, y_);
      },
      [&](const Intr::LogicAnd &y_) -> void {
        w_.writeArrayHeader(3);
        w_.writeInt32(18);
        Intr::logicand_fields_to_msgpack(w_, y_);
      },
      [&](const Intr::LogicOr &y_) -> void {
        w_.writeArrayHeader(3);
        w_.writeInt32(19);
        Intr::logicor_fields_to_msgpack(w_, y_);
      },
      [&](const Intr::LogicEq &y_) -> void {
        w_.writeArrayHeader(3);
        w_.writeInt32(20);
        Intr::logiceq_fields_to_msgpack(w_, y_);
      },
      [&](const Intr::LogicNeq &y_) -> void {
        w_.writeArrayHeader(3);
        w_.writeInt32(21);
        Intr::logicneq_fields_to_msgpack(w_, y_);
      },
      [&](const Intr::LogicLte &y_) -> void {
        w_.writeArrayHeader(3);
        w_.writeInt32(22);
        Intr::logiclte_fields_to_msgpack(w_, y_);
      },
      [&](const Intr::LogicGte &y_) -> void {
        w_.writeArrayHeader(3);
        w_.writeInt32(23);
        Intr::logicgte_fields_to_msgpack(w_, y_);
      },
      [&](const Intr::LogicLt &y_) -> void {
        w_.writeArrayHeader(3);
        w_.writeInt32(24);
        Intr::logiclt_fields_to_msgpack(w_, y_);
      },
      [&](const Intr::LogicGt &y_) -> void {
        w_.writeArrayHeader(3);
        w_.writeInt32(25);
        Intr::logicgt_fields_to_msgpack(w_, y_);
      });
}
}} // namespace Intr
namespace AtomicOp { namespace {

AtomicOp::Xchg xchg_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected AtomicOp::Xchg with 0 field(s)");
  return {};
}

AtomicOp::Add add_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected AtomicOp::Add with 0 field(s)");
  return {};
}

AtomicOp::Sub sub_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected AtomicOp::Sub with 0 field(s)");
  return {};
}

AtomicOp::And and_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected AtomicOp::And with 0 field(s)");
  return {};
}

AtomicOp::Or or_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected AtomicOp::Or with 0 field(s)");
  return {};
}

AtomicOp::Xor xor_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected AtomicOp::Xor with 0 field(s)");
  return {};
}

AtomicOp::Min min_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected AtomicOp::Min with 0 field(s)");
  return {};
}

AtomicOp::Max max_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected AtomicOp::Max with 0 field(s)");
  return {};
}

AtomicOp::Any any_from_msgpack(MsgpackReader &r_) {
  if (r_.nextIsArray()) {
    const auto n_ = r_.readArrayHeader();
    if (n_ == 0) throw std::runtime_error("Expected non-empty sum payload");
    const auto ord_ = r_.readInt32();
    switch (ord_) {
      case 0: return AtomicOp::xchg_fields_from_msgpack(r_, n_ - 1);
      case 1: return AtomicOp::add_fields_from_msgpack(r_, n_ - 1);
      case 2: return AtomicOp::sub_fields_from_msgpack(r_, n_ - 1);
      case 3: return AtomicOp::and_fields_from_msgpack(r_, n_ - 1);
      case 4: return AtomicOp::or_fields_from_msgpack(r_, n_ - 1);
      case 5: return AtomicOp::xor_fields_from_msgpack(r_, n_ - 1);
      case 6: return AtomicOp::min_fields_from_msgpack(r_, n_ - 1);
      case 7: return AtomicOp::max_fields_from_msgpack(r_, n_ - 1);
      default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
    }
  } else {
    const auto ord_ = r_.readInt32();
    switch (ord_) {
      case 0: return AtomicOp::xchg_fields_from_msgpack(r_, 0);
      case 1: return AtomicOp::add_fields_from_msgpack(r_, 0);
      case 2: return AtomicOp::sub_fields_from_msgpack(r_, 0);
      case 3: return AtomicOp::and_fields_from_msgpack(r_, 0);
      case 4: return AtomicOp::or_fields_from_msgpack(r_, 0);
      case 5: return AtomicOp::xor_fields_from_msgpack(r_, 0);
      case 6: return AtomicOp::min_fields_from_msgpack(r_, 0);
      case 7: return AtomicOp::max_fields_from_msgpack(r_, 0);
      default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
    }
  }
}

void any_to_msgpack(MsgpackWriter &w_, const AtomicOp::Any &x_) {
  x_.match_total([&](const AtomicOp::Xchg &y_) -> void { w_.writeInt32(0); }, [&](const AtomicOp::Add &y_) -> void { w_.writeInt32(1); },
                 [&](const AtomicOp::Sub &y_) -> void { w_.writeInt32(2); }, [&](const AtomicOp::And &y_) -> void { w_.writeInt32(3); },
                 [&](const AtomicOp::Or &y_) -> void { w_.writeInt32(4); }, [&](const AtomicOp::Xor &y_) -> void { w_.writeInt32(5); },
                 [&](const AtomicOp::Min &y_) -> void { w_.writeInt32(6); }, [&](const AtomicOp::Max &y_) -> void { w_.writeInt32(7); });
}
}} // namespace AtomicOp
namespace MemScope { namespace {

MemScope::Subgroup subgroup_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected MemScope::Subgroup with 0 field(s)");
  return {};
}

MemScope::Workgroup workgroup_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected MemScope::Workgroup with 0 field(s)");
  return {};
}

MemScope::Device device_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected MemScope::Device with 0 field(s)");
  return {};
}

MemScope::System system_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected MemScope::System with 0 field(s)");
  return {};
}

MemScope::Any any_from_msgpack(MsgpackReader &r_) {
  if (r_.nextIsArray()) {
    const auto n_ = r_.readArrayHeader();
    if (n_ == 0) throw std::runtime_error("Expected non-empty sum payload");
    const auto ord_ = r_.readInt32();
    switch (ord_) {
      case 0: return MemScope::subgroup_fields_from_msgpack(r_, n_ - 1);
      case 1: return MemScope::workgroup_fields_from_msgpack(r_, n_ - 1);
      case 2: return MemScope::device_fields_from_msgpack(r_, n_ - 1);
      case 3: return MemScope::system_fields_from_msgpack(r_, n_ - 1);
      default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
    }
  } else {
    const auto ord_ = r_.readInt32();
    switch (ord_) {
      case 0: return MemScope::subgroup_fields_from_msgpack(r_, 0);
      case 1: return MemScope::workgroup_fields_from_msgpack(r_, 0);
      case 2: return MemScope::device_fields_from_msgpack(r_, 0);
      case 3: return MemScope::system_fields_from_msgpack(r_, 0);
      default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
    }
  }
}

void any_to_msgpack(MsgpackWriter &w_, const MemScope::Any &x_) {
  x_.match_total(
      [&](const MemScope::Subgroup &y_) -> void { w_.writeInt32(0); }, [&](const MemScope::Workgroup &y_) -> void { w_.writeInt32(1); },
      [&](const MemScope::Device &y_) -> void { w_.writeInt32(2); }, [&](const MemScope::System &y_) -> void { w_.writeInt32(3); });
}
}} // namespace MemScope
namespace FunctionFpMode { namespace {

FunctionFpMode::Relaxed relaxed_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected FunctionFpMode::Relaxed with 0 field(s)");
  return {};
}

FunctionFpMode::Strict strict_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected FunctionFpMode::Strict with 0 field(s)");
  return {};
}

FunctionFpMode::Any any_from_msgpack(MsgpackReader &r_) {
  if (r_.nextIsArray()) {
    const auto n_ = r_.readArrayHeader();
    if (n_ == 0) throw std::runtime_error("Expected non-empty sum payload");
    const auto ord_ = r_.readInt32();
    switch (ord_) {
      case 0: return FunctionFpMode::relaxed_fields_from_msgpack(r_, n_ - 1);
      case 1: return FunctionFpMode::strict_fields_from_msgpack(r_, n_ - 1);
      default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
    }
  } else {
    const auto ord_ = r_.readInt32();
    switch (ord_) {
      case 0: return FunctionFpMode::relaxed_fields_from_msgpack(r_, 0);
      case 1: return FunctionFpMode::strict_fields_from_msgpack(r_, 0);
      default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
    }
  }
}

void any_to_msgpack(MsgpackWriter &w_, const FunctionFpMode::Any &x_) {
  x_.match_total([&](const FunctionFpMode::Relaxed &y_) -> void { w_.writeInt32(0); },
                 [&](const FunctionFpMode::Strict &y_) -> void { w_.writeInt32(1); });
}
}} // namespace FunctionFpMode
namespace ArgAccess { namespace {

ArgAccess::Read read_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected ArgAccess::Read with 0 field(s)");
  return {};
}

ArgAccess::Write write_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected ArgAccess::Write with 0 field(s)");
  return {};
}

ArgAccess::ReadWrite readwrite_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected ArgAccess::ReadWrite with 0 field(s)");
  return {};
}

ArgAccess::Any any_from_msgpack(MsgpackReader &r_) {
  if (r_.nextIsArray()) {
    const auto n_ = r_.readArrayHeader();
    if (n_ == 0) throw std::runtime_error("Expected non-empty sum payload");
    const auto ord_ = r_.readInt32();
    switch (ord_) {
      case 0: return ArgAccess::read_fields_from_msgpack(r_, n_ - 1);
      case 1: return ArgAccess::write_fields_from_msgpack(r_, n_ - 1);
      case 2: return ArgAccess::readwrite_fields_from_msgpack(r_, n_ - 1);
      default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
    }
  } else {
    const auto ord_ = r_.readInt32();
    switch (ord_) {
      case 0: return ArgAccess::read_fields_from_msgpack(r_, 0);
      case 1: return ArgAccess::write_fields_from_msgpack(r_, 0);
      case 2: return ArgAccess::readwrite_fields_from_msgpack(r_, 0);
      default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
    }
  }
}

void any_to_msgpack(MsgpackWriter &w_, const ArgAccess::Any &x_) {
  x_.match_total([&](const ArgAccess::Read &y_) -> void { w_.writeInt32(0); },
                 [&](const ArgAccess::Write &y_) -> void { w_.writeInt32(1); },
                 [&](const ArgAccess::ReadWrite &y_) -> void { w_.writeInt32(2); });
}
}} // namespace ArgAccess
namespace ArgSizeExpr { namespace {

ArgSizeExpr::Param param_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 1) throw std::runtime_error("Expected ArgSizeExpr::Param with 1 field(s)");
  const auto index = r_.readInt32();
  return ArgSizeExpr::Param(index);
}

void param_fields_to_msgpack(MsgpackWriter &w_, const ArgSizeExpr::Param &x_) { w_.writeInt32(x_.index); }

ArgSizeExpr::Const const_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 1) throw std::runtime_error("Expected ArgSizeExpr::Const with 1 field(s)");
  const auto value = r_.readInt64();
  return ArgSizeExpr::Const(value);
}

void const_fields_to_msgpack(MsgpackWriter &w_, const ArgSizeExpr::Const &x_) { w_.writeInt64(x_.value); }

ArgSizeExpr::Add add_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected ArgSizeExpr::Add with 2 field(s)");
  auto lhs = ArgSizeExpr::any_from_msgpack(r_);
  auto rhs = ArgSizeExpr::any_from_msgpack(r_);
  return {std::move(lhs), std::move(rhs)};
}

void add_fields_to_msgpack(MsgpackWriter &w_, const ArgSizeExpr::Add &x_) {
  ArgSizeExpr::any_to_msgpack(w_, x_.lhs);
  ArgSizeExpr::any_to_msgpack(w_, x_.rhs);
}

ArgSizeExpr::Mul mul_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected ArgSizeExpr::Mul with 2 field(s)");
  auto lhs = ArgSizeExpr::any_from_msgpack(r_);
  auto rhs = ArgSizeExpr::any_from_msgpack(r_);
  return {std::move(lhs), std::move(rhs)};
}

void mul_fields_to_msgpack(MsgpackWriter &w_, const ArgSizeExpr::Mul &x_) {
  ArgSizeExpr::any_to_msgpack(w_, x_.lhs);
  ArgSizeExpr::any_to_msgpack(w_, x_.rhs);
}

ArgSizeExpr::Min min_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected ArgSizeExpr::Min with 2 field(s)");
  auto lhs = ArgSizeExpr::any_from_msgpack(r_);
  auto rhs = ArgSizeExpr::any_from_msgpack(r_);
  return {std::move(lhs), std::move(rhs)};
}

void min_fields_to_msgpack(MsgpackWriter &w_, const ArgSizeExpr::Min &x_) {
  ArgSizeExpr::any_to_msgpack(w_, x_.lhs);
  ArgSizeExpr::any_to_msgpack(w_, x_.rhs);
}

ArgSizeExpr::Result result_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 1) throw std::runtime_error("Expected ArgSizeExpr::Result with 1 field(s)");
  auto bound = ArgSizeExpr::any_from_msgpack(r_);
  return ArgSizeExpr::Result(std::move(bound));
}

void result_fields_to_msgpack(MsgpackWriter &w_, const ArgSizeExpr::Result &x_) { ArgSizeExpr::any_to_msgpack(w_, x_.bound); }

ArgSizeExpr::Any any_from_msgpack(MsgpackReader &r_) {
  if (r_.nextIsArray()) {
    const auto n_ = r_.readArrayHeader();
    if (n_ == 0) throw std::runtime_error("Expected non-empty sum payload");
    const auto ord_ = r_.readInt32();
    switch (ord_) {
      case 0: return ArgSizeExpr::param_fields_from_msgpack(r_, n_ - 1);
      case 1: return ArgSizeExpr::const_fields_from_msgpack(r_, n_ - 1);
      case 2: return ArgSizeExpr::add_fields_from_msgpack(r_, n_ - 1);
      case 3: return ArgSizeExpr::mul_fields_from_msgpack(r_, n_ - 1);
      case 4: return ArgSizeExpr::min_fields_from_msgpack(r_, n_ - 1);
      case 5: return ArgSizeExpr::result_fields_from_msgpack(r_, n_ - 1);
      default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
    }
  } else {
    const auto ord_ = r_.readInt32();
    switch (ord_) {
      case 0:
      case 1:
      case 2:
      case 3:
      case 4:
      case 5: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
    }
  }
}

void any_to_msgpack(MsgpackWriter &w_, const ArgSizeExpr::Any &x_) {
  x_.match_total(
      [&](const ArgSizeExpr::Param &y_) -> void {
        w_.writeArrayHeader(2);
        w_.writeInt32(0);
        ArgSizeExpr::param_fields_to_msgpack(w_, y_);
      },
      [&](const ArgSizeExpr::Const &y_) -> void {
        w_.writeArrayHeader(2);
        w_.writeInt32(1);
        ArgSizeExpr::const_fields_to_msgpack(w_, y_);
      },
      [&](const ArgSizeExpr::Add &y_) -> void {
        w_.writeArrayHeader(3);
        w_.writeInt32(2);
        ArgSizeExpr::add_fields_to_msgpack(w_, y_);
      },
      [&](const ArgSizeExpr::Mul &y_) -> void {
        w_.writeArrayHeader(3);
        w_.writeInt32(3);
        ArgSizeExpr::mul_fields_to_msgpack(w_, y_);
      },
      [&](const ArgSizeExpr::Min &y_) -> void {
        w_.writeArrayHeader(3);
        w_.writeInt32(4);
        ArgSizeExpr::min_fields_to_msgpack(w_, y_);
      },
      [&](const ArgSizeExpr::Result &y_) -> void {
        w_.writeArrayHeader(2);
        w_.writeInt32(5);
        ArgSizeExpr::result_fields_to_msgpack(w_, y_);
      });
}
}} // namespace ArgSizeExpr
namespace PathStep { namespace {

PathStep::Field field_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 1) throw std::runtime_error("Expected PathStep::Field with 1 field(s)");
  auto name = r_.readString();
  return PathStep::Field(std::move(name));
}

void field_fields_to_msgpack(MsgpackWriter &w_, const PathStep::Field &x_) { w_.writeString(x_.name); }

PathStep::Deref deref_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected PathStep::Deref with 0 field(s)");
  return {};
}

PathStep::Index index_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 1) throw std::runtime_error("Expected PathStep::Index with 1 field(s)");
  const auto idx = r_.readInt32();
  return PathStep::Index(idx);
}

void index_fields_to_msgpack(MsgpackWriter &w_, const PathStep::Index &x_) { w_.writeInt32(x_.idx); }

PathStep::IndexDyn indexdyn_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 1) throw std::runtime_error("Expected PathStep::IndexDyn with 1 field(s)");
  auto idx = Term::any_from_msgpack(r_);
  return PathStep::IndexDyn(std::move(idx));
}

void indexdyn_fields_to_msgpack(MsgpackWriter &w_, const PathStep::IndexDyn &x_) { Term::any_to_msgpack(w_, x_.idx); }

PathStep::Any any_from_msgpack(MsgpackReader &r_) {
  if (r_.nextIsArray()) {
    const auto n_ = r_.readArrayHeader();
    if (n_ == 0) throw std::runtime_error("Expected non-empty sum payload");
    const auto ord_ = r_.readInt32();
    switch (ord_) {
      case 0: return PathStep::field_fields_from_msgpack(r_, n_ - 1);
      case 1: return PathStep::deref_fields_from_msgpack(r_, n_ - 1);
      case 2: return PathStep::index_fields_from_msgpack(r_, n_ - 1);
      case 3: return PathStep::indexdyn_fields_from_msgpack(r_, n_ - 1);
      default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
    }
  } else {
    const auto ord_ = r_.readInt32();
    switch (ord_) {
      case 0:
      case 2:
      case 3: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      case 1: return PathStep::deref_fields_from_msgpack(r_, 0);
      default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
    }
  }
}

void any_to_msgpack(MsgpackWriter &w_, const PathStep::Any &x_) {
  x_.match_total(
      [&](const PathStep::Field &y_) -> void {
        w_.writeArrayHeader(2);
        w_.writeInt32(0);
        PathStep::field_fields_to_msgpack(w_, y_);
      },
      [&](const PathStep::Deref &y_) -> void { w_.writeInt32(1); },
      [&](const PathStep::Index &y_) -> void {
        w_.writeArrayHeader(2);
        w_.writeInt32(2);
        PathStep::index_fields_to_msgpack(w_, y_);
      },
      [&](const PathStep::IndexDyn &y_) -> void {
        w_.writeArrayHeader(2);
        w_.writeInt32(3);
        PathStep::indexdyn_fields_to_msgpack(w_, y_);
      });
}
}} // namespace PathStep
namespace TypeKind { namespace {
}} // namespace TypeKind
namespace FunctionVisibility { namespace {

FunctionVisibility::Internal internal_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected FunctionVisibility::Internal with 0 field(s)");
  return {};
}

FunctionVisibility::Exported exported_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected FunctionVisibility::Exported with 0 field(s)");
  return {};
}

FunctionVisibility::Any any_from_msgpack(MsgpackReader &r_) {
  if (r_.nextIsArray()) {
    const auto n_ = r_.readArrayHeader();
    if (n_ == 0) throw std::runtime_error("Expected non-empty sum payload");
    const auto ord_ = r_.readInt32();
    switch (ord_) {
      case 0: return FunctionVisibility::internal_fields_from_msgpack(r_, n_ - 1);
      case 1: return FunctionVisibility::exported_fields_from_msgpack(r_, n_ - 1);
      default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
    }
  } else {
    const auto ord_ = r_.readInt32();
    switch (ord_) {
      case 0: return FunctionVisibility::internal_fields_from_msgpack(r_, 0);
      case 1: return FunctionVisibility::exported_fields_from_msgpack(r_, 0);
      default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
    }
  }
}

void any_to_msgpack(MsgpackWriter &w_, const FunctionVisibility::Any &x_) {
  x_.match_total([&](const FunctionVisibility::Internal &y_) -> void { w_.writeInt32(0); },
                 [&](const FunctionVisibility::Exported &y_) -> void { w_.writeInt32(1); });
}
}} // namespace FunctionVisibility
namespace Type { namespace {

Type::Float16 float16_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected Type::Float16 with 0 field(s)");
  return {};
}

Type::Float32 float32_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected Type::Float32 with 0 field(s)");
  return {};
}

Type::Float64 float64_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected Type::Float64 with 0 field(s)");
  return {};
}

Type::IntU8 intu8_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected Type::IntU8 with 0 field(s)");
  return {};
}

Type::IntU16 intu16_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected Type::IntU16 with 0 field(s)");
  return {};
}

Type::IntU32 intu32_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected Type::IntU32 with 0 field(s)");
  return {};
}

Type::IntU64 intu64_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected Type::IntU64 with 0 field(s)");
  return {};
}

Type::IntS8 ints8_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected Type::IntS8 with 0 field(s)");
  return {};
}

Type::IntS16 ints16_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected Type::IntS16 with 0 field(s)");
  return {};
}

Type::IntS32 ints32_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected Type::IntS32 with 0 field(s)");
  return {};
}

Type::IntS64 ints64_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected Type::IntS64 with 0 field(s)");
  return {};
}

Type::Nothing nothing_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected Type::Nothing with 0 field(s)");
  return {};
}

Type::Unit0 unit0_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected Type::Unit0 with 0 field(s)");
  return {};
}

Type::Bool1 bool1_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected Type::Bool1 with 0 field(s)");
  return {};
}

Type::Struct struct_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected Type::Struct with 2 field(s)");
  auto name = sym_from_msgpack(r_);
  std::vector<Type::Any> args;
  {
    const auto args_size = r_.readArrayHeader();
    args.reserve(args_size);
    for (size_t args_idx = 0; args_idx < args_size; ++args_idx) {
      auto args_elem = Type::any_from_msgpack(r_);
      args.emplace_back(std::move(args_elem));
    }
  }
  return {std::move(name), std::move(args)};
}

void struct_fields_to_msgpack(MsgpackWriter &w_, const Type::Struct &x_) {
  sym_to_msgpack(w_, x_.name);
  w_.writeArrayHeader(x_.args.size());
  for (const auto &v0_ : x_.args) {
    Type::any_to_msgpack(w_, v0_);
  }
}

Type::Struct struct_from_msgpack(MsgpackReader &r_) {
  const auto n_ = r_.readArrayHeader();
  return Type::struct_fields_from_msgpack(r_, n_);
}

void struct_to_msgpack(MsgpackWriter &w_, const Type::Struct &x_) {
  w_.writeArrayHeader(2);
  Type::struct_fields_to_msgpack(w_, x_);
}

Type::Ptr ptr_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected Type::Ptr with 2 field(s)");
  auto comp = Type::any_from_msgpack(r_);
  auto space = TypeSpace::any_from_msgpack(r_);
  return {std::move(comp), std::move(space)};
}

void ptr_fields_to_msgpack(MsgpackWriter &w_, const Type::Ptr &x_) {
  Type::any_to_msgpack(w_, x_.comp);
  TypeSpace::any_to_msgpack(w_, x_.space);
}

Type::Arr arr_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 3) throw std::runtime_error("Expected Type::Arr with 3 field(s)");
  auto comp = Type::any_from_msgpack(r_);
  const auto length = r_.readInt32();
  auto space = TypeSpace::any_from_msgpack(r_);
  return {std::move(comp), length, std::move(space)};
}

void arr_fields_to_msgpack(MsgpackWriter &w_, const Type::Arr &x_) {
  Type::any_to_msgpack(w_, x_.comp);
  w_.writeInt32(x_.length);
  TypeSpace::any_to_msgpack(w_, x_.space);
}

Type::Var var_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 2) throw std::runtime_error("Expected Type::Var with 2 field(s)");
  auto name = r_.readString();
  std::optional<int32_t> exactSizeInBytes;
  if (!r_.tryReadNil()) {
    const auto exactSizeInBytes_value = r_.readInt32();
    exactSizeInBytes = exactSizeInBytes_value;
  }
  return {std::move(name), exactSizeInBytes};
}

void var_fields_to_msgpack(MsgpackWriter &w_, const Type::Var &x_) {
  w_.writeString(x_.name);
  if (x_.exactSizeInBytes) {
    w_.writeInt32((*x_.exactSizeInBytes));
  } else {
    w_.writeNil();
  }
}

Type::Var var_from_msgpack(MsgpackReader &r_) {
  const auto n_ = r_.readArrayHeader();
  return Type::var_fields_from_msgpack(r_, n_);
}

void var_to_msgpack(MsgpackWriter &w_, const Type::Var &x_) {
  w_.writeArrayHeader(2);
  Type::var_fields_to_msgpack(w_, x_);
}

Type::Exec exec_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 3) throw std::runtime_error("Expected Type::Exec with 3 field(s)");
  std::vector<Type::Var> tpeVars;
  {
    const auto tpeVars_size = r_.readArrayHeader();
    tpeVars.reserve(tpeVars_size);
    for (size_t tpeVars_idx = 0; tpeVars_idx < tpeVars_size; ++tpeVars_idx) {
      auto tpeVars_elem = Type::var_from_msgpack(r_);
      tpeVars.emplace_back(std::move(tpeVars_elem));
    }
  }
  std::vector<Type::Any> args;
  {
    const auto args_size = r_.readArrayHeader();
    args.reserve(args_size);
    for (size_t args_idx = 0; args_idx < args_size; ++args_idx) {
      auto args_elem = Type::any_from_msgpack(r_);
      args.emplace_back(std::move(args_elem));
    }
  }
  auto rtn = Type::any_from_msgpack(r_);
  return {std::move(tpeVars), std::move(args), std::move(rtn)};
}

void exec_fields_to_msgpack(MsgpackWriter &w_, const Type::Exec &x_) {
  w_.writeArrayHeader(x_.tpeVars.size());
  for (const auto &v0_ : x_.tpeVars) {
    Type::var_to_msgpack(w_, v0_);
  }
  w_.writeArrayHeader(x_.args.size());
  for (const auto &v0_ : x_.args) {
    Type::any_to_msgpack(w_, v0_);
  }
  Type::any_to_msgpack(w_, x_.rtn);
}

Type::FnRef fnref_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 1) throw std::runtime_error("Expected Type::FnRef with 1 field(s)");
  auto name = sym_from_msgpack(r_);
  return Type::FnRef(std::move(name));
}

void fnref_fields_to_msgpack(MsgpackWriter &w_, const Type::FnRef &x_) { sym_to_msgpack(w_, x_.name); }

Type::Any any_from_msgpack(MsgpackReader &r_) {
  if (r_.nextIsArray()) {
    const auto n_ = r_.readArrayHeader();
    if (n_ == 0) throw std::runtime_error("Expected non-empty sum payload");
    const auto ord_ = r_.readInt32();
    switch (ord_) {
      case 0: return Type::float16_fields_from_msgpack(r_, n_ - 1);
      case 1: return Type::float32_fields_from_msgpack(r_, n_ - 1);
      case 2: return Type::float64_fields_from_msgpack(r_, n_ - 1);
      case 3: return Type::intu8_fields_from_msgpack(r_, n_ - 1);
      case 4: return Type::intu16_fields_from_msgpack(r_, n_ - 1);
      case 5: return Type::intu32_fields_from_msgpack(r_, n_ - 1);
      case 6: return Type::intu64_fields_from_msgpack(r_, n_ - 1);
      case 7: return Type::ints8_fields_from_msgpack(r_, n_ - 1);
      case 8: return Type::ints16_fields_from_msgpack(r_, n_ - 1);
      case 9: return Type::ints32_fields_from_msgpack(r_, n_ - 1);
      case 10: return Type::ints64_fields_from_msgpack(r_, n_ - 1);
      case 11: return Type::nothing_fields_from_msgpack(r_, n_ - 1);
      case 12: return Type::unit0_fields_from_msgpack(r_, n_ - 1);
      case 13: return Type::bool1_fields_from_msgpack(r_, n_ - 1);
      case 14: return Type::struct_fields_from_msgpack(r_, n_ - 1);
      case 15: return Type::ptr_fields_from_msgpack(r_, n_ - 1);
      case 16: return Type::arr_fields_from_msgpack(r_, n_ - 1);
      case 17: return Type::var_fields_from_msgpack(r_, n_ - 1);
      case 18: return Type::exec_fields_from_msgpack(r_, n_ - 1);
      case 19: return Type::fnref_fields_from_msgpack(r_, n_ - 1);
      default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
    }
  } else {
    const auto ord_ = r_.readInt32();
    switch (ord_) {
      case 14:
      case 15:
      case 16:
      case 17:
      case 18:
      case 19: throw std::runtime_error("Expected array payload for non-nullary sum ordinal");
      case 0: return Type::float16_fields_from_msgpack(r_, 0);
      case 1: return Type::float32_fields_from_msgpack(r_, 0);
      case 2: return Type::float64_fields_from_msgpack(r_, 0);
      case 3: return Type::intu8_fields_from_msgpack(r_, 0);
      case 4: return Type::intu16_fields_from_msgpack(r_, 0);
      case 5: return Type::intu32_fields_from_msgpack(r_, 0);
      case 6: return Type::intu64_fields_from_msgpack(r_, 0);
      case 7: return Type::ints8_fields_from_msgpack(r_, 0);
      case 8: return Type::ints16_fields_from_msgpack(r_, 0);
      case 9: return Type::ints32_fields_from_msgpack(r_, 0);
      case 10: return Type::ints64_fields_from_msgpack(r_, 0);
      case 11: return Type::nothing_fields_from_msgpack(r_, 0);
      case 12: return Type::unit0_fields_from_msgpack(r_, 0);
      case 13: return Type::bool1_fields_from_msgpack(r_, 0);
      default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
    }
  }
}

void any_to_msgpack(MsgpackWriter &w_, const Type::Any &x_) {
  x_.match_total([&](const Type::Float16 &y_) -> void { w_.writeInt32(0); }, [&](const Type::Float32 &y_) -> void { w_.writeInt32(1); },
                 [&](const Type::Float64 &y_) -> void { w_.writeInt32(2); }, [&](const Type::IntU8 &y_) -> void { w_.writeInt32(3); },
                 [&](const Type::IntU16 &y_) -> void { w_.writeInt32(4); }, [&](const Type::IntU32 &y_) -> void { w_.writeInt32(5); },
                 [&](const Type::IntU64 &y_) -> void { w_.writeInt32(6); }, [&](const Type::IntS8 &y_) -> void { w_.writeInt32(7); },
                 [&](const Type::IntS16 &y_) -> void { w_.writeInt32(8); }, [&](const Type::IntS32 &y_) -> void { w_.writeInt32(9); },
                 [&](const Type::IntS64 &y_) -> void { w_.writeInt32(10); }, [&](const Type::Nothing &y_) -> void { w_.writeInt32(11); },
                 [&](const Type::Unit0 &y_) -> void { w_.writeInt32(12); }, [&](const Type::Bool1 &y_) -> void { w_.writeInt32(13); },
                 [&](const Type::Struct &y_) -> void {
                   w_.writeArrayHeader(3);
                   w_.writeInt32(14);
                   Type::struct_fields_to_msgpack(w_, y_);
                 },
                 [&](const Type::Ptr &y_) -> void {
                   w_.writeArrayHeader(3);
                   w_.writeInt32(15);
                   Type::ptr_fields_to_msgpack(w_, y_);
                 },
                 [&](const Type::Arr &y_) -> void {
                   w_.writeArrayHeader(4);
                   w_.writeInt32(16);
                   Type::arr_fields_to_msgpack(w_, y_);
                 },
                 [&](const Type::Var &y_) -> void {
                   w_.writeArrayHeader(3);
                   w_.writeInt32(17);
                   Type::var_fields_to_msgpack(w_, y_);
                 },
                 [&](const Type::Exec &y_) -> void {
                   w_.writeArrayHeader(4);
                   w_.writeInt32(18);
                   Type::exec_fields_to_msgpack(w_, y_);
                 },
                 [&](const Type::FnRef &y_) -> void {
                   w_.writeArrayHeader(2);
                   w_.writeInt32(19);
                   Type::fnref_fields_to_msgpack(w_, y_);
                 });
}
}} // namespace Type
namespace Direction { namespace {

Direction::LocalToRemote localtoremote_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected Direction::LocalToRemote with 0 field(s)");
  return {};
}

Direction::RemoteToLocal remotetolocal_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected Direction::RemoteToLocal with 0 field(s)");
  return {};
}

Direction::RemoteToRemote remotetoremote_fields_from_msgpack(MsgpackReader &r_, size_t n_) {
  if (n_ != 0) throw std::runtime_error("Expected Direction::RemoteToRemote with 0 field(s)");
  return {};
}

Direction::Any any_from_msgpack(MsgpackReader &r_) {
  if (r_.nextIsArray()) {
    const auto n_ = r_.readArrayHeader();
    if (n_ == 0) throw std::runtime_error("Expected non-empty sum payload");
    const auto ord_ = r_.readInt32();
    switch (ord_) {
      case 0: return Direction::localtoremote_fields_from_msgpack(r_, n_ - 1);
      case 1: return Direction::remotetolocal_fields_from_msgpack(r_, n_ - 1);
      case 2: return Direction::remotetoremote_fields_from_msgpack(r_, n_ - 1);
      default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
    }
  } else {
    const auto ord_ = r_.readInt32();
    switch (ord_) {
      case 0: return Direction::localtoremote_fields_from_msgpack(r_, 0);
      case 1: return Direction::remotetolocal_fields_from_msgpack(r_, 0);
      case 2: return Direction::remotetoremote_fields_from_msgpack(r_, 0);
      default: throw std::out_of_range("Bad ordinal " + std::to_string(ord_));
    }
  }
}

void any_to_msgpack(MsgpackWriter &w_, const Direction::Any &x_) {
  x_.match_total([&](const Direction::LocalToRemote &y_) -> void { w_.writeInt32(0); },
                 [&](const Direction::RemoteToLocal &y_) -> void { w_.writeInt32(1); },
                 [&](const Direction::RemoteToRemote &y_) -> void { w_.writeInt32(2); });
}
}} // namespace Direction

std::vector<uint8_t> program_to_msgpack(const Program &x_) {
  return encodeInterned([&](MsgpackWriter &w_) { program_to_msgpack(w_, x_); });
}

Program program_from_msgpack(const uint8_t *begin_, const uint8_t *end_) {
  return decodeMaybeInterned(begin_, end_, [](MsgpackReader &r_) {
    auto value_ = program_from_msgpack(r_);
    return value_;
  });
}

Program program_from_msgpack(const std::vector<uint8_t> &xs_) { return program_from_msgpack(xs_.data(), xs_.data() + xs_.size()); }

std::vector<uint8_t> hashed_program_to_msgpack(const Program &x_) {
  return encodeInterned([&](MsgpackWriter &w_) {
    w_.writeArrayHeader(2);
    w_.writeString(std::string(ProgramHash));
    program_to_msgpack(w_, x_);
  });
}

Program hashed_program_from_msgpack(const uint8_t *begin_, const uint8_t *end_) {
  return decodeMaybeInterned(begin_, end_, [](MsgpackReader &r_) {
    const auto n_ = r_.readArrayHeader();
    if (n_ != 2) throw std::runtime_error("Expected versioned Program array of size 2");
    const auto hash_ = r_.readString();
    if (hash_ != ProgramHash) throw std::runtime_error("Expecting Program hash to be " + std::string(ProgramHash) + ", but was " + hash_);
    auto value_ = program_from_msgpack(r_);
    return value_;
  });
}

Program hashed_program_from_msgpack(const std::vector<uint8_t> &xs_) {
  return hashed_program_from_msgpack(xs_.data(), xs_.data() + xs_.size());
}

std::vector<uint8_t> interface_to_msgpack(const Interface &x_) {
  return encodeInterned([&](MsgpackWriter &w_) {
    w_.writeArrayHeader(2);
    w_.writeString(std::string(PackageHash));
    interface_to_msgpack(w_, x_);
  });
}

Interface interface_from_msgpack(const uint8_t *begin_, const uint8_t *end_) {
  return decodeMaybeInterned(begin_, end_, [](MsgpackReader &r_) {
    const auto n_ = r_.readArrayHeader();
    if (n_ != 2) throw std::runtime_error("Expected versioned Interface array of size 2");
    const auto hash_ = r_.readString();
    if (hash_ != PackageHash) throw std::runtime_error("Expecting Interface hash to be " + std::string(PackageHash) + ", but was " + hash_);
    auto value_ = interface_from_msgpack(r_);
    return value_;
  });
}

Interface interface_from_msgpack(const std::vector<uint8_t> &xs_) { return interface_from_msgpack(xs_.data(), xs_.data() + xs_.size()); }

std::vector<uint8_t> package_to_msgpack(const Package &x_) {
  return encodeInterned([&](MsgpackWriter &w_) {
    w_.writeArrayHeader(2);
    w_.writeString(std::string(PackageHash));
    package_to_msgpack(w_, x_);
  });
}

Package package_from_msgpack(const uint8_t *begin_, const uint8_t *end_) {
  return decodeMaybeInterned(begin_, end_, [](MsgpackReader &r_) {
    const auto n_ = r_.readArrayHeader();
    if (n_ != 2) throw std::runtime_error("Expected versioned Package array of size 2");
    const auto hash_ = r_.readString();
    if (hash_ != PackageHash) throw std::runtime_error("Expecting Package hash to be " + std::string(PackageHash) + ", but was " + hash_);
    auto value_ = package_from_msgpack(r_);
    return value_;
  });
}

Package package_from_msgpack(const std::vector<uint8_t> &xs_) { return package_from_msgpack(xs_.data(), xs_.data() + xs_.size()); }

std::vector<uint8_t> package_service_result_to_msgpack(const Package &x_) {
  return encodeInterned([&](MsgpackWriter &w_) {
    w_.writeArrayHeader(2);
    w_.writeString(std::string(PackageWireHash));
    package_to_msgpack(w_, x_);
  });
}

Package package_service_result_from_msgpack(const uint8_t *begin_, const uint8_t *end_) {
  return decodeMaybeInterned(begin_, end_, [](MsgpackReader &r_) {
    const auto n_ = r_.readArrayHeader();
    if (n_ != 2) throw std::runtime_error("Expected versioned package-service Package array of size 2");
    const auto hash_ = r_.readString();
    if (hash_ != PackageWireHash)
      throw std::runtime_error("Expecting package-service wire hash to be " + std::string(PackageWireHash) + ", but was " + hash_);
    auto value_ = package_from_msgpack(r_);
    return value_;
  });
}

Package package_service_result_from_msgpack(const std::vector<uint8_t> &xs_) {
  return package_service_result_from_msgpack(xs_.data(), xs_.data() + xs_.size());
}

std::vector<uint8_t> program_service_result_to_msgpack(const Program &x_) {
  return encodeInterned([&](MsgpackWriter &w_) {
    w_.writeArrayHeader(2);
    w_.writeString(std::string(PackageWireHash));
    program_to_msgpack(w_, x_);
  });
}

Program program_service_result_from_msgpack(const uint8_t *begin_, const uint8_t *end_) {
  return decodeMaybeInterned(begin_, end_, [](MsgpackReader &r_) {
    const auto n_ = r_.readArrayHeader();
    if (n_ != 2) throw std::runtime_error("Expected versioned package-service Program array of size 2");
    const auto hash_ = r_.readString();
    if (hash_ != PackageWireHash)
      throw std::runtime_error("Expecting package-service wire hash to be " + std::string(PackageWireHash) + ", but was " + hash_);
    auto value_ = program_from_msgpack(r_);
    return value_;
  });
}

Program program_service_result_from_msgpack(const std::vector<uint8_t> &xs_) {
  return program_service_result_from_msgpack(xs_.data(), xs_.data() + xs_.size());
}

std::vector<uint8_t> packagelinkrequest_to_msgpack(const PackageLinkRequest &x_) {
  return encodeInterned([&](MsgpackWriter &w_) {
    w_.writeArrayHeader(2);
    w_.writeString(std::string(PackageWireHash));
    packagelinkrequest_to_msgpack(w_, x_);
  });
}

PackageLinkRequest packagelinkrequest_from_msgpack(const uint8_t *begin_, const uint8_t *end_) {
  return decodeMaybeInterned(begin_, end_, [](MsgpackReader &r_) {
    const auto n_ = r_.readArrayHeader();
    if (n_ != 2) throw std::runtime_error("Expected versioned PackageLinkRequest array of size 2");
    const auto hash_ = r_.readString();
    if (hash_ != PackageWireHash)
      throw std::runtime_error("Expecting package-service wire hash to be " + std::string(PackageWireHash) + ", but was " + hash_);
    auto value_ = packagelinkrequest_from_msgpack(r_);
    return value_;
  });
}

PackageLinkRequest packagelinkrequest_from_msgpack(const std::vector<uint8_t> &xs_) {
  return packagelinkrequest_from_msgpack(xs_.data(), xs_.data() + xs_.size());
}

std::vector<uint8_t> programlinkrequest_to_msgpack(const ProgramLinkRequest &x_) {
  return encodeInterned([&](MsgpackWriter &w_) {
    w_.writeArrayHeader(2);
    w_.writeString(std::string(PackageWireHash));
    programlinkrequest_to_msgpack(w_, x_);
  });
}

ProgramLinkRequest programlinkrequest_from_msgpack(const uint8_t *begin_, const uint8_t *end_) {
  return decodeMaybeInterned(begin_, end_, [](MsgpackReader &r_) {
    const auto n_ = r_.readArrayHeader();
    if (n_ != 2) throw std::runtime_error("Expected versioned ProgramLinkRequest array of size 2");
    const auto hash_ = r_.readString();
    if (hash_ != PackageWireHash)
      throw std::runtime_error("Expecting package-service wire hash to be " + std::string(PackageWireHash) + ", but was " + hash_);
    auto value_ = programlinkrequest_from_msgpack(r_);
    return value_;
  });
}

ProgramLinkRequest programlinkrequest_from_msgpack(const std::vector<uint8_t> &xs_) {
  return programlinkrequest_from_msgpack(xs_.data(), xs_.data() + xs_.size());
}

std::vector<uint8_t> compilebundle_to_msgpack(const CompileBundle &x_) {
  return encodeInterned([&](MsgpackWriter &w_) {
    w_.writeArrayHeader(2);
    w_.writeString(std::string(CompileWireHash));
    compilebundle_to_msgpack(w_, x_);
  });
}

CompileBundle compilebundle_from_msgpack(const uint8_t *begin_, const uint8_t *end_) {
  return decodeMaybeInterned(begin_, end_, [](MsgpackReader &r_) {
    const auto n_ = r_.readArrayHeader();
    if (n_ != 2) throw std::runtime_error("Expected versioned CompileBundle array of size 2");
    const auto hash_ = r_.readString();
    if (hash_ != CompileWireHash)
      throw std::runtime_error("Expecting native compile wire hash to be " + std::string(CompileWireHash) + ", but was " + hash_);
    auto value_ = compilebundle_from_msgpack(r_);
    return value_;
  });
}

CompileBundle compilebundle_from_msgpack(const std::vector<uint8_t> &xs_) {
  return compilebundle_from_msgpack(xs_.data(), xs_.data() + xs_.size());
}

std::vector<uint8_t> structdefs_to_msgpack(const std::vector<StructDef> &x_) {
  return encodeInterned([&](MsgpackWriter &w_) {
    w_.writeArrayHeader(x_.size());
    for (const auto &v0_ : x_) {
      structdef_to_msgpack(w_, v0_);
    }
  });
}

std::vector<StructDef> structdefs_from_msgpack(const uint8_t *begin_, const uint8_t *end_) {
  return decodeMaybeInterned(begin_, end_, [](MsgpackReader &r_) {
    std::vector<StructDef> value_;
    {
      const auto value__size = r_.readArrayHeader();
      value_.reserve(value__size);
      for (size_t value__idx = 0; value__idx < value__size; ++value__idx) {
        auto value__elem = structdef_from_msgpack(r_);
        value_.emplace_back(std::move(value__elem));
      }
    }
    return value_;
  });
}

std::vector<StructDef> structdefs_from_msgpack(const std::vector<uint8_t> &xs_) {
  return structdefs_from_msgpack(xs_.data(), xs_.data() + xs_.size());
}

std::vector<uint8_t> hashed_structdefs_to_msgpack(const std::vector<StructDef> &x_) {
  return encodeInterned([&](MsgpackWriter &w_) {
    w_.writeArrayHeader(2);
    w_.writeString(std::string(ProgramHash));
    w_.writeArrayHeader(x_.size());
    for (const auto &v0_ : x_) {
      structdef_to_msgpack(w_, v0_);
    }
  });
}

std::vector<StructDef> hashed_structdefs_from_msgpack(const uint8_t *begin_, const uint8_t *end_) {
  return decodeMaybeInterned(begin_, end_, [](MsgpackReader &r_) {
    const auto n_ = r_.readArrayHeader();
    if (n_ != 2) throw std::runtime_error("Expected versioned StructDef list array of size 2");
    const auto hash_ = r_.readString();
    if (hash_ != ProgramHash) throw std::runtime_error("Expecting Program hash to be " + std::string(ProgramHash) + ", but was " + hash_);
    std::vector<StructDef> value_;
    {
      const auto value__size = r_.readArrayHeader();
      value_.reserve(value__size);
      for (size_t value__idx = 0; value__idx < value__size; ++value__idx) {
        auto value__elem = structdef_from_msgpack(r_);
        value_.emplace_back(std::move(value__elem));
      }
    }
    return value_;
  });
}

std::vector<StructDef> hashed_structdefs_from_msgpack(const std::vector<uint8_t> &xs_) {
  return hashed_structdefs_from_msgpack(xs_.data(), xs_.data() + xs_.size());
}

std::vector<uint8_t> compileresult_to_msgpack(const CompileResult &x_) {
  return encodeInterned([&](MsgpackWriter &w_) { compileresult_to_msgpack(w_, x_); });
}

CompileResult compileresult_from_msgpack(const uint8_t *begin_, const uint8_t *end_) {
  return decodeMaybeInterned(begin_, end_, [](MsgpackReader &r_) {
    auto value_ = compileresult_from_msgpack(r_);
    return value_;
  });
}

CompileResult compileresult_from_msgpack(const std::vector<uint8_t> &xs_) {
  return compileresult_from_msgpack(xs_.data(), xs_.data() + xs_.size());
}

std::vector<uint8_t> passrunresult_to_msgpack(const PassRunResult &x_) {
  return encodeInterned([&](MsgpackWriter &w_) { passrunresult_to_msgpack(w_, x_); });
}

PassRunResult passrunresult_from_msgpack(const uint8_t *begin_, const uint8_t *end_) {
  return decodeMaybeInterned(begin_, end_, [](MsgpackReader &r_) {
    auto value_ = passrunresult_from_msgpack(r_);
    return value_;
  });
}

PassRunResult passrunresult_from_msgpack(const std::vector<uint8_t> &xs_) {
  return passrunresult_from_msgpack(xs_.data(), xs_.data() + xs_.size());
}
} // namespace polyregion::polyast
