#include <iomanip>
#include <string>

#include "aspartame/all.hpp"
#include "ast.h"

using namespace std::string_literals;
using namespace polyregion::polyast;
using namespace polyregion;
using std::string;

using namespace aspartame;

string polyast::qualified(const Expr::Select &select) {
  return select.init | append(select.last) | mk_string(".", [](auto &x) { return x.symbol; });
}

Vec<Named> polyast::path(const Expr::Select &select) { return select.init | append(select.last) | to_vector(); }

Named polyast::head(const Expr::Select &select) { return select.init.empty() ? select.last : select.init.front(); }

Vec<Named> polyast::tail(const Expr::Select &select) {
  if (select.init.empty()) return {select.last};
  else {
    Vec<Named> xs(std::next(select.init.begin()), select.init.end());
    xs.push_back(select.last);
    return xs;
  }
}

Pair<Named, Vec<Named>> polyast::uncons(const Expr::Select &select) {
  if (select.init.empty()) return {{select.last}, {}};
  else {
    Vec<Named> xs(std::next(select.init.begin()), select.init.end());
    xs.push_back(select.last);
    return {select.init.front(), xs};
  }
}

Expr::Select polyast::selectNamed(const Expr::Select &select, const Named &that) { return Expr::Select(path(select), that); }

Expr::Select polyast::selectNamed(const Vec<Named> &names) {
  if (names.empty()) throw std::logic_error("Cannot create select from empty name paths");
  return Expr::Select(Vec<Named>(names.begin(), std::prev(names.end())), names.back());
}

Expr::Select polyast::selectNamed(const Named &name) { return Expr::Select({}, name); }

Expr::Select polyast::parent(const Expr::Select &select) {
  if (select.init.empty()) return select;
  return Expr::Select(Vec<Named>(select.init.begin(), std::prev(select.init.end())), select.init.back());
}

Type::Struct polyast::typeOf(const StructDef &def) { return Type::Struct(def.name); }

string polyast::repr(const CompileResult &compilation) {
  std::ostringstream os;
  os << "Compilation {"                                                                                            //
     << "\n  binary: " << (compilation.binary ? std::to_string(compilation.binary->size()) + " bytes" : "(empty)") //
     << "\n  messages: `" << compilation.messages << "`"                                                           //
     << "\n  features: `" << (compilation.features ^ mk_string(",")) << "`"                                        //
     << "\n  layouts: `\n"
     << (compilation.layouts ^ mk_string("\n", [](auto &l) { return repr(l) ^ indent(4); })) << "`" //
     << "\n  events:\n";

  for (auto &e : compilation.events) {
    os << "    [" << e.epochMillis << ", +" << static_cast<double>(e.elapsedNanos) / 1e6 << "ms] " << e.name;
    if (e.data.empty()) continue;
    os << ":\n";
    std::stringstream ss(e.data);
    string l;
    size_t ln = 0;
    while (std::getline(ss, l, '\n')) {
      ln++;
      os << "    " << std::setw(3) << ln << "│" << l << '\n';
    }
    os << "       ╰───\n";
  }
  os << "\n}";
  return os.str();
}

Opt<Type::Any> polyast::extractComponent(const Type::Any &t) {
  if (const auto p = t.get<Type::Ptr>()) return extractComponent(p->comp);
  if (const auto a = t.get<Type::Annotated>()) return extractComponent(a->tpe);
  return t;
}

Opt<size_t> polyast::primitiveSize(const Type::Any &t) {
  return t.match_total([&](const Type::Float16 &) -> Opt<size_t> { return 16 / 8; }, //
                       [&](const Type::Float32 &) -> Opt<size_t> { return 32 / 8; }, //
                       [&](const Type::Float64 &) -> Opt<size_t> { return 64 / 8; }, //

                       [&](const Type::IntU8 &) -> Opt<size_t> { return 8 / 8; },   //
                       [&](const Type::IntU16 &) -> Opt<size_t> { return 16 / 8; }, //
                       [&](const Type::IntU32 &) -> Opt<size_t> { return 32 / 8; }, //
                       [&](const Type::IntU64 &) -> Opt<size_t> { return 64 / 8; }, //

                       [&](const Type::IntS8 &) -> Opt<size_t> { return 8 / 8; },   //
                       [&](const Type::IntS16 &) -> Opt<size_t> { return 16 / 8; }, //
                       [&](const Type::IntS32 &) -> Opt<size_t> { return 32 / 8; }, //
                       [&](const Type::IntS64 &) -> Opt<size_t> { return 64 / 8; }, //

                       [&](const Type::Nothing &) -> Opt<size_t> { return {}; },  //
                       [&](const Type::Unit0 &) -> Opt<size_t> { return 8 / 8; }, //
                       [&](const Type::Bool1 &) -> Opt<size_t> { return 8 / 8; }, //

                       [&](const Type::Struct &) -> Opt<size_t> { return {}; }, [&](const Type::Ptr &) -> Opt<size_t> { return {}; },
                       [&](const Type::Annotated &x) -> Opt<size_t> { return primitiveSize(x.tpe); });
}

Pair<size_t, Opt<size_t>> polyast::countIndirectionsAndComponentSize(const Type::Any &t, const Map<Type::Struct, StructLayout> &table) {
  if (const auto s = t.get<Type::Struct>()) return {0, table ^ get_maybe(*s) ^ map([](auto &sl) { return sl.sizeInBytes; })};
  if (const auto a = t.get<Type::Annotated>()) return countIndirectionsAndComponentSize(a->tpe, table);
  if (const auto p = t.get<Type::Ptr>()) {
    auto [indirection, componentSize] = countIndirectionsAndComponentSize(p->comp, table);
    return {p->length ? indirection : indirection + 1, componentSize};
  }
  return {0, primitiveSize(t)};
}

bool polyast::isSelfOpaque(const Type::Any &tpe) {
  if (const auto a = tpe.get<Type::Annotated>()) return isSelfOpaque(a->tpe);
  if (const auto p = tpe.get<Type::Ptr>()) return p->length.has_value() && isSelfOpaque(p->comp);
  return true;
}

bool polyast::isSelfOpaque(const StructLayout &sl) {
  return sl.members ^ forall([](auto &m) { return isSelfOpaque(m.name.tpe); });
}

bool polyast::isOpaque(const StructLayout &sl, const std::unordered_map<Type::Struct, StructLayout> &table) {
  return isSelfOpaque(sl) &&
         sl.members ^ forall([&](auto &m) {
           return m.name.tpe.template get<Type::Struct>() ^
                  fold(
                      [&](auto &s) { return table ^ get_maybe(s) ^ map([&](auto &x) { return isOpaque(x, table); }) ^ get_or_else(false); },
                      []() { return true; });
         });
}

// ====================

Type::Ptr dsl::Ptr(const Type::Any &t, Opt<int32_t> l, const ::TypeSpace::Any &s) { return Tpe::Ptr(t, l, s); }
Expr::Any dsl::integral(const Type::Any &tpe, unsigned long long int x) {
  auto unsupported = [](auto &&t, auto &&v) -> Expr::Any {
    throw std::logic_error("Cannot create integral constant of type " + to_string(t) + " for value" + std::to_string(v));
  };
  return tpe.match_total(                                                  //
      [&](const Type::Float16 &) -> Expr::Any { return Float16Const(x); }, //
      [&](const Type::Float32 &) -> Expr::Any { return Float32Const(x); }, //
      [&](const Type::Float64 &) -> Expr::Any { return Float64Const(x); }, //

      [&](const Type::IntU8 &) -> Expr::Any { return IntU8Const(x); },   //
      [&](const Type::IntU16 &) -> Expr::Any { return IntU16Const(x); }, //
      [&](const Type::IntU32 &) -> Expr::Any { return IntU32Const(x); }, //
      [&](const Type::IntU64 &) -> Expr::Any { return IntU64Const(x); }, //

      [&](const Type::IntS8 &) -> Expr::Any { return IntS8Const(x); },   //
      [&](const Type::IntS16 &) -> Expr::Any { return IntS16Const(x); }, //
      [&](const Type::IntS32 &) -> Expr::Any { return IntS32Const(x); }, //
      [&](const Type::IntS64 &) -> Expr::Any { return IntS64Const(x); }, //

      [&](const Type::Nothing &t) -> Expr::Any { return unsupported(t, x); }, //
      [&](const Type::Unit0 &t) -> Expr::Any { return unsupported(t, x); },   //
      [&](const Type::Bool1 &) -> Expr::Any { return Bool1Const(x); },        //

      [&](const Type::Struct &t) -> Expr::Any { return unsupported(t, x); },   //
      [&](const Type::Ptr &t) -> Expr::Any { return unsupported(t, x); },      //
      [&](const Type::Annotated &t) -> Expr::Any { return unsupported(t, x); } //
  );
}
Expr::Any dsl::fractional(const Type::Any &tpe, long double x) {
  if (tpe.is<Type::Float64>()) return Float64Const(static_cast<double>(x));
  if (tpe.is<Type::Float32>()) return Float32Const(static_cast<float>(x));
  if (tpe.is<Type::Float16>()) return Float16Const(static_cast<float>(x));
  throw std::logic_error("Cannot create fractional constant of type " + to_string(tpe) + " for value" + std::to_string(x));
}
std::function<Expr::Any(Type::Any)> dsl::operator""_(unsigned long long int x) {
  return [=](const Type::Any &t) { return integral(t, x); };
}
std::function<Expr::Any(Type::Any)> dsl::operator""_(long double x) {
  return [=](const Type::Any &t) { return fractional(t, x); };
}
std::function<dsl::NamedBuilder(Type::Any)> dsl::operator""_(const char *name, size_t) {
  string name_(name);
  return [=](auto &&tpe) { return NamedBuilder{Named(name_, tpe)}; };
}

Stmt::Any dsl::let(const string &name, const Type::Any &tpe) { return Var(Named(name, tpe), {}); }
dsl::AssignmentBuilder dsl::let(const string &name) { return AssignmentBuilder{name}; }
Expr::IntrOp dsl::call(const Intr::Any &intr) { return IntrOp(intr); }
Expr::MathOp dsl::call(const Math::Any &intr) { return MathOp(intr); }
Expr::SpecOp dsl::call(const Spec::Any &intr) { return SpecOp(intr); }
std::function<Function(Vec<Stmt::Any>)> dsl::function(const string &name, const Vec<Arg> &args, const Type::Any &rtn,
                                                      const std::set<FunctionAttr::Any> &attrs) {
  return [=](auto &&stmts) { return Function(name, args, rtn, stmts, attrs); };
}
Stmt::Return dsl::ret(const Expr::Any &expr) { return Return(expr); }
Program dsl::program(const Vec<StructDef> &structs, const Vec<Function> &functions) { return {structs, functions}; }
Program dsl::program(const Function &function) { return Program({}, {function}); }

dsl::IndexBuilder::IndexBuilder(const Index &index) : index(index) {}
dsl::IndexBuilder::operator Expr::Any() const { return index; }
Stmt::Update dsl::IndexBuilder::operator=(const Expr::Any &term) const { return {index.lhs, index.idx, term}; }
dsl::NamedBuilder::NamedBuilder(const Named &named) : named(named) {}
dsl::NamedBuilder::operator Expr::Any() const { return Select({}, named); }
dsl::NamedBuilder::operator Expr::Select() const { return Select({}, named); }
// dsl::NamedBuilder::operator const Expr::Any() const { return Alias(Select({}, named)); }
dsl::NamedBuilder::operator Named() const { return named; }
Arg dsl::NamedBuilder::operator()() const { return Arg(named, {}); }

dsl::IndexBuilder dsl::NamedBuilder::operator[](const Expr::Any &idx) const {
  if (auto arr = named.tpe.get<Type::Ptr>()) {
    return IndexBuilder({Select({}, named), idx, arr->comp});
  }
  throw std::logic_error("Cannot index a reference to non-array type" + to_string(named));
}
dsl::Mut dsl::NamedBuilder::operator=(const Expr::Any &that) const { return Mut(Select({}, named), that); }

dsl::AssignmentBuilder::AssignmentBuilder(const string &name) : name(name) {}
Stmt::Any dsl::AssignmentBuilder::operator=(Expr::Any rhs) const { return Var(Named(name, rhs.tpe()), {rhs}); }
Stmt::Any dsl::AssignmentBuilder::operator=(Type::Any tpe) const { return Var(Named(name, tpe), {}); }
