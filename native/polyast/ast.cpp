#include "ast.h"

#include <string>

#include "aspartame/all.hpp"
#include "fmt/format.h"

using namespace std::string_literals;
using namespace polyregion::polyast;
using namespace polyregion;
using std::string;

using namespace aspartame;

static void renderCompileEvent(std::string &out, const CompileEvent &e, size_t depth) {
  const std::string prefix(4 + depth * 2, ' ');
  fmt::format_to(std::back_inserter(out), "{}[{}, +{}ms] {}", prefix, e.epochMillis, static_cast<double>(e.elapsedNanos) / 1e6, e.name);
  if (e.data.empty()) {
    out += '\n';
  } else if (e.data.find('\n') == std::string::npos) {
    fmt::format_to(std::back_inserter(out), ": {}\n", e.data);
  } else {
    out += ":\n";
    size_t ln = 0;
    for (size_t start = 0; start < e.data.size();) {
      const size_t nl = e.data.find('\n', start);
      const auto line = e.data.substr(start, nl == std::string::npos ? std::string::npos : nl - start);
      ++ln;
      fmt::format_to(std::back_inserter(out), "{}{:>3}│{}\n", prefix, ln, line);
      if (nl == std::string::npos) break;
      start = nl + 1;
    }
    fmt::format_to(std::back_inserter(out), "{}   ╰───\n", prefix);
  }
  for (auto &child : e.items)
    renderCompileEvent(out, child, depth + 1);
}

string polyast::qualified(const Term::Select &select) {
  std::string s = select.root.symbol;
  for (auto &step : select.steps) {
    step.match_total( //
        [&](const PathStep::Field &f) {
          s += ".";
          s += f.name;
        },                                          //
        [&](const PathStep::Deref &) { s += "->"; } //
    );
  }
  return s;
}

Term::Select polyast::selectNamed(const Named &name) { return Term::Select(name, {}, name.tpe); }

Term::Select polyast::selectField(const Term::Select &base, const Named &field) {
  auto steps = base.steps;
  steps.push_back(PathStep::Field(field.symbol));
  return Term::Select(base.root, steps, field.tpe);
}

Type::Struct polyast::typeOf(const StructDef &def) {
  Vector<Type::Any> args;
  for (auto &v : def.tpeVars)
    args.push_back(Type::Var(v));
  return Type::Struct(def.name, args);
}

string polyast::repr(const CompileResult &compilation) {
  std::string out;
  auto sink = std::back_inserter(out);
  fmt::format_to(sink, "Compilation {{\n  binary: {}\n  messages: {}\n  features: {}",
                 compilation.binary ? std::to_string(compilation.binary->size()) + " bytes" : "(empty)",
                 compilation.messages.empty() ? "(none)" : "`" + compilation.messages + "`",
                 compilation.features.empty() ? "(none)" : compilation.features ^ mk_string(","));
  if (compilation.layouts.empty()) out += "\n  layouts: (none)";
  else fmt::format_to(sink, "\n  layouts:\n{}", compilation.layouts ^ mk_string("\n", [](auto &l) { return repr(l) ^ indent(4); }));
  out += "\n  events:";
  if (compilation.events.empty()) out += " (none)";
  else out += '\n';

  for (auto &e : compilation.events)
    renderCompileEvent(out, e, 0);
  out += "}";
  return out;
}

Opt<Type::Any> polyast::extractComponent(const Type::Any &t) {
  if (const auto p = t.get<Type::Ptr>()) return extractComponent(p->comp);
  if (const auto a = t.get<Type::Arr>()) return extractComponent(a->comp);
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

                       [&](const Type::Struct &) -> Opt<size_t> { return {}; }, //
                       [&](const Type::Ptr &) -> Opt<size_t> { return {}; },    //
                       [&](const Type::Arr &) -> Opt<size_t> { return {}; },    //
                       [&](const Type::Var &) -> Opt<size_t> { return {}; },    //
                       [&](const Type::Exec &) -> Opt<size_t> { return {}; });
}

Pair<size_t, Opt<size_t>> polyast::countIndirectionsAndComponentSize(const Type::Any &t, const Map<Type::Struct, StructLayout> &table) {
  if (const auto s = t.get<Type::Struct>()) return {0, table ^ get_maybe(*s) ^ map([](auto &sl) { return sl.sizeInBytes; })};
  if (const auto p = t.get<Type::Ptr>()) {
    auto [indirection, componentSize] = countIndirectionsAndComponentSize(p->comp, table);
    return {indirection + 1, componentSize};
  }
  if (const auto a = t.get<Type::Arr>()) {
    auto [indirection, componentSize] = countIndirectionsAndComponentSize(a->comp, table);
    return {indirection, componentSize};
  }
  return {0, primitiveSize(t)};
}

bool polyast::isSelfOpaque(const Type::Any &tpe) {
  if (const auto a = tpe.get<Type::Arr>()) return isSelfOpaque(a->comp);
  if (const auto p = tpe.get<Type::Ptr>()) return false;
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

Type::Ptr dsl::Ptr(const Type::Any &t, const TypeSpace::Any &s) { return Type::Ptr(t, s); }
Type::Ptr dsl::Ptr(const Type::Any &t, std::optional<int32_t>, const TypeSpace::Any &s) { return Type::Ptr(t, s); }

std::vector<Stmt::Any> dsl::whileLoop(const std::vector<Stmt::Any> &prelude, const Term::Any &cond, const std::vector<Stmt::Any> &body) {
  std::vector<Stmt::Any> result = prelude;
  std::vector<Stmt::Any> loopBody = body;
  for (const auto &s : prelude) {
    if (auto v = s.get<Stmt::Var>(); v && v->expr) {
      loopBody.push_back(Stmt::Mut(Term::Select(v->name, {}, v->name.tpe), *v->expr));
    } else loopBody.push_back(s);
  }
  result.push_back(Stmt::While(cond, loopBody));
  return result;
}
Type::Arr dsl::Arr(const Type::Any &t, int32_t length, const TypeSpace::Any &s) { return Type::Arr(t, length, s); }
Type::Struct dsl::Struct(std::string name, Vector<Type::Any> args) { return Type::Struct(Sym({std::move(name)}), std::move(args)); }

Term::Any dsl::integral(const Type::Any &tpe, unsigned long long int x) {
  auto unsupported = [](auto &&t, auto &&v) -> Term::Any {
    throw std::logic_error("Cannot create integral constant of type " + to_string(t) + " for value " + std::to_string(v));
  };
  return tpe.match_total(                                                        //
      [&](const Type::Float16 &) -> Term::Any { return Term::Float16Const(x); }, //
      [&](const Type::Float32 &) -> Term::Any { return Term::Float32Const(x); }, //
      [&](const Type::Float64 &) -> Term::Any { return Term::Float64Const(x); }, //

      [&](const Type::IntU8 &) -> Term::Any { return Term::IntU8Const(x); },   //
      [&](const Type::IntU16 &) -> Term::Any { return Term::IntU16Const(x); }, //
      [&](const Type::IntU32 &) -> Term::Any { return Term::IntU32Const(x); }, //
      [&](const Type::IntU64 &) -> Term::Any { return Term::IntU64Const(x); }, //

      [&](const Type::IntS8 &) -> Term::Any { return Term::IntS8Const(x); },   //
      [&](const Type::IntS16 &) -> Term::Any { return Term::IntS16Const(x); }, //
      [&](const Type::IntS32 &) -> Term::Any { return Term::IntS32Const(x); }, //
      [&](const Type::IntS64 &) -> Term::Any { return Term::IntS64Const(x); }, //

      [&](const Type::Nothing &t) -> Term::Any { return unsupported(t, x); }, //
      [&](const Type::Unit0 &t) -> Term::Any { return unsupported(t, x); },   //
      [&](const Type::Bool1 &) -> Term::Any { return Term::Bool1Const(x); },  //

      [&](const Type::Struct &t) -> Term::Any { return unsupported(t, x); }, //
      [&](const Type::Ptr &t) -> Term::Any { return unsupported(t, x); },    //
      [&](const Type::Arr &t) -> Term::Any { return unsupported(t, x); },    //
      [&](const Type::Var &t) -> Term::Any { return unsupported(t, x); },    //
      [&](const Type::Exec &t) -> Term::Any { return unsupported(t, x); }    //
  );
}

Term::Any dsl::fractional(const Type::Any &tpe, long double x) {
  if (tpe.is<Type::Float64>()) return Term::Float64Const(static_cast<double>(x));
  if (tpe.is<Type::Float32>()) return Term::Float32Const(static_cast<float>(x));
  if (tpe.is<Type::Float16>()) return Term::Float16Const(static_cast<float>(x));
  throw std::logic_error("Cannot create fractional constant of type " + to_string(tpe) + " for value " + std::to_string(x));
}

std::function<Term::Any(Type::Any)> dsl::operator""_(unsigned long long int x) {
  return [=](const Type::Any &t) { return integral(t, x); };
}
std::function<Term::Any(Type::Any)> dsl::operator""_(long double x) {
  return [=](const Type::Any &t) { return fractional(t, x); };
}
std::function<dsl::NamedBuilder(Type::Any)> dsl::operator""_(const char *name, size_t) {
  string name_(name);
  return [=](auto &&tpe) { return NamedBuilder{Named(name_, tpe)}; };
}

Stmt::Any dsl::let(const string &name, const Type::Any &tpe) { return Stmt::Var(Named(name, tpe), {}, /*isMutable*/ false); }
dsl::AssignmentBuilder dsl::let(const string &name) { return AssignmentBuilder{name, /*isMutable*/ false}; }
dsl::AssignmentBuilder dsl::var(const string &name) { return AssignmentBuilder{name, /*isMutable*/ true}; }

Term::Select dsl::Select(const Vector<Named> &init, const Named &last) {
  if (init.empty()) return Term::Select(last, {}, last.tpe);
  Vector<PathStep::Any> steps;
  for (size_t i = 1; i < init.size(); ++i)
    steps.push_back(PathStep::Field(init[i].symbol));
  steps.push_back(PathStep::Field(last.symbol));
  return Term::Select(init.front(), steps, last.tpe);
}

Term::Select dsl::selectFromBuilders(const Vector<NamedBuilder> &init, const Named &last) {
  Vector<Named> namedInit;
  for (auto &nb : init)
    namedInit.push_back(nb.named);
  return dsl::Select(namedInit, last);
}

Expr::IntrOp dsl::call(const Intr::Any &intr) { return Expr::IntrOp(intr); }
Expr::MathOp dsl::call(const Math::Any &intr) { return Expr::MathOp(intr); }
Expr::SpecOp dsl::call(const Spec::Any &intr) { return Expr::SpecOp(intr); }

std::function<Function(Vector<Stmt::Any>)> dsl::function(const string &name, const Vector<Arg> &args, const Type::Any &rtn,
                                                         FunctionVisibility::Any visibility, FunctionFpMode::Any fpMode, bool isEntry) {
  return [=](auto &&stmts) {
    return Function(Sym({name}), {}, /*receiver*/ {}, args, /*moduleCaptures*/ {}, /*termCaptures*/ {}, rtn, stmts, visibility, fpMode,
                    isEntry);
  };
}

Stmt::Return dsl::ret(const Expr::Any &expr) { return Stmt::Return(expr); }
Stmt::Return dsl::ret(const Term::Any &term) { return Stmt::Return(Expr::Alias(term)); }

Program dsl::program(const Vector<StructDef> &structs, const Vector<Function> &functions) {
  if (functions.empty()) throw std::logic_error("dsl::program requires at least one (entry) function");
  return Program(functions.front(), Vector<Function>(std::next(functions.begin()), functions.end()), structs, PassPhase::Initial());
}
Program dsl::program(const Function &function) { return Program(function, {}, {}, PassPhase::Initial()); }

dsl::IndexBuilder::IndexBuilder(const Index &index) : index(index) {}
dsl::IndexBuilder::operator Expr::Any() const { return index; }
Stmt::Update dsl::IndexBuilder::operator=(const Term::Any &that) const {
  // The new shape demands lhs be a Term::Select; carry whatever the Index wraps.
  auto sel = index.lhs.get<Term::Select>();
  if (!sel) throw std::logic_error("IndexBuilder requires a Term::Select lhs to materialise an Update");
  return Stmt::Update(*sel, index.idx, that);
}

dsl::NamedBuilder::NamedBuilder(const Named &named) : named(named) {}
dsl::NamedBuilder::operator Term::Any() const { return Term::Select(named, {}, named.tpe); }
dsl::NamedBuilder::operator Term::Select() const { return Term::Select(named, {}, named.tpe); }
dsl::NamedBuilder::operator Named() const { return named; }
Arg dsl::NamedBuilder::operator()() const { return Arg(named, {}); }

dsl::IndexBuilder dsl::NamedBuilder::operator[](const Term::Any &idx) const {
  if (auto arr = named.tpe.get<Type::Ptr>()) {
    return IndexBuilder(Expr::Index(Term::Select(named, {}, named.tpe), idx, arr->comp));
  }
  if (auto arr = named.tpe.get<Type::Arr>()) {
    return IndexBuilder(Expr::Index(Term::Select(named, {}, named.tpe), idx, arr->comp));
  }
  throw std::logic_error("Cannot index a reference to non-array type " + to_string(named));
}

dsl::Mut dsl::NamedBuilder::operator=(const Expr::Any &that) const { return Stmt::Mut(Term::Select(named, {}, named.tpe), that); }

dsl::AssignmentBuilder::AssignmentBuilder(const string &name, bool isMutable) : name(name), isMutable(isMutable) {}
Stmt::Any dsl::AssignmentBuilder::operator=(Term::Any rhs) const { return Stmt::Var(Named(name, rhs.tpe()), Expr::Alias(rhs), isMutable); }
Stmt::Any dsl::AssignmentBuilder::operator=(Type::Any tpe) const { return Stmt::Var(Named(name, tpe), {}, isMutable); }
Stmt::Any dsl::AssignmentBuilder::operator=(const Expr::Any &rhs) const { return Stmt::Var(Named(name, rhs.tpe()), rhs, isMutable); }
