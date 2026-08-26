#include "ast.h"

#include <string>

#include "aspartame/all.hpp"
#include "fmt/format.h"

#include "polyast_codec.h"

using namespace std::string_literals;
using namespace polyregion::polyast;
using namespace polyregion;
using std::string;

using namespace aspartame;

string polyast::fqcn(const Sym &symbol) { return symbol.fqn ^ mk_string("."); }

string polyast::canonicalName(const TypeSpace::Any &space) {
  return space.match_total([](const TypeSpace::Global &) { return ""s; }, [](const TypeSpace::Local &) { return "^Local"s; },
                           [](const TypeSpace::Private &) { return "^Private"s; },
                           [](const TypeSpace::Constant &) { return "^Constant"s; });
}

string polyast::canonicalName(const Type::Any &tpe) {
  return tpe.match_total(
      [](const Type::Float16 &) { return "F16"s; }, [](const Type::Float32 &) { return "F32"s; },
      [](const Type::Float64 &) { return "F64"s; }, [](const Type::IntU8 &) { return "U8"s; }, [](const Type::IntU16 &) { return "U16"s; },
      [](const Type::IntU32 &) { return "U32"s; }, [](const Type::IntU64 &) { return "U64"s; }, [](const Type::IntS8 &) { return "I8"s; },
      [](const Type::IntS16 &) { return "I16"s; }, [](const Type::IntS32 &) { return "I32"s; }, [](const Type::IntS64 &) { return "I64"s; },
      [](const Type::Nothing &) { return "Nothing"s; }, [](const Type::Unit0 &) { return "Unit0"s; },
      [](const Type::Bool1 &) { return "Bool1"s; },
      [](const Type::Struct &s) {
        return fmt::format("{}<{}>", fqcn(s.name), s.args ^ mk_string(",", [](const auto &arg) { return canonicalName(arg); }));
      },
      [](const Type::Ptr &p) { return fmt::format("{}*{}", canonicalName(p.comp), canonicalName(p.space)); },
      [](const Type::Arr &a) { return fmt::format("{}[{}]{}", canonicalName(a.comp), a.length, canonicalName(a.space)); },
      [](const Type::Var &v) {
        return v.exactSizeInBytes ? fmt::format("#{}:size={}", v.name, *v.exactSizeInBytes) : fmt::format("#{}", v.name);
      },
      [](const Type::Exec &e) {
        return fmt::format("<{}>({}) => {}", e.tpeVars ^ mk_string(",", [](const auto &v) { return canonicalName(v); }),
                           e.args ^ mk_string(",", [](const auto &arg) { return canonicalName(arg); }), canonicalName(e.rtn));
      },
      [](const Type::FnRef &f) { return "&" + fqcn(f.name); });
}

string polyast::signatureKey(const Signature &signature) {
  const auto types = [](const auto &xs) { return xs ^ mk_string(",", [](const auto &tpe) { return canonicalName(tpe); }); };
  return fmt::format("{}{}<{}>({})[{};{}]:{}", signature.receiver ? canonicalName(*signature.receiver) + "." : "", fqcn(signature.name),
                     types(signature.tpeVars), types(signature.args), types(signature.moduleCaptures), types(signature.termCaptures),
                     canonicalName(signature.rtn));
}

std::variant<std::string, Package> polyregion::polyast::decodePackage(const uint8_t *begin, const uint8_t *end) noexcept {
  try {
    return package_from_msgpack(begin, end);
  } catch (const std::exception &e) {
    return std::string(e.what());
  }
}

std::variant<std::string, Program> polyregion::polyast::decodeHashedProgram(const uint8_t *begin, const uint8_t *end) noexcept {
  try {
    return hashed_program_from_msgpack(begin, end);
  } catch (const std::exception &e) {
    return std::string(e.what());
  }
}

static void renderCompileEvent(std::string &out, const CompileEvent &e, size_t depth) {
  const std::string prefix(4 + depth * 2, ' ');
  fmt::format_to(std::back_inserter(out), "{}[{}, +{}ms] {}", prefix, e.epochMillis, static_cast<double>(e.elapsedNanos) / 1e6, e.name);
  if (e.data.empty()) {
    out += '\n';
  } else if (e.data ^ none_match([](char c) { return c == '\n'; })) {
    fmt::format_to(std::back_inserter(out), ": {}\n", e.data);
  } else {
    out += ":\n";
    const auto lines = e.data ^ aspartame::lines();
    for (size_t i = 0; i < lines.size(); ++i)
      fmt::format_to(std::back_inserter(out), "{}{:>3}│{}\n", prefix, i + 1, lines[i]);
    fmt::format_to(std::back_inserter(out), "{}   ╰───\n", prefix);
  }
  for (const auto &child : e.items)
    renderCompileEvent(out, child, depth + 1);
}

string polyast::qualified(const Term::Select &select) {
  std::string s = select.root.symbol;
  for (auto &step : select.steps) {
    step.match_total( //
        [&](const PathStep::Field &f) {
          s += ".";
          s += f.name;
        },                                                                         //
        [&](const PathStep::Deref &) { s += "->"; },                               //
        [&](const PathStep::Index &i) { s += "[" + std::to_string(i.idx) + "]"; }, //
        [&](const PathStep::IndexDyn &i) { s += "[" + repr(i.idx) + "]"; }         //
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
  return Type::Struct(def.name, def.tpeVars ^ map([](const auto &variable) -> Type::Any { return variable; }));
}

string polyast::repr(const CompileResult &compilation) {
  std::string out;
  auto sink = std::back_inserter(out);
  fmt::format_to(sink, "Compilation {{\n  binary: {}\n  messages: {}\n  features: {}",
                 compilation.binary ? std::to_string(compilation.binary->size()) + " bytes" : "(empty)",
                 compilation.messages.empty() ? "(none)" : "`" + compilation.messages + "`",
                 compilation.features.empty() ? "(none)" : compilation.features ^ mk_string(","));
  if (compilation.layouts.empty()) out += "\n  layouts: (none)";
  else fmt::format_to(sink, "\n  layouts:\n{}", compilation.layouts ^ mk_string("\n", [](const auto &l) { return repr(l) ^ indent(4); }));
  out += "\n  events:";
  if (compilation.events.empty()) out += " (none)";
  else out += '\n';

  for (const auto &e : compilation.events)
    renderCompileEvent(out, e, 0);
  out += "}";
  return out;
}

Opt<Type::Any> polyast::extractComponent(const Type::Any &t) {
  if (const auto p = t.get<Type::Ptr>()) return extractComponent(p->comp);
  if (const auto a = t.get<Type::Arr>()) return extractComponent(a->comp);
  return t;
}

Opt<Sym> polyast::calleeSym(const Expr::Invoke &ivk) {
  if (auto f = ivk.callee.get<Type::FnRef>()) return std::move(f->name);
  return {};
}

Sym polyast::calleeName(const Expr::Invoke &ivk) {
  if (auto s = calleeSym(ivk)) return std::move(*s);
  throw std::logic_error("callee is not a concrete function: " + repr(ivk.callee));
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
                       [&](const Type::Exec &) -> Opt<size_t> { return {}; },   //
                       [&](const Type::FnRef &) -> Opt<size_t> { return {}; });
}

Pair<size_t, Opt<size_t>> polyast::countIndirectionsAndComponentSize(const Type::Any &t, const Map<Type::Struct, StructLayout> &table) {
  if (const auto s = t.get<Type::Struct>()) return {0, table ^ get_maybe(*s) | map([](const auto &sl) { return sl.sizeInBytes; })};
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
  return sl.members ^ forall([](const auto &m) { return isSelfOpaque(m.name.tpe); });
}

bool polyast::isOpaque(const StructLayout &sl, const std::unordered_map<Type::Struct, StructLayout> &table) {
  return isSelfOpaque(sl)
         && sl.members ^ forall([&](const auto &m) {
              return m.name.tpe.template get<Type::Struct>() //
                     ^ fold(
                         [&](const auto &s) {
                           return table ^ get_maybe(s) | map([&](const auto &x) { return isOpaque(x, table); }) | get_or_else(false);
                         },
                         []() { return true; });
            });
}

// ====================

Type::Ptr dsl::Ptr(const Type::Any &t, const TypeSpace::Any &s) { return Type::Ptr(t, s); }

std::vector<Stmt::Any> dsl::whileLoop(const std::vector<Stmt::Any> &prelude, const Term::Any &cond, const std::vector<Stmt::Any> &body) {
  std::vector<Stmt::Any> result = prelude;
  const auto loopBody = body ^ concat(prelude ^ map([](const auto &s) -> Stmt::Any {
                                        if (auto v = s.template get<Stmt::Var>(); v && v->expr)
                                          return Stmt::Mut(Term::Select(v->name, {}, v->name.tpe), *v->expr);
                                        return s;
                                      }));
  result.push_back(Stmt::While(cond, loopBody));
  return result;
}
Type::Arr dsl::Arr(const Type::Any &t, int32_t length, const TypeSpace::Any &s) { return Type::Arr(t, length, s); }
Type::Struct dsl::Struct(std::string name, Vector<Type::Any> args) { return Type::Struct(Sym({std::move(name)}), std::move(args)); }

Term::Any dsl::integral(const Type::Any &tpe, unsigned long long int x) {
  auto unsupported = [](const auto &t, const auto &v) -> Term::Any {
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
      [&](const Type::Exec &t) -> Term::Any { return unsupported(t, x); },   //
      [&](const Type::FnRef &t) -> Term::Any { return unsupported(t, x); }   //
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
  return [=](const auto &tpe) { return NamedBuilder{Named(name_, tpe)}; };
}

Stmt::Any dsl::let(const string &name, const Type::Any &tpe) { return Stmt::Var(Named(name, tpe), {}, /*isMutable*/ false); }
dsl::AssignmentBuilder dsl::let(const string &name) { return AssignmentBuilder{name, /*isMutable*/ false}; }
dsl::AssignmentBuilder dsl::var(const string &name) { return AssignmentBuilder{name, /*isMutable*/ true}; }

Term::Select dsl::Select(const Vector<Named> &init, const Named &last) {
  if (init.empty()) return Term::Select(last, {}, last.tpe);
  auto steps = init | drop(1) | map([](const auto &n) -> PathStep::Any { return PathStep::Field(n.symbol); }) | to_vector();
  steps.push_back(PathStep::Field(last.symbol));
  return Term::Select(init.front(), steps, last.tpe);
}

Term::Select dsl::selectFromBuilders(const Vector<NamedBuilder> &init, const Named &last) {
  return dsl::Select(init ^ map([](const auto &nb) { return nb.named; }), last);
}

Expr::IntrOp dsl::call(const Intr::Any &intr) { return Expr::IntrOp(intr); }
Expr::MathOp dsl::call(const Math::Any &intr) { return Expr::MathOp(intr); }
Expr::SpecOp dsl::call(const Spec::Any &intr) { return Expr::SpecOp(intr); }

std::function<Function(Vector<Stmt::Any>)> dsl::function(const string &name, const Vector<Arg> &args, const Type::Any &rtn,
                                                         FunctionVisibility::Any visibility, FunctionFpMode::Any fpMode,
                                                         CallConvention::Any convention) {
  return [=](const auto &stmts) {
    return Function(
        FunctionDecl(Sym({name}), {}, /*receiver*/ {}, args, /*moduleCaptures*/ {}, /*termCaptures*/ {}, rtn, FunctionAffinity::Offload()),
        stmts, visibility, fpMode, convention);
  };
}

Stmt::Return dsl::ret(const Expr::Any &expr) { return Stmt::Return(expr); }
Stmt::Return dsl::ret(const Term::Any &term) { return Stmt::Return(Expr::Alias(term)); }

Program dsl::program(const Vector<StructDef> &structs, const Vector<Function> &functions) {
  if (functions.empty()) throw std::logic_error("dsl::program requires at least one (entry) function");
  return Program(functions.front(), Vector<Function>(std::next(functions.begin()), functions.end()), structs, PassPhase::Initial(), {});
}
Program dsl::program(const Function &function) { return Program(function, {}, {}, PassPhase::Initial(), {}); }

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
