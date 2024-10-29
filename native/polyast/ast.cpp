#include <iomanip>
#include <string>

#include "ast.h"

#include "aspartame/all.hpp"
#include "fmt/core.h"

using namespace std::string_literals;
using namespace polyregion::polyast;
using namespace polyregion;
using std::string;

using namespace aspartame;

[[nodiscard]] string polyast::repr(const SourcePosition &pos) { return fmt::format("{}:{}{}", pos.file, pos.line, pos.col ^ mk_string()); }

[[nodiscard]] string polyast::repr(const Type::Any &type) {
  return type.match_total(                          //
      [](const Type::Float16 &) { return "f16"s; }, //
      [](const Type::Float32 &) { return "f32"s; }, //
      [](const Type::Float64 &) { return "f64"s; }, //

      [](const Type::IntS8 &) { return "i8"s; },   //
      [](const Type::IntS16 &) { return "i16"s; }, //
      [](const Type::IntS32 &) { return "i32"s; }, //
      [](const Type::IntS64 &) { return "i64"s; }, //

      [](const Type::IntU8 &) { return "u8"s; },   //
      [](const Type::IntU16 &) { return "u16"s; }, //
      [](const Type::IntU32 &) { return "u32"s; }, //
      [](const Type::IntU64 &) { return "u64"s; }, //

      [](const Type::Unit0 &) { return "Unit"s; },                      //
      [](const Type::Bool1 &) { return "Bool"s; },                      //
      [](const Type::Nothing &) { return "Nothing"s; },                 //
      [](const Type::Struct &x) { return fmt::format("@{}", x.name); }, //
      [](const Type::Ptr &x) {
        return fmt::format("Ptr[{}{}]{}", repr(x.component), x.length ? "*" + std::to_string(*x.length) : "", repr(x.space));
      },
      [](const Type::Annotated &x) {
        return fmt::format("{} /*{}; {}*/", repr(x.tpe), x.pos ^ mk_string("", show_repr), x.comment ^ get_or_else(""));
      } //
  );
}

[[nodiscard]] string polyast::repr(const Named &x) { return fmt::format("{}: {}", x.symbol, repr(x.tpe)); }

[[nodiscard]] string polyast::repr(const Intr::Any &expr) {
  return expr.match_total(                                                                    //
      [](const Intr::BNot &x) { return fmt::format("'~{}", repr(x.x)); },                     //
      [](const Intr::LogicNot &x) { return fmt::format("'!{}", repr(x.x)); },                 //
      [](const Intr::Pos &x) { return fmt::format("'+{}", repr(x.x)); },                      //
      [](const Intr::Neg &x) { return fmt::format("'-{}", repr(x.x)); },                      //
      [](const Intr::Add &x) { return fmt::format("{} '+ {}", repr(x.x), repr(x.y)); },       //
      [](const Intr::Sub &x) { return fmt::format("{} '- {}", repr(x.x), repr(x.y)); },       //
      [](const Intr::Mul &x) { return fmt::format("{} '* {}", repr(x.x), repr(x.y)); },       //
      [](const Intr::Div &x) { return fmt::format("{} '/ {}", repr(x.x), repr(x.y)); },       //
      [](const Intr::Rem &x) { return fmt::format("{} '% {}", repr(x.x), repr(x.y)); },       //
      [](const Intr::Min &x) { return fmt::format("'min({}, {})", repr(x.x), repr(x.y)); },   //
      [](const Intr::Max &x) { return fmt::format("'max({}, {})", repr(x.x), repr(x.y)); },   //
      [](const Intr::BAnd &x) { return fmt::format("{} '& {}", repr(x.x), repr(x.y)); },      //
      [](const Intr::BOr &x) { return fmt::format("{} '| {}", repr(x.x), repr(x.y)); },       //
      [](const Intr::BXor &x) { return fmt::format("{} '^ {}", repr(x.x), repr(x.y)); },      //
      [](const Intr::BSL &x) { return fmt::format("{} '<< {}", repr(x.x), repr(x.y)); },      //
      [](const Intr::BSR &x) { return fmt::format("{} '>> {}", repr(x.x), repr(x.y)); },      //
      [](const Intr::BZSR &x) { return fmt::format("{} '>>> {}", repr(x.x), repr(x.y)); },    //
      [](const Intr::LogicAnd &x) { return fmt::format("{} '&& {}", repr(x.x), repr(x.y)); }, //
      [](const Intr::LogicOr &x) { return fmt::format("{} '|| {}", repr(x.x), repr(x.y)); },  //
      [](const Intr::LogicEq &x) { return fmt::format("{} '== {}", repr(x.x), repr(x.y)); },  //
      [](const Intr::LogicNeq &x) { return fmt::format("{} '!= {}", repr(x.x), repr(x.y)); }, //
      [](const Intr::LogicLte &x) { return fmt::format("{} '<= {}", repr(x.x), repr(x.y)); }, //
      [](const Intr::LogicGte &x) { return fmt::format("{} '>= {}", repr(x.x), repr(x.y)); }, //
      [](const Intr::LogicLt &x) { return fmt::format("{} '< {}", repr(x.x), repr(x.y)); },   //
      [](const Intr::LogicGt &x) { return fmt::format("{} '> {}", repr(x.x), repr(x.y)); });
}

[[nodiscard]] string polyast::repr(const Spec::Any &expr) {
  return expr.match_total(                                                                         //
      [](const Spec::Assert &x) { return "'assert"s; },                                            //
      [](const Spec::GpuBarrierGlobal &x) { return "'gpuBarrierGlobal"s; },                        //
      [](const Spec::GpuBarrierLocal &x) { return "'gpuBarrierLocal"s; },                          //
      [](const Spec::GpuBarrierAll &x) { return "'gpuBarrierAll"s; },                              //
      [](const Spec::GpuFenceGlobal &x) { return "'gpuFenceGlobal"s; },                            //
      [](const Spec::GpuFenceLocal &x) { return "'gpuFenceLocal"s; },                              //
      [](const Spec::GpuFenceAll &x) { return "'gpuFenceAll"s; },                                  //
      [](const Spec::GpuGlobalIdx &x) { return fmt::format("'gpuGlobalIdx({})", repr(x.dim)); },   //
      [](const Spec::GpuGlobalSize &x) { return fmt::format("'gpuGlobalSize({})", repr(x.dim)); }, //
      [](const Spec::GpuGroupIdx &x) { return fmt::format("'gpuGroupIdx({})", repr(x.dim)); },     //
      [](const Spec::GpuGroupSize &x) { return fmt::format("'gpuGroupSize({})", repr(x.dim)); },   //
      [](const Spec::GpuLocalIdx &x) { return fmt::format("'gpuLocalIdx({})", repr(x.dim)); },     //
      [](const Spec::GpuLocalSize &x) { return fmt::format("'gpuLocalSize({})", repr(x.dim)); });
}

[[nodiscard]] string polyast::repr(const Math::Any &expr) {
  return expr.match_total(                                                                      //
      [](const Math::Abs &x) { return fmt::format("'abs({})", repr(x.x)); },                    //
      [](const Math::Sin &x) { return fmt::format("'sin({})", repr(x.x)); },                    //
      [](const Math::Cos &x) { return fmt::format("'cos({})", repr(x.x)); },                    //
      [](const Math::Tan &x) { return fmt::format("'tan({})", repr(x.x)); },                    //
      [](const Math::Asin &x) { return fmt::format("'asin({})", repr(x.x)); },                  //
      [](const Math::Acos &x) { return fmt::format("'acos({})", repr(x.x)); },                  //
      [](const Math::Atan &x) { return fmt::format("'atan({})", repr(x.x)); },                  //
      [](const Math::Sinh &x) { return fmt::format("'sinh({})", repr(x.x)); },                  //
      [](const Math::Cosh &x) { return fmt::format("'cosh({})", repr(x.x)); },                  //
      [](const Math::Tanh &x) { return fmt::format("'tanh({})", repr(x.x)); },                  //
      [](const Math::Signum &x) { return fmt::format("'signum({})", repr(x.x)); },              //
      [](const Math::Round &x) { return fmt::format("'round({})", repr(x.x)); },                //
      [](const Math::Ceil &x) { return fmt::format("'ceil({})", repr(x.x)); },                  //
      [](const Math::Floor &x) { return fmt::format("'floor({})", repr(x.x)); },                //
      [](const Math::Rint &x) { return fmt::format("'rint({})", repr(x.x)); },                  //
      [](const Math::Sqrt &x) { return fmt::format("'sqrt({})", repr(x.x)); },                  //
      [](const Math::Cbrt &x) { return fmt::format("'cbrt({})", repr(x.x)); },                  //
      [](const Math::Exp &x) { return fmt::format("'exp({})", repr(x.x)); },                    //
      [](const Math::Expm1 &x) { return fmt::format("'expm1({})", repr(x.x)); },                //
      [](const Math::Log &x) { return fmt::format("'log({})", repr(x.x)); },                    //
      [](const Math::Log1p &x) { return fmt::format("'log1p({})", repr(x.x)); },                //
      [](const Math::Log10 &x) { return fmt::format("'log10({})", repr(x.x)); },                //
      [](const Math::Pow &x) { return fmt::format("'pow({}, {})", repr(x.x), repr(x.y)); },     //
      [](const Math::Atan2 &x) { return fmt::format("'atan2({}, {})", repr(x.x), repr(x.y)); }, //
      [](const Math::Hypot &x) { return fmt::format("'hypot({}, {})", repr(x.x), repr(x.y)); });
}

[[nodiscard]] string polyast::repr(const Expr::Any &expr) {
  return expr.match_total(                                                         //
      [](const Expr::Float16Const &x) { return fmt::format("f16({})", x.value); }, //
      [](const Expr::Float32Const &x) { return fmt::format("f32({})", x.value); }, //
      [](const Expr::Float64Const &x) { return fmt::format("f64({})", x.value); }, //

      [](const Expr::IntU8Const &x) { return fmt::format("u8({})", x.value); },   //
      [](const Expr::IntU16Const &x) { return fmt::format("u16({})", x.value); }, //
      [](const Expr::IntU32Const &x) { return fmt::format("u32({})", x.value); }, //
      [](const Expr::IntU64Const &x) { return fmt::format("u64({})", x.value); }, //
      [](const Expr::IntS8Const &x) { return fmt::format("i8({})", x.value); },   //
      [](const Expr::IntS16Const &x) { return fmt::format("i16({})", x.value); }, //
      [](const Expr::IntS32Const &x) { return fmt::format("i32({})", x.value); }, //
      [](const Expr::IntS64Const &x) { return fmt::format("i64({})", x.value); }, //

      [](const Expr::Bool1Const &x) { return fmt::format("bool({})", x.value ? "true" : "false"); }, //
      [](const Expr::Unit0Const &x) { return "unit()"s; },

      [](const Expr::SpecOp &x) { return repr(x.op); }, //
      [](const Expr::MathOp &x) { return repr(x.op); }, //
      [](const Expr::IntrOp &x) { return repr(x.op); }, //

      [](const Expr::Select &x) { return x.init | append(x.last) | mk_string(".", show_repr); },
      [](const Expr::Poison &x) { return fmt::format("Poison({})", repr(x.tpe)); },

      [](const Expr::Cast &x) { return "(" + repr(x.from) + ".to[" + repr(x.as) + "])"; },
      [](const Expr::Index &x) { return repr(x.lhs) + "[" + repr(x.idx) + "]:" + repr(x.component); },
      [](const Expr::RefTo &x) {
        string str = "&(" + repr(x.lhs) + ")";
        if (x.idx) str += "[" + repr(*x.idx) + "]";
        return str + ": " + repr(x.tpe);
      },
      [](const Expr::Alloc &x) { return "new [" + repr(x.tpe) + "*" + repr(x.size) + "]"; },
      [](const Expr::Invoke &x) { return x.name + (x.args ^ mk_string("(", ",", ")", show_repr)) + ": " + repr(x.tpe); },

      [](const Expr::Annotated &x) {
        return fmt::format("{} /*{}; {}*/", repr(x.expr), x.pos ^ mk_string("", show_repr), x.comment ^ get_or_else(""));
      }
      //
  );
}

[[nodiscard]] string polyast::repr(const Stmt::Any &stmt) {
  return stmt.match_total( //
      [](const Stmt::Block &x) { return "{ \n" + (x.stmts ^ mk_string("\n", show_repr) ^ indent(2)) + "}"; },
      [](const Stmt::Comment &x) { return "/* " + x.value + " */"; },
      [](const Stmt::Var &x) { return "var " + repr(x.name) + " = " + (x.expr ? repr(*x.expr) : "_"); },
      [](const Stmt::Mut &x) { return repr(x.name) + " := " + repr(x.expr); },
      [](const Stmt::Update &x) { return repr(x.lhs) + "[" + repr(x.idx) + "] = " + repr(x.value); },
      [](const Stmt::While &x) {
        const auto tests = x.tests ^ mk_string("\n", show_repr);
        return "while({" + tests + ";" + repr(x.cond) + "}){\n" + (x.body ^ mk_string("\n", show_repr) ^ indent(2)) + "\n}";
      },
      [](const Stmt::Break &) { return "break;"s; },   //
      [](const Stmt::Cont &) { return "continue;"s; }, //
      [](const Stmt::Cond &x) {
        const auto elseStmts = x.falseBr.empty() //
                                   ? "\n}"
                                   : "\n} else {\n" + (x.falseBr ^ mk_string("\n", show_repr) ^ indent(2)) + "\n}";
        return "if(" + repr(x.cond) + ") { \n" + (x.trueBr ^ mk_string("\n", show_repr) ^ indent(2)) + elseStmts;
      },
      [](const Stmt::Return &x) { return "return " + repr(x.value); },
      [](const Stmt::Annotated &x) {
        return fmt::format("{} /*{}; {}*/", repr(x.expr), x.pos ^ mk_string("", show_repr), x.comment ^ get_or_else(""));
      }

  );
}

[[nodiscard]] string polyast::repr(const Arg &arg) { return repr(arg.named); }

[[nodiscard]] string polyast::repr(const TypeSpace::Any &space) {
  return space.match_total( //
      [&](const TypeSpace::Global &x) { return "^Global"; }, [&](const TypeSpace::Local &x) { return "^Local"; });
}

[[nodiscard]] string polyast::repr(const Signature &s) {
  return fmt::format("{}({}): {}", s.name, s.args ^ mk_string(", ", show_repr), repr(s.rtn));
}

[[nodiscard]] string polyast::repr(const Function &fn) {
  string str;
  str += fn.name;
  str += fn.args ^ mk_string("(", ",", ")", show_repr);
  str += ": " + repr(fn.rtn);
  str += " = {\n" + (fn.body ^ mk_string("\n", show_repr) ^ indent(2)) + "\n}";
  return str;
}

[[nodiscard]] string polyast::repr(const StructDef &def) {
  return fmt::format("struct {} {}", def.name, def.members ^ mk_string("{", ", ", "}", show_repr));
}

[[nodiscard]] string polyast::repr(const Program &program) {
  return fmt::format("{}\n{}\n", program.structs ^ mk_string("\n", show_repr), program.functions ^ mk_string("\n", show_repr));
}

string polyast::qualified(const Expr::Select &select) {
  return select.init | append(select.last) | mk_string(".", [](auto &x) { return x.symbol; });
}

std::vector<Named> polyast::path(const Expr::Select &select) { return select.init | append(select.last) | to_vector(); }

Named polyast::head(const Expr::Select &select) { return select.init.empty() ? select.last : select.init.front(); }

std::vector<Named> polyast::tail(const Expr::Select &select) {
  if (select.init.empty()) return {select.last};
  else {
    std::vector<Named> xs(std::next(select.init.begin()), select.init.end());
    xs.push_back(select.last);
    return xs;
  }
}

std::pair<Named, std::vector<Named>> polyast::uncons(const Expr::Select &select) {
  if (select.init.empty()) return {{select.last}, {}};
  else {
    std::vector<Named> xs(std::next(select.init.begin()), select.init.end());
    xs.push_back(select.last);
    return {select.init.front(), xs};
  }
}

string polyast::repr(const polyast::CompileResult &compilation) {
  std::ostringstream os;
  os << "Compilation {"                                                                                            //
     << "\n  binary: " << (compilation.binary ? std::to_string(compilation.binary->size()) + " bytes" : "(empty)") //
     << "\n  messages: `" << compilation.messages << "`"                                                           //
     << "\n  features: `" << (compilation.features ^ mk_string(",")) << "`"                                        //
     << "\n  layouts: `" << (compilation.layouts ^ mk_string("\n    ")) << "`"                                     //
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

Type::Ptr dsl::Ptr(const Type::Any &t, std::optional<int32_t> l, const ::TypeSpace::Any &s) { return Tpe::Ptr(t, l, s); }
// Type::Struct dsl::Struct(string name,  std::vector<Type::Any> members) { return {name,   args, {}}; }
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
Expr::IntrOp dsl::invoke(const Intr::Any &intr) { return IntrOp(intr); }
Expr::MathOp dsl::invoke(const Math::Any &intr) { return MathOp(intr); }
Expr::SpecOp dsl::invoke(const Spec::Any &intr) { return SpecOp(intr); }
std::function<Function(std::vector<Stmt::Any>)> dsl::function(const string &name, const std::vector<Arg> &args, const Type::Any &rtn,
                                                              const std::set<FunctionAttr::Any> &attrs) {
  return [=](auto &&stmts) { return Function(name, args, rtn, stmts, attrs); };
}
Stmt::Return dsl::ret(const Expr::Any &expr) { return Return(expr); }
Program dsl::program(const std::vector<StructDef> &structs, const std::vector<Function> &functions) { return {structs, functions}; }
Program dsl::program(const Function &function) { return Program({}, {function}); }

dsl::IndexBuilder::IndexBuilder(const Index &index) : index(index) {}
dsl::IndexBuilder::operator Expr::Any() const { return index; }
Stmt::Update dsl::IndexBuilder::operator=(const Expr::Any &term) const { return {index.lhs, index.idx, term}; }
dsl::NamedBuilder::NamedBuilder(const Named &named) : named(named) {}
dsl::NamedBuilder::operator Expr::Any() const { return Select({}, named); }
// dsl::NamedBuilder::operator const Expr::Any() const { return Alias(Select({}, named)); }
dsl::NamedBuilder::operator Named() const { return named; }
Arg dsl::NamedBuilder::operator()() const { return Arg(named, {}); }

dsl::IndexBuilder dsl::NamedBuilder::operator[](const Expr::Any &idx) const {
  if (auto arr = named.tpe.get<Type::Ptr>()) {
    return IndexBuilder({Select({}, named), idx, arr->component});
  }
  throw std::logic_error("Cannot index a reference to non-array type" + to_string(named));
}
dsl::Mut dsl::NamedBuilder::operator=(const Expr::Any &that) const { return Mut(Select({}, named), that); }

dsl::AssignmentBuilder::AssignmentBuilder(const string &name) : name(name) {}
Stmt::Any dsl::AssignmentBuilder::operator=(Expr::Any rhs) const { return Var(Named(name, rhs.tpe()), {rhs}); }
Stmt::Any dsl::AssignmentBuilder::operator=(Type::Any tpe) const { return Var(Named(name, tpe), {}); }
