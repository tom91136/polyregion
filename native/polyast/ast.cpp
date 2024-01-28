#include <string>
#include <iomanip>

#include "ast.h"
#include "utils.hpp"

using namespace std::string_literals;
using namespace polyregion::polyast;
using namespace polyregion;
using std::string;

[[nodiscard]] string polyast::repr(const Sym &sym) {
  return mk_string<string>(
      sym.fqn, [](auto &&x) { return x; }, ".");
}

[[nodiscard]] string polyast::repr(const InvokeSignature &ivk) {
  string str;
  if (ivk.receiver) str += repr(*ivk.receiver) + ".";
  str += repr(ivk.name);
  str += "<" +
         mk_string<Type::Any>(
             ivk.tpeVars, [](auto &x) { return repr(x); }, ", ") +
         ">";
  str += "(" +
         mk_string<Type::Any>(
             ivk.args, [&](auto x) { return repr(x); }, ",") +
         ")";
  str += "[" +
         mk_string<Type::Any>(
             ivk.captures, [&](auto x) { return repr(x); }, ",") +
         "]";
  str += ": " + repr(ivk.rtn);
  return str;
}

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

      [](const Type::Unit0 &) { return "Unit"s; },                             //
      [](const Type::Bool1 &) { return "Bool"s; },                             //
      [](const Type::Nothing &) { return "Nothing"s; },                        //
      [](const Type::Struct &x) {
        string args = "<";
        for (size_t i = 0; i < x.tpeVars.size(); ++i) {
          args += x.tpeVars[i];
          args += "=";
          args += repr(x.args[i]);
        }
        args += ">";
        auto parents = mk_string<Sym>(
            x.parents, [](auto &x) { return repr(x); }, ",");
        return "@" + repr(x.name) + args + (x.parents.empty() ? "" : "<:{" + parents + "}");
      },                                                                   //
      [](const Type::Ptr &x) { return "Ptr[" + repr(x.component) + (x.length ? "*" + std::to_string(*x.length) : "") + "]"; }, //
      [](const Type::Var &x) { return "Var[" + x.name + "]"; },                //
      [](const Type::Exec &) { return "Exec[???]"s; }                          //
  );
}

[[nodiscard]] string polyast::repr(const Named &path) { return "(" + path.symbol + ": " + repr(path.tpe) + ")"; }

[[nodiscard]] string polyast::repr(const Term::Any &ref) {
  return ref.match_total(
      [](const Term::Select &x) {
        return x.init.empty() //
                   ? repr(x.last)
                   : mk_string<Named>(
                         x.init, [&](auto x) { return repr(x); }, ".") +
                         "." + repr(x.last);
      },
      [](const Term::Poison &x) { return "Poison(" + repr(x.tpe) + ")"; },

      [](const Term::Float16Const &x) { return "f16(" + std::to_string(x.value) + ")"; }, //
      [](const Term::Float32Const &x) { return "f32(" + std::to_string(x.value) + ")"; }, //
      [](const Term::Float64Const &x) { return "f64(" + std::to_string(x.value) + ")"; }, //

      [](const Term::IntU8Const &x) { return "u8(" + std::to_string(x.value) + ")"; },   //
      [](const Term::IntU16Const &x) { return "u16(" + std::to_string(x.value) + ")"; }, //
      [](const Term::IntU32Const &x) { return "u32(" + std::to_string(x.value) + ")"; }, //
      [](const Term::IntU64Const &x) { return "u64(" + std::to_string(x.value) + ")"; }, //
      [](const Term::IntS8Const &x) { return "i8(" + std::to_string(x.value) + ")"; },   //
      [](const Term::IntS16Const &x) { return "i16(" + std::to_string(x.value) + ")"; }, //
      [](const Term::IntS32Const &x) { return "i32(" + std::to_string(x.value) + ")"; }, //
      [](const Term::IntS64Const &x) { return "i64(" + std::to_string(x.value) + ")"; }, //

      [](const Term::Bool1Const &x) { return "bool(" + (x.value ? "true"s : "false"s) + ")"; }, //
      [](const Term::Unit0Const &x) { return "unit()"s; }                                       //

  );
}

[[nodiscard]] string polyast::repr(const Intr::Any &expr) {
  return expr.match_total(                                                             //
      [](const Intr::BNot &x) { return "'~" + repr(x.x); },                            //
      [](const Intr::LogicNot &x) { return "'!" + repr(x.x); },                        //
      [](const Intr::Pos &x) { return "'+" + repr(x.x); },                             //
      [](const Intr::Neg &x) { return "'-" + repr(x.x); },                             //
      [](const Intr::Add &x) { return repr(x.x) + " '+ " + repr(x.y); },                 //
      [](const Intr::Sub &x) { return repr(x.x) + " '- " + repr(x.y); },                 //
      [](const Intr::Mul &x) { return repr(x.x) + " '* " + repr(x.y); },                 //
      [](const Intr::Div &x) { return repr(x.x) + " '/ " + repr(x.y); },                 //
      [](const Intr::Rem &x) { return repr(x.x) + " '% " + repr(x.y); },                 //
      [](const Intr::Min &x) { return "'min(" + repr(x.x) + ", " + repr(x.y) + ")"; }, //
      [](const Intr::Max &x) { return "'max(" + repr(x.x) + ", " + repr(x.y) + ")"; }, //
      [](const Intr::BAnd &x) { return repr(x.x) + " '& " + repr(x.y); },                //
      [](const Intr::BOr &x) { return repr(x.x) + " '| " + repr(x.y); },                 //
      [](const Intr::BXor &x) { return repr(x.x) + " '^ " + repr(x.y); },                //
      [](const Intr::BSL &x) { return repr(x.x) + " '<< " + repr(x.y); },                //
      [](const Intr::BSR &x) { return repr(x.x) + " '>> " + repr(x.y); },                //
      [](const Intr::BZSR &x) { return repr(x.x) + " '>>> " + repr(x.y); },              //
      [](const Intr::LogicAnd &x) { return repr(x.x) + " '&& " + repr(x.y); },           //
      [](const Intr::LogicOr &x) { return repr(x.x) + " '|| " + repr(x.y); },            //
      [](const Intr::LogicEq &x) { return repr(x.x) + " '== " + repr(x.y); },            //
      [](const Intr::LogicNeq &x) { return repr(x.x) + " '!= " + repr(x.y); },           //
      [](const Intr::LogicLte &x) { return repr(x.x) + " '<= " + repr(x.y); },           //
      [](const Intr::LogicGte &x) { return repr(x.x) + " '>= " + repr(x.y); },           //
      [](const Intr::LogicLt &x) { return repr(x.x) + " '< " + repr(x.y); },             //
      [](const Intr::LogicGt &x) { return repr(x.x) + " '> " + repr(x.y); });
}

[[nodiscard]] string polyast::repr(const Spec::Any &expr) {
  return expr.match_total(                                                                //
      [](const Spec::Assert &x) { return "'assert"s; },                                   //
      [](const Spec::GpuBarrierGlobal &x) { return "'gpuBarrierGlobal"s; },               //
      [](const Spec::GpuBarrierLocal &x) { return "'gpuBarrierLocal"s; },                 //
      [](const Spec::GpuBarrierAll &x) { return "'gpuBarrierAll"s; },                     //
      [](const Spec::GpuFenceGlobal &x) { return "'gpuFenceGlobal"s; },                   //
      [](const Spec::GpuFenceLocal &x) { return "'gpuFenceLocal"s; },                     //
      [](const Spec::GpuFenceAll &x) { return "'gpuFenceAll"s; },                         //
      [](const Spec::GpuGlobalIdx &x) { return "'gpuGlobalIdx(" + repr(x.dim) + ")"; },   //
      [](const Spec::GpuGlobalSize &x) { return "'gpuGlobalSize(" + repr(x.dim) + ")"; }, //
      [](const Spec::GpuGroupIdx &x) { return "'gpuGroupIdx(" + repr(x.dim) + ")"; },     //
      [](const Spec::GpuGroupSize &x) { return "'gpuGroupSize(" + repr(x.dim) + ")"; },   //
      [](const Spec::GpuLocalIdx &x) { return "'gpuLocalIdx(" + repr(x.dim) + ")"; },     //
      [](const Spec::GpuLocalSize &x) { return "'gpuLocalSize(" + repr(x.dim) + ")"; });
}

[[nodiscard]] string polyast::repr(const Math::Any &expr) {
  return expr.match_total(                                                                 //
      [](const Math::Abs &x) { return "'abs(" + repr(x.x) + ")"; },                        //
      [](const Math::Sin &x) { return "'sin(" + repr(x.x) + ")"; },                        //
      [](const Math::Cos &x) { return "'cos(" + repr(x.x) + ")"; },                        //
      [](const Math::Tan &x) { return "'tan(" + repr(x.x) + ")"; },                        //
      [](const Math::Asin &x) { return "'asin(" + repr(x.x) + ")"; },                      //
      [](const Math::Acos &x) { return "'acos(" + repr(x.x) + ")"; },                      //
      [](const Math::Atan &x) { return "'atan(" + repr(x.x) + ")"; },                      //
      [](const Math::Sinh &x) { return "'sinh(" + repr(x.x) + ")"; },                      //
      [](const Math::Cosh &x) { return "'cosh(" + repr(x.x) + ")"; },                      //
      [](const Math::Tanh &x) { return "'tanh(" + repr(x.x) + ")"; },                      //
      [](const Math::Signum &x) { return "'signum(" + repr(x.x) + ")"; },                  //
      [](const Math::Round &x) { return "'round(" + repr(x.x) + ")"; },                    //
      [](const Math::Ceil &x) { return "'ceil(" + repr(x.x) + ")"; },                      //
      [](const Math::Floor &x) { return "'floor(" + repr(x.x) + ")"; },                    //
      [](const Math::Rint &x) { return "'rint(" + repr(x.x) + ")"; },                      //
      [](const Math::Sqrt &x) { return "'sqrt(" + repr(x.x) + ")"; },                      //
      [](const Math::Cbrt &x) { return "'cbrt(" + repr(x.x) + ")"; },                      //
      [](const Math::Exp &x) { return "'exp(" + repr(x.x) + ")"; },                        //
      [](const Math::Expm1 &x) { return "'expm1(" + repr(x.x) + ")"; },                    //
      [](const Math::Log &x) { return "'log(" + repr(x.x) + ")"; },                        //
      [](const Math::Log1p &x) { return "'log1p(" + repr(x.x) + ")"; },                    //
      [](const Math::Log10 &x) { return "'log10(" + repr(x.x) + ")"; },                    //
      [](const Math::Pow &x) { return "'pow(" + repr(x.x) + ", " + repr(x.y) + ")"; },     //
      [](const Math::Atan2 &x) { return "'atan2(" + repr(x.x) + ", " + repr(x.y) + ")"; }, //
      [](const Math::Hypot &x) { return "'hypot(" + repr(x.x) + ", " + repr(x.y) + ")"; });
}

[[nodiscard]] string polyast::repr(const Expr::Any &expr) {
  return expr.match_total(                              //
      [](const Expr::SpecOp &x) { return repr(x.op); }, //
      [](const Expr::MathOp &x) { return repr(x.op); }, //
      [](const Expr::IntrOp &x) { return repr(x.op); }, //
      [](const Expr::Cast &x) { return "(" + repr(x.from) + ".to[" + repr(x.as) + "])"; },
      [](const Expr::Alias &x) { return "(~>" + repr(x.ref) + ")"; },
      [](const Expr::Invoke &x) {
        return (x.receiver ? repr(*x.receiver) : "<module>") + "." + repr(x.name) + "(" +
               mk_string<Term::Any>(
                   x.args, [&](auto x) { return repr(x); }, ",") +
               ")" + ": " + repr(x.tpe);
      },
      [](const Expr::Index &x) { return repr(x.lhs) + "[" + repr(x.idx) + "]:" + repr(x.component); },
      [](const Expr::RefTo &x) {
        string str = "&(" + repr(x.lhs) + ")";
        if (x.idx) str += "[" + repr(*x.idx) + "]";
        return str + ": " + repr(x.tpe);
      },
      [](const Expr::Alloc &x) { return "new [" + repr(x.tpe) + "*" + repr(x.size) + "]"; });
}

[[nodiscard]] string polyast::repr(const Stmt::Any &stmt) {
  return stmt.match_total( //
      [](const Stmt::Block &x) {
        return "{ \n" +
               indent(2, mk_string<Stmt::Any>(
                             x.stmts, [&](auto x) { return repr(x); }, "\n")) +
               "}";
      },
      [](const Stmt::Comment &x) { return "/* " + x.value + " */"; },
      [](const Stmt::Var &x) { return "var " + repr(x.name) + " = " + (x.expr ? repr(*x.expr) : "_"); },
      [](const Stmt::Mut &x) { return repr(x.name) + " := " + repr(x.expr); },
      [](const Stmt::Update &x) { return repr(x.lhs) + "[" + repr(x.idx) + "] = " + repr(x.value); },
      [](const Stmt::While &x) {
        auto tests = mk_string<Stmt::Any>(
            x.tests, [&](auto x) { return repr(x); }, "\n");

        return "while({" + tests + ";" + repr(x.cond) + "}){\n" +
               indent(2, mk_string<Stmt::Any>(
                             x.body, [&](auto x) { return repr(x); }, "\n")) +
               "\n}";
      },
      [](const Stmt::Break &x) { return "break;"s; }, [](const Stmt::Cont &x) { return "continue;"s; },
      [](const Stmt::Cond &x) {
        auto elseStmts = x.falseBr.empty() //
                             ? "\n}"
                             : "\n} else {\n" +
                                   indent(2, mk_string<Stmt::Any>(
                                                 x.falseBr, [&](auto x) { return repr(x); }, "\n")) +
                                   "\n}";
        return "if(" + repr(x.cond) + ") { \n" +
               indent(2, mk_string<Stmt::Any>(
                             x.trueBr, [&](auto x) { return repr(x); }, "\n")) +
               elseStmts;
      },
      [](const Stmt::Return &x) { return "return " + repr(x.value); });
}

[[nodiscard]] string polyast::repr(const Arg &arg) { return repr(arg.named); }

[[nodiscard]] string polyast::repr(const TypeSpace::Any &space) {
  return space.match_total( //
      [&](const TypeSpace::Global &x) { return "^Global"; }, [&](const TypeSpace::Local &x) { return "^Local"; });
}

[[nodiscard]] string polyast::repr(const Function &fn) {
  string str;
  if (fn.receiver) str += repr(*fn.receiver) + ".";
  str += repr(fn.name);
  str += "<" +
         mk_string<string>(
             fn.tpeVars, [](auto &x) { return x; }, ", ") +
         ">";
  str += "(" +
         mk_string<Arg>(
             fn.args, [&](auto x) { return repr(x); }, ",") +
         ")";
  str += "[" +
         mk_string<Arg>(
             fn.moduleCaptures, [&](auto x) { return repr(x); }, ",") +
         ";" +
         mk_string<Arg>(
             fn.termCaptures, [&](auto x) { return repr(x); }, ",") +
         "]";
  str += ": " + repr(fn.rtn);
  str += " = {\n" +
         indent(2, mk_string<Stmt::Any>(
                       fn.body, [&](auto x) { return repr(x); }, "\n")) +
         "\n}";
  return str;
}

[[nodiscard]] string polyast::repr(const StructDef &def) {

  return "struct " + repr(def.name) + (def.parents.empty() ? "" : ": ") +
         mk_string<Sym>(
             def.parents, [](auto &&x) { return repr(x); }, ", ") +
         " { " +
         mk_string<StructMember>(
             def.members, [](auto &&x) { return (x.isMutable ? "mut " : "") + repr(x.named); }, ",") +
         " }";
}

[[nodiscard]] string polyast::repr(const Program &program) {
  auto defs = mk_string<StructDef>(
      program.defs, [](auto &&x) { return repr(x); }, "\n");
  auto fns = mk_string<Function>(
      program.functions, [](auto &&x) { return repr(x); }, "\n");
  return defs + "\n" + fns + "\n" + repr(program.entry);
}

string polyast::qualified(const Term::Select &select) {
  return select.init.empty() //
             ? select.last.symbol
             : polyregion::mk_string<Named>(
                   select.init, [](auto &n) { return n.symbol; }, ".") +
                   "." + select.last.symbol;
}

string polyast::qualified(const Sym &sym) {
  return polyregion::mk_string<string>(
      sym.fqn, [](auto &n) { return n; }, ".");
}

std::vector<Named> polyast::path(const Term::Select &select) {
  std::vector<Named> xs(select.init);
  xs.push_back(select.last);
  return xs;
}

Named polyast::head(const Term::Select &select) { return select.init.empty() ? select.last : select.init.front(); }

std::vector<Named> polyast::tail(const Term::Select &select) {
  if (select.init.empty()) return {select.last};
  else {
    std::vector<Named> xs(std::next(select.init.begin()), select.init.end());
    xs.push_back(select.last);
    return xs;
  }
}

std::pair<Named, std::vector<Named>> polyast::uncons(const Term::Select &select) {
  if (select.init.empty()) return {{select.last}, {}};
  else {
    std::vector<Named> xs(std::next(select.init.begin()), select.init.end());
    xs.push_back(select.last);
    return {select.init.front(), xs};
  }
}

std::optional<polyast::Target> polyast::targetFromOrdinal(std::underlying_type_t<polyast::Target> ordinal) {
  auto target = static_cast<Target>(ordinal);
  switch (target) {
    case Target::Object_LLVM_HOST:
    case Target::Object_LLVM_x86_64:
    case Target::Object_LLVM_AArch64:
    case Target::Object_LLVM_ARM:
    case Target::Object_LLVM_NVPTX64:
    case Target::Object_LLVM_AMDGCN:
    case Target::Source_C_OpenCL1_1:
    case Target::Source_C_C11:
    case Target::Source_C_Metal1_0:
    case Target::Object_LLVM_SPIRV32:
    case Target::Object_LLVM_SPIRV64:
      return target;
      // XXX do not add default here, see  -Werror=switch
  }
}

std::optional<polyast::OptLevel> polyast::optFromOrdinal(std::underlying_type_t<polyast::OptLevel> ordinal) {
  auto target = static_cast<OptLevel>(ordinal);
  switch (target) {
    case OptLevel::O0:
    case OptLevel::O1:
    case OptLevel::O2:
    case OptLevel::O3:
    case OptLevel::Ofast:
      return target;
      // XXX do not add default here, see  -Werror=switch
  }
}

string polyast::repr(const polyast::CompileResult &compilation) {
  std::ostringstream os;
  os << "Compilation {"                                                                                            //
     << "\n  binary: " << (compilation.binary ? std::to_string(compilation.binary->size()) + " bytes" : "(empty)") //
     << "\n  messages: `" << compilation.messages << "`"                                                           //
     << "\n  features: `"
     << mk_string<string>(
            compilation.features, [](auto x) { return x; }, ",")
     << "`"
     << "\n  layouts: `"
     << mk_string<CompileLayout>(
            compilation.layouts, [](auto x) { return to_string(x); }, "\n    ")
     << "`"
     << "\n  events:\n";

  for (auto &e : compilation.events) {
    os << "    [" << e.epochMillis << ", +" << (double(e.elapsedNanos) / 1e6) << "ms] " << e.name;
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
Type::Struct dsl::Struct(Sym name, std::vector<string> tpeVars, std::vector<Type::Any> args) { return {name, tpeVars, args, {}}; }
Term::Any dsl::integral(const Type::Any &tpe, unsigned long long int x) {
  auto unsupported = [](auto &&t, auto &&v) -> Term::Any {
    throw std::logic_error("Cannot create integral constant of type " + to_string(t) + " for value" + std::to_string(v));
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

      [&](const Type::Unit0 &t) -> Term::Any { return unsupported(t, x); },   //
      [&](const Type::Bool1 &) -> Term::Any { return Term::Bool1Const(x); },  //
      [&](const Type::Nothing &t) -> Term::Any { return unsupported(t, x); }, //
      [&](const Type::Struct &t) -> Term::Any { return unsupported(t, x); },  //
      [&](const Type::Ptr &t) -> Term::Any { return unsupported(t, x); },     //
      [&](const Type::Var &t) -> Term::Any { return unsupported(t, x); },     //
      [&](const Type::Exec &t) -> Term::Any { return unsupported(t, x); }     //

  );
}
Term::Any dsl::fractional(const Type::Any &tpe, long double x) {
  if (tpe.is<Type::Float64>()) return Term::Float64Const(static_cast<double>(x));
  if (tpe.is<Type::Float32>()) return Term::Float32Const(static_cast<float>(x));
  if (tpe.is<Type::Float16>()) return Term::Float16Const(static_cast<float>(x));
  throw std::logic_error("Cannot create fractional constant of type " + to_string(tpe) + " for value" + std::to_string(x));
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

Stmt::Any dsl::let(const string &name, const Type::Any &tpe) { return Var(Named(name, tpe), {}); }
dsl::AssignmentBuilder dsl::let(const string &name) { return AssignmentBuilder{name}; }
Expr::IntrOp dsl::invoke(const Intr::Any &intr) { return Expr::IntrOp(intr); }
Expr::MathOp dsl::invoke(const Math::Any &intr) { return Expr::MathOp(intr); }
Expr::SpecOp dsl::invoke(const Spec::Any &intr) { return Expr::SpecOp(intr); }
std::function<Function(std::vector<Stmt::Any>)> dsl::function(const string &name, const std::vector<Arg> &args, const Type::Any &rtn) {
  return [=](auto &&stmts) { return Function(Sym({name}), {}, {}, args, {}, {}, rtn, stmts); };
}
Stmt::Return dsl::ret(const Expr::Any &expr) { return Return(expr); }
Stmt::Return dsl::ret(const Term::Any &term) { return Return(Alias(term)); }
Program dsl::program(Function entry, std::vector<StructDef> defs, std::vector<Function> functions) {
  return Program(entry, functions, defs);
}

dsl::IndexBuilder::IndexBuilder(const Expr::Index &index) : index(index) {}
dsl::IndexBuilder::operator Expr::Any() const { return index; }
Stmt::Update dsl::IndexBuilder::operator=(const Term::Any &term) const { return {index.lhs, index.idx, term}; }
dsl::NamedBuilder::NamedBuilder(const Named &named) : named(named) {}
dsl::NamedBuilder::operator Term::Any() const { return Select({}, named); }
// dsl::NamedBuilder::operator const Expr::Any() const { return Alias(Select({}, named)); }
dsl::NamedBuilder::operator Named() const { return named; }
Arg dsl::NamedBuilder::operator()() const { return Arg(named, {}); }

dsl::IndexBuilder dsl::NamedBuilder::operator[](const Term::Any &idx) const {
  if (auto arr = named.tpe.get<Type::Ptr>(); arr) {
    return IndexBuilder({Select({}, named), idx, arr->component});
  } else {
    throw std::logic_error("Cannot index a reference to non-array type" + to_string(named));
  }
}

dsl::AssignmentBuilder::AssignmentBuilder(const string &name) : name(name) {}
Stmt::Any dsl::AssignmentBuilder::operator=(Expr::Any rhs) const { return Var(Named(name, rhs.tpe()), {rhs}); }
Stmt::Any dsl::AssignmentBuilder::operator=(Term::Any rhs) const { return Var(Named(name, rhs.tpe()), {Alias(rhs)}); }
Stmt::Any dsl::AssignmentBuilder::operator=(Type::Any tpe) const { return Var(Named(name, tpe), {}); }
