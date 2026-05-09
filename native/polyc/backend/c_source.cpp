
#include "c_source.h"

#include <set>

#include "aspartame/all.hpp"
#include "fmt/core.h"

using namespace aspartame;
using namespace polyregion;
using namespace polyast;
using namespace std::string_literals;

struct CLAddressSpaceTracePass {

  struct StackScope {
    Map<std::string, Named> vars;
  };

  struct SpacedTerm {
    Term::Any actual;
    TypeSpace::Any space = TypeSpace::Private();
  };

  struct SpacedExpr {
    Expr::Any actual;
    TypeSpace::Any space = TypeSpace::Private();
  };

  // Map a Term: only Term::Select needs scope rewiring; all other Term variants are pure atoms.
  static SpacedTerm mapTerm(const Term::Any &term, StackScope &scope) {
    if (auto sel = term.template get<Term::Select>()) {
      // First step rebinds via the scope (in case the root was a captured arg whose type carries
      // an inferred address space).
      const auto rebound = scope.vars ^ get_or_default(sel->root.symbol, sel->root);
      // Walk the steps to recover a final type and any pointer-space hints from struct fields.
      // Since PathStep::Field carries no type, we just take the result type from sel->tpe and the
      // space from rebound's tpe if it is a Ptr (as the most common origin for spaced selects).
      const auto space = rebound.tpe.template get<Type::Ptr>() ^ //
                         map([](auto &p) { return p.space; }) ^  //
                         get_or_else(TypeSpace::Private().widen());
      return SpacedTerm{Term::Select(rebound, sel->steps, sel->tpe), space};
    }
    return SpacedTerm{term};
  }

  static SpacedExpr mapExpr(const Expr::Any &expr, StackScope &scope) {
    auto mapTerm_ = [&](const Term::Any &t) { return mapTerm(t, scope); };
    auto mapTerm0_ = [&](const Term::Any &t) { return mapTerm(t, scope).actual; };
    return expr.match_total(
        [&](const Expr::Alias &x) -> SpacedExpr {
          auto st = mapTerm_(x.ref);
          return {Expr::Alias(st.actual), st.space};
        },
        [&](const Expr::SpecOp &x) -> SpacedExpr { return {Expr::SpecOp(x.op.modify_all<Term::Any>(mapTerm0_))}; },
        [&](const Expr::IntrOp &x) -> SpacedExpr { return {Expr::IntrOp(x.op.modify_all<Term::Any>(mapTerm0_))}; },
        [&](const Expr::MathOp &x) -> SpacedExpr { return {Expr::MathOp(x.op.modify_all<Term::Any>(mapTerm0_))}; },
        [&](const Expr::Cast &x) -> SpacedExpr {
          auto st = mapTerm_(x.from);
          return {Expr::Cast(st.actual, x.as), st.space};
        },
        [&](const Expr::Invoke &x) -> SpacedExpr { return {x.modify_all<Term::Any>(mapTerm0_)}; },
        [&](const Expr::Index &x) -> SpacedExpr {
          auto stLhs = mapTerm_(x.lhs);
          auto stIdx = mapTerm_(x.idx);
          return {Expr::Index(stLhs.actual, stIdx.actual, x.comp), stLhs.space};
        },
        [&](const Expr::RefTo &x) -> SpacedExpr {
          auto stLhs = mapTerm_(x.lhs);
          return {Expr::RefTo(stLhs.actual, x.idx ^ map(mapTerm0_), x.comp, stLhs.space), stLhs.space};
        },
        [&](const Expr::Alloc &x) -> SpacedExpr { return {Expr::Alloc(x.comp, mapTerm0_(x.size), x.space)}; });
  }

  static Function mapFn(const Function &fn) {

    StackScope scope{.vars = fn.args                                                                  //
                             | flat_map([&](auto &arg) { return arg.template collect_all<Named>(); }) //
                             | filter([](auto &n) { return n.tpe.template is<Type::Ptr>(); })         //
                             | map([](auto &n) { return std::pair(n.symbol, n); })                    //
                             | to<Map>()};

    auto body = fn.body ^ map([&](auto &s) {
                  return s
                      .template modify_all<Stmt::Var>([&](auto &var) { //
                        if (auto expr = var.expr ^ map([&](auto &e) { return mapExpr(e, scope); })) {
                          auto name = var.name;
                          if (auto ptr = expr->actual.tpe().template get<Type::Ptr>()) {
                            name = Named(var.name.symbol, Type::Ptr(ptr->comp, expr->space));
                          }
                          scope.vars.emplace(name.symbol, name);
                          return Stmt::Var(name, expr->actual, var.isMutable);
                        }
                        scope.vars.emplace(var.name.symbol, var.name);
                        return Stmt::Var(var.name, {}, var.isMutable);
                      })
                      .template modify_all<Expr::Any>([&](auto &e) { return mapExpr(e, scope).actual; }); //
                });

    const auto tracedRtnTpes = body                                                                        //
                               | flat_map([&](auto &s) { return s.template collect_all<Stmt::Return>(); }) //
                               | map([&](auto &r) { return r.value.tpe(); })                               //
                               | distinct()                                                                //
                               | to_vector();                                                              //
    return Function(fn.name, fn.tpeVars, fn.receiver, fn.args, fn.moduleCaptures, fn.termCaptures, tracedRtnTpes[0], body, fn.visibility,
                    fn.fpMode, fn.isEntry);
  }

  static Program execute(const Program &p) {
    auto fns = p.functions ^ map([](const Function &f) {
                 const auto kernel = f.isEntry;
                 auto remapSpace = [&](auto &s) {
                   return s.match_total(
                       [&](const TypeSpace::Global &) { return kernel ? TypeSpace::Global().widen() : TypeSpace::Private().widen(); }, //
                       [&](const TypeSpace::Local &x) { return x.widen(); },                                                           //
                       [&](const TypeSpace::Private &x) { return x.widen(); });
                 };
                 return CLAddressSpaceTracePass::mapFn(
                     f.withArgs(f.args ^ map([&](auto &arg) { return arg.template modify_all<TypeSpace::Any>(remapSpace); })));
               });

    auto sigOf = [](const Expr::Invoke &inv) {
      return Signature(inv.name, /*tpeVars*/ {}, /*receiver*/ {}, inv.args ^ map([](auto &e) { return e.tpe(); }),
                       /*moduleCaptures*/ {}, /*termCaptures*/ {}, inv.rtn);
    };

    Map<Signature, std::shared_ptr<Function>> functionTable;
    fns | for_each([&](auto &f) {
      const Signature sig(f.name, /*tpeVars*/ {}, /*receiver*/ {}, f.args ^ map([](auto &e) { return e.named.tpe; }),
                          /*moduleCaptures*/ {}, /*termCaptures*/ {}, f.rtn);
      functionTable[sig] = std::make_shared<Function>(f);
    });

    while (true) {
      const auto specialised =
          functionTable                                                                                //
          | flat_map([&](auto, auto &f) { return f->template collect_all<Expr::Invoke>(); })           //
          | collect([&](auto &inv) -> std::optional<std::pair<Signature, std::shared_ptr<Function>>> { //
              if (const auto sig = sigOf(inv); !functionTable.contains(sig)) {
                if (auto spec =
                        functionTable ^ find([&](auto &lhs, auto) { return lhs.name == sig.name && lhs.args.size() == sig.args.size(); })) {
                  const auto fn = *spec->second;
                  const auto args = fn.args                                                                           //
                                    | zip(sig.args)                                                                   //
                                    | map([](auto &arg, auto &tpe) { return arg.withNamed(arg.named.withTpe(tpe)); }) //
                                    | to_vector();

                  return std::pair{sig, std::make_shared<Function>(CLAddressSpaceTracePass::mapFn(fn.withArgs(args)))};
                }
              }
              return {};
            }) //
          | to<Map>();
      if (specialised.empty()) break;
      functionTable.insert(specialised.begin(), specialised.end());
    }

    const auto spaceSpecialisedName = [](const Sym &name, const std::vector<TypeSpace::Any> &ts) -> Sym {
      auto suffix = ts ^ mk_string("", [&](auto &s) {
                      return s.match_total([&](TypeSpace::Global) { return "g"; }, //
                                           [&](TypeSpace::Local) { return "l"; },  //
                                           [&](TypeSpace::Private) { return "p"; });
                    });
      auto fqn = name.fqn;
      if (!fqn.empty()) fqn.back() = fqn.back() + "_" + suffix;
      else fqn.push_back("_" + suffix);
      return Sym(fqn);
    };

    const auto spaceSpecialisedFns =     //
        functionTable                    //
        | values()                       //
        | map([&](auto &f) -> Function { //
            auto spaces = [&](auto &a) { return a.template collect_all<TypeSpace::Any>(); };
            return f
                ->template modify_all<Expr::Invoke>(
                    [&](auto &inv) { return inv.withName(spaceSpecialisedName(inv.name, inv.args ^ flat_map(spaces))); })
                .withName(f->isEntry ? f->name : spaceSpecialisedName(f->name, f->args ^ flat_map(spaces)));
          }) //
        | to_vector();

    if (spaceSpecialisedFns.empty()) return p;
    return Program(spaceSpecialisedFns.front(), Vector<Function>(std::next(spaceSpecialisedFns.begin()), spaceSpecialisedFns.end()), p.defs,
                   p.phase);
  }
};

static std::string normalise(const std::string &s) {
  return s                                   //
         ^ replace_all(" ", "_")             //
         ^ replace_all("&", "_")             //
         ^ replace_all(",", "_")             //
         ^ replace_all("*", "_")             //
         ^ replace_all("+", "_")             //
         ^ replace_all("/", "_")             //
         ^ replace_all("<", "_")             //
         ^ replace_all(">", "_")             //
         ^ replace_all("#", "_")             //
         ^ replace_all(":", "_")             //
         ^ replace_all("(", "_")             //
         ^ replace_all(")", "_")             //
         ^ replace_all(".", "_")             //
         ^ replace_all("global", "_global")  //
         ^ replace_all("local", "_local")    //
         ^ replace_all("kernel", "_kernel"); //
}

static std::string normalise(const Sym &s) { return normalise(repr(s)); }

std::string backend::CSource::mkTpe(const Type::Any &tpe) {
  switch (dialect) {
    case Dialect::C11:
    case Dialect::MSL1_0:
      return tpe.match_total([&](const Type::Float16 &) { return "__fp16"s; }, //
                             [&](const Type::Float32 &) { return "float"s; },  //
                             [&](const Type::Float64 &) { return "double"s; }, //

                             [&](const Type::IntU8 &) { return "uint8_t"s; },   //
                             [&](const Type::IntU16 &) { return "uint16_t"s; }, //
                             [&](const Type::IntU32 &) { return "uint32_t"s; }, //
                             [&](const Type::IntU64 &) { return "uint64_t"s; }, //

                             [&](const Type::IntS8 &) { return "int8_t"s; },   //
                             [&](const Type::IntS16 &) { return "int16_t"s; }, //
                             [&](const Type::IntS32 &) { return "int32_t"s; }, //
                             [&](const Type::IntS64 &) { return "int64_t"s; }, //

                             [&](const Type::Nothing &) { return "/*nothing*/"s; }, //
                             [&](const Type::Unit0 &) { return "void"s; },          //
                             [&](const Type::Bool1 &) { return "bool"s; },          //

                             [&](const Type::Struct &x) { return normalise(x.name); },                           //
                             [&](const Type::Ptr &x) { return fmt::format("{}*", mkTpe(x.comp)); },              //
                             [&](const Type::Arr &x) { return fmt::format("{}[{}]", mkTpe(x.comp), x.length); }, //
                             [&](const Type::Var &x) -> std::string { throw std::logic_error("Type::Var should be erased"); },
                             [&](const Type::Exec &x) -> std::string { throw std::logic_error("Type::Exec should be erased"); });
    case Dialect::OpenCL1_1:
      return tpe.match_total([&](const Type::Float16 &) { return "half"s; },   //
                             [&](const Type::Float32 &) { return "float"s; },  //
                             [&](const Type::Float64 &) { return "double"s; }, //

                             [&](const Type::IntU8 &) { return "uchar"s; },   //
                             [&](const Type::IntU16 &) { return "ushort"s; }, //
                             [&](const Type::IntU32 &) { return "uint"s; },   //
                             [&](const Type::IntU64 &) { return "ulong"s; },  //

                             [&](const Type::IntS8 &) { return "char"s; },   //
                             [&](const Type::IntS16 &) { return "short"s; }, //
                             [&](const Type::IntS32 &) { return "int"s; },   //
                             [&](const Type::IntS64 &) { return "long"s; },  //

                             [&](const Type::Nothing &) { return "/*nothing*/"s; }, //
                             [&](const Type::Unit0 &) { return "void"s; },          //
                             [&](const Type::Bool1 &) { return "char"s; },          //

                             [&](const Type::Struct &x) { return normalise(x.name); }, //
                             [&](const Type::Ptr &x) {
                               auto prefix = x.space.match_total([&](TypeSpace::Global) { return "global"; },  //
                                                                 [&](TypeSpace::Local) { return "local"; },    //
                                                                 [&](TypeSpace::Private) { return "private"; } //
                               );
                               auto comp = mkTpe(x.comp);
                               return fmt::format("{} {}*", prefix, comp);
                             }, //
                             [&](const Type::Arr &x) {
                               auto prefix = x.space.match_total([&](TypeSpace::Global) { return "global"; },  //
                                                                 [&](TypeSpace::Local) { return "local"; },    //
                                                                 [&](TypeSpace::Private) { return "private"; } //
                               );
                               auto comp = mkTpe(x.comp);
                               return fmt::format("{} {}[{}]", prefix, comp, x.length);
                             }, //
                             [&](const Type::Var &x) -> std::string { throw std::logic_error("Type::Var should be erased"); },
                             [&](const Type::Exec &x) -> std::string { throw std::logic_error("Type::Exec should be erased"); });
  }
}

std::string backend::CSource::mkTerm(const Term::Any &term) {
  return term.match_total([](const Term::Float16Const &x) { return fmt::format("{}", x.value); },   //
                          [](const Term::Float32Const &x) { return fmt::format("{}.f", x.value); }, //
                          [](const Term::Float64Const &x) { return fmt::format("{}", x.value); },   //

                          [](const Term::IntU8Const &x) { return fmt::format("{}", x.value); },  //
                          [](const Term::IntU16Const &x) { return fmt::format("{}", x.value); }, //
                          [](const Term::IntU32Const &x) { return fmt::format("{}", x.value); }, //
                          [](const Term::IntU64Const &x) { return fmt::format("{}", x.value); }, //

                          [](const Term::IntS8Const &x) { return fmt::format("{}", x.value); },  //
                          [](const Term::IntS16Const &x) { return fmt::format("{}", x.value); }, //
                          [](const Term::IntS32Const &x) { return fmt::format("{}", x.value); }, //
                          [](const Term::IntS64Const &x) { return fmt::format("{}", x.value); }, //

                          [](const Term::Unit0Const &) { return "/*void*/"s; },                                   //
                          [](const Term::Bool1Const &x) { return x.value ? "true"s : "false"s; },                 //
                          [&](const Term::NullPtrConst &x) { return fmt::format("NULL /*{}*/", mkTpe(x.comp)); }, //
                          [&](const Term::Poison &x) { return fmt::format("(NULL /*{}*/)", repr(x.tpe)); },       //
                          [&](const Term::Select &x) {
                            std::string acc = normalise(x.root.symbol);
                            for (auto &step : x.steps) {
                              step.match_total(
                                  [&](const PathStep::Field &f) {
                                    acc += ".";
                                    acc += normalise(f.name);
                                  },
                                  [&](const PathStep::Deref &) { acc = "(*" + acc + ")"; });
                            }
                            return acc;
                          });
}

std::string backend::CSource::mkExpr(const Expr::Any &expr) {
  return expr.match_total(
      [&](const Expr::Alias &x) { return mkTerm(x.ref); },
      [&](const Expr::SpecOp &x) {
        struct DialectAccessor {
          std::string cl, msl;
        };
        const auto gpuIntr = [&](const DialectAccessor &accessor) -> std::string {
          switch (dialect) {
            case Dialect::C11: throw std::logic_error(to_string(x) + " not supported for C11");
            case Dialect::MSL1_0: return accessor.msl;
            case Dialect::OpenCL1_1: return accessor.cl;
          }
        };
        const auto gpuDimIntr = [&](const DialectAccessor &accessor, const Term::Any &index) -> std::string {
          switch (dialect) {
            case Dialect::C11: throw std::logic_error(to_string(x) + " not supported for C11");
            case Dialect::MSL1_0: return fmt::format("{}[{}]", accessor.msl, mkTerm(index));
            case Dialect::OpenCL1_1: return fmt::format("{}({})", accessor.cl, mkTerm(index));
          }
        };
        return x.op.match_total(
            [&](const Spec::Assert &v) { return "abort()"s; }, //
            [&](const Spec::GpuBarrierGlobal &v) {
              return gpuIntr({.cl = "barrier(CLK_GLOBAL_MEM_FENCE)", //
                              .msl = "threadgroup_barrier(metal::mem_flags::mem_none)"});
            },
            [&](const Spec::GpuBarrierLocal &v) {
              return gpuIntr({.cl = "barrier(CLK_LOCAL_MEM_FENCE)", //
                              .msl = "threadgroup_barrier(metal::mem_flags::mem_none)"});
            },
            [&](const Spec::GpuBarrierAll &v) {
              return gpuIntr({.cl = "barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE)", //
                              .msl = "threadgroup_barrier(metal::mem_flags::mem_none)"});
            },
            [&](const Spec::GpuFenceGlobal &v) {
              return gpuIntr({.cl = "mem_fence(CLK_GLOBAL_MEM_FENCE)", //
                              .msl = "threadgroup_barrier(metal::mem_flags::mem_device)"});
            },
            [&](const Spec::GpuFenceLocal &v) {
              return gpuIntr({.cl = "mem_fence(CLK_LOCAL_MEM_FENCE)", //
                              .msl = "threadgroup_barrier(metal::mem_flags::mem_threadgroup)"});
            },
            [&](const Spec::GpuFenceAll &v) {
              return gpuIntr({.cl = "mem_fence(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE)", //
                              .msl = "threadgroup_barrier(metal::mem_flags::mem_device)"});
            },
            [&](const Spec::GpuGlobalIdx &v) { return gpuDimIntr({.cl = "get_global_id", .msl = "__get_global_id__"}, v.dim); },      //
            [&](const Spec::GpuGlobalSize &v) { return gpuDimIntr({.cl = "get_global_size", .msl = "__get_global_size__"}, v.dim); }, //
            [&](const Spec::GpuGroupIdx &v) { return gpuDimIntr({.cl = "get_group_id", .msl = "__get_group_id__"}, v.dim); },         //
            [&](const Spec::GpuGroupSize &v) { return gpuDimIntr({.cl = "get_num_groups", .msl = "__get_num_groups__"}, v.dim); },    //
            [&](const Spec::GpuLocalIdx &v) { return gpuDimIntr({.cl = "get_local_id", .msl = "__get_local_id__"}, v.dim); },         //
            [&](const Spec::GpuLocalSize &v) { return gpuDimIntr({.cl = "get_local_size", .msl = "__get_local_size__"}, v.dim); }     //
        );
      },
      [&](const Expr::IntrOp &x) {
        return x.op.match_total([&](const Intr::Pos &v) { return fmt::format("(+{})", mkTerm(v.x)); },
                                [&](const Intr::Neg &v) { return fmt::format("(-{})", mkTerm(v.x)); },
                                [&](const Intr::BNot &v) { return fmt::format("(~{})", mkTerm(v.x)); },
                                [&](const Intr::LogicNot &v) { return fmt::format("(!{})", mkTerm(v.x)); },
                                [&](const Intr::Add &v) { return fmt::format("({} + {})", mkTerm(v.x), mkTerm(v.y)); },
                                [&](const Intr::Sub &v) { return fmt::format("({} - {})", mkTerm(v.x), mkTerm(v.y)); },
                                [&](const Intr::Mul &v) { return fmt::format("({} * {})", mkTerm(v.x), mkTerm(v.y)); },
                                [&](const Intr::Div &v) { return fmt::format("({} / {})", mkTerm(v.x), mkTerm(v.y)); },
                                [&](const Intr::Rem &v) { return fmt::format("({} % {})", mkTerm(v.x), mkTerm(v.y)); },
                                [&](const Intr::Min &v) { return fmt::format("min({}, {})", mkTerm(v.x), mkTerm(v.y)); },
                                [&](const Intr::Max &v) { return fmt::format("max({}, {})", mkTerm(v.x), mkTerm(v.y)); },
                                [&](const Intr::BAnd &v) { return fmt::format("({} & {})", mkTerm(v.x), mkTerm(v.y)); },
                                [&](const Intr::BOr &v) { return fmt::format("({} | {})", mkTerm(v.x), mkTerm(v.y)); },
                                [&](const Intr::BXor &v) { return fmt::format("({} ^ {})", mkTerm(v.x), mkTerm(v.y)); },
                                [&](const Intr::BSL &v) { return fmt::format("({} << {})", mkTerm(v.x), mkTerm(v.y)); },
                                [&](const Intr::BSR &v) { return fmt::format("({} >> {})", mkTerm(v.x), mkTerm(v.y)); },
                                [&](const Intr::BZSR &v) { return fmt::format("({} <<< {})", mkTerm(v.x), mkTerm(v.y)); },
                                [&](const Intr::LogicAnd &v) { return fmt::format("({} && {})", mkTerm(v.x), mkTerm(v.y)); },
                                [&](const Intr::LogicOr &v) { return fmt::format("({} || {})", mkTerm(v.x), mkTerm(v.y)); },
                                [&](const Intr::LogicEq &v) { return fmt::format("({} == {})", mkTerm(v.x), mkTerm(v.y)); },
                                [&](const Intr::LogicNeq &v) { return fmt::format("({} != {})", mkTerm(v.x), mkTerm(v.y)); },
                                [&](const Intr::LogicLte &v) { return fmt::format("({} <= {})", mkTerm(v.x), mkTerm(v.y)); },
                                [&](const Intr::LogicGte &v) { return fmt::format("({} >= {})", mkTerm(v.x), mkTerm(v.y)); },
                                [&](const Intr::LogicLt &v) { return fmt::format("({} < {})", mkTerm(v.x), mkTerm(v.y)); },
                                [&](const Intr::LogicGt &v) { return fmt::format("({} > {})", mkTerm(v.x), mkTerm(v.y)); });
      },
      [&](const Expr::MathOp &x) {
        return x.op.match_total([&](const Math::Abs &v) { return fmt::format("abs({})", mkTerm(v.x)); },
                                [&](const Math::Sin &v) { return fmt::format("sin({})", mkTerm(v.x)); },
                                [&](const Math::Cos &v) { return fmt::format("cos({})", mkTerm(v.x)); },
                                [&](const Math::Tan &v) { return fmt::format("tan({})", mkTerm(v.x)); },
                                [&](const Math::Asin &v) { return fmt::format("asin({})", mkTerm(v.x)); },
                                [&](const Math::Acos &v) { return fmt::format("acos({})", mkTerm(v.x)); },
                                [&](const Math::Atan &v) { return fmt::format("atan({})", mkTerm(v.x)); },
                                [&](const Math::Sinh &v) { return fmt::format("sinh({})", mkTerm(v.x)); },
                                [&](const Math::Cosh &v) { return fmt::format("cosh({})", mkTerm(v.x)); },
                                [&](const Math::Tanh &v) { return fmt::format("tanh({})", mkTerm(v.x)); },
                                [&](const Math::Signum &v) { return fmt::format("signum({})", mkTerm(v.x)); },
                                [&](const Math::Round &v) { return fmt::format("round({})", mkTerm(v.x)); },
                                [&](const Math::Ceil &v) { return fmt::format("ceil({})", mkTerm(v.x)); },
                                [&](const Math::Floor &v) { return fmt::format("floor({})", mkTerm(v.x)); },
                                [&](const Math::Rint &v) { return fmt::format("rint({})", mkTerm(v.x)); },
                                [&](const Math::Sqrt &v) { return fmt::format("sqrt({})", mkTerm(v.x)); },
                                [&](const Math::Cbrt &v) { return fmt::format("cbrt({})", mkTerm(v.x)); },
                                [&](const Math::Exp &v) { return fmt::format("exp({})", mkTerm(v.x)); },
                                [&](const Math::Expm1 &v) { return fmt::format("expm1({})", mkTerm(v.x)); },
                                [&](const Math::Log &v) { return fmt::format("log({})", mkTerm(v.x)); },
                                [&](const Math::Log1p &v) { return fmt::format("log1p({})", mkTerm(v.x)); },
                                [&](const Math::Log10 &v) { return fmt::format("log10({})", mkTerm(v.x)); },
                                [&](const Math::Pow &v) { return fmt::format("pow({}, {})", mkTerm(v.x), mkTerm(v.y)); },
                                [&](const Math::Atan2 &v) { return fmt::format("atan2({}, {})", mkTerm(v.x), mkTerm(v.y)); },
                                [&](const Math::Hypot &v) { return fmt::format("hypot({}, {})", mkTerm(v.x), mkTerm(v.y)); });
      },
      [&](const Expr::Cast &x) { return fmt::format("(({}) {})", mkTpe(x.as), mkTerm(x.from)); },
      [&](const Expr::Invoke &x) {
        return fmt::format("{}({})", normalise(x.name), x.args ^ mk_string(", ", [&](auto &arg) { return mkTerm(arg); }));
      }, //
      [&](const Expr::Index &x) { return fmt::format("{}[{}]", mkTerm(x.lhs), mkTerm(x.idx)); },
      [&](const Expr::RefTo &x) {
        std::string str = fmt::format("&({} /*{}*/)", mkTerm(x.lhs), mkTpe(x.comp));
        if (x.idx) str += fmt::format("[{}]", mkTerm(*x.idx));
        return str;
      },
      [&](const Expr::Alloc &x) { return fmt::format("{{/*{}*/}}", to_string(x)); });
}

std::string backend::CSource::mkStmt(const Stmt::Any &stmt) {
  return stmt.match_total( //
      [&](const Stmt::Var &x) {
        if (x.name.tpe.is<Type::Unit0>()) return x.expr ? fmt::format("{};", mkExpr(*x.expr)) : ""s;
        return fmt::format("{} {}{};", mkTpe(x.name.tpe), normalise(x.name.symbol), x.expr ? " = " + mkExpr(*x.expr) : "");
      },
      [&](const Stmt::Mut &x) { return fmt::format("{} = {};", mkTerm(x.name), mkExpr(x.expr)); },
      [&](const Stmt::Update &x) { return fmt::format("{}[{}] = {};", mkTerm(x.lhs), mkTerm(x.idx), mkTerm(x.value)); },
      [&](const Stmt::While &x) {
        const auto body = x.body | mk_string("\n", [&](auto &s) { return mkStmt(s); });
        return fmt::format("while({}) {{\n{}\n}}", mkTerm(x.cond), body ^ indent(2));
      },
      [&](const Stmt::ForRange &x) {
        const auto body = x.body | mk_string("\n", [&](auto &s) { return mkStmt(s); });
        const auto induction = normalise(x.induction.symbol);
        return fmt::format("for({} {} = {}; {} < {}; {} += {}) {{\n{}\n}}",     //
                           mkTpe(x.induction.tpe), induction, mkTerm(x.lbIncl), //
                           induction, mkTerm(x.ubExcl), induction, mkTerm(x.step), body ^ indent(2));
      },
      [&](const Stmt::Break &) { return "break;"s; },   //
      [&](const Stmt::Cont &) { return "continue;"s; }, //
      [&](const Stmt::Cond &x) {
        auto trueBr = x.trueBr ^ mk_string("{\n", "\n", "\n}", [&](auto &s) { return mkStmt(s) ^ indent(2); });
        if (x.falseBr.empty()) {
          return fmt::format("if ({}) {}", mkTerm(x.cond), trueBr);
        } else {
          auto falseBr = x.falseBr ^ mk_string("{\n", "\n", "\n}", [&](auto &s) { return mkStmt(s) ^ indent(2); });
          return fmt::format("if ({}) {} else {}", mkTerm(x.cond), trueBr, falseBr);
        }
      },
      [&](const Stmt::Return &x) { return "return " + mkExpr(x.value) + ";"; }, //
      // Annotations carry no codegen meaning; unwrap and recurse.
      [&](const Stmt::Annotated &x) { return mkStmt(x.inner); });
}

std::string backend::CSource::mkFnProto(const Function &fnTree) {

  const auto entry = fnTree.isEntry;

  std::vector<std::string> argExprs =
      fnTree.args | zip_with_index() | map([&](auto &arg, auto idx) {
        auto tpe = mkTpe(arg.named.tpe);
        auto name = normalise(arg.named.symbol);
        std::string decl;
        switch (dialect) {
          case Dialect::OpenCL1_1: {
            decl = fmt::format("{} {}", tpe, name);
            break;
          }
          case Dialect::MSL1_0: {
            // Scalar:      device $T &$name      [[ buffer($i) ]]
            // Global:      device $T  $name      [[ buffer($i) ]]
            // Local:  threadgroup $T &$name [[ threadgroup($i) ]]
            // query:              $T &$name           [[ $type ]]
            if (auto arr = arg.named.tpe.template get<Type::Ptr>()) {
              decl = arr->space.match_total(
                  [&](TypeSpace::Global) { return fmt::format("device {} {} [[buffer({})]]", tpe, name, idx); },          //
                  [&](TypeSpace::Local) { return fmt::format("threadgroup {} {} [[threadgroup({})]]", tpe, name, idx); }, //
                  [&](TypeSpace::Private) { return fmt::format("device {} &{} [[buffer({})]]", tpe, name, idx); }         //
              );
            } else decl = fmt::format("device {} &{} [[buffer({})]]", tpe, name, idx);

            break;
          }
          default: break;
        }
        return decl;
      }) |
      to_vector();

  if (dialect == Dialect::MSL1_0) {

    std::set<std::pair<std::string, std::string>> iargs; // ordered set for consistency
    for (auto &stmt : fnTree.body) {
      auto findIntrinsics = [&](Expr::Any &expr) {
        if (auto spec = expr.get<Expr::SpecOp>(); spec) {
          if (spec->op.is<Spec::GpuGlobalIdx>()) iargs.emplace("get_global_id", "thread_position_in_grid");
          if (spec->op.is<Spec::GpuGlobalSize>()) iargs.emplace("get_global_size", "threads_per_grid");
          if (spec->op.is<Spec::GpuGroupIdx>()) iargs.emplace("get_group_id", "threadgroup_position_in_grid");
          if (spec->op.is<Spec::GpuGroupSize>()) iargs.emplace("get_num_groups", "threadgroups_per_grid");
          if (spec->op.is<Spec::GpuLocalIdx>()) iargs.emplace("get_local_id", "thread_position_in_threadgroup");
          if (spec->op.is<Spec::GpuLocalSize>()) iargs.emplace("get_local_size", "threads_per_threadgroup");
        }
      };
      if (auto var = stmt.get<Stmt::Var>(); var) {
        if (var->expr) findIntrinsics(*var->expr);
      } else if (auto mut = stmt.get<Stmt::Mut>(); mut) {
        findIntrinsics(mut->expr);
      }
      // Cond.cond is now a Term (atomic) - SpecOp lives in Expr only, so nothing to find here.
    }
    for (auto &[name, attr] : iargs)
      argExprs.emplace_back(fmt::format("uint3 __{}__ [[ {} ]]", name, attr));
  }

  std::string fnPrefix;
  switch (dialect) {
    case Dialect::C11: fnPrefix = ""; break;
    case Dialect::MSL1_0:
    case Dialect::OpenCL1_1:
      if (entry) {
        fnPrefix = "kernel ";
      }
      break;
    default: fnPrefix = "";
  }

  return fmt::format("{}{} {}({})",
                     fnPrefix,               //
                     mkTpe(fnTree.rtn),      //
                     normalise(fnTree.name), //
                     argExprs | mk_string(", "));
}

std::string backend::CSource::mkFn(const Function &fnTree) {
  return fmt::format("{} {}", mkFnProto(fnTree),
                     fnTree.body ^ mk_string("{\n", "\n", "\n}", [&](auto &s) { return mkStmt(s) ^ indent(2); }));
}

CompileResult backend::CSource::compileProgram(const Program &program_, const compiletime::OptLevel &opt) {
  const auto tracePassStart = compiler::nowMono();
  auto program = CLAddressSpaceTracePass::execute(program_);
  CompileEvent cltpEvent(compiler::nowMs(), compiler::elapsedNs(tracePassStart), "polyast_cltp", repr(program));

  const auto start = compiler::nowMono();

  // work out the dependencies between structs first
  auto structsAndDeps = program.defs ^ map([&](auto &def) {
                          const auto deps = def.members //
                                            | collect([&](auto &m) {
                                                return m.tpe.template get<Type::Ptr>()                                         //
                                                       ^ flat_map([](auto &p) { return p.comp.template get<Type::Struct>(); }) //
                                                       ^ or_else_maybe(m.tpe.template get<Type::Struct>());                    //
                                              })                                                                               //
                                            | map([&](auto &s) { return s.name; });                                            //
                          return std::pair{def, deps};
                        }) ^
                        to<Map>();

  std::vector<std::string> fragments;
  switch (dialect) {
    case Dialect::C11: fragments.emplace_back("#include <stdint.h>\n#include <stdbool.h>"); break;
    default: break;
  }

  Set<Sym> resolved;
  while (resolved.size() != program.defs.size()) {
    const auto noDeps = structsAndDeps                      //
                        | filter([&](auto &s, auto &deps) { //
                            return !resolved.contains(s.name) && deps | forall([&](auto &d) { return resolved.contains(d); });
                          }) //
                        | keys();
    if (noDeps.empty()) {
      fragments.push_back(fmt::format("// Some structs cannot be resolved due to recursive dependencies"));
      break;
    }
    for (auto s : noDeps) {
      fragments.push_back(fmt::format(
          "typedef struct {} {};\n",
          s.members | mk_string("{\n", "\n", "\n}", [&](auto &m) { return fmt::format("  {} {};", mkTpe(m.tpe), normalise(m.symbol)); }),
          normalise(s.name)));
      resolved.emplace(s.name);
    }
  }

  // Forward declare all fns
  fragments.emplace_back(program.functions | mk_string("\n", [&](auto &fn) { return fmt::format("{};", mkFnProto(fn)); }));
  fragments.emplace_back("\n");

  for (auto &f : program.functions)
    fragments.emplace_back(mkFn(f));

  auto code = fragments | mk_string("\n");

  std::string dialectName;
  switch (dialect) {
    case Dialect::C11: dialectName = "c11"; break;
    case Dialect::OpenCL1_1: dialectName = "opencl1_1"; break;
    case Dialect::MSL1_0: dialectName = "msl1"; break;
    default: dialectName = "unknown";
  }

  return {std::vector<int8_t>(code.begin(), code.end()),
          {},
          {cltpEvent, {compiler::nowMs(), compiler::elapsedNs(start), fmt::format("polyast_to_{}_c", dialectName), code}},
          {},
          ""};
}
std::vector<StructLayout> backend::CSource::resolveLayouts(const std::vector<StructDef> &defs) { return std::vector<StructLayout>(); }
