
#include "c_source.h"
#include "aspartame/all.hpp"
#include "fmt/core.h"

#include <set>

using namespace aspartame;
using namespace polyregion;
using namespace polyast;
using namespace std::string_literals;

struct CLAddressSpaceTracePass {

  struct StackScope {
    Map<std::string, Named> vars;
  };

  struct SpacedExpr {
    Expr::Any actual;
    TypeSpace::Any space = TypeSpace::Private();
  };

  static SpacedExpr mapExpr(const Expr::Any &expr, StackScope &scope) {
    auto mapExpr_ = [&](auto &x) { return mapExpr(x, scope); };
    auto mapExpr0_ = [&](auto &x) { return mapExpr(x, scope).actual; };
    return expr.match_total(                                           //
        [](const Expr::Float16Const &x) -> SpacedExpr { return {x}; }, //
        [](const Expr::Float32Const &x) -> SpacedExpr { return {x}; }, //
        [](const Expr::Float64Const &x) -> SpacedExpr { return {x}; }, //

        [](const Expr::IntU8Const &x) -> SpacedExpr { return {x}; },  //
        [](const Expr::IntU16Const &x) -> SpacedExpr { return {x}; }, //
        [](const Expr::IntU32Const &x) -> SpacedExpr { return {x}; }, //
        [](const Expr::IntU64Const &x) -> SpacedExpr { return {x}; }, //

        [](const Expr::IntS8Const &x) -> SpacedExpr { return {x}; },  //
        [](const Expr::IntS16Const &x) -> SpacedExpr { return {x}; }, //
        [](const Expr::IntS32Const &x) -> SpacedExpr { return {x}; }, //
        [](const Expr::IntS64Const &x) -> SpacedExpr { return {x}; }, //

        [](const Expr::Unit0Const &x) -> SpacedExpr { return {x}; }, //
        [](const Expr::Bool1Const &x) -> SpacedExpr { return {x}; }, //

        [&](const Expr::SpecOp &x) -> SpacedExpr { return {x.modify_all<Expr::Any>(mapExpr0_)}; },
        [&](const Expr::IntrOp &x) -> SpacedExpr { return {x.modify_all<Expr::Any>(mapExpr0_)}; },
        [&](const Expr::MathOp &x) -> SpacedExpr { return {x.modify_all<Expr::Any>(mapExpr0_)}; },
        [&](const Expr::Select &x) -> SpacedExpr {
          if (x.init.empty()) {
            const auto last = scope.vars ^ get_or_default(x.last.symbol, x.last);
            return SpacedExpr{Expr::Select({}, last),
                              last.tpe.get<Type::Ptr>() ^ map([](auto &p) { return p.space; }) ^ get_or_else(TypeSpace::Private().widen())};
          }
          const auto init = x.init             //
                            | zip_with_index() //
                            | map([&](auto &n, auto idx) { return idx == 0 ? scope.vars ^ get_or_default(n.symbol, n) : n; }) | to_vector();
          const auto space = (init                                                               //
                              | append(x.last)                                                   //
                              | collect([](auto &n) { return n.tpe.template get<Type::Ptr>(); }) //
                              | map([](auto &p) { return p.space; })                             //
                              | last_maybe()) ^
                             get_or_else(TypeSpace::Private().widen());
          return SpacedExpr{Expr::Select(init, x.last), space};
        },                                                                                         //
        [&](const Expr::Poison &x) -> SpacedExpr { return {x.modify_all<Expr::Any>(mapExpr0_)}; }, //
        [&](const Expr::Cast &x) -> SpacedExpr {
          auto [from, s] = mapExpr_(x.from);
          return {Expr::Cast(from, x.as), s};
        },
        [&](const Expr::Invoke &x) -> SpacedExpr { return {x.modify_all<Expr::Any>(mapExpr0_)}; },
        [&](const Expr::Index &x) -> SpacedExpr { return {x.modify_all<Expr::Any>(mapExpr0_)}; },
        [&](const Expr::RefTo &x) -> SpacedExpr {
          auto [lhs, s] = mapExpr_(x.lhs);
          return {Expr::RefTo(lhs, x.idx ^ map(mapExpr0_), x.comp, s), s};
        },
        [&](const Expr::Alloc &x) -> SpacedExpr { return {x.modify_all<Expr::Any>(mapExpr0_)}; },
        [&](const Expr::Annotated &x) -> SpacedExpr {
          auto [e, s] = mapExpr_(x.expr);
          return {Expr::Annotated(e, x.pos, x.comment), s};
        });
  }

  static Function mapFn(const Function &fn) {

    StackScope scope{.vars = fn.args                                                              //
                             | flat_map([&](auto &arg) { return arg.template collect_all<Named>(); }) //
                             | filter([](auto &n) { return n.tpe.template is<Type::Ptr>(); })     //
                             | map([](auto &n) { return std::pair(n.symbol, n); })                //
                             | to<Map>()};

    auto body = fn.body ^ map([&](auto &s) {
                  return s
                      .template modify_all<Stmt::Var>([&](auto &var) { //
                        if (auto expr = var.expr ^ map([&](auto &e) { return mapExpr(e, scope); })) {
                          auto name = var.name;
                          if (auto ptr = expr->actual.tpe().template get<Type::Ptr>()) {
                            name = Named(var.name.symbol, Type::Ptr(ptr->comp, ptr->length, expr->space));
                          }
                          scope.vars.emplace(name.symbol, name);
                          return Stmt::Var(name, expr->actual);
                        }
                        scope.vars.emplace(var.name.symbol, var.name);
                        return Stmt::Var(var.name, {});
                      })
                      .template modify_all<Expr::Any>([&](auto &e) { return mapExpr(e, scope).actual; }); //
                });

    const auto tracedRtnTpes = body                                                                    //
                               | flat_map([&](auto &s) { return s.template collect_all<Stmt::Return>(); }) //
                               | map([&](auto &r) { return r.value.tpe(); })                           //
                               | distinct()                                                            //
                               | to_vector();                                                          //
    if (tracedRtnTpes.size() != 1) {
      body.emplace_back(Stmt::Comment(
          fmt::format("CLASTP: Return type diverged for function {}, types={}", fn.name, tracedRtnTpes | mk_string(", ", show_repr))));
    }
    return Function(fn.name, fn.args, tracedRtnTpes[0], body, fn.attrs);
  }

  static Program execute(const Program &p) {
    auto fns = p.functions ^ map([](const Function &f) {
                 const auto kernel = f.attrs.contains(FunctionAttr::Entry());
                 auto remapSpace = [&](auto &s) {
                   return s.match_total(
                       [&](const TypeSpace::Global &) { return kernel ? TypeSpace::Global().widen() : TypeSpace::Private().widen(); }, //
                       [&](const TypeSpace::Local &x) { return x.widen(); },                                                           //
                       [&](const TypeSpace::Private &x) { return x.widen(); });
                 };
                 return CLAddressSpaceTracePass::mapFn(
                     f.withArgs(f.args ^ map([&](auto &arg) { return arg.template modify_all<TypeSpace::Any>(remapSpace); })));
               });

    auto sigOf = [](const Expr::Invoke &inv) { return Signature(inv.name, inv.args ^ map([](auto &e) { return e.tpe(); }), inv.rtn); };

    Map<Signature, std::shared_ptr<Function>> functionTable;
    fns | for_each([&](auto &f) {
      const Signature sig(f.name, f.args ^ map([](auto &e) { return e.named.tpe; }), f.rtn);
      functionTable[sig] = std::make_shared<Function>(f);
    });

    while (true) {
      const auto specialised =
          functionTable                                                                                //
          | flat_map([&](auto, auto &f) { return f->template collect_all<Expr::Invoke>(); })               //
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

    const auto spaceSpecialisedName = [](const auto &name, const std::vector<TypeSpace::Any> &ts) {
      return fmt::format("{}_{}", //
                         name,    //
                         ts ^ mk_string("", [&](auto &s) {
                           return s.match_total([&](TypeSpace::Global) { return "g"; }, //
                                                [&](TypeSpace::Local) { return "l"; },  //
                                                [&](TypeSpace::Private) { return "p"; });
                         }));
    };

    const auto spaceSpecialisedFns =     //
        functionTable                    //
        | values()                       //
        | map([&](auto &f) -> Function { //
            auto spaces = [&](auto &a) { return a.template collect_all<TypeSpace::Any>(); };
            return f
                ->template modify_all<Expr::Invoke>(
                    [&](auto &inv) { return inv.withName(spaceSpecialisedName(inv.name, inv.args ^ flat_map(spaces))); })
                .withName(f->attrs.contains(FunctionAttr::Entry()) ? f->name : spaceSpecialisedName(f->name, f->args ^ flat_map(spaces)));
          }) //
        | to_vector();

    return Program(p.structs, spaceSpecialisedFns);
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

                             [&](const Type::Struct &x) { return normalise(x.name); },              //
                             [&](const Type::Ptr &x) { return fmt::format("{}*", mkTpe(x.comp)); }, //
                             [&](const Type::Annotated &x) {
                               return fmt::format("{} /*{};{}*/", mkTpe(x.tpe), x.pos ^ map(show_repr) ^ get_or_else(""),
                                                  x.comment ^ get_or_else(""));
                             } //
      );
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
                             [&](const Type::Annotated &x) {
                               return fmt::format("{} /*{};{}*/", mkTpe(x.tpe), x.pos ^ map(show_repr) ^ get_or_else(""),
                                                  x.comment ^ get_or_else(""));
                             } //
      );
  }
}

std::string backend::CSource::mkExpr(const Expr::Any &expr) {
  return expr.match_total(
      [](const Expr::Float16Const &x) { return fmt::format("{}", x.value); },   //
      [](const Expr::Float32Const &x) { return fmt::format("{}.f", x.value); }, //
      [](const Expr::Float64Const &x) { return fmt::format("{}", x.value); },   //

      [](const Expr::IntU8Const &x) { return fmt::format("{}", x.value); },  //
      [](const Expr::IntU16Const &x) { return fmt::format("{}", x.value); }, //
      [](const Expr::IntU32Const &x) { return fmt::format("{}", x.value); }, //
      [](const Expr::IntU64Const &x) { return fmt::format("{}", x.value); }, //

      [](const Expr::IntS8Const &x) { return fmt::format("{}", x.value); },  //
      [](const Expr::IntS16Const &x) { return fmt::format("{}", x.value); }, //
      [](const Expr::IntS32Const &x) { return fmt::format("{}", x.value); }, //
      [](const Expr::IntS64Const &x) { return fmt::format("{}", x.value); }, //

      [](const Expr::Unit0Const &x) { return "/*void*/"s; },                  //
      [](const Expr::Bool1Const &x) { return x.value ? "true"s : "false"s; }, //

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
        const auto gpuDimIntr = [&](const DialectAccessor &accessor, const Expr::Any &index) -> std::string {
          switch (dialect) {
            case Dialect::C11: throw std::logic_error(to_string(x) + " not supported for C11");
            case Dialect::MSL1_0: return fmt::format("{}[{}]", accessor.msl, mkExpr(index));
            case Dialect::OpenCL1_1: return fmt::format("{}({})", accessor.cl, mkExpr(index));
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
        return x.op.match_total([&](const Intr::Pos &v) { return fmt::format("(+{})", mkExpr(v.x)); },
                                [&](const Intr::Neg &v) { return fmt::format("(-{})", mkExpr(v.x)); },
                                [&](const Intr::BNot &v) { return fmt::format("(~{})", mkExpr(v.x)); },
                                [&](const Intr::LogicNot &v) { return fmt::format("(!{})", mkExpr(v.x)); },
                                [&](const Intr::Add &v) { return fmt::format("({} + {})", mkExpr(v.x), mkExpr(v.y)); },
                                [&](const Intr::Sub &v) { return fmt::format("({} - {})", mkExpr(v.x), mkExpr(v.y)); },
                                [&](const Intr::Mul &v) { return fmt::format("({} * {})", mkExpr(v.x), mkExpr(v.y)); },
                                [&](const Intr::Div &v) { return fmt::format("({} / {})", mkExpr(v.x), mkExpr(v.y)); },
                                [&](const Intr::Rem &v) { return fmt::format("({} % {})", mkExpr(v.x), mkExpr(v.y)); },
                                [&](const Intr::Min &v) { return fmt::format("min({}, {})", mkExpr(v.x), mkExpr(v.y)); },
                                [&](const Intr::Max &v) { return fmt::format("max({}, {})", mkExpr(v.x), mkExpr(v.y)); },
                                [&](const Intr::BAnd &v) { return fmt::format("({} & {})", mkExpr(v.x), mkExpr(v.y)); },
                                [&](const Intr::BOr &v) { return fmt::format("({} | {})", mkExpr(v.x), mkExpr(v.y)); },
                                [&](const Intr::BXor &v) { return fmt::format("({} ^ {})", mkExpr(v.x), mkExpr(v.y)); },
                                [&](const Intr::BSL &v) { return fmt::format("({} << {})", mkExpr(v.x), mkExpr(v.y)); },
                                [&](const Intr::BSR &v) { return fmt::format("({} >> {})", mkExpr(v.x), mkExpr(v.y)); },
                                [&](const Intr::BZSR &v) { return fmt::format("({} <<< {})", mkExpr(v.x), mkExpr(v.y)); },
                                [&](const Intr::LogicAnd &v) { return fmt::format("({} && {})", mkExpr(v.x), mkExpr(v.y)); },
                                [&](const Intr::LogicOr &v) { return fmt::format("({} || {})", mkExpr(v.x), mkExpr(v.y)); },
                                [&](const Intr::LogicEq &v) { return fmt::format("({} == {})", mkExpr(v.x), mkExpr(v.y)); },
                                [&](const Intr::LogicNeq &v) { return fmt::format("({} != {})", mkExpr(v.x), mkExpr(v.y)); },
                                [&](const Intr::LogicLte &v) { return fmt::format("({} <= {})", mkExpr(v.x), mkExpr(v.y)); },
                                [&](const Intr::LogicGte &v) { return fmt::format("({} >= {})", mkExpr(v.x), mkExpr(v.y)); },
                                [&](const Intr::LogicLt &v) { return fmt::format("({} < {})", mkExpr(v.x), mkExpr(v.y)); },
                                [&](const Intr::LogicGt &v) { return fmt::format("({} > {})", mkExpr(v.x), mkExpr(v.y)); });
      },
      [&](const Expr::MathOp &x) {
        return x.op.match_total([&](const Math::Abs &v) { return fmt::format("abs({})", mkExpr(v.x)); },
                                [&](const Math::Sin &v) { return fmt::format("sin({})", mkExpr(v.x)); },
                                [&](const Math::Cos &v) { return fmt::format("cos({})", mkExpr(v.x)); },
                                [&](const Math::Tan &v) { return fmt::format("tan({})", mkExpr(v.x)); },
                                [&](const Math::Asin &v) { return fmt::format("asin({})", mkExpr(v.x)); },
                                [&](const Math::Acos &v) { return fmt::format("acos({})", mkExpr(v.x)); },
                                [&](const Math::Atan &v) { return fmt::format("atan({})", mkExpr(v.x)); },
                                [&](const Math::Sinh &v) { return fmt::format("sinh({})", mkExpr(v.x)); },
                                [&](const Math::Cosh &v) { return fmt::format("cosh({})", mkExpr(v.x)); },
                                [&](const Math::Tanh &v) { return fmt::format("tanh({})", mkExpr(v.x)); },
                                [&](const Math::Signum &v) { return fmt::format("signum({})", mkExpr(v.x)); },
                                [&](const Math::Round &v) { return fmt::format("round({})", mkExpr(v.x)); },
                                [&](const Math::Ceil &v) { return fmt::format("ceil({})", mkExpr(v.x)); },
                                [&](const Math::Floor &v) { return fmt::format("floor({})", mkExpr(v.x)); },
                                [&](const Math::Rint &v) { return fmt::format("rint({})", mkExpr(v.x)); },
                                [&](const Math::Sqrt &v) { return fmt::format("sqrt({})", mkExpr(v.x)); },
                                [&](const Math::Cbrt &v) { return fmt::format("cbrt({})", mkExpr(v.x)); },
                                [&](const Math::Exp &v) { return fmt::format("exp({})", mkExpr(v.x)); },
                                [&](const Math::Expm1 &v) { return fmt::format("expm1({})", mkExpr(v.x)); },
                                [&](const Math::Log &v) { return fmt::format("log({})", mkExpr(v.x)); },
                                [&](const Math::Log1p &v) { return fmt::format("log1p({})", mkExpr(v.x)); },
                                [&](const Math::Log10 &v) { return fmt::format("log10({})", mkExpr(v.x)); },
                                [&](const Math::Pow &v) { return fmt::format("pow({}, {})", mkExpr(v.x), mkExpr(v.y)); },
                                [&](const Math::Atan2 &v) { return fmt::format("atan2({}, {})", mkExpr(v.x), mkExpr(v.y)); },
                                [&](const Math::Hypot &v) { return fmt::format("hypot({}, {})", mkExpr(v.x), mkExpr(v.y)); });
      },
      [&](const Expr::Select &x) {
        return x.init ^
               fold_left(std::string{},
                         [&](auto &&acc, auto &n) {
                           acc += normalise(n.symbol);
                           acc += n.tpe.template is<Type::Ptr>() ? "->" : ".";
                           return acc;
                         }) ^
               concat(normalise(x.last.symbol));
      },                                                                                //
      [&](const Expr::Poison &x) { return fmt::format("(NULL /*{}*/)", repr(x.tpe)); }, //
      [&](const Expr::Cast &x) { return fmt::format("(({}) {})", mkTpe(x.as), mkExpr(x.from)); },
      [&](const Expr::Invoke &x) {
        return fmt::format("{}({})", normalise(x.name), x.args ^ mk_string(", ", [&](auto &arg) { return mkExpr(arg); }));
      }, //
      [&](const Expr::Index &x) { return fmt::format("{}[{}]", mkExpr(x.lhs), mkExpr(x.idx)); },
      [&](const Expr::RefTo &x) {
        std::string str = fmt::format("&({} /*{}*/)", mkExpr(x.lhs), mkTpe(x.comp));
        if (x.idx) str += fmt::format("[{}]", mkExpr(*x.idx));
        return str;
      },
      [&](const Expr::Alloc &x) { return fmt::format("{{/*{}*/}}", to_string(x)); },
      [&](const Expr::Annotated &x) {
        return fmt::format("{} /*{};{}*/", mkExpr(x.expr), x.pos ^ map(show_repr) ^ get_or_else(""), x.comment ^ get_or_else(""));
      } //

  );
}

std::string backend::CSource::mkStmt(const Stmt::Any &stmt) {
  return stmt.match_total( //
      [&](const Stmt::Block &x) { return x.stmts ^ mk_string("\n", [&](auto &x) { return mkStmt(x); }); },
      [&](const Stmt::Comment &x) {
        return x.value ^ split("\n") | map([](auto &l) { return fmt::format("// {}", l); }) | mk_string("\n");
      },
      [&](const Stmt::Var &x) {
        if (x.name.tpe.is<Type::Unit0>()) return fmt::format("{};", mkExpr(*x.expr));
        return fmt::format("{} {}{};", mkTpe(x.name.tpe), normalise(x.name.symbol), x.expr ? (" = " + mkExpr(*x.expr)) : "");
      },
      [&](const Stmt::Mut &x) { return fmt::format("{} = {};", mkExpr(x.name), mkExpr(x.expr)); },
      [&](const Stmt::Update &x) { return fmt::format("{}[{}] = {};", mkExpr(x.lhs), mkExpr(x.idx), mkExpr(x.value)); },
      [&](const Stmt::While &x) {
        auto body = x.body | mk_string("\n", [&](auto &stmt) { return mkStmt(stmt); });
        auto tests = x.tests | mk_string("\n", [&](auto &stmt) { return mkStmt(stmt); });
        auto whileBody = fmt::format("{}\n  if(!{}) break;\n{}", tests ^ indent(2), mkExpr(x.cond), body ^ indent(2));
        return fmt::format("while(true) {{\n{}\n}}", whileBody);
      },
      [&](const Stmt::Break &) { return "break;"s; },   //
      [&](const Stmt::Cont &) { return "continue;"s; }, //
      [&](const Stmt::Cond &x) {
        auto trueBr = x.trueBr ^ mk_string("{\n", "\n", "\n}", [&](auto x) { return mkStmt(x) ^ indent(2); });
        if (x.falseBr.empty()) {
          return fmt::format("if ({}) {}", mkExpr(x.cond), trueBr);
        } else {
          auto falseBr = x.falseBr ^ mk_string("{\n", "\n", "\n}", [&](auto x) { return mkStmt(x) ^ indent(2); });
          return fmt::format("if ({}) {} else {}", mkExpr(x.cond), trueBr, falseBr);
        }
      },
      [&](const Stmt::Return &x) { return "return " + mkExpr(x.value) + ";"; }, //
      [&](const Stmt::Annotated &x) {
        return fmt::format("{} /*{};{}*/", mkStmt(x.stmt), x.pos ^ map(show_repr) ^ get_or_else(""), x.comment ^ get_or_else(""));
      } //
  );
}

std::string backend::CSource::mkFnProto(const Function &fnTree) {

  const auto entry = fnTree.attrs.contains(FunctionAttr::Entry());

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
      } else if (auto cond = stmt.get<Stmt::Cond>(); cond) {
        findIntrinsics(cond->cond);
      }
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
  auto structsAndDeps = program.structs ^ map([&](auto &def) {
                          auto deps = def.members | collect([&](auto &m) { return m.tpe.template get<Type::Struct>(); }) |
                                      map([&](auto &s) { return s.name; });
                          return std::pair{def, deps};
                        }) ^
                        to<Map>();

  std::vector<std::string> fragments;
  switch (dialect) {
    case Dialect::C11: fragments.emplace_back("#include <stdint.h>\n#include <stdbool.h>"); break;
    default: break;
  }

  Set<std::string> resolved;
  while (resolved.size() != program.structs.size()) {
    auto noDeps =
        structsAndDeps | filter([&](auto &, auto &deps) { return deps | forall([&](auto &d) { return resolved.contains(d); }); }) | keys();
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
