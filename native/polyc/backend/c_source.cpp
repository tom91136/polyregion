#include <iostream>

#include "aspartame/all.hpp"
#include "c_source.h"
// #include "polyregion/utils.hpp"

#include "fmt/core.h"
#include <set>

using namespace aspartame;
using namespace polyregion;
using namespace std::string_literals;

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

                             [&](const Type::Struct &x) { return x.name; },                              //
                             [&](const Type::Ptr &x) { return fmt::format("{}*", mkTpe(x.component)); }, //
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

                             [&](const Type::Struct &x) { return x.name; },                              //
                             [&](const Type::Ptr &x) { return fmt::format("{}*", mkTpe(x.component)); }, //
                             [&](const Type::Annotated &x) {
                               return fmt::format("{} /*{};{}*/", mkTpe(x.tpe), x.pos ^ map(show_repr) ^ get_or_else(""),
                                                  x.comment ^ get_or_else(""));
                             } //
      );
  }
}

std::string backend::CSource::mkExpr(const Expr::Any &expr) {
  return expr.match_total(
      [](const Expr::Float16Const &x) { return fmt::format("{}", x.value); },  //
      [](const Expr::Float32Const &x) { return fmt::format("{}f", x.value); }, //
      [](const Expr::Float64Const &x) { return fmt::format("{}", x.value); },  //

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
      [&](const Expr::Select &x) { return polyast::qualified(x); },                     //
      [&](const Expr::Poison &x) { return fmt::format("(NULL /*{}*/)", repr(x.tpe)); }, //

      [&](const Expr::Cast &x) { return fmt::format("(({}) {})", mkTpe(x.as), mkExpr(x.from)); },
      [&](const Expr::Invoke &x) { return "???"s; }, //
      [&](const Expr::Index &x) { return fmt::format("{}[{}]", mkExpr(x.lhs), mkExpr(x.idx)); },
      [&](const Expr::RefTo &x) {
        std::string str = fmt::format("&{}", mkExpr(x.lhs));
        if (x.idx) str += fmt::format("[{}]", mkExpr(*x.idx));
        return str;
      },
      [&](const Expr::Alloc &x) { return fmt::format("{{/*{}*/}}", to_string(x)); },
      [&](const Expr::Annotated &x) {
        return fmt::format("{} /*{};{}*/", mkExpr(x), x.pos ^ map(show_repr) ^ get_or_else(""), x.comment ^ get_or_else(""));
      } //

  );
}

std::string backend::CSource::mkStmt(const Stmt::Any &stmt) {
  return stmt.match_total( //
      [&](const Stmt::Block &x) { return x.stmts | mk_string("\n", [&](auto &x) { return mkStmt(x); }); },
      [&](const Stmt::Comment &x) {
        return x.value ^ split("\n") | map([](auto &l) { return fmt::format("// {}", l); }) | mk_string("\n");
      },
      [&](const Stmt::Var &x) {
        if (x.name.tpe.is<Type::Unit0>()) return fmt::format("{};", mkExpr(*x.expr));
        return fmt::format("{} {}{};", mkTpe(x.name.tpe), x.name.symbol, x.expr ? (" = " + mkExpr(*x.expr)) : "");
      },
      [&](const Stmt::Mut &x) { return fmt::format("{} = {};", mkExpr(x.name), mkExpr(x.expr)); },
      [&](const Stmt::Update &x) { return fmt::format("{}[{}] = {};", mkExpr(x.lhs), mkExpr(x.idx), mkExpr(x.value)); },
      [&](const Stmt::While &x) {
        auto body = x.body | mk_string("{\n", "\n", "\n}", [&](auto &stmt) { return mkStmt(stmt) ^ indent(2); });
        auto tests = x.tests | mk_string("\n", [&](auto &stmt) { return mkStmt(stmt); });
        auto whileBody = fmt::format("{}\nif(!{}) break;\n{}", tests, mkExpr(x.cond), body);
        return fmt::format("while(true) {{\n{}\n}}", whileBody);
      },
      [&](const Stmt::Break &x) { return "break;"s; },   //
      [&](const Stmt::Cont &x) { return "continue;"s; }, //
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
        return fmt::format("{} /*{};{}*/", mkStmt(x), x.pos ^ map(show_repr) ^ get_or_else(""), x.comment ^ get_or_else(""));
      } //
  );
}

std::string backend::CSource ::mkFn(const Function &fnTree) {

  std::vector<std::string> argExprs =
      fnTree.args | zip_with_index() | map([&](auto &arg, auto idx) {
        auto tpe = mkTpe(arg.named.tpe);
        auto name = arg.named.symbol;
        std::string decl;
        switch (dialect) {
          case Dialect::OpenCL1_1: {
            if (auto arr = arg.named.tpe.template get<Type::Ptr>()) {
              decl = arr->space.match_total([&](TypeSpace::Global) { return fmt::format("global {} {}", tpe, name); },
                                            [&](TypeSpace::Local) { return fmt::format("local {} {}", tpe, name); });
            } else decl = fmt::format("{} {}", tpe, name);

            break;
          }
          case Dialect::MSL1_0: {
            // Scalar:      device $T &$name      [[ buffer($i) ]]
            // Global:      device $T  $name      [[ buffer($i) ]]
            // Local:  threadgroup $T &$name [[ threadgroup($i) ]]
            // query:              $T &$name           [[ $type ]]
            if (auto arr = arg.named.tpe.template get<Type::Ptr>()) {
              decl = arr->space.match_total(
                  [&](TypeSpace::Global) { return fmt::format("device {} {} [[buffer({})]]", tpe, name, idx); },         //
                  [&](TypeSpace::Local) { return fmt::format("threadgroup {} {} [[threadgroup({})]]", tpe, name, idx); } //
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
    case Dialect::OpenCL1_1: fnPrefix = "kernel "; break;
    default: fnPrefix = "";
  }

  // TODO OpenCL: collect types and see if we have any Double and prepend:
  //  #pragma OPENCL EXTENSION cl_khr_fp64 : enable
  //   Possible extensions:
  //   cl_khr_fp64                    Double precision floating-point
  //   cl_khr_int64_base_atomics      64-bit integer base atomic operations
  //   cl_khr_int64_extended_atomics  64-bit integer extended atomic operations
  //   cl_khr_fp16                    Half-precision floating-point

  // TODO OpenCL implement memory space prefix for arguments
  //  Possible values:
  //  constant, local, global, private

  return fmt::format("{}{} {}({}) {}",
                     fnPrefix,                   //
                     mkTpe(fnTree.rtn),          //
                     fnTree.name,                //
                     argExprs | mk_string(", "), //
                     fnTree.body ^ mk_string("{\n", "\n", "\n}", [&](auto &s) { return mkStmt(s) ^ indent(2); }));
}

polyast::CompileResult backend::CSource::compileProgram(const Program &program, const compiletime::OptLevel &opt) {
  auto start = compiler::nowMono();

  auto structDefs =
      program.structs | mk_string("\n", [&](auto &s) {
        return fmt::format("typedef struct {}",
                           s.members | mk_string("{", "\n", "}", [&](auto &m) { return fmt::format("  {} {};", mkTpe(m.tpe), m.symbol); }));
      });

  std::vector<std::string> fragments;
  switch (dialect) {
    case Dialect::C11: fragments.emplace_back("#include <stdint.h>\n#include <stdbool.h>"); break;
    default: break;
  }

  fragments.emplace_back(structDefs);
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
          {{compiler::nowMs(), compiler::elapsedNs(start), "polyast_to_" + dialectName + "_c", code}},
          {},
          ""};
}
std::vector<polyast::StructLayout> backend::CSource::resolveLayouts(const std::vector<StructDef> &defs) {
  return std::vector<StructLayout>();
}
