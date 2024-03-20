#include <iostream>

#include "c_source.h"
#include "utils.hpp"
#include <set>

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

          [&](const Type::Bool1 &) { return "bool"s; },                   //
          [&](const Type::Unit0 &) { return "void"s; },                   //
          [&](const Type::Nothing &) { return "/*nothing*/"s; },          //
          [&](const Type::Struct &x) { return qualified(x.name); },       //
          [&](const Type::Ptr &x) { return mkTpe(x.component) + "*"; }, //
          [&](const Type::Var &) { return "/*type var*/"s; },             //
          [&](const Type::Exec &) { return "/*exec*/"s; }                 //
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

          [&](const Type::Bool1 &) { return "char"s; },                   //
          [&](const Type::Unit0 &) { return "void"s; },                   //
          [&](const Type::Nothing &) { return "/*nothing*/"s; },          //
          [&](const Type::Struct &x) { return qualified(x.name); },       //
          [&](const Type::Ptr &x) { return mkTpe(x.component) + "*"; }, //
          [&](const Type::Var &) { return "/*type var*/"s; },             //
          [&](const Type::Exec &) { return "/*exec*/"s; }                 //
      );
  }
}

std::string backend::CSource::mkRef(const Term::Any &ref) {
  return ref.match_total([](const Term::Select &x) { return polyast::qualified(x); },               //
                         [](const Term::Poison &x) { return "(NULL /* " + repr(x.tpe) + " */)"s; }, //
      [](const Term::Unit0Const &x) { return "/*void*/"s; },                     //
      [](const Term::Bool1Const &x) { return x.value ? "true"s : "false"s; },    //

      [](const Term::IntU8Const &x) { return std::to_string(x.value); },  //
      [](const Term::IntU16Const &x) { return std::to_string(x.value); }, //
      [](const Term::IntU32Const &x) { return std::to_string(x.value); }, //
      [](const Term::IntU64Const &x) { return std::to_string(x.value); }, //

      [](const Term::IntS8Const &x) { return std::to_string(x.value); },  //
      [](const Term::IntS16Const &x) { return std::to_string(x.value); }, //
      [](const Term::IntS32Const &x) { return std::to_string(x.value); }, //
      [](const Term::IntS64Const &x) { return std::to_string(x.value); }, //

      [](const Term::Float64Const &x) { return std::to_string(x.value); },       //
      [](const Term::Float32Const &x) { return std::to_string(x.value) + "f"; }, //
      [](const Term::Float16Const &x) { return std::to_string(x.value) + ""; }   //
  );                                                                             // FIXME escape string
}

std::string backend::CSource::mkExpr(const Expr::Any &expr, const std::string &key) {
  return expr.match_total(
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
            case Dialect::MSL1_0: return accessor.msl + "[" + mkRef(index) + "]";
            case Dialect::OpenCL1_1: return accessor.cl + "(" + mkRef(index) + ")";
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
        return x.op.match_total([&](const Intr::Pos &v) { return "(+" + mkRef(v.x) + ")"; },                           //
                                [&](const Intr::Neg &v) { return "(-" + mkRef(v.x) + ")"; },                           //
            [&](const Intr::BNot &v) { return "(~" + mkRef(v.x) + ")"; },                          //
            [&](const Intr::LogicNot &v) { return "(!" + mkRef(v.x) + ")"; },                      //
            [&](const Intr::Add &v) { return "(" + mkRef(v.x) + " + " + mkRef(v.y) + ")"; },       //
            [&](const Intr::Sub &v) { return "(" + mkRef(v.x) + " - " + mkRef(v.y) + ")"; },       //
            [&](const Intr::Mul &v) { return "(" + mkRef(v.x) + " * " + mkRef(v.y) + ")"; },       //
            [&](const Intr::Div &v) { return "(" + mkRef(v.x) + " / " + mkRef(v.y) + ")"; },       //
            [&](const Intr::Rem &v) { return "(" + mkRef(v.x) + " % " + mkRef(v.y) + ")"; },       //
            [&](const Intr::Min &v) { return "min(" + mkRef(v.x) + ", " + mkRef(v.y) + ")"; },     //
            [&](const Intr::Max &v) { return "max(" + mkRef(v.x) + ", " + mkRef(v.y) + ")"; },     //
            [&](const Intr::BAnd &v) { return "(" + mkRef(v.x) + " & " + mkRef(v.y) + ")"; },      //
            [&](const Intr::BOr &v) { return "(" + mkRef(v.x) + " | " + mkRef(v.y) + ")"; },       //
            [&](const Intr::BXor &v) { return "(" + mkRef(v.x) + " ^ " + mkRef(v.y) + ")"; },      //
            [&](const Intr::BSL &v) { return "(" + mkRef(v.x) + " << " + mkRef(v.y) + ")"; },      //
            [&](const Intr::BSR &v) { return "(" + mkRef(v.x) + " >> " + mkRef(v.y) + ")"; },      //
            [&](const Intr::BZSR &v) { return "(" + mkRef(v.x) + " <<< " + mkRef(v.y) + ")"; },    //
            [&](const Intr::LogicAnd &v) { return "(" + mkRef(v.x) + " && " + mkRef(v.y) + ")"; }, //
            [&](const Intr::LogicOr &v) { return "(" + mkRef(v.x) + " || " + mkRef(v.y) + ")"; },  //
            [&](const Intr::LogicEq &v) { return "(" + mkRef(v.x) + " == " + mkRef(v.y) + ")"; },  //
            [&](const Intr::LogicNeq &v) { return "(" + mkRef(v.x) + " != " + mkRef(v.y) + ")"; }, //
            [&](const Intr::LogicLte &v) { return "(" + mkRef(v.x) + " <= " + mkRef(v.y) + ")"; }, //
            [&](const Intr::LogicGte &v) { return "(" + mkRef(v.x) + " >= " + mkRef(v.y) + ")"; }, //
            [&](const Intr::LogicLt &v) { return "(" + mkRef(v.x) + " < " + mkRef(v.y) + ")"; },   //
            [&](const Intr::LogicGt &v) { return "(" + mkRef(v.x) + " > " + mkRef(v.y) + ")"; }    //
        );
      },
      [&](const Expr::MathOp &x) {
        return x.op.match_total([&](const Math::Abs &v) { return "abs(" + mkRef(v.x) + ")"; },                         //
                                [&](const Math::Sin &v) { return "sin(" + mkRef(v.x) + ")"; },                         //
            [&](const Math::Cos &v) { return "cos(" + mkRef(v.x) + ")"; },                         //
            [&](const Math::Tan &v) { return "tan(" + mkRef(v.x) + ")"; },                         //
            [&](const Math::Asin &v) { return "asin(" + mkRef(v.x) + ")"; },                       //
            [&](const Math::Acos &v) { return "acos(" + mkRef(v.x) + ")"; },                       //
            [&](const Math::Atan &v) { return "atan(" + mkRef(v.x) + ")"; },                       //
            [&](const Math::Sinh &v) { return "sinh(" + mkRef(v.x) + ")"; },                       //
            [&](const Math::Cosh &v) { return "cosh(" + mkRef(v.x) + ")"; },                       //
            [&](const Math::Tanh &v) { return "tanh(" + mkRef(v.x) + ")"; },                       //
            [&](const Math::Signum &v) { return "signum(" + mkRef(v.x) + ")"; },                   //
            [&](const Math::Round &v) { return "round(" + mkRef(v.x) + ")"; },                     //
            [&](const Math::Ceil &v) { return "ceil(" + mkRef(v.x) + ")"; },                       //
            [&](const Math::Floor &v) { return "floor(" + mkRef(v.x) + ")"; },                     //
            [&](const Math::Rint &v) { return "rint(" + mkRef(v.x) + ")"; },                       //
            [&](const Math::Sqrt &v) { return "sqrt(" + mkRef(v.x) + ")"; },                       //
            [&](const Math::Cbrt &v) { return "cbrt(" + mkRef(v.x) + ")"; },                       //
            [&](const Math::Exp &v) { return "exp(" + mkRef(v.x) + ")"; },                         //
            [&](const Math::Expm1 &v) { return "expm1(" + mkRef(v.x) + ")"; },                     //
            [&](const Math::Log &v) { return "log(" + mkRef(v.x) + ")"; },                         //
            [&](const Math::Log1p &v) { return "log1p(" + mkRef(v.x) + ")"; },                     //
            [&](const Math::Log10 &v) { return "log10(" + mkRef(v.x) + ")"; },                     //
            [&](const Math::Pow &v) { return "pow(" + mkRef(v.x) + ", " + mkRef(v.y) + ")"; },     //
            [&](const Math::Atan2 &v) { return "atan2(" + mkRef(v.x) + ", " + mkRef(v.y) + ")"; }, //
            [&](const Math::Hypot &v) { return "hypot(" + mkRef(v.x) + ", " + mkRef(v.y) + ")"; }  //
        );
      },
      [&](const Expr::Cast &x) { return "((" + mkTpe(x.as) + ") " + mkRef(x.from) + ")"; },
      [&](const Expr::Alias &x) { return mkRef(x.ref); },                            //
      [&](const Expr::Invoke &x) { return "???"s; },                                 //
      [&](const Expr::Index &x) { return mkRef(x.lhs) + "[" + mkRef(x.idx) + "]"; }, //
      [&](const Expr::RefTo &x) {
        std::string str = "&" + mkRef(x.lhs);
        if (x.idx) str += "[" + mkRef(*x.idx) + "]";
        return str;
      },                                                                 //
      [&](const Expr::Alloc &x) { return "{/*" + to_string(x) + "*/}"; } //
  );
}

std::string backend::CSource::mkStmt(const Stmt::Any &stmt) {
  return stmt.match_total(
      [&](const Stmt::Block &x) {
        return mk_string<Stmt::Any>(
            x.stmts, [&](auto x) { return mkStmt(x); }, "\n");
      },
      [&](const Stmt::Comment &x) {
        auto lines = split(x.value, '\n');
        auto commented = map_vec<std::string, std::string>(lines, [](auto &&line) { return "// " + line; });
        return mk_string<std::string>(
            commented, [](auto &&x) { return x; }, "\n");
      },
      [&](const Stmt::Var &x) {
        if (x.name.tpe.is<Type::Unit0>()) {
          return mkExpr(*x.expr, x.name.symbol) + ";";
        }
        auto line = mkTpe(x.name.tpe) + " " + x.name.symbol + (x.expr ? (" = " + mkExpr(*x.expr, x.name.symbol)) : "") + ";";
        return line;
      },
      [&](const Stmt::Mut &x) {
        if (x.copy) {
          return "memcpy(" + //
                 mkRef(x.name) + ", " + mkExpr(x.expr, "?") + ", sizeof(" + mkTpe(x.expr.tpe()) + "));";
        } else {
          return mkRef(x.name) + " = " + mkExpr(x.expr, "?") + ";";
        }
      },
      [&](const Stmt::Update &x) {
        auto idx = mkRef(x.idx);
        auto val = mkRef(x.value);
        return mkRef(x.lhs) + "[" + idx + "] = " + val + ";";
      },
      [&](const Stmt::While &x) {
        auto body = mk_string<Stmt::Any>(
            x.body, [&](auto &stmt) { return mkStmt(stmt); }, "\n");

        auto tests = mk_string<Stmt::Any>(
            x.tests, [&](auto &stmt) { return mkStmt(stmt); }, "\n");

        auto whileBody = tests + "\nif(!" + mkRef(x.cond) + ") break;" + "\n" + body;

        return "while(true) {\n" + indent(2, whileBody) + "\n}";
      },
      [&](const Stmt::Break &x) { return "break;"s; },   //
      [&](const Stmt::Cont &x) { return "continue;"s; }, //
      [&](const Stmt::Cond &x) {
        auto elseStmts = x.falseBr.empty() //
                             ? "\n}"
                             : "\n} else {\n" +
                                   indent(2, mk_string<Stmt::Any>(
                                                 x.falseBr, [&](auto x) { return mkStmt(x); }, "\n")) +
                                   "}";

        return "if(" + mkExpr(x.cond, "if") + ") {\n" +
               indent(2, mk_string<Stmt::Any>(
                             x.trueBr, [&](auto x) { return mkStmt(x); }, "\n")) +
               elseStmts;
      },
      [&](const Stmt::Return &x) { return "return " + mkExpr(x.value, "rtn") + ";"; } //
  );
}

std::string backend::CSource ::mkFn(const Function &fnTree) {

  std::vector<Arg> allArgs;
  if (fnTree.receiver) allArgs.insert(allArgs.begin(), *fnTree.receiver);
  allArgs.insert(allArgs.begin(), fnTree.args.begin(), fnTree.args.end());
  allArgs.insert(allArgs.begin(), fnTree.moduleCaptures.begin(), fnTree.moduleCaptures.end());
  allArgs.insert(allArgs.begin(), fnTree.termCaptures.begin(), fnTree.termCaptures.end());

  std::vector<std::string> argExprs(allArgs.size());
  for (size_t i = 0; i < allArgs.size(); ++i) {
    auto arg = allArgs[i];
    auto tpe = mkTpe(arg.named.tpe);
    auto name = arg.named.symbol;
    std::string decl;
    switch (dialect) {
      case Dialect::OpenCL1_1: {
        if (auto arr = arg.named.tpe.get<Type::Ptr>(); arr) {
          decl = arr->space.match_total([&](TypeSpace::Global _) { return "global " + tpe + " " + name; }, //
                                        [&](TypeSpace::Local _) { return "local " + tpe + " " + name; });
        } else
          decl = tpe + " " + name;
        break;
      }
      case Dialect::MSL1_0: {
        // Scalar:      device $T &$name      [[ buffer($i) ]]
        // Global:      device $T  $name      [[ buffer($i) ]]
        // Local:  threadgroup $T &$name [[ threadgroup($i) ]]
        // query:              $T &$name           [[ $type ]]
        auto idx = std::to_string(i);
        if (auto arr = arg.named.tpe.get<Type::Ptr>(); arr) {
          decl = arr->space.match_total(
              [&](TypeSpace::Global _) { return "device " + tpe + " " + name + " [[buffer(" + idx + ")]]"; },         //
              [&](TypeSpace::Local _) { return "threadgroup " + tpe + " " + name + " [[threadgroup(" + idx + ")]]"; } //
          );
        } else {
          decl = "device " + tpe + " &" + name + " [[buffer(" + idx + ")]]";
        };
        break;
      }
      default: break;
    }
    argExprs[i] = decl;
  }

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
      argExprs.push_back("uint3 __" + name + "__ [[ " + attr + " ]]");
  }

  auto args = mk_string<std::string>(
      argExprs, [&](auto x) { return x; }, ", ");

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

  auto prototype = fnPrefix + mkTpe(fnTree.rtn) + " " + qualified(fnTree.name) + "(" + args + ")";

  auto body = mk_string<Stmt::Any>(
      fnTree.body, [&](auto &stmt) { return mkStmt(stmt); }, "\n");

  return prototype + " {\n" + indent(2, body) + "\n}";
}

polyast::CompileResult backend::CSource::compileProgram(const Program &program, const compiletime::OptLevel &opt) {
  auto start = compiler::nowMono();

  auto structDefs = mk_string<StructDef>(
      program.defs,
      [&](auto &x) {
        return std::accumulate(                                                                                       //
                   x.members.begin(), x.members.end(),                                                                //
                   "typedef struct {"s,                                                                               //
                   [&](auto &&acc, auto m) { return acc + "\n  " + mkTpe(m.named.tpe) + " " + m.named.symbol + ";"; } //
                   ) +
               "\n} " + qualified(x.name) + ";";
      },
      "\n");

  std::vector<std::string> lines;

  switch (dialect) {
    case Dialect::C11: lines.emplace_back("#include <stdint.h>\n#include <stdbool.h>"); break;
    default: break;
  }

  lines.push_back(structDefs);
  lines.push_back(mkFn(program.entry));
  for (auto &f : program.functions)
    lines.push_back(mkFn(f));

  auto def = mk_string<std::string>(lines, [](auto x) { return x; }, "\n");

  std::vector<int8_t> data(def.begin(), def.end());

  std::string dialectName;
  switch (dialect) {
    case Dialect::C11: dialectName = "c11"; break;
    case Dialect::OpenCL1_1: dialectName = "opencl1_1"; break;
    case Dialect::MSL1_0: dialectName = "msl1"; break;
    default: dialectName = "unknown";
  }

  return {data, {}, {{compiler::nowMs(), compiler::elapsedNs(start), "polyast_to_" + dialectName + "_c", def}}, {}, ""};
}
std::vector<polyast::CompileLayout> backend::CSource::resolveLayouts(const std::vector<StructDef> &defs, const compiletime::OptLevel &opt) {
  return std::vector<polyast::CompileLayout>();
}
