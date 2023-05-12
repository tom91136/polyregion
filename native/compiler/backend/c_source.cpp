#include <iostream>

#include "c_source.h"
#include "utils.hpp"
#include "variants.hpp"
#include <set>

using namespace polyregion;
using namespace std::string_literals;

std::string backend::CSource::mkTpe(const Type::Any &tpe) {

  switch (dialect) {
    case Dialect::C11:
    case Dialect::MSL1_0:
      return variants::total(
          *tpe,                                                           //
          [&](const Type::Float &x) { return "float"s; },                 //
          [&](const Type::Double &x) { return "double"s; },               //
          [&](const Type::Bool &x) { return "bool"s; },                   //
          [&](const Type::Byte &x) { return "int8_t"s; },                 //
          [&](const Type::Char &x) { return "uint16_t"s; },               //
          [&](const Type::Short &x) { return "int16_t"s; },               //
          [&](const Type::Int &x) { return "int32_t"s; },                 //
          [&](const Type::Long &x) { return "int64_t"s; },                //
          [&](const Type::Unit &x) { return "void"s; },                   //
          [&](const Type::Nothing &x) { return "/*nothing*/"s; },         //
          [&](const Type::Struct &x) { return qualified(x.name); },       //
          [&](const Type::Array &x) { return mkTpe(x.component) + "*"; }, //
          [&](const Type::Var &x) { return "/*type var*/"s; },            //
          [&](const Type::Exec &x) { return "/*exec*/"s; }                //
      );
    case Dialect::OpenCL1_1:
      return variants::total(
          *tpe,                                                           //
          [&](const Type::Float &x) { return "float"s; },                 //
          [&](const Type::Double &x) { return "double"s; },               //
          [&](const Type::Bool &x) { return "char"s; },                   //
          [&](const Type::Byte &x) { return "char"s; },                   //
          [&](const Type::Char &x) { return "ushort"s; },                 //
          [&](const Type::Short &x) { return "short"s; },                 //
          [&](const Type::Int &x) { return "int"s; },                     //
          [&](const Type::Long &x) { return "long"s; },                   //
          [&](const Type::Unit &x) { return "void"s; },                   //
          [&](const Type::Nothing &x) { return "/*nothing*/"s; },         //
          [&](const Type::Struct &x) { return qualified(x.name); },       //
          [&](const Type::Array &x) { return mkTpe(x.component) + "*"; }, //
          [&](const Type::Var &x) { return "/*type var*/"s; },            //
          [&](const Type::Exec &x) { return "/*exec*/"s; }                //
      );
  }
}

std::string backend::CSource::mkRef(const Term::Any &ref) {
  return variants::total(
      *ref,                                                                      //
      [](const Term::Select &x) { return polyast::qualified(x); },               //
      [](const Term::Poison &x) { return "(NULL /* " + repr(x.tpe) + " */)"s; }, //
      [](const Term::UnitConst &x) { return "/*void*/"s; },                      //
      [](const Term::BoolConst &x) { return x.value ? "true"s : "false"s; },     //
      [](const Term::ByteConst &x) { return std::to_string(x.value); },          //
      [](const Term::CharConst &x) { return std::to_string(x.value); },          //
      [](const Term::ShortConst &x) { return std::to_string(x.value); },         //
      [](const Term::IntConst &x) { return std::to_string(x.value); },           //
      [](const Term::LongConst &x) { return std::to_string(x.value); },          //
      [](const Term::DoubleConst &x) { return std::to_string(x.value); },        //
      [](const Term::FloatConst &x) { return std::to_string(x.value) + "f"; }    //
  );                                                                             // FIXME escape string
}

std::string backend::CSource::mkExpr(const Expr::Any &expr, const std::string &key) {
  return variants::total(
      *expr, //
      [&](const Expr::NullaryIntrinsic &x) {
        switch (dialect) {
          case Dialect::C11:
          case Dialect::MSL1_0:
            return variants::total(
                *x.kind, //
                [](const NullaryIntrinsicKind::Assert &) { return "abort()"s; },
                [](const NullaryIntrinsicKind::GpuGlobalIdxX &) { return "___get_global_id___.x"s; },
                [](const NullaryIntrinsicKind::GpuGlobalIdxY &) { return "___get_global_id___.y"s; },
                [](const NullaryIntrinsicKind::GpuGlobalIdxZ &) { return "___get_global_id___.z"s; },
                [](const NullaryIntrinsicKind::GpuGlobalSizeX &) { return "___get_global_size___.x"s; },
                [](const NullaryIntrinsicKind::GpuGlobalSizeY &) { return "___get_global_size___.y"s; },
                [](const NullaryIntrinsicKind::GpuGlobalSizeZ &) { return "___get_global_size___.z"s; },
                [](const NullaryIntrinsicKind::GpuGroupIdxX &) { return "___get_group_id___.x"s; },
                [](const NullaryIntrinsicKind::GpuGroupIdxY &) { return "___get_group_id___.y"s; },
                [](const NullaryIntrinsicKind::GpuGroupIdxZ &) { return "___get_group_id___.z"s; },
                [](const NullaryIntrinsicKind::GpuGroupSizeX &) { return "___get_num_groups___.x"s; },
                [](const NullaryIntrinsicKind::GpuGroupSizeY &) { return "___get_num_groups___.y"s; },
                [](const NullaryIntrinsicKind::GpuGroupSizeZ &) { return "___get_num_groups___.z"s; },
                [](const NullaryIntrinsicKind::GpuLocalIdxX &) { return "___get_local_id___.x"s; },
                [](const NullaryIntrinsicKind::GpuLocalIdxY &) { return "___get_local_id___.y"s; },
                [](const NullaryIntrinsicKind::GpuLocalIdxZ &) { return "___get_local_id___.z"s; },
                [](const NullaryIntrinsicKind::GpuLocalSizeX &) { return "___get_local_size___.x"s; },
                [](const NullaryIntrinsicKind::GpuLocalSizeY &) { return "___get_local_size___.y"s; },
                [](const NullaryIntrinsicKind::GpuLocalSizeZ &) { return "___get_local_size___.z"s; },
                [](const NullaryIntrinsicKind::GpuGroupBarrier &) {
                  return "threadgroup_barrier(metal::mem_flags::mem_threadgroup)"s;
                },
                [](const NullaryIntrinsicKind::GpuGroupFence &) {
                  return "threadgroup_barrier(metal::mem_flags::mem_threadgroup)"s;
                });
          case Dialect::OpenCL1_1:
            return variants::total(
                *x.kind, //
                [](const NullaryIntrinsicKind::Assert &) { return "abort()"s; },
                [](const NullaryIntrinsicKind::GpuGlobalIdxX &) { return "get_global_id(0)"s; },
                [](const NullaryIntrinsicKind::GpuGlobalIdxY &) { return "get_global_id(1)"s; },
                [](const NullaryIntrinsicKind::GpuGlobalIdxZ &) { return "get_global_id(2)"s; },
                [](const NullaryIntrinsicKind::GpuGlobalSizeX &) { return "get_global_size(0)"s; },
                [](const NullaryIntrinsicKind::GpuGlobalSizeY &) { return "get_global_size(1)"s; },
                [](const NullaryIntrinsicKind::GpuGlobalSizeZ &) { return "get_global_size(2)"s; },
                [](const NullaryIntrinsicKind::GpuGroupIdxX &) { return "get_group_id(0)"s; },
                [](const NullaryIntrinsicKind::GpuGroupIdxY &) { return "get_group_id(1)"s; },
                [](const NullaryIntrinsicKind::GpuGroupIdxZ &) { return "get_group_id(2)"s; },
                [](const NullaryIntrinsicKind::GpuGroupSizeX &) { return "get_num_groups(0)"s; },
                [](const NullaryIntrinsicKind::GpuGroupSizeY &) { return "get_num_groups(1)"s; },
                [](const NullaryIntrinsicKind::GpuGroupSizeZ &) { return "get_num_groups(2)"s; },
                [](const NullaryIntrinsicKind::GpuLocalIdxX &) { return "get_local_id(0)"s; },
                [](const NullaryIntrinsicKind::GpuLocalIdxY &) { return "get_local_id(1)"s; },
                [](const NullaryIntrinsicKind::GpuLocalIdxZ &) { return "get_local_id(2)"s; },
                [](const NullaryIntrinsicKind::GpuLocalSizeX &) { return "get_local_size(0)"s; },
                [](const NullaryIntrinsicKind::GpuLocalSizeY &) { return "get_local_size(1)"s; },
                [](const NullaryIntrinsicKind::GpuLocalSizeZ &) { return "get_local_size(2)"s; },
                [](const NullaryIntrinsicKind::GpuGroupBarrier &) { return "barrier(CLK_LOCAL_MEM_FENCE)"s; },
                [](const NullaryIntrinsicKind::GpuGroupFence &) { return "mem_fence(CLK_LOCAL_MEM_FENCE)"s; });
        }
      },
      [&](const Expr::UnaryIntrinsic &x) {
        auto op = variants::total(
            *x.kind,                                                 //
            [](const UnaryIntrinsicKind::Sin &) { return "sin"; },   //
            [](const UnaryIntrinsicKind::Cos &) { return "cos"; },   //
            [](const UnaryIntrinsicKind::Tan &) { return "tan"; },   //
            [](const UnaryIntrinsicKind::Asin &) { return "asin"; }, //
            [](const UnaryIntrinsicKind::Acos &) { return "acos"; }, //
            [](const UnaryIntrinsicKind::Atan &) { return "atan"; }, //
            [](const UnaryIntrinsicKind::Sinh &) { return "sinh"; }, //
            [](const UnaryIntrinsicKind::Cosh &) { return "cosh"; }, //
            [](const UnaryIntrinsicKind::Tanh &) { return "tanh"; }, //

            [](const UnaryIntrinsicKind::Signum &) { return "signum"; }, //
            [](const UnaryIntrinsicKind::Abs &) { return "abs"; },       //
            [](const UnaryIntrinsicKind::Round &) { return "round"; },   //
            [](const UnaryIntrinsicKind::Ceil &) { return "ceil"; },     //
            [](const UnaryIntrinsicKind::Floor &) { return "floor"; },   //
            [](const UnaryIntrinsicKind::Rint &) { return "rint"; },     //

            [](const UnaryIntrinsicKind::Sqrt &) { return "sqrt"; },   //
            [](const UnaryIntrinsicKind::Cbrt &) { return "cbrt"; },   //
            [](const UnaryIntrinsicKind::Exp &) { return "exp"; },     //
            [](const UnaryIntrinsicKind::Expm1 &) { return "expm1"; }, //
            [](const UnaryIntrinsicKind::Log &) { return "log"; },     //
            [](const UnaryIntrinsicKind::Log1p &) { return "log1p"; }, //
            [](const UnaryIntrinsicKind::Log10 &) { return "log10"; }, //
            [](const UnaryIntrinsicKind::BNot &) { return "~"; },      //
            [](const UnaryIntrinsicKind::Pos &) { return "+"; },       //
            [](const UnaryIntrinsicKind::Neg &) { return "-"; },       //
            [](const UnaryIntrinsicKind::LogicNot &x) { return "!"; });
        return std::string(op) + "(" + mkRef(x.lhs) + ")";
      },
      [&](const Expr::BinaryIntrinsic &x) {
        auto op = variants::total(
            *x.kind,                                              //
            [](const BinaryIntrinsicKind::Add &) { return "+"; }, //
            [](const BinaryIntrinsicKind::Sub &) { return "-"; }, //
            [](const BinaryIntrinsicKind::Mul &) { return "*"; }, //
            [](const BinaryIntrinsicKind::Div &) { return "/"; }, //
            [](const BinaryIntrinsicKind::Rem &) { return "%"; }, //

            [](const BinaryIntrinsicKind::Pow &) { return "**"; }, //

            [](const BinaryIntrinsicKind::Min &) { return "min"; }, //
            [](const BinaryIntrinsicKind::Max &) { return "max"; }, //

            [](const BinaryIntrinsicKind::Atan2 &) { return "atan2"; }, //
            [](const BinaryIntrinsicKind::Hypot &) { return "hypot"; }, //

            [](const BinaryIntrinsicKind::BAnd &) { return "&"; },   //
            [](const BinaryIntrinsicKind::BOr &) { return "|"; },    //
            [](const BinaryIntrinsicKind::BXor &) { return "^"; },   //
            [](const BinaryIntrinsicKind::BSL &) { return "<<"; },   //
            [](const BinaryIntrinsicKind::BSR &) { return ">>"; },   //
            [](const BinaryIntrinsicKind::BZSR &) { return ">>>"; }, //

            [](const BinaryIntrinsicKind::LogicEq &x) { return "=="; },  //
            [](const BinaryIntrinsicKind::LogicNeq &x) { return "!="; }, //
            [](const BinaryIntrinsicKind::LogicAnd &x) { return "&&"; }, //
            [](const BinaryIntrinsicKind::LogicOr &x) { return "||"; },  //
            [](const BinaryIntrinsicKind::LogicLte &x) { return "<="; }, //
            [](const BinaryIntrinsicKind::LogicGte &x) { return ">="; }, //
            [](const BinaryIntrinsicKind::LogicLt &x) { return "<"; },   //
            [](const BinaryIntrinsicKind::LogicGt &x) { return ">"; }    //
        );
        return mkRef(x.lhs) + " " + std::string(op) + " " + mkRef(x.rhs);
      },
      [&](const Expr::Cast &x) { return "((" + mkTpe(x.as) + ") " + mkRef(x.from) + ")"; },
      [&](const Expr::Alias &x) { return mkRef(x.ref); },                            //
      [&](const Expr::Invoke &x) { return "???"s; },                                 //
      [&](const Expr::Index &x) { return mkRef(x.lhs) + "[" + mkRef(x.idx) + "]"; }, //
      [&](const Expr::Alloc &x) { return "{/*" + to_string(x) + "*/}"; }             //
  );
}

std::string backend::CSource::mkStmt(const Stmt::Any &stmt) {
  return variants::total(
      *stmt, //
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
        if (holds<Type::Unit>(x.name.tpe)) {
          return mkExpr(*x.expr, x.name.symbol) + ";";
        }
        auto line =
            mkTpe(x.name.tpe) + " " + x.name.symbol + (x.expr ? (" = " + mkExpr(*x.expr, x.name.symbol)) : "") + ";";
        return line;
      },
      [&](const Stmt::Mut &x) {
        if (x.copy) {
          return "memcpy(" + //
                 mkRef(x.name) + ", " + mkExpr(x.expr, "?") + ", sizeof(" + mkTpe(tpe(x.expr)) + "));";
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
    auto decl = mkTpe(arg.named.tpe) + " " + arg.named.symbol;
    switch (dialect) {
      case Dialect::OpenCL1_1: {
        if (auto arr = get_opt<Type::Array>(arg.named.tpe); arr) {
          decl = variants::total(
              *arr->space,                                           //
              [&](TypeSpace::Global _) { return "global " + decl; }, //
              [&](TypeSpace::Local _) { return "local " + decl; });
        };
        break;
      }
      case Dialect::MSL1_0: {

        // Scalar:      device $T &$name      [[ buffer($i) ]]
        // Global:      device $T  $name      [[ buffer($i) ]]
        // Local:  threadgroup $T &$name [[ threadgroup($i) ]]
        // query:              $T &$name           [[ $type ]]

        auto idx = std::to_string(i);
        if (auto arr = get_opt<Type::Array>(arg.named.tpe); arr) {
          decl = variants::total(
              *arr->space,                                                                                  //
              [&](TypeSpace::Global _) { return "device " + decl + " [[ buffer(" + idx + ") ]]"; },         //
              [&](TypeSpace::Local _) { return "threadgroup " + decl + " [[ threadgroup(" + idx + ") ]]"; } //
          );
        } else {
          decl = "device &" + decl + " [[ buffer(" + idx + ") ]]";
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
        if (auto intr = get_opt<Expr::NullaryIntrinsic>(expr); intr) {
          if (holds<NullaryIntrinsicKind::GpuGlobalIdxX>(intr->kind) ||
              holds<NullaryIntrinsicKind::GpuGlobalIdxY>(intr->kind) ||
              holds<NullaryIntrinsicKind::GpuGlobalIdxZ>(intr->kind)) {
            iargs.emplace("get_global_id", "thread_position_in_grid");
          }
          if (holds<NullaryIntrinsicKind::GpuGlobalSizeX>(intr->kind) ||
              holds<NullaryIntrinsicKind::GpuGlobalSizeY>(intr->kind) ||
              holds<NullaryIntrinsicKind::GpuGlobalSizeZ>(intr->kind)) {
            iargs.emplace("get_global_size", "threads_per_grid");
          }
          if (holds<NullaryIntrinsicKind::GpuGroupIdxX>(intr->kind) ||
              holds<NullaryIntrinsicKind::GpuGroupIdxY>(intr->kind) ||
              holds<NullaryIntrinsicKind::GpuGroupIdxZ>(intr->kind)) {
            iargs.emplace("get_group_id", "threadgroup_position_in_grid");
          };
          if (holds<NullaryIntrinsicKind::GpuGroupSizeX>(intr->kind) ||
              holds<NullaryIntrinsicKind::GpuGroupSizeY>(intr->kind) ||
              holds<NullaryIntrinsicKind::GpuGroupSizeZ>(intr->kind)) {
            iargs.emplace("get_num_groups", "threadgroups_per_grid");
          };
          if (holds<NullaryIntrinsicKind::GpuLocalIdxX>(intr->kind) ||
              holds<NullaryIntrinsicKind::GpuLocalIdxY>(intr->kind) ||
              holds<NullaryIntrinsicKind::GpuLocalIdxZ>(intr->kind)) {
            iargs.emplace("get_local_id", "thread_position_in_threadgroup");
          };
          if (holds<NullaryIntrinsicKind::GpuLocalSizeX>(intr->kind) ||
              holds<NullaryIntrinsicKind::GpuLocalSizeY>(intr->kind) ||
              holds<NullaryIntrinsicKind::GpuLocalSizeZ>(intr->kind)) {
            iargs.emplace("get_local_size", "threads_per_threadgroup");
          };
        }
      };
      if (auto var = get_opt<Stmt::Var>(stmt); var) {
        if (var->expr) findIntrinsics(*var->expr);
      } else if (auto mut = get_opt<Stmt::Mut>(stmt); mut) {
        findIntrinsics(mut->expr);
      } else if (auto cond = get_opt<Stmt::Cond>(stmt); cond) {
        findIntrinsics(cond->cond);
      }
    }
    for (auto &[name, attr] : iargs)
      argExprs.push_back("uint3 ___" + name + "___ [[ " + attr + " ]]");
  }

  auto args = mk_string<std::string>(
      argExprs, [&](auto x) { return x; }, ", ");

  std::string fnPrefix;
  switch (dialect) {
    case Dialect::C11: fnPrefix = ""; break;
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

compiler::Compilation backend::CSource::compileProgram(const Program &program, const compiler::Opt &opt) {
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
    case Dialect::C11: lines.push_back("#include <stdint.h>\n#include <stdbool.h>"); break;
    default: break;
  }

  lines.push_back(structDefs);
  lines.push_back(mkFn(program.entry));
  for (auto &f : program.functions)
    lines.push_back(mkFn(f));

  auto def = mk_string<std::string>(lines, std::identity{}, "\n");

  std::cout << def << std::endl;
  std::vector<char> data(def.begin(), def.end());

  std::string dialectName;
  switch (dialect) {
    case Dialect::C11: dialectName = "c11"; break;
    case Dialect::OpenCL1_1: dialectName = "opencl1_1"; break;
    case Dialect::MSL1_0: dialectName = "msl1"; break;
    default: dialectName = "unknown";
  }

  return {data, {}, {{compiler::nowMs(), compiler::elapsedNs(start), "polyast_to_" + dialectName + "_c", def}}};
}
std::vector<compiler::Layout> backend::CSource::resolveLayouts(const std::vector<StructDef> &defs,
                                                               const compiler::Opt &opt) {
  return std::vector<compiler::Layout>();
}
