#include "polyast_repr.h"

#include "aspartame/all.hpp"
#include "fmt/core.h"

using namespace aspartame;
using namespace std::string_literals;

namespace polyregion::polyast {

std::string repr(const Sym &s) { return (s.fqn | mk_string("."s)); }

std::string repr(const SourcePosition &t) {
  return fmt::format("{}:{}{}", t.file, t.line, t.col ^ map([&](const int32_t &c) { return fmt::format(":{}", c); }) ^ get_or_else(""s));
}

std::string repr(const TypeSpace::Any &t) {
  return [&] {
    if (t.is<TypeSpace::Global>()) {
      return ""s;
    }
    if (t.is<TypeSpace::Local>()) {
      return "^Local"s;
    }
    if (t.is<TypeSpace::Private>()) {
      return "^Private"s;
    }
    if (t.is<TypeSpace::Constant>()) {
      return "^Constant"s;
    }

    throw std::logic_error(fmt::format("Unhandled match case for t (of type TypeSpace::Any) at {}:{})", __FILE__, __LINE__));
  }();
}

std::string repr(const Region::Any &r) {
  return [&] {
    if (auto _x = r.get<Region::Rooted>()) {
      return fmt::format("@{}", _x->root.symbol);
    }
    if (r.is<Region::Opaque>()) {
      return "@opaque"s;
    }

    throw std::logic_error(fmt::format("Unhandled match case for r (of type Region::Any) at {}:{})", __FILE__, __LINE__));
  }();
}

std::string repr(const TypeKind::Any &k) {
  return [&] {
    if (k.is<TypeKind::None>()) {
      return "None"s;
    }
    if (k.is<TypeKind::Ref>()) {
      return "Ref"s;
    }
    if (k.is<TypeKind::Integral>()) {
      return "Integral"s;
    }
    if (k.is<TypeKind::Fractional>()) {
      return "Fractional"s;
    }

    throw std::logic_error(fmt::format("Unhandled match case for k (of type TypeKind::Any) at {}:{})", __FILE__, __LINE__));
  }();
}

std::string repr(const PathStep::Any &s) {
  return [&] {
    if (auto _x = s.get<PathStep::Field>()) {
      return fmt::format(".{}", _x->name);
    }
    if (s.is<PathStep::Deref>()) {
      return "->*"s;
    }
    if (auto _x = s.get<PathStep::Index>()) {
      return fmt::format("[{}]", _x->idx);
    }
    if (auto _x = s.get<PathStep::IndexDyn>()) {
      return fmt::format("[{}]", repr(_x->idx));
    }

    throw std::logic_error(fmt::format("Unhandled match case for s (of type PathStep::Any) at {}:{})", __FILE__, __LINE__));
  }();
}

std::string repr(const Type::Any &t) {
  return [&] {
    if (t.is<Type::Float16>()) {
      return "F16"s;
    }
    if (t.is<Type::Float32>()) {
      return "F32"s;
    }
    if (t.is<Type::Float64>()) {
      return "F64"s;
    }
    if (t.is<Type::IntU8>()) {
      return "U8"s;
    }
    if (t.is<Type::IntU16>()) {
      return "U16"s;
    }
    if (t.is<Type::IntU32>()) {
      return "U32"s;
    }
    if (t.is<Type::IntU64>()) {
      return "U64"s;
    }
    if (t.is<Type::IntS8>()) {
      return "I8"s;
    }
    if (t.is<Type::IntS16>()) {
      return "I16"s;
    }
    if (t.is<Type::IntS32>()) {
      return "I32"s;
    }
    if (t.is<Type::IntS64>()) {
      return "I64"s;
    }
    if (t.is<Type::Nothing>()) {
      return "Nothing"s;
    }
    if (t.is<Type::Unit0>()) {
      return "Unit0"s;
    }
    if (t.is<Type::Bool1>()) {
      return "Bool1"s;
    }
    if (auto _x = t.get<Type::Struct>()) {
      return fmt::format("{}<{}>", repr(_x->name), (_x->args | map([&](const Type::Any &_v7_0) { return repr(_v7_0); }) | mk_string(","s)));
    }
    if (auto _x = t.get<Type::Ptr>()) {
      return fmt::format("{}*{}", repr(_x->comp), repr(_x->space));
    }
    if (auto _x = t.get<Type::Arr>()) {
      return fmt::format("{}[{}]{}", repr(_x->comp), _x->length, repr(_x->space));
    }
    if (auto _x = t.get<Type::Var>()) {
      return fmt::format("#{}", _x->name);
    }
    if (auto _x = t.get<Type::Exec>()) {
      return fmt::format("<{}>({}) => {}", (_x->tpeVars | mk_string(","s)),
                         (_x->args | map([&](const Type::Any &_v7_0) { return repr(_v7_0); }) | mk_string(","s)), repr(_x->rtn));
    }

    throw std::logic_error(fmt::format("Unhandled match case for t (of type Type::Any) at {}:{})", __FILE__, __LINE__));
  }();
}

std::string repr(const Named &n) { return fmt::format("{}", n.symbol); }

std::string repr(const Term::Any &t) {
  return [&] {
    if (auto _x = t.get<Term::Float16Const>()) {
      return fmt::format("f16({})", _x->value);
    }
    if (auto _x = t.get<Term::Float32Const>()) {
      return fmt::format("f32({})", _x->value);
    }
    if (auto _x = t.get<Term::Float64Const>()) {
      return fmt::format("f64({})", _x->value);
    }
    if (auto _x = t.get<Term::IntU8Const>()) {
      return fmt::format("u8({})", _x->value);
    }
    if (auto _x = t.get<Term::IntU16Const>()) {
      return fmt::format("u16({})", _x->value);
    }
    if (auto _x = t.get<Term::IntU32Const>()) {
      return fmt::format("u32({})", _x->value);
    }
    if (auto _x = t.get<Term::IntU64Const>()) {
      return fmt::format("u64({})", _x->value);
    }
    if (auto _x = t.get<Term::IntS8Const>()) {
      return fmt::format("i8({})", _x->value);
    }
    if (auto _x = t.get<Term::IntS16Const>()) {
      return fmt::format("i16({})", _x->value);
    }
    if (auto _x = t.get<Term::IntS32Const>()) {
      return fmt::format("i32({})", _x->value);
    }
    if (auto _x = t.get<Term::IntS64Const>()) {
      return fmt::format("i64({})", _x->value);
    }
    if (t.is<Term::Unit0Const>()) {
      return "unit0(())"s;
    }
    if (auto _x = t.get<Term::Bool1Const>()) {
      return fmt::format("bool1({})", _x->value);
    }
    if (auto _x = t.get<Term::NullPtrConst>()) {
      return fmt::format("nullptr[{}, {}{}]", repr(_x->comp), repr(_x->space), repr(_x->region));
    }
    if (auto _x = t.get<Term::StringConst>()) {
      return fmt::format("str({})", _x->value);
    }
    if (auto _x = t.get<Term::Poison>()) {
      return fmt::format("__poison__ /* poison of type {} */", repr(_x->t));
    }
    if (auto _x = t.get<Term::Select>()) {
      return fmt::format("{}: {}{}", _x->root.symbol, repr(_x->root.tpe),
                         (_x->steps | map([&](const PathStep::Any &_v7_0) { return repr(_v7_0); }) | mk_string(""s)));
    }

    throw std::logic_error(fmt::format("Unhandled match case for t (of type Term::Any) at {}:{})", __FILE__, __LINE__));
  }();
}

std::string repr(const Expr::Any &e) {
  return [&] {
    if (auto _x = e.get<Expr::Alias>()) {
      return repr(_x->ref);
    }
    if (auto _x = e.get<Expr::SpecOp>()) {
      return [&] {
        if (auto _z = _x->op.get<Spec::Assert>()) {
          return fmt::format("'assert({}, {})", repr(_z->code), repr(_z->message));
        }
        if (_x->op.is<Spec::GpuBarrierGlobal>()) {
          return "'gpuBarrierGlobal"s;
        }
        if (_x->op.is<Spec::GpuBarrierLocal>()) {
          return "'gpuBarrierLocal"s;
        }
        if (_x->op.is<Spec::GpuBarrierAll>()) {
          return "'gpuBarrierAll"s;
        }
        if (_x->op.is<Spec::GpuFenceGlobal>()) {
          return "'gpuFenceGlobal"s;
        }
        if (_x->op.is<Spec::GpuFenceLocal>()) {
          return "'gpuFenceLocal"s;
        }
        if (_x->op.is<Spec::GpuFenceAll>()) {
          return "'gpuFenceAll"s;
        }
        if (auto _z = _x->op.get<Spec::GpuGlobalIdx>()) {
          return fmt::format("'gpuGlobalIdx({})", repr(_z->dim));
        }
        if (auto _z = _x->op.get<Spec::GpuGlobalSize>()) {
          return fmt::format("'gpuGlobalSize({})", repr(_z->dim));
        }
        if (auto _z = _x->op.get<Spec::GpuGroupIdx>()) {
          return fmt::format("'gpuGroupIdx({})", repr(_z->dim));
        }
        if (auto _z = _x->op.get<Spec::GpuGroupSize>()) {
          return fmt::format("'gpuGroupSize({})", repr(_z->dim));
        }
        if (auto _z = _x->op.get<Spec::GpuLocalIdx>()) {
          return fmt::format("'gpuLocalIdx({})", repr(_z->dim));
        }
        if (auto _z = _x->op.get<Spec::GpuLocalSize>()) {
          return fmt::format("'gpuLocalSize({})", repr(_z->dim));
        }

        throw std::logic_error(fmt::format("Unhandled match case for _x->op (of type Spec::Any) at {}:{})", __FILE__, __LINE__));
      }();
    }
    if (auto _x = e.get<Expr::MathOp>()) {
      return [&] {
        if (auto _z = _x->op.get<Math::Abs>()) {
          return fmt::format("'abs({})", repr(_z->x));
        }
        if (auto _z = _x->op.get<Math::Sin>()) {
          return fmt::format("'sin({})", repr(_z->x));
        }
        if (auto _z = _x->op.get<Math::Cos>()) {
          return fmt::format("'cos({})", repr(_z->x));
        }
        if (auto _z = _x->op.get<Math::Tan>()) {
          return fmt::format("'tan({})", repr(_z->x));
        }
        if (auto _z = _x->op.get<Math::Asin>()) {
          return fmt::format("'asin({})", repr(_z->x));
        }
        if (auto _z = _x->op.get<Math::Acos>()) {
          return fmt::format("'acos({})", repr(_z->x));
        }
        if (auto _z = _x->op.get<Math::Atan>()) {
          return fmt::format("'atan({})", repr(_z->x));
        }
        if (auto _z = _x->op.get<Math::Sinh>()) {
          return fmt::format("'sinh({})", repr(_z->x));
        }
        if (auto _z = _x->op.get<Math::Cosh>()) {
          return fmt::format("'cosh({})", repr(_z->x));
        }
        if (auto _z = _x->op.get<Math::Tanh>()) {
          return fmt::format("'tanh({})", repr(_z->x));
        }
        if (auto _z = _x->op.get<Math::Signum>()) {
          return fmt::format("'signum({})", repr(_z->x));
        }
        if (auto _z = _x->op.get<Math::Round>()) {
          return fmt::format("'round({})", repr(_z->x));
        }
        if (auto _z = _x->op.get<Math::Ceil>()) {
          return fmt::format("'ceil({})", repr(_z->x));
        }
        if (auto _z = _x->op.get<Math::Floor>()) {
          return fmt::format("'floor({})", repr(_z->x));
        }
        if (auto _z = _x->op.get<Math::Rint>()) {
          return fmt::format("'rint({})", repr(_z->x));
        }
        if (auto _z = _x->op.get<Math::Sqrt>()) {
          return fmt::format("'sqrt({})", repr(_z->x));
        }
        if (auto _z = _x->op.get<Math::Cbrt>()) {
          return fmt::format("'cbrt({})", repr(_z->x));
        }
        if (auto _z = _x->op.get<Math::Exp>()) {
          return fmt::format("'exp({})", repr(_z->x));
        }
        if (auto _z = _x->op.get<Math::Expm1>()) {
          return fmt::format("'expm1({})", repr(_z->x));
        }
        if (auto _z = _x->op.get<Math::Log>()) {
          return fmt::format("'log({})", repr(_z->x));
        }
        if (auto _z = _x->op.get<Math::Log1p>()) {
          return fmt::format("'log1p({})", repr(_z->x));
        }
        if (auto _z = _x->op.get<Math::Log10>()) {
          return fmt::format("'log10({})", repr(_z->x));
        }
        if (auto _z = _x->op.get<Math::Pow>()) {
          return fmt::format("'pow({}, {})", repr(_z->x), repr(_z->y));
        }
        if (auto _z = _x->op.get<Math::Atan2>()) {
          return fmt::format("'atan2({}, {})", repr(_z->x), repr(_z->y));
        }
        if (auto _z = _x->op.get<Math::Hypot>()) {
          return fmt::format("'hypot({}, {})", repr(_z->x), repr(_z->y));
        }

        throw std::logic_error(fmt::format("Unhandled match case for _x->op (of type Math::Any) at {}:{})", __FILE__, __LINE__));
      }();
    }
    if (auto _x = e.get<Expr::IntrOp>()) {
      return [&] {
        if (auto _z = _x->op.get<Intr::BNot>()) {
          return fmt::format("('~{})", repr(_z->x));
        }
        if (auto _z = _x->op.get<Intr::LogicNot>()) {
          return fmt::format("('!{})", repr(_z->x));
        }
        if (auto _z = _x->op.get<Intr::Pos>()) {
          return fmt::format("('+{})", repr(_z->x));
        }
        if (auto _z = _x->op.get<Intr::Neg>()) {
          return fmt::format("('-{})", repr(_z->x));
        }
        if (auto _z = _x->op.get<Intr::Add>()) {
          return fmt::format("({} '+ {})", repr(_z->x), repr(_z->y));
        }
        if (auto _z = _x->op.get<Intr::Sub>()) {
          return fmt::format("({} '- {})", repr(_z->x), repr(_z->y));
        }
        if (auto _z = _x->op.get<Intr::Mul>()) {
          return fmt::format("({} '* {})", repr(_z->x), repr(_z->y));
        }
        if (auto _z = _x->op.get<Intr::Div>()) {
          return fmt::format("({} '/ {})", repr(_z->x), repr(_z->y));
        }
        if (auto _z = _x->op.get<Intr::Rem>()) {
          return fmt::format("({} '% {})", repr(_z->x), repr(_z->y));
        }
        if (auto _z = _x->op.get<Intr::Min>()) {
          return fmt::format("'min({}, {})", repr(_z->x), repr(_z->y));
        }
        if (auto _z = _x->op.get<Intr::Max>()) {
          return fmt::format("'max({}, {})", repr(_z->x), repr(_z->y));
        }
        if (auto _z = _x->op.get<Intr::BAnd>()) {
          return fmt::format("({} '& {})", repr(_z->x), repr(_z->y));
        }
        if (auto _z = _x->op.get<Intr::BOr>()) {
          return fmt::format("({} '| {})", repr(_z->x), repr(_z->y));
        }
        if (auto _z = _x->op.get<Intr::BXor>()) {
          return fmt::format("({} '^ {})", repr(_z->x), repr(_z->y));
        }
        if (auto _z = _x->op.get<Intr::BSL>()) {
          return fmt::format("({} '<< {})", repr(_z->x), repr(_z->y));
        }
        if (auto _z = _x->op.get<Intr::BSR>()) {
          return fmt::format("({} '>> {})", repr(_z->x), repr(_z->y));
        }
        if (auto _z = _x->op.get<Intr::BZSR>()) {
          return fmt::format("({} '>>> {})", repr(_z->x), repr(_z->y));
        }
        if (auto _z = _x->op.get<Intr::LogicAnd>()) {
          return fmt::format("({} '&& {})", repr(_z->x), repr(_z->y));
        }
        if (auto _z = _x->op.get<Intr::LogicOr>()) {
          return fmt::format("({} '|| {})", repr(_z->x), repr(_z->y));
        }
        if (auto _z = _x->op.get<Intr::LogicEq>()) {
          return fmt::format("({} '== {})", repr(_z->x), repr(_z->y));
        }
        if (auto _z = _x->op.get<Intr::LogicNeq>()) {
          return fmt::format("({} '!= {})", repr(_z->x), repr(_z->y));
        }
        if (auto _z = _x->op.get<Intr::LogicLte>()) {
          return fmt::format("({} '<= {})", repr(_z->x), repr(_z->y));
        }
        if (auto _z = _x->op.get<Intr::LogicGte>()) {
          return fmt::format("({} '>= {})", repr(_z->x), repr(_z->y));
        }
        if (auto _z = _x->op.get<Intr::LogicLt>()) {
          return fmt::format("({} '< {})", repr(_z->x), repr(_z->y));
        }
        if (auto _z = _x->op.get<Intr::LogicGt>()) {
          return fmt::format("({} '> {})", repr(_z->x), repr(_z->y));
        }

        throw std::logic_error(fmt::format("Unhandled match case for _x->op (of type Intr::Any) at {}:{})", __FILE__, __LINE__));
      }();
    }
    if (auto _x = e.get<Expr::Cast>()) {
      return fmt::format("({}).to[{}]", repr(_x->from), repr(_x->as));
    }
    if (auto _x = e.get<Expr::Index>()) {
      return fmt::format("({}).index[{}]({})", repr(_x->lhs), repr(_x->comp), repr(_x->idx));
    }
    if (auto _x = e.get<Expr::RefTo>()) {
      return fmt::format("({}).refTo[{}, {}{}]({})", repr(_x->lhs), repr(_x->comp), repr(_x->space), repr(_x->region),
                         _x->idx ^ map([&](const Term::Any &_v7_0) { return repr(_v7_0); }) ^ get_or_else(""s));
    }
    if (auto _x = e.get<Expr::Alloc>()) {
      return fmt::format("alloc[{}, {}{}]({})", repr(_x->comp), repr(_x->space), repr(_x->region), repr(_x->size));
    }
    if (auto _x = e.get<Expr::Invoke>()) {
      return fmt::format("{}{}<{}>({}): {}",
                         _x->receiver ^ map([&](const Term::Any &r) { return fmt::format("{}.", repr(r)); }) ^ get_or_else(""s),
                         repr(_x->name), (_x->tpeArgs | map([&](const Type::Any &_v7_0) { return repr(_v7_0); }) | mk_string(","s)),
                         (_x->args | map([&](const Term::Any &_v7_0) { return repr(_v7_0); }) | mk_string(", "s)), repr(_x->rtn));
    }
    if (auto _x = e.get<Expr::ForeignCall>()) {
      return fmt::format("{}({}): {}", _x->name, (_x->args | map([&](const Term::Any &_v7_0) { return repr(_v7_0); }) | mk_string(", "s)),
                         repr(_x->rtn));
    }
    if (auto _x = e.get<Expr::OffsetOf>()) {
      return fmt::format("offsetof({}, {})", repr(_x->structTpe), _x->field);
    }
    if (auto _x = e.get<Expr::SizeOf>()) {
      return fmt::format("sizeof({})", repr(_x->forTpe));
    }

    throw std::logic_error(fmt::format("Unhandled match case for e (of type Expr::Any) at {}:{})", __FILE__, __LINE__));
  }();
}

std::string repr(const Stmt::Any &stmt) {
  return [&] {
    if (auto _x = stmt.get<Stmt::Var>()) {
      return fmt::format("{} {}: {} = {}", (_x->isMutable ? "var"s : "val"s), _x->name.symbol, repr(_x->name.tpe),
                         _x->expr ^ map([&](const Expr::Any &_v7_0) { return repr(_v7_0); }) ^ get_or_else("_"s));
    }
    if (auto _x = stmt.get<Stmt::Mut>()) {
      return fmt::format("{} = {}", repr(_x->name), repr(_x->expr));
    }
    if (auto _x = stmt.get<Stmt::Update>()) {
      return fmt::format("({}).update({}) = {}", repr(_x->lhs), repr(_x->idx), repr(_x->value));
    }
    if (auto _x = stmt.get<Stmt::While>()) {
      return fmt::format("while({}){}\n{}\n{}", repr(_x->cond), "{"s,
                         (_x->body | map([&](const Stmt::Any &_v8_0) { return repr(_v8_0); }) | mk_string("\n"s)) ^ indent(2), "}"s);
    }
    if (auto _x = stmt.get<Stmt::ForRange>()) {
      return fmt::format("for({}: {} = {}; < {}; += {}){}\n{}\n{}", _x->induction.symbol, repr(_x->induction.tpe), repr(_x->lbIncl),
                         repr(_x->ubExcl), repr(_x->step), "{"s,
                         (_x->body | map([&](const Stmt::Any &_v8_0) { return repr(_v8_0); }) | mk_string("\n"s)) ^ indent(2), "}"s);
    }
    if (stmt.is<Stmt::Break>()) {
      return "break;"s;
    }
    if (stmt.is<Stmt::Cont>()) {
      return "continue;"s;
    }
    if (auto _x = stmt.get<Stmt::Return>()) {
      return fmt::format("return {}", repr(_x->value));
    }
    if (auto _x = stmt.get<Stmt::Cond>()) {
      return fmt::format(
          "if({}) {}\n{}\n{}{}", repr(_x->cond), "{"s,
          (_x->trueBr | map([&](const Stmt::Any &_v8_0) { return repr(_v8_0); }) | mk_string("\n"s)) ^ indent(2), "}"s,
          (_x->falseBr.empty()
               ? ""s
               : fmt::format(" else {}\n{}\n{}", "{"s,
                             (_x->falseBr | map([&](const Stmt::Any &_v10_0) { return repr(_v10_0); }) | mk_string("\n"s)) ^ indent(2),
                             "}"s)));
    }
    if (auto _x = stmt.get<Stmt::Annotated>()) {
      return fmt::format("{}{}{}", repr(_x->inner),
                         _x->pos ^ map([&](const SourcePosition &p) { return fmt::format(" /* {} */", repr(p)); }) ^ get_or_else(""s),
                         _x->comment ^ map([&](const std::string &c) { return fmt::format(" /* {} */", c); }) ^ get_or_else(""s));
    }

    throw std::logic_error(fmt::format("Unhandled match case for stmt (of type Stmt::Any) at {}:{})", __FILE__, __LINE__));
  }();
}

std::string repr(const Arg &a) {
  return fmt::format("{}: {}{}", a.named.symbol, repr(a.named.tpe),
                     a.pos ^ map([&](const SourcePosition &s) { return fmt::format("/* {} */", repr(s)); }) ^ get_or_else(""s));
}

std::string repr(const FunctionVisibility::Any &v) {
  return [&] {
    if (v.is<FunctionVisibility::Internal>()) {
      return "Internal"s;
    }
    if (v.is<FunctionVisibility::Exported>()) {
      return "Exported"s;
    }

    throw std::logic_error(fmt::format("Unhandled match case for v (of type FunctionVisibility::Any) at {}:{})", __FILE__, __LINE__));
  }();
}

std::string repr(const FunctionFpMode::Any &m) {
  return [&] {
    if (m.is<FunctionFpMode::Relaxed>()) {
      return "FPRelaxed"s;
    }
    if (m.is<FunctionFpMode::Strict>()) {
      return "FPStrict"s;
    }

    throw std::logic_error(fmt::format("Unhandled match case for m (of type FunctionFpMode::Any) at {}:{})", __FILE__, __LINE__));
  }();
}

std::string repr(const Signature &f) {
  return fmt::format("def {}{}<{}>({}): {} /* mod={} term={} */",
                     f.receiver ^ map([&](const Type::Any &r) { return fmt::format("{}.", repr(r)); }) ^ get_or_else(""s), repr(f.name),
                     (f.tpeVars | mk_string(","s)), (f.args | map([&](const Type::Any &_v5_0) { return repr(_v5_0); }) | mk_string(", "s)),
                     repr(f.rtn), (f.moduleCaptures | map([&](const Type::Any &_v5_0) { return repr(_v5_0); }) | mk_string(","s)),
                     (f.termCaptures | map([&](const Type::Any &_v5_0) { return repr(_v5_0); }) | mk_string(","s)));
}

std::string repr(const Function &f) {
  return fmt::format(
      "def {}{}<{}>({}): {} /* vis={} fp={} entry={} mod={} term={} */ {}\n{}\n{}",
      f.receiver ^ map([&](const Arg &r) { return fmt::format("{}.", repr(r)); }) ^ get_or_else(""s), repr(f.name),
      (f.tpeVars | mk_string(","s)),
      (f.args | map([&](const Arg &a) { return fmt::format("{}: {}", a.named.symbol, repr(a.named.tpe)); }) | mk_string(", "s)),
      repr(f.rtn), repr(f.visibility), repr(f.fpMode), f.isEntry,
      (f.moduleCaptures | map([&](const Arg &_v5_0) { return repr(_v5_0); }) | mk_string(","s)),
      (f.termCaptures | map([&](const Arg &_v5_0) { return repr(_v5_0); }) | mk_string(","s)), "{"s,
      (f.body | map([&](const Stmt::Any &_v6_0) { return repr(_v6_0); }) | mk_string("\n"s)) ^ indent(2), "}"s);
}

std::string repr(const StructDef &s) {
  return fmt::format("class {}<{}>({}) <: {}", repr(s.name), (s.tpeVars | mk_string(","s)),
                     (s.members | map([&](const Named &m) { return fmt::format("{}: {}", m.symbol, repr(m.tpe)); }) | mk_string(", "s)),
                     (s.parents | map([&](const Type::Struct &_v5_0) { return repr(_v5_0); }) | mk_string(", "s)));
}

std::string repr(const Program &s) {
  return fmt::format("{}\n{}\n{}", (s.defs | map([&](const StructDef &_v5_0) { return repr(_v5_0); }) | mk_string("\n"s)), repr(s.entry),
                     (s.functions | map([&](const Function &_v5_0) { return repr(_v5_0); }) | mk_string("\n"s)));
}

std::string repr(const StructLayout &l) {
  return fmt::format("StructLayout[{}, sizeInBytes={}, align={}]{}\n{}\n{}", l.name, l.sizeInBytes, l.alignment, "{"s,
                     (l.members | map([&](const StructLayoutMember &m) {
                        return fmt::format("{}: {} (+{},{})", m.name.symbol, repr(m.name.tpe), m.offsetInBytes, m.sizeInBytes);
                      }) |
                      mk_string("\n"s)) ^
                         indent(2),
                     "}"s);
}

} // namespace polyregion::polyast
