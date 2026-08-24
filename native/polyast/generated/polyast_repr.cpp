#include "polyast_repr.h"

#include "aspartame/all.hpp"
#include "fmt/core.h"

using namespace aspartame;
using namespace std::string_literals;

namespace polyregion::polyast {

std::string repr(const Sym &s) { return fqcn(s); }

std::string repr(const SourcePosition &t) {
  return fmt::format("{}:{}{}", t.file, t.line, t.col ^ map([&](const int32_t &c) { return fmt::format(":{}", c); }) ^ get_or_else(""s));
}

std::string repr(const TypeSpace::Any &t) { return canonicalName(t); }

std::string repr(const AtomicOp::Any &o) {
  return [&] {
    if (o.is<AtomicOp::Xchg>()) {
      return "Xchg"s;
    }
    if (o.is<AtomicOp::Add>()) {
      return "Add"s;
    }
    if (o.is<AtomicOp::Sub>()) {
      return "Sub"s;
    }
    if (o.is<AtomicOp::And>()) {
      return "And"s;
    }
    if (o.is<AtomicOp::Or>()) {
      return "Or"s;
    }
    if (o.is<AtomicOp::Xor>()) {
      return "Xor"s;
    }
    if (o.is<AtomicOp::Min>()) {
      return "Min"s;
    }
    if (o.is<AtomicOp::Max>()) {
      return "Max"s;
    }

    throw std::logic_error(fmt::format("Unhandled match case for o (of type AtomicOp::Any) at {}:{})", __FILE__, __LINE__));
  }();
}

std::string repr(const MemScope::Any &s) {
  return [&] {
    if (s.is<MemScope::Subgroup>()) {
      return "subgroup"s;
    }
    if (s.is<MemScope::Workgroup>()) {
      return "workgroup"s;
    }
    if (s.is<MemScope::Device>()) {
      return "device"s;
    }
    if (s.is<MemScope::System>()) {
      return "system"s;
    }

    throw std::logic_error(fmt::format("Unhandled match case for s (of type MemScope::Any) at {}:{})", __FILE__, __LINE__));
  }();
}

std::string repr(const MemOrder::Any &o) {
  return [&] {
    if (o.is<MemOrder::Relaxed>()) {
      return "relaxed"s;
    }
    if (o.is<MemOrder::Acquire>()) {
      return "acquire"s;
    }
    if (o.is<MemOrder::Release>()) {
      return "release"s;
    }
    if (o.is<MemOrder::AcqRel>()) {
      return "acqrel"s;
    }
    if (o.is<MemOrder::SeqCst>()) {
      return "seqcst"s;
    }

    throw std::logic_error(fmt::format("Unhandled match case for o (of type MemOrder::Any) at {}:{})", __FILE__, __LINE__));
  }();
}

std::string repr(const Direction::Any &d) {
  return [&] {
    if (d.is<Direction::LocalToRemote>()) {
      return "localToRemote"s;
    }
    if (d.is<Direction::RemoteToLocal>()) {
      return "remoteToLocal"s;
    }
    if (d.is<Direction::RemoteToRemote>()) {
      return "remoteToRemote"s;
    }

    throw std::logic_error(fmt::format("Unhandled match case for d (of type Direction::Any) at {}:{})", __FILE__, __LINE__));
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

std::string repr(const Type::Any &t) { return canonicalName(t); }

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
        if (_x->op.is<Spec::GpuLaneIdx>()) {
          return "'gpuLaneIdx"s;
        }
        if (_x->op.is<Spec::GpuSubgroupSize>()) {
          return "'gpuSubgroupSize"s;
        }
        if (auto _z = _x->op.get<Spec::GpuShuffleDown>()) {
          return fmt::format("'gpuShuffleDown({}, {}, {}, {})", repr(_z->value), repr(_z->delta), repr(_z->width), repr(_z->mask));
        }
        if (auto _z = _x->op.get<Spec::GpuShuffleUp>()) {
          return fmt::format("'gpuShuffleUp({}, {}, {}, {})", repr(_z->value), repr(_z->delta), repr(_z->width), repr(_z->mask));
        }
        if (auto _z = _x->op.get<Spec::GpuShuffleIdx>()) {
          return fmt::format("'gpuShuffleIdx({}, {}, {}, {})", repr(_z->value), repr(_z->srcLane), repr(_z->width), repr(_z->mask));
        }
        if (auto _z = _x->op.get<Spec::GpuShuffleXor>()) {
          return fmt::format("'gpuShuffleXor({}, {}, {}, {})", repr(_z->value), repr(_z->laneMask), repr(_z->width), repr(_z->mask));
        }
        if (auto _z = _x->op.get<Spec::GpuSubgroupBarrier>()) {
          return fmt::format("'gpuSubgroupBarrier({})", repr(_z->mask));
        }
        if (auto _z = _x->op.get<Spec::GpuBallot>()) {
          return fmt::format("'gpuBallot({}, {})", repr(_z->mask), repr(_z->pred));
        }
        if (auto _z = _x->op.get<Spec::GpuVoteAny>()) {
          return fmt::format("'gpuVoteAny({}, {})", repr(_z->mask), repr(_z->pred));
        }
        if (auto _z = _x->op.get<Spec::GpuVoteAll>()) {
          return fmt::format("'gpuVoteAll({}, {})", repr(_z->mask), repr(_z->pred));
        }
        if (auto _z = _x->op.get<Spec::GpuAtomicRMW>()) {
          return fmt::format("'gpuAtomic{}({}, {}, {}, {})", repr(_z->op), repr(_z->ptr), repr(_z->value), repr(_z->scope),
                             repr(_z->order));
        }
        if (auto _z = _x->op.get<Spec::GpuAtomicCAS>()) {
          return fmt::format("'gpuAtomicCAS({}, {}, {}, {}, {})", repr(_z->ptr), repr(_z->expected), repr(_z->desired), repr(_z->scope),
                             repr(_z->order));
        }
        if (auto _z = _x->op.get<Spec::GpuGroupReduce>()) {
          return fmt::format("'gpuGroupReduce{}({})", repr(_z->op), repr(_z->value));
        }
        if (auto _z = _x->op.get<Spec::GpuGroupInclusiveScan>()) {
          return fmt::format("'gpuGroupInclusiveScan{}({})", repr(_z->op), repr(_z->value));
        }
        if (auto _z = _x->op.get<Spec::GpuGroupExclusiveScan>()) {
          return fmt::format("'gpuGroupExclusiveScan{}({})", repr(_z->op), repr(_z->value));
        }
        if (auto _z = _x->op.get<Spec::RemoteLaunch>()) {
          return fmt::format("'remoteLaunch({}, {}[{}], <{}, {}, {}>, <{}, {}, {}>, {}, [{}])", repr(_z->context), repr(_z->kernel),
                             (_z->tpeArgs | map([&](const Type::Any &_v9_0) { return repr(_v9_0); }) | mk_string(", "s)), repr(_z->gridX),
                             repr(_z->gridY), repr(_z->gridZ), repr(_z->blockX), repr(_z->blockY), repr(_z->blockZ), repr(_z->shmem),
                             (_z->args | map([&](const Term::Any &_v9_0) { return repr(_v9_0); }) | mk_string(", "s)));
        }
        if (auto _z = _x->op.get<Spec::RemoteAlloc>()) {
          return fmt::format("'remoteAlloc({}, {})", repr(_z->context), repr(_z->bytes));
        }
        if (auto _z = _x->op.get<Spec::RemoteFree>()) {
          return fmt::format("'remoteFree({}, {})", repr(_z->context), repr(_z->ptr));
        }
        if (auto _z = _x->op.get<Spec::RemoteMemcpy>()) {
          return fmt::format("'remoteMemcpy({}, {}, {}, {}, {})", repr(_z->context), repr(_z->dst), repr(_z->src), repr(_z->bytes),
                             repr(_z->direction));
        }
        if (auto _z = _x->op.get<Spec::RemoteSync>()) {
          return fmt::format("'remoteSync({})", repr(_z->context));
        }
        if (auto _z = _x->op.get<Spec::GpuVolatileLoad>()) {
          return fmt::format("'gpuVolatileLoad({})", repr(_z->ptr));
        }
        if (auto _z = _x->op.get<Spec::GpuVolatileStore>()) {
          return fmt::format("'gpuVolatileStore({}, {})", repr(_z->ptr), repr(_z->value));
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
                         repr(_x->callee), (_x->tpeArgs | map([&](const Type::Any &_v7_0) { return repr(_v7_0); }) | mk_string(","s)),
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
    if (auto _x = stmt.get<Stmt::Raise>()) {
      return fmt::format("raise {} /* {} */", repr(_x->value), _x->exceptionKind.sourceName);
    }
    if (stmt.is<Stmt::Rethrow>()) {
      return "rethrow"s;
    }
    if (auto _x = stmt.get<Stmt::Try>()) {
      return fmt::format(
          "try {}\n{}\n{}{}{}", "{"s, (_x->body | map([&](const Stmt::Any &_v8_0) { return repr(_v8_0); }) | mk_string("\n"s)) ^ indent(2),
          "}"s, (_x->handlers | map([&](const Handler &_v7_0) { return repr(_v7_0); }) | mk_string(""s)),
          (_x->fin.empty()
               ? ""s
               : fmt::format(" finally {}\n{}\n{}", "{"s,
                             (_x->fin | map([&](const Stmt::Any &_v10_0) { return repr(_v10_0); }) | mk_string("\n"s)) ^ indent(2), "}"s)));
    }

    throw std::logic_error(fmt::format("Unhandled match case for stmt (of type Stmt::Any) at {}:{})", __FILE__, __LINE__));
  }();
}

std::string repr(const Handler &h) {
  return fmt::format(" catch ({}{}{}) {}\n{}\n{}",
                     h.binder ^ map([&](const Named &b) { return fmt::format("{}: ", b.symbol); }) ^ get_or_else(""s),
                     h.caught ^ map([&](const ExceptionKind &_v5_0) { return repr(_v5_0.tpe); }) ^ get_or_else("_"s),
                     h.caught ^ map([&](const ExceptionKind &x) { return fmt::format(" /* {} */", x.sourceName); }) ^ get_or_else(""s),
                     "{"s, (h.body | map([&](const Stmt::Any &_v6_0) { return repr(_v6_0); }) | mk_string("\n"s)) ^ indent(2), "}"s);
}

std::string repr(const Arg &a) {
  return fmt::format("{}: {}{}{}", a.named.symbol, repr(a.named.tpe), a.boundary ^ map([&](const ArgBoundary &b) {
                                                                        return fmt::format(" /* {} {} */", repr(b.access), repr(b.extent));
                                                                      }) ^ get_or_else(""s),
                     a.pos ^ map([&](const SourcePosition &s) { return fmt::format(" /* {} */", repr(s)); }) ^ get_or_else(""s));
}

std::string repr(const ArgAccess::Any &a) {
  return [&] {
    if (a.is<ArgAccess::Read>()) {
      return "read"s;
    }
    if (a.is<ArgAccess::Write>()) {
      return "write"s;
    }
    if (a.is<ArgAccess::ReadWrite>()) {
      return "read-write"s;
    }

    throw std::logic_error(fmt::format("Unhandled match case for a (of type ArgAccess::Any) at {}:{})", __FILE__, __LINE__));
  }();
}

std::string repr(const ArgSizeExpr::Any &s) {
  return [&] {
    if (auto _x = s.get<ArgSizeExpr::Param>()) {
      return fmt::format("arg[{}]", _x->index);
    }
    if (auto _x = s.get<ArgSizeExpr::Const>()) {
      return std::to_string(_x->value);
    }
    if (auto _x = s.get<ArgSizeExpr::Add>()) {
      return fmt::format("({} + {})", repr(_x->lhs), repr(_x->rhs));
    }
    if (auto _x = s.get<ArgSizeExpr::Mul>()) {
      return fmt::format("({} * {})", repr(_x->lhs), repr(_x->rhs));
    }
    if (auto _x = s.get<ArgSizeExpr::Min>()) {
      return fmt::format("min({}, {})", repr(_x->lhs), repr(_x->rhs));
    }

    throw std::logic_error(fmt::format("Unhandled match case for s (of type ArgSizeExpr::Any) at {}:{})", __FILE__, __LINE__));
  }();
}

std::string repr(const ArgExtent::Any &e) {
  return [&] {
    if (auto _x = e.get<ArgExtent::Elements>()) {
      return fmt::format("elements({})", repr(_x->size));
    }
    if (auto _x = e.get<ArgExtent::Bytes>()) {
      return fmt::format("bytes({})", repr(_x->size));
    }

    throw std::logic_error(fmt::format("Unhandled match case for e (of type ArgExtent::Any) at {}:{})", __FILE__, __LINE__));
  }();
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

std::string repr(const FunctionAffinity::Any &a) {
  return [&] {
    if (a.is<FunctionAffinity::Offload>()) {
      return "Offload"s;
    }
    if (a.is<FunctionAffinity::Host>()) {
      return "Host"s;
    }

    throw std::logic_error(fmt::format("Unhandled match case for a (of type FunctionAffinity::Any) at {}:{})", __FILE__, __LINE__));
  }();
}

std::string repr(const CallConvention::Any &c) {
  return [&] {
    if (c.is<CallConvention::RegularCall>()) {
      return "RegularCall"s;
    }
    if (c.is<CallConvention::OffloadEntry>()) {
      return "OffloadEntry"s;
    }

    throw std::logic_error(fmt::format("Unhandled match case for c (of type CallConvention::Any) at {}:{})", __FILE__, __LINE__));
  }();
}

std::string repr(const Signature &f) {
  return fmt::format("def {}{}<{}>({}): {} /* mod={} term={} */",
                     f.receiver ^ map([&](const Type::Any &r) { return fmt::format("{}.", repr(r)); }) ^ get_or_else(""s), repr(f.name),
                     (f.tpeVars | map([&](const Type::Var &_v5_0) { return repr(_v5_0); }) | mk_string(","s)),
                     (f.args | map([&](const Type::Any &_v5_0) { return repr(_v5_0); }) | mk_string(", "s)), repr(f.rtn),
                     (f.moduleCaptures | map([&](const Type::Any &_v5_0) { return repr(_v5_0); }) | mk_string(","s)),
                     (f.termCaptures | map([&](const Type::Any &_v5_0) { return repr(_v5_0); }) | mk_string(","s)));
}

std::string repr(const Function &f) {
  return fmt::format("{} /* vis={} fp={} convention={} implements={} requires={} */ {}\n{}\n{}", repr(f.decl), repr(f.visibility),
                     repr(f.fpMode), repr(f.convention),
                     f.implements ^ map([&](const Sym &_v5_0) { return repr(_v5_0); }) ^ get_or_else(""s),
                     (f.requiredCapabilities | mk_string(","s)), "{"s,
                     (f.body | map([&](const Stmt::Any &_v6_0) { return repr(_v6_0); }) | mk_string("\n"s)) ^ indent(2), "}"s);
}

std::string repr(const FunctionDecl &f) {
  return fmt::format("def {}{}<{}>({}): {} /* affinity={} mod={} term={} */",
                     f.receiver ^ map([&](const Arg &r) { return fmt::format("{}.", repr(r)); }) ^ get_or_else(""s), repr(f.name),
                     (f.tpeVars | map([&](const Type::Var &_v5_0) { return repr(_v5_0); }) | mk_string(","s)),
                     (f.args | map([&](const Arg &_v5_0) { return repr(_v5_0); }) | mk_string(", "s)), repr(f.rtn), repr(f.affinity),
                     (f.moduleCaptures | map([&](const Arg &_v5_0) { return repr(_v5_0); }) | mk_string(","s)),
                     (f.termCaptures | map([&](const Arg &_v5_0) { return repr(_v5_0); }) | mk_string(","s)));
}

std::string repr(const MetaEntry &m) { return fmt::format("{}={}", m.key, m.value); }

std::string repr(const Interface &l) {
  return fmt::format("interface {} {}\n{}\n{}\n{}", repr(l.name), "{"s,
                     (l.declarations | map([&](const FunctionDecl &_v6_0) { return repr(_v6_0); }) | mk_string("\n"s)) ^ indent(2),
                     (l.metadata | map([&](const MetaEntry &_v6_0) { return repr(_v6_0); }) | mk_string("\n"s)) ^ indent(2), "}"s);
}

std::string repr(const StructDef &s) {
  return fmt::format("class {}<{}>({}) <: {}", repr(s.name),
                     (s.tpeVars | map([&](const Type::Var &_v5_0) { return repr(_v5_0); }) | mk_string(","s)),
                     (s.members | map([&](const Named &m) { return fmt::format("{}: {}", m.symbol, repr(m.tpe)); }) | mk_string(", "s)),
                     (s.parents | map([&](const Type::Struct &_v5_0) { return repr(_v5_0); }) | mk_string(", "s)));
}

std::string repr(const Program &s) {
  return fmt::format("{}\n{}\n{}", (s.defs | map([&](const StructDef &_v5_0) { return repr(_v5_0); }) | mk_string("\n"s)),
                     s.entry ^ map([&](const Function &_v5_0) { return repr(_v5_0); }) ^ get_or_else(""s),
                     (s.functions | map([&](const Function &_v5_0) { return repr(_v5_0); }) | mk_string("\n"s)));
}

std::string repr(const StructLayout &l) {
  return fmt::format("StructLayout[{}, sizeInBytes={}, align={}]{}\n{}\n{}", l.name, l.sizeInBytes, l.alignment, "{"s,
                     (l.members | map([&](const StructLayoutMember &m) {
                        return fmt::format("{}: {} (+{},{})", m.name.symbol, repr(m.name.tpe), m.offsetInBytes, m.sizeInBytes);
                      })
                      | mk_string("\n"s))
                         ^ indent(2),
                     "}"s);
}

} // namespace polyregion::polyast
