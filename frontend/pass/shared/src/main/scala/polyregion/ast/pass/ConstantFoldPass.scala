package polyregion.ast.pass

import polyregion.ast.{PolyAST as p, *, given}
import polyregion.ast.Traversal.*

object ConstantFoldPass extends ProgramPass {

  type Env = Map[p.Named, p.Term]

  override def apply(program: p.Program, log: Log): p.Program =
    p.Program(
      run(program.entry, log.subLog(s"ConstantFoldPass on ${program.entry.name}")),
      program.functions.map(f => run(f, log.subLog(s"ConstantFoldPass on ${f.name}"))),
      program.defs,
      program.phase
    )

  private def run(f: p.Function, log: Log): p.Function = {
    val (n, reduced) = doUntilNotEq(f) { (_, f) =>
      val mutated = f.collectAll[p.Stmt].collect { case p.Stmt.Mut(p.Term.Select(name, _, _), _) => name }.toSet
      // Names whose address is taken via RefTo can be mutated indirectly through the resulting
      // pointer; const-binding them and substituting their reads would be unsound. The backend
      // also rejects RefTo of a constant, so even a pure-read const-bind followed by RefTo
      // produces a semantic error -- exclude these names from the const env entirely.
      val addressTaken =
        f.collectAll[p.Expr].collect { case p.Expr.RefTo(p.Term.Select(name, _, _), _, _, _) => name }.toSet
      f.copy(body = foldStmts(f.body, Map.empty, mutated ++ addressTaken, log)._1)
    }
    log.info(s"Constant fold is stable after ${n} passes")
    reduced
  }

  private def foldStmts(stmts: List[p.Stmt], env0: Env, mutated: Set[p.Named], log: Log): (List[p.Stmt], Env) = {
    val (rev, envOut) = stmts.foldLeft((List.empty[p.Stmt], env0)) { case ((acc, env), s) =>
      val (out, env1) = foldStmt(s, env, mutated, log)
      (out reverse_::: acc, env1)
    }
    (rev.reverse, envOut)
  }

  private def foldStmt(s: p.Stmt, env: Env, mutated: Set[p.Named], log: Log): (List[p.Stmt], Env) = s match {
    case p.Stmt.Var(name, None, mut) => (p.Stmt.Var(name, None, mut) :: Nil, env)
    case p.Stmt.Var(name, Some(e), mut) =>
      val folded = foldExpr(e, env)
      val env2 = folded match {
        case p.Expr.Alias(c) if isConstTerm(c) && !mut && !mutated.contains(name) =>
          log.info(s"const-bind ${name.repr} = ${c.repr}")
          env + (name -> c)
        case _ => env
      }
      (p.Stmt.Var(name, Some(folded), mut) :: Nil, env2)

    case p.Stmt.Mut(lhs, e) => (p.Stmt.Mut(lhs, foldExpr(e, env)) :: Nil, env)

    case p.Stmt.Update(lhs, idx, value) => (p.Stmt.Update(lhs, foldTerm(idx, env), foldTerm(value, env)) :: Nil, env)

    case p.Stmt.While(cond, body) =>
      val c2 = foldTerm(cond, env)
      c2 match {
        case p.Term.Bool1Const(false) =>
          log.info("while(false) -> drop")
          (Nil, env)
        case _ =>
          (p.Stmt.While(c2, foldStmts(body, env, mutated, log)._1) :: Nil, env)
      }

    case p.Stmt.ForRange(induction, lb, ub, step, body) =>
      val lb2   = foldTerm(lb, env)
      val ub2   = foldTerm(ub, env)
      val step2 = foldTerm(step, env)
      val empty = (asLong(lb2), asLong(ub2), asLong(step2)) match {
        case (Some(l), Some(u), Some(s)) if s > 0 => l >= u
        case (Some(l), Some(u), Some(s)) if s < 0 => l <= u
        case _                                    => false
      }
      if (empty) {
        log.info(s"for(${induction.repr} = ${lb2.repr}; < ${ub2.repr}; += ${step2.repr}) is empty -> drop")
        (Nil, env)
      } else (p.Stmt.ForRange(induction, lb2, ub2, step2, foldStmts(body, env, mutated, log)._1) :: Nil, env)

    case p.Stmt.Break => (p.Stmt.Break :: Nil, env)
    case p.Stmt.Cont  => (p.Stmt.Cont :: Nil, env)

    case p.Stmt.Cond(cond, t, f) =>
      val c2 = foldTerm(cond, env)
      c2 match {
        case p.Term.Bool1Const(true) =>
          log.info("if(true) { t } else { f } -> t")
          foldStmts(t, env, mutated, log)
        case p.Term.Bool1Const(false) =>
          log.info("if(false) { t } else { f } -> f")
          foldStmts(f, env, mutated, log)
        case _ =>
          (p.Stmt.Cond(c2, foldStmts(t, env, mutated, log)._1, foldStmts(f, env, mutated, log)._1) :: Nil, env)
      }

    case p.Stmt.Return(value) => (p.Stmt.Return(foldExpr(value, env)) :: Nil, env)

    case p.Stmt.Annotated(inner, pos, c) =>
      val (xs, env2) = foldStmt(inner, env, mutated, log)
      (xs.map(p.Stmt.Annotated(_, pos, c)), env2)
  }

  private def foldExpr(e: p.Expr, env: Env): p.Expr = e match {
    case p.Expr.Alias(t)   => p.Expr.Alias(foldTerm(t, env))
    case p.Expr.SpecOp(op) => p.Expr.SpecOp(op.modifyAll[p.Term](foldTerm(_, env)))
    case p.Expr.MathOp(op) => p.Expr.MathOp(op.modifyAll[p.Term](foldTerm(_, env)))
    case p.Expr.IntrOp(op) =>
      val op2 = op.modifyAll[p.Term](foldTerm(_, env))
      tryFoldIntr(op2) match {
        case Some(c) => p.Expr.Alias(c)
        case None    => p.Expr.IntrOp(op2)
      }
    case p.Expr.Cast(from, as) =>
      val from2 = foldTerm(from, env)
      tryFoldCast(from2, as) match {
        case Some(c) => p.Expr.Alias(c)
        case None    => p.Expr.Cast(from2, as)
      }
    case p.Expr.Index(lhs, idx, comp) => p.Expr.Index(foldTerm(lhs, env), foldTerm(idx, env), comp)
    case p.Expr.RefTo(lhs, idx, comp, sp) =>
      p.Expr.RefTo(foldTerm(lhs, env), idx.map(foldTerm(_, env)), comp, sp)
    case p.Expr.Alloc(comp, size, sp) => p.Expr.Alloc(comp, foldTerm(size, env), sp)
    case p.Expr.Invoke(n, ts, recv, args, rtn) =>
      p.Expr.Invoke(n, ts, recv.map(foldTerm(_, env)), args.map(foldTerm(_, env)), rtn)
  }

  private def foldTerm(t: p.Term, env: Env): p.Term = t match {
    case p.Term.Select(root, Nil, _) => env.getOrElse(root, t)
    case other                       => other
  }

  private def isConstTerm(t: p.Term): Boolean = t match {
    case _: p.Term.Float16Const | _: p.Term.Float32Const | _: p.Term.Float64Const | _: p.Term.IntU8Const |
        _: p.Term.IntU16Const | _: p.Term.IntU32Const | _: p.Term.IntU64Const | _: p.Term.IntS8Const |
        _: p.Term.IntS16Const | _: p.Term.IntS32Const | _: p.Term.IntS64Const | _: p.Term.Bool1Const |
        p.Term.Unit0Const | _: p.Term.NullPtrConst =>
      true
    case _ => false
  }

  private def asLong(t: p.Term): Option[Long] = t match {
    case p.Term.IntS8Const(v)  => Some(v.toLong)
    case p.Term.IntS16Const(v) => Some(v.toLong)
    case p.Term.IntS32Const(v) => Some(v.toLong)
    case p.Term.IntS64Const(v) => Some(v)
    case p.Term.IntU8Const(v)  => Some(v.toLong & 0xffL)
    case p.Term.IntU16Const(v) => Some(v.toLong & 0xffffL)
    case p.Term.IntU32Const(v) => Some(v.toLong & 0xffffffffL)
    // signed bit-pattern is fine for shifts/bitwise.
    case p.Term.IntU64Const(v) => Some(v)
    case _                     => None
  }

  private def asDouble(t: p.Term): Option[Double] = t match {
    case p.Term.Float16Const(v) => Some(v.toDouble)
    case p.Term.Float32Const(v) => Some(v.toDouble)
    case p.Term.Float64Const(v) => Some(v)
    case _                      => None
  }

  private def asBool(t: p.Term): Option[Boolean] = t match {
    case p.Term.Bool1Const(v) => Some(v)
    case _                    => None
  }

  private def intTerm(tpe: p.Type, v: Long): Option[p.Term] = tpe match {
    case p.Type.IntS8  => Some(p.Term.IntS8Const(v.toByte))
    case p.Type.IntS16 => Some(p.Term.IntS16Const(v.toShort))
    case p.Type.IntS32 => Some(p.Term.IntS32Const(v.toInt))
    case p.Type.IntS64 => Some(p.Term.IntS64Const(v))
    case p.Type.IntU8  => Some(p.Term.IntU8Const(v.toByte))
    case p.Type.IntU16 => Some(p.Term.IntU16Const(v.toChar))
    case p.Type.IntU32 => Some(p.Term.IntU32Const(v.toInt))
    case p.Type.IntU64 => Some(p.Term.IntU64Const(v))
    case _             => None
  }

  private def floatTerm(tpe: p.Type, v: Double): Option[p.Term] = tpe match {
    case p.Type.Float16 => Some(p.Term.Float16Const(v.toFloat))
    case p.Type.Float32 => Some(p.Term.Float32Const(v.toFloat))
    case p.Type.Float64 => Some(p.Term.Float64Const(v))
    case _              => None
  }

  private def tryFoldIntr(op: p.Intr): Option[p.Term] = op match {
    case p.Intr.Neg(x, t)   => foldUnaryNumeric(x, t)(_.unary_-)(-_)
    case p.Intr.Pos(x, t)   => foldUnaryNumeric(x, t)(identity)(identity)
    case p.Intr.BNot(x, t)  => asLong(x).flatMap(v => intTerm(t, ~v))
    case p.Intr.LogicNot(x) => asBool(x).map(v => p.Term.Bool1Const(!v))

    case p.Intr.Add(x, y, t) => foldBinNumeric(x, y, t)(_ + _)(_ + _)
    case p.Intr.Sub(x, y, t) => foldBinNumeric(x, y, t)(_ - _)(_ - _)
    case p.Intr.Mul(x, y, t) => foldBinNumeric(x, y, t)(_ * _)(_ * _)
    case p.Intr.Div(x, y, t) =>
      // never fold integer /0; preserve the runtime trap.
      (asLong(x), asLong(y)) match {
        case (Some(_), Some(0L)) if t.kind == p.Type.Kind.Integral => None
        case (Some(a), Some(b)) if t.kind == p.Type.Kind.Integral  => intTerm(t, signExtend(t, a) / signExtend(t, b))
        case _ =>
          (asDouble(x), asDouble(y)) match {
            case (Some(a), Some(b)) => floatTerm(t, a / b)
            case _                  => None
          }
      }
    case p.Intr.Rem(x, y, t) =>
      (asLong(x), asLong(y)) match {
        case (Some(_), Some(0L)) if t.kind == p.Type.Kind.Integral => None
        case (Some(a), Some(b)) if t.kind == p.Type.Kind.Integral  => intTerm(t, signExtend(t, a) % signExtend(t, b))
        case _ =>
          (asDouble(x), asDouble(y)) match {
            case (Some(a), Some(b)) => floatTerm(t, a % b)
            case _                  => None
          }
      }
    case p.Intr.Min(x, y, t) => foldBinNumeric(x, y, t)(math.min(_, _))(math.min(_, _))
    case p.Intr.Max(x, y, t) => foldBinNumeric(x, y, t)(math.max(_, _))(math.max(_, _))

    case p.Intr.BAnd(x, y, t) =>
      for {
        a <- asLong(x)
        b <- asLong(y)
        r <- intTerm(t, a & b)
      } yield r
    case p.Intr.BOr(x, y, t) =>
      for {
        a <- asLong(x)
        b <- asLong(y)
        r <- intTerm(t, a | b)
      } yield r
    case p.Intr.BXor(x, y, t) =>
      for {
        a <- asLong(x)
        b <- asLong(y)
        r <- intTerm(t, a ^ b)
      } yield r
    case p.Intr.BSL(x, y, t) =>
      for {
        a <- asLong(x)
        b <- asLong(y)
        r <- intTerm(t, a << b)
      } yield r
    case p.Intr.BSR(x, y, t) =>
      for {
        a <- asLong(x)
        b <- asLong(y)
        r <- intTerm(t, a >> b)
      } yield r
    case p.Intr.BZSR(x, y, t) =>
      for {
        a <- asLong(x)
        b <- asLong(y)
        r <- intTerm(t, a >>> b)
      } yield r

    case p.Intr.LogicAnd(x, y) =>
      (asBool(x), asBool(y)) match { case (Some(a), Some(b)) => Some(p.Term.Bool1Const(a && b)); case _ => None }
    case p.Intr.LogicOr(x, y) =>
      (asBool(x), asBool(y)) match { case (Some(a), Some(b)) => Some(p.Term.Bool1Const(a || b)); case _ => None }
    case p.Intr.LogicEq(x, y) => foldCmp(x, y)(_ == _)(_ == _).orElse(boolEq(x, y))
    case p.Intr.LogicNeq(x, y) =>
      foldCmp(x, y)(_ != _)(_ != _).orElse(boolEq(x, y).map(b => p.Term.Bool1Const(!b.value)))
    case p.Intr.LogicLt(x, y)  => foldCmp(x, y)(_ < _)(_ < _)
    case p.Intr.LogicLte(x, y) => foldCmp(x, y)(_ <= _)(_ <= _)
    case p.Intr.LogicGt(x, y)  => foldCmp(x, y)(_ > _)(_ > _)
    case p.Intr.LogicGte(x, y) => foldCmp(x, y)(_ >= _)(_ >= _)
  }

  private def foldUnaryNumeric(x: p.Term, t: p.Type)(longF: Long => Long)(doubleF: Double => Double): Option[p.Term] =
    asLong(x).flatMap(v => intTerm(t, longF(v))).orElse(asDouble(x).flatMap(v => floatTerm(t, doubleF(v))))

  private def foldBinNumeric(x: p.Term, y: p.Term, t: p.Type)(longF: (Long, Long) => Long)(
      doubleF: (Double, Double) => Double
  ): Option[p.Term] =
    (asLong(x), asLong(y)) match {
      case (Some(a), Some(b)) if t.kind == p.Type.Kind.Integral => intTerm(t, longF(a, b))
      case _ =>
        (asDouble(x), asDouble(y)) match {
          case (Some(a), Some(b)) => floatTerm(t, doubleF(a, b))
          case _                  => None
        }
    }

  private def foldCmp(x: p.Term, y: p.Term)(longF: (Long, Long) => Boolean)(
      doubleF: (Double, Double) => Boolean
  ): Option[p.Term] =
    (asLong(x), asLong(y)) match {
      case (Some(a), Some(b)) => Some(p.Term.Bool1Const(longF(a, b)))
      case _ =>
        (asDouble(x), asDouble(y)) match {
          case (Some(a), Some(b)) => Some(p.Term.Bool1Const(doubleF(a, b)))
          case _                  => None
        }
    }

  private def boolEq(x: p.Term, y: p.Term): Option[p.Term.Bool1Const] = (asBool(x), asBool(y)) match {
    case (Some(a), Some(b)) => Some(p.Term.Bool1Const(a == b))
    case _                  => None
  }

  private def tryFoldCast(from: p.Term, as: p.Type): Option[p.Term] =
    if (from.tpe == as) Some(from)
    else
      asLong(from)
        .flatMap { v =>
          as match {
            case p.Type.Float16 => Some(p.Term.Float16Const(v.toFloat))
            case p.Type.Float32 => Some(p.Term.Float32Const(v.toFloat))
            case p.Type.Float64 => Some(p.Term.Float64Const(v.toDouble))
            case _              => intTerm(as, v)
          }
        }
        .orElse(asDouble(from).flatMap { v =>
          as match {
            case p.Type.Float16 | p.Type.Float32 | p.Type.Float64 => floatTerm(as, v)
            case _                                                => intTerm(as, v.toLong)
          }
        })
        .orElse(asBool(from).flatMap { v =>
          as match {
            case p.Type.Bool1 => Some(p.Term.Bool1Const(v))
            case _            => intTerm(as, if (v) 1L else 0L)
          }
        })

  // Sign-extend so narrower signed types (IntS8/16/32) match runtime semantics after the
  // surrounding intTerm truncation; IntS64 and unsigned types pass through unchanged.
  private def signExtend(t: p.Type, v: Long): Long = t match {
    case p.Type.IntS8  => v.toByte.toLong
    case p.Type.IntS16 => v.toShort.toLong
    case p.Type.IntS32 => v.toInt.toLong
    case _             => v
  }

}
