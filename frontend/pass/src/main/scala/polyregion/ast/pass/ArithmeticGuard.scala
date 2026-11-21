package polyregion.ast.pass

import polyregion.ast.{Log, PolyAST as p}

object ArithmeticGuard extends ProgramPass {

  private val DivideByZero = "/ by zero"

  private val Unsigned: Map[p.Type, p.Type] = Map(
    p.Type.IntS8  -> p.Type.IntU8,
    p.Type.IntS16 -> p.Type.IntU16,
    p.Type.IntS32 -> p.Type.IntU32,
    p.Type.IntS64 -> p.Type.IntU64
  )

  private def constant(tpe: p.Type, value: Int): p.Term = tpe match {
    case p.Type.IntS8  => p.Term.IntS8Const(value.toByte)
    case p.Type.IntS16 => p.Term.IntS16Const(value.toShort)
    case p.Type.IntS32 => p.Term.IntS32Const(value)
    case p.Type.IntS64 => p.Term.IntS64Const(value.toLong)
    case p.Type.IntU8  => p.Term.IntU8Const(value.toByte)
    case p.Type.IntU16 => p.Term.IntU16Const(value.toChar)
    case p.Type.IntU32 => p.Term.IntU32Const(value)
    case p.Type.IntU64 => p.Term.IntU64Const(value.toLong)
    case other         => throw IllegalArgumentException(s"no integral constant of ${other.repr}")
  }

  private object Division {
    def unapply(expr: p.Expr): Option[Division] = expr match {
      case p.Expr.IntrOp(p.Intr.Div(x, y, tpe)) if Unsigned.contains(tpe) =>
        Some(Division(remainder = false, x, y, tpe))
      case p.Expr.IntrOp(p.Intr.Rem(x, y, tpe)) if Unsigned.contains(tpe) => Some(Division(remainder = true, x, y, tpe))
      case _                                                              => None
    }
  }

  private case class Division(remainder: Boolean, x: p.Term, y: p.Term, tpe: p.Type)

  private val GuardedMeta = "#arithmetic_guarded"

  override def apply(program: p.Program, log: Log): p.Program =
    if (program.metadata.exists(_.key == GuardedMeta)) program
    else
      program.copy(
        entry = program.entry.map(guard(_)),
        functions = program.functions.map(guard(_)),
        metadata = p.MetaEntry(GuardedMeta, "") :: program.metadata
      )

  private def guard(f: p.Function): p.Function = {
    val fresh = Binders.freshNamer(f, "_arith_")
    f.copy(body = Binders.mapStmts(f.body) {
      case p.Stmt.Var(name, Some(Division(division)), _) =>
        p.Stmt.Var(name, Some(p.Expr.Alias(constant(name.tpe, 0))), isMutable = true) ::
          guarded(division, fresh, p.Stmt.Mut(p.Term.Select(name, Nil, name.tpe), _))
      case p.Stmt.Mut(target, Division(division)) => guarded(division, fresh, p.Stmt.Mut(target, _))
      case p.Stmt.Return(Division(division)) =>
        val result                = fresh(division.tpe)
        val select: p.Term.Select = p.Term.Select(result, Nil, result.tpe)
        p.Stmt.Var(result, Some(p.Expr.Alias(constant(result.tpe, 0))), isMutable = true) ::
          guarded(division, fresh, p.Stmt.Mut(select, _)) ::: List(p.Stmt.Return(p.Expr.Alias(select)))
      case other => List(other)
    })
  }

  private def guarded(division: Division, fresh: p.Type => p.Named, assign: p.Expr => p.Stmt): List[p.Stmt] = {
    val Division(remainder, x, y, tpe) = division
    def compare(term: p.Term, value: Int): (List[p.Stmt], p.Term) = {
      val flag = fresh(p.Type.Bool1)
      List(p.Stmt.Var(flag, Some(p.Expr.IntrOp(p.Intr.LogicEq(term, constant(tpe, value)))))) ->
        p.Term.Select(flag, Nil, flag.tpe)
    }
    val (zeroTest, isZero)   = compare(y, 0)
    val (minusTest, isMinus) = compare(y, -1)
    val raise = p.Stmt.Var(
      fresh(p.Type.Unit0),
      Some(
        p.Expr.SpecOp(
          p.Spec.Assert(p.Term.IntU32Const(p.Enums.AssertCode.Arithmetic.value), p.Term.StringConst(DivideByZero))
        )
      )
    )
    val minusOne =
      if (remainder) List(assign(p.Expr.Alias(constant(tpe, 0))))
      else {
        val unsigned = Unsigned(tpe)
        val bits     = fresh(unsigned)
        val negated  = fresh(unsigned)
        val result   = fresh(tpe)
        List(
          p.Stmt.Var(bits, Some(p.Expr.Cast(x, unsigned))),
          p.Stmt.Var(
            negated,
            Some(p.Expr.IntrOp(p.Intr.Sub(constant(unsigned, 0), p.Term.Select(bits, Nil, unsigned), unsigned)))
          ),
          p.Stmt.Var(result, Some(p.Expr.Cast(p.Term.Select(negated, Nil, unsigned), tpe))),
          assign(p.Expr.Alias(p.Term.Select(result, Nil, tpe)))
        )
      }
    val plain = assign(
      p.Expr.IntrOp(if (remainder) p.Intr.Rem(x, y, tpe) else p.Intr.Div(x, y, tpe))
    )
    zeroTest ::: minusTest ::: List(p.Stmt.Cond(isZero, List(raise), List(p.Stmt.Cond(isMinus, minusOne, List(plain)))))
  }
}
