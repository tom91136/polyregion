package polyregion.ast.pass

import polyregion.ast.{PolyAST as p, *, given}
import polyregion.ast.Traversal.*

// rewrites early `Return`s into an exit flag so a body can be spliced into a larger one. a return stores its
// value through `onReturn`, raises the flag and leaves the enclosing loops with a `Break` chain; the code after
// a returning construct is fenced by the flag
// examples:
//   if c { ret x }; rest            ->  if c { onReturn(x); flag = true }; if !flag { rest }
//   while w { if c { ret x }; b }   ->  while w { if c { onReturn(x); flag = true; break }; b }; if flag { break } else { rest }
private[pass] object ReturnUnwind {

  def containsReturn(s: p.Stmt): Boolean =
    s.collectFirst_[p.Stmt] { case r: p.Stmt.Return => r }.isDefined

  def needsUnwind(stmts: List[p.Stmt]): Boolean = stmts.exists {
    case t: p.Stmt.Try                 => t.blocks.exists(_.exists(containsReturn)) || t.blocks.exists(needsUnwind)
    case p.Stmt.While(_, b)            => b.exists(containsReturn) || needsUnwind(b)
    case p.Stmt.ForRange(body = b)     => b.exists(containsReturn) || needsUnwind(b)
    case p.Stmt.Cond(_, t, f)          => needsUnwind(t) || needsUnwind(f)
    case p.Stmt.Annotated(inner, _, _) => needsUnwind(List(inner))
    case _                             => false
  }

  def apply(
      stmts: List[p.Stmt],
      flag: p.Term.Select,
      onReturn: p.Expr => List[p.Stmt],
      pending: () => p.Named
  ): List[p.Stmt] = {
    def unwind(stmts: List[p.Stmt], inLoop: Boolean): List[p.Stmt] = {
      def guard(tail: List[p.Stmt]): List[p.Stmt] =
        if (inLoop) List(p.Stmt.Cond(flag, List(p.Stmt.Break), tail))
        else if (tail.isEmpty) Nil
        else List(p.Stmt.Cond(flag, Nil, tail))

      def finalizer(fin: List[p.Stmt]): List[p.Stmt] = {
        val rewritten = unwind(fin, inLoop = false)
        if (!fin.exists(containsReturn)) rewritten
        else {
          val savedName            = pending()
          val saved: p.Term.Select = p.Term.Select(savedName, Nil, p.Type.Bool1)
          val save                 = p.Stmt.Var(savedName, Some(p.Expr.Alias(flag)))
          val clear                = p.Stmt.Mut(flag, p.Expr.Alias(p.Term.Bool1Const(false)))
          val restore              = p.Stmt.Mut(flag, p.Expr.Alias(saved))
          save :: clear :: rewritten ::: List(p.Stmt.Cond(flag, Nil, List(restore)))
        }
      }

      stmts match {
        case Nil => Nil
        case p.Stmt.Return(e) :: _ =>
          onReturn(e) ::: List(p.Stmt.Mut(flag, p.Expr.Alias(p.Term.Bool1Const(true)))) :::
            Option.when(inLoop)(p.Stmt.Break).toList
        case p.Stmt.Cond(c, t, f) :: rest =>
          val head = p.Stmt.Cond(c, unwind(t, inLoop), unwind(f, inLoop))
          val tail = unwind(rest, inLoop)
          if (t.exists(containsReturn) || f.exists(containsReturn)) head :: guard(tail) else head :: tail
        case p.Stmt.While(c, b) :: rest =>
          val head = p.Stmt.While(c, unwind(b, inLoop = true))
          val tail = unwind(rest, inLoop)
          if (b.exists(containsReturn)) head :: guard(tail) else head :: tail
        case p.Stmt.ForRange(i, lb, ub, step, b) :: rest =>
          val head = p.Stmt.ForRange(i, lb, ub, step, unwind(b, inLoop = true))
          val tail = unwind(rest, inLoop)
          if (b.exists(containsReturn)) head :: guard(tail) else head :: tail
        case (t: p.Stmt.Try) :: rest =>
          val head = p.Stmt.Try(
            unwind(t.body, inLoop = false),
            t.handlers.map(h => h.copy(body = unwind(h.body, inLoop = false))),
            finalizer(t.fin)
          )
          val tail = unwind(rest, inLoop)
          if (containsReturn(t)) head :: guard(tail) else head :: tail
        case p.Stmt.Annotated(inner, pos, comment) :: rest =>
          val head = unwind(List(inner), inLoop).map(p.Stmt.Annotated(_, pos, comment))
          val tail = unwind(rest, inLoop)
          if (containsReturn(inner)) head ::: guard(tail) else head ::: tail
        case other :: rest => other :: unwind(rest, inLoop)
      }
    }
    unwind(stmts, inLoop = false)
  }
}
