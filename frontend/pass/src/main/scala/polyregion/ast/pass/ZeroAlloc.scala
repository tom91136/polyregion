package polyregion.ast.pass

import polyregion.ast.{Log, PolyAST as p}

object ZeroAlloc extends ProgramPass {

  override def apply(program: p.Program, log: Log): p.Program = {
    val defs = program.defs.map(d => d.name -> d).toMap
    program.copy(entry = program.entry.map(fill(_, defs)), functions = program.functions.map(fill(_, defs)))
  }

  private def isHeap(a: p.Expr.Alloc): Boolean = a.space == p.Type.Space.Global && a.region == p.Region.Opaque

  private def fill(f: p.Function, defs: Map[p.Sym, p.StructDef]): p.Function = {
    val fresh = Binders.freshNamer(f, "_zero_")
    def zeroed(stmt: p.Stmt, ptr: p.Term.Select, a: p.Expr.Alloc): List[p.Stmt] =
      zeroFill(ptr, a, defs, fresh).fold(List(stmt))(stmt :: _)
    f.copy(body = Binders.mapStmts(f.body) {
      case s @ p.Stmt.Var(n, Some(a @ p.Expr.Alloc(comp = _)), _) if isHeap(a) =>
        zeroed(s, p.Term.Select(n, Nil, n.tpe), a)
      case s @ p.Stmt.Mut(lhs, a @ p.Expr.Alloc(comp = _)) if isHeap(a) => zeroed(s, lhs, a)
      case other                                                        => List(other)
    })
  }

  private def zeroFill(
      ptr: p.Term.Select,
      a: p.Expr.Alloc,
      defs: Map[p.Sym, p.StructDef],
      fresh: p.Type => p.Named
  ): Option[List[p.Stmt]] =
    for {
      zeroIdx <- Zero.term(a.size.tpe)
      oneIdx  <- Zero.one(a.size.tpe)
      value <- Zero.term(a.comp).map(t => (Nil, t)).orElse {
        val z = fresh(a.comp)
        Zero.declare(defs, z, mutable = false).map(_ -> p.Term.Select(z, Nil, z.tpe))
      }
    } yield {
      val (setup, zero)    = value
      val idx              = fresh(a.size.tpe)
      val more             = fresh(p.Type.Bool1)
      def i: p.Term.Select = p.Term.Select(idx, Nil, idx.tpe)
      def m: p.Term.Select = p.Term.Select(more, Nil, more.tpe)
      def inRange: p.Expr  = p.Expr.IntrOp(p.Intr.LogicLt(i, a.size))
      setup ::: List(
        p.Stmt.Var(idx, Some(p.Expr.Alias(zeroIdx)), isMutable = true),
        p.Stmt.Var(more, Some(inRange), isMutable = true),
        p.Stmt.While(
          m,
          List(
            p.Stmt.Update(ptr, i, zero),
            p.Stmt.Mut(i, p.Expr.IntrOp(p.Intr.Add(i, oneIdx, idx.tpe))),
            p.Stmt.Mut(m, inRange)
          )
        )
      )
    }
}
