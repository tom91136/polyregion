package polyregion.ast.pass

import polyregion.ast.Traversal.*
import polyregion.ast.{PolyAST as p, *, given}

private[pass] object Binders {

  def substVars[A](t: p.Type, env: Map[String, A])(bound: (p.Type.Var, A) => p.Type): p.Type = t match {
    case variable @ p.Type.Var(name, _)  => env.get(name).fold(variable)(bound(variable, _))
    case p.Type.Struct(name, args)       => p.Type.Struct(name, args.map(substVars(_, env)(bound)))
    case p.Type.Ptr(comp, space)         => p.Type.Ptr(substVars(comp, env)(bound), space)
    case p.Type.Arr(comp, length, space) => p.Type.Arr(substVars(comp, env)(bound), length, space)
    case p.Type.Exec(vars, args, rtn) =>
      val nested = env -- vars.map(_.name)
      p.Type.Exec(vars, args.map(substVars(_, nested)(bound)), substVars(rtn, nested)(bound))
    case other => other
  }

  def substType(t: p.Type, env: Map[String, p.Type]): p.Type = substVars(t, env)((_, bound) => bound)

  def renameVars(t: p.Type, env: Map[String, String]): p.Type =
    substVars(t, env)((variable, name) => variable.copy(name = name))

  def protectExec(prefix: String)(tpe: p.Type): p.Type = tpe match {
    case p.Type.Exec(vars, args, rtn) if !vars.forall(_.name.startsWith(prefix)) =>
      val renamed: List[p.Type.Var] =
        vars.map(variable => p.Type.Var(prefix + variable.name, variable.exactSizeInBytes))
      val env = vars.map(_.name).zip(renamed.map(_.name)).toMap
      p.Type.Exec(renamed, args.map(renameVars(_, env)), renameVars(rtn, env))
    case other => other
  }

  def unprotectExec(prefix: String)(tpe: p.Type): p.Type = tpe match {
    case variable @ p.Type.Var(name, _) if name.startsWith(prefix) =>
      variable.copy(name = name.stripPrefix(prefix))
    case p.Type.Exec(vars, args, rtn) =>
      p.Type.Exec(vars.map(variable => variable.copy(name = variable.name.stripPrefix(prefix))), args, rtn)
    case other => other
  }

  def renameBinders(rename: p.Named => p.Named)(stmt: p.Stmt): p.Stmt = stmt match {
    case p.Stmt.Var(n, expr, isMutable)         => p.Stmt.Var(rename(n), expr, isMutable)
    case p.Stmt.ForRange(i, lb, ub, step, body) => p.Stmt.ForRange(rename(i), lb, ub, step, body)
    case p.Stmt.Try(b, hs, fin) => p.Stmt.Try(b, hs.map(h => h.copy(binder = h.binder.map(rename))), fin)
    case other                  => other
  }

  def lowerNested(lower: List[p.Stmt] => List[p.Stmt])(stmt: p.Stmt): List[p.Stmt] = stmt match {
    case annotated @ p.Stmt.Annotated(inner, _, _) =>
      lower(List(inner)) match {
        case init :+ last => init :+ annotated.copy(inner = last)
        case Nil          => Nil
        case other        => other
      }
    case tried @ p.Stmt.Try(body, handlers, fin) =>
      List(
        tried.copy(
          body = lower(body),
          handlers = handlers.map(h => h.copy(body = lower(h.body))),
          fin = lower(fin)
        )
      )
    case other => List(other)
  }

  def freshNamer(function: p.Function, prefix: String): p.Type => p.Named = {
    val occupied = scala.collection.mutable.Set.from(
      (function.receiver.toList ::: function.args ::: function.moduleCaptures ::: function.termCaptures)
        .map(_.named.symbol) :::
        function.collectAll[p.Stmt].flatMap {
          case p.Stmt.Var(name, _, _)                 => List(name.symbol)
          case p.Stmt.ForRange(induction = induction) => List(induction.symbol)
          case _                                      => Nil
        }
    )
    val candidates = Iterator.from(0).map(index => s"$prefix$index")
    tpe => {
      var symbol = candidates.next()
      while (!occupied.add(symbol)) symbol = candidates.next()
      p.Named(symbol, tpe)
    }
  }
}
