package polyregion.ast.pass

import polyregion.ast.{Log, PolyAST as p, *, given}
import polyregion.ast.Traversal.*

object PassTest {

  object NoopLog extends Log {
    def info(message: String, details: String*): Unit = ()
    def subLog(name: String): Log                     = this
  }

  def sym(parts: String*): p.Sym                                = p.Sym(parts.toList)
  def named(name: String, tpe: p.Type = p.Type.IntS32): p.Named = p.Named(name, tpe)
  def arg(name: String, tpe: p.Type = p.Type.IntS32): p.Arg     = p.Arg(named(name, tpe))

  // The old DSL exposed `select(...)` returning an `Expr.Select`. Under the new shape, Select is
  // a Term variant; for tests that want it as an Expr we wrap with Alias.
  def selectT(n: p.Named): p.Term.Select                                = p.Term.Select(n, Nil, n.tpe)
  def selectT(name: String, tpe: p.Type = p.Type.IntS32): p.Term.Select = selectT(named(name, tpe))
  def select(n: p.Named): p.Expr                                        = p.Expr.Alias(selectT(n))
  def select(name: String, tpe: p.Type = p.Type.IntS32): p.Expr         = select(named(name, tpe))

  def fieldOf(root: p.Named)(name: String, t: p.Type): p.Term.Select =
    p.Term.Select(root, List(p.PathStep.Field(name)), t)

  def fn(
      name: String,
      args: List[p.Arg] = Nil,
      rtn: p.Type = p.Type.Unit0,
      body: List[p.Stmt] = List(p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))),
      tpeVars: List[String] = Nil,
      moduleCaptures: List[p.Arg] = Nil,
      termCaptures: List[p.Arg] = Nil,
      visibility: p.Function.Visibility = p.Function.Visibility.Exported,
      fpMode: p.Function.FpMode = p.Function.FpMode.Relaxed,
      isEntry: Boolean = false
  ): p.Function =
    p.Function(sym(name), tpeVars, None, args, moduleCaptures, termCaptures, rtn, body, visibility, fpMode, isEntry)

  def entry(
      args: List[p.Arg] = Nil,
      body: List[p.Stmt] = List(p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))),
      moduleCaptures: List[p.Arg] = Nil,
      termCaptures: List[p.Arg] = Nil
  ): p.Function =
    fn(p.Conventions.EntryName, args, p.Type.Unit0, body, Nil, moduleCaptures, termCaptures, isEntry = true)

  def program(
      entry: p.Function,
      functions: List[p.Function] = Nil,
      defs: List[p.StructDef] = Nil
  ): p.Program = p.Program(entry, functions, defs)
}
