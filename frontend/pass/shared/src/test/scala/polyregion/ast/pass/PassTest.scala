package polyregion.ast.pass

import polyregion.ast.{Log, PolyAST as p, *, given}
import polyregion.ast.Traversal.*

object PassTest {

  object NoopLog extends Log {
    def info(message: String, details: String*): Unit = ()
    def subLog(name: String): Log                     = this
  }

  def sym(parts: String*): p.Sym                                       = p.Sym(parts.toList)
  def named(name: String, tpe: p.Type = p.Type.IntS32): p.Named        = p.Named(name, tpe)
  def arg(name: String, tpe: p.Type = p.Type.IntS32): p.Arg            = p.Arg(named(name, tpe))
  def select(n: p.Named): p.Expr.Select                                = p.Expr.Select(Nil, n)
  def select(name: String, tpe: p.Type = p.Type.IntS32): p.Expr.Select = select(named(name, tpe))

  def fn(
      name: String,
      args: List[p.Arg] = Nil,
      rtn: p.Type = p.Type.Unit0,
      body: List[p.Stmt] = List(p.Stmt.Return(p.Expr.Unit0Const)),
      tpeVars: List[String] = Nil,
      moduleCaptures: List[p.Arg] = Nil,
      termCaptures: List[p.Arg] = Nil,
      attrs: Set[p.Function.Attr] = Set.empty
  ): p.Function =
    p.Function(sym(name), tpeVars, None, args, moduleCaptures, termCaptures, rtn, body, attrs)

  def entry(
      args: List[p.Arg] = Nil,
      body: List[p.Stmt] = List(p.Stmt.Return(p.Expr.Unit0Const)),
      moduleCaptures: List[p.Arg] = Nil,
      termCaptures: List[p.Arg] = Nil
  ): p.Function =
    fn("_main", args, p.Type.Unit0, body, Nil, moduleCaptures, termCaptures, Set(p.Function.Attr.Entry))

  def program(
      entry: p.Function,
      functions: List[p.Function] = Nil,
      defs: List[p.StructDef] = Nil
  ): p.Program = p.Program(entry, functions, defs)
}
