package polyregion.ast

import polyregion.ast.PolyAST as p

object Core {

  private val t: p.Type.Var = p.Type.Var("T")

  private def callable(name: String, args: List[p.Type], rtn: p.Type): p.Arg =
    p.Arg(p.Named(name, p.Type.Exec(Nil, args, rtn)))

  private def declaration(
      name: String,
      typeVariables: List[p.Type.Var],
      args: List[p.Arg],
      rtn: p.Type
  ): p.FunctionDecl =
    p.FunctionDecl(
      p.Sym(List("polyregion", "core", name)),
      typeVariables,
      None,
      args,
      Nil,
      Nil,
      rtn,
      p.Function.Affinity.Host
    )

  private val n = p.Arg(p.Named("n", p.Type.IntS64))

  val interfaceDef: p.Interface = p.Interface(
    p.Sym(List("polyregion", "core")),
    List(
      declaration("for_each", Nil, List(n, callable("body", List(p.Type.IntS64), p.Type.Unit0)), p.Type.Unit0),
      declaration(
        "transform_reduce",
        List(t),
        List(n, p.Arg(p.Named("init", t)), callable("map", List(p.Type.IntS64), t), callable("combine", List(t, t), t)),
        t
      )
    )
  )
}
