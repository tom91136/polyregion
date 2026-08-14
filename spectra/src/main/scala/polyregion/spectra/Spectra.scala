package polyregion.spectra

import polyregion.ast.PolyAST as p

object Spectra {

  private val t = p.Type.Var("T")
  private val u = p.Type.Var("U")
  private val k = p.Type.Var("K")
  private val v = p.Type.Var("V")

  private def param(index: Int): p.Arg.SizeExpr            = p.Arg.SizeExpr.Param(index)
  private def sum(lhs: Int, rhs: Int): p.Arg.SizeExpr      = p.Arg.SizeExpr.Add(param(lhs), param(rhs))
  private def elements(size: p.Arg.SizeExpr): p.Arg.Extent = p.Arg.Extent.Elements(size)

  private def buffer(
      name: String,
      component: p.Type,
      access: p.Arg.Access,
      size: p.Arg.SizeExpr
  ): p.Arg =
    p.Arg(
      p.Named(name, p.Type.Ptr(component, p.Type.Space.Global)),
      boundary = Some(p.Arg.Boundary(access, elements(size)))
    )

  private def read(name: String, component: p.Type = t, size: p.Arg.SizeExpr): p.Arg =
    buffer(name, component, p.Arg.Access.Read, size)
  private def write(name: String, component: p.Type = t, size: p.Arg.SizeExpr): p.Arg =
    buffer(name, component, p.Arg.Access.Write, size)
  private def readWrite(name: String, component: p.Type = t, size: p.Arg.SizeExpr): p.Arg =
    buffer(name, component, p.Arg.Access.ReadWrite, size)
  private def count(name: String): p.Arg                   = p.Arg(p.Named(name, p.Type.IntS32))
  private def scalar(name: String, tpe: p.Type = t): p.Arg = p.Arg(p.Named(name, tpe))
  private def callable(name: String, args: List[p.Type], rtn: p.Type): p.Arg =
    p.Arg(p.Named(name, p.Type.Exec(Nil, args, rtn)))

  private def declaration(
      name: String,
      typeVariables: List[String],
      args: List[p.Arg],
      rtn: p.Type
  ): p.FunctionDecl =
    p.FunctionDecl(
      p.Sym(List("spectra", name)),
      typeVariables,
      None,
      args,
      Nil,
      Nil,
      rtn,
      p.Function.Affinity.Host
    )

  private val T   = List("T")
  private val TU  = List("T", "U")
  private val TUV = List("T", "U", "V")
  private val KV  = List("K", "V")

  private val declarations: List[p.FunctionDecl] = List(
    declaration(
      "reduce",
      T,
      List(read("in", size = param(1)), count("n"), scalar("init"), callable("op", List(t, t), t)),
      t
    ),
    declaration(
      "transform",
      TU,
      List(read("in", size = param(2)), write("out", u, param(2)), count("n"), callable("op", List(t), u)),
      p.Type.Unit0
    ),
    declaration(
      "inclusive_scan",
      T,
      List(read("in", size = param(2)), write("out", size = param(2)), count("n"), callable("op", List(t, t), t)),
      p.Type.Unit0
    ),
    declaration(
      "exclusive_scan",
      T,
      List(
        read("in", size = param(2)),
        write("out", size = param(2)),
        count("n"),
        scalar("init"),
        callable("op", List(t, t), t)
      ),
      p.Type.Unit0
    ),
    declaration(
      "sort",
      T,
      List(readWrite("data", size = param(1)), count("n"), callable("op", List(t, t), p.Type.Bool1)),
      p.Type.Unit0
    ),
    declaration(
      "count_if",
      T,
      List(read("in", size = param(1)), count("n"), callable("op", List(t), p.Type.Bool1)),
      p.Type.IntS32
    ),
    declaration(
      "min_element",
      T,
      List(read("in", size = param(1)), count("n"), callable("op", List(t, t), p.Type.Bool1)),
      t
    ),
    declaration(
      "max_element",
      T,
      List(read("in", size = param(1)), count("n"), callable("op", List(t, t), p.Type.Bool1)),
      t
    ),
    declaration(
      "find_if",
      T,
      List(read("in", size = param(1)), count("n"), callable("op", List(t), p.Type.Bool1)),
      p.Type.IntS32
    ),
    declaration(
      "is_sorted",
      T,
      List(read("data", size = param(1)), count("n"), callable("op", List(t, t), p.Type.Bool1)),
      p.Type.Bool1
    ),
    declaration(
      "merge",
      T,
      List(
        read("a", size = param(1)),
        count("na"),
        read("b", size = param(3)),
        count("nb"),
        write("out", size = sum(1, 3)),
        callable("op", List(t, t), p.Type.Bool1)
      ),
      p.Type.Unit0
    ),
    declaration(
      "set_difference",
      T,
      List(
        read("a", size = param(1)),
        count("na"),
        read("b", size = param(3)),
        count("nb"),
        write("out", size = param(1)),
        callable("op", List(t, t), p.Type.Bool1)
      ),
      p.Type.IntS32
    ),
    declaration(
      "set_intersection",
      T,
      List(
        read("a", size = param(1)),
        count("na"),
        read("b", size = param(3)),
        count("nb"),
        write("out", size = param(5)),
        count("out_n"),
        callable("op", List(t, t), p.Type.Bool1)
      ),
      p.Type.IntS32
    ),
    declaration(
      "set_union",
      T,
      List(
        read("a", size = param(1)),
        count("na"),
        read("b", size = param(3)),
        count("nb"),
        write("out", size = sum(1, 3)),
        callable("op", List(t, t), p.Type.Bool1)
      ),
      p.Type.IntS32
    ),
    declaration(
      "includes",
      T,
      List(
        read("a", size = param(1)),
        count("na"),
        read("b", size = param(3)),
        count("nb"),
        callable("op", List(t, t), p.Type.Bool1)
      ),
      p.Type.Bool1
    ),
    declaration(
      "copy_if",
      T,
      List(
        read("in", size = param(1)),
        count("n"),
        write("out", size = param(1)),
        callable("op", List(t), p.Type.Bool1)
      ),
      p.Type.IntS32
    ),
    declaration(
      "remove_copy_if",
      T,
      List(
        read("in", size = param(1)),
        count("n"),
        write("out", size = param(1)),
        callable("op", List(t), p.Type.Bool1)
      ),
      p.Type.IntS32
    ),
    declaration(
      "unique_copy",
      T,
      List(
        read("in", size = param(1)),
        count("n"),
        write("out", size = param(1)),
        callable("eq", List(t, t), p.Type.Bool1)
      ),
      p.Type.IntS32
    ),
    declaration(
      "remove_if",
      T,
      List(readWrite("data", size = param(1)), count("n"), callable("op", List(t), p.Type.Bool1)),
      p.Type.IntS32
    ),
    declaration(
      "unique",
      T,
      List(readWrite("data", size = param(1)), count("n"), callable("eq", List(t, t), p.Type.Bool1)),
      p.Type.IntS32
    ),
    declaration(
      "sort_by_key",
      KV,
      List(
        readWrite("keys", k, param(2)),
        readWrite("values", v, param(2)),
        count("n"),
        callable("op", List(k, k), p.Type.Bool1)
      ),
      p.Type.Unit0
    ),
    declaration(
      "for_each",
      T,
      List(readWrite("data", size = param(1)), count("n"), callable("op", List(t), t)),
      p.Type.Unit0
    ),
    declaration(
      "gather",
      T,
      List(
        read("map", p.Type.IntS32, param(1)),
        count("n"),
        read("in", size = param(3)),
        count("in_n"),
        write("out", size = param(1))
      ),
      p.Type.Unit0
    ),
    declaration(
      "scatter",
      T,
      List(
        read("in", size = param(1)),
        count("n"),
        read("map", p.Type.IntS32, param(1)),
        write("out", size = param(4)),
        count("out_n")
      ),
      p.Type.Unit0
    ),
    declaration("copy", T, List(read("in", size = param(1)), count("n"), write("out", size = param(1))), p.Type.Unit0),
    declaration("reverse", T, List(readWrite("data", size = param(1)), count("n")), p.Type.Unit0),
    declaration("fill", T, List(write("out", size = param(1)), count("n"), scalar("v")), p.Type.Unit0),
    declaration(
      "transform_reduce",
      TU,
      List(
        read("in", size = param(1)),
        count("n"),
        scalar("init", u),
        callable("map", List(t), u),
        callable("op", List(u, u), u)
      ),
      u
    ),
    declaration(
      "transform_inclusive_scan",
      TU,
      List(
        read("in", size = param(2)),
        write("out", u, param(2)),
        count("n"),
        callable("map", List(t), u),
        callable("op", List(u, u), u)
      ),
      p.Type.Unit0
    ),
    declaration(
      "transform_exclusive_scan",
      TU,
      List(
        read("in", size = param(2)),
        write("out", u, param(2)),
        count("n"),
        scalar("init", u),
        callable("map", List(t), u),
        callable("op", List(u, u), u)
      ),
      p.Type.Unit0
    ),
    declaration(
      "reduce_by_key",
      KV,
      List(
        read("keys", k, param(4)),
        read("vals", v, param(4)),
        write("kout", k, param(4)),
        write("vout", v, param(4)),
        count("n"),
        callable("eq", List(k, k), p.Type.Bool1),
        callable("op", List(v, v), v)
      ),
      p.Type.IntS32
    ),
    declaration(
      "inclusive_scan_by_key",
      KV,
      List(
        read("keys", k, param(3)),
        read("vals", v, param(3)),
        write("out", v, param(3)),
        count("n"),
        callable("eq", List(k, k), p.Type.Bool1),
        callable("op", List(v, v), v)
      ),
      p.Type.Unit0
    ),
    declaration(
      "exclusive_scan_by_key",
      KV,
      List(
        read("keys", k, param(3)),
        read("vals", v, param(3)),
        write("out", v, param(3)),
        count("n"),
        scalar("init", v),
        callable("eq", List(k, k), p.Type.Bool1),
        callable("op", List(v, v), v)
      ),
      p.Type.Unit0
    ),
    declaration(
      "transform_binary",
      TUV,
      List(
        read("a", size = param(3)),
        read("b", u, param(3)),
        write("out", v, param(3)),
        count("n"),
        callable("op", List(t, u), v)
      ),
      p.Type.Unit0
    ),
    declaration(
      "inner_product",
      TUV,
      List(
        read("a", size = param(2)),
        read("b", u, param(2)),
        count("n"),
        scalar("init", v),
        callable("op_reduce", List(v, v), v),
        callable("op_product", List(t, u), v)
      ),
      v
    ),
    declaration(
      "equal",
      T,
      List(
        read("a", size = param(2)),
        read("b", size = param(2)),
        count("n"),
        callable("op", List(t, t), p.Type.Bool1)
      ),
      p.Type.Bool1
    ),
    declaration(
      "mismatch",
      T,
      List(
        read("a", size = param(2)),
        read("b", size = param(2)),
        count("n"),
        callable("op", List(t, t), p.Type.Bool1)
      ),
      p.Type.IntS32
    ),
    declaration(
      "search",
      T,
      List(
        read("in", size = param(1)),
        count("n"),
        read("sub", size = param(3)),
        count("m"),
        callable("op", List(t, t), p.Type.Bool1)
      ),
      p.Type.IntS32
    ),
    declaration(
      "search_n",
      T,
      List(
        read("in", size = param(1)),
        count("n"),
        scalar("count", p.Type.IntS32),
        scalar("value"),
        callable("op", List(t, t), p.Type.Bool1)
      ),
      p.Type.IntS32
    ),
    declaration(
      "all_of",
      T,
      List(read("in", size = param(1)), count("n"), callable("op", List(t), p.Type.Bool1)),
      p.Type.Bool1
    ),
    declaration(
      "any_of",
      T,
      List(read("in", size = param(1)), count("n"), callable("op", List(t), p.Type.Bool1)),
      p.Type.Bool1
    ),
    declaration(
      "none_of",
      T,
      List(read("in", size = param(1)), count("n"), callable("op", List(t), p.Type.Bool1)),
      p.Type.Bool1
    ),
    declaration(
      "find_if_not",
      T,
      List(read("in", size = param(1)), count("n"), callable("op", List(t), p.Type.Bool1)),
      p.Type.IntS32
    ),
    declaration("generate", T, List(write("out", size = param(1)), count("n"), callable("op", Nil, t)), p.Type.Unit0),
    declaration("generate_n", T, List(write("out", size = param(1)), count("n"), callable("op", Nil, t)), p.Type.Unit0),
    declaration(
      "tabulate",
      T,
      List(write("out", size = param(1)), count("n"), callable("op", List(p.Type.IntS32), t)),
      p.Type.Unit0
    ),
    declaration(
      "lower_bound",
      T,
      List(read("in", size = param(1)), count("n"), scalar("value"), callable("op", List(t, t), p.Type.Bool1)),
      p.Type.IntS32
    ),
    declaration(
      "upper_bound",
      T,
      List(read("in", size = param(1)), count("n"), scalar("value"), callable("op", List(t, t), p.Type.Bool1)),
      p.Type.IntS32
    ),
    declaration(
      "binary_search",
      T,
      List(read("in", size = param(1)), count("n"), scalar("value"), callable("op", List(t, t), p.Type.Bool1)),
      p.Type.Bool1
    ),
    declaration(
      "partition",
      T,
      List(readWrite("data", size = param(1)), count("n"), callable("op", List(t), p.Type.Bool1)),
      p.Type.Unit0
    ),
    declaration(
      "stable_partition",
      T,
      List(readWrite("data", size = param(1)), count("n"), callable("op", List(t), p.Type.Bool1)),
      p.Type.Unit0
    ),
    declaration(
      "partition_point",
      T,
      List(read("in", size = param(1)), count("n"), callable("op", List(t), p.Type.Bool1)),
      p.Type.IntS32
    ),
    declaration(
      "is_partitioned",
      T,
      List(read("data", size = param(1)), count("n"), callable("op", List(t), p.Type.Bool1)),
      p.Type.Bool1
    ),
    declaration(
      "is_sorted_until",
      T,
      List(read("data", size = param(1)), count("n"), callable("op", List(t, t), p.Type.Bool1)),
      p.Type.IntS32
    ),
    declaration(
      "stable_sort",
      T,
      List(readWrite("data", size = param(1)), count("n"), callable("op", List(t, t), p.Type.Bool1)),
      p.Type.Unit0
    ),
    declaration(
      "for_each_n",
      T,
      List(readWrite("data", size = param(1)), count("n"), callable("op", List(t), t)),
      p.Type.Unit0
    ),
    declaration(
      "minmax_element",
      T,
      List(
        read("in", size = param(1)),
        count("n"),
        write("min_out", size = p.Arg.SizeExpr.Const(1)),
        write("max_out", size = p.Arg.SizeExpr.Const(1)),
        callable("op", List(t, t), p.Type.Bool1)
      ),
      p.Type.Unit0
    ),
    declaration(
      "copy_n",
      T,
      List(read("in", size = param(1)), count("n"), write("out", size = param(1))),
      p.Type.Unit0
    ),
    declaration(
      "reverse_copy",
      T,
      List(read("in", size = param(1)), count("n"), write("out", size = param(1))),
      p.Type.Unit0
    ),
    declaration("fill_n", T, List(write("out", size = param(1)), count("n"), scalar("v")), p.Type.Unit0),
    declaration(
      "swap_ranges",
      T,
      List(readWrite("a", size = param(1)), count("n"), readWrite("b", size = param(1))),
      p.Type.Unit0
    ),
    declaration("count", T, List(read("in", size = param(1)), count("n"), scalar("value")), p.Type.IntS32),
    declaration("find", T, List(read("in", size = param(1)), count("n"), scalar("value")), p.Type.IntS32),
    declaration("remove", T, List(readWrite("data", size = param(1)), count("n"), scalar("value")), p.Type.IntS32),
    declaration(
      "remove_copy",
      T,
      List(read("in", size = param(1)), count("n"), write("out", size = param(1)), scalar("value")),
      p.Type.IntS32
    ),
    declaration(
      "replace",
      T,
      List(readWrite("io", size = param(1)), count("n"), scalar("oldv"), scalar("newv")),
      p.Type.Unit0
    ),
    declaration(
      "replace_copy",
      T,
      List(read("in", size = param(2)), write("out", size = param(2)), count("n"), scalar("oldv"), scalar("newv")),
      p.Type.Unit0
    ),
    declaration(
      "replace_if",
      T,
      List(readWrite("data", size = param(1)), count("n"), scalar("new_value"), callable("op", List(t), p.Type.Bool1)),
      p.Type.Unit0
    ),
    declaration(
      "replace_copy_if",
      T,
      List(
        read("in", size = param(2)),
        write("out", size = param(2)),
        count("n"),
        scalar("new_value"),
        callable("op", List(t), p.Type.Bool1)
      ),
      p.Type.Unit0
    ),
    declaration(
      "sequence",
      T,
      List(write("out", size = param(1)), count("n"), scalar("init"), scalar("step")),
      p.Type.Unit0
    ),
    declaration(
      "adjacent_difference",
      T,
      List(read("in", size = param(2)), write("out", size = param(2)), count("n"), callable("op", List(t, t), t)),
      p.Type.Unit0
    )
  )

  val interfaceDef: p.InterfaceDef = p.InterfaceDef(p.Sym("spectra"), declarations)
}
