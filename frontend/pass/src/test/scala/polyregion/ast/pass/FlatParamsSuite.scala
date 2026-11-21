package polyregion.ast.pass

import polyregion.ast.{PolyAST as p, *, given}
import PassTest.*

class FlatParamsSuite extends munit.FunSuite {

  private val i32 = p.Type.IntS32

  private def param(name: String, tpe: p.Type = i32) = p.Arg(named(name, tpe))

  private val error = param(p.Conventions.ErrorArg, p.Type.Ptr(p.Type.IntS8, p.Type.Space.Global))

  private val decl = p
    .FunctionDecl(
      sym("f"),
      Nil,
      Some(param("self")),
      List(error, param("x"), param("unit", p.Type.Unit0)),
      List(param("module")),
      List(param("term")),
      p.Type.Unit0,
      p.Function.Affinity.Offload
    )

  private def symbols(args: List[p.Arg]) = args.map(_.named.symbol)

  test("the flat order leads with the error buffer, then the receiver, captures, and arguments") {
    assertEquals(symbols(decl.flatParams), List(p.Conventions.ErrorArg, "self", "module", "term", "x", "unit"))
  }

  test("the invoke parameters exclude the error buffer and keep the receiver first") {
    assertEquals(symbols(decl.invokeParams), List("self", "module", "term", "x", "unit"))
  }

  test("the call parameters exclude the error buffer and the receiver") {
    assertEquals(symbols(decl.callParams), List("module", "term", "x", "unit"))
  }

  test("the runtime parameters drop unit parameters after substitution") {
    assertEquals(symbols(decl.runtimeParams()), List(p.Conventions.ErrorArg, "self", "module", "term", "x"))
    val generic = decl.copy(args = List(param("t", p.Type.Var("T"))))
    val erase: p.Type => p.Type = {
      case _: p.Type.Var => p.Type.Unit0
      case other         => other
    }
    assertEquals(symbols(generic.runtimeParams(erase)), List("self", "module", "term"))
  }

  test("a signature lists its call parameters in the order of the declaration") {
    val plain = decl.copy(args = decl.args.filterNot(_ == error))
    assertEquals(plain.signature.callParams, plain.callParams.map(_.named.tpe))
  }
}
