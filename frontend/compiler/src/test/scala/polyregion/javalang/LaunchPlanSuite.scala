package polyregion.javalang

import polyregion.ast.PolyAST as p

class LaunchPlanSuite extends munit.FunSuite {

  private val ints = p.Type.Ptr(p.Type.IntS32, p.Type.Space.Global)

  private def entry(captures: p.Named*): p.Function = entryOf(None, captures*)

  private def entryOf(receiver: Option[p.Named], captures: p.Named*): p.Function = p.Function(
    decl = p.FunctionDecl(
      name = p.Sym("entry"),
      tpeVars = Nil,
      receiver = receiver.map(p.Arg(_)),
      args = Nil,
      moduleCaptures = Nil,
      termCaptures = captures.toList.map(p.Arg(_)),
      rtn = p.Type.Unit0,
      affinity = p.Function.Affinity.Offload
    ),
    body = Nil,
    visibility = p.Function.Visibility.Exported,
    fpMode = p.Function.FpMode.Relaxed,
    convention = p.CallConvention.OffloadEntry
  )

  test("a launch source follows its parameter when the sources are listed in another order") {
    val xs     = p.Named("xs", ints)
    val length = p.Named("#len0", p.Type.IntS32)
    val slots = Offloader
      .slotsOf(
        entry(xs, length),
        List(
          TreeRemapper.Captured("#len0", TreeRemapper.Capture.Length(0)),
          TreeRemapper.Captured("xs", TreeRemapper.Capture.Arg(0))
        ),
        hasResult = false
      )
      .fold(e => fail(e.getMessage), identity)
    assertEquals(slots.map(s => s.name -> s.source), List("xs" -> "a0", "#len0" -> "l0"))
  }

  test("the receiver keeps its place ahead of the captures") {
    val self = p.Named("this", p.Type.Struct(p.Sym("Cell"), Nil))
    val xs   = p.Named("xs", ints)
    val slots = Offloader
      .slotsOf(
        entryOf(Some(self), xs),
        List(
          TreeRemapper.Captured("xs", TreeRemapper.Capture.Arg(0)),
          TreeRemapper.Captured("this", TreeRemapper.Capture.Receiver)
        ),
        hasResult = false
      )
      .fold(e => fail(e.getMessage), identity)
    assertEquals(slots.map(s => s.name -> s.source), List("this" -> "t", "xs" -> "a0"))
  }
}
