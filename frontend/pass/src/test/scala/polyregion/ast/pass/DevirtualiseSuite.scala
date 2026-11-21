package polyregion.ast.pass

import polyregion.ast.{PolyAST as p, *}
import polyregion.ast.Interpreter
import polyregion.ast.Interpreter.V
import PassTest.*

class DevirtualiseSuite extends munit.FunSuite {

  private val i32  = p.Type.IntS32
  private val g    = p.Type.Space.Global
  private val errT = p.Type.Ptr(p.Type.IntS8, g)

  private def struct(name: String): p.Type.Struct = p.Type.Struct(sym(name), Nil)
  private def ptr(t: p.Type): p.Type              = p.Type.Ptr(t, g)

  private val shape  = struct("Shape")
  private val circle = struct("Circle")
  private val rect   = struct("Rect")

  private def structDef(tpe: p.Type.Struct, members: (String, p.Type)*): p.StructDef =
    p.StructDef(
      tpe.name,
      Nil,
      named(p.Conventions.DispatchTag) :: members.toList.map((n, t) => named(n, t)),
      if (tpe == shape) Nil else List(shape)
    )

  private val defs = List(
    structDef(shape),
    structDef(circle, "r" -> i32),
    structDef(rect, "w"   -> i32, "h" -> i32)
  )

  private def method(
      owner: p.Type.Struct,
      name: String,
      args: List[p.Arg],
      rtn: p.Type,
      body: List[p.Stmt]
  ): p.Function = p.Function(
    p.FunctionDecl(
      sym(owner.name.fqn.head, name),
      Nil,
      Some(p.Arg(named("this", ptr(owner)))),
      args,
      Nil,
      Nil,
      rtn,
      p.Function.Affinity.Offload
    ),
    body,
    p.Function.Visibility.Internal,
    p.Function.FpMode.Relaxed,
    p.CallConvention.RegularCall
  )

  private def field(owner: p.Type.Struct, name: String): p.Term.Select =
    p.Term.Select(named("this", ptr(owner)), List(p.PathStep.Deref, p.PathStep.Field(name)), i32)

  private def mul(x: p.Term, y: p.Term): p.Expr = p.Expr.IntrOp(p.Intr.Mul(x, y, i32))
  private def ret(e: p.Expr): p.Stmt            = p.Stmt.Return(e)
  private val unit                              = p.Expr.Alias(p.Term.Unit0Const)

  private val k = named("k")

  private val areaAbstract = method(shape, "area", List(p.Arg(k)), i32, List(ret(p.Expr.Alias(p.Term.Poison(i32)))))
  private val areaCircle = method(
    circle,
    "area",
    List(p.Arg(k)),
    i32,
    List(
      p.Stmt.Var(named("rr"), Some(mul(field(circle, "r"), field(circle, "r")))),
      p.Stmt.Var(named("a"), Some(mul(selectT("rr"), selectT(k)))),
      ret(select("a"))
    )
  )
  private val areaRect = method(
    rect,
    "area",
    List(p.Arg(k)),
    i32,
    List(
      p.Stmt.Var(named("a"), Some(mul(field(rect, "w"), field(rect, "h")))),
      p.Stmt.Var(named("b"), Some(mul(selectT("a"), selectT(k)))),
      ret(select("b"))
    )
  )
  private val growAbstract = method(shape, "grow", List(p.Arg(k)), p.Type.Unit0, List(ret(unit)))
  private val growRect = method(
    rect,
    "grow",
    List(p.Arg(k)),
    p.Type.Unit0,
    List(
      p.Stmt.Var(named("w"), Some(p.Expr.IntrOp(p.Intr.Add(field(rect, "w"), selectT(k), i32)))),
      p.Stmt.Mut(field(rect, "w"), select("w")),
      ret(unit)
    )
  )
  private val growCircle = method(circle, "grow", List(p.Arg(k)), p.Type.Unit0, List(ret(unit)))

  private val receiver = named("s", ptr(shape))
  private val out      = named("out", ptr(i32))

  private val table = Map(
    sym("Shape", "area") -> List(
      Devirtualise.Candidate(circle, sym("Circle", "area"), 0),
      Devirtualise.Candidate(rect, sym("Rect", "area"), 1)
    ),
    sym("Shape", "grow") -> List(
      Devirtualise.Candidate(circle, sym("Circle", "grow"), 0),
      Devirtualise.Candidate(rect, sym("Rect", "grow"), 1)
    )
  )

  private def invoke(name: String, rtn: p.Type): p.Expr.Invoke =
    p.Expr.Invoke(p.Type.FnRef(sym("Shape", name)), Nil, Some(selectT(receiver)), List(selectT(k)), rtn)

  private def program(body: List[p.Stmt]): p.Program =
    PassTest.program(
      entry(args = List(p.Arg(out), p.Arg(receiver), p.Arg(k)), body = body :+ ret(unit)),
      List(areaAbstract, areaCircle, areaRect, growAbstract, growCircle, growRect),
      defs
    )

  private val areaBody = List(
    p.Stmt.Var(named("r"), Some(invoke("area", i32))),
    p.Stmt.Update(selectT(out), p.Term.IntS32Const(0), selectT("r"))
  )

  private def lowered(in: p.Program): p.Program =
    StructuredExit(FnInline(Devirtualise(table)(in, NoopLog), NoopLog), NoopLog)

  private case class Run(out: Long, code: Int, message: String, rect: (Long, Long))

  private def run(in: p.Program, tag: Int, scale: Long): Run = {
    val prog = lowered(in)
    val vm   = Interpreter.Vm(prog)
    val dst  = vm.alloc(4L)
    val obj  = vm.alloc(12L)
    val err  = vm.alloc(4L + p.Conventions.assertMessageLimit)
    vm.store(dst, V.I(-1L), i32)
    vm.store(obj, V.I(tag.toLong), i32)
    vm.store(obj + 4, V.I(3L), i32)
    vm.store(obj + 8, V.I(5L), i32)
    val args = prog.entry.required.args.map { a =>
      a.named.symbol match {
        case p.Conventions.ErrorArg => errT        -> V.I(err)
        case "out"                  => a.named.tpe -> V.I(dst)
        case "s"                    => a.named.tpe -> V.I(obj)
        case "k"                    => i32         -> V.I(scale)
        case other                  => fail(s"unexpected argument $other")
      }
    }
    vm.call(p.Conventions.EntryName, args)
    def i32At(a: Long): Long = vm.load(a, i32) match { case V.I(v) => v; case _ => fail("non-integral") }
    val message = String(
      Iterator
        .from(4)
        .map(o => vm.load(err + o, p.Type.IntU8) match { case V.I(v) => v.toInt; case _ => 0 })
        .takeWhile(_ != 0)
        .map(_.toByte)
        .toArray,
      java.nio.charset.StandardCharsets.UTF_8
    )
    Run(i32At(dst), i32At(err).toInt, message, (i32At(obj + 4), i32At(obj + 8)))
  }

  test("a call dispatches on the receiver tag") {
    val in = program(areaBody)
    assertEquals(run(in, 0, 2).out, 18L)
    assertEquals(run(in, 1, 2).out, 30L)
  }

  test("a tag outside the candidate set raises and no arm runs") {
    val r = run(program(areaBody), 7, 2)
    assertEquals(r.out, -1L)
    assertEquals(r.code, p.Enums.AssertCode.Error.value)
    assert(r.message.contains("Shape.area"), r.message)
  }

  test("a returned call dispatches") {
    val caller = fn(
      "callArea",
      List(p.Arg(receiver), p.Arg(k)),
      i32,
      List(ret(invoke("area", i32))),
      visibility = p.Function.Visibility.Internal
    )
    val in = program(
      List(
        p.Stmt.Var(
          named("r"),
          Some(p.Expr.Invoke(p.Type.FnRef(sym("callArea")), Nil, None, List(selectT(receiver), selectT(k)), i32))
        ),
        p.Stmt.Update(selectT(out), p.Term.IntS32Const(0), selectT("r"))
      )
    )
    val withCaller = in.copy(functions = caller :: in.functions)
    assertEquals(run(withCaller, 0, 2).out, 18L)
    assertEquals(run(withCaller, 1, 2).out, 30L)
    assertEquals(run(withCaller, 5, 2).code, p.Enums.AssertCode.Error.value)
  }

  test("a unit call dispatches and runs the selected arm only") {
    val grow = program(List(p.Stmt.Var(named("u", p.Type.Unit0), Some(invoke("grow", p.Type.Unit0)))))
    assertEquals(run(grow, 1, 4).rect, (7L, 5L))
    assertEquals(run(grow, 0, 4).rect, (3L, 5L))
    assertEquals(run(grow, 3, 4).code, p.Enums.AssertCode.Error.value)
  }

  test("a candidate receives only the arguments it selects") {
    val j            = named("j")
    val pickAbstract = method(shape, "pick", List(p.Arg(k), p.Arg(j)), i32, List(ret(p.Expr.Alias(p.Term.Poison(i32)))))
    val pickCircle = method(
      circle,
      "pick",
      List(p.Arg(j)),
      i32,
      List(
        p.Stmt.Var(named("a"), Some(p.Expr.IntrOp(p.Intr.Add(field(circle, "r"), selectT(j), i32)))),
        ret(select("a"))
      )
    )
    val pickRect = method(rect, "pick", List(p.Arg(k), p.Arg(j)), i32, List(ret(mul(selectT(k), selectT(j)))))
    val picks = Map(
      sym("Shape", "pick") -> List(
        Devirtualise.Candidate(circle, sym("Circle", "pick"), 0, Some(List(1))),
        Devirtualise.Candidate(rect, sym("Rect", "pick"), 1)
      )
    )
    val call = p.Expr.Invoke(
      p.Type.FnRef(sym("Shape", "pick")),
      Nil,
      Some(selectT(receiver)),
      List(selectT(k), p.Term.IntS32Const(100)),
      i32
    )
    val in = PassTest.program(
      entry(
        args = List(p.Arg(out), p.Arg(receiver), p.Arg(k)),
        body = List(
          p.Stmt.Var(named("r"), Some(call)),
          p.Stmt.Update(selectT(out), p.Term.IntS32Const(0), selectT("r")),
          ret(unit)
        )
      ),
      List(pickAbstract, pickCircle, pickRect),
      defs
    )
    val prog = StructuredExit(FnInline(Devirtualise(picks)(in, NoopLog), NoopLog), NoopLog)
    val vm   = Interpreter.Vm(prog)
    def run(tag: Long): Long = {
      val dst = vm.alloc(4L)
      val obj = vm.alloc(12L)
      val err = vm.alloc(4L + p.Conventions.assertMessageLimit)
      vm.store(obj, V.I(tag), i32)
      vm.store(obj + 4, V.I(3L), i32)
      vm.store(obj + 8, V.I(5L), i32)
      val args = prog.entry.required.args.map { a =>
        a.named.symbol match {
          case p.Conventions.ErrorArg => errT        -> V.I(err)
          case "out"                  => a.named.tpe -> V.I(dst)
          case "s"                    => a.named.tpe -> V.I(obj)
          case "k"                    => i32         -> V.I(2L)
          case other                  => fail(s"unexpected argument $other")
        }
      }
      vm.call(p.Conventions.EntryName, args)
      vm.load(dst, i32) match { case V.I(v) => v; case _ => fail("non-integral") }
    }
    assertEquals(run(0), 103L)
    assertEquals(run(1), 200L)
  }

  test("an abstract struct is sized to hold its widest candidate and each candidate leads with the tag") {
    val plain = List(
      p.StructDef(shape.name, Nil, Nil, Nil),
      p.StructDef(circle.name, Nil, List(named("r", p.Type.Float64)), List(shape)),
      p.StructDef(rect.name, Nil, List(named("w"), named("h")), List(shape))
    )
    val tags = Devirtualise.tagsOf(List(rect.name, circle.name))
    assertEquals(tags, Map(circle.name -> 0, rect.name -> 1))
    val out = Devirtualise
      .decorate(plain, Map(shape.name -> List(circle.name, rect.name)), tags)
      .map(d => d.name -> d.members)
      .toMap
    val tag = named(p.Conventions.DispatchTag)
    assertEquals(
      out(shape.name),
      List(tag, named("#slack"), named("#pad", p.Type.Arr(p.Type.IntS64, 2, p.Type.Space.Global)))
    )
    assertEquals(out(circle.name), List(tag, named("r", p.Type.Float64)))
    assertEquals(out(rect.name), List(tag, named("w"), named("h")))
  }

  test("calls outside the table are untouched") {
    val in = program(areaBody)
    assertEquals(Devirtualise(Map.empty)(in, NoopLog), in)
    assertEquals(Devirtualise(Map(sym("Other", "area") -> Nil))(in, NoopLog), in)
  }

  test("the rewritten program verifies") {
    val result = Verify(Devirtualise(table)(program(areaBody), NoopLog), NoopLog, verifyFunction = true)
    assertEquals(result.flatMap(_._2), Nil)
  }
}
