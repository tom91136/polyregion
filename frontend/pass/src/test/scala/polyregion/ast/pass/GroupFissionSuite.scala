package polyregion.ast.pass

import polyregion.ast.{PolyAST as p, *, given}
import polyregion.ast.Interpreter
import polyregion.ast.Interpreter.V
import polyregion.ast.Traversal.*
import PassTest.*

class GroupFissionSuite extends munit.FunSuite {

  private val i32   = p.Type.IntS32
  private val u32   = p.Type.IntU32
  private val bool  = p.Type.Bool1
  private val g     = p.Type.Space.Global
  private val local = p.Type.Space.Local
  private val out   = named("out", p.Type.Ptr(i32, g))
  private val flag  = named("flag", p.Type.Ptr(i32, g))
  private val outT  = selectT(out)

  private def i(v: Int): p.Term = p.Term.IntS32Const(v)
  private def u(v: Int): p.Term = p.Term.IntU32Const(v)
  private def ret               = p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
  private def alias(t: p.Term)  = p.Expr.Alias(t)
  private def spec(op: p.Spec)  = p.Expr.SpecOp(op)
  private def let(n: String, e: p.Expr, t: p.Type = i32, mut: Boolean = false) =
    p.Stmt.Var(named(n, t), Some(e), mut)
  private def lt(a: p.Term, b: p.Term)  = p.Expr.IntrOp(p.Intr.LogicLt(a, b))
  private def eq(a: p.Term, b: p.Term)  = p.Expr.IntrOp(p.Intr.LogicEq(a, b))
  private def neq(a: p.Term, b: p.Term) = p.Expr.IntrOp(p.Intr.LogicNeq(a, b))
  private def add(a: p.Term, b: p.Term) = p.Expr.IntrOp(p.Intr.Add(a, b, i32))
  private def lid                       = spec(p.Spec.GpuLocalIdx(u(0)))
  private def gid                       = spec(p.Spec.GpuGlobalIdx(u(0)))
  private def barrier                   = let("b", spec(p.Spec.GpuBarrierLocal), p.Type.Unit0)
  private def volatileLoad              = spec(p.Spec.GpuVolatileLoad(selectT(flag), i32))
  private def cas(expected: p.Term) =
    spec(p.Spec.GpuAtomicCAS(selectT(flag), expected, i(1), p.MemScope.Device, p.MemOrder.SeqCst, i32))
  private def rmw(op: p.AtomicOp, v: p.Term) =
    spec(p.Spec.GpuAtomicRMW(op, selectT(flag), v, p.MemScope.Device, p.MemOrder.SeqCst, i32))
  private def update(idx: p.Term, v: p.Term)       = p.Stmt.Update(outT, idx, v)
  private def localArray(n: String, len: Int = 16) = named(n, p.Type.Arr(i32, len, local))
  private def localDecl(n: String)                 = p.Stmt.Var(localArray(n), None)

  private val ptrArgs = List(p.Arg(out), p.Arg(flag))

  private def run(body: List[p.Stmt], args: List[p.Arg] = ptrArgs): p.Program =
    GroupFission(program(entry(args = args, body = body :+ ret)), NoopLog)

  private def abi(out: p.Program): Option[String] = out.meta(p.Conventions.HostAbiMeta + p.Conventions.EntryName)
  private def stmts(out: p.Program): List[p.Stmt] = out.entry.required.body.flatMap(_.collectAll[p.Stmt])
  private def localVars(out: p.Program): List[p.Named] = stmts(out).collect {
    case p.Stmt.Var(n @ p.Named(_, p.Type.Arr(_, _, `local`), _), _, _) => n
  }
  private def specs(out: p.Program): List[p.Spec] =
    out.entry.required.collectWhere[p.Expr] { case p.Expr.SpecOp(op) => op }
  private def loop(out: p.Program): p.Stmt.ForRange =
    out.entry.required.body.collectFirst { case l: p.Stmt.ForRange => l }.get
  private def waits(body: List[p.Stmt]): Boolean =
    WaitLoops.find(entry(args = ptrArgs, body = body :+ ret)).isDefined

  test("a kernel that cannot observe the group is flat and untouched") {
    val body = List(let("id", gid, u32), update(i(0), i(1)))
    val in   = program(entry(args = ptrArgs, body = body :+ ret))
    val res  = GroupFission(in, NoopLog)
    assertEquals(abi(res), Some("1"))
    assertEquals(res.entry, in.entry)
  }

  test("a local id makes a barrier-free kernel group, with a lane loop over the local size") {
    val res = run(List(let("l", lid, u32), update(i(0), i(1))))
    assertEquals(abi(res), Some("2"))
    val body = res.entry.required.body
    assert(body.exists {
      case p.Stmt.Var(n, Some(p.Expr.SpecOp(p.Spec.GpuLocalSize(_))), _) => n.symbol == "#gf_lanes"
      case _                                                             => false
    })
    val l = loop(res)
    assertEquals(l.induction.symbol, "#gf_lid")
    assertEquals(l.lbIncl, u(0))
    assertEquals(l.ubExcl, selectT("#gf_lanes", u32))
    assertEquals(l.step, u(1))
    assert(!specs(res).exists(_.isInstanceOf[p.Spec.GpuLocalIdx]))
    assertEquals(body.last, ret)
  }

  test("global ids become group * lanes + lane and other dimensions are left to the backend") {
    val res = run(List(let("a", gid, u32), let("b", spec(p.Spec.GpuGlobalIdx(u(1))), u32), let("c", lid, u32)))
    assertEquals(abi(res), Some("2"))
    val exprs = res.entry.required.collectAll[p.Expr]
    assert(exprs.contains(p.Expr.IntrOp(p.Intr.Add(selectT("#gf_base", u32), selectT("#gf_lid", u32), u32))))
    assert(exprs.contains(p.Expr.IntrOp(p.Intr.Mul(selectT("#gf_group", u32), selectT("#gf_lanes", u32), u32))))
    assert(specs(res).contains(p.Spec.GpuGlobalIdx(u(1))))
    assert(!specs(res).contains(p.Spec.GpuGlobalIdx(u(0))))
  }

  test("group and size specs are read once above the lane loops") {
    val gs = List(
      spec(p.Spec.GpuGroupIdx(u(0))),
      spec(p.Spec.GpuGroupSize(u(0))),
      spec(p.Spec.GpuLocalSize(u(0))),
      spec(p.Spec.GpuGlobalSize(u(0)))
    )
    val body = gs.zipWithIndex.map((e, k) => let(s"u$k", e, u32)) ::: List(let("l", lid, u32))
    List(body, body.take(4) ::: List(barrier, let("l", lid, u32))).foreach { b =>
      val res    = run(b)
      val inLoop = res.entry.required.body.flatMap(_.collectWhere[p.Stmt] { case l: p.Stmt.ForRange => l })
      val inside = inLoop.flatMap(_.collectWhere[p.Expr] { case p.Expr.SpecOp(op) => op })
      assert(!inside.exists(_.isInstanceOf[p.Spec.GpuGroupIdx]), inside.toString)
      assert(!inside.exists(_.isInstanceOf[p.Spec.GpuGroupSize]), inside.toString)
      assert(!inside.exists(_.isInstanceOf[p.Spec.GpuLocalSize]), inside.toString)
      assert(!inside.exists(_.isInstanceOf[p.Spec.GpuGlobalSize]), inside.toString)
      val outside = res.entry.required.body.flatMap {
        case p.Stmt.Var(_, Some(p.Expr.SpecOp(op)), _) => List(op)
        case _                                         => Nil
      }
      assertEquals(outside.count(_.isInstanceOf[p.Spec.GpuGroupIdx]), 1)
      assertEquals(outside.count(_.isInstanceOf[p.Spec.GpuGroupSize]), 1)
      assertEquals(outside.count(_.isInstanceOf[p.Spec.GpuLocalSize]), 1)
      assertEquals(outside.count(_.isInstanceOf[p.Spec.GpuGlobalSize]), 1)
    }
  }

  test("fields of captured structs are read once above the lane loops") {
    val capSym = sym("Cap")
    val cap    = named("cap", p.Type.Struct(capSym, Nil))
    val ptr    = p.Type.Ptr(i32, g)
    val defs   = List(p.StructDef(capSym, Nil, List(named("dst", ptr), named("n", i32)), Nil))
    val dst    = fieldOf(cap)("dst", ptr)
    val n      = fieldOf(cap)("n", i32)
    List(Nil, List(barrier)).foreach { sync =>
      val body = List(let("l", lid, u32), let("w", add(n, i(1)))) ::: sync ::: List(
        p.Stmt.Update(dst, selectT("l", u32), selectT("w")),
        p.Stmt.Update(dst, i(0), n)
      )
      val res = GroupFission(program(entry(args = List(p.Arg(cap)), body = body :+ ret), defs = defs), NoopLog)
      val inside = laneLoops(res.entry.required.body).flatMap(_.body.flatMap(_.collectWhere[p.Term] {
        case s: p.Term.Select => s
      }))
      assert(!inside.exists(s => s.root == cap && s.steps.nonEmpty), inside.toString)
      val reads = res.entry.required.body.collect {
        case p.Stmt.Var(_, Some(p.Expr.Alias(s: p.Term.Select)), _) if s.root == cap => s.steps
      }
      assertEquals(reads.sortBy(_.toString), List(p.PathStep.Field("dst")) :: List(p.PathStep.Field("n")) :: Nil)
    }
  }

  test("a non-literal dimension keeps the lane abi") {
    val dim = named("dim", u32)
    val in  = List(let("l", spec(p.Spec.GpuLocalIdx(selectT(dim))), u32))
    val res = run(in, ptrArgs :+ p.Arg(dim))
    assertEquals(abi(res), Some("0"))
    assertEquals(res.entry.required.body.dropRight(1), in)
  }

  test("a value-returning entry keeps the lane abi") {
    val body = List(let("l", lid, u32), p.Stmt.Return(alias(i(7))))
    val e = fn(
      p.Conventions.EntryName,
      ptrArgs,
      i32,
      body,
      convention = p.CallConvention.OffloadEntry
    )
    val res = GroupFission(program(e), NoopLog)
    assertEquals(abi(res), Some("0"))
    assertEquals(res.entry, Some(e))
  }

  test("barrier kernels are split into regions with no barrier left") {
    val res = run(List(localDecl("s"), let("l", lid, u32), barrier))
    assertEquals(abi(res), Some("2"))
    assert(!specs(res).exists(GroupVocabulary.isBarrier), specs(res).toString)
    assertEquals(localVars(res).map(_.symbol), List("s"))
    assertEquals(Verify(res, NoopLog, verifyFunction = true).flatMap(_._2), Nil)
  }

  test("local arrays are declared once above the lane loop") {
    val s = localArray("s")
    val body = List(
      localDecl("s"),
      p.Stmt.ForRange(
        named("k"),
        i(0),
        i(4),
        i(1),
        List(p.Stmt.Var(localArray("t", 8), Some(alias(p.Term.Poison(localArray("t", 8).tpe)))), update(i(0), i(2)))
      ),
      p.Stmt.Cond(p.Term.Bool1Const(true), List(p.Stmt.Var(localArray("u"), None)), Nil),
      let("l", lid, u32),
      p.Stmt.Update(selectT(s), i(0), i(1))
    )
    val res = run(body)
    assertEquals(abi(res), Some("2"))
    val top = res.entry.required.body
    assertEquals(localVars(res).map(_.symbol), List("s", "t", "u"))
    val firstLoop = top.indexWhere(_.isInstanceOf[p.Stmt.ForRange])
    assertEquals(
      top.take(firstLoop).count {
        case p.Stmt.Var(n, None, _) => n.tpe.isInstanceOf[p.Type.Arr]
        case _                      => false
      },
      3
    )
    val inner = loop(res).body.flatMap(_.collectAll[p.Stmt])
    assert(inner.forall {
      case p.Stmt.Var(n, _, _) => !n.tpe.isInstanceOf[p.Type.Arr]
      case _                   => true
    })
    assert(inner.exists {
      case p.Stmt.Mut(lhs, _) => lhs.root.symbol == "t"
      case _                  => false
    })
  }

  test("lowered entries verify") {
    val k = named("k")
    val bodies = List(
      List(localDecl("s"), let("l", lid, u32), let("g", gid, u32), p.Stmt.Update(selectT(localArray("s")), i(0), i(1))),
      List(
        let("l", lid, u32),
        p.Stmt.ForRange(
          k,
          i(0),
          i(3),
          i(1),
          List(let("hit", eq(selectT(k), i(1)), bool), p.Stmt.Cond(selectT("hit", bool), List(ret), Nil))
        ),
        update(i(0), i(1))
      )
    )
    bodies.foreach { body =>
      val res = run(body)
      assertEquals(abi(res), Some("2"))
      assertEquals(Verify(res, NoopLog, verifyFunction = true).flatMap(_._2), Nil)
    }
  }

  test("names declared more than once are made unique before hoisting") {
    val body = List(
      p.Stmt.Cond(p.Term.Bool1Const(true), List(localDecl("s"), let("x", alias(i(1)))), Nil),
      p.Stmt.Cond(p.Term.Bool1Const(false), List(localDecl("s"), let("x", alias(i(2)))), Nil),
      let("l", lid, u32)
    )
    val res    = run(body)
    val locals = localVars(res).map(_.symbol)
    val decls  = stmts(res).collect { case p.Stmt.Var(n, _, _) => n.symbol }
    assertEquals(locals.size, 2)
    assertEquals(locals.distinct.size, 2)
    assertEquals(decls.distinct.size, decls.size)
  }

  test("uniquified names keep their uses") {
    val x = named("x")
    val body = List(
      let("l", lid, u32),
      p.Stmt.Cond(p.Term.Bool1Const(true), List(p.Stmt.Var(x, Some(alias(i(5)))), update(i(0), selectT(x))), Nil),
      p.Stmt.Cond(p.Term.Bool1Const(true), List(p.Stmt.Var(x, Some(alias(i(9)))), update(i(1), selectT(x))), Nil)
    )
    val res = run(body)
    val vm  = Interpreter.Vm(res)
    val ptr = vm.alloc(16)
    vm.call(p.Conventions.EntryName, List(out.tpe -> V.I(ptr), flag.tpe -> V.I(0)))
    assertEquals(vm.load(ptr, i32), V.I(5))
    assertEquals(vm.load(ptr + 4, i32), V.I(9))
  }

  test("a parameter written by a lane is copied per lane") {
    val n = named("n", i32)
    val body = List(
      let("l", lid, u32),
      let("li", p.Expr.Cast(selectT("l", u32), i32), i32),
      let("nl", add(selectT(n), selectT("li", i32)), i32),
      p.Stmt.Mut(selectT(n), alias(selectT("nl", i32))),
      update(selectT("l", u32), selectT(n))
    )
    val res = GroupFission(program(entry(args = ptrArgs :+ p.Arg(n), body = body :+ ret)), NoopLog)
    assertEquals(abi(res), Some("2"))
    val vm = Interpreter.Vm(res)
    vm.localSize = 5
    val ptr = vm.alloc(32)
    (0 until 8).foreach(k => vm.store(ptr + 4L * k, V.I(-1), i32))
    vm.call(p.Conventions.EntryName, List(out.tpe -> V.I(ptr), flag.tpe -> V.I(0), n.tpe -> V.I(100)))
    assertEquals(
      (0 until 8).map(k => vm.load(ptr + 4L * k, i32)).toList,
      (0 until 8).map(k => V.I(if (k < 5) 100 + k else -1)).toList
    )
  }

  private def runBoth(body: List[p.Stmt]): (List[V], List[V]) = {
    def exec(prog: p.Program): List[V] = {
      val vm  = Interpreter.Vm(prog)
      val ptr = vm.alloc(16)
      (0 until 4).foreach(k => vm.store(ptr + 4L * k, V.I(-1), i32))
      vm.call(p.Conventions.EntryName, List(out.tpe -> V.I(ptr), flag.tpe -> V.I(0)))
      (0 until 4).map(k => vm.load(ptr + 4L * k, i32)).toList
    }
    val plain   = program(entry(args = ptrArgs, body = body :+ ret))
    val grouped = GroupFission(program(entry(args = ptrArgs, body = (let("l", lid, u32) :: body) :+ ret)), NoopLog)
    assertEquals(abi(grouped), Some("2"))
    (exec(plain), exec(grouped))
  }

  test("an early return is fenced by an exit flag instead of a Return") {
    val body = List(
      update(i(0), i(1)),
      p.Stmt.Cond(p.Term.Bool1Const(true), List(update(i(1), i(2)), ret), Nil),
      update(i(2), i(3))
    )
    val (plain, grouped) = runBoth(body)
    assertEquals(grouped, plain)
    assertEquals(plain, List(V.I(1), V.I(2), V.I(-1), V.I(-1)))
    val res = GroupFission(program(entry(args = ptrArgs, body = (let("l", lid, u32) :: body) :+ ret)), NoopLog)
    assertEquals(loop(res).body.flatMap(_.collectAll[p.Stmt]).count(_.isInstanceOf[p.Stmt.Return]), 0)
    assert(loop(res).body.exists {
      case p.Stmt.Var(n, _, true) => n.symbol == "#gf_exit"
      case _                      => false
    })
  }

  test("a return inside nested loops leaves every loop and skips the tail") {
    val k = named("k")
    val j = named("j")
    val body = List(
      p.Stmt.ForRange(
        k,
        i(0),
        i(3),
        i(1),
        List(
          p.Stmt.ForRange(
            j,
            i(0),
            i(3),
            i(1),
            List(
              let("sum", add(selectT(k), selectT(j))),
              update(i(0), selectT("sum")),
              let("hit", eq(selectT(k), i(1)), bool),
              p.Stmt.Cond(selectT("hit", bool), List(ret), Nil)
            )
          ),
          update(i(3), i(7))
        )
      ),
      update(i(2), i(8))
    )
    val (plain, grouped) = runBoth(body)
    assertEquals(plain, List(V.I(1), V.I(-1), V.I(-1), V.I(7)))
    assertEquals(grouped, plain)
  }

  test("an effectful trailing return expression is kept") {
    val sink = p.Expr.ForeignCall("sink", List(i(1)), p.Type.Unit0)
    val res = GroupFission(
      program(entry(args = ptrArgs, body = List(let("l", lid, u32), p.Stmt.Return(sink)))),
      NoopLog
    )
    assertEquals(abi(res), Some("2"))
    assert(loop(res).body.exists {
      case p.Stmt.Var(_, Some(`sink`), _) => true
      case _                              => false
    })
  }

  test("invariant violations name the function and the construct") {
    def failure(body: List[p.Stmt]): String =
      intercept[IllegalStateException](
        GroupFission(program(entry(args = ptrArgs, body = body :+ ret)), NoopLog)
      ).getMessage
    val call = p.Expr.Invoke(p.Type.FnRef(sym("callee")), Nil, None, Nil, i32)
    List(
      List(let("r", call))            -> "Expr.Invoke",
      List(raise(i(1), "int"))        -> "Stmt.Raise",
      List(p.Stmt.Try(Nil, Nil, Nil)) -> "Stmt.Try",
      List(p.Stmt.Rethrow)            -> "Stmt.Rethrow",
      List(
        let("a", spec(p.Spec.Assert(p.Term.IntU32Const(1), p.Term.StringConst("m"))), p.Type.Unit0)
      )                                                                      -> "Spec.Assert",
      List(let("r", spec(p.Spec.GpuGroupReduce(p.AtomicOp.Add, i(1), i32)))) -> "Spec.GpuGroupReduce",
      List(let("r", spec(p.Spec.RemoteSync(i(0)))))                          -> "Spec.RemoteSync",
      List(let("x", alias(i(0)), mut = true), p.Stmt.Mut(selectT("x"), spec(p.Spec.GpuBarrierLocal)))
        -> "a barrier outside a variable initialiser"
    ).foreach { (body, construct) =>
      val message = failure(body)
      assert(message.contains(p.Conventions.EntryName), message)
      assert(message.contains(construct), message)
    }
  }

  test("a marked entry and a non-entry program are left alone") {
    val marked = program(entry(args = ptrArgs, body = List(let("l", lid, u32), ret)))
      .copy(metadata = List(p.MetaEntry(p.Conventions.HostAbiMeta + p.Conventions.EntryName, "0")))
    assertEquals(GroupFission(marked, NoopLog), marked)
    val regular = program(fn("f", body = List(let("l", lid, u32), ret)))
    assertEquals(GroupFission(regular, NoopLog), regular)
    val twice = run(List(let("l", lid, u32)))
    assertEquals(GroupFission(twice, NoopLog), twice)
  }

  test("output is deterministic") {
    val body = List(
      localDecl("s"),
      let("l", lid, u32),
      let("g", gid, u32),
      p.Stmt.Cond(p.Term.Bool1Const(true), List(update(i(0), i(1)), ret), Nil),
      update(i(1), i(2))
    )
    def encoded(prog: p.Program): Seq[Byte] = {
      val sink = MsgPack.ArrayByteOutput()
      MsgPack.encodeTo(prog, sink)
      sink.toByteArray.toSeq
    }
    assertEquals(encoded(run(body)), encoded(run(body)))
  }

  test("is a configured pipeline step") {
    val built = PassPipelineParser.parseStep("GroupFission").flatMap(PassRegistry.build)
    assert(built.isRight, built.toString)
    assertEquals(GroupFission.phase, p.Pass.Phase.PostMono)
  }

  test("a spin on a volatile flag falls back to the lane abi with a note") {
    val v = named("v")
    val spin = p.Stmt.Annotated(
      p.Stmt.While(
        selectT("c", bool),
        List(
          p.Stmt.Var(v, Some(volatileLoad)),
          p.Stmt.Mut(selectT("c", bool), eq(selectT(v), i(0)))
        )
      ),
      Some(p.SourcePosition("spin.cpp", 12, Some(3)))
    )
    val body = List(let("l", lid, u32), let("c", alias(p.Term.Bool1Const(true)), bool, mut = true), spin)
    val in   = program(entry(args = ptrArgs, body = body :+ ret))
    val res  = GroupFission(in, NoopLog)
    assertEquals(abi(res), Some("0"))
    assertEquals(res.entry, in.entry)
    val note = res.meta(p.Conventions.HostAbiNoteMeta + p.Conventions.EntryName).get
    assert(note.contains("loop at spin.cpp:12:3"), note)
    assert(note.contains("waits on a volatile or atomic value; launching with one thread per lane"), note)
  }

  test("classifier: a loop exiting on a volatile read waits") {
    val v = named("v")
    assert(
      waits(
        List(
          p.Stmt.While(
            p.Term.Bool1Const(true),
            List(
              p.Stmt.Var(v, Some(volatileLoad)),
              let("done", neq(selectT(v), i(0)), bool),
              p.Stmt.Cond(selectT("done", bool), List(p.Stmt.Break), Nil)
            )
          )
        )
      )
    )
  }

  test("classifier: a flag set under a volatile condition waits") {
    val body = List(
      let("done", alias(p.Term.Bool1Const(false)), bool, mut = true),
      let("notDone", alias(p.Term.Bool1Const(true)), bool, mut = true),
      p.Stmt.While(
        selectT("notDone", bool),
        List(
          let("v", volatileLoad),
          let("seen", neq(selectT("v"), i(0)), bool),
          p.Stmt.Cond(
            selectT("seen", bool),
            List(
              p.Stmt.Mut(selectT("done", bool), alias(p.Term.Bool1Const(true))),
              p.Stmt.Mut(selectT("notDone", bool), alias(p.Term.Bool1Const(false)))
            ),
            Nil
          )
        )
      )
    )
    assert(waits(body))
  }

  test("classifier: a lock acquired by compare-and-swap against a constant waits, also across a barrier") {
    def lock(extra: List[p.Stmt]) = List(
      let("got", alias(p.Term.Bool1Const(true)), bool, mut = true),
      let("old", alias(i(1)), i32, mut = true),
      p.Stmt.While(
        selectT("got", bool),
        List(p.Stmt.Mut(selectT("old"), cas(i(0))), p.Stmt.Mut(selectT("got", bool), neq(selectT("old"), i(0))))
      )
    ) ::: extra
    assert(waits(lock(Nil)))
    assert(waits(lock(List(barrier, update(i(0), i(1))))))
    val res = GroupFission(program(entry(args = ptrArgs, body = lock(List(barrier)) :+ ret)), NoopLog)
    assertEquals(abi(res), Some("0"))
    assert(res.meta(p.Conventions.HostAbiNoteMeta + p.Conventions.EntryName).isDefined)
  }

  test("classifier: a compare-and-swap retry loop whose expected value is re-read does not wait") {
    val body = List(
      p.Stmt.Var(named("old"), Some(p.Expr.Index(selectT(flag), i(0), i32)), isMutable = true),
      let("again", alias(p.Term.Bool1Const(true)), bool, mut = true),
      p.Stmt.While(
        selectT("again", bool),
        List(
          p.Stmt.Var(named("prev"), Some(cas(selectT("old")))),
          let("failed", neq(selectT("prev"), selectT("old")), bool),
          p.Stmt.Mut(selectT("again", bool), alias(selectT("failed", bool))),
          p.Stmt.Mut(selectT("old"), alias(selectT("prev")))
        )
      )
    )
    assert(!waits(body))
  }

  test("classifier: atomic counters hand out work and do not wait") {
    val body = List(
      let("n", alias(i(8)), i32),
      let("i", rmw(p.AtomicOp.Add, i(1)), i32, mut = true),
      let("more", lt(selectT("i"), selectT("n")), bool, mut = true),
      p.Stmt.While(
        selectT("more", bool),
        List(
          update(i(0), selectT("i")),
          p.Stmt.Mut(selectT("i"), rmw(p.AtomicOp.Add, i(1))),
          p.Stmt.Mut(selectT("more", bool), lt(selectT("i"), selectT("n")))
        )
      )
    )
    assert(!waits(body))
  }

  test("classifier: an exchange loop waits and an unused atomic result does not") {
    val spin = List(
      let("busy", alias(p.Term.Bool1Const(true)), bool, mut = true),
      p.Stmt.While(
        selectT("busy", bool),
        List(let("old", rmw(p.AtomicOp.Xchg, i(1))), p.Stmt.Mut(selectT("busy", bool), neq(selectT("old"), i(0))))
      )
    )
    assert(waits(spin))
    val unused = List(
      p.Stmt.ForRange(
        named("k"),
        i(0),
        i(4),
        i(1),
        List(let("old", rmw(p.AtomicOp.Max, selectT("k"))), update(selectT("k"), i(1)))
      )
    )
    assert(!waits(unused))
  }

  test("classifier: a plain loop around a barrier does not wait") {
    val body = List(p.Stmt.ForRange(named("k"), i(0), i(4), i(1), List(update(selectT("k"), i(1)), barrier)))
    assert(!waits(body))
    val res = GroupFission(program(entry(args = ptrArgs, body = body :+ ret)), NoopLog)
    assertEquals(abi(res), Some("2"))
    assertEquals(res.meta(p.Conventions.HostAbiNoteMeta + p.Conventions.EntryName), None)
  }

  test("classifier: the reported position is the innermost annotated loop") {
    val v = named("v")
    val inner = p.Stmt.Annotated(
      p.Stmt.While(
        p.Term.Bool1Const(true),
        List(
          p.Stmt.Var(v, Some(volatileLoad)),
          let("done", neq(selectT(v), i(0)), bool),
          p.Stmt.Cond(selectT("done", bool), List(ret), Nil)
        )
      ),
      Some(p.SourcePosition("k.cpp", 7, None))
    )
    val outer = p.Stmt.Annotated(
      p.Stmt.ForRange(named("k"), i(0), i(2), i(1), List(inner)),
      Some(p.SourcePosition("k.cpp", 5, None))
    )
    val found = WaitLoops.find(entry(args = ptrArgs, body = List(outer, ret)))
    assertEquals(found, Some(Some(p.SourcePosition("k.cpp", 7, None))))
  }

  // barrier kernels, run one call per group by the interpreter with the work-group state from a bump allocator

  private val U32Arr                                         = p.Type.Arr(u32, 64, local)
  private val scratch                                        = named("s", U32Arr)
  private def uadd(a: p.Term, b: p.Term)                     = p.Expr.IntrOp(p.Intr.Add(a, b, u32))
  private def usub(a: p.Term, b: p.Term)                     = p.Expr.IntrOp(p.Intr.Sub(a, b, u32))
  private def urem(a: p.Term, b: p.Term)                     = p.Expr.IntrOp(p.Intr.Rem(a, b, u32))
  private def udiv(a: p.Term, b: p.Term)                     = p.Expr.IntrOp(p.Intr.Div(a, b, u32))
  private def gt(a: p.Term, b: p.Term)                       = p.Expr.IntrOp(p.Intr.LogicGt(a, b))
  private def lanesE                                         = spec(p.Spec.GpuLocalSize(u(0)))
  private def sel(n: String, t: p.Type = u32): p.Term.Select = selectT(n, t)
  private def readS(n: String, idx: p.Term)                  = let(n, p.Expr.Index(selectT(scratch), idx, u32), u32)
  private def writeS(idx: p.Term, v: p.Term)                 = p.Stmt.Update(selectT(scratch), idx, v)
  private def emit(idx: p.Term, v: p.Term): List[p.Stmt] = {
    val n = s"emit${idx.repr.hashCode.abs}${v.repr.hashCode.abs}"
    List(let(n, p.Expr.Cast(v, i32), i32), update(idx, selectT(n, i32)))
  }
  private def scratchDecl                                             = p.Stmt.Var(scratch, None)
  private def cond(c: p.Term, t: List[p.Stmt], f: List[p.Stmt] = Nil) = p.Stmt.Cond(c, t, f)
  private def ifLet(n: String, e: p.Expr)(t: p.Stmt*): List[p.Stmt] =
    List(let(n, e, bool), cond(sel(n, bool), t.toList))

  private def grouped(body: List[p.Stmt], uniform: Boolean = false): p.Program = {
    val res = GroupFission.lower(program(entry(args = ptrArgs, body = body :+ ret)), NoopLog, uniform)
    assertEquals(abi(res), Some("2"))
    assertEquals(Verify(res, NoopLog, verifyFunction = true).flatMap(_._2), Nil)
    res
  }

  private def counted(prog: p.Program): p.Program = prog.copy(entry = prog.entry.map { f =>
    f.copy(body = f.body.modifyAll[p.Stmt] {
      case p.Stmt.While(c, (v @ p.Stmt.Var(n, _, _)) :: rest) if n.symbol.startsWith("#gf_idle") =>
        val tick = p.Stmt.Var(named("#tick", p.Type.Unit0), Some(p.Expr.ForeignCall("sweep", Nil, p.Type.Unit0)))
        p.Stmt.While(c, tick :: v :: rest)
      case other => other
    })
  })

  private def runGroups(
      prog: p.Program,
      lanes: Int,
      groups: Int = 1,
      slots: Int = 64,
      init: Int => Int = _ => -1,
      extra: List[(p.Type, V)] = Nil
  ): (List[V], Long) = {
    val vm     = Interpreter.Vm(counted(prog), globalSize = lanes.toLong * groups)
    var sweeps = 0L
    vm.localSize = lanes
    vm.foreign = (name, args, _) =>
      name match {
        case p.Conventions.HostGroupState =>
          V.I(vm.alloc(args.head._2 match {
            case V.I(bytes) => bytes
            case _          => 0L
          }))
        case "sweep" =>
          sweeps += 1
          V.U
        case other => sys.error(s"unexpected foreign call $other")
      }
    val ptr = vm.alloc(4L * slots)
    (0 until slots).foreach(k => vm.store(ptr + 4L * k, V.I(init(k).toLong), i32))
    (0 until groups).foreach { g =>
      vm.groupIdx = g
      vm.call(p.Conventions.EntryName, List(out.tpe -> V.I(ptr), flag.tpe -> V.I(0)) ::: extra)
    }
    ((0 until slots).map(k => vm.load(ptr + 4L * k, i32)).toList, sweeps)
  }

  private def expect(slots: Int = 64, init: Int => Int = _ => -1)(f: PartialFunction[Int, Int]): List[V] =
    (0 until slots).map(k => V.I(f.applyOrElse(k, init).toLong)).toList

  // pc stores of region `k`, the values of `pc[lid] = v` inside its guarded lane loop
  private def transitions(prog: p.Program): Map[Int, List[Int]] = {
    def stores(xs: List[p.Stmt]): List[Int] = xs.flatMap(_.collectWhere[p.Stmt] {
      case p.Stmt.Update(p.Term.Select(pc, Nil, _), _, p.Term.IntU32Const(v)) if pc.symbol.startsWith("#gf_pc") => v
    })
    prog.entry.required.body
      .flatMap(_.collectWhere[p.Stmt] {
        case p.Stmt.ForRange(
              body = List(
                _,
                p.Stmt.Var(_, Some(p.Expr.IntrOp(p.Intr.LogicEq(_, p.Term.IntU32Const(k)))), _),
                p.Stmt.Cond(_, code, _)
              )
            ) =>
          k -> stores(code)
      })
      .toMap
  }

  private val ParkedBit = 0x80000000
  private val ExitPc    = 0x7fffffff

  private def listing3(trailing: Boolean): List[p.Stmt] = List(
    scratchDecl,
    let("l", lid, u32),
    let("g", gid, u32),
    let("a", p.Expr.Index(outT, sel("g"), i32), i32),
    let("au", p.Expr.Cast(sel("a", i32), u32), u32),
    writeS(sel("l"), sel("au")),
    barrier,
    let("bound", uadd(sel("l"), u(2)), u32),
    p.Stmt.ForRange(
      named("i", u32),
      u(0),
      sel("bound"),
      u(1),
      List(
        readS("cur", sel("l")),
        let("next", uadd(sel("cur"), sel("i")), u32),
        writeS(sel("l"), sel("next"))
      ) ::: ifLet("early", lt(sel("i"), u(2)))(let("b2", spec(p.Spec.GpuBarrierLocal), p.Type.Unit0))
    )
  ) ::: (if (trailing)
           List(
             let("b3", spec(p.Spec.GpuBarrierLocal), p.Type.Unit0),
             let("l1", uadd(sel("l"), u(1)), u32),
             let("nb", urem(sel("l1"), selectT("lanes", u32)), u32)
           )
         else List(let("nb", alias(sel("l")), u32))) ::: (readS("fin", sel("nb")) :: emit(sel("g"), sel("fin")))

  private def listing3Kernel(trailing: Boolean) =
    grouped(let("lanes", lanesE, u32) :: listing3(trailing))

  private def listing3Expected(lanes: Int, groups: Int, trailing: Boolean): List[V] = {
    def tri(l: Int)  = (1 + l) * (2 + l) / 2
    def init(k: Int) = k
    expect(init = init) {
      case k if k < lanes * groups =>
        val (g, l) = (k / lanes, k % lanes)
        val nb     = if (trailing) (l + 1) % lanes else l
        init(g * lanes + nb) + tri(nb)
    }
  }

  test("paper listing 3: lanes leave the barrier loop after different trip counts") {
    List(false, true).foreach { trailing =>
      val prog = listing3Kernel(trailing)
      List(1 -> 1, 2 -> 1, 3 -> 5, 8 -> 2, 16 -> 3).foreach { (lanes, groups) =>
        val (got, _) = runGroups(prog, lanes, groups, init = k => k)
        assertEquals(got, listing3Expected(lanes, groups, trailing), s"lanes=$lanes groups=$groups trailing=$trailing")
      }
    }
  }

  test("a tree reduction over a local array sums every lane") {
    val body = List(
      scratchDecl,
      let("l", lid, u32),
      let("v", uadd(sel("l"), u(1)), u32),
      writeS(sel("l"), sel("v")),
      barrier,
      let("half", udiv(selectT("lanes", u32), u(2)), u32),
      let("stride", alias(sel("half")), u32, mut = true),
      let("more", gt(sel("stride"), u(0)), bool, mut = true),
      p.Stmt.While(
        sel("more", bool),
        ifLet("active", lt(sel("l"), sel("stride")))(
          let("o", uadd(sel("l"), sel("stride")), u32),
          readS("x", sel("l")),
          readS("y", sel("o")),
          let("z", uadd(sel("x"), sel("y")), u32),
          writeS(sel("l"), sel("z"))
        ) ::: List(
          let("b2", spec(p.Spec.GpuBarrierLocal), p.Type.Unit0),
          let("nx", udiv(sel("stride"), u(2)), u32),
          p.Stmt.Mut(sel("stride"), alias(sel("nx"))),
          p.Stmt.Mut(sel("more", bool), gt(sel("stride"), u(0)))
        )
      )
    ) ::: ifLet("first", eq(sel("l"), u(0)))(
      (readS("total", u(0)) :: let("grp", spec(p.Spec.GpuGroupIdx(u(0))), u32) :: emit(sel("grp"), sel("total")))*
    )
    val prog = grouped(let("lanes", lanesE, u32) :: body)
    List(1, 2, 8, 64).foreach { lanes =>
      val (got, _) = runGroups(prog, lanes, groups = 3)
      assertEquals(got, expect() { case g if g < 3 => lanes * (lanes + 1) / 2 }, s"lanes=$lanes")
    }
  }

  test("an early exit before one or two barriers retires the lane") {
    val body = List(
      scratchDecl,
      let("l", lid, u32),
      writeS(sel("l"), sel("l")),
      let("lr", urem(sel("l"), u(2)), u32)
    ) ::: ifLet("odd", eq(sel("lr"), u(1)))(ret) ::: List(
      barrier,
      readS("mine", sel("l")),
      let("ten", uadd(sel("mine"), u(10)), u32),
      writeS(sel("l"), sel("ten")),
      let("b2", spec(p.Spec.GpuBarrierLocal), p.Type.Unit0),
      let("l2", uadd(sel("l"), u(2)), u32),
      let("nb", urem(sel("l2"), selectT("lanes", u32)), u32),
      readS("theirs", sel("nb"))
    ) ::: emit(sel("l"), sel("theirs"))
    val prog = grouped(let("lanes", lanesE, u32) :: body)
    List(2, 8, 10).foreach { lanes =>
      val (got, _) = runGroups(prog, lanes)
      assertEquals(got, expect() { case l if l < lanes && l % 2 == 0 => (l + 2) % lanes + 10 }, s"lanes=$lanes")
    }
  }

  test("a lane-varying continue and break of a dissolved loop meet at the next barrier") {
    val i = named("i", u32)
    val continued = List(
      scratchDecl,
      let("l", lid, u32),
      let("acc", alias(u(0)), u32, mut = true),
      p.Stmt.ForRange(
        i,
        u(0),
        u(4),
        u(1),
        List(let("li", uadd(sel("l"), selectT(i)), u32), let("par", urem(sel("li"), u(2)), u32)) :::
          ifLet("skip", eq(sel("par"), u(0)))(p.Stmt.Cont) ::: List(
            let("na", uadd(sel("acc"), selectT(i)), u32),
            p.Stmt.Mut(sel("acc"), alias(sel("na"))),
            writeS(sel("l"), sel("acc")),
            let("b2", spec(p.Spec.GpuBarrierLocal), p.Type.Unit0)
          )
      ),
      barrier,
      let("l1", uadd(sel("l"), u(1)), u32),
      let("nb", urem(sel("l1"), selectT("lanes", u32)), u32),
      readS("theirs", sel("nb")),
      let("tens", p.Expr.IntrOp(p.Intr.Mul(sel("acc"), u(10), u32)), u32),
      let("both", uadd(sel("tens"), sel("theirs")), u32)
    ) ::: emit(sel("l"), sel("both"))
    def acc(l: Int) = (0 until 4).filter(k => (l + k) % 2 == 1).sum
    val contProg    = grouped(let("lanes", lanesE, u32) :: continued)
    List(2, 5, 8).foreach { lanes =>
      val (got, _) = runGroups(contProg, lanes)
      assertEquals(
        got,
        expect() { case l if l < lanes => acc(l) * 10 + acc((l + 1) % lanes) },
        s"continue lanes=$lanes"
      )
    }

    val broken = List(
      scratchDecl,
      let("l", lid, u32),
      let("hundred", uadd(sel("l"), u(100)), u32),
      writeS(sel("l"), sel("hundred")),
      let("stop", urem(sel("l"), u(4)), u32),
      p.Stmt.ForRange(
        i,
        u(0),
        u(8),
        u(1),
        ifLet("hit", eq(selectT(i), sel("stop")))(p.Stmt.Break) ::: List(
          writeS(sel("l"), selectT(i)),
          let("b2", spec(p.Spec.GpuBarrierLocal), p.Type.Unit0)
        )
      ),
      barrier,
      let("l1", uadd(sel("l"), u(1)), u32),
      let("nb", urem(sel("l1"), selectT("lanes", u32)), u32),
      readS("theirs", sel("nb"))
    ) ::: emit(sel("l"), sel("theirs"))
    def last(l: Int) = if (l % 4 == 0) 100 + l else l % 4 - 1
    val breakProg    = grouped(let("lanes", lanesE, u32) :: broken)
    List(3, 4, 9).foreach { lanes =>
      val (got, _) = runGroups(breakProg, lanes)
      assertEquals(got, expect() { case l if l < lanes => last((l + 1) % lanes) }, s"break lanes=$lanes")
    }
  }

  test("nested conditionals hold barriers under uniform and lane-varying predicates") {
    val uniform = List(
      scratchDecl,
      let("l", lid, u32),
      writeS(sel("l"), sel("l")),
      let("wide", gt(selectT("lanes", u32), u(2)), bool),
      cond(
        sel("wide", bool),
        List(
          let("wider", gt(selectT("lanes", u32), u(4)), bool),
          cond(sel("wider", bool), List(barrier, let("l1", uadd(sel("l"), u(1)), u32)), List(barrier))
        )
      ),
      let("b2", spec(p.Spec.GpuBarrierLocal), p.Type.Unit0),
      let("nb", usub(selectT("lanes", u32), u(1)), u32),
      let("rev", usub(sel("nb"), sel("l")), u32),
      readS("theirs", sel("rev"))
    ) ::: emit(sel("l"), sel("theirs"))
    val uniformProg = grouped(let("lanes", lanesE, u32) :: uniform)
    List(2, 4, 8).foreach { lanes =>
      val (got, _) = runGroups(uniformProg, lanes)
      assertEquals(got, expect() { case l if l < lanes => lanes - 1 - l }, s"uniform lanes=$lanes")
    }

    val varying = List(
      scratchDecl,
      let("l", lid, u32),
      writeS(sel("l"), sel("l")),
      let("half", udiv(selectT("lanes", u32), u(2)), u32),
      let("low", lt(sel("l"), sel("half")), bool),
      cond(
        sel("low", bool),
        List(let("lp", urem(sel("l"), u(2)), u32), let("even", eq(sel("lp"), u(0)), bool)) :+ cond(
          sel("even", bool),
          List(
            barrier,
            let("l1", uadd(sel("l"), u(1)), u32),
            readS("x", sel("l1")),
            readS("y", sel("l")),
            let("z", uadd(sel("x"), sel("y")), u32),
            writeS(sel("l"), sel("z"))
          )
        )
      ),
      let("b2", spec(p.Spec.GpuBarrierLocal), p.Type.Unit0),
      readS("mine", sel("l"))
    ) ::: emit(sel("l"), sel("mine"))
    val varyingProg = grouped(let("lanes", lanesE, u32) :: varying)
    List(4, 8, 12).foreach { lanes =>
      val (got, _) = runGroups(varyingProg, lanes)
      assertEquals(
        got,
        expect() { case l if l < lanes => if (l < lanes / 2 && l % 2 == 0) 2 * l + 1 else l },
        s"lanes=$lanes"
      )
    }
  }

  test("a return nested in a barrier-free loop of a barrier kernel exits the lane") {
    val j = named("j", u32)
    val body = List(
      scratchDecl,
      let("l", lid, u32),
      writeS(sel("l"), sel("l")),
      let("third", urem(sel("l"), u(3)), u32),
      p.Stmt.ForRange(
        j,
        u(0),
        u(4),
        u(1),
        List(let("two", eq(selectT(j), u(2)), bool), let("zero", eq(sel("third"), u(0)), bool)) :::
          ifLet("both", p.Expr.IntrOp(p.Intr.LogicAnd(sel("two", bool), sel("zero", bool))))(ret)
      ),
      barrier,
      let("l1", uadd(sel("l"), u(1)), u32),
      let("nb", urem(sel("l1"), selectT("lanes", u32)), u32),
      readS("theirs", sel("nb"))
    ) ::: emit(sel("l"), sel("theirs"))
    val prog = grouped(let("lanes", lanesE, u32) :: body)
    assert(!prog.entry.required.body.flatMap(_.collectAll[p.Stmt]).exists {
      case p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const)) => false
      case p.Stmt.Return(_)                               => true
      case _                                              => false
    })
    val (got, _) = runGroups(prog, 7)
    assertEquals(got, expect() { case l if l < 7 && l % 3 != 0 => (l + 1) % 7 })
  }

  test("transitions: parks set bit 31, returns exit, and only loop back edges go to a lower region") {
    val prog  = listing3Kernel(trailing = true)
    val moves = transitions(prog)
    val all   = moves.values.flatten.toList
    assert(all.exists(v => (v & ParkedBit) != 0), all.toString)
    assert(all.contains(ExitPc), all.toString)
    val backwards = moves.toList.flatMap((k, vs) => vs.filter(v => (v & ParkedBit) == 0 && v != ExitPc && v <= k))
    assertEquals(backwards.distinct.size, 1, moves.toString)
    assert(moves.keySet == (0 until moves.size).toSet, moves.toString)
    assertEquals(moves(0).filter(v => (v & ParkedBit) == 0), Nil, "the entry region ends at the first barrier")
  }

  test("classification: uniform values are shared, crossing values get lane slots, local values stay") {
    val body = List(
      scratchDecl,
      let("l", lid, u32),
      let("half", udiv(selectT("lanes", u32), u(2)), u32),
      let("seed", uadd(sel("l"), u(3)), u32),
      writeS(sel("l"), sel("seed")),
      readS("mine", sel("l")),
      let("tmp", uadd(sel("l"), u(4)), u32),
      barrier,
      let("sum", uadd(sel("mine"), sel("half")), u32)
    ) ::: emit(sel("l"), sel("sum"))
    val prog  = grouped(let("lanes", lanesE, u32) :: body)
    val decls = prog.entry.required.body.flatMap(_.collectWhere[p.Stmt] { case p.Stmt.Var(n, _, _) => n.symbol })
    assert(decls.exists(_.startsWith("mine#gf_view")), decls.toString)
    assert(!decls.exists(_.startsWith("half#gf_view")), decls.toString)
    assert(!decls.exists(_.startsWith("tmp#gf_view")), decls.toString)
    assert(!decls.exists(_.startsWith("lanes#gf_view")), decls.toString)
    val shared = prog.entry.required.body.takeWhile {
      case p.Stmt.ForRange(_, _, _, _, _) => false
      case _                              => true
    }
    assert(shared.exists {
      case p.Stmt.Var(n, None, true) => n.symbol == "half"
      case _                         => false
    })
    val (got, _) = runGroups(prog, 6)
    assertEquals(got, expect() { case l if l < 6 => l + 3 + 3 })
  }

  private def declared(prog: p.Program): List[String] =
    prog.entry.required.body.flatMap(_.collectWhere[p.Stmt] { case p.Stmt.Var(n, _, _) => n.symbol })
  private def stateCalls(prog: p.Program): Int = prog.entry.required.collectAll[p.Expr].count {
    case p.Expr.ForeignCall(p.Conventions.HostGroupState, _, _) => true
    case _                                                      => false
  }

  test("classification: values that are pure in the lane index are recomputed, not spilled") {
    val body = List(
      let("l", lid, u32),
      let("shifted", uadd(sel("l"), u(3)), u32),
      let("quad", urem(sel("shifted"), u(4)), u32),
      barrier,
      let("sum", uadd(sel("quad"), sel("shifted")), u32),
      let("again", uadd(sel("sum"), sel("l")), u32)
    ) ::: emit(sel("l"), sel("again"))
    val unviewed = List("l", "shifted", "quad")
    List(false, true).foreach { uniform =>
      val prog  = grouped(let("lanes", lanesE, u32) :: body, uniform)
      val names = declared(prog)
      unviewed.foreach(n => assert(!names.exists(_.startsWith(s"$n#gf_view")), s"$n $names"))
      assert(names.exists(_.startsWith("shifted#gf_re")), names.toString)
      assert(names.exists(_.startsWith("quad#gf_re")), names.toString)
      assert(names.exists(_.startsWith("l#gf_re")), names.toString)
      assertEquals(stateCalls(prog), if (uniform) 0 else 1)
      val (got, _) = runGroups(prog, 6)
      assertEquals(got, expect() { case l if l < 6 => (l + 3) % 4 + (l + 3) + l })
    }
  }

  test("classification: a recomputed division is only by a literal, other lane-pure values are spilled") {
    val body = List(
      let("l", lid, u32),
      let("by", uadd(selectT("lanes", u32), u(1)), u32),
      let("safe", udiv(sel("l"), u(3)), u32),
      let("unsafe", udiv(sel("l"), sel("by")), u32),
      barrier,
      let("sum", uadd(sel("safe"), sel("unsafe")), u32)
    ) ::: emit(sel("l"), sel("sum"))
    val names = declared(grouped(let("lanes", lanesE, u32) :: body, uniform = true))
    assert(names.exists(_.startsWith("safe#gf_re")), names.toString)
    assert(names.exists(_.startsWith("unsafe#gf_view")), names.toString)
  }

  test("classification: a private array crossing a barrier is a per-lane aggregate") {
    val buf = named("buf", p.Type.Arr(u32, 4, p.Type.Space.Private))
    val k   = named("k", u32)
    val body = List(
      let("l", lid, u32),
      p.Stmt.Var(buf, None),
      p.Stmt.ForRange(
        k,
        u(0),
        u(4),
        u(1),
        List(
          let("tenl", p.Expr.IntrOp(p.Intr.Mul(sel("l"), u(10), u32)), u32),
          let("v", uadd(sel("tenl"), selectT(k)), u32),
          p.Stmt.Update(selectT(buf), selectT(k), sel("v"))
        )
      ),
      barrier,
      let("b0", p.Expr.Index(selectT(buf), u(0), u32), u32),
      let("b3", p.Expr.Index(selectT(buf), u(3), u32), u32),
      let("bs", uadd(sel("b0"), sel("b3")), u32)
    ) ::: emit(sel("l"), sel("bs"))
    val prog = grouped(let("lanes", lanesE, u32) :: body)
    assertEquals(prog.defs.map(_.name.repr), List("#gf_lane.buf"))
    val (got, _) = runGroups(prog, 5)
    assertEquals(got, expect() { case l if l < 5 => 20 * l + 3 })
  }

  test("classification: an address taken before a barrier points at lane storage") {
    val x  = named("x", u32)
    val px = named("px", p.Type.Ptr(u32, p.Type.Space.Private))
    val body = List(
      let("l", lid, u32),
      p.Stmt.Var(x, Some(uadd(sel("l"), u(7)))),
      p.Stmt.Var(px, Some(p.Expr.RefTo(selectT(x), None, u32, p.Type.Space.Private, p.Region.Rooted(x)))),
      barrier,
      let("v", p.Expr.Index(selectT(px), u(0), u32), u32)
    ) ::: emit(sel("l"), sel("v"))
    val prog  = grouped(body)
    val decls = prog.entry.required.body.flatMap(_.collectWhere[p.Stmt] { case p.Stmt.Var(n, _, _) => n.symbol })
    assert(decls.exists(_.startsWith("x#gf_view")), decls.toString)
    val (got, _) = runGroups(prog, 4)
    assertEquals(got, expect() { case l if l < 4 => l + 7 })
  }

  test("a parameter written by a lane is copied per lane across a barrier") {
    val n = named("n", i32)
    val body = List(
      let("l", lid, u32),
      let("li", p.Expr.Cast(sel("l"), i32), i32),
      let("nl", add(selectT(n), sel("li", i32)), i32),
      p.Stmt.Mut(selectT(n), alias(sel("nl", i32))),
      barrier,
      update(sel("l"), selectT(n))
    )
    val prog = GroupFission(program(entry(args = ptrArgs :+ p.Arg(n), body = body :+ ret)), NoopLog)
    assertEquals(abi(prog), Some("2"))
    val (got, _) = runGroups(prog, 5, extra = List(i32 -> V.I(100)))
    assertEquals(got, expect() { case l if l < 5 => 100 + l })
  }

  test("a whole-array assignment to a crossing array keeps the lane abi") {
    val a = named("a", p.Type.Arr(u32, 4, p.Type.Space.Private))
    val b = named("b", p.Type.Arr(u32, 4, p.Type.Space.Private))
    val body = List(
      let("l", lid, u32),
      p.Stmt.Var(b, None),
      p.Stmt.Var(a, Some(alias(selectT(b)))),
      barrier,
      let("v", p.Expr.Index(selectT(a), u(0), u32), u32)
    ) ::: emit(sel("l"), sel("v"))
    val in  = program(entry(args = ptrArgs, body = body :+ ret))
    val res = GroupFission(in, NoopLog)
    assertEquals(abi(res), Some("0"))
    assertEquals(res.entry, in.entry)
  }

  test("a barrier under a rare uniform predicate in a long loop completes with bounded sweeps") {
    val i = named("i", u32)
    val body = List(
      scratchDecl,
      let("l", lid, u32),
      writeS(sel("l"), u(0)),
      p.Stmt.ForRange(
        i,
        u(0),
        u(1000),
        u(1),
        List(
          readS("c", sel("l")),
          let("c1", uadd(sel("c"), u(1)), u32),
          writeS(sel("l"), sel("c1")),
          let("r", urem(selectT(i), u(250)), u32)
        ) :::
          ifLet("rare", eq(sel("r"), u(0)))(let("b2", spec(p.Spec.GpuBarrierLocal), p.Type.Unit0))
      ),
      barrier,
      let("l1", uadd(sel("l"), u(1)), u32),
      let("nb", urem(sel("l1"), selectT("lanes", u32)), u32),
      readS("theirs", sel("nb"))
    ) ::: emit(sel("l"), sel("theirs"))
    val prog         = grouped(let("lanes", lanesE, u32) :: body)
    val (got, sweep) = runGroups(prog, 4)
    assertEquals(got, expect() { case l if l < 4 => 1000 })
    // one sweep per loop iteration plus two per round (6 rounds: entry, 4 in the loop, the trailing barrier)
    assertEquals(sweep, 1012L)
  }

  test("static and dynamic local arrays are declared once above the driver") {
    val dyn = named("d", p.Type.Arr(u32, 0, local))
    val body = List(
      scratchDecl,
      let("l", lid, u32),
      writeS(sel("l"), sel("l")),
      barrier,
      p.Stmt.Var(dyn, None),
      p.Stmt.Update(selectT(dyn), sel("l"), sel("l")),
      let("b2", spec(p.Spec.GpuBarrierLocal), p.Type.Unit0),
      let("x", p.Expr.Index(selectT(dyn), u(0), u32), u32)
    ) ::: emit(sel("l"), sel("x"))
    val prog     = grouped(body)
    val prologue = prog.entry.required.body.takeWhile(!_.isInstanceOf[p.Stmt.ForRange])
    assertEquals(localVars(prog).map(_.symbol), List("s", "d"))
    assert(prologue.exists {
      case p.Stmt.Var(n, None, _) => n.symbol == "d"
      case _                      => false
    })
  }

  test("barrier kernel output is deterministic") {
    def encoded(prog: p.Program): Seq[Byte] = {
      val sink = MsgPack.ArrayByteOutput()
      MsgPack.encodeTo(prog, sink)
      sink.toByteArray.toSeq
    }
    assertEquals(encoded(listing3Kernel(trailing = true)), encoded(listing3Kernel(trailing = true)))
    val carried = List(
      scratchDecl,
      let("l", lid, u32),
      let("shifted", uadd(sel("l"), u(3)), u32),
      writeS(sel("l"), sel("shifted")),
      readS("seen", sel("l")),
      barrier,
      let("sum", uadd(sel("seen"), sel("shifted")), u32)
    ) ::: emit(sel("l"), sel("sum"))
    def lanes() = grouped(let("lanes", lanesE, u32) :: carried, uniform = true)
    assertEquals(encoded(lanes()), encoded(lanes()))
  }

  private def machine(prog: p.Program): Boolean =
    prog.entry.required.body
      .flatMap(_.collectWhere[p.Stmt] { case p.Stmt.Var(n, _, _) => n.symbol })
      .exists(_.startsWith("#gf_pc"))
  private def laneLoops(stmts: List[p.Stmt]): List[p.Stmt.ForRange] = stmts.flatMap {
    case l: p.Stmt.ForRange if l.induction.symbol == "#gf_lid" => List(l)
    case p.Stmt.ForRange(body = b)                             => laneLoops(b)
    case p.Stmt.While(_, b)                                    => laneLoops(b)
    case p.Stmt.Cond(_, t, f)                                  => laneLoops(t) ::: laneLoops(f)
    case _                                                     => Nil
  }
  private def topLevelLaneLoops(prog: p.Program): Int =
    prog.entry.required.body.count { case l: p.Stmt.ForRange => l.induction.symbol == "#gf_lid"; case _ => false }

  private trait Expected { def apply(lanes: Int, groups: Int): List[V] }

  // runs a kernel under both lowerings and checks them against the expected output
  private def agrees(body: List[p.Stmt], shapes: List[(Int, Int)], expected: Expected) = {
    val fast = grouped(let("lanes", lanesE, u32) :: body, uniform = true)
    val slow = grouped(let("lanes", lanesE, u32) :: body)
    assert(!machine(fast), "the kernel should lower to lane loops")
    assert(machine(slow))
    shapes.foreach { (lanes, groups) =>
      assertEquals(
        runGroups(fast, lanes, groups)._1,
        expected(lanes, groups),
        s"lane loops lanes=$lanes groups=$groups"
      )
      assertEquals(runGroups(slow, lanes, groups)._1, expected(lanes, groups), s"machine lanes=$lanes groups=$groups")
    }
    fast
  }

  test("uniform barriers lower to one lane loop per stretch with no region state") {
    val body = List(
      scratchDecl,
      let("l", lid, u32),
      writeS(sel("l"), sel("l")),
      barrier,
      let("nb", usub(selectT("lanes", u32), u(1)), u32),
      let("rev", usub(sel("nb"), sel("l")), u32),
      readS("theirs", sel("rev"))
    ) ::: emit(sel("l"), sel("theirs"))
    val fast = agrees(
      body,
      List(1 -> 1, 2 -> 3, 7 -> 2, 64 -> 1),
      (lanes, groups) => expect() { case l if l < lanes => lanes - 1 - l }
    )
    assertEquals(topLevelLaneLoops(fast), 2)
    val names = fast.entry.required.body.flatMap(_.collectWhere[p.Stmt] { case p.Stmt.Var(n, _, _) => n.symbol })
    assert(!names.exists(n => n.startsWith("#gf_cnt") || n.startsWith("#gf_runnable")), names.toString)
    assert(!fast.entry.required.collectAll[p.Expr].exists {
      case p.Expr.SpecOp(op) => GroupVocabulary.isBarrier(op)
      case _                 => false
    })
  }

  test("a uniform loop around a barrier runs outside the lane loops") {
    val body = List(
      scratchDecl,
      let("l", lid, u32),
      let("v", uadd(sel("l"), u(1)), u32),
      writeS(sel("l"), sel("v")),
      barrier,
      let("half", udiv(selectT("lanes", u32), u(2)), u32),
      let("stride", alias(sel("half")), u32, mut = true),
      let("more", gt(sel("stride"), u(0)), bool, mut = true),
      p.Stmt.While(
        sel("more", bool),
        ifLet("active", lt(sel("l"), sel("stride")))(
          let("o", uadd(sel("l"), sel("stride")), u32),
          readS("x", sel("l")),
          readS("y", sel("o")),
          let("z", uadd(sel("x"), sel("y")), u32),
          writeS(sel("l"), sel("z"))
        ) ::: List(
          let("b2", spec(p.Spec.GpuBarrierLocal), p.Type.Unit0),
          let("nx", udiv(sel("stride"), u(2)), u32),
          p.Stmt.Mut(sel("stride"), alias(sel("nx"))),
          p.Stmt.Mut(sel("more", bool), gt(sel("stride"), u(0)))
        )
      )
    ) ::: ifLet("first", eq(sel("l"), u(0)))(
      (readS("total", u(0)) :: let("grp", spec(p.Spec.GpuGroupIdx(u(0))), u32) :: emit(sel("grp"), sel("total")))*
    )
    val fast = agrees(
      body,
      List(1 -> 1, 2 -> 3, 8 -> 2, 64 -> 3),
      (lanes, groups) => expect() { case g if g < groups => lanes * (lanes + 1) / 2 }
    )
    val loops = fast.entry.required.body.collect { case w: p.Stmt.While => w }
    assertEquals(loops.size, 1)
    assert(laneLoops(loops.head.body).nonEmpty)
    assert(laneLoops(List(loops.head)).forall(l => !l.body.exists(_.isInstanceOf[p.Stmt.While])))
  }

  test("uniform conditionals and counted loops around barriers keep per-lane state across lane loops") {
    val i = named("i", u32)
    val body = List(
      scratchDecl,
      let("l", lid, u32),
      let("acc", alias(u(0)), u32, mut = true),
      let("wide", gt(selectT("lanes", u32), u(2)), bool),
      p.Stmt.ForRange(
        i,
        u(0),
        u(3),
        u(1),
        List(let("v", uadd(sel("l"), selectT(i)), u32), writeS(sel("l"), sel("v")), barrier) :::
          List(
            let("l1", uadd(sel("l"), u(1)), u32),
            let("nb", urem(sel("l1"), selectT("lanes", u32)), u32),
            readS("theirs", sel("nb")),
            let("na", uadd(sel("acc"), sel("theirs")), u32),
            p.Stmt.Mut(sel("acc"), alias(sel("na"))),
            let("b2", spec(p.Spec.GpuBarrierLocal), p.Type.Unit0)
          ) ::: List(cond(sel("wide", bool), List(let("b3", spec(p.Spec.GpuBarrierLocal), p.Type.Unit0))))
      )
    ) ::: emit(sel("l"), sel("acc"))
    val fast = agrees(
      body,
      List(1 -> 1, 2 -> 2, 5 -> 3, 16 -> 1),
      (lanes, groups) => expect() { case l if l < lanes => (0 until 3).map(k => (l + 1) % lanes + k).sum }
    )
    assertEquals(laneLoops(fast.entry.required.body).size, 4)
  }

  test("a lane-varying construct around a barrier keeps the region machine") {
    val varyingBound = let("lanes", lanesE, u32) :: listing3(trailing = true)
    assert(machine(GroupFission(program(entry(args = ptrArgs, body = varyingBound :+ ret)), NoopLog)))
    val early = List(
      scratchDecl,
      let("l", lid, u32),
      let("lr", urem(sel("l"), u(2)), u32)
    ) ::: ifLet("odd", eq(sel("lr"), u(1)))(ret) ::: List(barrier, writeS(sel("l"), sel("l")))
    assert(machine(GroupFission(program(entry(args = ptrArgs, body = early :+ ret)), NoopLog)))
    val i = named("i", u32)
    val broken = List(
      scratchDecl,
      let("l", lid, u32),
      p.Stmt.ForRange(
        i,
        u(0),
        u(8),
        u(1),
        ifLet("hit", eq(selectT(i), u(3)))(p.Stmt.Break) ::: List(writeS(sel("l"), selectT(i)), barrier)
      )
    )
    val prog = GroupFission(program(entry(args = ptrArgs, body = broken :+ ret)), NoopLog)
    assert(machine(prog))
    val (got, _) = runGroups(prog, 4)
    assertEquals(got, expect() { case _ => -1 })
  }

  test("a loop whose exit varies by lane does not make its variables agree") {
    val i = named("i", u32)
    val j = named("j", u32)
    val body = List(
      scratchDecl,
      let("l", lid, u32),
      let("n", alias(u(0)), u32, mut = true),
      p.Stmt.ForRange(
        j,
        u(0),
        u(6),
        u(1),
        ifLet("stop", eq(sel("l"), selectT(j)))(p.Stmt.Break) ::: List(
          let("n1", uadd(sel("n"), u(1)), u32),
          p.Stmt.Mut(sel("n"), alias(sel("n1")))
        )
      ),
      p.Stmt.ForRange(i, u(0), sel("n"), u(1), List(writeS(sel("l"), selectT(i)), barrier))
    )
    val prog = GroupFission(program(entry(args = ptrArgs, body = body :+ ret)), NoopLog)
    assert(machine(prog))
  }
}
