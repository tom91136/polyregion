package polyregion.ast.pass

import polyregion.ast.{PolyAST as p, *, given}
import polyregion.ast.Traversal.*
import PassTest.*

class SpecialisationSuite extends munit.FunSuite {

  // Spec: each generic function (one with non-empty tpeVars) should be cloned per unique
  // applied type at the call sites; the original generic should be replaced/removed; remaining
  // call sites should reference the specialised functions with concrete types and no tpeArgs.

  test("non-generic program is unchanged") {
    val helper = fn("h", args = List(arg("a")), rtn = p.Type.IntS32, body = List(p.Stmt.Return(select("a"))))
    val prog   = program(entry(), List(helper))
    val out    = Specialisation(prog, NoopLog)
    assertEquals(out.functions.map(_.name), List(helper.name))
  }

  test("generic function called with one concrete type produces a specialised function") {
    val tArg = arg("a", p.Type.Var("T"))
    val generic = fn(
      "id",
      args = List(tArg),
      rtn = p.Type.Var("T"),
      body = List(p.Stmt.Return(select(tArg.named))),
      tpeVars = List("T")
    )
    val callSite =
      p.Expr.Invoke(p.Type.FnRef(generic.name), List(p.Type.IntS32), None, List(p.Term.IntS32Const(1)), p.Type.IntS32)
    val e   = entry(body = List(p.Stmt.Var(named("r"), Some(callSite)), p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))))
    val out = Specialisation(program(e, List(generic)), NoopLog)

    val genericLeft = out.functions.exists(f => f.name == generic.name && f.tpeVars.nonEmpty)
    assert(!genericLeft, s"generic should be removed; remaining: ${out.functions.map(f => f.name -> f.tpeVars)}")

    val specialised = out.functions.filter(_.tpeVars.isEmpty)
    assert(specialised.nonEmpty, "expected at least one specialised function with no tpeVars")
  }

  test("infers generic arguments omitted after an explicit prefix") {
    val generic = fn(
      "partial",
      args = List(arg("value", p.Type.Var("U"))),
      rtn = p.Type.Var("U"),
      body = List(p.Stmt.Return(select("value"))),
      tpeVars = List("T", "U")
    )
    val call = p.Expr.Invoke(
      p.Type.FnRef(generic.name),
      List(p.Type.IntS32),
      None,
      List(p.Term.Float32Const(1.0f)),
      p.Type.Float32
    )
    val out = Specialisation(
      program(entry(body = List(p.Stmt.Return(call))), List(generic)),
      NoopLog
    )

    assertEquals(out.functions.map(_.tpeVars), List(Nil))
    assert(out.functions.head.collectAll[p.Type].contains(p.Type.Float32))
    assertEquals(out.entry.collectAll[p.Expr].collect { case invoke: p.Expr.Invoke => invoke.tpeArgs }, List(Nil))
  }

  test("generic overloads retain every specialised signature") {
    val name         = sym("overloaded.constructor")
    val tpe          = p.Type.Var("T")
    val self         = p.Type.Ptr(p.Type.Struct(sym("box"), List(tpe)), p.Type.Space.Global)
    val defaultCtor  = fn(name.fqcn, args = List(arg("#this", self)), tpeVars = List("T"))
    val valueCtor    = fn(name.fqcn, args = List(arg("#this", self), arg("value", tpe)), tpeVars = List("T"))
    val concreteSelf = p.Type.Ptr(p.Type.Struct(sym("box"), List(p.Type.IntS32)), p.Type.Space.Global)
    val callDefault = p.Expr.Invoke(
      p.Type.FnRef(name),
      List(p.Type.IntS32),
      None,
      List(p.Term.Poison(concreteSelf)),
      p.Type.Unit0
    )
    val callValue = p.Expr.Invoke(
      p.Type.FnRef(name),
      List(p.Type.IntS32),
      None,
      List(p.Term.Poison(concreteSelf), p.Term.IntS32Const(1)),
      p.Type.Unit0
    )
    val e = entry(body =
      List(
        p.Stmt.Var(named("default", p.Type.Unit0), Some(callDefault)),
        p.Stmt.Var(named("value", p.Type.Unit0), Some(callValue)),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )

    val out         = Specialisation(program(e, List(defaultCtor, valueCtor)), NoopLog)
    val specialised = out.functions.filter(_.tpeVars.isEmpty)
    val targets     = out.entry.collectAll[p.Expr].collect { case invoke: p.Expr.Invoke => invoke.calleeName }
    assertEquals(specialised.map(_.args.size).sorted, List(1, 2))
    assertEquals(specialised.map(_.name).distinct.size, 2)
    assertEquals(targets.toSet, specialised.map(_.name).toSet)
  }

  test("specialisation selects overloads by complete call signature") {
    val name  = sym("mixed.overload")
    val unary = fn(name.fqcn, args = List(arg("x", p.Type.Var("T"))), tpeVars = List("T"))
    val binary = fn(
      name.fqcn,
      args = List(arg("x", p.Type.Var("T")), arg("y", p.Type.Var("U"))),
      tpeVars = List("T", "U")
    )
    val call = p.Expr.Invoke(
      p.Type.FnRef(name),
      List(p.Type.IntS32),
      None,
      List(p.Term.IntS32Const(1)),
      p.Type.Unit0
    )
    val e   = entry(body = List(p.Stmt.Return(call)))
    val out = Specialisation(program(e, List(binary, unary)), NoopLog)

    val specialised = out.functions.filter(_.tpeVars.isEmpty)
    assertEquals(specialised.map(_.args.size), List(1))
    assertEquals(specialised.flatMap(_.collectAll[p.Type].collect { case x: p.Type.Var => x }), Nil)
  }

  test("generic remote launch produces and references a specialised function") {
    val generic = fn(
      "generic.kernel",
      args = List(arg("value", p.Type.Var("T"))),
      tpeVars = List("T")
    )
    val one     = p.Term.IntU32Const(1)
    val context = p.Term.NullPtrConst(p.Type.IntU8, p.Type.Space.Global, p.Region.Opaque)
    val launch = p.Spec.RemoteLaunch(
      context = context,
      kernel = p.Term.Poison(p.Type.FnRef(generic.name)),
      tpeArgs = List(p.Type.IntS32),
      gridX = one,
      gridY = one,
      gridZ = one,
      blockX = one,
      blockY = one,
      blockZ = one,
      shmem = p.Term.IntU32Const(0),
      args = List(p.Term.IntS32Const(1))
    )
    val e = entry(
      body = List(
        p.Stmt.Var(named("launch", p.Type.Unit0), Some(p.Expr.SpecOp(launch))),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )

    val out         = Specialisation(program(e, List(generic)), NoopLog)
    val specialised = out.functions.filter(_.tpeVars.isEmpty)
    val launches    = out.entry.collectAll[p.Expr].collect { case p.Expr.SpecOp(x: p.Spec.RemoteLaunch) => x }

    assertEquals(specialised.size, 1)
    assertEquals(launches.size, 1)
    assertEquals(launches.head.kernel.tpe, p.Type.FnRef(specialised.head.name))
    assertEquals(launches.head.tpeArgs, Nil)
  }

  test("remote launch infers missing generic arguments from operands") {
    val generic = fn(
      "inferred.kernel",
      args = List(arg("value", p.Type.Var("T"))),
      tpeVars = List("T")
    )
    val one     = p.Term.IntU32Const(1)
    val context = p.Term.NullPtrConst(p.Type.IntU8, p.Type.Space.Global, p.Region.Opaque)
    val launch = p.Spec.RemoteLaunch(
      context = context,
      kernel = p.Term.Poison(p.Type.FnRef(generic.name)),
      tpeArgs = Nil,
      gridX = one,
      gridY = one,
      gridZ = one,
      blockX = one,
      blockY = one,
      blockZ = one,
      shmem = p.Term.IntU32Const(0),
      args = List(p.Term.IntS32Const(1))
    )
    val e = entry(body =
      List(
        p.Stmt.Var(named("launch", p.Type.Unit0), Some(p.Expr.SpecOp(launch))),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )

    val out      = Specialisation(program(e, List(generic)), NoopLog)
    val expected = Specialisation.monomorphicName(generic.name, List(p.Type.IntS32))
    val launches = out.entry.collectAll[p.Expr].collect { case p.Expr.SpecOp(x: p.Spec.RemoteLaunch) => x }

    assert(out.functions.exists(_.name == expected), out.functions.map(_.name).mkString(", "))
    assertEquals(launches.map(_.kernel.tpe), List(p.Type.FnRef(expected)))
    assertEquals(launches.map(_.tpeArgs), List(Nil))
  }

  test("remote launch normalizes an address-of struct argument after resolving its generic kernel") {
    val element     = p.Type.Var("T")
    val boxName     = sym("launch.Box")
    val box         = p.Type.Struct(boxName, List(element))
    val generic     = fn("addressed.kernel", args = List(arg("value", box)), tpeVars = List("T"))
    val concreteBox = p.Type.Struct(boxName, List(p.Type.IntS32))
    val boxPointer  = p.Type.Ptr(concreteBox, p.Type.Space.Global)
    val storage     = named("storage", boxPointer)
    val one         = p.Term.IntU32Const(1)
    val context     = p.Term.NullPtrConst(p.Type.IntU8, p.Type.Space.Global, p.Region.Opaque)
    val launch = p.Spec.RemoteLaunch(
      context = context,
      kernel = p.Term.Poison(p.Type.FnRef(generic.name)),
      tpeArgs = Nil,
      gridX = one,
      gridY = one,
      gridZ = one,
      blockX = one,
      blockY = one,
      blockZ = one,
      shmem = p.Term.IntU32Const(0),
      args = List(p.Term.Select(storage, Nil, boxPointer))
    )
    val e = entry(
      args = List(p.Arg(storage)),
      body = List(
        p.Stmt.Var(named("launch", p.Type.Unit0), Some(p.Expr.SpecOp(launch))),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )

    val out      = Specialisation(program(e, List(generic)), NoopLog)
    val expected = Specialisation.monomorphicName(generic.name, List(p.Type.IntS32))
    val launches = out.entry.collectAll[p.Expr].collect { case p.Expr.SpecOp(value: p.Spec.RemoteLaunch) => value }

    assert(out.functions.exists(_.name == expected), out.functions.map(_.name).mkString(", "))
    assertEquals(launches.map(_.kernel.tpe), List(p.Type.FnRef(expected)))
    assertEquals(launches.flatMap(_.args).map(_.tpe), List(concreteBox))
    assertEquals(
      launches.flatMap(_.args).collect { case select: p.Term.Select => select.steps },
      List(List(p.PathStep.Deref))
    )
  }

  test("specialises a generic launch reached through an internal monomorphic helper") {
    val element = p.Type.Var("T")
    val kernel  = fn("nested.kernel", args = List(arg("value", element)), tpeVars = List("T"))
    val one     = p.Term.IntU32Const(1)
    val context = p.Term.NullPtrConst(p.Type.IntU8, p.Type.Space.Global, p.Region.Opaque)
    val helper = fn(
      "nested.helper",
      body = List(
        p.Stmt.Var(
          named("launch", p.Type.Unit0),
          Some(
            p.Expr.SpecOp(
              p.Spec.RemoteLaunch(
                context,
                p.Term.Poison(p.Type.FnRef(kernel.name)),
                List(p.Type.IntS32),
                one,
                one,
                one,
                one,
                one,
                one,
                p.Term.IntU32Const(0),
                List(p.Term.IntS32Const(1))
              )
            )
          )
        ),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )
    val invoke = p.Expr.Invoke(p.Type.FnRef(helper.name), Nil, None, Nil, p.Type.Unit0)
    val e = entry(
      body = List(
        p.Stmt.Var(named("invoke", p.Type.Unit0), Some(invoke)),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )

    val out      = Specialisation(program(e, List(helper, kernel)), NoopLog)
    val expected = Specialisation.monomorphicName(kernel.name, List(p.Type.IntS32))
    val launches = out.functions
      .find(_.name == helper.name)
      .toList
      .flatMap(_.collectAll[p.Expr].collect { case p.Expr.SpecOp(value: p.Spec.RemoteLaunch) => value })

    assert(out.functions.exists(_.name == expected), out.functions.map(_.name).mkString(", "))
    assertEquals(launches.map(_.kernel.tpe), List(p.Type.FnRef(expected)))
    assertEquals(launches.map(_.tpeArgs), List(Nil))
  }

  test("materialises an implicit scalar conversion for a generic remote launch") {
    val element = p.Type.Var("T")
    val kernel = fn(
      "converted.kernel",
      args = List(arg("value", element), arg("offset", p.Type.IntS64)),
      tpeVars = List("T")
    )
    val one     = p.Term.IntU32Const(1)
    val context = p.Term.NullPtrConst(p.Type.IntU8, p.Type.Space.Global, p.Region.Opaque)
    val launch = p.Spec.RemoteLaunch(
      context,
      p.Term.Poison(p.Type.FnRef(kernel.name)),
      Nil,
      one,
      one,
      one,
      one,
      one,
      one,
      p.Term.IntU32Const(0),
      List(p.Term.IntS32Const(1), p.Term.IntS32Const(2))
    )
    val e = entry(
      body = List(
        p.Stmt.Var(named("launch", p.Type.Unit0), Some(p.Expr.SpecOp(launch))),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )

    val out        = Specialisation(program(e, List(kernel)), NoopLog)
    val statements = out.entry.toList.flatMap(_.body)
    val casts      = statements.collect { case p.Stmt.Var(_, Some(cast: p.Expr.Cast), _) => cast }
    val rewritten  = out.entry.collectAll[p.Expr].collect { case p.Expr.SpecOp(value: p.Spec.RemoteLaunch) => value }

    assertEquals(casts.map(_.tpe), List(p.Type.IntS64))
    assertEquals(rewritten.map(_.args.map(_.tpe)), List(List(p.Type.IntS32, p.Type.IntS64)))
    assertEquals(rewritten.map(_.tpeArgs), List(Nil))
  }

  test("materialises a scalar conversion for a non-generic remote launch in exception cleanup") {
    val kernel  = fn("cleanup.converted.kernel", args = List(arg("offset", p.Type.IntS64)))
    val one     = p.Term.IntU32Const(1)
    val context = p.Term.NullPtrConst(p.Type.IntU8, p.Type.Space.Global, p.Region.Opaque)
    val launch = p.Spec.RemoteLaunch(
      context,
      p.Term.Poison(p.Type.FnRef(kernel.name)),
      Nil,
      one,
      one,
      one,
      one,
      one,
      one,
      p.Term.IntU32Const(0),
      List(p.Term.IntS32Const(2))
    )
    val cleanup = List(p.Stmt.Var(named("cleanupLaunch", p.Type.Unit0), Some(p.Expr.SpecOp(launch))))
    val e = entry(
      body = List(
        raise(p.Term.IntS32Const(1), "int", cleanup),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )

    val out        = Specialisation(program(e, List(kernel)), NoopLog)
    val statements = out.entry.toList.flatMap(_.collectWhere[p.Stmt] { case statement => statement })
    val casts      = statements.collect { case p.Stmt.Var(_, Some(cast: p.Expr.Cast), _) => cast }
    val rewritten  = out.entry.collectWhere[p.Expr] { case p.Expr.SpecOp(value: p.Spec.RemoteLaunch) => value }

    assertEquals(casts.map(_.tpe), List(p.Type.IntS64))
    assertEquals(rewritten.map(_.args.map(_.tpe)), List(List(p.Type.IntS64)))
    assertEquals(Verify(out, NoopLog, verifyFunction = true).flatMap(_._2), Nil)
  }

  test("remote launch accepts a by-value capture for a pointer-shaped generic kernel argument") {
    val element     = p.Type.Var("T")
    val captureName = sym("launch.Capture")
    val capture     = p.Type.Struct(captureName, List(element))
    val generic = fn(
      "capture.kernel",
      args = List(arg("capture", p.Type.Ptr(capture, p.Type.Space.Global))),
      tpeVars = List("T")
    )
    val concreteCapture = p.Type.Struct(captureName, List(p.Type.IntS32))
    val one             = p.Term.IntU32Const(1)
    val context         = p.Term.NullPtrConst(p.Type.IntU8, p.Type.Space.Global, p.Region.Opaque)
    val launch = p.Spec.RemoteLaunch(
      context = context,
      kernel = p.Term.Poison(p.Type.FnRef(generic.name)),
      tpeArgs = List(p.Type.IntS32),
      gridX = one,
      gridY = one,
      gridZ = one,
      blockX = one,
      blockY = one,
      blockZ = one,
      shmem = p.Term.IntU32Const(0),
      args = List(p.Term.Poison(concreteCapture))
    )
    val e = entry(body =
      List(
        p.Stmt.Var(named("launch", p.Type.Unit0), Some(p.Expr.SpecOp(launch))),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )

    val out         = Specialisation(program(e, List(generic)), NoopLog)
    val expected    = Specialisation.monomorphicName(generic.name, List(p.Type.IntS32))
    val rewritten   = out.entry.collectAll[p.Expr].collect { case p.Expr.SpecOp(value: p.Spec.RemoteLaunch) => value }
    val specialised = out.functions.find(_.name == expected)

    assert(specialised.nonEmpty, out.functions.map(_.name).mkString(", "))
    assertEquals(rewritten.map(_.kernel.tpe), List(p.Type.FnRef(expected)))
    assertEquals(rewritten.flatMap(_.args).map(_.tpe), List(concreteCapture))
  }

  test("remote launch completes partial type arguments from captures and ordinary arguments") {
    val captureName = sym("partial.capture.Box")
    val generic = fn(
      "partial.capture.kernel",
      args = List(arg("value", p.Type.Var("U"))),
      tpeVars = List("T", "U"),
      moduleCaptures = List(arg("capture", p.Type.Struct(captureName, List(p.Type.Var("T")))))
    )
    val one     = p.Term.IntU32Const(1)
    val context = p.Term.NullPtrConst(p.Type.IntU8, p.Type.Space.Global, p.Region.Opaque)
    val launch = p.Spec.RemoteLaunch(
      context = context,
      kernel = p.Term.Poison(p.Type.FnRef(generic.name)),
      tpeArgs = List(p.Type.IntS32),
      gridX = one,
      gridY = one,
      gridZ = one,
      blockX = one,
      blockY = one,
      blockZ = one,
      shmem = p.Term.IntU32Const(0),
      args = List(
        p.Term.Poison(p.Type.Struct(captureName, List(p.Type.IntS32))),
        p.Term.Float32Const(1.0f)
      )
    )
    val out = Specialisation(
      program(
        entry(body = List(p.Stmt.Return(p.Expr.SpecOp(launch)))),
        List(generic)
      ),
      NoopLog
    )
    val expected = Specialisation.monomorphicName(generic.name, List(p.Type.IntS32, p.Type.Float32))

    assert(out.functions.exists(_.name == expected), out.functions.map(_.name).mkString(", "))
    assertEquals(
      out.entry.collectAll[p.Expr].collect { case p.Expr.SpecOp(value: p.Spec.RemoteLaunch) => value.kernel.tpe },
      List(p.Type.FnRef(expected))
    )
  }

  test("materialises an erased callable carried in the remote capture ABI") {
    val callback = fn("captured.callable.callback")
    val erased   = p.Type.Ptr(p.Type.Nothing, p.Type.Space.Global)
    val callable = arg("callable", erased)
    val one      = p.Term.IntU32Const(1)
    val context  = p.Term.NullPtrConst(p.Type.IntU8, p.Type.Space.Global, p.Region.Opaque)
    val launcher = fn(
      "captured.callable.launcher",
      args = List(arg("value", p.Type.Var("T"))),
      tpeVars = List("T", "Callable"),
      termCaptures = List(callable),
      body = List(
        p.Stmt.Var(
          named("nested", p.Type.Unit0),
          Some(
            p.Expr.SpecOp(
              p.Spec.RemoteLaunch(
                context,
                selectT(callable.named),
                Nil,
                one,
                one,
                one,
                one,
                one,
                one,
                p.Term.IntU32Const(0),
                Nil
              )
            )
          )
        ),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )
    val launch = p.Spec.RemoteLaunch(
      context,
      p.Term.Poison(p.Type.FnRef(launcher.name)),
      List(p.Type.IntS32, p.Type.FnRef(callback.name)),
      one,
      one,
      one,
      one,
      one,
      one,
      p.Term.IntU32Const(0),
      List(p.Term.Poison(p.Type.FnRef(callback.name)), p.Term.IntS32Const(1))
    )
    val out = Specialisation(
      program(entry(body = List(p.Stmt.Return(p.Expr.SpecOp(launch)))), List(launcher, callback)),
      NoopLog
    )
    val specialised = out.functions
      .find(_.name == Specialisation.monomorphicName(launcher.name, List(p.Type.IntS32, p.Type.FnRef(callback.name))))
      .getOrElse(fail("missing specialised captured-callable launcher"))

    assertEquals(
      specialised.collectAll[p.Expr].collect { case p.Expr.SpecOp(value: p.Spec.RemoteLaunch) => value.kernel.tpe },
      List(p.Type.FnRef(callback.name))
    )
  }

  test("remote launch inference selects the matching overloaded kernel") {
    val name = sym("overloaded.kernel")
    val unary = fn(
      name.fqcn,
      args = List(arg("value", p.Type.Var("T"))),
      tpeVars = List("T")
    )
    val binary = fn(
      name.fqcn,
      args = List(arg("left", p.Type.Var("T")), arg("right", p.Type.Var("T"))),
      tpeVars = List("T")
    )
    val one     = p.Term.IntU32Const(1)
    val context = p.Term.NullPtrConst(p.Type.IntU8, p.Type.Space.Global, p.Region.Opaque)
    val launch = p.Spec.RemoteLaunch(
      context = context,
      kernel = p.Term.Poison(p.Type.FnRef(name)),
      tpeArgs = Nil,
      gridX = one,
      gridY = one,
      gridZ = one,
      blockX = one,
      blockY = one,
      blockZ = one,
      shmem = p.Term.IntU32Const(0),
      args = List(p.Term.IntS32Const(1))
    )
    val e = entry(body =
      List(
        p.Stmt.Var(named("launch", p.Type.Unit0), Some(p.Expr.SpecOp(launch))),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )

    val out      = Specialisation(program(e, List(binary, unary)), NoopLog)
    val retained = out.functions.filter(_.tpeVars.isEmpty)
    val launches = out.entry.collectAll[p.Expr].collect { case p.Expr.SpecOp(x: p.Spec.RemoteLaunch) => x }

    assertEquals(retained.map(_.args.size), List(1))
    assertEquals(launches.map(_.kernel.tpe), retained.map(fn => p.Type.FnRef(fn.name)))
  }

  test("nested generic launchers retain the inferred specialised kernel") {
    val valueT  = p.Type.Var("T")
    val kernel  = fn("nested.kernel", args = List(arg("value", valueT)), tpeVars = List("T"))
    val one     = p.Term.IntU32Const(1)
    val context = p.Term.NullPtrConst(p.Type.IntU8, p.Type.Space.Global, p.Region.Opaque)
    val launcher = fn(
      "nested.launcher",
      args = List(arg("value", valueT)),
      tpeVars = List("T"),
      body = List(
        p.Stmt.Var(
          named("launch", p.Type.Unit0),
          Some(
            p.Expr.SpecOp(
              p.Spec.RemoteLaunch(
                context = context,
                kernel = p.Term.Poison(p.Type.FnRef(kernel.name)),
                tpeArgs = Nil,
                gridX = one,
                gridY = one,
                gridZ = one,
                blockX = one,
                blockY = one,
                blockZ = one,
                shmem = p.Term.IntU32Const(0),
                args = List(selectT("value", valueT))
              )
            )
          )
        ),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )
    val call = p.Expr.Invoke(
      p.Type.FnRef(launcher.name),
      List(p.Type.IntS32),
      None,
      List(p.Term.IntS32Const(1)),
      p.Type.Unit0
    )
    val e = entry(body = List(p.Stmt.Return(call)))

    val out              = Specialisation(program(e, List(launcher, kernel)), NoopLog)
    val expectedKernel   = Specialisation.monomorphicName(kernel.name, List(p.Type.IntS32))
    val expectedLauncher = Specialisation.monomorphicName(launcher.name, List(p.Type.IntS32))
    val specialisedLauncher =
      out.functions.find(_.name == expectedLauncher).getOrElse(fail("missing specialised launcher"))
    val launches = specialisedLauncher.collectAll[p.Expr].collect { case p.Expr.SpecOp(x: p.Spec.RemoteLaunch) => x }

    assert(out.functions.exists(_.name == expectedKernel), out.functions.map(_.name).mkString(", "))
    assertEquals(launches.map(_.kernel.tpe), List(p.Type.FnRef(expectedKernel)))
  }

  test("generic remote launch in a retained helper produces its specialised function") {
    val generic = fn(
      "helper.generic.kernel",
      args = List(arg("value", p.Type.Var("T"))),
      tpeVars = List("T")
    )
    val one     = p.Term.IntU32Const(1)
    val context = p.Term.NullPtrConst(p.Type.IntU8, p.Type.Space.Global, p.Region.Opaque)
    val launch = p.Spec.RemoteLaunch(
      context = context,
      kernel = p.Term.Poison(p.Type.FnRef(generic.name)),
      tpeArgs = List(p.Type.IntS32),
      gridX = one,
      gridY = one,
      gridZ = one,
      blockX = one,
      blockY = one,
      blockZ = one,
      shmem = p.Term.IntU32Const(0),
      args = List(p.Term.IntS32Const(1))
    )
    val helper = fn(
      "helper",
      body = List(
        p.Stmt.Var(named("launch", p.Type.Unit0), Some(p.Expr.SpecOp(launch))),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )
    val helperCall = p.Expr.Invoke(p.Type.FnRef(helper.name), Nil, None, Nil, p.Type.Unit0)
    val e = entry(
      body = List(
        p.Stmt.Var(named("helperCall", p.Type.Unit0), Some(helperCall)),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )

    val out          = Specialisation(program(e, List(helper, generic)), NoopLog)
    val expectedName = Specialisation.monomorphicName(generic.name, List(p.Type.IntS32))

    assert(out.functions.exists(_.name == expectedName), out.functions.map(_.name).mkString(", "))
  }

  test("callable type arguments become concrete before their invocation is discovered") {
    val callback = fn(
      "callback",
      args = List(arg("value", p.Type.IntS32)),
      rtn = p.Type.IntS32,
      body = List(p.Stmt.Return(select("value")))
    )
    val generic = fn(
      "apply",
      args = List(arg("value", p.Type.IntS32)),
      rtn = p.Type.IntS32,
      body = List(
        p.Stmt.Return(
          p.Expr.Invoke(
            p.Type.Var("Callable"),
            Nil,
            None,
            List(selectT("value", p.Type.IntS32)),
            p.Type.IntS32
          )
        )
      ),
      tpeVars = List("Callable")
    )
    val call = p.Expr.Invoke(
      p.Type.FnRef(generic.name),
      List(p.Type.FnRef(callback.name)),
      None,
      List(p.Term.IntS32Const(1)),
      p.Type.IntS32
    )
    val e = entry(body = List(p.Stmt.Return(call)))

    val out = Specialisation(program(e, List(generic, callback)), NoopLog)
    val specialised =
      out.functions.find(_.name == Specialisation.monomorphicName(generic.name, List(p.Type.FnRef(callback.name))))
    val invokes = specialised.toList.flatMap(_.collectAll[p.Expr].collect { case x: p.Expr.Invoke => x })

    assertEquals(invokes.map(_.callee), List(p.Type.FnRef(callback.name)))
  }

  test("generic member functions take type arguments from pointer receivers") {
    val boxName   = sym("Box")
    val boxT      = p.Type.Struct(boxName, List(p.Type.Var("T")))
    val boxInt    = p.Type.Struct(boxName, List(p.Type.IntS32))
    val boxTPtr   = p.Type.Ptr(boxT, p.Type.Space.Private)
    val boxIntPtr = p.Type.Ptr(boxInt, p.Type.Space.Private)
    val member = fn(
      "Box.apply",
      args = List(arg("value", p.Type.Var("T"))),
      rtn = p.Type.Var("T"),
      body = List(p.Stmt.Return(select("value", p.Type.Var("T")))),
      tpeVars = List("T")
    ).modifyDecl(_.copy(receiver = Some(arg("self", boxTPtr))))
    val call = p.Expr.Invoke(
      p.Type.FnRef(member.name),
      Nil,
      Some(p.Term.Poison(boxIntPtr)),
      List(p.Term.IntS32Const(1)),
      p.Type.IntS32
    )
    val e   = entry(body = List(p.Stmt.Return(call)))
    val out = Specialisation(program(e, List(member)), NoopLog)

    val expected = Specialisation.monomorphicName(member.name, List(p.Type.IntS32))
    assert(out.functions.exists(_.name == expected), out.functions.map(_.name).mkString(", "))
    val invoke = out.entry.collectAll[p.Expr].collectFirst { case x: p.Expr.Invoke => x }.get
    assertEquals(invoke.callee, p.Type.FnRef(expected))
  }

  test("pointer receiver arguments combine with compiler-supplied method type arguments") {
    val boxName   = sym("Box")
    val boxT      = p.Type.Struct(boxName, List(p.Type.Var("T")))
    val boxInt    = p.Type.Struct(boxName, List(p.Type.IntS32))
    val boxTPtr   = p.Type.Ptr(boxT, p.Type.Space.Private)
    val boxIntPtr = p.Type.Ptr(boxInt, p.Type.Space.Private)
    val member = fn(
      "Box.convert",
      args = List(arg("value", p.Type.Var("U"))),
      rtn = p.Type.Var("U"),
      body = List(p.Stmt.Return(select("value", p.Type.Var("U")))),
      tpeVars = List("T", "U")
    ).modifyDecl(_.copy(receiver = Some(arg("self", boxTPtr))))
    val call = p.Expr.Invoke(
      p.Type.FnRef(member.name),
      List(p.Type.Float32),
      Some(p.Term.Poison(boxIntPtr)),
      List(p.Term.Float32Const(1.5f)),
      p.Type.Float32
    )
    val out = Specialisation(program(entry(body = List(p.Stmt.Return(call))), List(member)), NoopLog)

    val expected = Specialisation.monomorphicName(member.name, List(p.Type.IntS32, p.Type.Float32))
    assert(out.functions.exists(_.name == expected), out.functions.map(_.name).mkString(", "))
  }

  test("infers method type arguments after receiver and explicit prefixes") {
    val boxName   = sym("Box")
    val boxT      = p.Type.Struct(boxName, List(p.Type.Var("T")))
    val boxInt    = p.Type.Struct(boxName, List(p.Type.IntS32))
    val boxTPtr   = p.Type.Ptr(boxT, p.Type.Space.Private)
    val boxIntPtr = p.Type.Ptr(boxInt, p.Type.Space.Private)
    val member = fn(
      "Box.invoke",
      args = List(arg("value", p.Type.Var("V"))),
      rtn = p.Type.Var("V"),
      body = List(p.Stmt.Return(select("value", p.Type.Var("V")))),
      tpeVars = List("T", "U", "V")
    ).modifyDecl(_.copy(receiver = Some(arg("self", boxTPtr))))
    val call = p.Expr.Invoke(
      p.Type.FnRef(member.name),
      List(p.Type.Float32),
      Some(p.Term.Poison(boxIntPtr)),
      List(p.Term.IntS64Const(1)),
      p.Type.IntS64
    )
    val out = Specialisation(program(entry(body = List(p.Stmt.Return(call))), List(member)), NoopLog)

    val expected = Specialisation.monomorphicName(member.name, List(p.Type.IntS32, p.Type.Float32, p.Type.IntS64))
    assert(out.functions.exists(_.name == expected), out.functions.map(_.name).mkString(", "))
    assertEquals(
      out.entry.collectAll[p.Expr].collectFirst { case invoke: p.Expr.Invoke => invoke.callee },
      Some(p.Type.FnRef(expected))
    )
  }

  test("matches an erased dependent function pointer with its concrete callable") {
    val callback = fn(
      "erased.callback",
      rtn = p.Type.IntS32,
      body = List(p.Stmt.Return(p.Expr.Alias(p.Term.IntS32Const(1))))
    )
    val erased = p.Type.Ptr(p.Type.Nothing, p.Type.Space.Global)
    val generic = fn(
      "erased.invoke",
      args = List(arg("value", p.Type.Var("T")), arg("callback", erased)),
      rtn = p.Type.IntS32,
      body = List(p.Stmt.Return(p.Expr.Alias(p.Term.IntS32Const(1)))),
      tpeVars = List("T", "Callable")
    )
    val call = p.Expr.Invoke(
      p.Type.FnRef(generic.name),
      List(p.Type.IntS32, p.Type.FnRef(callback.name)),
      None,
      List(p.Term.IntS32Const(1), p.Term.Poison(p.Type.FnRef(callback.name))),
      p.Type.IntS32
    )
    val out = Specialisation(program(entry(body = List(p.Stmt.Return(call))), List(generic, callback)), NoopLog)

    val expected = Specialisation.monomorphicName(generic.name, List(p.Type.IntS32, p.Type.FnRef(callback.name)))
    assert(out.functions.exists(_.name == expected), out.functions.map(_.name).mkString(", "))
    assertEquals(
      out.entry.collectAll[p.Expr].collectFirst { case invoke: p.Expr.Invoke => invoke.callee },
      Some(p.Type.FnRef(expected))
    )
  }

  test("materialises an erased launch parameter as its concrete callable") {
    val callback    = fn("erased.launch.callback")
    val erased      = p.Type.Ptr(p.Type.Nothing, p.Type.Space.Global)
    val callableArg = arg("callback", erased)
    val one         = p.Term.IntU32Const(1)
    val context     = p.Term.NullPtrConst(p.Type.IntU8, p.Type.Space.Global, p.Region.Opaque)
    val generic = fn(
      "erased.launch",
      args = List(callableArg),
      body = List(
        p.Stmt.Var(
          named("launch", p.Type.Unit0),
          Some(
            p.Expr.SpecOp(
              p.Spec.RemoteLaunch(
                context,
                selectT(callableArg.named),
                Nil,
                one,
                one,
                one,
                one,
                one,
                one,
                p.Term.IntU32Const(0),
                Nil
              )
            )
          )
        ),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      ),
      tpeVars = List("Callable")
    )
    val call = p.Expr.Invoke(
      p.Type.FnRef(generic.name),
      List(p.Type.FnRef(callback.name)),
      None,
      List(p.Term.Poison(p.Type.FnRef(callback.name))),
      p.Type.Unit0
    )
    val out = Specialisation(program(entry(body = List(p.Stmt.Return(call))), List(generic, callback)), NoopLog)

    val launch = out.functions
      .find(_.name == Specialisation.monomorphicName(generic.name, List(p.Type.FnRef(callback.name))))
      .toList
      .flatMap(_.collectAll[p.Expr].collect { case p.Expr.SpecOp(x: p.Spec.RemoteLaunch) => x })
    assertEquals(launch.map(_.kernel.tpe), List(p.Type.FnRef(callback.name)))
  }

  test("specialises a non-generic erased launcher for each concrete callable") {
    val longSuffix  = "vendor_component" * 128
    val first       = fn(s"erased.launch.first.$longSuffix")
    val second      = fn(s"erased.launch.second.$longSuffix")
    val erased      = p.Type.Ptr(p.Type.Nothing, p.Type.Space.Global)
    val callableArg = arg("callback", erased)
    val one         = p.Term.IntU32Const(1)
    val context     = p.Term.NullPtrConst(p.Type.IntU8, p.Type.Space.Global, p.Region.Opaque)
    val launcher = fn(
      "erased.launch.forward",
      args = List(callableArg),
      body = List(
        p.Stmt.Var(
          named("launch", p.Type.Unit0),
          Some(
            p.Expr.SpecOp(
              p.Spec.RemoteLaunch(
                context,
                selectT(callableArg.named),
                Nil,
                one,
                one,
                one,
                one,
                one,
                one,
                p.Term.IntU32Const(0),
                Nil
              )
            )
          )
        ),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )
    def invoke(target: p.Function) =
      p.Expr.Invoke(
        p.Type.FnRef(launcher.name),
        Nil,
        None,
        List(p.Term.Poison(p.Type.FnRef(target.name))),
        p.Type.Unit0
      )
    val e = entry(
      body = List(
        p.Stmt.Var(named("first", p.Type.Unit0), Some(invoke(first))),
        p.Stmt.Var(named("second", p.Type.Unit0), Some(invoke(second))),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )

    val specialised = Specialisation(program(e, List(launcher, first, second)), NoopLog)
    val invoked = specialised.entry.collectAll[p.Expr].collect { case invoke: p.Expr.Invoke => invoke.calleeName }.toSet
    val targets = specialised.functions
      .filter(function => invoked(function.name))
      .flatMap(_.collectAll[p.Expr].collect { case p.Expr.SpecOp(value: p.Spec.RemoteLaunch) => value.kernel.tpe })

    assertEquals(targets.toSet, Set(p.Type.FnRef(first.name), p.Type.FnRef(second.name)))
    assertEquals(invoked.size, 2)
    assert(invoked.forall(_.fqcn.length < 128), invoked)
    val repeated = Specialisation(program(e, List(launcher, first, second)), NoopLog)
    assertEquals(
      repeated.entry.collectAll[p.Expr].collect { case invoke: p.Expr.Invoke => invoke.calleeName }.toSet,
      invoked
    )
  }

  test("preserves enclosing type arguments for a materialised generic launch target") {
    val callback    = fn("erased.generic.callback")
    val kernelName  = sym("erased.generic.kernel")
    val erased      = p.Type.Ptr(p.Type.Nothing, p.Type.Space.Global)
    val callableArg = arg("kernel", erased)
    val one         = p.Term.IntU32Const(1)
    val context     = p.Term.NullPtrConst(p.Type.IntU8, p.Type.Space.Global, p.Region.Opaque)
    val launcher = fn(
      "erased.generic.launcher",
      args = List(callableArg),
      body = List(
        p.Stmt.Var(
          named("launch", p.Type.Unit0),
          Some(
            p.Expr.SpecOp(
              p.Spec.RemoteLaunch(
                context,
                selectT(callableArg.named),
                Nil,
                one,
                one,
                one,
                one,
                one,
                one,
                p.Term.IntU32Const(0),
                Nil
              )
            )
          )
        ),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      ),
      tpeVars = List("T", "Callable")
    )
    val typeArgs = List(p.Type.IntS32, p.Type.FnRef(callback.name))
    val call = p.Expr.Invoke(
      p.Type.FnRef(launcher.name),
      typeArgs,
      None,
      List(p.Term.Poison(p.Type.FnRef(kernelName))),
      p.Type.Unit0
    )
    val out = Specialisation(program(entry(body = List(p.Stmt.Return(call))), List(launcher, callback)), NoopLog)
    val launcherName = out.entry
      .collectAll[p.Expr]
      .collectFirst { case invoke: p.Expr.Invoke => invoke.calleeName }
      .getOrElse(fail("missing specialised launcher call"))
    val specialisedLauncher = out.functions
      .find(_.name == launcherName)
      .getOrElse(fail("missing specialised launcher"))
    val launches = specialisedLauncher.collectAll[p.Expr].collect { case p.Expr.SpecOp(value: p.Spec.RemoteLaunch) =>
      value
    }

    assertEquals(launches.map(_.kernel.tpe), List(p.Type.FnRef(kernelName)))
    assertEquals(launches.map(_.tpeArgs), List(typeArgs))
  }

  test("resolves an immutable erased launch-target alias to its concrete callable") {
    val kernel      = fn("erased.alias.kernel")
    val erased      = p.Type.Ptr(p.Type.Nothing, p.Type.Space.Global)
    val callableArg = arg("kernel", erased)
    val local       = named("kernel.local", erased)
    val one         = p.Term.IntU32Const(1)
    val context     = p.Term.NullPtrConst(p.Type.IntU8, p.Type.Space.Global, p.Region.Opaque)
    val launcher = fn(
      "erased.alias.launcher",
      args = List(callableArg),
      body = List(
        p.Stmt.Var(local, Some(p.Expr.Alias(selectT(callableArg.named))), isMutable = false),
        p.Stmt.Var(
          named("launch", p.Type.Unit0),
          Some(
            p.Expr.SpecOp(
              p.Spec.RemoteLaunch(
                context,
                selectT(local),
                Nil,
                one,
                one,
                one,
                one,
                one,
                one,
                p.Term.IntU32Const(0),
                Nil
              )
            )
          )
        ),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )
    val call = p.Expr.Invoke(
      p.Type.FnRef(launcher.name),
      Nil,
      None,
      List(p.Term.Poison(p.Type.FnRef(kernel.name))),
      p.Type.Unit0
    )

    val out = Specialisation(program(entry(body = List(p.Stmt.Return(call))), List(launcher, kernel)), NoopLog)
    val launcherName = out.entry
      .collectAll[p.Expr]
      .collectFirst { case invoke: p.Expr.Invoke => invoke.calleeName }
      .getOrElse(fail("missing specialised launcher call"))
    val target = out.functions
      .find(_.name == launcherName)
      .toList
      .flatMap(_.collectAll[p.Expr].collect { case p.Expr.SpecOp(value: p.Spec.RemoteLaunch) => value.kernel.tpe })

    assertEquals(target, List(p.Type.FnRef(kernel.name)))
  }

  test("captured generic calls match the pre-capture-patching invocation shape") {
    val capture = arg("captured", p.Type.IntS32)
    val generic = fn(
      "captured.generic",
      args = List(arg("value", p.Type.Var("T"))),
      tpeVars = List("T"),
      moduleCaptures = List(capture)
    )
    val call = p.Expr.Invoke(
      p.Type.FnRef(generic.name),
      List(p.Type.IntS32),
      None,
      List(p.Term.IntS32Const(1)),
      p.Type.Unit0
    )

    val out      = Specialisation(program(entry(body = List(p.Stmt.Return(call))), List(generic)), NoopLog)
    val expected = Specialisation.monomorphicName(generic.name, List(p.Type.IntS32))

    assert(out.functions.exists(_.name == expected), out.functions.map(_.name).mkString(", "))
    assertEquals(
      out.entry.collectAll[p.Expr].collect { case invoke: p.Expr.Invoke => invoke.callee },
      List(p.Type.FnRef(expected))
    )
  }

  test("does not substitute type variables shadowed by callable binders") {
    val callable = p.Type.Exec(List(p.Type.Var("T")), List(p.Type.Var("T")), p.Type.Var("T"))
    val generic = fn(
      "callable.shadow",
      args = List(arg("callback", callable), arg("value", p.Type.Var("T"))),
      tpeVars = List("T")
    )
    val call = p.Expr.Invoke(
      p.Type.FnRef(generic.name),
      List(p.Type.IntS32),
      None,
      List(p.Term.Poison(callable), p.Term.IntS32Const(1)),
      p.Type.Unit0
    )

    val out = Specialisation(program(entry(body = List(p.Stmt.Return(call))), List(generic)), NoopLog)
    val specialised = out.functions
      .find(_.name == Specialisation.monomorphicName(generic.name, List(p.Type.IntS32)))
      .getOrElse(fail(s"missing specialised callable helper: ${out.functions.map(_.repr).mkString("; ")}"))

    assertEquals(specialised.args.map(_.named.tpe), List(callable, p.Type.IntS32))
  }

  test("rejects polymorphic recursion with a bounded diagnostic") {
    val boxName = sym("Box")
    val t       = p.Type.Var("T")
    val nested  = p.Type.Struct(boxName, List(t))
    val generic = fn(
      "recursive",
      args = List(arg("value", t)),
      tpeVars = List("T"),
      body = List(
        p.Stmt.Return(
          p.Expr.Invoke(
            p.Type.FnRef(sym("recursive")),
            List(nested),
            None,
            List(p.Term.Poison(nested)),
            p.Type.Unit0
          )
        )
      )
    )
    val call = p.Expr.Invoke(
      p.Type.FnRef(generic.name),
      List(p.Type.IntS32),
      None,
      List(p.Term.IntS32Const(1)),
      p.Type.Unit0
    )

    val error = intercept[IllegalStateException] {
      Specialisation(program(entry(body = List(p.Stmt.Return(call))), List(generic)), NoopLog)
    }

    assert(error.getMessage.contains("polymorphic recursion"))
  }

  test("same-name generic overload delegation is not polymorphic recursion") {
    val name = sym("generic.delegation")
    val leaf = fn(
      name.fqcn,
      args = List(arg("left", p.Type.Var("U")), arg("right", p.Type.Var("U"))),
      tpeVars = List("U")
    )
    val delegating = fn(
      name.fqcn,
      args = List(arg("value", p.Type.Var("T"))),
      tpeVars = List("T"),
      body = List(
        p.Stmt.Return(
          p.Expr.Invoke(
            p.Type.FnRef(name),
            List(p.Type.Float32),
            None,
            List(p.Term.Float32Const(1.0f), p.Term.Float32Const(2.0f)),
            p.Type.Unit0
          )
        )
      )
    )
    val invoke = p.Expr.Invoke(
      p.Type.FnRef(name),
      List(p.Type.IntS32),
      None,
      List(p.Term.IntS32Const(1)),
      p.Type.Unit0
    )

    val out = Specialisation(program(entry(body = List(p.Stmt.Return(invoke))), List(leaf, delegating)), NoopLog)

    assertEquals(out.functions.map(_.name).distinct.size, 2)
    assertEquals(out.functions.map(_.args.size).sorted, List(1, 2))
  }

  test("type-changing recursion may stabilise at an existing specialisation") {
    val name     = sym("stable.recursion")
    val boxedInt = p.Type.Struct(sym("StableBox"), List(p.Type.IntS32))
    val generic = fn(
      name.fqcn,
      args = List(arg("value", p.Type.Var("T"))),
      tpeVars = List("T"),
      body = List(
        p.Stmt.Return(
          p.Expr.Invoke(
            p.Type.FnRef(name),
            List(boxedInt),
            None,
            List(p.Term.Poison(boxedInt)),
            p.Type.Unit0
          )
        )
      )
    )
    val invoke = p.Expr.Invoke(
      p.Type.FnRef(name),
      List(p.Type.IntS32),
      None,
      List(p.Term.IntS32Const(1)),
      p.Type.Unit0
    )

    val out = Specialisation(program(entry(body = List(p.Stmt.Return(invoke))), List(generic)), NoopLog)

    assertEquals(out.functions.size, 2)
    assertEquals(out.functions.map(_.name).distinct.size, 2)
  }

  test("an exact remote overload outranks a scalar-convertible overload") {
    val name = sym("scalar.overload.kernel")
    def kernel(argumentType: p.Type, marker: Int) = fn(
      name.fqcn,
      args = List(arg("value", argumentType)),
      tpeVars = List("Unused"),
      body = List(
        p.Stmt.Var(named("marker"), Some(p.Expr.Alias(p.Term.IntS32Const(marker))), isMutable = false),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )
    val converted = kernel(p.Type.IntS64, 1)
    val exact     = kernel(p.Type.IntS32, 2)
    val one       = p.Term.IntU32Const(1)
    val launch = p.Spec.RemoteLaunch(
      p.Term.NullPtrConst(p.Type.IntU8, p.Type.Space.Global, p.Region.Opaque),
      p.Term.Poison(p.Type.FnRef(name)),
      List(p.Type.IntS32),
      one,
      one,
      one,
      one,
      one,
      one,
      p.Term.IntU32Const(0),
      List(p.Term.IntS32Const(7))
    )

    val out =
      Specialisation(program(entry(body = List(p.Stmt.Return(p.Expr.SpecOp(launch)))), List(converted, exact)), NoopLog)
    val target = out.entry
      .collectWhere[p.Expr] { case p.Expr.SpecOp(value: p.Spec.RemoteLaunch) => value.kernel.tpe }
      .collectFirst { case p.Type.FnRef(value) => value }
      .getOrElse(fail("missing specialised launch target"))
    val selected = out.functions.find(_.name == target).getOrElse(fail("missing selected kernel"))

    assertEquals(selected.collectWhere[p.Term] { case value: p.Term.IntS32Const => value.value }, List(2))
  }
}
