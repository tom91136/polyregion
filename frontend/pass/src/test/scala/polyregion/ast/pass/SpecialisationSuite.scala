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
}
