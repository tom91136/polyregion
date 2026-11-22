package polyregion

import polyregion.ast.{CodeGen, MsgPack, PolyAST as p}
import polyregion.jvm.compiler as ct
import polyregion.jvm.runtime as rt
import polyregion.scalalang.Platforms

import java.nio.file.Files
import java.nio.{ByteBuffer, ByteOrder}

class ContextSuite extends munit.FunSuite {

  private val PtrTpe      = p.Type.Ptr(p.Type.IntS32, p.Type.Space.Global)
  private val HostKind    = 1
  private val ObjectFmt   = 2
  private val ManagedKind = 2

  private def select(tpe: p.Type, n: String): (p.Named, p.Term.Select) = {
    val named = p.Named(n, tpe)
    named -> p.Term.Select(named, Nil, tpe)
  }

  private def fillProgram: p.Program = {
    val (out, outRef) = select(PtrTpe, "out")
    val (gid, gidRef) = select(p.Type.IntU32, "gid")
    val (idx, idxRef) = select(p.Type.IntS32, "idx")
    val body = List(
      p.Stmt.Var(gid, Some(p.Expr.SpecOp(p.Spec.GpuGlobalIdx(p.Term.IntU32Const(0)))), false),
      p.Stmt.Var(idx, Some(p.Expr.Cast(gidRef, p.Type.IntS32)), false),
      p.Stmt.Update(outRef, idxRef, idxRef),
      p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
    )
    val decl = p.FunctionDecl(
      p.Sym("fill"),
      Nil,
      None,
      List(p.Arg(out)),
      Nil,
      Nil,
      p.Type.Unit0,
      p.Function.Affinity.Offload
    )
    p.Program(
      Some(
        p.Function(
          decl,
          body,
          p.Function.Visibility.Exported,
          p.Function.FpMode.Relaxed,
          p.CallConvention.OffloadEntry
        )
      ),
      Nil,
      Nil
    )
  }

  private def assertProgram(code: p.Enums.AssertCode, message: String): p.Program = {
    val (out, outRef) = select(PtrTpe, "out")
    val body = List(
      p.Stmt.Update(outRef, p.Term.IntS32Const(0), p.Term.IntS32Const(7)),
      p.Stmt.Var(
        p.Named("trap", p.Type.Unit0),
        Some(p.Expr.SpecOp(p.Spec.Assert(p.Term.IntU32Const(code.value), p.Term.StringConst(message)))),
        false
      ),
      p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
    )
    val decl = p.FunctionDecl(
      p.Sym("trap"),
      Nil,
      None,
      List(p.Arg(out)),
      Nil,
      Nil,
      p.Type.Unit0,
      p.Function.Affinity.Offload
    )
    p.Program(
      Some(
        p.Function(
          decl,
          body,
          p.Function.Visibility.Exported,
          p.Function.FpMode.Relaxed,
          p.CallConvention.OffloadEntry
        )
      ),
      Nil,
      Nil
    )
  }

  private def probeProgram: p.Program = {
    val ctxTpe          = p.Spec.ContextType
    val outTpe          = p.Type.Ptr(p.Type.IntU64, p.Type.Space.Global)
    val (ctx, ctxRef)   = select(ctxTpe, "ctx")
    val (out, outRef)   = select(outTpe, "out")
    val (mem, memRef)   = select(ctxTpe, "mem")
    val (adr, adrRef)   = select(p.Type.IntU64, "adr")
    val (seed, seedRef) = select(p.Type.IntU64, "seed")
    val body = List(
      p.Stmt.Var(mem, Some(p.Expr.SpecOp(p.Spec.RemoteAlloc(ctxRef, p.Term.IntU64Const(64)))), false),
      p.Stmt.Var(adr, Some(p.Expr.Cast(memRef, p.Type.IntU64)), false),
      p.Stmt.Update(outRef, p.Term.IntS32Const(0), seedRef),
      p.Stmt.Update(outRef, p.Term.IntS32Const(1), adrRef),
      p.Stmt.Var(p.Named("freed", p.Type.Unit0), Some(p.Expr.SpecOp(p.Spec.RemoteFree(ctxRef, memRef))), false),
      p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
    )
    val decl = p.FunctionDecl(
      p.Sym("probe"),
      Nil,
      None,
      List(p.Arg(ctx), p.Arg(out), p.Arg(seed)),
      Nil,
      Nil,
      p.Type.Unit0,
      p.Function.Affinity.Host
    )
    p.Program(
      Some(
        p.Function(
          decl,
          body,
          p.Function.Visibility.Exported,
          p.Function.FpMode.Relaxed,
          p.CallConvention.RegularCall
        )
      ),
      Nil,
      Nil
    )
  }

  private def compile(program: p.Program, target: ct.Target, arch: String): ct.Compilation = {
    val compiler = ct.Compiler.create()
    try
      compiler.compile(
        MsgPack.encode(CodeGen.programVersioned(program)),
        false,
        ct.Options.of(target, arch),
        ct.Opt.O3.value
      )
    finally compiler.close()
  }

  private def hostFunction(
      name: String,
      params: List[p.Named],
      body: List[p.Stmt],
      visibility: p.Function.Visibility = p.Function.Visibility.Exported
  ): p.Function = {
    val decl = p.FunctionDecl(
      p.Sym(name),
      Nil,
      None,
      params.map(p.Arg(_)),
      Nil,
      Nil,
      p.Type.Unit0,
      p.Function.Affinity.Host
    )
    p.Function(
      decl,
      body :+ p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const)),
      visibility,
      p.Function.FpMode.Relaxed,
      p.CallConvention.RegularCall
    )
  }

  private def hostProgram(name: String, params: List[p.Named], body: List[p.Stmt]): p.Program =
    p.Program(Some(hostFunction(name, params, body)), Nil, Nil)

  private def usableProgram: p.Program = {
    val (ctx, mem) = (p.Named("ctx", p.Spec.ContextType), p.Named("mem", p.Spec.ContextType))
    val context    = p.Term.Select(ctx, Nil, ctx.tpe)
    hostProgram(
      "usable",
      List(ctx),
      List(
        p.Stmt.Var(mem, Some(p.Expr.SpecOp(p.Spec.RemoteAlloc(context, p.Term.IntU64Const(64))))),
        p.Stmt.Var(
          p.Named("freed", p.Type.Unit0),
          Some(p.Expr.SpecOp(p.Spec.RemoteFree(context, p.Term.Select(mem, Nil, mem.tpe))))
        )
      )
    )
  }

  private def emptyLaunch(ctx: p.Named, args: List[p.Term]): p.Stmt = {
    val one = p.Term.IntU64Const(1)
    p.Stmt.Var(
      p.Named("launched", p.Type.Unit0),
      Some(
        p.Expr.SpecOp(
          p.Spec.RemoteLaunch(
            p.Term.Select(ctx, Nil, ctx.tpe),
            p.Term.Unit0Const,
            Nil,
            one,
            one,
            one,
            one,
            one,
            one,
            p.Term.IntU64Const(0),
            args
          )
        )
      ),
      isMutable = false
    )
  }

  private def hostPointerLaunch: p.Program = {
    val (ctx, buf) = (p.Named("ctx", p.Spec.ContextType), p.Named("buf", PtrTpe))
    val owned      = p.Named("owned", p.Spec.ContextType)
    val allocation = p.Stmt.Var(
      owned,
      Some(p.Expr.SpecOp(p.Spec.RemoteAlloc(p.Term.Select(ctx, Nil, ctx.tpe), p.Term.IntU64Const(64))))
    )
    hostProgram("refuse", List(ctx, buf), List(allocation, emptyLaunch(ctx, List(p.Term.Select(buf, Nil, buf.tpe)))))
  }

  private def refusal(program: p.Program): String = {
    val images = List(program, usableProgram).map { p =>
      val image = Files.createTempFile("refusal", ".o")
      Files.write(image, compile(p, ct.Target.LLVM_HOST, "native").program)
      image
    }
    val (code, output) =
      try
        ChildJvm.run(
          "polyregion.RefusalProbe",
          Map("REFUSAL_REFUSE" -> images.head.toString, "REFUSAL_USABLE" -> images(1).toString)
        )
      finally images.foreach(Files.deleteIfExists)
    assertEquals(code, 0, output)
    output
  }

  private def hostDevice: rt.DeviceInfo = rt.Context
    .platforms()
    .flatMap(_.devices)
    .find(d => d.platformKind == HostKind && d.moduleFormat == ObjectFmt)
    .getOrElse(fail("no host object device in the catalogue"))

  private def nvidiaDevice: Option[rt.DeviceInfo] =
    rt.Context.platforms().flatMap(_.devices).find(d => d.platformKind == ManagedKind && d.cudaArchitectureMajor > 0)

  private def direct(bytes: Int): ByteBuffer = ByteBuffer.allocateDirect(bytes).order(ByteOrder.nativeOrder)

  private def withRemote[A](bytes: Int)(body: (rt.Context, Long) => A): A = {
    val ctx = rt.Context.create(hostDevice)
    try {
      val remote = ctx.malloc(bytes)
      try body(ctx, remote)
      finally ctx.free(remote)
    } finally ctx.close()
  }

  private def ptrArgs(ptr: Long): (Array[Byte], Array[Byte]) =
    (Array(rt.Type.PTR.value), ByteBuffer.allocate(8).order(ByteOrder.nativeOrder).putLong(ptr).array)

  private val ProbeName = "probe@host"

  private def ensureProbe(): Unit = synchronized {
    if (!rt.Context.hostProgramLoaded(ProbeName))
      rt.Context.loadHostProgram(ProbeName, compile(probeProgram, ct.Target.LLVM_HOST, "native").program)
  }

  private def probe(ctx: rt.Context, seed: Long): (Long, Long) = {
    val out   = direct(16)
    val types = Array(rt.Type.PTR.value, rt.Type.LONG.value)
    val data = ByteBuffer
      .allocate(16)
      .order(ByteOrder.nativeOrder)
      .putLong(Platforms.pointerOfDirectBuffer(out))
      .putLong(seed)
      .array
    ctx.callHost(Array.empty, ProbeName, "probe", types, data)
    (out.getLong(0), out.getLong(8))
  }

  test("platforms and devices are enumerated") {
    val platforms = rt.Context.platforms()
    assert(platforms.nonEmpty)
    platforms.zipWithIndex.foreach { (platform, i) =>
      assertEquals(platform.index, i)
      assert(platform.name.nonEmpty)
      platform.devices.zipWithIndex.foreach { (device, j) =>
        assertEquals(device.platform, i)
        assertEquals(device.index, j)
        assert(device.name.nonEmpty)
      }
    }
    assertEquals(hostDevice.isGpu, false)
  }

  test("a created context can be destroyed once") {
    val ctx = rt.Context.create(hostDevice)
    assertNotEquals(ctx, rt.Context.current())
    ctx.close()
    ctx.close()
  }

  test("a destroyed context rejects every operation") {
    val ctx = rt.Context.create(hostDevice)
    ctx.close()
    intercept[rt.PolyregionRuntimeException](ctx.malloc(16))
    intercept[rt.PolyregionRuntimeException](ctx.free(1))
    intercept[rt.PolyregionRuntimeException](ctx.select(Array.empty[rt.ModuleImage]))
    intercept[rt.PolyregionRuntimeException](
      ctx.launch(Array.empty, "m", "k", Array.empty, Array.empty, rt.Policy(rt.Dim3(1, 1, 1)))
    )
  }

  test("a context for a missing device is rejected") {
    intercept[rt.PolyregionRuntimeException](rt.Context.create(Int.MaxValue, 0))
    intercept[rt.PolyregionRuntimeException](rt.Context.create(hostDevice.platform, Int.MaxValue))
  }

  test("the default context is not destroyed by close") {
    val ctx = rt.Context.current()
    ctx.close()
    assertEquals(rt.Context.current(), ctx)
    val bytes = ctx.malloc(16)
    assertNotEquals(bytes, 0L)
    ctx.free(bytes)
  }

  private def checkFill(ctx: rt.Context): Unit = {
    val n             = 64
    val target        = ct.Target.LLVM_HOST
    val c             = compile(fillProgram, target, "native")
    val image         = rt.ModuleImage("fill@host", target, c.features, c.program)
    val buffer        = direct(n * 4)
    val (types, data) = ptrArgs(Platforms.pointerOfDirectBuffer(buffer))
    assertEquals(ctx.select(Array(image)), 0)
    ctx.launch(Array(image), image.name, "fill", types, data, rt.Policy(rt.Dim3(n, 1, 1)))
    (0 until n).foreach(i => assertEquals(buffer.getInt(i * 4), i))
  }

  test("a host kernel launches through a created context") {
    val ctx = rt.Context.create(hostDevice)
    try checkFill(ctx)
    finally ctx.close()
  }

  test("a host kernel launches through the default context") {
    checkFill(rt.Context.current())
  }

  private val trapCounter = java.util.concurrent.atomic.AtomicInteger()

  private def launchAssert(ctx: rt.Context, code: p.Enums.AssertCode, message: String): Unit = {
    val target        = ct.Target.LLVM_HOST
    val c             = compile(assertProgram(code, message), target, "native")
    val image         = rt.ModuleImage(s"trap${trapCounter.incrementAndGet()}@host", target, c.features, c.program)
    val buffer        = direct(4)
    val (types, data) = ptrArgs(Platforms.pointerOfDirectBuffer(buffer))
    assertEquals(ctx.select(Array(image)), 0)
    ctx.launch(Array(image), image.name, "trap", types, data, rt.Policy(rt.Dim3(1, 1, 1)))
  }

  test("a kernel that asserts advertises the error argument feature") {
    val c = compile(assertProgram(p.Enums.AssertCode.Assert, "no"), ct.Target.LLVM_HOST, "native")
    assert(c.features.contains(p.Conventions.ErrorArg), c.features.mkString(","))
    val plain = compile(fillProgram, ct.Target.LLVM_HOST, "native")
    assert(!plain.features.contains(p.Conventions.ErrorArg), plain.features.mkString(","))
  }

  test("an asserting entry takes the error block as its leading argument") {
    val c = compile(assertProgram(p.Enums.AssertCode.Assert, "no"), ct.Target.LLVM_HOST, "native")
    assertEquals(c.entryArgs.toList, List(p.Conventions.ErrorArg, "out"))
  }

  test("a kernel assert raises an AssertionError carrying the device message") {
    val ctx = rt.Context.create(hostDevice)
    try {
      val e = intercept[AssertionError](launchAssert(ctx, p.Enums.AssertCode.Assert, "device says no"))
      assertEquals(e.getMessage, "device says no")
    } finally ctx.close()
  }

  test("a kernel recursion overflow raises a runtime exception, not a VM error") {
    val ctx = rt.Context.create(hostDevice)
    try {
      val e = intercept[rt.PolyregionRuntimeException](
        launchAssert(ctx, p.Enums.AssertCode.RecursionLimit, "recursion depth exceeded 1024")
      )
      assert(e.getMessage.contains("RLIM"), e.getMessage)
      assert(e.getMessage.contains("recursion depth exceeded 1024"), e.getMessage)
    } finally ctx.close()
  }

  test("a kernel JvmException assert raises the named exception class carrying the message") {
    val ctx = rt.Context.create(hostDevice)
    try {
      val e = intercept[IllegalArgumentException](
        launchAssert(ctx, p.Enums.AssertCode.JvmException, "java.lang.IllegalArgumentException: step cannot be 0.")
      )
      assertEquals(e.getMessage, "step cannot be 0.")
      val empty = intercept[NoSuchElementException](
        launchAssert(ctx, p.Enums.AssertCode.JvmException, "java.util.NoSuchElementException: head on empty Range")
      )
      assertEquals(empty.getMessage, "head on empty Range")
    } finally ctx.close()
  }

  test("a kernel JvmException assert naming no loadable class raises a runtime exception") {
    val ctx = rt.Context.create(hostDevice)
    try {
      val e = intercept[rt.PolyregionRuntimeException](
        launchAssert(ctx, p.Enums.AssertCode.JvmException, "no.such.Failure: boom")
      )
      assert(e.getMessage.contains("no.such.Failure: boom"), e.getMessage)
    } finally ctx.close()
  }

  private def childLoaderWithFailure(): (java.net.URLClassLoader, String) = {
    val dir    = java.nio.file.Files.createTempDirectory("child-failure")
    val source = dir.resolve("childside").resolve("Outer.java")
    java.nio.file.Files.createDirectories(source.getParent)
    java.nio.file.Files.writeString(
      source,
      "package childside; public class Outer { public static class Failure extends RuntimeException { public Failure(String m) { super(m); } } }"
    )
    val compiler = javax.tools.ToolProvider.getSystemJavaCompiler
    assertEquals(compiler.run(null, null, null, "-d", dir.toString, source.toString), 0)
    (new java.net.URLClassLoader(Array(dir.toUri.toURL), getClass.getClassLoader), "childside.Outer$Failure")
  }

  test("a kernel JvmException assert resolves its class through the thread context loader") {
    val (loader, name) = childLoaderWithFailure()
    val ctx            = rt.Context.create(hostDevice)
    val thread         = Thread.currentThread
    val previous       = thread.getContextClassLoader
    try {
      thread.setContextClassLoader(loader)
      val e = intercept[RuntimeException](launchAssert(ctx, p.Enums.AssertCode.JvmException, s"$name: from child"))
      assertEquals(e.getClass.getName, name)
      assert(e.getClass.getClassLoader eq loader)
      assertEquals(e.getMessage, "from child")
    } finally {
      thread.setContextClassLoader(previous)
      ctx.close()
      loader.close()
    }
  }

  test("a kernel JvmException assert naming a class outside the thread context loader raises a runtime exception") {
    val (loader, name) = childLoaderWithFailure()
    val ctx            = rt.Context.create(hostDevice)
    try {
      val e =
        intercept[rt.PolyregionRuntimeException](launchAssert(ctx, p.Enums.AssertCode.JvmException, s"$name: hidden"))
      assert(e.getMessage.contains(s"$name: hidden"), e.getMessage)
    } finally {
      ctx.close()
      loader.close()
    }
  }

  test("a kernel JvmException assert naming a class that is not a Throwable raises a runtime exception") {
    val ctx = rt.Context.create(hostDevice)
    try {
      val e = intercept[rt.PolyregionRuntimeException](
        launchAssert(ctx, p.Enums.AssertCode.JvmException, "java.lang.String: boom")
      )
      assert(e.getMessage.contains("java.lang.String: boom"), e.getMessage)
    } finally ctx.close()
  }

  test("any other kernel assert code raises a runtime exception naming the code and message") {
    val ctx = rt.Context.create(hostDevice)
    try {
      val e = intercept[rt.PolyregionRuntimeException](
        launchAssert(ctx, p.Enums.AssertCode.Exception, "java.lang.ArithmeticException: / by zero")
      )
      assert(e.getMessage.contains("EXCP"), e.getMessage)
      assert(e.getMessage.contains("java.lang.ArithmeticException: / by zero"), e.getMessage)
    } finally ctx.close()
  }

  test("a kernel assert on the default context raises an AssertionError") {
    val e = intercept[AssertionError](launchAssert(rt.Context.current(), p.Enums.AssertCode.Assert, "default says no"))
    assertEquals(e.getMessage, "default says no")
  }

  test("a context launches again after a kernel assert") {
    val ctx = rt.Context.create(hostDevice)
    try {
      intercept[AssertionError](launchAssert(ctx, p.Enums.AssertCode.Assert, "first"))
      intercept[AssertionError](launchAssert(ctx, p.Enums.AssertCode.Assert, "second"))
      checkFill(ctx)
    } finally ctx.close()
  }

  test("a kernel image for another platform kind is not selected") {
    val ctx = rt.Context.create(hostDevice)
    try {
      val c     = compile(fillProgram, ct.Target.LLVM_NVPTX64, "sm_80")
      val image = rt.ModuleImage("fill@nv", ct.Target.LLVM_NVPTX64, c.features, c.program)
      assertEquals(ctx.select(Array(image)), -1)
      val e = intercept[rt.PolyregionRuntimeException] {
        val (types, data) = ptrArgs(1)
        ctx.launch(Array(image), image.name, "fill", types, data, rt.Policy(rt.Dim3(1, 1, 1)))
      }
      assert(e.getMessage.contains("fill@nv"), e.getMessage)
    } finally ctx.close()
  }

  test("select reports the most severe refusal as a negated outcome") {
    val ctx = rt.Context.create(hostDevice)
    try {
      def image(name: String, target: ct.Target, features: Array[String], bytes: Array[Byte]) =
        rt.ModuleImage(name, target, features, bytes)
      val format                           = image("format", ct.Target.LLVM_NVPTX64, Array.empty, Array[Byte](0))
      val lacks                            = image("lacks", ct.Target.LLVM_HOST, Array("storage8"), Array[Byte](0))
      val unprepared                       = image("unprepared", ct.Target.LLVM_HOST, Array.empty, Array.empty)
      def outcome(images: rt.ModuleImage*) = ctx.select(images.toArray)
      assertEquals(outcome(format), -rt.LoadOutcome.FORMAT.value.toInt)
      assertEquals(outcome(lacks), -rt.LoadOutcome.LACKS.value.toInt)
      assertEquals(outcome(unprepared), -rt.LoadOutcome.UNPREPARED.value.toInt)
      assertEquals(outcome(format, lacks), -rt.LoadOutcome.LACKS.value.toInt)
      assertEquals(outcome(lacks, format), -rt.LoadOutcome.LACKS.value.toInt)
      assertEquals(outcome(lacks, unprepared), -rt.LoadOutcome.UNPREPARED.value.toInt)
    } finally ctx.close()
  }

  test("a host program is called with the context as its first argument") {
    ensureProbe()
    val ctx = rt.Context.create(hostDevice)
    try {
      val (seed, adr) = probe(ctx, 0x5eed1234L)
      assertEquals(seed, 0x5eed1234L)
      assertNotEquals(adr, 0L)
    } finally ctx.close()
  }

  test("a host program on a destroyed context fails with the context message") {
    ensureProbe()
    val ctx = rt.Context.create(hostDevice)
    ctx.close()
    val e = intercept[rt.PolyregionRuntimeException](probe(ctx, 1))
    assert(e.getMessage.startsWith("Context ") && e.getMessage.endsWith(" is closed"), e.getMessage)
  }

  test("an unknown host program is rejected") {
    val ctx = rt.Context.create(hostDevice)
    try {
      val e = intercept[rt.PolyregionRuntimeException](
        ctx.callHost(Array.empty, "missing@host", "probe", Array.empty, Array.empty)
      )
      assertEquals(e.getMessage, "Host program `missing@host` is not loaded")
    } finally ctx.close()
  }

  test("an unknown host entry is rejected") {
    ensureProbe()
    val ctx = rt.Context.create(hostDevice)
    try {
      val e = intercept[rt.PolyregionRuntimeException](
        ctx.callHost(Array.empty, ProbeName, "absent", Array.empty, Array.empty)
      )
      assertEquals(e.getMessage, s"Host program `$ProbeName` has no entry `absent`")
    } finally ctx.close()
  }

  test("host programs run concurrently on separate contexts") {
    ensureProbe()
    val failures = java.util.concurrent.ConcurrentLinkedQueue[Throwable]()
    val threads = (0 until 8).map { t =>
      new Thread(
        (
            () =>
              try {
                val ctx = rt.Context.create(hostDevice)
                try
                  (0 until 50).foreach { i =>
                    val seed          = t * 1000L + i
                    val (echoed, adr) = probe(ctx, seed)
                    assertEquals(echoed, seed)
                    assertNotEquals(adr, 0L)
                  }
                finally ctx.close()
              } catch { case e: Throwable => failures.add(e) }
        ): Runnable
      )
    }
    threads.foreach(_.start())
    threads.foreach(_.join())
    assertEquals(failures.size, 0, failures.toString)
  }

  test("copyFromDevice rejects more bytes than the destination holds") {
    withRemote(16) { (ctx, remote) =>
      intercept[rt.PolyregionRuntimeException](ctx.copyFromDevice(remote, direct(4), 8))
    }
  }

  test("copyToDevice rejects more bytes than the source holds") {
    withRemote(16) { (ctx, remote) =>
      intercept[rt.PolyregionRuntimeException](ctx.copyToDevice(direct(4), remote, 8))
    }
  }

  test("copyToDevice counts the bytes from the source position") {
    withRemote(16) { (ctx, remote) =>
      val source = direct(16)
      (0 until 16).foreach(i => source.put(i, i.toByte))
      source.position(8)
      intercept[rt.PolyregionRuntimeException](ctx.copyToDevice(source, remote, 9))
      ctx.copyToDevice(source, remote, 8)
      val back = direct(8)
      ctx.copyFromDevice(remote, back, 8)
      (0 until 8).foreach(i => assertEquals(back.get(i), (8 + i).toByte))
    }
  }

  test("copyToDeviceAsync rejects more bytes than the source holds") {
    withRemote(16) { (ctx, remote) =>
      intercept[rt.PolyregionRuntimeException](ctx.copyToDevice(direct(4), remote, 8, (_: Throwable) => ()))
    }
  }

  test("copyFromDeviceAsync rejects more bytes than the destination holds") {
    withRemote(16) { (ctx, remote) =>
      intercept[rt.PolyregionRuntimeException](ctx.copyFromDevice(remote, direct(4), 8, (_: Throwable) => ()))
    }
  }

  test("a kernel launches on an NVIDIA device through a created context") {
    nvidiaDevice match {
      case None => assume(false, "no CUDA device in the catalogue")
      case Some(device) =>
        val n      = 256
        val target = ct.Target.LLVM_NVPTX64
        val c      = compile(fillProgram, target, s"sm_${device.cudaArchitectureMajor}${device.cudaArchitectureMinor}")
        val image  = rt.ModuleImage("fill@nv", target, c.features, c.program)
        val ctx    = rt.Context.create(device)
        try {
          val remote = ctx.malloc(n * 4)
          try {
            val (types, data) = ptrArgs(remote)
            assertEquals(ctx.select(Array(image)), 0)
            ctx.launch(Array(image), image.name, "fill", types, data, rt.Policy(rt.Dim3(n, 1, 1)))
            val back = direct(n * 4)
            ctx.copyFromDevice(remote, back, n * 4)
            (0 until n).foreach(i => assertEquals(back.getInt(i * 4), i))
          } finally ctx.free(remote)
        } finally ctx.close()
    }
  }

  private def raiseSeven: p.Stmt = p.Stmt.Raise(p.Term.IntS32Const(7), p.ExceptionKind(p.Type.IntS32, "int"), Nil)

  private def allocation(ctx: p.Named): p.Stmt =
    p.Stmt.Var(
      p.Named("owned", p.Spec.ContextType),
      Some(p.Expr.SpecOp(p.Spec.RemoteAlloc(p.Term.Select(ctx, Nil, ctx.tpe), p.Term.IntU64Const(64))))
    )

  private def hostRaise(name: String, program: p.Program): rt.PolyregionRuntimeException = {
    val module = s"$name@host"
    rt.Context.loadHostProgram(module, compile(program, ct.Target.LLVM_HOST, "native").program)
    val ctx = rt.Context.create(hostDevice)
    try intercept[rt.PolyregionRuntimeException](ctx.callHost(Array.empty, module, name, Array.empty, Array.empty))
    finally ctx.close()
  }

  private def assertRaised(e: rt.PolyregionRuntimeException): Unit =
    assert(e.getMessage.contains("EXCP") && e.getMessage.contains("int"), e.getMessage)

  test("a raise escaping a host helper after a remote operation reaches the JVM") {
    val ctx = p.Named("ctx", p.Spec.ContextType)
    val helper =
      hostFunction("raisingHelper", List(ctx), List(allocation(ctx), raiseSeven), p.Function.Visibility.Internal)
    val call = p.Stmt.Var(
      p.Named("called", p.Type.Unit0),
      Some(
        p.Expr.Invoke(p.Type.FnRef(helper.decl.name), Nil, None, List(p.Term.Select(ctx, Nil, ctx.tpe)), p.Type.Unit0)
      )
    )
    val entry = hostFunction("raiseThroughHelper", List(ctx), List(call))
    assertRaised(hostRaise("raiseThroughHelper", p.Program(Some(entry), List(helper), Nil)))
  }

  test("a raise escaping a host entry after a remote operation reaches the JVM") {
    val ctx = p.Named("ctx", p.Spec.ContextType)
    assertRaised(
      hostRaise("raiseAfterAlloc", hostProgram("raiseAfterAlloc", List(ctx), List(allocation(ctx), raiseSeven)))
    )
  }

  test("a raise escaping a host entry with no remote operation reaches the JVM") {
    val ctx = p.Named("ctx", p.Spec.ContextType)
    assertRaised(hostRaise("raiseOnly", hostProgram("raiseOnly", List(ctx), List(raiseSeven))))
  }

  test("a host program marshalling a host pointer through a created managed context raises and leaves it usable") {
    assume(
      rt.Context.platforms().flatMap(_.devices).exists(_.platformKind == ManagedKind),
      "no managed device in the catalogue"
    )
    val output = refusal(hostPointerLaunch)
    assert(output.contains("raised: ") && output.contains("created context cannot marshal"), output)
    assert(output.contains("usable: remoteMallocs=1"), output)
    assert(output.contains("owned: remoteAllocations=0"), output)
  }
}
