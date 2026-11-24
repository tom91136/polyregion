package polyregion

import polyregion.ast.generated.PolyPackageWireSchema
import polyregion.ast.{CodeGen, MsgPack, PolyAST as p}
import polyregion.jvm.twin.Layout
import polyregion.jvm.{compiler as ct, runtime as rt}
import polyregion.scalalang.{LaunchConfig, LaunchRuntime}

import java.nio.{ByteBuffer, ByteOrder}
import java.util.concurrent.atomic.AtomicLong

object SpectraRaw {

  val i32    = p.Type.IntS32
  val g      = p.Type.Space.Global
  val ptrI32 = p.Type.Ptr(i32, g)

  private val Counter = AtomicLong(0)

  def select(tpe: p.Type, name: String): (p.Named, p.Term.Select) = {
    val named = p.Named(name, tpe)
    named -> p.Term.Select(named, Nil, tpe)
  }

  def config(hostOnly: Boolean): LaunchConfig = {
    val configured = sys.env.get("POLYREGION_TARGETS").toList.flatMap(_.split(',')).map(_.trim).filter(_.nonEmpty)
    val targets    = if (hostOnly) List("host@native") else configured
    val settings =
      s"polyregion.library.path=${sys.props.getOrElse("polyregion.library.path", "")}" ::
        sys.props
          .get("polyregion.library.capabilities")
          .toList
          .flatMap(_.split(','))
          .filter(_.nonEmpty)
          .map(c => s"polyregion.library.capabilities=$c") ::: targets.map(t => s"polyregion.targets=$t")
    LaunchConfig.fromSettings(settings).fold(error => throw IllegalStateException(error), identity)
  }

  def callable(name: String, arg: p.Named, rtn: p.Type, body: List[p.Stmt]): p.Function =
    p.Function(
      p.FunctionDecl(p.Sym(name), Nil, None, List(p.Arg(arg)), Nil, Nil, rtn, p.Function.Affinity.Host),
      body,
      p.Function.Visibility.Internal,
      p.Function.FpMode.Relaxed,
      p.CallConvention.RegularCall
    )

  def compile(op: p.Function, hostOnly: Boolean): (String, Array[Byte], p.Compile.Bundle) = {
    val entry = s"__polyregion_raw_launch_${Counter.getAndIncrement()}_${System.nanoTime().toHexString}"
    val args  = List(ptrI32, ptrI32, i32, p.Type.FnRef(op.name))
    val root = p.Function(
      p.FunctionDecl(
        p.Sym(entry),
        Nil,
        None,
        args.zipWithIndex.map((tpe, index) => p.Arg(p.Named(s"arg$index", tpe))),
        Nil,
        Nil,
        p.Type.Unit0,
        p.Function.Affinity.Host
      ),
      Nil,
      p.Function.Visibility.Exported,
      p.Function.FpMode.Relaxed,
      p.CallConvention.RegularCall,
      Some(p.Sym("spectra.transform"))
    )
    val request = p.Program.LinkRequest(
      Nil,
      p.Program(None, List(root, op), Nil),
      Nil,
      List(p.Program.TypeSize(i32, 4))
    )
    val bytes    = MsgPack.encode(MsgPack.Versioned(PolyPackageWireSchema.Hash, request))
    val cfg      = config(hostOnly)
    val compiler = ct.Compiler.create()
    try {
      val raw = compiler.compileInterface(
        "spectra",
        cfg.roots,
        "spectra.transform",
        entry,
        bytes,
        cfg.capabilityList,
        cfg.targetList,
        cfg.opt,
        0
      )
      val bundles =
        raw.toList.map(b => MsgPack.decode[MsgPack.Versioned[p.Compile.Bundle]](b).fold(throw _, v => b -> v.t))
      val usable = bundles.filter(_._2.unfit.isEmpty)
      assert(usable.nonEmpty, bundles.flatMap(_._2.unfit).distinct.mkString("; "))
      val (b, bundle) = usable.head
      (entry, b, bundle)
    } finally compiler.close()
  }

  def rawLaunch(ctx: rt.Context, op: p.Function, xs: Array[Int], out: Array[Int]): Unit = {
    val (entry, bytes, bundle) = compile(op, hostOnly = true)
    val chunks                 = java.util.Base64.getEncoder.encodeToString(bytes).grouped(30000).toArray
    val group                  = LaunchRuntime.Group(s"$entry@g0", chunks, bundle.remoteModules.isEmpty)
    val site                   = LaunchRuntime.site(entry, Array(group), Layout.encodeAll(java.util.List.of()))
    val marshal                = LaunchRuntime.marshal(site, ctx)
    try
      LaunchRuntime.invoke(
        ctx,
        site,
        marshal,
        Array(
          LaunchRuntime.array(marshal, xs),
          LaunchRuntime.array(marshal, out),
          LaunchRuntime.int(marshal, xs.length)
        )
      )
    finally marshal.close()
  }

  private def hostProgram(name: String, args: List[p.Named], body: List[p.Stmt]): p.Program =
    p.Program(
      Some(
        p.Function(
          p.FunctionDecl(
            p.Sym(name),
            Nil,
            None,
            (p.Named("ctx", p.Spec.ContextType) :: args).map(p.Arg(_)),
            Nil,
            Nil,
            p.Type.Unit0,
            p.Function.Affinity.Host
          ),
          body,
          p.Function.Visibility.Exported,
          p.Function.FpMode.Relaxed,
          p.CallConvention.RegularCall
        )
      ),
      Nil,
      Nil
    )

  def dirtyHeap(bytes: Int): Unit = {
    val (n, nRef)     = select(i32, "n")
    val (out, _)      = select(ptrI32, "out")
    val (blk, blkRef) = select(ptrI32, "blk")
    val (scr, _)      = select(ptrI32, "scr")
    val (len, lenRef) = select(p.Type.IntU64, "len")
    val program = hostProgram(
      "dirty",
      List(out, n),
      List(
        p.Stmt.Var(blk, Some(p.Expr.Alloc(i32, nRef, g, p.Region.Opaque))),
        p.Stmt.Var(len, Some(p.Expr.Cast(p.Term.IntS32Const(bytes), p.Type.IntU64))),
        p.Stmt.Var(scr, Some(p.Expr.ForeignCall("memset", List(blkRef, p.Term.IntS32Const(0x5a), lenRef), ptrI32))),
        p.Stmt.Var(p.Named("freed", p.Type.Unit0), Some(p.Expr.ForeignCall("free", List(blkRef), p.Type.Unit0))),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )
    val compiler = ct.Compiler.create()
    val compiled =
      try
        compiler.compile(
          MsgPack.encode(CodeGen.programVersioned(program)),
          false,
          ct.Options.of(ct.Target.LLVM_HOST, "native"),
          ct.Opt.O2.value
        )
      finally compiler.close()
    val name = s"raw-dirty-$bytes"
    if (!rt.Context.hostProgramLoaded(name)) rt.Context.loadHostProgram(name, compiled.program)
    val data = ByteBuffer
      .allocate(12)
      .order(ByteOrder.nativeOrder)
      .putLong(scalaPointer(ByteBuffer.allocateDirect(4)))
      .putInt(bytes / 4)
      .array
    rt.Context.current().callHost(Array.empty, name, "dirty", Array(rt.Type.PTR.value, rt.Type.INT.value), data)
  }

  private def scalaPointer(buffer: ByteBuffer): Long = polyregion.scalalang.Platforms.pointerOfDirectBuffer(buffer)
}
