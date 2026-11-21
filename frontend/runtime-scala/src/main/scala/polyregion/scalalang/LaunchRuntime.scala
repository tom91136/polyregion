package polyregion.scalalang

import polyregion.ast.{MsgPack, PolyAST as p, given}
import polyregion.jvm.compiler.Target
import polyregion.jvm.twin.Marshal
import polyregion.jvm.runtime as rt

import java.nio.{ByteBuffer, ByteOrder}
import java.util.Base64
import java.util.concurrent.ConcurrentHashMap

final class LaunchGroup(val program: String, bundle: Array[String], targets: Array[Byte]) {
  private lazy val decoded: p.Compile.Bundle =
    MsgPack.decode[MsgPack.Versioned[p.Compile.Bundle]](Base64.getDecoder.decode(bundle.mkString)).fold(throw _, _.t)

  lazy val host: Array[Byte] = decoded.hostObject.toArray

  lazy val modules: Array[rt.ModuleImage] = decoded.remoteModules
    .zip(targets)
    .map { (module, target) =>
      rt.ModuleImage(
        module.moduleName,
        Target.VALUES.find(_.value() == target).getOrElse(throw IllegalStateException(s"unknown target $target")),
        module.features.toArray,
        module.image.toArray
      )
    }
    .toArray
}

final class LaunchSite(val entry: String, val groups: Array[LaunchGroup])

sealed trait LaunchArg {
  def address: Long
}

object LaunchRuntime {

  private val loaded = ConcurrentHashMap.newKeySet[String]()

  private final class Cell(val buffer: ByteBuffer) extends LaunchArg {
    val address: Long = Platforms.pointerOfDirectBuffer(buffer)
  }

  private final class Pointer(val address: Long) extends LaunchArg

  def marshal(): Marshal = Marshal.inPlace(null, Platforms.directBufferFromPointer(_, _))

  private def scalar(bytes: Int)(write: ByteBuffer => Unit): LaunchArg = {
    val buffer = Marshal.allocate(bytes)
    write(buffer)
    Cell(buffer)
  }

  def int(x: Int): LaunchArg         = scalar(4)(_.putInt(0, x))
  def long(x: Long): LaunchArg       = scalar(8)(_.putLong(0, x))
  def float(x: Float): LaunchArg     = scalar(4)(_.putFloat(0, x))
  def double(x: Double): LaunchArg   = scalar(8)(_.putDouble(0, x))
  def short(x: Short): LaunchArg     = scalar(2)(_.putShort(0, x))
  def byte(x: Byte): LaunchArg       = scalar(1)(_.put(0, x))
  def char(x: Char): LaunchArg       = scalar(2)(_.putChar(0, x))
  def boolean(x: Boolean): LaunchArg = scalar(1)(_.put(0, if (x) 1.toByte else 0.toByte))

  def array(m: Marshal, xs: AnyRef): LaunchArg = Pointer(m.array(xs))

  final class EnvironmentBuilder(marshal: Marshal, size: Int) {
    private val buffer                                     = Marshal.allocate(size)
    def int(offset: Int, x: Int): EnvironmentBuilder       = { buffer.putInt(offset, x); this }
    def long(offset: Int, x: Long): EnvironmentBuilder     = { buffer.putLong(offset, x); this }
    def float(offset: Int, x: Float): EnvironmentBuilder   = { buffer.putFloat(offset, x); this }
    def double(offset: Int, x: Double): EnvironmentBuilder = { buffer.putDouble(offset, x); this }
    def short(offset: Int, x: Short): EnvironmentBuilder   = { buffer.putShort(offset, x); this }
    def byte(offset: Int, x: Byte): EnvironmentBuilder     = { buffer.put(offset, x); this }
    def char(offset: Int, x: Char): EnvironmentBuilder     = { buffer.putChar(offset, x); this }
    def boolean(offset: Int, x: Boolean): EnvironmentBuilder = {
      buffer.put(offset, if (x) 1.toByte else 0.toByte); this
    }
    def array(offset: Int, xs: AnyRef): EnvironmentBuilder = { buffer.putLong(offset, marshal.array(xs)); this }
    def done: LaunchArg                                    = Pointer(marshal.stage(buffer))
  }

  def environment(m: Marshal, size: Int): EnvironmentBuilder = EnvironmentBuilder(m, size)

  def result(bytes: Int): LaunchArg = scalar(bytes)(_ => ())

  def resultInt(x: LaunchArg): Int         = cell(x).buffer.getInt(0)
  def resultLong(x: LaunchArg): Long       = cell(x).buffer.getLong(0)
  def resultFloat(x: LaunchArg): Float     = cell(x).buffer.getFloat(0)
  def resultDouble(x: LaunchArg): Double   = cell(x).buffer.getDouble(0)
  def resultShort(x: LaunchArg): Short     = cell(x).buffer.getShort(0)
  def resultByte(x: LaunchArg): Byte       = cell(x).buffer.get(0)
  def resultChar(x: LaunchArg): Char       = cell(x).buffer.getChar(0)
  def resultBoolean(x: LaunchArg): Boolean = cell(x).buffer.get(0) != 0

  private def cell(x: LaunchArg): Cell = x.asInstanceOf[Cell]

  private val sites = ConcurrentHashMap[String, LaunchSite]()

  def site(entry: String, groups: Array[LaunchGroup]): LaunchSite =
    sites.computeIfAbsent(entry, _ => LaunchSite(entry, groups))

  def invoke(ctx: rt.Context, site: LaunchSite, marshal: Marshal, args: Array[LaunchArg]): Unit = {
    val group = site.groups
      .find(g => g.modules.isEmpty || ctx.select(g.modules) >= 0)
      .getOrElse(throw rt.PolyregionRuntimeException(s"No compiled variant of `${site.entry}` accepts $ctx"))
    if (loaded.add(group.program) && !rt.Context.hostProgramLoaded(group.program))
      rt.Context.loadHostProgram(group.program, group.host)
    val types = Array.fill[Byte](args.length)(rt.Type.PTR.value)
    val data  = ByteBuffer.allocate(8 * args.length).order(ByteOrder.nativeOrder)
    args.foreach(arg => data.putLong(arg.address))
    ctx.callHost(group.modules, group.program, site.entry, types, data.array)
    marshal.settle()
  }
}
