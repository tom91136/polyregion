package polyregion.scalalang

import polyregion.ast.{MsgPack, PolyAST as p}
import polyregion.jvm.compiler.Target
import polyregion.jvm.runtime as rt
import polyregion.jvm.twin.{Marshal, Twins}

import java.nio.{ByteBuffer, ByteOrder}
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.locks.ReentrantLock

object LaunchRuntime {

  sealed trait Arg {
    def address: Long
  }

  private case class Cell(buffer: ByteBuffer, address: Long) extends Arg

  private case class Pointer(address: Long) extends Arg

  private val Targets: Map[Byte, Target] = Target.VALUES.map(target => target.value() -> target).toMap

  class Group(val program: String, bundle: Array[String], val hostOnly: Boolean) {
    private lazy val decoded: p.Compile.Bundle =
      MsgPack.decode[MsgPack.Versioned[p.Compile.Bundle]](java.util.Base64.getDecoder.decode(bundle.mkString)) match {
        case Left(error)      => throw error
        case Right(versioned) => versioned.t
      }

    private lazy val modules: Array[rt.ModuleImage] = decoded.remoteModules.map { module =>
      rt.ModuleImage(
        module.moduleName,
        Targets.getOrElse(module.target.toByte, throw IllegalStateException(s"unknown target ${module.target}")),
        module.features.toArray,
        module.image.toArray
      )
    }.toArray

    def accepts(ctx: rt.Context): Boolean = hostOnly || ctx.select(modules) >= 0

    def load(): Unit =
      ensureLoaded(program, rt.Context.hostProgramLoaded(program))(
        rt.Context.loadHostProgram(program, decoded.hostObject.toArray)
      )

    def call(ctx: rt.Context, entry: String, types: Array[Byte], data: Array[Byte]): Unit =
      ctx.callHost(modules, program, entry, types, data)
  }

  class Site(val entry: String, groups: Array[Group], layouts: String) {
    private val ordered = groups.sortBy(_.hostOnly)

    val twins: Twins = Twins.decode(layouts)

    def select(ctx: rt.Context): Group = ordered
      .find(_.accepts(ctx))
      .getOrElse(throw rt.PolyregionRuntimeException(s"No compiled variant of `$entry` accepts $ctx"))
  }

  private val loaded = ConcurrentHashMap.newKeySet[String]()

  private val loadLock = ReentrantLock()

  private[scalalang] def ensureLoaded(program: String, isLoaded: => Boolean)(load: => Unit): Unit =
    if (!loaded.contains(program)) {
      loadLock.lock()
      try
        if (!loaded.contains(program)) { if (!isLoaded) load; loaded.add(program) }
      finally loadLock.unlock()
    }

  private val sites = ConcurrentHashMap[String, Site]()

  def marshal(site: Site): Marshal = Marshal.inPlace(site.twins, Platforms.directBufferFromPointer(_, _))

  private def scalar(m: Marshal, bytes: Int)(write: ByteBuffer => Unit): Arg = {
    val buffer = Marshal.allocate(bytes)
    write(buffer)
    Cell(buffer, m.stage(buffer))
  }

  def int(m: Marshal, x: Int): Arg         = scalar(m, 4)(_.putInt(0, x))
  def long(m: Marshal, x: Long): Arg       = scalar(m, 8)(_.putLong(0, x))
  def float(m: Marshal, x: Float): Arg     = scalar(m, 4)(_.putFloat(0, x))
  def double(m: Marshal, x: Double): Arg   = scalar(m, 8)(_.putDouble(0, x))
  def short(m: Marshal, x: Short): Arg     = scalar(m, 2)(_.putShort(0, x))
  def byte(m: Marshal, x: Byte): Arg       = scalar(m, 1)(_.put(0, x))
  def char(m: Marshal, x: Char): Arg       = scalar(m, 2)(_.putChar(0, x))
  def boolean(m: Marshal, x: Boolean): Arg = scalar(m, 1)(_.put(0, if (x) 1.toByte else 0.toByte))

  def array(m: Marshal, xs: Array[?]): Arg = Pointer(m.array(xs))

  class EnvironmentBuilder(marshal: Marshal, size: Int) {
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
    def array(offset: Int, xs: Array[?]): EnvironmentBuilder = { buffer.putLong(offset, marshal.array(xs)); this }
    def done: Arg                                            = Pointer(marshal.stage(buffer))
  }

  def environment(m: Marshal, size: Int): EnvironmentBuilder = EnvironmentBuilder(m, size)

  def result(m: Marshal, bytes: Int): Arg = scalar(m, bytes)(_ => ())

  private def read(x: Arg): ByteBuffer = x match {
    case Cell(buffer, _) => buffer
    case Pointer(_)      => throw IllegalArgumentException("a pointer holds no result")
  }

  def resultInt(x: Arg): Int         = read(x).getInt(0)
  def resultLong(x: Arg): Long       = read(x).getLong(0)
  def resultFloat(x: Arg): Float     = read(x).getFloat(0)
  def resultDouble(x: Arg): Double   = read(x).getDouble(0)
  def resultShort(x: Arg): Short     = read(x).getShort(0)
  def resultByte(x: Arg): Byte       = read(x).get(0)
  def resultChar(x: Arg): Char       = read(x).getChar(0)
  def resultBoolean(x: Arg): Boolean = read(x).get(0) != 0

  def site(entry: String, groups: Array[Group], layouts: String): Site =
    sites.computeIfAbsent(entry, _ => Site(entry, groups, layouts))

  def invoke(ctx: rt.Context, site: Site, marshal: Marshal, args: Array[Arg]): Unit = {
    val group = site.select(ctx)
    group.load()
    val types = Array.fill[Byte](args.length)(rt.Type.PTR.value)
    val data  = ByteBuffer.allocate(8 * args.length).order(ByteOrder.nativeOrder)
    args.foreach(arg => data.putLong(arg.address))
    group.call(ctx, site.entry, types, data.array)
    marshal.settle()
  }
}
