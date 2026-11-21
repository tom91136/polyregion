package polyregion.scalalang

import polyregion.jvm.runtime as rt

import java.nio.ByteBuffer

class Platform(r: rt.Platform) {
  private lazy val index                   = Contexts.platformIndex(r.name)
  lazy val name: String                    = r.name
  lazy val properties: Map[String, String] = r.properties.map(p => p.key -> p.value).toMap
  inline def devices: Vector[Device] =
    r.devices.zipWithIndex.map((d, i) => Device(d, Contexts.of(index, i))).toVector
}

private def residentOf(x: AnyRef): AnyRef = x match {
  case b: Buffer[?] => b.backing
  case other        => other
}

private def buffersOf(xs: Seq[AnyRef]): Array[ByteBuffer] = xs.map(residentOf).toArray.map {
  case b: ByteBuffer => b
  case other         => throw IllegalArgumentException(s"Expected a Buffer or a direct ByteBuffer, got $other")
}

// uploads on entry and writes back on exit, also when the body throws; host writes to a resident inside the scope are
// not tracked: a discrete device never sees them and the write-back overwrites them, on a shared address space they
// alias the device copy
extension (ctx: rt.Context) {
  def resident[A](xs: AnyRef*)(body: => A): A = ctx.residentValue(xs.map(residentOf).toArray, _ => body)
}

@SuppressWarnings(Array("bonsai:duplicate-code"))
class Ops(ctx: rt.Context) {

  inline def task[A](inline f: => A)(cb: Callback[A]): Unit = {
    val result = scala.collection.mutable.ListBuffer[A](null.asInstanceOf[A])
    polyregion.scalalang.compiletime.offload0(
      ctx,
      {
        case Left(e) => cb(Left(e))
        case Right(()) =>
          val value = result(0)
          try cb(Right(value))
          catch { case e: Throwable => cb(Left(e)) }
      }
    ) { result(0) = f; () }
  }

  inline def foreach(inline x: Range)
  /*                     */ (inline f: Int => Unit)(cb: Callback[Unit]): Unit = {
    val startX = x.start
    val stepX  = x.step
    polyregion.scalalang.compiletime.offload1(ctx, x, cb) {
      f(Support.linearise(startX, stepX)(polyregion.scalalang.intrinsics.gpuGlobalIdx(0))) //
      ()
    }
  }

  inline def foreach(inline x: Range, inline y: Range)
  /*                     */ (inline f: (Int, Int) => Unit)(cb: Callback[Unit]): Unit = {
    val startX = x.start
    val stepX  = x.step
    val startY = y.start
    val stepY  = y.step
    polyregion.scalalang.compiletime.offload2(ctx, x, y, cb) {
      f(
        Support.linearise(startX, stepX)(polyregion.scalalang.intrinsics.gpuGlobalIdx(0)),
        Support.linearise(startY, stepY)(polyregion.scalalang.intrinsics.gpuGlobalIdx(1))
      ) //
      ()
    }
  }

  inline def foreach(inline x: Range, inline y: Range, inline z: Range)
  /*                     */ (inline f: (Int, Int, Int) => Unit)(cb: Callback[Unit]): Unit = {
    val startX = x.start
    val stepX  = x.step
    val startY = y.start
    val stepY  = y.step
    val startZ = z.start
    val stepZ  = z.step
    polyregion.scalalang.compiletime.offload3(ctx, x, y, z, cb) {
      f(
        Support.linearise(startX, stepX)(polyregion.scalalang.intrinsics.gpuGlobalIdx(0)),
        Support.linearise(startY, stepY)(polyregion.scalalang.intrinsics.gpuGlobalIdx(1)),
        Support.linearise(startZ, stepZ)(polyregion.scalalang.intrinsics.gpuGlobalIdx(2))
      ) //
      ()
    }
  }
}

class Device(val underlying: rt.Device, val context: rt.Context) {
  def on(context: rt.Context): Device      = Device(underlying, context)
  lazy val name: String                    = underlying.name
  lazy val properties: Map[String, String] = underlying.properties.map(p => p.key -> p.value).toMap
  lazy val ops: Ops                        = Ops(context)
}

// the callback runs on the launching thread before the launch returns
private[scalalang] trait Lift[R[_]] {
  def apply[A](start: Callback[A] => Unit): R[A]
}

@SuppressWarnings(Array("bonsai:duplicate-code"))
class LiftedOps[R[_]] private[scalalang] (ops: Ops, lift: Lift[R]) {

  inline def task[A](inline f: => A): R[A] = lift[A](cb => ops.task[A](f)(cb))

  inline def foreach(inline x: Range)
  /*                     */ (inline f: Int => Unit): R[Unit] = lift[Unit](cb => ops.foreach(x)(f)(cb))

  inline def foreach(inline x: Range, inline y: Range)
  /*                     */ (inline f: (Int, Int) => Unit): R[Unit] = lift[Unit](cb => ops.foreach(x, y)(f)(cb))

  inline def foreach(inline x: Range, inline y: Range, inline z: Range)
  /*                     */ (inline f: (Int, Int, Int) => Unit): R[Unit] =
    lift[Unit](cb => ops.foreach(x, y, z)(f)(cb))
}

class LiftedDevice[R[_]] private[scalalang] (device: Device, lift: Lift[R]) {
  def underlying: rt.Device                    = device.underlying
  def context: rt.Context                      = device.context
  def on(context: rt.Context): LiftedDevice[R] = LiftedDevice(device.on(context), lift)
  lazy val name: String                        = device.name
  lazy val properties: Map[String, String]     = device.properties
  lazy val ops: LiftedOps[R]                   = LiftedOps(device.ops, lift)
}

class LiftedPlatform[R[_]] private[scalalang] (platform: Platform, lift: Lift[R]) {
  lazy val name: String                    = platform.name
  lazy val properties: Map[String, String] = platform.properties
  def devices: Vector[LiftedDevice[R]]     = platform.devices.map(LiftedDevice(_, lift))
}

object callback {
  lazy val CUDA: Platform   = Platform(Platforms.CUDA())
  lazy val HIP: Platform    = Platform(Platforms.HIP())
  lazy val HSA: Platform    = Platform(Platforms.HSA())
  lazy val OpenCL: Platform = Platform(Platforms.OpenCL())
  lazy val Host: Device = {
    val platform = Platforms.Relocatable()
    Device(platform.devices()(0), Contexts.of(Contexts.platformIndex(platform.name), 0))
  }
}

object blocking {

  import java.util.concurrent.CountDownLatch
  import java.util.concurrent.atomic.AtomicReference

  type Id[A] = A

  @SuppressWarnings(Array("bonsai:unbounded-wait"))
  private val Latched: Lift[Id] = new Lift[Id] {
    def apply[A](start: Callback[A] => Unit): A = {
      val latch = CountDownLatch(1)
      val ref   = AtomicReference[Either[Throwable, A]]()
      start { e =>
        ref.set(e); latch.countDown()
      }
      latch.await()
      ref.get.fold(throw _, identity)
    }
  }

  private def await(start: rt.Completion => Unit): Unit = Latched[Unit](cb =>
    try start(failure => cb(if (failure == null) Right(()) else Left(failure)))
    catch { case e: Throwable => cb(Left(e)) }
  )

  def invalidate(ctx: rt.Context, xs: AnyRef*): Unit = {
    rt.Queue.requireNotCompletionThread()
    await(ctx.queue.invalidateAll(_, buffersOf(xs)*))
  }
  def sync(ctx: rt.Context, xs: AnyRef*): Unit = {
    rt.Queue.requireNotCompletionThread()
    await(ctx.queue.syncAll(_, buffersOf(xs)*))
  }
  // blocks until pending copies on the buffers settle
  def release(ctx: rt.Context, xs: AnyRef*): Unit = ctx.queue.releaseAll(buffersOf(xs)*)

  lazy val CUDA: LiftedPlatform[Id]   = LiftedPlatform(callback.CUDA, Latched)
  lazy val HIP: LiftedPlatform[Id]    = LiftedPlatform(callback.HIP, Latched)
  lazy val HSA: LiftedPlatform[Id]    = LiftedPlatform(callback.HSA, Latched)
  lazy val OpenCL: LiftedPlatform[Id] = LiftedPlatform(callback.OpenCL, Latched)
  lazy val Host: LiftedDevice[Id]     = LiftedDevice(callback.Host, Latched)
}

object future {

  import scala.concurrent.{Future, Promise}
  private val FutureLift: Lift[Future] = new Lift[Future] {
    def apply[A](start: Callback[A] => Unit): Future[A] = {
      val p = Promise[A]()
      start(e => p.tryComplete(e.toTry))
      p.future
    }
  }

  private def suspend(start: rt.Completion => Unit): Future[Unit] = FutureLift[Unit](cb =>
    try start(failure => cb(if (failure == null) Right(()) else Left(failure)))
    catch { case e: Throwable => cb(Left(e)) }
  )

  def invalidate(ctx: rt.Context, xs: AnyRef*): Future[Unit] = suspend(ctx.queue.invalidateAll(_, buffersOf(xs)*))
  def sync(ctx: rt.Context, xs: AnyRef*): Future[Unit]       = suspend(ctx.queue.syncAll(_, buffersOf(xs)*))
  // blocks until pending copies on the buffers settle
  def release(ctx: rt.Context, xs: AnyRef*): Unit = ctx.queue.releaseAll(buffersOf(xs)*)

  lazy val CUDA: LiftedPlatform[Future]   = LiftedPlatform(callback.CUDA, FutureLift)
  lazy val HIP: LiftedPlatform[Future]    = LiftedPlatform(callback.HIP, FutureLift)
  lazy val OpenCL: LiftedPlatform[Future] = LiftedPlatform(callback.OpenCL, FutureLift)
  lazy val Host: LiftedDevice[Future]     = LiftedDevice(callback.Host, FutureLift)
}
