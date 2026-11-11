package polyregion.scalalang

import fansi.ErrorMode.Throw
import polyregion.jvm.compiler.Options
import polyregion.jvm.{Loader, runtime as rt}

import java.util.concurrent.TimeUnit
import scala.reflect.ClassTag

type Suspend[F[_]] = [A] => (Callback[A] => Unit) => F[A]

inline val DefaultQueueCapacity = 10000L

class Platform[F[_], T <: Target](r: rt.Platform, f: Suspend[F]) {
  lazy val name: String                    = r.name
  lazy val properties: Map[String, String] = r.properties.map(p => p.key -> p.value).toMap
  inline def devices: Vector[Device[F, T]] = r.devices.map(d => Device[F, T](d, f)).toVector
}

@SuppressWarnings(Array("bonsai:duplicate-code"))
trait JitOps[F[_], O](d: rt.Device.Queue, suspend: Suspend[F]) {

  inline def task[C <: O, A](inline f: => A): F[A] = suspend { cb =>
    val result = scala.collection.mutable.ListBuffer[A](null.asInstanceOf[A])
    polyregion.scalalang.compiletime.offloadJit0[C](
      d,
      {
        case Left(e) => cb(Left(e))
        case Right(()) =>
          val value = result(0)
          try cb(Right(value))
          catch { case e: Throwable => cb(Left(e)) }
      }
    ) { result(0) = f; () }
  }

  inline def foreach[C <: O](inline x: Range)
  /*                     */ (inline f: Int => Unit): F[Unit] = suspend { cb =>
    val startX = x.start
    val stepX  = x.step
    polyregion.scalalang.compiletime.offloadJit1[C](d, x, cb) {
      f(Support.linearise(startX, stepX)(polyregion.scalalang.intrinsics.gpuGlobalIdx(0)))
      ()
    }
  }

  inline def foreach[C <: O](inline x: Range, inline y: Range)
  /*                     */ (inline f: (Int, Int) => Unit): F[Unit] = suspend { cb =>
    val startX = x.start
    val stepX  = x.step
    val startY = y.start
    val stepY  = y.step
    polyregion.scalalang.compiletime.offloadJit2[C](d, x, y, cb) {
      f(
        Support.linearise(startX, stepX)(polyregion.scalalang.intrinsics.gpuGlobalIdx(0)),
        Support.linearise(startY, stepY)(polyregion.scalalang.intrinsics.gpuGlobalIdx(1))
      )
      ()
    }
  }

  inline def foreach[C <: O](inline x: Range, inline y: Range, inline z: Range)
  /*                     */ (inline f: (Int, Int, Int) => Unit): F[Unit] = suspend { cb =>
    val startX = x.start
    val stepX  = x.step
    val startY = y.start
    val stepY  = y.step
    val startZ = z.start
    val stepZ  = z.step
    polyregion.scalalang.compiletime.offloadJit3[C](d, x, y, z, cb) {
      f(
        Support.linearise(startX, stepX)(polyregion.scalalang.intrinsics.gpuGlobalIdx(0)),
        Support.linearise(startY, stepY)(polyregion.scalalang.intrinsics.gpuGlobalIdx(1)),
        Support.linearise(startZ, stepZ)(polyregion.scalalang.intrinsics.gpuGlobalIdx(2))
      )
      ()
    }
  }

  inline def reduce[C <: O, A](inline x: Range)
  /*                       */ (inline f: Int => A)
  /*                       */ (inline g: (A, A) => A): F[A] = ???
  inline def reduce[C <: O, A](inline x: Range, inline y: Range)
  /*                       */ (inline f: (Int, Int) => A)
  /*                       */ (inline g: (A, A) => A): F[A] = ???
  inline def reduce[C <: O, A](inline x: Range, inline y: Range, inline z: Range)
  /*                       */ (inline f: (Int, Int, Int) => A)
  /*                       */ (inline g: (A, A) => A): F[A] = ???
}

trait AotOps[F[_], B](q: rt.Device.Queue, suspend: Suspend[F]) {

  inline def task[O <: B, A](inline f: => A): F[A] = suspend { cb =>
//     val result = Buffer.ofDim[A](1)
    val result = scala.collection.mutable.ListBuffer[A](null.asInstanceOf[A])
    // val result = Array.ofDim[A](1)
    polyregion.scalalang.compiletime.offload0[O](
      q,
      {
        case Left(e) => cb(Left(e))
        case Right(()) =>
          val value = result(0)
          try cb(Right(value))
          catch { case e: Throwable => cb(Left(e)) }
      }
    ) { result(0) = f; () }
  }

  inline def foreach[O <: B](inline x: Range)
  /*                     */ (inline f: Int => Unit): F[Unit] = suspend { cb =>
    val startX = x.start
    val stepX  = x.step
    polyregion.scalalang.compiletime.offload1[O](q, x, cb) {
      f(Support.linearise(startX, stepX)(polyregion.scalalang.intrinsics.gpuGlobalIdx(0))) //
      ()
    }
  }

  inline def foreach[O <: B](inline x: Range, inline y: Range)
  /*                     */ (inline f: (Int, Int) => Unit): F[Unit] = suspend { cb =>
    val startX = x.start
    val stepX  = x.step
    val startY = y.start
    val stepY  = y.step
    polyregion.scalalang.compiletime.offload2[O](q, x, y, cb) {
      f(
        Support.linearise(startX, stepX)(polyregion.scalalang.intrinsics.gpuGlobalIdx(0)),
        Support.linearise(startY, stepY)(polyregion.scalalang.intrinsics.gpuGlobalIdx(1))
      ) //
      ()
    }
  }

  inline def foreach[O <: B](inline x: Range, inline y: Range, inline z: Range)
  /*                     */ (inline f: (Int, Int, Int) => Unit): F[Unit] = suspend { cb =>
    val startX = x.start
    val stepX  = x.step
    val startY = y.start
    val stepY  = y.step
    val startZ = z.start
    val stepZ  = z.step
    polyregion.scalalang.compiletime.offload3[O](q, x, y, z, cb) {
      f(
        Support.linearise(startX, stepX)(polyregion.scalalang.intrinsics.gpuGlobalIdx(0)),
        Support.linearise(startY, stepY)(polyregion.scalalang.intrinsics.gpuGlobalIdx(1)),
        Support.linearise(startZ, stepZ)(polyregion.scalalang.intrinsics.gpuGlobalIdx(2))
      ) //
      ()
    }
  }

  inline def reduce[O <: B, A](inline x: Range)
  /*                       */ (inline f: Int => A)
  /*                       */ (inline g: (A, A) => A): F[A] = ???
  inline def reduce[O <: B, A](inline x: Range, inline y: Range)
  /*                       */ (inline f: (Int, Int) => A)
  /*                       */ (inline g: (A, A) => A): F[A] = ???
  inline def reduce[O <: B, A](inline x: Range, inline y: Range, inline z: Range)
  /*                       */ (inline f: (Int, Int, Int) => A)
  /*                       */ (inline g: (A, A) => A): F[A] = ???
}

trait DeviceQueue[F[_]](d: rt.Device.Queue, f: Suspend[F]) {

  inline def invalidate_(xs: AnyRef*): this.type = { d.invalidateAll(null, xs*); this }
  inline def invalidate(xs: AnyRef*): F[Unit]    = f(cb => d.invalidateAll(() => cb(Right(())), xs*))

  inline def sync: F[Unit] = f(cb => d.syncAll(() => cb(Right(()))))

  inline def release_(xs: AnyRef*): this.type = { d.releaseAll(xs*); this }
  inline def release(xs: AnyRef*): Unit       = d.releaseAll(xs*)

  // inline def foreach(inline x: Int)
  // /*             */ (using inline o: O)
  // /*             */ (inline f: Int => Unit): F[Unit] = foreach(0 until x)(f)
  // inline def foreach(inline x: Int, inline y: Int)
  // /*             */ (using inline o: O)
  // /*             */ (inline f: (Int, Int) => Unit): F[Unit] = foreach(0 until x, 0 until y)(f)
  // inline def foreach(inline x: Int, inline y: Int, inline z: Int)
  // /*             */ (using inline o: O)
  // /*             */ (inline f: (Int, Int, Int) => Unit): F[Unit] = foreach(0 until x, 0 until y, 0 until z)(f)

  // inline def reduce[A](inline x: Int)
  // /*               */ (using inline o: O)
  // /*               */ (inline g: (A, A) => A)
  // /*               */ (inline f: Int => A): F[A] = reduce[A](0 until x)(g)(f)
  // inline def reduce[A](inline x: Int, inline y: Int)
  // /*               */ (using inline o: O)
  // /*               */ (inline g: (A, A) => A)
  // /*               */ (inline f: (Int, Int) => A): F[A] = reduce[A](0 until x, 0 until y)(g)(f)
  // inline def reduce[A](inline x: Int, inline y: Int, inline z: Int)
  // /*               */ (using inline o: O)
  // /*               */ (inline g: (A, A) => A)
  // /*               */ (inline f: (Int, Int, Int) => A): F[A] = reduce[A](0 until x, 0 until y, 0 until z)(g)(f)
}

class JitDevice[F[_], T](d: rt.Device.Queue, f: Suspend[F]) extends JitOps[F, T](d, f) {
  class Queue(d: rt.Device.Queue) extends JitOps[F, T](d, f) with DeviceQueue[F](d, f)
  inline def mkQueue: Queue = Queue(d.device.createQueue(DefaultQueueCapacity))
}

class AotDevice[F[_], T](d: rt.Device.Queue, f: Suspend[F]) extends AotOps[F, T](d, f) {
  class Queue(d: rt.Device.Queue) extends AotOps[F, T](d, f) with DeviceQueue[F](d, f)
  inline def mkQueue: Queue = Queue(d.device.createQueue(DefaultQueueCapacity))
}

class Device[F[_], T <: Target](val underlying: rt.Device, f: Suspend[F]) {
  lazy val name: String                                 = underlying.name
  lazy val properties: Map[String, String]              = underlying.properties.map(p => p.key -> p.value).toMap
  lazy val jit: JitDevice[F, Config[? <: T, ?]]         = JitDevice(underlying.createQueue(DefaultQueueCapacity), f)
  lazy val aot: AotDevice[F, Config[? <: T, ?] | Tuple] = AotDevice(underlying.createQueue(DefaultQueueCapacity), f)
}

inline def liftCUDA[F[_]](lift: Suspend[F]): Platform[F, Target.NVPTX64[?]] = Platform(Platforms.CUDA(), lift)
inline def liftHIP[F[_]](lift: Suspend[F]): Platform[F, Target.AMDGCN[?]]   = Platform(Platforms.HIP(), lift)
inline def liftHSA[F[_]](lift: Suspend[F]): Platform[F, Target.AMDGCN[?]]   = Platform(Platforms.HSA(), lift)
inline def liftOpenCL[F[_]](lift: Suspend[F]): Platform[F, Target.OpenCL_C] = Platform(Platforms.OpenCL(), lift)
inline def liftHost[F[_]](lift: Suspend[F]): Device[F, Target.CPU] = Device(Platforms.Relocatable().devices()(0), lift)

object blocking {

  import java.util.concurrent.CountDownLatch
  import java.util.concurrent.atomic.AtomicReference

  type Id[A] = A

  @SuppressWarnings(Array("bonsai:unbounded-wait"))
  private val Latched: Suspend[Id] = [A] => { (cb: Callback[A] => Unit) =>
    val latch = CountDownLatch(1)
    val ref   = AtomicReference[Either[Throwable, A]]()
    cb { e =>
      ref.set(e); latch.countDown()
    }
    latch.await()
    ref.get.fold(throw _, identity)
  }

  lazy val CUDA: Platform[Id, Target.NVPTX64[?]] = liftCUDA(Latched)
  lazy val HIP: Platform[Id, Target.AMDGCN[?]]   = liftHIP(Latched)
  lazy val HSA: Platform[Id, Target.AMDGCN[?]]   = liftHSA(Latched)
  lazy val OpenCL: Platform[Id, Target.OpenCL_C] = liftOpenCL(Latched)
  lazy val Host: Device[Id, Target.CPU]          = liftHost(Latched)

}

object future {

  import scala.concurrent.{Future, Promise}
  private val SuspendFuture: Suspend[Future] = [A] => { (cb: Callback[A] => Unit) =>
    val p = Promise[A]()
    cb(e => p.tryComplete(e.toTry))
    p.future
  }

  lazy val CUDA: Platform[Future, Target.NVPTX64[?]] = liftCUDA(SuspendFuture)
  lazy val HIP: Platform[Future, Target.AMDGCN[?]]   = liftHIP(SuspendFuture)
  lazy val OpenCL: Platform[Future, Target.OpenCL_C] = liftOpenCL(SuspendFuture)
  lazy val Host: Device[Future, Target.CPU]          = liftHost(SuspendFuture)

  // extension [A](xs: Seq[A]) {

  //   inline def foreach[F[_], O](using d: JitDevice[F, O]#Queue, o: O)(f: A => Unit): Unit =
  //     d.foreach(0 until xs.length)(i => f(xs(i)))
  // }

}
