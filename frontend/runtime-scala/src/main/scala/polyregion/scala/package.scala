package polyregion.scala

import fansi.ErrorMode.Throw
import polyregion.jvm.compiler.Options
import polyregion.jvm.{Loader, compiler as cp, runtime as rt}

import java.util.concurrent.TimeUnit
import scala.compiletime.constValue
import scala.reflect.ClassTag

type Callback[A]   = Either[Throwable, A] => Unit
type Suspend[F[_]] = [A] => (Callback[A] => Unit) => F[A]

class Platform[F[_], T <: Target](r: rt.Platform, f: Suspend[F]) {
  lazy val name: String                    = r.name
  lazy val properties: Map[String, String] = r.properties.map(p => p.key -> p.value).toMap
  inline def devices: Vector[Device[F, T]] = r.devices.map(d => Device[F, T](d, f)).toVector
}

trait JitOps[F[_], O](d: rt.Device.Queue, f: Suspend[F]) {

  inline def task[A](using inline o: O)(f: => A): F[A] = ???

  inline def foreach(inline x: Range)
  /*             */ (using inline o: O)
  /*             */ (inline f: Int => Unit): F[Unit] = {
    println("s" + f)
    ???
  }
  inline def foreach(inline x: Range, inline y: Range)
  /*             */ (using inline o: O)
  /*             */ (inline f: (Int, Int) => Unit): this.type = ???
  inline def foreach(inline x: Range, inline y: Range, inline z: Range)
  /*             */ (using inline o: O)
  /*             */ (inline f: (Int, Int, Int) => Unit): this.type = ???

  inline def reduce[A](inline x: Range)
  /*               */ (using inline o: O)
  /*               */ (inline f: Int => A)
  /*               */ (inline g: (A, A) => A): F[A] = ???

  inline def reduce[A](inline x: Range, inline y: Range)
  /*               */ (using inline o: O)
  /*               */ (inline f: (Int, Int) => A)
  /*               */ (inline g: (A, A) => A): F[A] = ???

  inline def reduce[A](inline x: Range, inline y: Range, inline z: Range)
  /*               */ (using inline o: O)
  /*               */ (inline f: (Int, Int, Int) => A)
  /*               */ (inline g: (A, A) => A): F[A] = ???
}

trait AotOps[F[_], B](q: rt.Device.Queue, suspend: Suspend[F]) {

  inline def task[O <: B, A](inline f: => A): F[A] = suspend { cb =>
//     val result = Buffer.ofDim[A](1)
    val result = scala.collection.mutable.ListBuffer[A](null.asInstanceOf[A])
    // val result = Array.ofDim[A](1)
    polyregion.scala.compiletime.offload0[O](
      q,
      {
        case Left(e) => cb(Left(e))
        case Right(()) =>
          try cb(Right(result(0)))
          catch { case e: Throwable => cb(Left(e)) }
      }
    ) { result(0) = f; () }
  }

  inline def foreach[O <: B](inline x: Range)
  /*                     */ (inline f: Int => Unit): F[Unit] = suspend { cb =>
    val startX = x.start
    val stepX  = x.step
    polyregion.scala.compiletime.offload1[O](q, x, cb) {
      f(Support.linearise(startX, stepX)(polyregion.scala.intrinsics.gpuGlobalIdxX)) //
      ()
    }
  }

  inline def foreach[O <: B](inline x: Range, inline y: Range)
  /*                     */ (inline f: (Int, Int) => Unit): F[Unit] = suspend { cb =>
    val startX = x.start
    val stepX  = x.step
    val startY = y.start
    val stepY  = y.step
    polyregion.scala.compiletime.offload2[O](q, x, y, cb) {
      f(
        Support.linearise(startX, stepX)(polyregion.scala.intrinsics.gpuGlobalIdxX),
        Support.linearise(startY, stepY)(polyregion.scala.intrinsics.gpuGlobalIdxY)
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
    polyregion.scala.compiletime.offload3[O](q, x, y, z, cb) {
      f(
        Support.linearise(startX, stepX)(polyregion.scala.intrinsics.gpuGlobalIdxX),
        Support.linearise(startY, stepY)(polyregion.scala.intrinsics.gpuGlobalIdxY),
        Support.linearise(startZ, stepZ)(polyregion.scala.intrinsics.gpuGlobalIdxZ)
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

  inline def invalidate_(xs: AnyRef*): this.type = { d.invalidateAll(null, xs: _*); this }
  inline def invalidate(xs: AnyRef*): F[Unit]    = f(cb => d.invalidateAll(() => cb(Right(())), xs: _*))

  inline def sync: F[Unit] = f(cb => d.syncAll(() => cb(Right(()))))

  inline def release_(xs: AnyRef*): this.type = { d.releaseAll(xs: _*); this }
  inline def release(xs: AnyRef*): Unit       = d.releaseAll(xs: _*)

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
  inline def mkQueue: Queue = Queue(d.device.createQueue)
}

class AotDevice[F[_], T](d: rt.Device.Queue, f: Suspend[F]) extends AotOps[F, T](d, f) {
  class Queue(d: rt.Device.Queue) extends AotOps[F, T](d, f) with DeviceQueue[F](d, f)
  inline def mkQueue: Queue = Queue(d.device.createQueue)
}

class Device[F[_], T <: Target](val underlying: rt.Device, f: Suspend[F]) {
  lazy val name: String                                 = underlying.name
  lazy val properties: Map[String, String]              = underlying.properties.map(p => p.key -> p.value).toMap
  lazy val jit: JitDevice[F, Config[_ <: T, _]]         = JitDevice(underlying.createQueue, f)
  lazy val aot: AotDevice[F, Config[_ <: T, _] | Tuple] = AotDevice(underlying.createQueue, f)
}

enum Opt(val value: cp.Opt) {
  case O0    extends Opt(cp.Opt.O0)
  case O1    extends Opt(cp.Opt.O1)
  case O2    extends Opt(cp.Opt.O2)
  case O3    extends Opt(cp.Opt.O3)
  case Ofast extends Opt(cp.Opt.Ofast)
}
object Opt {
  type O0    = Opt.O0.type
  type O1    = Opt.O1.type
  type O2    = Opt.O2.type
  type O3    = Opt.O3.type
  type Ofast = Opt.Ofast.type
}

sealed trait Target {
  type Arch <: cp.Target & Singleton
  type UArch <: String & Singleton
  def arch: Arch
  def uarch: String
}

object Target {

  private object Aux {
    import scala.compiletime.*
    import scala.deriving.*

    private type T[A <: String, U <: String] = Target { type Arch = A; type UArch = U }

    transparent inline given deriveArchNames[A](using inline m: Mirror.Of[A]): Tuple = inline m match {
      case _: Mirror.SumOf[A] => deriveSum[m.MirroredElemTypes]
      case _: Mirror.ProductOf[A] =>
        inline scala.compiletime.erasedValue[m.MirroredMonoType] match {
          case _: T[arch, uarch] => constValue[arch] *: EmptyTuple
          case _                 => EmptyTuple
        }
    }

    private transparent inline def deriveSum[Xs <: Tuple]: Tuple = inline erasedValue[Xs] match {
      case _: (T[arch, uarch] *: ts) => constValue[arch] *: deriveSum[ts]
      case _: (t *: ts)              => deriveArchNames[t](using summonInline[Mirror.Of[t]]) ++ deriveSum[ts]
      case _: EmptyTuple             => EmptyTuple
    }

  }

  // final val ArchNames: Tuple = Aux.deriveArchNames[Target]

  // val code = scala.compiletime.codeOf(Aux.deriveArchNames[Target])

  import cp.Target as cpt
  private transparent inline def valueOf[T <: Singleton: ValueOf] = summon[ValueOf[T]].value

  type OpenCL_C = OpenCL_C.type
  val OpenCL_C = new Target {
    type Arch  = cp.Target.C_OpenCL1_1.type;
    type UArch = "auto"
    val arch  = valueOf[Arch]
    val uarch = constValue[UArch]
  }

  sealed trait CPU extends Target
  val Host = new CPU {
    type Arch = cp.Target.LLVM_HOST.type; type UArch = "native"
    val arch = valueOf[Arch]; val uarch = constValue[UArch]
  }
  case class X86(uarch: String)     extends CPU    { type Arch = cpt.LLVM_X86_64.type; val arch = valueOf[Arch]  }
  case class AArch64(uarch: String) extends CPU    { type Arch = cpt.LLVM_AARCH64.type; val arch = valueOf[Arch] }
  case class ARM(uarch: String)     extends CPU    { type Arch = cpt.LLVM_ARM.type; val arch = valueOf[Arch]     }
  case class NVPTX64(uarch: String) extends Target { type Arch = cpt.LLVM_NVPTX64.type; val arch = valueOf[Arch] }
  case class AMDGCN(uarch: String)  extends Target { type Arch = cpt.LLVM_AMDGCN.type; val arch = valueOf[Arch]  }
  case class SPIRV64(uarch: String) extends Target { type Arch = cpt.LLVM_SPIRV64.type; val arch = valueOf[Arch] }

  private type SString = String & Singleton

  object X86 {
    inline def apply[A <: SString](inline uarch: A) = new X86(uarch) { override type UArch = A }
    final val Znver2                                = X86("znver2")
  }
  object AArch64 {
    inline def apply[A <: SString](inline uarch: A) = new AArch64(uarch) { override type UArch = A }
    val AppleM1                                     = AArch64("apple-m1")
    val A64fx                                       = AArch64("a64fx")
  }
  object ARM {
    inline def apply[A <: SString](inline uarch: A) = new ARM(uarch) { override type UArch = A }
  }

  object NVPTX64 {
    inline def apply[A <: SString](inline uarch: A) = new NVPTX64(uarch) { override type UArch = A }
    final val SM52                                  = NVPTX64("sm_52")
    final val SM61                                  = NVPTX64("sm_61")
    final val SM80                                  = NVPTX64("sm_80")
  }
  object AMDGCN {
    inline def apply[A <: SString](inline uarch: A) = new AMDGCN(uarch) { override type UArch = A }
    final val gfx906                                = AMDGCN("gfx906")
    final val gfx803                                = AMDGCN("gfx803")
  }
  object SPIRV64 {
    inline def apply[A <: SString](inline uarch: A) = new SPIRV64(uarch) { override type UArch = A }
  }

}

case class Config[T <: Target, O <: Opt](targets: List[(T, O)]) {
  type Target = T
  type Opt    = O

  infix def ++(that: Config[T, O]) = Config(targets ++ that.targets)
}
object Config {
  def apply[T <: Target, O <: Opt](opt: O, target: T*): Config[T, O] = Config(target.map(_ -> opt).toList)
  def apply[T <: Target](target: T*): Config[T, Opt]                 = Config(Opt.O3, target: _*)
}

object Platforms {
  val platforms = rt.Platforms.create()
  sys.addShutdownHook(platforms.close())
  export platforms.*
}

inline def liftCUDA[F[_]](lift: Suspend[F]): Platform[F, Target.NVPTX64]    = Platform(Platforms.CUDA(), lift)
inline def liftHIP[F[_]](lift: Suspend[F]): Platform[F, Target.AMDGCN]      = Platform(Platforms.HIP(), lift)
inline def liftHSA[F[_]](lift: Suspend[F]): Platform[F, Target.AMDGCN]      = Platform(Platforms.HSA(), lift)
inline def liftOpenCL[F[_]](lift: Suspend[F]): Platform[F, Target.OpenCL_C] = Platform(Platforms.OpenCL(), lift)
inline def liftHost[F[_]](lift: Suspend[F]): Device[F, Target.CPU] = Device(Platforms.Relocatable().devices()(0), lift)

object blocking {

  import java.util.concurrent.CountDownLatch
  import java.util.concurrent.atomic.AtomicReference

  type Id[A] = A

  private val Latched: Suspend[Id] = [A] => { (cb: Callback[A] => Unit) =>
    val latch = CountDownLatch(1)
    val ref   = AtomicReference[Either[Throwable, A]]()
    cb { e =>
      ref.set(e); latch.countDown()
    }
    latch.await()
    ref.get.fold(throw _, identity)
  }

  lazy val CUDA: Platform[Id, Target.NVPTX64]    = liftCUDA(Latched)
  lazy val HIP: Platform[Id, Target.AMDGCN]      = liftHIP(Latched)
  lazy val HSA: Platform[Id, Target.AMDGCN]      = liftHSA(Latched)
  lazy val OpenCL: Platform[Id, Target.OpenCL_C] = liftOpenCL(Latched)
  lazy val Host: Device[Id, Target.CPU]          = liftHost(Latched)

}

object future {

  import scala.concurrent.{Future, Promise}
  private val SuspendFuture: Suspend[Future] = [A] => { (cb: Callback[A] => Unit) =>
    val p = Promise[A]
    cb(e => p.tryComplete(e.toTry))
    p.future
  }

  lazy val CUDA: Platform[Future, Target.NVPTX64]    = liftCUDA(SuspendFuture)
  lazy val HIP: Platform[Future, Target.AMDGCN]      = liftHIP(SuspendFuture)
  lazy val OpenCL: Platform[Future, Target.OpenCL_C] = liftOpenCL(SuspendFuture)
  lazy val Host: Device[Future, Target.CPU]          = liftHost(SuspendFuture)

  // extension [A](xs: Seq[A]) {

  //   inline def foreach[F[_], O](using d: JitDevice[F, O]#Queue, o: O)(f: A => Unit): Unit =
  //     d.foreach(0 until xs.length)(i => f(xs(i)))
  // }

}
