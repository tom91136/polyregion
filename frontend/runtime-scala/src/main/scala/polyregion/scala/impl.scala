package polyregion.scala

import polyregion.jvm.compiler.Options

import scala.collection.mutable
import scala.concurrent.{Future, Promise}

enum Optimisation {
  case O0, O1, O2, O3, Ofast
}

sealed trait Arch
enum CUDAArch extends Arch {
  case SM51, SM80
}

sealed trait CompileOptions
object CompileOptions {
  case class JIT(opt: Optimisation, archHint: Option[Arch] = None) extends CompileOptions
  case class AOT[A <: Arch](arch: A, opt: Optimisation)            extends CompileOptions
}

type Suspend[F[_]] = [A] => ((Either[Throwable, A] => Unit) => Unit) => F[A]

private val suspendFuture: Suspend[Future] = [A] => { (cb: (Either[Throwable, A] => Unit) => Unit) =>
  val p = Promise[A]
  cb(e => p.tryComplete(e.toTry): Unit)
  p.future
}

class Runtime[F[_], O <: CompileOptions](r: polyregion.jvm.runtime.Runtime)(f: Suspend[F]) {
  def jit: Runtime[F, CompileOptions.JIT] = ???

  def name: String                    = r.name
  def properties: Map[String, String] = r.properties.map(p => p.key -> p.value).toMap
  def devices: Vector[Device[F, O]]   = r.devices.map(d => new Device[F, O](d)(f)).toVector
}

class VarWitness(val xs : mutable.WeakHashMap[AnyRef, Buffer[Any]]){
  
  
  
}

class DeviceQueue[F[_], O <: CompileOptions]( //
    d: polyregion.jvm.runtime.Device,         //
    q: polyregion.jvm.runtime.Device.Queue    //
)(f: Suspend[F]) {
  
  
  inline def sync  : F[Unit] = ???

  inline def use(xs : AnyRef*) : F[Unit] = ???
  
  inline def task[A](using inline o: O)(inline f: => A): F[A] = ???

  inline def foreach(inline x: Range)
  /*             */ (using inline o: O)
  /*             */ (inline f: Int => Unit): F[Unit] = ???

  inline def foreach(inline x: Range, inline y: Range)
  /*             */ (using inline o: O)
  /*             */ (inline f: (Int, Int) => Unit): F[Unit] = ???

  inline def foreach(inline x: Range, inline y: Range, inline z: Range)
  /*             */ (using inline o: O)
  /*             */ (inline f: (Int, Int, Int) => Unit): F[Unit] = ???

  inline def reduce[A](inline x: Range)
  /*               */ (using inline o: O)
  /*               */ (inline g: (A, A) => A)
  /*               */ (inline f: Int => A): F[A] = ???

  inline def reduce[A](inline x: Range, inline y: Range)
  /*               */ (using inline o: O)
  /*               */ (inline g: (A, A) => A)
  /*               */ (inline f: (Int, Int) => A): F[A] = ???

  inline def reduce[A](inline x: Range, inline y: Range, inline z: Range)
  /*               */ (using inline o: O)
  /*               */ (inline g: (A, A) => A)
  /*               */ (inline f: (Int, Int, Int) => A): F[A] = ???

  inline def foreach(inline x: Int)
  /*             */ (using inline o: O)
  /*             */ (inline f: Int => Unit): F[Unit] = foreach(0 until x)(f)
  inline def foreach(inline x: Int, inline y: Int)
  /*             */ (using inline o: O)
  /*             */ (inline f: (Int, Int) => Unit): F[Unit] = foreach(0 until x, 0 until y)(f)
  inline def foreach(inline x: Int, inline y: Int, inline z: Int)
  /*             */ (using inline o: O)
  /*             */ (inline f: (Int, Int, Int) => Unit): F[Unit] = foreach(0 until x, 0 until y, 0 until z)(f)

  inline def reduce[A](inline x: Int)
  /*               */ (using inline o: O)
  /*               */ (inline g: (A, A) => A)
  /*               */ (inline f: Int => A): F[A] = reduce[A](0 until x)(g)(f)
  inline def reduce[A](inline x: Int, inline y: Int)
  /*               */ (using inline o: O)
  /*               */ (inline g: (A, A) => A)
  /*               */ (inline f: (Int, Int) => A): F[A] = reduce[A](0 until x, 0 until y)(g)(f)
  inline def reduce[A](inline x: Int, inline y: Int, inline z: Int)
  /*               */ (using inline o: O)
  /*               */ (inline g: (A, A) => A)
  /*               */ (inline f: (Int, Int, Int) => A): F[A] = reduce[A](0 until x, 0 until y, 0 until z)(g)(f)
}

class Device[F[_], O <: CompileOptions](d: polyregion.jvm.runtime.Device)(f: Suspend[F])
    extends DeviceQueue[F, O](d, d.createQueue())(f) {
  def jit: Device[F, CompileOptions.JIT] = ???

  def name: String                      = d.name
  def properties(): Map[String, String] = d.properties.map(p => p.key -> p.value).toMap

  def queue: DeviceQueue[F, O] = ???

}

class FutureRuntime[O <: CompileOptions](r: polyregion.jvm.runtime.Runtime) extends Runtime[Future, O](r)(suspendFuture)
class FutureDevice[O <: CompileOptions](d: polyregion.jvm.runtime.Device)   extends Device[Future, O](d)(suspendFuture)

//
//trait CUDADevice[F[_]] extends Device[F, CompileOptions.AOT[CUDAArch]]
//
//class CUDARuntime[F[_]] extends Runtime[F, CompileOptions.AOT[CUDAArch]] {
//  override inline def name: String             = "CUDA"
//  override def properties: Map[String, String] = Map.empty
//  override def devices: Vector[CUDADevice[F]]  = Vector.empty
//  def jit: Runtime[F, CompileOptions.JIT]      = ???
//}

//

val HostRuntime: Device[Future, CompileOptions] = ???
val Native: Device[Future, CompileOptions]      = ???
def a = {
  given CompileOptions = ???

  Native.foreach(2)(i => ())

  val q = Native.queue

  q.foreach(2)(i => ())
}
