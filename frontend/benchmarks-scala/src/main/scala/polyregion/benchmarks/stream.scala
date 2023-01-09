package polyregion.benchmarks

import org.openjdk.jmh.annotations.*
import java.util.concurrent.TimeUnit
import polyregion.scalalang.compiletime.*
import polyregion.scalalang.*
import concurrent.ExecutionContext.Implicits.global
import net.openhft.affinity.AffinityLock

@State(Scope.Thread) class StreamData {

  @Param(
    Array(
//      "1000",
//      "100000"
      "33554432"
//      "67108864"
    )
  )
  var size: Int = _

  @Param(
    Array(
      "10"
    )
  )
  var iter: Int = _

  @Param(
    Array(
      "0.4"
    )
  )
  var scalar: Double = _

  var a: Buffer[Double] = _
  var b: Buffer[Double] = _
  var c: Buffer[Double] = _

  var a_ : Array[Double] = _
  var b_ : Array[Double] = _
  var c_ : Array[Double] = _

  @Setup(Level.Iteration) def setup(): Unit = {
    // val l = AffinityLock.acquireLock(0)
    a = Buffer.fill(size)(0.1)
    b = Buffer.fill(size)(0.2)
    c = Buffer.fill(size)(0.0)

    a_ = Array.fill(size)(0.1)
    b_ = Array.fill(size)(0.2)
    c_ = Array.fill(size)(0.0)
    // l.close()
  }

}

@OutputTimeUnit(TimeUnit.MILLISECONDS)
@BenchmarkMode(Array(Mode.AverageTime))
class Stream {

//  @Benchmark
//  def copyJVM(data: StreamData) =
//    foreachJVMPar(0 until data.size)(i => data.c_(i) = data.a_(i))

//  @Benchmark
//  def mulJVM(data: StreamData) =
//    foreachJVMPar(0 until data.size)(i => data.b_(i) = data.scalar * data.c_(i))

//  @Benchmark
//  def addJVM(data: StreamData) =
//    foreachJVMPar(0 until data.size)(i => data.c_(i) = data.a_(i) + data.b_(i))

//  @Benchmark
//  def triadJVM(data: StreamData) =
//    foreachJVMPar(0 until data.size)(i => data.a_(i) = data.b_(i) + (data.scalar * data.b_(i)))

//  @Benchmark
//  def nstreamJVM(data: StreamData) =
//    foreachJVMPar(0 until data.size)(i => data.a_(i) = data.b_(i) * data.scalar * data.b_(i))

//  @Benchmark
//  def copyPolyregion(data: StreamData) =
//    foreachPar(0 until data.size)(i => data.c(i) = data.a(i))

//  @Benchmark
//  def mulPolyregion(data: StreamData) =
//    foreachPar(0 until data.size)(i => data.b(i) = data.scalar * data.c(i))

//  @Benchmark
//  def addPolyregion(data: StreamData) =
//    foreachPar(0 until data.size)(i => data.c(i) = data.a(i) + data.b(i))

//  @Benchmark
//  def triadPolyregion(data: StreamData) =
//    foreachPar(0 until data.size)(i => data.a(i) = data.b(i) + (data.scalar * data.b(i)))

//  @Benchmark
//  def nstreamPolyregion(data: StreamData) =
//    foreachPar(0 until data.size)(i => data.a(i) = data.b(i) * data.scalar * data.b(i))

  @Benchmark
  def streamJVM(data: StreamData) = {
    // val a   = AffinityLock.acquireLock(0)
    var res = 0d
    for (_ <- 0 until data.iter) {
      foreachJVMPar(0 until data.size)(i => data.c_(i) = data.a_(i))
      foreachJVMPar(0 until data.size)(i => data.b_(i) = data.scalar * data.c_(i))
      foreachJVMPar(0 until data.size)(i => data.c_(i) = data.a_(i) + data.b_(i))
      foreachJVMPar(0 until data.size)(i => data.a_(i) = data.b_(i) + (data.scalar * data.b_(i)))
      foreachJVMPar(0 until data.size)(i => data.a_(i) = data.b_(i) * data.scalar * data.b_(i))
      res = reduceJVMPar[Double](0 until data.size)(res, i => data.a(i) + data.b(i))(_ + _)
    }
    // a.close()
    res
  }

  @Benchmark
  def streamPolyregion(data: StreamData) = {
    // val a   = AffinityLock.acquireLock(0)
    var res = 0d
    for (_ <- 0 until data.iter) {
      foreachPar(0 until data.size)(i => data.c(i) = data.a(i))
      foreachPar(0 until data.size)(i => data.b(i) = data.scalar * data.c(i))
      foreachPar(0 until data.size)(i => data.c(i) = data.a(i) + data.b(i))
      foreachPar(0 until data.size)(i => data.a(i) = data.b(i) + (data.scalar * data.b(i)))
      foreachPar(0 until data.size)(i => data.a(i) = data.b(i) * data.scalar * data.b(i))
      res = reducePar[Double](0 until data.size)(res, i => data.a(i) + data.b(i))(_ + _)
    }
    // a.close()
    res
  }

}
