package polyregion.benchmarks

import org.openjdk.jmh.annotations._
import java.util.concurrent.TimeUnit
import polyregion.compiletime._
import polyregion.Runtime._

@OutputTimeUnit(TimeUnit.MILLISECONDS)
@BenchmarkMode(Array(Mode.AverageTime))
class Stream {

  @State(Scope.Thread) class Data {

    @Param(
      Array(
        "1000",
        "100000"
      )
    )
    var size: Int = _

    var a: Buffer[Float] = _
    var b: Buffer[Float] = _
    var c: Buffer[Float] = _

    @Setup(Level.Iteration) def setup(): Unit = {
      a = Buffer.tabulate(size)(_.toFloat)
      b = Buffer.tabulate(size)(_.toFloat)
      c = Buffer.tabulate(size)(_.toFloat)
    }

  }

  @Benchmark
  def jvmCopy(data: Data) =
    foreachJVM(0 to data.size)(i => data.a(i) = data.b(i))

  @Benchmark
  def polyregionCopy(data: Data) = {
    val a = data.a
    val b = data.b
    val c = data.c
    foreach(0 to data.size)(i => a(i) = b(i))
  }

}
