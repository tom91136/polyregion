package polyregion.benchmarks

import org.openjdk.jmh.runner.Runner
import org.openjdk.jmh.runner.options.{OptionsBuilder, TimeValue}

object Main {

  final val Iterations = 3
  final val Forks      = 1
  final val Time       = TimeValue.seconds(1)

  def main(args: Array[String]): Unit = {
    val opts = new OptionsBuilder()
      .warmupTime(Time)
      .warmupIterations(Iterations)
      .measurementTime(Time)
      .measurementIterations(Iterations)
      .forks(Forks)
      .include(classOf[Stream].getCanonicalName)
      .shouldFailOnError(true)
      .build()

    new Runner(opts).run()
  }

}
