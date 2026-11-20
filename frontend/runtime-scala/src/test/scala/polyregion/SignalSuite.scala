package polyregion

import polyregion.jvm.runtime.Context

import java.nio.file.{Files, Path}

object SignalProbe {

  private final class Box(val value: Int)

  private def read(box: Box): Int =
    try box.value
    catch case _: NullPointerException => -1

  def main(args: Array[String]): Unit = {
    Context.platforms()
    val loaded = Files.readString(Path.of("/proc/self/maps")).contains("libpocl")
    println(s"pocl=$loaded")
    val box = new Box(1)
    var sum = 0L
    var i   = 0
    while (i < 1_000_000) {
      sum += read(box)
      i += 1
    }
    var caught = 0
    i = 0
    while (i < 2_000_000) {
      if (read(if (i % 64 == 0) null else box) < 0) caught += 1
      i += 1
    }
    println(s"survived caught=$caught")
    sys.exit(if (caught == (2_000_000 + 63) / 64 && sum == 1_000_000L) 0 else 1)
  }
}

class SignalSuite extends munit.FunSuite {

  test("a JVM that enumerates OpenCL keeps its own SIGSEGV handler") {
    val (code, output) = ChildJvm.run(
      "polyregion.SignalProbe",
      ChildJvm.pocl() + ("POLYINVOKE_DISABLE_BACKENDS" -> "HSA,HIP,LevelZero")
    )
    assume(output.contains("pocl=true") || code != 0, s"pocl is not loadable here:\n$output")
    assertEquals(code, 0, output)
    assert(output.contains("survived"), output)
  }
}
