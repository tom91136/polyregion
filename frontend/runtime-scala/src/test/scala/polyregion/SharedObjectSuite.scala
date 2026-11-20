package polyregion

import polyregion.jvm.runtime.Context

object SharedObjectProbe {
  def main(args: Array[String]): Unit =
    for {
      platform <- Context.platforms() if platform.name.contains("SharedObject")
      device   <- platform.devices
    }
      println(s"shared-object-format=${device.moduleFormat}")
}

class SharedObjectSuite extends munit.FunSuite {

  private val DsoFormat = 3

  test("a shared object device without a C compiler accepts only shared object images") {
    val (code, output) = ChildJvm.run(
      "polyregion.SharedObjectProbe",
      Map(
        "POLYINVOKE_C_COMPILER"       -> "/nonexistent/cc",
        "POLYINVOKE_DISABLE_BACKENDS" -> "HSA,HIP,LevelZero,OpenCL,Vulkan,CUDA"
      )
    )
    assertEquals(code, 0, output)
    assert(output.contains(s"shared-object-format=$DsoFormat"), output)
  }
}
