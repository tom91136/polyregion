package polyregion.examples

import polyregion.scalalang.*
import polyregion.scalalang.blocking.*

object JitSmoke {
  def main(args: Array[String]): Unit = {
    val r: Int = Host.jit.task[Config[Target.Host.type, Opt.O0], Int] {
      val a = 20
      val b = 22
      a + b
    }
    println(s"host r=$r")
    if (r != 42) { System.err.println(s"FAIL: expected 42, got $r"); sys.exit(1) }
    println("host JIT OK")
  }
}
