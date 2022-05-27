package polyregion.examples

import java.nio.{ByteBuffer, ByteOrder}

object CheckApi {

  def main(args: Array[String]): Unit = {

    import polyregion.jvm.runtime.*

    Runtimes.load()

    val rs = Array(Runtimes.OpenCL())

    rs.foreach { r =>
      println(r)
      val xs = r.devices()
      println(xs.map(x => s"\t$x\n${x.properties().map(x => s"\t\t$x").mkString("\n")}").mkString("\n"))

      xs.foreach { d0 =>

        println(s"D0 = $d0")
        val m = d0.malloc(10, Access.RW)
        println(s"alloc = ${m}")
        d0.free(m)

        val kern = "kernel void add(int a, int b, global int* xs ){ xs[get_global_id(0)] = get_global_id(0) + a + b; } "
        d0.loadModule("a", kern.getBytes)

        val xs = d0.malloc(java.lang.Integer.BYTES, Access.RW)

        val data = ByteBuffer.allocate(java.lang.Integer.BYTES * 3).order(ByteOrder.nativeOrder())

        val q0 = d0.createQueue();

        q0.enqueueInvokeAsync(
          "a",
          "add",
          java.util.Arrays.asList(
            Arg.of(1),
            Arg.of(2),
            new Arg(Type.INT, data)
          ),
          new Arg(Type.VOID, ByteBuffer.allocate(1)),
          new Policy(new Dim3(1, 1, 0)),
          () => {
            println("Done!")
            val x = Array.ofDim[Int](3)
            data.asIntBuffer().get(x)
            println(x.toList)
          }
        )

//        q0.close();
//        d0.close();
//          r.close()
      }

//      r.close()
    }

//    val q = d0.createQueue()
//    d0.createQueue()
//    println(q)
//    d0.loadModule("", Array())
    println("Exit")

  }

}
