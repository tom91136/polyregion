package polyregion.scalalang

import polyregion.jvm.runtime as rt

import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.{CountDownLatch, TimeUnit}

class LaunchRuntimeSuite extends munit.FunSuite {

  private def group(program: String, hostOnly: Boolean): LaunchRuntime.Group =
    new LaunchRuntime.Group(program, Array.empty[String], hostOnly) {
      override def accepts(ctx: rt.Context): Boolean = true
    }

  test("a concurrent first use waits for the load in progress") {
    val started    = CountDownLatch(1)
    val release    = CountDownLatch(1)
    val loaded     = AtomicBoolean(false)
    val seenLoaded = AtomicBoolean(false)
    val loader = Thread(() =>
      LaunchRuntime.ensureLoaded("concurrent-first-use", false) {
        started.countDown()
        release.await()
        loaded.set(true)
      }
    )
    loader.start()
    assert(started.await(10, TimeUnit.SECONDS))
    val waiter = Thread(() =>
      LaunchRuntime.ensureLoaded("concurrent-first-use", false)(fail("the program is loaded twice")) match {
        case () => seenLoaded.set(loaded.get)
      }
    )
    waiter.start()
    waiter.join(500)
    release.countDown()
    loader.join(10000)
    waiter.join(10000)
    assert(seenLoaded.get, "the second caller returned before the program was loaded")
  }

  test("a failed load is retried by the next call") {
    val attempts = java.util.concurrent.atomic.AtomicInteger(0)
    intercept[IllegalStateException](
      LaunchRuntime.ensureLoaded("failed-load", false)(throw IllegalStateException("load failed"))
    )
    LaunchRuntime.ensureLoaded("failed-load", false)(attempts.incrementAndGet())
    assertEquals(attempts.get, 1)
  }

  test("a host-only variant is the fallback after every device variant") {
    val host   = group("host-only", hostOnly = true)
    val device = group("device", hostOnly = false)
    val site =
      LaunchRuntime.Site("ordering", Array(host, device), polyregion.jvm.twin.Layout.encodeAll(java.util.List.of()))
    assertEquals(site.select(null).program, "device")
  }

  test("a launch keeps its scalar arguments reachable until the marshal closes") {
    val marshal = LaunchRuntime.marshal(
      LaunchRuntime.Site("scalars", Array.empty, polyregion.jvm.twin.Layout.encodeAll(java.util.List.of()))
    )
    try {
      LaunchRuntime.int(marshal, 7)
      assertEquals(marshal.stagedBuffers, 1)
    } finally marshal.close()
  }
}
