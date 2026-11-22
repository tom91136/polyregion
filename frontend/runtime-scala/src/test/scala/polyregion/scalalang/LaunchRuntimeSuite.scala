package polyregion.scalalang

import polyregion.jvm.runtime as rt

import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.{CountDownLatch, TimeUnit}

class LaunchRuntimeSuite extends munit.FunSuite {

  private class FixedGroup(program: String, hostOnly: Boolean, refused: Option[rt.LoadOutcome])
      extends LaunchRuntime.Group(program, Array.empty[String], hostOnly) {
    override def refusal(ctx: rt.Context): Option[rt.LoadOutcome] = refused
  }

  private def group(program: String, hostOnly: Boolean): LaunchRuntime.Group = FixedGroup(program, hostOnly, None)

  private def refusedSite(outcomes: rt.LoadOutcome*): LaunchRuntime.Site = LaunchRuntime.Site(
    s"refused-${outcomes.mkString("-")}",
    outcomes.zipWithIndex.map((o, i) => FixedGroup(s"g$i", false, Some(o))).toArray,
    polyregion.jvm.twin.Layout.encodeAll(java.util.List.of())
  )

  test("a concurrent first use waits for the load in progress") {
    val started    = CountDownLatch(1)
    val release    = CountDownLatch(1)
    val loaded     = AtomicBoolean(false)
    val seenLoaded = AtomicBoolean(false)
    val loader = Thread(() =>
      LaunchRuntime.ensureLoaded("concurrent-first-use", false) {
        started.countDown()
        assert(release.await(10, TimeUnit.SECONDS))
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

  test("a site refused only for lacking features raises a capability gap") {
    import rt.LoadOutcome.*
    intercept[rt.CapabilityGapException](refusedSite(LACKS).select(null))
    intercept[rt.CapabilityGapException](refusedSite(FORMAT, LACKS).select(null))
    intercept[rt.CapabilityGapException](refusedSite(LACKS, LACKS, FORMAT).select(null))
  }

  test("a site refused on format alone or blocked anywhere raises a plain runtime exception") {
    import rt.LoadOutcome.*
    List(refusedSite(FORMAT), refusedSite(LACKS, UNPREPARED), refusedSite(NO_DEVICE, LACKS), refusedSite()).foreach {
      site =>
        val e = intercept[rt.PolyregionRuntimeException](site.select(null))
        assert(!e.isInstanceOf[rt.CapabilityGapException], e.toString)
    }
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
