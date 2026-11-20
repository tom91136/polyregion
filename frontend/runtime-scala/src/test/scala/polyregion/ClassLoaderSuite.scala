package polyregion

import polyregion.IsolatedBindings.isolated

class ClassLoaderSuite extends munit.FunSuite {

  test("polyc-JNI binds natives in two classloaders") {
    val (a, b) = (isolated(), isolated())
    val first  = a.openCompiler()
    try {
      assert(a.compilerTriplet().nonEmpty)
      assert(b.compilerTriplet().nonEmpty)
      assertNotEquals(a.loader, b.loader)
    } finally a.close(first)
  }

  test("polyinvoke-JNI binds natives in two classloaders") {
    val (a, b) = (isolated(), isolated())
    val first  = a.openPlatforms()
    try {
      assertNotEquals(a.platformsAddress(), 0L)
      assertNotEquals(b.platformsAddress(), 0L)
      assert(a.dynamicPlatform().getClass.getClassLoader eq a.loader)
      assert(b.dynamicPlatform().getClass.getClassLoader eq b.loader)
    } finally a.close(first)
  }

  test("polyrt-JNI binds natives in two classloaders") {
    val (a, b) = (isolated(), isolated())
    assertEquals(b.contextPlatformCount(), a.contextPlatformCount())
  }

  test("a classloader keeps its natives after another classloader releases the library") {
    val (a, b)   = (isolated(), isolated())
    val (ca, cb) = (a.openCompiler(), b.openCompiler())
    a.close(ca)
    try assert(b.compilerTriplet().nonEmpty)
    finally b.close(cb)
  }

  test("a library reloads in a fresh classloader after every holder released it") {
    assert(isolated().compilerTriplet().nonEmpty)
    assert(isolated().compilerTriplet().nonEmpty)
    assert(isolated().compilerTriplet().nonEmpty)
  }

  private def lifecycle(scenario: String): Unit = {
    val (code, output) = ChildJvm.run("polyregion.ClassLoaderLifecycle", Map("LIFECYCLE_SCENARIO" -> scenario))
    assertEquals(code, 0, output)
  }

  test("closing a library twice keeps it loaded for another classloader") {
    lifecycle("double-close")
  }

  test("a stray native release keeps the library loaded for another classloader") {
    lifecycle("native-double-release")
  }

  test("loading still works after a classloader that never closed its library is collected") {
    lifecycle("dead-owner")
  }

  test("a library whose class is missing from the loader fails to load with an exception") {
    lifecycle("missing-class")
  }
}
