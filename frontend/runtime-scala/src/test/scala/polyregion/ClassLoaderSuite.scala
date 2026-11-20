package polyregion

import java.net.{URL, URLClassLoader}
import java.nio.ByteBuffer

class ClassLoaderSuite extends munit.FunSuite {

  private val bindings: Array[URL] =
    Array(classOf[polyregion.jvm.Loader].getProtectionDomain.getCodeSource.getLocation)

  private def isolated(): URLClassLoader = new URLClassLoader(bindings, ClassLoader.getPlatformClassLoader)

  private class Session(val loader: URLClassLoader) {
    private def call(className: String, method: String, instance: Option[AnyRef], args: (Class[?], AnyRef)*): AnyRef =
      Class
        .forName(className, true, loader)
        .getMethod(method, args.map(_._1)*)
        .invoke(instance.orNull, args.map(_._2)*)

    private def construct(className: String, factory: String): AnyRef = call(className, factory, None)

    private def close(instance: AnyRef): Unit = instance.getClass.getMethod("close").invoke(instance)

    private def withCompiler[A](f: AnyRef => A): A = {
      val compiler = construct("polyregion.jvm.compiler.Compiler", "create")
      try f(compiler)
      finally close(compiler)
    }

    private def withPlatforms[A](f: AnyRef => A): A = {
      val platforms = construct("polyregion.jvm.runtime.Platforms", "create")
      try f(platforms)
      finally close(platforms)
    }

    def compilerTriplet(): String =
      withCompiler(c => c.getClass.getMethod("hostTriplet").invoke(c).asInstanceOf[String])

    def platformsAddress(): Long =
      withPlatforms { p =>
        val buffer = ByteBuffer.allocateDirect(8)
        p.getClass.getMethod("pointerOfDirectBuffer", classOf[java.nio.Buffer]).invoke(p, buffer).asInstanceOf[Long]
      }

    def dynamicPlatform(): AnyRef =
      withPlatforms(p => p.getClass.getMethod("Dynamic").invoke(p))

    def contextPlatformCount(): Int =
      call("polyregion.jvm.runtime.Context", "platforms", None).asInstanceOf[Array[?]].length

    def openCompiler(): AnyRef       = construct("polyregion.jvm.compiler.Compiler", "create")
    def openPlatforms(): AnyRef      = construct("polyregion.jvm.runtime.Platforms", "create")
    def shut(instance: AnyRef): Unit = close(instance)
  }

  test("polyc-JNI binds natives in two classloaders") {
    val (a, b) = (new Session(isolated()), new Session(isolated()))
    val first  = a.openCompiler()
    try {
      assert(a.compilerTriplet().nonEmpty)
      assert(b.compilerTriplet().nonEmpty)
      assertNotEquals(a.loader, b.loader)
    } finally a.shut(first)
  }

  test("polyinvoke-JNI binds natives in two classloaders") {
    val (a, b) = (new Session(isolated()), new Session(isolated()))
    val first  = a.openPlatforms()
    try {
      assertNotEquals(a.platformsAddress(), 0L)
      assertNotEquals(b.platformsAddress(), 0L)
      assert(a.dynamicPlatform().getClass.getClassLoader eq a.loader)
      assert(b.dynamicPlatform().getClass.getClassLoader eq b.loader)
      assert(a.dynamicPlatform().getClass.getClassLoader eq a.loader)
    } finally a.shut(first)
  }

  test("polyrt-JNI binds natives in two classloaders") {
    val (a, b) = (new Session(isolated()), new Session(isolated()))
    assertEquals(b.contextPlatformCount(), a.contextPlatformCount())
  }

  test("a classloader keeps its natives after another classloader releases the library") {
    val (a, b)   = (new Session(isolated()), new Session(isolated()))
    val (ca, cb) = (a.openCompiler(), b.openCompiler())
    a.shut(ca)
    try assert(b.compilerTriplet().nonEmpty)
    finally b.shut(cb)
  }

  test("a library reloads in a fresh classloader after every holder released it") {
    val a = new Session(isolated())
    assert(a.compilerTriplet().nonEmpty)
    val b = new Session(isolated())
    assert(b.compilerTriplet().nonEmpty)
    assert(new Session(isolated()).compilerTriplet().nonEmpty)
  }
}
