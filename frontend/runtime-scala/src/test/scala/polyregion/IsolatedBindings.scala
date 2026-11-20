package polyregion

import java.net.{URL, URLClassLoader}
import java.nio.ByteBuffer
import java.nio.file.Paths

object IsolatedBindings {

  val bindings: Array[URL] =
    Array(classOf[polyregion.jvm.Loader].getProtectionDomain.getCodeSource.getLocation)

  class Loader(blocked: Set[String]) extends URLClassLoader(bindings, ClassLoader.getPlatformClassLoader) {
    override def loadClass(name: String, resolve: Boolean): Class[?] =
      if (blocked.contains(name)) throw new ClassNotFoundException(name) else super.loadClass(name, resolve)
  }

  def isolated(blocked: Set[String] = Set.empty): Session = new Session(new Loader(blocked))

  class Session(val loader: ClassLoader) {
    private def forName(className: String): Class[?] = Class.forName(className, true, loader)

    private def call(className: String, method: String, args: (Class[?], AnyRef)*): AnyRef =
      forName(className).getMethod(method, args.map(_._1)*).invoke(null, args.map(_._2)*)

    def close(instance: AnyRef): Unit = instance.getClass.getMethod("close").invoke(instance)

    def openCompiler(): AnyRef  = call("polyregion.jvm.compiler.Compiler", "create")
    def openPlatforms(): AnyRef = call("polyregion.jvm.runtime.Platforms", "create")

    def openLibrary(name: String): AnyRef = {
      val file = call(
        "polyregion.jvm.Loader",
        "searchAndCopyResourceIfNeeded",
        classOf[String]             -> System.mapLibraryName(name),
        classOf[java.nio.file.Path] -> Paths.get(".")
      ).asInstanceOf[java.util.Optional[java.nio.file.Path]].get.toAbsolutePath.toString
      call("polyregion.jvm.NativeLibrary", "load", classOf[String] -> file)
    }

    def handle(library: AnyRef): Long = library.getClass.getField("handle").getLong(library)

    def releaseNative(handle: Long): Unit = {
      val method = forName("polyregion.jvm.Natives").getDeclaredMethod("dynamicLibraryRelease0", classOf[Long])
      method.setAccessible(true)
      method.invoke(null, Long.box(handle))
    }

    def hostTriplet(compiler: AnyRef): String =
      compiler.getClass.getMethod("hostTriplet").invoke(compiler).asInstanceOf[String]

    def hostTriplet0(): String = {
      val method = forName("polyregion.jvm.compiler.Compiler").getDeclaredMethod("hostTriplet0")
      method.setAccessible(true)
      method.invoke(null).asInstanceOf[String]
    }

    def withCompiler[A](f: AnyRef => A): A = {
      val compiler = openCompiler()
      try f(compiler)
      finally close(compiler)
    }

    def withPlatforms[A](f: AnyRef => A): A = {
      val platforms = openPlatforms()
      try f(platforms)
      finally close(platforms)
    }

    def compilerTriplet(): String = withCompiler(hostTriplet)

    def platformsAddress(): Long =
      withPlatforms { p =>
        val buffer = ByteBuffer.allocateDirect(8)
        p.getClass.getMethod("pointerOfDirectBuffer", classOf[java.nio.Buffer]).invoke(p, buffer).asInstanceOf[Long]
      }

    def dynamicPlatform(): AnyRef = withPlatforms(p => p.getClass.getMethod("Dynamic").invoke(p))

    def contextPlatformCount(): Int = call("polyregion.jvm.runtime.Context", "platforms").asInstanceOf[Array[?]].length
  }
}
