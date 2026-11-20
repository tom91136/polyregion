package polyregion

import polyregion.IsolatedBindings.isolated

import java.lang.ref.WeakReference

object ClassLoaderLifecycle {

  private def leak(): WeakReference[ClassLoader] = {
    val session = isolated()
    session.openLibrary("polyc-JNI")
    new WeakReference(session.loader)
  }

  def main(args: Array[String]): Unit =
    System.getenv("LIFECYCLE_SCENARIO") match {
      case "double-close" =>
        val (a, b)   = (isolated(), isolated())
        val (la, lb) = (a.openLibrary("polyc-JNI"), b.openLibrary("polyc-JNI"))
        a.close(la)
        a.close(la)
        require(b.hostTriplet0().nonEmpty)
        b.close(lb)
      case "native-double-release" =>
        val (a, b)   = (isolated(), isolated())
        val (la, lb) = (a.openLibrary("polyc-JNI"), b.openLibrary("polyc-JNI"))
        a.releaseNative(a.handle(la))
        a.releaseNative(a.handle(la))
        require(b.hostTriplet0().nonEmpty)
        b.close(lb)
      case "dead-owner" =>
        val ref = leak()
        var i   = 0
        while (ref.get != null && i < 200) {
          System.gc()
          Thread.sleep(25)
          i += 1
        }
        require(ref.get == null, "the leaked classloader was not collected")
        isolated().openLibrary("polyinvoke-JNI")
      case "missing-class" =>
        val session = isolated(Set("polyregion.jvm.compiler.Compiler"))
        val failure =
          try { session.openLibrary("polyc-JNI"); None }
          catch { case e: Throwable => Some(e) }
        require(failure.nonEmpty, "loading succeeded without the Compiler class")
      case other => sys.error(s"unknown scenario $other")
    }
}
