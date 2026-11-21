package polyregion.scalalang

object Loaders {
  def under[A](loader: ClassLoader)(f: => A): A = {
    val thread   = Thread.currentThread
    val previous = thread.getContextClassLoader
    thread.setContextClassLoader(loader)
    try f
    finally thread.setContextClassLoader(previous)
  }
}
