package polyregion.scalalang

import polyregion.jvm.runtime as rt

// invoked exactly once, possibly on the launching thread before the launch returns
type Callback[A] = Either[Throwable, A] => Unit

object Platforms {
  val platforms = rt.Platforms.create()
  sys.addShutdownHook(platforms.close())
  export platforms.*
}
