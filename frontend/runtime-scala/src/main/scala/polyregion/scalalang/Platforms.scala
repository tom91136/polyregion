package polyregion.scalalang

import polyregion.jvm.runtime as rt

object Platforms {
  val platforms: rt.Platforms = rt.Platforms.create()
  sys.addShutdownHook(platforms.close())
  export platforms.*
}
