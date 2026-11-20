package polyregion.scalalang

import polyregion.jvm.runtime as rt

import java.util.concurrent.ConcurrentHashMap

object Contexts {

  private val created = ConcurrentHashMap[(Int, Int), rt.Context]()

  def platformIndex(platformName: String): Int = rt.Context.platforms().filter(_.name == platformName) match {
    case Array(platform) => platform.index
    case matches =>
      throw new IllegalArgumentException(s"Expected one polyrt platform named `$platformName`, found ${matches.length}")
  }

  def of(platformIndex: Int, device: Int): rt.Context =
    created.computeIfAbsent((platformIndex, device), k => rt.Context.create(k._1, k._2))
}
