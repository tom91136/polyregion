package polyregion.ast

object Env {

  inline val PolyregionDebug   = "POLYREGION_DEBUG"
  inline val PolyregionPassLog = "POLYREGION_PASS_LOG"

  def debugEnabled: Boolean = sys.env.contains(PolyregionDebug)

  inline def debug(inline s: => Any): Unit = if (debugEnabled) println(s)
}
