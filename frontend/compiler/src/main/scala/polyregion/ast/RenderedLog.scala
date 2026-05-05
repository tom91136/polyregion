package polyregion.ast

import scala.annotation.targetName
import scala.collection.mutable.ArrayBuffer
import scala.util.Try

case class RenderedLog(name: String, lines: ArrayBuffer[(String, Vector[String]) | RenderedLog]) extends Log {

  @targetName("append") infix def subLog(name: String): RenderedLog = {
    val sub = RenderedLog(name); lines += sub; sub
  }

  @targetName("append") infix def +=(log: RenderedLog): Unit          = lines += log
  @targetName("appendAll") infix def ++=(log: Seq[RenderedLog]): Unit = lines ++= log
  def info(message: String, details: String*): Unit                   = lines += (message -> details.toVector)

//  def info(message: String, details: String*): Result[Log] = info_(message, details*).success

  def render(nesting: Int = 0): Vector[String] =
    Try {
      val colour = RenderedLog.Colours(nesting % RenderedLog.Colours.size)
      val attr   = colour ++ fansi.Reversed.On ++ fansi.Bold.On
      val indent = colour("┃ ")

      ((colour("┏━") ++ attr(s" ${name} ") ++ colour("")) +: lines.toVector
        .flatMap {
          case log: RenderedLog => log.render(nesting + 1).map(indent ++ _)
          case (line, details) =>
            ((colour ++ fansi.Underlined.On)(s"▓ $line ▓")) +: details.flatMap { l =>
              l.linesIterator.toList match {
                case x :: xs =>
                  ((colour("┃ ╰ ") ++ s"$x") :: xs.map(x => indent ++ s"  $x")).toVector
                case Nil => Vector.empty
              }
            }
        } :+ colour(s"┗━${"━" * (name.length + 2)}"))
        .map(_.render)
    }.recover { case e: Exception =>
      Vector(s"Cannot render:${e}")
    }.get
}

object RenderedLog {
  private val Colours: Vector[fansi.Attr] = Vector(
    // fansi.Color.Red,
    fansi.Color.Green,
    fansi.Color.Yellow,
    fansi.Color.Blue,
    fansi.Color.Magenta,
    fansi.Color.Cyan,
    fansi.Color.LightGray,
    fansi.Color.DarkGray,
    fansi.Color.LightRed,
    fansi.Color.LightGreen,
    fansi.Color.LightYellow,
    fansi.Color.LightBlue,
    fansi.Color.LightMagenta,
    fansi.Color.LightCyan
  )

  def apply(name: String): RenderedLog = RenderedLog(name, ArrayBuffer.empty)
}
