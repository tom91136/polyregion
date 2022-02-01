package polyregion.compiler

import polyregion.Buffer
import polyregion.ast.{PolyAst => p}
import polyregion.*

object Symbols {
  val JavaLang      = "java" :: "lang" :: Nil
  val Scala         = "scala" :: Nil
  val ScalaMath     = "scala" :: "math" :: "package$" :: Nil
  val JavaMath      = "java" :: "lang" :: "Math$" :: Nil
  val SeqMutableOps = "scala" :: "collection" :: "mutable" :: "SeqOps" :: Nil
  val SeqOps        = "scala" :: "collection" :: "SeqOps" :: Nil
  val Buffer        = p.Sym[Buffer[_]]
}
