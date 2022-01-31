package polyregion.compiler

import polyregion.ast.PolyAst
import polyregion.Buffer
import polyregion.internal.*

object Symbols {
  val JavaLang      = "java" :: "lang" :: Nil
  val Scala         = "scala" :: Nil
  val ScalaMath     = "scala" :: "math" :: "package$" :: Nil
  val JavaMath      = "java" :: "lang" :: "Math$" :: Nil
  val SeqMutableOps = "scala" :: "collection" :: "mutable" :: "SeqOps" :: Nil
  val SeqOps        = "scala" :: "collection" :: "SeqOps" :: Nil
  val Buffer        = PolyAst.Sym[Buffer[_]]
}
