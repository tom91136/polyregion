package polyregion.scala

import polyregion.ast.{PolyAst as p, *}
import polyregion.scala.Buffer

import scala.reflect.ClassTag

object Symbols {
  val JavaLang      = "java" :: "lang" :: Nil
  val Scala         = "scala" :: Nil
  val ScalaMath     = "scala" :: "math" :: "package$" :: Nil
  val JavaMath      = "java" :: "lang" :: "Math$" :: Nil
  val SeqMutableOps = "scala" :: "collection" :: "mutable" :: "SeqOps" :: Nil
  val SeqOps        = "scala" :: "collection" :: "SeqOps" :: Nil

  val Buffer   = p.Sym[Buffer[_]]
  val ClassTag = p.Sym[ClassTag[_]]

  val Array       = p.Sym("scala" :: "Array" :: Nil)
  val ArrayModule = "scala" :: "Array$" :: Nil

  // Array don't delegate to aneeds a ClassTag which it can't have for *

  // val Array         = p.Sym(classOf[Array[Int]])

  // println(">"+Symbols.ClassTag)
  // println(">"+Symbols.Array)

}
