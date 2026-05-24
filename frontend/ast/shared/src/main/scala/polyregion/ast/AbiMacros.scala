package polyregion.ast

import scala.quoted.*

object AbiMacros {

  transparent inline def cName[T <: Singleton](inline prefix: String): String = ${ cNameImpl[T]('prefix) }

  private def cNameImpl[T: Type](prefix: Expr[String])(using Quotes): Expr[String] =
    Expr(prefix.valueOrAbort + pascalToSnake(simpleObjectName[T]))

  private def simpleObjectName[T: Type](using Quotes): String =
    quotes.reflect.TypeRepr.of[T].typeSymbol.name.stripSuffix("$")

  private[ast] def pascalToSnake(s: String): String =
    s.iterator.zipWithIndex.map { case (c, i) =>
      if (c.isUpper && i > 0) "_" + c.toLower
      else if (c.isUpper) c.toLower.toString
      else c.toString
    }.mkString
}
