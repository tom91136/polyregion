package polyregion.javalang

import com.sun.tools.javac.code.Symbol as JSymbol

import scala.jdk.CollectionConverters.*

object InterfaceStub {

  final val Annotation = "polyregion.javalang.Interface"

  def identityOf(sym: JSymbol): Option[String] =
    sym.getAnnotationMirrors.asScala.collectFirst {
      case annotation if annotation.getAnnotationType.toString == Annotation =>
        annotation.getElementValues.asScala.values.headOption.map(_.getValue.toString)
    }.flatten
}
