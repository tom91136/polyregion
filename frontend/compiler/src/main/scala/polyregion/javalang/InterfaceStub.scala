package polyregion.javalang

import com.sun.tools.javac.code.Symbol as JSymbol

import scala.jdk.CollectionConverters.*

object InterfaceStub {

  final val Annotation = "polyregion.javalang.Interface"
  final val Sugar      = "polyregion.javalang.LaunchSugar"

  final val Descriptor: String      = s"L${Annotation.replace('.', '/')};"
  final val SugarDescriptor: String = s"L${Sugar.replace('.', '/')};"

  def refusal(callee: String, identity: String): String =
    s"`$callee` launches `$identity` and cannot be called inside an offload region"

  def identityOf(sym: JSymbol): Option[String] =
    sym.getAnnotationMirrors.asScala.collectFirst {
      case annotation if annotation.getAnnotationType.toString == Annotation =>
        annotation.getElementValues.asScala.values.headOption.map(_.getValue.toString)
    }.flatten

  def sugarOf(sym: JSymbol): Option[(String, Int, Int)] =
    sym.getAnnotationMirrors.asScala.collectFirst {
      case annotation if annotation.getAnnotationType.toString == Sugar =>
        val values = annotation.getElementValues.asScala.map { (key, value) =>
          key.getSimpleName.toString -> value.getValue.toString
        }.toMap
        for {
          identity <- values.get("value")
          rank     <- values.get("rank").map(_.toInt)
          map      <- values.get("map").map(_.toInt)
        } yield (identity, rank, map)
    }.flatten
}
