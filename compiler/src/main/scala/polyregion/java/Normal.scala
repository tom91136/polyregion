package polyregion.java

import polyregion.Outside

import javax.annotation.processing.{ProcessingEnvironment, RoundEnvironment}
import javax.lang.model.element.TypeElement
import javax.tools.Diagnostic.Kind
import scala.collection.immutable.ArraySeq

object Normal{

  def handle(annotations: _root_.java.util.Set[_ <: TypeElement], roundEnv: RoundEnvironment, processingEnv: ProcessingEnvironment): Boolean = {

    Outside.foo
//    println(System.getProperty("java.class.path"))
//    println(classOf[cats.Functor[?]])
//    println(scala.collection.immutable.Vector.apply(1))
//    println(ArraySeq(1))
    import _root_.scala.jdk.CollectionConverters.*

    for (annotation <- annotations.asScala) {
      val annotatedElements = roundEnv.getElementsAnnotatedWith(annotation)
      for (element <- annotatedElements.asScala) {
        processingEnv.getMessager.printMessage(Kind.WARNING, "Got elem: " + element.getSimpleName.toString)
      }
    }
    true
  }

}