package polyregion.java

import com.sun.source.util.Trees
import polyregion.Outside

import javax.annotation.processing.{ProcessingEnvironment, RoundEnvironment}
import javax.lang.model.element.TypeElement
import javax.tools.Diagnostic.Kind
import scala.collection.immutable.ArraySeq
import scala.jdk.CollectionConverters.*

object Compiler {

  def compile(
      trees: Trees,
      annotations: java.util.Set[_ <: TypeElement],
      roundEnv: RoundEnvironment,
      processingEnv: ProcessingEnvironment
  ): Boolean = {
    for (annotation <- annotations.asScala) {
      val annotatedElements = roundEnv.getElementsAnnotatedWith(annotation)
      for (element <- annotatedElements.asScala) {

        println(s"Tree=${trees.getTree(element)}")



        processingEnv.getMessager.printMessage(Kind.WARNING, "Got elem: " + element.getSimpleName.toString)
      }
    }
    true
  }

}
