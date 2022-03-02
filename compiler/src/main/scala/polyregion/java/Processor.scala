package polyregion.java

//import com.google.auto.service.AutoService
import com.sun.source.util.Trees

import _root_.java.lang.reflect.AccessibleObject
import _root_.java.lang.reflect.Field
import _root_.java.lang.reflect.InvocationHandler
import _root_.java.lang.reflect.Method
import _root_.java.lang.reflect.Proxy
import _root_.java.util
import javax.annotation.processing.AbstractProcessor
import javax.annotation.processing.ProcessingEnvironment
import javax.annotation.processing.Processor
import javax.annotation.processing.RoundEnvironment
import javax.annotation.processing.SupportedAnnotationTypes
import javax.annotation.processing.SupportedSourceVersion
import javax.lang.model.SourceVersion
import javax.lang.model.element.Element
import javax.lang.model.element.TypeElement
import javax.tools.Diagnostic
import javax.tools.Diagnostic.Kind
import sun.misc.Unsafe

import scala.annotation.tailrec
import scala.collection.immutable.ArraySeq

object Processor {

  private def getUnsafe = try {
    val theUnsafe = classOf[Unsafe].getDeclaredField("theUnsafe")
    theUnsafe.setAccessible(true)
    theUnsafe.get(null).asInstanceOf[Unsafe]
  } catch {
    case e: Exception =>
      null
  }

  def setAccessible[T <: AccessibleObject](accessor: T): T = {
    accessor.setAccessible(true)
    accessor
  }

  @tailrec def traverseUp[A <: AccessibleObject](cls: Class[?])(f: Class[?] => A): Option[A] =
    cls match {
      case null => None
      case x =>
        try Option(f(x)).map(setAccessible(_))
        catch {
          case _: ReflectiveOperationException => traverseUp(cls.getSuperclass)(f)
        }
    }

  def getMethod(c: Class[_], mName: String, parameterTypes: Class[_]*): Method =
    traverseUp(c)(c => c.getDeclaredMethod(mName, parameterTypes*)).get

  def getField(c: Class[_], fName: String): Field =
    traverseUp(c)(c => c.getDeclaredField(fName)).get

  private def getOwnModule = try {
    val m = getMethod(classOf[Class[_]], "getModule")
    m.invoke(classOf[Processor])
  } catch {
    case e: Exception =>
      null
  }

  /* call public api: ModuleLayer.boot().findModule("jdk.compiler").get();
			  but use reflection because we don't want this code to crash on jdk1.7 and below.
			  In that case, none of this stuff was needed in the first place, so we just exit via
			  the catch block and do nothing.
   */
  private def getJdkCompilerModule = try {
    val cModuleLayer = Class.forName("java.lang.ModuleLayer")
    val mBoot        = cModuleLayer.getDeclaredMethod("boot")
    val bootLayer    = mBoot.invoke(null)
    val cOptional    = Class.forName("java.util.Optional")
    val mFindModule  = cModuleLayer.getDeclaredMethod("findModule", classOf[String])
    val oCompilerO   = mFindModule.invoke(bootLayer, "jdk.compiler")
    cOptional.getDeclaredMethod("get").invoke(oCompilerO)
  } catch {
    case e: Exception =>
      null
  }

  private def getFirstFieldOffset(unsafe: Unsafe) = try
    unsafe.objectFieldOffset(classOf[Nothing].getDeclaredField("first"))
  catch {
    case e: NoSuchFieldException =>
      // can't happen.
      throw new RuntimeException(e)
    case e: SecurityException =>
      // can't happen
      throw new RuntimeException(e)
  }

  def findCls(name: String): Option[Class[_]] =
    try
      Option(Class.forName(name))
    catch { case e: ClassNotFoundException => None }

  /** Useful from jdk9 and up; required from jdk16 and up. This code is supposed to gracefully do nothing on jdk8 and
    * below, as this operation isn't needed there.
    */
  def addOpensForLombok(): Unit = findCls("java.lang.Module").foreach { cModule =>
    val unsafe            = getUnsafe
    val jdkCompilerModule = getJdkCompilerModule
    val ownModule         = getOwnModule
    val allPkgs = Array(
      "com.sun.tools.javac.code",
      "com.sun.tools.javac.comp",
      "com.sun.tools.javac.file",
      "com.sun.tools.javac.main",
      "com.sun.tools.javac.model",
      "com.sun.tools.javac.parser",
      "com.sun.tools.javac.processing",
      "com.sun.tools.javac.tree",
      "com.sun.tools.javac.util",
      "com.sun.tools.javac.jvm"
    )
    try {
      val m                = cModule.getDeclaredMethod("implAddOpens", classOf[String], cModule)
      val firstFieldOffset = getFirstFieldOffset(unsafe)
      unsafe.putBooleanVolatile(m, firstFieldOffset, true)
      for (p <- allPkgs) m.invoke(jdkCompilerModule, p, ownModule)
    } catch {
      case ignore: Exception =>
    }
  }

}

@SupportedAnnotationTypes(Array("polyregion.java.Offload"))
@SupportedSourceVersion(SourceVersion.RELEASE_8)
class Processor extends AbstractProcessor {

  /** Gradle incremental processing
    */
  private def tryGetDelegateField(delegateClass: Class[_], instance: Any) = try
    Processor.getField(delegateClass, "delegate").get(instance)
  catch {
    case e: Exception =>
      null
  }

  /** Kotlin incremental processing
    */
  private def tryGetProcessingEnvField(delegateClass: Class[_], instance: Any) = try
    Processor.getField(delegateClass, "processingEnv").get(instance)
  catch {
    case e: Exception =>
      null
  }

  /** IntelliJ IDEA >= 2020.3
    */
  private def tryGetProxyDelegateToField(delegateClass: Class[_], instance: Any) = try {
    val handler = Proxy.getInvocationHandler(instance)
    Processor.getField(handler.getClass, "val$delegateTo").get(handler)
  } catch {
    case e: Exception =>
      null
  }

  private[polyregion] var t: Trees = _

  override def init(processingEnv: ProcessingEnvironment): Unit = {
    super.init(processingEnv)

    type JavaCEnv = Any // com.sun.tools.javac.processing.JavacProcessingEnvironment

    def getJavacProcessingEnvironment(procEnv: Any): JavaCEnv = {
      Processor.addOpensForLombok()
      if (procEnv.isInstanceOf[JavaCEnv]) return procEnv.asInstanceOf[JavaCEnv]
      // try to find a "delegate" field in the object, and use this to try to obtain a JavacProcessingEnvironment
      var procEnvClass = procEnv.getClass
      while (procEnvClass != null) {
        var delegate = tryGetDelegateField(procEnvClass, procEnv)
        if (delegate == null) delegate = tryGetProxyDelegateToField(procEnvClass, procEnv)
        if (delegate == null) delegate = tryGetProcessingEnvField(procEnvClass, procEnv)
        if (delegate != null) return getJavacProcessingEnvironment(delegate)
        // delegate field was not found, try on superclass

        procEnvClass = procEnvClass.getSuperclass
      }
      processingEnv.getMessager.printMessage(
        Kind.WARNING,
        "Can't get the delegate of the gradle IncrementalProcessingEnvironment. Lombok won't work."
      )
      null
    }
//    val a = getJavacProcessingEnvironment(processingEnv)
    t = Trees.instance(processingEnv)
    processingEnv.getMessager.printMessage(Kind.WARNING, s"In processor, tree=$t")
  }

  override def process(annotations: _root_.java.util.Set[_ <: TypeElement], roundEnv: RoundEnvironment): Boolean =
    Compiler.compile(t, annotations, roundEnv, processingEnv)
}
