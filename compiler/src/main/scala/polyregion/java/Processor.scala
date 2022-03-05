package polyregion.java

import com.sun.source.tree.{
  IdentifierTree,
  LambdaExpressionTree,
  MemberSelectTree,
  MethodInvocationTree,
  MethodTree,
  SwitchTree,
  VariableTree
}
import com.sun.source.util.*
import polyregion.__UnsafeObject
import polyregion.java.Processor.addOpens

import _root_.java.lang.reflect.*
import _root_.java.util
import javax.annotation.processing.*
import javax.lang.model.SourceVersion
import javax.lang.model.element.{Element, Modifier, TypeElement}
import javax.tools.Diagnostic
import javax.tools.Diagnostic.Kind
import scala.annotation.tailrec
import scala.collection.immutable.ArraySeq
import scala.jdk.CollectionConverters.*
import scala.util.{Failure, Success, Try}

object Processor {

  private def getUnsafe = try {
    val theUnsafe = classOf[sun.misc.Unsafe].getDeclaredField("theUnsafe")
    theUnsafe.setAccessible(true)
    theUnsafe.get(null).asInstanceOf[sun.misc.Unsafe]
  } catch {
    case e: Exception =>
      null
  }

  @tailrec def traverseUp[A <: AccessibleObject](cls: Class[?])(f: Class[?] => A): Option[A] = cls match {
    case null => None
    case x =>
      try {
        val ao = Option(f(x))
        ao.foreach(_.setAccessible(true))
        ao
      } catch {
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

  private def getFirstFieldOffset(unsafe: sun.misc.Unsafe) = try
    unsafe.objectFieldOffset(classOf[__UnsafeObject].getDeclaredField("first"))
  catch {
    case e: NoSuchFieldException =>
      // can't happen.
      throw new RuntimeException(e)
    case e: SecurityException =>
      // can't happen
      throw new RuntimeException(e)
  }

  def findCls(name: String): Option[Class[_]] =
    try Option(Class.forName(name))
    catch { case e: ClassNotFoundException => None }

  def addOpens(packages: Seq[String]): Try[Unit] = findCls("java.lang.Module") match {
    case None => Success(())
    case Some(mod) =>
      val unsafe            = getUnsafe
      val jdkCompilerModule = getJdkCompilerModule
      val ownModule         = getOwnModule
      Try {
        val m                = mod.getDeclaredMethod("implAddOpens", classOf[String], mod)
        val firstFieldOffset = getFirstFieldOffset(unsafe)
        unsafe.putBooleanVolatile(m, firstFieldOffset, true)
        packages.foreach(m.invoke(jdkCompilerModule, _, ownModule))
      }
  }

}

@SupportedSourceVersion(SourceVersion.RELEASE_8)
class Processor extends AbstractProcessor {

  override def getSupportedAnnotationTypes: util.Set[String] =
    java.util.Collections.singleton[String]("*") // "polyregion.java.Offload"

  override def getSupportedSourceVersion: SourceVersion = SourceVersion.RELEASE_8

  private[polyregion] var t: Trees = _

  def resolveJavacTools: Try[(JavacTask, Trees)] = Try {
    JavacTask.instance(processingEnv) ->
      Trees.instance(processingEnv)
  }

  override def init(processingEnv: ProcessingEnvironment): Unit = {
    super.init(processingEnv)

    val allPkgs = Seq(
      "com.sun.tools.javac.code",
      "com.sun.tools.javac.comp",
      "com.sun.tools.javac.file",
      "com.sun.tools.javac.main",
      "com.sun.tools.javac.model",
      "com.sun.tools.javac.parser",
      "com.sun.tools.javac.processing",
      "com.sun.tools.javac.tree",
      "com.sun.tools.javac.util",
      "com.sun.tools.javac.jvm",
      "com.sun.tools.javac.api"
    )

    (for {
      _ <- addOpens(allPkgs).recoverWith(e => Failure(new RuntimeException("Unable to add opens at runtime", e)))
      tasks <- Try(JavacTask.instance(processingEnv)).recoverWith(e =>
        Failure(new RuntimeException("Unable to obtain an instance of JavacTask", e))
      )
      trees <- Try(Trees.instance(processingEnv)).recoverWith(e =>
        Failure(new RuntimeException("Unable to obtain an instance of Trees", e))
      )
    } yield tasks.addTaskListener(new TaskListener {
      override def started(e: TaskEvent): Unit = () // don't care
      override def finished(e: TaskEvent): Unit = {
        println(s"EV, end  =${e}")
//        println(e.getCompilationUnit)
        (e.getKind, e.getCompilationUnit) match {
          case (TaskEvent.Kind.GENERATE, cu) if cu != null =>
            val x = new TreeScanner[List[MethodTree], Unit] {
              override def visitMethod(node: MethodTree, p: Unit) = (
                node.getModifiers.getFlags.asScala.toSet,
                node.getName.toString,
                node.getParameters.asScala
                  .map(_.getType)
                  .collect {
                    case i: MemberSelectTree => i.getIdentifier.toString //  general case of unqualified use
                    case i: IdentifierTree   => i.getName.toString       // in case the call site imports the class
                  }
                  .toList
              ) match {
                case (modifiers, "$deserializeLambda$", "SerializedLambda" :: Nil)
                    if modifiers == Set(Modifier.PRIVATE, Modifier.STATIC) =>
                  node :: Nil
                case _ => Nil
              }

              override def reduce(l: List[MethodTree], r: List[MethodTree]) = (l, r) match {
                case (null, null) => Nil
                case (null, r)    => r
                case (l, null)    => l
                case (l, r)       => l ::: r
              }
            }
            val xs = x.scan(cu, ())
            println(">>" + xs.map(_.getName))


//
//              println(e.getCompilationUnit)
//
//              val x = new TreeScanner[String, Unit] {
//
//                override def visitSwitch(node: SwitchTree, p: Unit) = {
//                  println(s"!!!!!${node}")
//                  ""
//                }
//                override def visitMethodInvocation(node: MethodInvocationTree, p: Unit): String = {
//                  println(s">>>${node} args=${node.getArguments} ${node.getKind} ${node.getMethodSelect}")
//
//                  ""
//                }
//
//                override def reduce(r1: String, r2: String) = r1 ++ r2
//              }
//
//              x.scan(e.getCompilationUnit, ())

          case _ =>
        }

      }
    })) match {
      case Success(()) =>
      case Failure(e) =>
        e.printStackTrace()
        processingEnv.getMessager.printMessage(Diagnostic.Kind.ERROR, e.getMessage)
    }

//
//    JavacTask
//      .instance(processingEnv)
//      .addTaskListener(new TaskListener {
//        override def started(e: TaskEvent) = println(s"EV, Start=${e}")
//        override def finished(e: TaskEvent) = {
//
//          e.getKind match {
//            case TaskEvent.Kind.GENERATE =>
////            processingEnv.getFiler.createSourceFile("", null)
//              println(e.getCompilationUnit)
//
//              val x = new TreeScanner[String, Unit] {
//
//                override def visitSwitch(node: SwitchTree, p: Unit) = {
//                  println(s"!!!!!${node}")
//                  ""
//                }
//                override def visitMethodInvocation(node: MethodInvocationTree, p: Unit): String = {
//                  println(s">>>${node} args=${node.getArguments} ${node.getKind} ${node.getMethodSelect}")
//
//                  ""
//                }
//
//                override def reduce(r1: String, r2: String) = r1 ++ r2
//              }
//
//              x.scan(e.getCompilationUnit, ())
//
//            case _ =>
//          }
//
//          println(s"EV, end  =${e}")
//        }
//      })
////    println(com.sun.tools.javac.api.MultiTaskListener.instance(a.getContext).add())
//    t = Trees.instance(processingEnv)
//    processingEnv.getMessager.printMessage(Kind.WARNING, s"In processor, tree=$t")
  }

  override def process(annotations: _root_.java.util.Set[_ <: TypeElement], roundEnv: RoundEnvironment): Boolean =
    true
//    Compiler.compile(t, annotations, roundEnv, processingEnv)
}
