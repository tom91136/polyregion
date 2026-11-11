package polyregion.javalang

import cats.Eval
import cats.data.EitherT
import cats.syntax.all.*
import com.sun.source.tree.*
import com.sun.source.util.*
import com.sun.tools.javac.code.Symbol.MethodSymbol
import net.bytebuddy.description.modifier.*
import net.bytebuddy.implementation.{FieldAccessor, InvocationHandlerAdapter, MethodCall}
import net.bytebuddy.matcher.ElementMatchers
import polyregion.__UnsafeObject
import polyregion.ast.*
import polyregion.javalang.Processor.addOpens

import java.lang.reflect.*
import java.util
import java.nio.file.Paths
import javax.annotation.processing.*
import javax.lang.model.SourceVersion
import javax.lang.model.element.*
import javax.tools.Diagnostic.Kind
import javax.tools.{Diagnostic, JavaFileManager, JavaFileObject, StandardLocation}
import scala.annotation.tailrec
import scala.collection.immutable.ArraySeq
import scala.collection.mutable
import scala.jdk.CollectionConverters.*
import scala.reflect.ClassTag
import scala.util.{Failure, Success, Try, Using}

object Processor {

  private val javacPackages = Seq(
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

  def getMethod(c: Class[?], mName: String, parameterTypes: Class[?]*): Method =
    traverseUp(c)(c => c.getDeclaredMethod(mName, parameterTypes*)).get

  private def getOwnModule = try {
    val m = getMethod(classOf[Class[?]], "getModule")
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
    case e @ (_: NoSuchFieldException | _: SecurityException) =>
      // can't happen.
      throw RuntimeException(e)
  }

  def findCls(name: String): Option[Class[?]] =
    try Option(Class.forName(name))
    catch { case e: ClassNotFoundException => None }

  def addOpens(packages: Seq[String]): Result[Unit] = findCls("java.lang.Module") match {
    case None => ().success
    case Some(mod) =>
      val unsafe            = getUnsafe
      val jdkCompilerModule = getJdkCompilerModule
      val ownModule         = getOwnModule
      Either.catchNonFatal {
        val m                = mod.getDeclaredMethod("implAddOpens", classOf[String], mod)
        val firstFieldOffset = getFirstFieldOffset(unsafe)
        unsafe.putBooleanVolatile(m, firstFieldOffset, true)
        packages.foreach(m.invoke(jdkCompilerModule, _, ownModule))
      }
  }

//  def resolveMethodSignature(
//      processingEnvironment: ProcessingEnvironment,
//      cu: CompilationUnitTree,
//      m: MethodTree
//  ): String = {
//    def tpeToSig(t: Tree): String =
//      t match {
//        case i: IdentifierTree    => i.getName.toString
//        case x: PrimitiveTypeTree => x.getPrimitiveTypeKind.toString
//      }
//
////    try{
//    println(">!" + Trees.instance(processingEnvironment).getPath(cu, m))
//
////      println(Trees.instance(processingEnvironment).getElement( Trees.instance(processingEnvironment).getPath(cu, m)).asInstanceOf[ExecutableElement].getReturnType   )
////    } catch {case e => }
//
//    m.getParameters.asScala.map(_.getType).map(tpeToSig(_)).mkString("(", ",", ")") + tpeToSig(m.getReturnType)
//  }

//  def createClassLookup(xs: List[CompilationUnitTree]): Map[String, (CompilationUnitTree, ClassTree)] = xs
//    .map { cu =>
//      val pkg = Option(cu.getPackageName).fold(Nil)(collectTree(_) {
//        case v: MemberSelectTree => v.getIdentifier.toString :: Nil
//        case v: IdentifierTree   => v.getName.toString :: Nil
//        case _                   => Nil
//      })
//
//      def resolveClassTrees(scope: List[String], xs: List[Tree]): List[(String, ClassTree)] = xs match {
//        case Nil => Nil
//        case (x: ClassTree) :: xs =>
//          val current = scope :+ x.getSimpleName.toString
//          (current.mkString(".") -> x)
//            :: resolveClassTrees(current, x.getMembers.asScala.toList)
//            ::: resolveClassTrees(scope, xs)
//        case (_ :: xs) => resolveClassTrees(scope, xs)
//      }
//
//      println(s"[[]] ${cu.getPackage} => ${cu.getTypeDecls.asScala.toList
//        .collect { case x: ClassTree => x.getSimpleName.toString }}")
//
//      resolveClassTrees(pkg, cu.getTypeDecls.asScala.toList).map((fqcn, c) => fqcn -> (cu, c)).toMap
//    }
//    .foldLeft(Map.empty)(_ ++ _)

}

class Processor extends AbstractProcessor {
  import polyregion.ast.Env.debug

  private val symbolTable = mutable.Map[String, TypeElement]()
  private val methodTable = mutable.Map[String, MethodTree]()

  override def getSupportedAnnotationTypes: java.util.Set[String] =
    java.util.Collections.singleton[String]("*")

  override def getSupportedSourceVersion: SourceVersion = SourceVersion.RELEASE_8

  override def init(processingEnv: ProcessingEnvironment): Unit = {
    super.init(processingEnv)
    processingEnv.getMessager.printMessage(Diagnostic.Kind.NOTE, "Entered processor")

    debug(getClass.getClassLoader)

    val markerClosures =
      Set("polyregion.javalang.OffloadRunnable", "polyregion.javalang.OffloadFunction")

    (for {
      _ <- addOpens(Processor.javacPackages).adaptError(e => RuntimeException("Unable to add opens at runtime", e))
      tasks <- Either
        .catchNonFatal(JavacTask.instance(processingEnv))
        .adaptError(e => RuntimeException("Unable to obtain an instance of JavacTask", e))
      _ <- Either
        .catchNonFatal(Trees.instance(processingEnv))
        .adaptError(e => RuntimeException("Unable to obtain an instance of Trees", e))
    } yield {

      val visited = mutable.Set[JavaFileObject]()

//      val enterVisited = mutable.Map[JavaFileObject, CompilationUnitTree]()
//      val allClasses   = mutable.Map[String, (CompilationUnitTree, ClassTree)]()

      val pool = mutable.Set[MethodTree]()

      val generated = mutable.Set[String]()

      tasks.addTaskListener(new TaskListener {
        override def started(e: TaskEvent): Unit = {
          debug(s"[STARTED]  =>  ${e} ${visited}")
          (e.getKind, e.getCompilationUnit, e.getSourceFile) match {

            case (TaskEvent.Kind.GENERATE, cu, file) if cu != null && !visited.contains(file) =>
              // we do any compilation BEFORE (at the start) GENERATE kicks in as the tree is gone (probably gc'd) after GENERATE
              visited += file

              val reflector = LambdaOutliner.Reflector(symbolTable.toMap, methodTable.toMap)

              val r = LambdaOutliner.extractDeserializeLambdaMethods(
                reflector,
                markerClosures,
                cu
              )

              val s = r.flatMap { xs =>
                xs.traverse((invokeFn, meta) =>
                  LambdaOutliner.findMethod(reflector, cu, invokeFn, meta).map(m => meta.implCls -> m)
                )
              }

              val xs = s match {
                case Left(x)      => throw x
                case Right(value) => value
              }

              debug(
                s">>>> Closures for ${e.getTypeElement}\n" + xs
                  .map { case (cls, m) => s"  ->  ${m.getName.toString.padTo(30, ' ')} @ $cls${m.toString}" }
                  .mkString("\n")
              )

              xs.filterNot(x => pool.contains(x._2)).foreach { case (cls: String, x: MethodTree) =>
                pool += x

                val pkg  = s"polyregion.$$gen$$"
                val name = s"${cls.replace('.', '$')}$$${x.getName}"
                val fqcn = s"$pkg.$name"
                debug(s"Write ${fqcn}")
                if (!generated.contains(fqcn)) {
                  val gen =
                    processingEnv.getFiler.createResource(StandardLocation.CLASS_OUTPUT, pkg, s"$name.class", null)
                  Using.resource(gen.openOutputStream()) { os =>
                    val bytes = new net.bytebuddy.ByteBuddy()
                      .subclass(classOf[ /* polyregion.java.BinaryOffloadExecutable */ Nothing])
                      .name(fqcn)
                      .modifiers(Visibility.PUBLIC, TypeManifestation.FINAL)
                      .defineField(
                        "$BINARY$",
                        classOf[scala.Array[Byte]],
                        Visibility.PUBLIC,
                        Ownership.STATIC,
                        FieldManifestation.FINAL
                      )
                      .method(ElementMatchers.named("binary"))
                      .intercept(FieldAccessor.ofField("$BINARY$"))
                      .make()
                      .getBytes

                    os.write(bytes)
                    os.flush()
                  }
                  generated += fqcn
                } else {
                  debug(s"OVERWRITE ${fqcn}")
                }

              }

            case _ =>
          }

        }

        override def finished(e: TaskEvent): Unit = {
          debug(s"[  ENDED]  =>  ${e}")
          e.getKind match {
            case TaskEvent.Kind.ENTER =>
//              enterVisited += (e.getSourceFile -> e.getCompilationUnit)
            case TaskEvent.Kind.ANNOTATION_PROCESSING =>
//              allClasses ++= createClassLookup(enterVisited.values.toList)
            case _ => // don't care
          }
        }

      })
    }) match {
      case Right(()) =>

      case Left(e) =>
        e.printStackTrace()

        processingEnv.getMessager.printMessage(Diagnostic.Kind.ERROR, e.getMessage)
    }

  }

  override def process(annotations: java.util.Set[? <: TypeElement], roundEnv: RoundEnvironment): Boolean = {

    @tailrec def collectDeclaredElements(xs: List[Element], acc: List[TypeElement] = Nil): List[TypeElement] =
      xs match {
        case Nil => acc
        case (x: TypeElement) :: xs =>
          collectDeclaredElements(x.getEnclosedElements.asScala.toList ::: xs, x :: acc)
        case _ :: xs => collectDeclaredElements(xs, acc)
      }

    val classes = collectDeclaredElements(roundEnv.getRootElements.asScala.toList)

    symbolTable ++= classes.map(e => e.getQualifiedName.toString -> e)

    methodTable ++= classes.flatMap { c =>
      // we only care about normal method and ctors, any other type of executable cannot be referred directly
      c.getEnclosedElements.asScala
        .collect {
          case e: ExecutableElement if e.getKind == ElementKind.METHOD =>
            s"${c.getQualifiedName}.${e.getSimpleName}" -> Trees.instance(processingEnv).getTree(e)
          case e: ExecutableElement if e.getKind == ElementKind.CONSTRUCTOR =>
            s"${c.getQualifiedName}.<init>" -> Trees.instance(processingEnv).getTree(e)
        }
    }.toMap

    false
  }
}
