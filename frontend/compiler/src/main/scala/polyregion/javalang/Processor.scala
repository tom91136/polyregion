package polyregion.javalang

import cats.syntax.all.*
import com.sun.source.tree.*
import com.sun.source.util.*
import com.sun.tools.javac.code.Symbol.MethodSymbol
import com.sun.tools.javac.tree.JCTree
import polyregion.__UnsafeObject
import polyregion.ast.*
import polyregion.javalang.LambdaOutliner.collectTree
import polyregion.javalang.Processor.addOpens

import java.lang.reflect.*
import javax.annotation.processing.*
import javax.lang.model.SourceVersion
import javax.lang.model.element.*
import javax.tools.{Diagnostic, StandardLocation}
import scala.annotation.tailrec
import scala.collection.mutable
import scala.jdk.CollectionConverters.*
import sun.misc.Unsafe

object Processor {

  final val DeferErrorsOption     = "polyregion.deferErrors"
  final val ProcessorDigestOption = "polyregion.processorDigest"

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

  private def getUnsafe: Result[Unsafe] = Either.catchNonFatal {
    val theUnsafe = classOf[Unsafe].getDeclaredField("theUnsafe")
    theUnsafe.setAccessible(true)
    classOf[Unsafe].cast(theUnsafe.get(classOf[Unsafe]))
  }

  @tailrec def traverseUp[A <: AccessibleObject](cls: Class[?])(f: Class[?] => A): Option[A] = Option(cls) match {
    case None => None
    case Some(x) =>
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

  private def getOwnModule: Result[Object] = Either.catchNonFatal {
    val m = getMethod(classOf[Class[?]], "getModule")
    m.invoke(classOf[Processor])
  }

  // reflective so a JDK without modules exits through the catch rather than failing to link
  private def getJdkCompilerModule: Result[Object] = Either.catchNonFatal {
    val cModuleLayer = Class.forName("java.lang.ModuleLayer")
    val mBoot        = cModuleLayer.getDeclaredMethod("boot")
    val bootLayer    = mBoot.invoke(cModuleLayer)
    val cOptional    = Class.forName("java.util.Optional")
    val mFindModule  = cModuleLayer.getDeclaredMethod("findModule", classOf[String])
    val oCompilerO   = mFindModule.invoke(bootLayer, "jdk.compiler")
    cOptional.getDeclaredMethod("get").invoke(oCompilerO)
  }

  private def getFirstFieldOffset(unsafe: Unsafe) = try
    unsafe.objectFieldOffset(classOf[__UnsafeObject].getDeclaredField("first"))
  catch { case e @ (_: NoSuchFieldException | _: SecurityException) => throw RuntimeException(e) }

  def findCls(name: String): Option[Class[?]] =
    try Option(Class.forName(name))
    catch { case _: ClassNotFoundException => None }

  def addOpens(packages: Seq[String]): Result[Unit] = findCls("java.lang.Module") match {
    case None => ().success
    case Some(mod) =>
      for {
        unsafe            <- getUnsafe
        jdkCompilerModule <- getJdkCompilerModule
        ownModule         <- getOwnModule
        _ <- Either.catchNonFatal {
          val m                = mod.getDeclaredMethod("implAddOpens", classOf[String], mod)
          val firstFieldOffset = getFirstFieldOffset(unsafe)
          unsafe.putBooleanVolatile(m, firstFieldOffset, true)
          packages.foreach(m.invoke(jdkCompilerModule, _, ownModule))
        }
      } yield ()
  }
}

class Processor extends AbstractProcessor {
  import polyregion.ast.Env.debug

  private val symbolTable = mutable.Map.empty[String, TypeElement]
  private val methodTable = mutable.Map.empty[String, MethodTree]

  private def deferErrors: Boolean =
    processingEnv.getOptions.getOrDefault(Processor.DeferErrorsOption, "false") == "true"

  // lowering can throw rather than return a Left, and javac raises AssertionError on some post-Lower nodes
  private def guard[A](f: => Result[A]): Result[A] =
    try f
    catch { case t: Throwable => Left(t) }

  private def writeClass(env: ProcessingEnvironment, pkg: String, name: String, bytes: scala.Array[Byte]): Unit = {
    val os = env.getFiler.createResource(StandardLocation.CLASS_OUTPUT, pkg, s"$name.class").openOutputStream()
    try os.write(bytes)
    finally os.close()
  }

  override def getSupportedOptions: java.util.Set[String] =
    java.util.Set.of[String](Processor.DeferErrorsOption, Processor.ProcessorDigestOption)

  override def getSupportedAnnotationTypes: java.util.Set[String] =
    java.util.Collections.singleton[String]("*")

  override def getSupportedSourceVersion: SourceVersion = SourceVersion.latestSupported

  private def declaredMethods(tree: Tree): List[(MethodSymbol, MethodTree)] = collectTree(tree) {
    case m: JCTree.JCMethodDecl => Option(m.sym).map(_ -> m).toList
    case _                      => Nil
  }

  private def declaredClasses(tree: Tree): List[(TypeElement, ClassTree)] = collectTree(tree) {
    case c: JCTree.JCClassDecl => Option(c.sym).map(_ -> c).toList
    case _                     => Nil
  }

  private class RegionListener(trees: Trees) extends TaskListener {

    private val markerClosures = Set("polyregion.javalang.OffloadRunnable", "polyregion.javalang.OffloadFunction")

    private val visited = mutable.Set.empty[TypeElement]

    private val pool = mutable.Set.empty[MethodTree]

    private val generated = mutable.Set.empty[String]

    // a member or local class is lifted out of the unit by GENERATE, so its members are caught while still nested
    private val nestedTrees = mutable.Map.empty[MethodSymbol, MethodTree]

    // lowering rewrites a class declaration in place, so the lifted class keeps the tree seen after analysis
    private val classTrees = mutable.Map.empty[TypeElement, ClassTree]

    override def started(e: TaskEvent): Unit = {
      debug(s"[STARTED]  =>  $e $visited")
      (e.getKind, Option(e.getCompilationUnit), e.getSourceFile) match {

        // compile at the start of GENERATE, the tree is gone by the time it finishes
        case (TaskEvent.Kind.GENERATE, Some(unit), _) if Option(e.getTypeElement).exists(visited.add) =>
          val scope: Tree = classTrees.getOrElse(e.getTypeElement, unit)

          val reflector = LambdaOutliner.Reflector(symbolTable.toMap, methodTable.toMap)

          // bodies javac synthesises after processing (a lambda's impl method) are invisible to `methodTable`
          val localTrees = guard {
            Either.catchNonFatal(declaredMethods(unit).toMap ++ declaredMethods(scope))
          }.getOrElse(Map.empty[MethodSymbol, MethodTree]) ++ nestedTrees

          // javac throws on some post-Lower nodes (`LetExpr is not part of a public API`), which would abort every region in the file
          val xs = guard {
            LambdaOutliner.extractDeserializeLambdaMethods(markerClosures, scope).flatMap { xs =>
              xs.traverse((invokeFn, meta) =>
                LambdaOutliner.findMethod(reflector, scope, invokeFn, meta).map(m => meta.implCls -> m)
              )
            }
          } match {
            case Right(value) => value
            case Left(x) =>
              if (!deferErrors) throw x
              processingEnv.getMessager.printMessage(
                Diagnostic.Kind.WARNING,
                s"[PolyRegion] Cannot outline regions in ${e.getSourceFile.getName}: $x"
              )
              Nil
          }

          debug(
            s">>>> Closures for ${e.getTypeElement}\n" + xs
              .map { case (cls, m) => s"  ->  ${m.getName.toString.padTo(30, ' ')} @ $cls${m.toString}" }
              .mkString("\n")
          )

          xs.filterNot(x => pool.contains(x._2)).foreach { case (cls: String, x: MethodTree) =>
            pool += x

            val pkg  = "polyregion.$gen$"
            val name = s"${cls.replace('.', '$')}$$${x.getName}"
            val fqcn = s"$pkg.$name"
            if (generated.contains(fqcn)) debug(s"Skip already generated $fqcn")
            else {
              debug(s"Write $fqcn")
              guard(
                Offloader.compile(
                  TreeRemapper.Sources(processingEnv, trees, symbolTable.toMap, methodTable.toMap, localTrees),
                  x,
                  fqcn
                )
              ) match {
                case Left(failure) =>
                  val reason = s"[PolyRegion] Cannot offload $cls.${x.getName}: ${failure.getMessage}"
                  debug(failure)
                  if (!deferErrors) processingEnv.getMessager.printMessage(Diagnostic.Kind.ERROR, reason)
                  else
                    Offloader.unsupported(fqcn, reason) match {
                      case Right(stub) =>
                        processingEnv.getMessager.printMessage(Diagnostic.Kind.WARNING, reason)
                        writeClass(processingEnv, pkg, name, stub.bytes)
                        generated += fqcn
                      case Left(stubError) =>
                        processingEnv.getMessager
                          .printMessage(Diagnostic.Kind.ERROR, s"$reason (and the stub failed: $stubError)")
                    }
                case Right(exe) =>
                  exe.twins.foreach { twin =>
                    val at = twin.fqcn.lastIndexOf('.')
                    writeClass(processingEnv, twin.fqcn.take(math.max(at, 0)), twin.fqcn.drop(at + 1), twin.bytes)
                  }
                  writeClass(processingEnv, pkg, name, exe.bytes)
                  generated += fqcn
              }
            }

          }

        case _ =>
      }

    }

    override def finished(e: TaskEvent): Unit = {
      debug(s"[  ENDED]  =>  $e")
      (e.getKind, Option(e.getCompilationUnit)) match {
        case (TaskEvent.Kind.ANALYZE, Some(cu)) =>
          guard(Either.catchNonFatal {
            declaredMethods(cu).foreach(nestedTrees += _)
            declaredClasses(cu).foreach(classTrees += _)
          })
          ()
        case _ =>
      }
    }
  }

  override def init(processingEnv: ProcessingEnvironment): Unit = {
    super.init(processingEnv)
    processingEnv.getMessager.printMessage(Diagnostic.Kind.NOTE, "Entered processor")

    debug(getClass.getClassLoader)

    (for {
      _ <- addOpens(Processor.javacPackages).adaptError(e => RuntimeException("Unable to add opens at runtime", e))
      tasks <- Either
        .catchNonFatal(JavacTask.instance(processingEnv))
        .adaptError(e => RuntimeException("Unable to obtain an instance of JavacTask", e))
      trees <- Either
        .catchNonFatal(Trees.instance(processingEnv))
        .adaptError(e => RuntimeException("Unable to obtain an instance of Trees", e))
    } yield tasks.addTaskListener(RegionListener(trees))) match {
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
        case (x: TypeElement) :: rest =>
          collectDeclaredElements(x.getEnclosedElements.asScala.toList ::: rest, x :: acc)
        case _ :: rest => collectDeclaredElements(rest, acc)
      }

    val classes = collectDeclaredElements(roundEnv.getRootElements.asScala.toList)

    symbolTable ++= classes.map(e => e.getQualifiedName.toString -> e)

    methodTable ++= classes.flatMap { c =>
      // no other executable can be named at a call site
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
