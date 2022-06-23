package polyregion.java

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
import polyregion.java.Processor.{addOpens, collectTree}

import _root_.java.lang.reflect.*
import _root_.java.util
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

  def asCanonicalName(clsName: String): String = clsName.replace('/', '.').replace('$', '.')

  def collectTree[A](t: Tree)(f: Tree => List[A]): List[A] = new TreeScanner[List[A], Unit] {
    private type U = Unit
    override def reduce(l: List[A], r: List[A]) = (l, r) match {
      case (null, null) => Nil
      case (null, r)    => r
      case (l, null)    => l
      case (l, r)       => r ::: l
    }
    // Commented lines are trees introduced after Java 8
    private inline def v[T <: Tree](n: T, inline g: (T, Unit) => List[A]) = reduce(f(n), g(n, ()))
    override def visitCompilationUnit(n: CompilationUnitTree, p: U)       = v(n, super.visitCompilationUnit)
    // override def visitPackage(n: PackageTree, p: U)                             = v(n, super.visitPackage)
    override def visitImport(n: ImportTree, p: U)                     = v(n, super.visitImport)
    override def visitClass(n: ClassTree, p: U)                       = v(n, super.visitClass)
    override def visitMethod(n: MethodTree, p: U)                     = v(n, super.visitMethod)
    override def visitVariable(n: VariableTree, p: U)                 = v(n, super.visitVariable)
    override def visitEmptyStatement(n: EmptyStatementTree, p: U)     = v(n, super.visitEmptyStatement)
    override def visitBlock(n: BlockTree, p: U)                       = v(n, super.visitBlock)
    override def visitDoWhileLoop(n: DoWhileLoopTree, p: U)           = v(n, super.visitDoWhileLoop)
    override def visitWhileLoop(n: WhileLoopTree, p: U)               = v(n, super.visitWhileLoop)
    override def visitForLoop(n: ForLoopTree, p: U)                   = v(n, super.visitForLoop)
    override def visitEnhancedForLoop(n: EnhancedForLoopTree, p: U)   = v(n, super.visitEnhancedForLoop)
    override def visitLabeledStatement(n: LabeledStatementTree, p: U) = v(n, super.visitLabeledStatement)
    override def visitSwitch(n: SwitchTree, p: U)                     = v(n, super.visitSwitch)
    // override def visitSwitchExpression(n: SwitchExpressionTree, p: U)           = v(n, super.visitSwitchExpression)
    override def visitCase(n: CaseTree, p: U)                                   = v(n, super.visitCase)
    override def visitSynchronized(n: SynchronizedTree, p: U)                   = v(n, super.visitSynchronized)
    override def visitTry(n: TryTree, p: U)                                     = v(n, super.visitTry)
    override def visitCatch(n: CatchTree, p: U)                                 = v(n, super.visitCatch)
    override def visitConditionalExpression(n: ConditionalExpressionTree, p: U) = v(n, super.visitConditionalExpression)
    override def visitIf(n: IfTree, p: U)                                       = v(n, super.visitIf)
    override def visitExpressionStatement(n: ExpressionStatementTree, p: U)     = v(n, super.visitExpressionStatement)
    override def visitBreak(n: BreakTree, p: U)                                 = v(n, super.visitBreak)
    override def visitContinue(n: ContinueTree, p: U)                           = v(n, super.visitContinue)
    override def visitReturn(n: ReturnTree, p: U)                               = v(n, super.visitReturn)
    override def visitThrow(n: ThrowTree, p: U)                                 = v(n, super.visitThrow)
    override def visitAssert(n: AssertTree, p: U)                               = v(n, super.visitAssert)
    override def visitMethodInvocation(n: MethodInvocationTree, p: U)           = v(n, super.visitMethodInvocation)
    override def visitNewClass(n: NewClassTree, p: U)                           = v(n, super.visitNewClass)
    override def visitNewArray(n: NewArrayTree, p: U)                           = v(n, super.visitNewArray)
    override def visitLambdaExpression(n: LambdaExpressionTree, p: U)           = v(n, super.visitLambdaExpression)
    override def visitParenthesized(n: ParenthesizedTree, p: U)                 = v(n, super.visitParenthesized)
    override def visitAssignment(n: AssignmentTree, p: U)                       = v(n, super.visitAssignment)
    override def visitCompoundAssignment(n: CompoundAssignmentTree, p: U)       = v(n, super.visitCompoundAssignment)
    override def visitUnary(n: UnaryTree, p: U)                                 = v(n, super.visitUnary)
    override def visitBinary(n: BinaryTree, p: U)                               = v(n, super.visitBinary)
    override def visitTypeCast(n: TypeCastTree, p: U)                           = v(n, super.visitTypeCast)
    override def visitInstanceOf(n: InstanceOfTree, p: U)                       = v(n, super.visitInstanceOf)
    // override def visitBindingPattern(n: BindingPatternTree, p: U)               = v(n, super.visitBindingPattern)
    // override def visitDefaultCaseLabel(n: DefaultCaseLabelTree, p: U)           = v(n, super.visitDefaultCaseLabel)
    override def visitArrayAccess(n: ArrayAccessTree, p: U)   = v(n, super.visitArrayAccess)
    override def visitMemberSelect(n: MemberSelectTree, p: U) = v(n, super.visitMemberSelect)
    // override def visitParenthesizedPattern(n: ParenthesizedPatternTree, p: U)   = v(n, super.visitParenthesizedPattern)
    // override def visitGuardedPattern(n: GuardedPatternTree, p: U)               = v(n, super.visitGuardedPattern)
    override def visitMemberReference(n: MemberReferenceTree, p: U)     = v(n, super.visitMemberReference)
    override def visitIdentifier(n: IdentifierTree, p: U)               = v(n, super.visitIdentifier)
    override def visitLiteral(n: LiteralTree, p: U)                     = v(n, super.visitLiteral)
    override def visitPrimitiveType(n: PrimitiveTypeTree, p: U)         = v(n, super.visitPrimitiveType)
    override def visitArrayType(n: ArrayTypeTree, p: U)                 = v(n, super.visitArrayType)
    override def visitParameterizedType(n: ParameterizedTypeTree, p: U) = v(n, super.visitParameterizedType)
    override def visitUnionType(n: UnionTypeTree, p: U)                 = v(n, super.visitUnionType)
    override def visitIntersectionType(n: IntersectionTypeTree, p: U)   = v(n, super.visitIntersectionType)
    override def visitTypeParameter(n: TypeParameterTree, p: U)         = v(n, super.visitTypeParameter)
    override def visitWildcard(n: WildcardTree, p: U)                   = v(n, super.visitWildcard)
    override def visitModifiers(n: ModifiersTree, p: U)                 = v(n, super.visitModifiers)
    override def visitAnnotation(n: AnnotationTree, p: U)               = v(n, super.visitAnnotation)
    override def visitAnnotatedType(n: AnnotatedTypeTree, p: U)         = v(n, super.visitAnnotatedType)
    // override def visitModule(n: ModuleTree, p: U)                               = v(n, super.visitModule)
    // override def visitExports(n: ExportsTree, p: U)                             = v(n, super.visitExports)
    // override def visitOpens(n: OpensTree, p: U)                                 = v(n, super.visitOpens)
    // override def visitProvides(n: ProvidesTree, p: U)                           = v(n, super.visitProvides)
    // override def visitRequires(n: RequiresTree, p: U)                           = v(n, super.visitRequires)
    // override def visitUses(n: UsesTree, p: U)                                   = v(n, super.visitUses)
    override def visitOther(n: Tree, p: U)              = v(n, super.visitOther)
    override def visitErroneous(n: ErroneousTree, p: U) = v(n, super.visitErroneous)
    // override def visitYield(n: YieldTree, p: U)                                 = v(n, super.visitYield)

  }.scan(t, ())

  def collectTrees[A](xs: List[Tree])(f: Tree => List[A]): List[A] = xs.flatMap(collectTree(_)(f))

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

  private val symbolTable = mutable.Map[String, TypeElement]()
  private val methodTable = mutable.Map[String, MethodTree]()

  private var trees: Trees = _

  override def getSupportedAnnotationTypes: _root_.java.util.Set[String] =
    _root_.java.util.Collections.singleton[String]("*")

  override def getSupportedSourceVersion: SourceVersion = SourceVersion.RELEASE_8

  override def init(processingEnv: ProcessingEnvironment): Unit = {
    super.init(processingEnv)
    processingEnv.getMessager.printMessage(Diagnostic.Kind.NOTE, "Entered processor")

    System.out.println(getClass.getClassLoader)

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

    val markerClosures =
      Set("polyregion.java.OffloadRunnable", "polyregion.java.OffloadFunction")

    (for {
      _ <- addOpens(allPkgs).adaptError(e => new RuntimeException("Unable to add opens at runtime", e))
      tasks <- Either
        .catchNonFatal(JavacTask.instance(processingEnv))
        .adaptError(e => new RuntimeException("Unable to obtain an instance of JavacTask", e))
      trees <- Either
        .catchNonFatal(Trees.instance(processingEnv))
        .adaptError(e => new RuntimeException("Unable to obtain an instance of Trees", e))
      _ = this.trees = trees
    } yield {

      val visited = mutable.Set[JavaFileObject]()

//      val enterVisited = mutable.Map[JavaFileObject, CompilationUnitTree]()
//      val allClasses   = mutable.Map[String, (CompilationUnitTree, ClassTree)]()

      val pool = mutable.Set[MethodTree]()

      val generated = mutable.Set[String]()

      tasks.addTaskListener(new TaskListener {
        override def started(e: TaskEvent): Unit = {
          println(s"[STARTED]  =>  ${e} ${visited}")
          (e.getKind, e.getCompilationUnit, e.getSourceFile) match {

            case (TaskEvent.Kind.GENERATE, cu, file) if cu != null && !visited.contains(file) =>
              // we do any compilation BEFORE (at the start) GENERATE kicks in as the tree is gone (probably gc'd) after GENERATE
              visited += file

              val reflector = LambdaOutliner.Reflector(processingEnv, trees, symbolTable.toMap, methodTable.toMap)

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

              println(
                s">>>> Closures for ${e.getTypeElement}\n" + xs
                  .map { case (cls, m) => s"  ->  ${m.getName.toString.padTo(30, ' ')} @ $cls${m.toString}" }
                  .mkString("\n")
              )

              xs.filterNot(x => pool.contains(x._2)).foreach { case (cls: String, x: MethodTree) =>
                pool += x

                val pkg  = s"polyregion.$$gen$$"
                val name = s"${cls.replace('.', '$')}$$${x.getName}"
                val fqcn = s"$pkg.$name"
                println(s"Write ${fqcn}")
                if (!generated.contains(fqcn)) {
                  val gen =
                    processingEnv.getFiler.createResource(StandardLocation.CLASS_OUTPUT, pkg, s"$name.class", null)
                  val os = gen.openOutputStream()



                  val bytes = new net.bytebuddy.ByteBuddy()
                    .subclass(classOf[ /* _root_.polyregion.java.BinaryOffloadExecutable */ Nothing])
                    .name(fqcn)
                    .modifiers(Visibility.PUBLIC, TypeManifestation.FINAL)
                    .defineField(
                      "$BINARY$",
                      classOf[scala.Array[Byte]] ,
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
                  os.close()
                  generated += fqcn
                } else {
                  println(s"OVERWRITE ${fqcn}")
                }

              }

            case _ =>
          }

        }

        override def finished(e: TaskEvent): Unit = {
          println(s"[  ENDED]  =>  ${e}")
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

  override def process(annotations: _root_.java.util.Set[_ <: TypeElement], roundEnv: RoundEnvironment): Boolean = {

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
