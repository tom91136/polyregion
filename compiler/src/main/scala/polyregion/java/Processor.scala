package polyregion.java

import cats.Eval
import cats.data.EitherT
import cats.syntax.all.*

import com.sun.source.tree.*
import com.sun.source.util.*
import polyregion.__UnsafeObject
import polyregion.java.Processor.{addOpens, collectTree, extractInvokeStaticTargets, findDeserialiseLambdaMethods}
import polyregion.backend.compiler.*
import _root_.java.lang.reflect.*
import _root_.java.util
import javax.annotation.processing.*
import javax.lang.model.SourceVersion
import javax.lang.model.element.{Element, Modifier, TypeElement}
import javax.tools.{Diagnostic, JavaFileObject}
import javax.tools.Diagnostic.Kind
import scala.annotation.tailrec
import scala.collection.immutable.ArraySeq
import scala.collection.mutable
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

  def collectTree[A](t: Tree)(f: Tree => List[A]): List[A] = new TreeScanner[List[A], Unit] {
    private type U = Unit
    override def reduce(l: List[A], r: List[A]) = (l, r) match {
      case (null, null) => Nil
      case (null, r)    => r
      case (l, null)    => l
      case (l, r)       => r ::: l
    }
    private inline def v[T <: Tree](n: T, inline g: (T, Unit) => List[A])       = reduce(f(n), g(n, ()))
    override def visitCompilationUnit(n: CompilationUnitTree, p: U)             = v(n, super.visitCompilationUnit)
    override def visitPackage(n: PackageTree, p: U)                             = v(n, super.visitPackage)
    override def visitImport(n: ImportTree, p: U)                               = v(n, super.visitImport)
    override def visitClass(n: ClassTree, p: U)                                 = v(n, super.visitClass)
    override def visitMethod(n: MethodTree, p: U)                               = v(n, super.visitMethod)
    override def visitVariable(n: VariableTree, p: U)                           = v(n, super.visitVariable)
    override def visitEmptyStatement(n: EmptyStatementTree, p: U)               = v(n, super.visitEmptyStatement)
    override def visitBlock(n: BlockTree, p: U)                                 = v(n, super.visitBlock)
    override def visitDoWhileLoop(n: DoWhileLoopTree, p: U)                     = v(n, super.visitDoWhileLoop)
    override def visitWhileLoop(n: WhileLoopTree, p: U)                         = v(n, super.visitWhileLoop)
    override def visitForLoop(n: ForLoopTree, p: U)                             = v(n, super.visitForLoop)
    override def visitEnhancedForLoop(n: EnhancedForLoopTree, p: U)             = v(n, super.visitEnhancedForLoop)
    override def visitLabeledStatement(n: LabeledStatementTree, p: U)           = v(n, super.visitLabeledStatement)
    override def visitSwitch(n: SwitchTree, p: U)                               = v(n, super.visitSwitch)
    override def visitSwitchExpression(n: SwitchExpressionTree, p: U)           = v(n, super.visitSwitchExpression)
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
    override def visitBindingPattern(n: BindingPatternTree, p: U)               = v(n, super.visitBindingPattern)
    override def visitDefaultCaseLabel(n: DefaultCaseLabelTree, p: U)           = v(n, super.visitDefaultCaseLabel)
    override def visitArrayAccess(n: ArrayAccessTree, p: U)                     = v(n, super.visitArrayAccess)
    override def visitMemberSelect(n: MemberSelectTree, p: U)                   = v(n, super.visitMemberSelect)
    override def visitParenthesizedPattern(n: ParenthesizedPatternTree, p: U)   = v(n, super.visitParenthesizedPattern)
    override def visitGuardedPattern(n: GuardedPatternTree, p: U)               = v(n, super.visitGuardedPattern)
    override def visitMemberReference(n: MemberReferenceTree, p: U)             = v(n, super.visitMemberReference)
    override def visitIdentifier(n: IdentifierTree, p: U)                       = v(n, super.visitIdentifier)
    override def visitLiteral(n: LiteralTree, p: U)                             = v(n, super.visitLiteral)
    override def visitPrimitiveType(n: PrimitiveTypeTree, p: U)                 = v(n, super.visitPrimitiveType)
    override def visitArrayType(n: ArrayTypeTree, p: U)                         = v(n, super.visitArrayType)
    override def visitParameterizedType(n: ParameterizedTypeTree, p: U)         = v(n, super.visitParameterizedType)
    override def visitUnionType(n: UnionTypeTree, p: U)                         = v(n, super.visitUnionType)
    override def visitIntersectionType(n: IntersectionTypeTree, p: U)           = v(n, super.visitIntersectionType)
    override def visitTypeParameter(n: TypeParameterTree, p: U)                 = v(n, super.visitTypeParameter)
    override def visitWildcard(n: WildcardTree, p: U)                           = v(n, super.visitWildcard)
    override def visitModifiers(n: ModifiersTree, p: U)                         = v(n, super.visitModifiers)
    override def visitAnnotation(n: AnnotationTree, p: U)                       = v(n, super.visitAnnotation)
    override def visitAnnotatedType(n: AnnotatedTypeTree, p: U)                 = v(n, super.visitAnnotatedType)
    override def visitModule(n: ModuleTree, p: U)                               = v(n, super.visitModule)
    override def visitExports(n: ExportsTree, p: U)                             = v(n, super.visitExports)
    override def visitOpens(n: OpensTree, p: U)                                 = v(n, super.visitOpens)
    override def visitProvides(n: ProvidesTree, p: U)                           = v(n, super.visitProvides)
    override def visitRequires(n: RequiresTree, p: U)                           = v(n, super.visitRequires)
    override def visitUses(n: UsesTree, p: U)                                   = v(n, super.visitUses)
    override def visitOther(n: Tree, p: U)                                      = v(n, super.visitOther)
    override def visitErroneous(n: ErroneousTree, p: U)                         = v(n, super.visitErroneous)
    override def visitYield(n: YieldTree, p: U)                                 = v(n, super.visitYield)

  }.scan(t, ())

  def collectTrees[A](xs: List[Tree])(f: Tree => List[A]): List[A] = xs.flatMap(collectTree(_)(f))

  def extractInvokeStaticTargets(
      interfaceNames: Set[String],
      deserialiseLambdaMethod: MethodTree,
      cu: CompilationUnitTree
  ): Result[List[MethodTree]] = collectTree(deserialiseLambdaMethod.getBody) {
    case s: SwitchTree => s :: Nil
    case _             => Nil
  } match {
    case sourceNameSwitch :: targetSwitch :: Nil =>
      // The general structure of $deserializeLambda$ looks like this:
      //  /*synthetic*/ private static $deserializeLambda$(final SerializedLambda l)
      //    int target;
      //    switch(l.getImplMethodName().hashCode()) {
      //     case hashCodeOfName: if (name.equals("$INVOKE_STATIC_FN_NAME")) target = $N; break;
      //     ...
      //    }
      //    switch(target) {
      //     case $N: if (l.getImplMethodKind() == 6 && # 6 for invoke dynamic
      //                  l.getFunctionalInterfaceClass().equals("$FN_IFACE") &&
      //                  l.getFunctionalInterfaceMethodName().equals("$FN_SAM_NAME") &&
      //                  l.getFunctionalInterfaceMethodSignature().equals("$FN_SAM_SIG") &&
      //                  l.getImplClass().equals("$FN_DECL_SITE") &&
      //                  l.getImplMethodSignature().equals("$INVOKE_STATIC_FN_SIG"))
      //                    return java.lang.invoke.LambdaMetafactory.$...;
      //     ...
      //    }
      //   ...
      // }
      val invokeCases = sourceNameSwitch.getCases.asScala.toList.traverse { tree =>
        (
          collectTree(tree) {
            case dest: AssignmentTree if dest.getExpression.getKind == Tree.Kind.INT_LITERAL =>
              dest.getExpression.asInstanceOf[LiteralTree].getValue.asInstanceOf[Int] :: Nil
            case _ => Nil
          },
          collectTree(tree) {
            case dest: LiteralTree if dest.getKind == Tree.Kind.STRING_LITERAL =>
              dest.getValue.toString :: Nil
            case _ => Nil
          }
        ) match {
          case (targetCase :: Nil, invokeStaticName :: Nil) => (targetCase, invokeStaticName).success
          case (cases, names) =>
            s"Expecting one int literal target and one String literal invokeStatic from case body, target was $cases and invokeStatic was $names. Tree:\n$tree".fail
        }
      }

      targetSwitch.getCases.asScala.toList.traverse { tree =>



        val rr = for{
          c <-
            if (tree.getExpression.getKind == Tree.Kind.INT_LITERAL) {
              tree.getExpression.asInstanceOf[LiteralTree].getValue.asInstanceOf[Int].success
            } else s"Case expression is not an int literal. Tree:\n$tree".fail
            name = tree.getStatements.asScala.flatMap(collectTree(_) {
            case a: MethodInvocationTree =>
              collectTree(a.getMethodSelect) {
                case i: MemberSelectTree => i.getIdentifier.toString :: Nil
                case _                   => Nil
              } match {
                case "getFunctionalInterfaceClass" :: "equals" :: Nil =>
                  collectTrees(a.getArguments.asScala.toList) {
                    case s: LiteralTree if s.getKind == Tree.Kind.STRING_LITERAL =>
                      s.getValue.asInstanceOf[String] :: Nil
                    case _ => Nil
                  }
                case _ => Nil
              }
            case _ => Nil
          }).toList match {
            case x :: Nil => x.success
            case xs => s"Expecting one string literal as args to getFunctionalInterfaceClass().equals(), got $xs. Tree:\n$tree".fail
          }
        } yield (c->name)



        println(">>" + rr)

        ().success

      }

      Nil.success

    case xs =>
      s"Expecting two switch statements from method body of $$deserializeLambda$$, got ${xs.size}. Tree:\n:${xs
        .mkString("\n")}".fail
  }

  def findDeserialiseLambdaMethods(cu: CompilationUnitTree): List[MethodTree] = collectTree(cu) {
    case m: MethodTree =>
      (
        m.getModifiers.getFlags.asScala.toSet,
        m.getName.toString,
        m.getParameters.asScala
          .map(_.getType)
          .collect {
            case i: MemberSelectTree => i.getIdentifier.toString // general case of unqualified use
            case i: IdentifierTree   => i.getName.toString       // in case the call site imports the class
          }
          .toList
      ) match {
        case (modifiers, "$deserializeLambda$", "SerializedLambda" :: Nil)
            if modifiers == Set(Modifier.PRIVATE, Modifier.STATIC) =>
          m :: Nil
        case _ => Nil
      }
    case _ => Nil
  }

}

@SupportedSourceVersion(SourceVersion.RELEASE_8)
class Processor extends AbstractProcessor {

  override def getSupportedAnnotationTypes: _root_.java.util.Set[String] =
    _root_.java.util.Collections.singleton[String]("*")

  override def getSupportedSourceVersion: SourceVersion = SourceVersion.RELEASE_8

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
      _ <- addOpens(allPkgs).adaptError(e => new RuntimeException("Unable to add opens at runtime", e))
      tasks <- Either
        .catchNonFatal(JavacTask.instance(processingEnv))
        .adaptError(e => new RuntimeException("Unable to obtain an instance of JavacTask", e))
      trees <- Either
        .catchNonFatal(Trees.instance(processingEnv))
        .adaptError(e => new RuntimeException("Unable to obtain an instance of Trees", e))
    } yield {
      val visited = mutable.Set[JavaFileObject]()

      tasks.addTaskListener(new TaskListener {
        override def started(e: TaskEvent): Unit = () // don't care
        override def finished(e: TaskEvent): Unit = {
          println(s"EV, end  =${e}")
          (e.getKind, e.getCompilationUnit, e.getSourceFile) match {
            case (TaskEvent.Kind.GENERATE, cu, file) if cu != null && !visited.contains(file) =>
              visited += file

              findDeserialiseLambdaMethods(cu) match {
                case x :: Nil =>
                  extractInvokeStaticTargets(
                    Set("polyregion/java/OffloadFunction", "polyregion/java/OffloadRunnable"),
                    x,
                    cu
                  )

                case Nil => // skip, no serialisable lambda found
                case xs =>
                  throw new AssertionError(
                    s"More than one $$deserializeLambda$$ found, something is broken! Implementations:\n${xs.mkString("\n")}"
                  )
              }

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
      })
    }) match {
      case Right(()) =>
      case Left(e) =>
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
