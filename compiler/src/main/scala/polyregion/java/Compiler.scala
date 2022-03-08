package polyregion.java

import com.sun.source.tree.*
import com.sun.source.util.{SimpleTreeVisitor, TreeScanner, Trees}
import com.sun.tools.javac.tree.JCTree
import com.sun.tools.javac.tree.JCTree.JCLambda
import polyregion.ast.PolyAst as p

import javax.annotation.processing.{ProcessingEnvironment, RoundEnvironment}
import javax.lang.model.element.{Element, TypeElement}
import javax.tools.Diagnostic.Kind
import javax.tools.{JavaFileManager, StandardLocation}
import scala.collection.immutable.ArraySeq
import scala.jdk.CollectionConverters.*

object Compiler {

  def visit(elem: Element, trees: Trees, t: Tree, e: ProcessingEnvironment): String = {


//    t.accept(new TreeVisitor[String, Unit]{
//      override def visitAnnotatedType(node: AnnotatedTypeTree, ignore: Unit) = s">> $node"
//      override def visitAnnotation(node: AnnotationTree, ignore: Unit) = s">> $node"
//
//      override def visitMethodInvocation(node: MethodInvocationTree, ignore: Unit) = s">> $node => ${node.getArguments.asScala.map(visit(_)).mkString("\n")}"
//      override def visitAssert(node: AssertTree, ignore: Unit) = s">> $node"
//      override def visitAssignment(node: AssignmentTree, ignore: Unit) = s">> $node"
//      override def visitCompoundAssignment(node: CompoundAssignmentTree, ignore: Unit) = s">> $node"
//      override def visitBinary(node: BinaryTree, ignore: Unit) = {
//
//
//        node.getKind match {
//          case Tree.Kind.MULTIPLY => p.BinaryIntrinsicKind.Mul
//          case Tree.Kind.DIVIDE =>  p.BinaryIntrinsicKind.Div
//          case Tree.Kind.REMAINDER => p.BinaryIntrinsicKind.Rem
//          case Tree.Kind.PLUS => p.BinaryIntrinsicKind.Add
//          case Tree.Kind.MINUS => p.BinaryIntrinsicKind.Sub
//          case Tree.Kind.LEFT_SHIFT => p.BinaryIntrinsicKind.BSL
//          case Tree.Kind.RIGHT_SHIFT => p.BinaryIntrinsicKind.BSR
//          case Tree.Kind.UNSIGNED_RIGHT_SHIFT => p.BinaryIntrinsicKind.BZSR
//          case Tree.Kind.LESS_THAN => p.BinaryLogicIntrinsicKind.Lt
//          case Tree.Kind.GREATER_THAN =>p.BinaryLogicIntrinsicKind.Gt
//          case Tree.Kind.LESS_THAN_EQUAL => p.BinaryLogicIntrinsicKind.Lte
//          case Tree.Kind.GREATER_THAN_EQUAL => p.BinaryLogicIntrinsicKind.Gte
//          case Tree.Kind.EQUAL_TO => p.BinaryLogicIntrinsicKind.Eq
//          case Tree.Kind.NOT_EQUAL_TO =>p.BinaryLogicIntrinsicKind.Neq
//          case Tree.Kind.AND => p.BinaryIntrinsicKind.BAnd
//          case Tree.Kind.XOR => p.BinaryIntrinsicKind.BXor
//          case Tree.Kind.OR => p.BinaryIntrinsicKind.BOr
//          case Tree.Kind.CONDITIONAL_AND => p.BinaryLogicIntrinsicKind.And
//          case Tree.Kind.CONDITIONAL_OR =>p.BinaryLogicIntrinsicKind.Or
//          case _ => ???
//        }
//
//        s"${visit(node.getLeftOperand)} ${node.getKind} ${visit(node.getRightOperand)}"
//
//      }
//      override def visitBlock(node: BlockTree, ignore: Unit) = s">> $node"
//      override def visitBreak(node: BreakTree, ignore: Unit) = s">> $node"
//      override def visitCase(node: CaseTree, ignore: Unit) = s">> $node"
//      override def visitCatch(node: CatchTree, ignore: Unit) = s">> $node"
//      override def visitClass(node: ClassTree, ignore: Unit) = {
//
//        s"saw ${node} ${node.getMembers.asScala.toList.map(visit(_)).mkString("\n")}"
//      }
//      override def visitConditionalExpression(node: ConditionalExpressionTree, ignore: Unit) = s">> $node"
//      override def visitContinue(node: ContinueTree, ignore: Unit) = s">> $node"
//      override def visitDoWhileLoop(node: DoWhileLoopTree, ignore: Unit) = s">> $node"
//      override def visitErroneous(node: ErroneousTree, ignore: Unit) = s">> $node"
//      override def visitExpressionStatement(node: ExpressionStatementTree, ignore: Unit) = s">> $node"
//      override def visitEnhancedForLoop(node: EnhancedForLoopTree, ignore: Unit) = s">> $node"
//      override def visitForLoop(node: ForLoopTree, ignore: Unit) = s">> $node"
//      override def visitIdentifier(node: IdentifierTree, ignore: Unit) = {
//        "ID{"+node.getName.toString+"}"
//      }
//      override def visitIf(node: IfTree, ignore: Unit) = s">> $node"
//      override def visitImport(node: ImportTree, ignore: Unit) = s">> $node"
//      override def visitArrayAccess(node: ArrayAccessTree, ignore: Unit) = s">> $node"
//      override def visitLabeledStatement(node: LabeledStatementTree, ignore: Unit) = s">> $node"
//      override def visitLiteral(node: LiteralTree, ignore: Unit) = {
//        s">> $node"
//      }
//      override def visitBindingPattern(node: BindingPatternTree, ignore: Unit) = s">> $node"
//      override def visitDefaultCaseLabel(node: DefaultCaseLabelTree, ignore: Unit) = s">> $node"
//      override def visitMethod(node: MethodTree, ignore: Unit) = {
//        val xs = Option(node.getBody).toList.flatMap(_.getStatements.asScala.toList).map(s => visit(s))
//        xs.mkString("\n")
//      }
//      override def visitModifiers(node: ModifiersTree, ignore: Unit) = s">> $node"
//      override def visitNewArray(node: NewArrayTree, ignore: Unit) = s">> $node"
//      override def visitGuardedPattern(node: GuardedPatternTree, ignore: Unit) = s">> $node"
//      override def visitParenthesizedPattern(node: ParenthesizedPatternTree, ignore: Unit) = s">> $node"
//      override def visitNewClass(node: NewClassTree, ignore: Unit) = s">> $node"
//      override def visitLambdaExpression(node: LambdaExpressionTree, ignore: Unit) = s"LLLL $node"
//      override def visitPackage(node: PackageTree, ignore: Unit) = s">> $node"
//      override def visitParenthesized(node: ParenthesizedTree, ignore: Unit) = s">> $node"
//      override def visitReturn(node: ReturnTree, ignore: Unit) = s">> $node"
//
//      override def visitMemberSelect(node: MemberSelectTree, ignore: Unit) = s">> $node"
//      override def visitMemberReference(node: MemberReferenceTree, ignore: Unit) = s">> $node"
//
//      override def visitEmptyStatement(node: EmptyStatementTree, ignore: Unit) = s">> $node"
//
//      override def visitSwitch(node: SwitchTree, ignore: Unit) = s">> $node"
//      override def visitSwitchExpression(node: SwitchExpressionTree, ignore: Unit) = s">> $node"
//
//      override def visitSynchronized(node: SynchronizedTree, ignore: Unit) = s">> $node"
//      override def visitThrow(node: ThrowTree, ignore: Unit) = s">> $node"
//
//      override def visitCompilationUnit(node: CompilationUnitTree, ignore: Unit) = s">> $node"
//      override def visitTry(node: TryTree, ignore: Unit) = s">> $node"
//
//      override def visitParameterizedType(node: ParameterizedTypeTree, ignore: Unit) = s">> $node"
//      override def visitUnionType(node: UnionTypeTree, ignore: Unit) = s">> $node"
//      override def visitIntersectionType(node: IntersectionTypeTree, ignore: Unit) = s">> $node"
//      override def visitArrayType(node: ArrayTypeTree, ignore: Unit) = s">> $node"
//      override def visitTypeCast(node: TypeCastTree, ignore: Unit) = s">> $node"
//      override def visitPrimitiveType(node: PrimitiveTypeTree, ignore: Unit) = node.getPrimitiveTypeKind.toString
//
//      override def visitTypeParameter(node: TypeParameterTree, ignore: Unit) = s">> $node"
//      override def visitInstanceOf(node: InstanceOfTree, ignore: Unit) = s">> $node"
//      override def visitUnary(node: UnaryTree, ignore: Unit) = s">> $node"
//      override def visitVariable(node: VariableTree, ignore: Unit) = {
//        s">> $node"
//
////        val expr = if(node.getInitializer == null){
////          "_"
////        }else {
////          visit(node.getInitializer)
////        }
////
//////        println(s"t=${node.getInitializer.getKind}")
////
////        (s"${node.getName}:${visit(node.getType)} = ${node.getInitializer}")
//      }
//      override def visitWhileLoop(node: WhileLoopTree, ignore: Unit) = s">> $node"
//      override def visitWildcard(node: WildcardTree, ignore: Unit) = s">> $node"
//      override def visitModule(node: ModuleTree, ignore: Unit) = s">> $node"
//      override def visitExports(node: ExportsTree, ignore: Unit) = s">> $node"
//      override def visitOpens(node: OpensTree, ignore: Unit) = s">> $node"
//      override def visitProvides(node: ProvidesTree, ignore: Unit) = s">> $node"
//      override def visitRequires(node: RequiresTree, ignore: Unit) = s">> $node"
//      override def visitUses(node: UsesTree, ignore: Unit) = s">> $node"
//      override def visitOther(node: Tree, ignore: Unit) = s">> $node"
//      override def visitYield(node: YieldTree, ignore: Unit) = s">> $node"
//    }, ())

    val x = new TreeScanner[String, Unit] {

      override def visitMethodInvocation(node: MethodInvocationTree, p: Unit): String = {
        println(">>>" + node)
        val cu = trees.getPath(elem).getCompilationUnit
        println(s">>> at ${trees.getSourcePositions.getStartPosition(cu, node)}")
        println(cu.getLineMap.getLineNumber(trees.getSourcePositions.getStartPosition(cu, node)))

        ""
      }

      override def visitLambdaExpression(node: LambdaExpressionTree, p: Unit) = {
        println(">>>" + node)

        val cu = trees.getPath(elem).getCompilationUnit
        println(s">>> at ${trees.getSourcePositions.getStartPosition(cu, node)}")

        println(cu.getLineMap.getLineNumber(trees.getSourcePositions.getStartPosition(cu, node)))

//        println(trees.getElement(trees.getPath(trees.getPath(elem).getCompilationUnit, node)))
//        println("tag="+trees.get)

//        println(e.getElementUtils.getAllMembers(trees.getElement(trees.getPath(elem)).asInstanceOf ).asScala.toList)

        node.toString
      }
      override def reduce(r1: String, r2: String) = r1 ++ r2
    }

    println(">" + elem.getEnclosedElements.asScala.toList)

    x.scan(t, ())

  }

  def compile(
      trees: Trees,
      annotations: java.util.Set[_ <: TypeElement],
      roundEnv: RoundEnvironment,
      processingEnv: ProcessingEnvironment
  ): Boolean = {

    println(s"Roots = ${roundEnv.getRootElements.asScala.toList}")

    roundEnv.getRootElements.asScala.foreach { e =>
      println(visit(e, trees, trees.getTree(e), processingEnv))
    }

//    val xs = processingEnv.getFiler.getResource(StandardLocation.CLASS_OUTPUT, "polyregion", "AnnotationTest.class")
//    try{
//      println(s">>>${xs.getCharContent(false)}")
//    } catch {
//      case e : Throwable => e.printStackTrace()
//    }

//    processingEnv.getElementUtils.getBinaryName()

//    for (annotation <- annotations.asScala) {
//      val annotatedElements = roundEnv.getElementsAnnotatedWith(annotation)
//      for (element <- annotatedElements.asScala) {
//
//        println("Got elem: " + element.getSimpleName.toString)
//        println(s"Tree=${trees.getTree(element)}")
//
//
//
//
//        println("-----")
//        println(visit(trees.getTree(element)).indent(2))
//        println("-----")
//
//
//
//
////        processingEnv.getMessager.printMessage(Kind.WARNING, )
//      }
//    }
    true
  }

}
