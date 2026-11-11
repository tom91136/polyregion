package polyregion.javalang

import cats.syntax.all.*
import com.sun.source.tree.*
import com.sun.source.util.TreeScanner
import polyregion.ast.*

import scala.annotation.nowarn
import javax.lang.model.`type`.{ExecutableType, TypeKind, TypeMirror}
import javax.lang.model.element.{ExecutableElement, Modifier, TypeElement}
import javax.lang.model.util.TypeKindVisitor6
import scala.jdk.CollectionConverters.*

object LambdaOutliner {

  case class Reflector(
      classTable: Map[String, TypeElement],
      methodTable: Map[String, MethodTree]
  )

  private def resolveExecutableElementDesc(e: ExecutableElement) = {
    def mirrorToDesc(m: TypeMirror) = m.getKind match {
      case TypeKind.BOOLEAN  => "Z"
      case TypeKind.BYTE     => "B"
      case TypeKind.SHORT    => "S"
      case TypeKind.INT      => "I"
      case TypeKind.LONG     => "J"
      case TypeKind.CHAR     => "C"
      case TypeKind.FLOAT    => "F"
      case TypeKind.DOUBLE   => "D"
      case TypeKind.ARRAY    => "["
      case TypeKind.VOID     => "V"
      case TypeKind.DECLARED => s"L${m};"
      case illegal           => s"<illegal:$illegal>"
    }
    val tpe = e.asType().asInstanceOf[ExecutableType]
    tpe.getParameterTypes.asScala.map(mirrorToDesc(_)).mkString("(", "", ")") + mirrorToDesc(tpe.getReturnType)
  }

  case class FnMetadata(interfaceCls: String, implCls: String, methodSignature: String)

  def findMethod(
      reflector: Reflector,
      cu: CompilationUnitTree,
      invokeFn: String,
      target: FnMetadata
  ): Result[MethodTree] =
    // first, search in the local CompilationUnit as $deserializeLambda$ and the corresponding synthetic static
    // method must be defined there we only need to match the method name as the generated name is unique
    // it's also not possible to reflect this method for the signatures because it's synthesised after RoundEnvironment
    collectTree(cu) {
      case m: MethodTree if m.getName.toString == invokeFn => m :: Nil
      case _                                               => Nil
    } match {
      case x :: Nil => x.success // found the direct tree
      case Nil      =>
        // we didn't find a synthetic method so the target is a real method defined somewhere that can be hopefully reflected
        reflector.classTable.get(target.implCls) match {
          case None =>
            s"Cannot find class ${target.implCls} (which defines method $invokeFn), classes visible include:\n${reflector.classTable.keys
                .mkString("\n")}".fail
          case Some(clsElem) =>
            val matchingReflectedMethods = clsElem.getEnclosedElements.asScala.toList.collect {
              case e: ExecutableElement
                  if e.getSimpleName.toString == invokeFn &&
                    resolveExecutableElementDesc(e) == target.methodSignature =>
                e
            }
            def unresolved(kind: String, outcome: String) =
              s"$kind $invokeFn with signature ${target.methodSignature} (defined in $clsElem) $outcome, " +
                s"available methods in the same class are:\n${matchingReflectedMethods.mkString("\n")}"
            matchingReflectedMethods match {
              case x :: Nil =>
                // TODO workout <init>
                reflector.methodTable.get(s"${clsElem.getQualifiedName}.${x.getSimpleName}") match {
                  case Some(m) => m.success
                  case None    => s"Method $x (defined in $clsElem) has a non-reified tree".fail
                }
              case Nil => unresolved("Method", "not found").fail
              case xs  => unresolved("More tha one overload of", "found").fail
            }
        }
      case xs => s"More than one method match the name $invokeFn, all signatures: ${xs}".fail
    }

  def extractDeserializeLambdaMethods(
      reflector: Reflector,
      allowedIfaceNames: Set[String],
      cu: CompilationUnitTree
  ): Result[List[(String, FnMetadata)]] = {
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

    // find the correct synthetic $deserializeLambda$ method
    val deserializeLambdaMethods = collectTree(cu) {
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

    deserializeLambdaMethods match {
      case m :: Nil =>
        collectTree(m.getBody) {
          case s: SwitchTree => s :: Nil
          case _             => Nil
        } match {
          case sourceNameSwitch :: targetSwitch :: Nil =>
            // first switch, extract the target's method name and the target label
            val labeledInvokeFn = sourceNameSwitch.getCases.asScala.toList.traverse { tree =>
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
                case (label :: Nil, invokeFnName :: Nil) => (label, invokeFnName).success
                case (cases, names) =>
                  s"Expecting one int literal target and one String literal invokeFn from case body, target was $cases and invokeFn was $names. Tree:\n$tree".fail
              }
            }

            // second switch, extract the assertion values on the label
            val labeledTarget = targetSwitch.getCases.asScala.toList.traverse { tree =>
              for {
                label <-
                  (if (tree.getExpression.getKind == Tree.Kind.INT_LITERAL) {
                     tree.getExpression.asInstanceOf[LiteralTree].getValue.asInstanceOf[Int].success
                   } else s"Case expression is not an int literal. Tree:\n$tree".fail): @nowarn
                target <- tree.getStatements.asScala.toList.flatMap(collectTree(_) {
                  case a: MethodInvocationTree =>
                    collectTree(a.getMethodSelect) {
                      case i: MemberSelectTree => i.getIdentifier.toString :: Nil
                      case _                   => Nil
                    } match {
                      case ("getFunctionalInterfaceClass" | "getImplClass" |
                          "getImplMethodSignature") :: "equals" :: Nil =>
                        collectTrees(a.getArguments.asScala.toList) {
                          case s: LiteralTree if s.getKind == Tree.Kind.STRING_LITERAL =>
                            s.getValue.asInstanceOf[String] :: Nil
                          case _ => Nil
                        }
                      case _ => Nil
                    }
                  case _ => Nil
                }) match {
                  // we get repeats of the same triple if we refer to the same method reference multiple times
                  case s0 :: s1 :: s2 :: xs if xs.isEmpty || xs.grouped(3).forall(_ == List(s0, s1, s2)) =>
                    FnMetadata(
                      interfaceCls = asCanonicalName(s0),
                      implCls = asCanonicalName(s1),
                      methodSignature = s2
                    ).success
                  case xs =>
                    (s"Expecting three string literal as args to " +
                      s"getFunctionalInterfaceClass().equals(), " +
                      s"getImplClass().equals(), " +
                      s"and getImplMethodSignature().equals()" +
                      s", got $xs. Tree:\n$tree").fail
                }

              } yield label -> target
            }

            // we got both invoke target and assertion data, keep the ones that conform to our SAM interface and find the tree of this method
            for {
              invokeFns <- labeledInvokeFn
              targets   <- labeledTarget
              interfaceLookup = targets.toMap
              method <- invokeFns.traverseFilter { (label, invokeFn) =>
                interfaceLookup.get(label) match {
                  case None =>
                    s"Label $label ($invokeFn) does not have a corresponding target case in the second switch".fail
                  case Some(target) if allowedIfaceNames contains target.interfaceCls =>
                    Some(invokeFn -> target).success
                  case _ => None.success
                }
              }
            } yield method
          case bad =>
            (s"Expecting two switch statements from method body of $$deserializeLambda$$, got ${bad.size}. " +
              s"Tree:\n:${bad.mkString("\n")}").fail
        }
      case Nil => Nil.success // skip, class doesn't have serialisable lambda defined at all
      case xs =>
        s"More than one $$deserializeLambda$$ found, something is broken! Implementations:\n${xs.mkString("\n")}".fail
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

}
