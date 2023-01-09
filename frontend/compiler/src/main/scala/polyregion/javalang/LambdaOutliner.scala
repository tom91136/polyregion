package polyregion.javalang

import cats.syntax.all.*
import com.sun.source.tree.*
import com.sun.source.util.Trees
import polyregion.ast.*
import polyregion.javalang.Processor.{asCanonicalName, collectTree, collectTrees}

import scala.annotation.nowarn
import javax.annotation.processing.ProcessingEnvironment
import javax.lang.model.`type`.{ExecutableType, TypeKind, TypeMirror}
import javax.lang.model.element.{ExecutableElement, Modifier, TypeElement}
import javax.lang.model.util.TypeKindVisitor6
import scala.jdk.CollectionConverters.*

object LambdaOutliner {

  case class Reflector(
      env: ProcessingEnvironment,
      trees: Trees,
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
            matchingReflectedMethods match {
              case x :: Nil =>
                // TODO workout <init>
                reflector.methodTable.get(s"${clsElem.getQualifiedName}.${x.getSimpleName}") match {
                  case Some(m) => m.success
                  case None    => s"Method $x (defined in $clsElem) has a non-reified tree".fail
                }
              case Nil =>
                (s"Method $invokeFn with signature ${target.methodSignature} (defined in $clsElem) not found, " +
                  s"available methods in the same class are:\n${matchingReflectedMethods.mkString("\n")}").fail
              case xs =>
                (s"More tha one overload of $invokeFn with signature ${target.methodSignature} (defined in $clsElem) found, " +
                  s"available methods in the same class are:\n${matchingReflectedMethods.mkString("\n")}").fail
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

}
