package polyregion.data

object compileTime {

  import scala.quoted.*

  inline def primaryCtorApplyTerms[T, U, TC[x] <: x => U] =
    ${ primaryCtorApplyTermsImpl[T, U, TC] }

  def primaryCtorApplyTermsImpl[T: Type, U: Type, TC[x] <: x => U: Type](using quotes: Quotes): Expr[List[U]] = {
    import quotes.reflect.*

    val tpe    = TypeRepr.of[T].dealias.simplified
    val symbol = tpe.typeSymbol

    def res[A: Type: ToExpr](e: A) = Expr.summon[TC[A]] match {
      case Some(x) => '{ ${ x }(${ Expr(e) }) }
      case _       => report.errorAndAbort(s"No implicit found for ${TypeRepr.of[TC[A]].show}")
    }

    def matchCtor(t: Tree, parent: Option[TypeRepr]): Option[Expr[List[U]]] = t match {

      case a @ Apply(Select(New(tpeTree), "<init>"), args) if parent.forall(tpeTree.tpe =:= _) =>
        Some(Expr.ofList(args.map {
          case i @ Ident(x) =>
            if (i.symbol.owner == symbol) res(x :: Nil)
            else report.errorAndAbort(s"Ident ${i.show} not part of class ${symbol.fullName}")
          case Literal(BooleanConstant(x)) => res(x)
          case Literal(ByteConstant(x))    => res(x)
          case Literal(ShortConstant(x))   => res(x)
          case Literal(IntConstant(x))     => res(x)
          case Literal(LongConstant(x))    => res(x)
          case Literal(FloatConstant(x))   => res(x)
          case Literal(DoubleConstant(x))  => res(x)
          case Literal(CharConstant(x))    => res(x)
          case Literal(StringConstant(x))  => res(x)
          // case Literal(UnitConstant)       => res(())
          // case Literal(NullConstant)       => res(null)
          // case Literal(ClassOfConstant(x)) => res(x)
          case s @ Select(a, b) =>
            // println(s"  >${a.tpe.show}"
            val applied = TypeRepr.of[TC].appliedTo(s.tpe.dealias.simplified)
            Implicits.search(applied) match {
              case imp: ImplicitSearchSuccess => '{ ${ imp.tree.asExpr }.asInstanceOf[TC[Any]](${ s.asExpr }) }
              case _                          => report.errorAndAbort(s"No implicit found for ${applied.show}")
            }
        }))
      case _ => None
    }

    // Scala implements enum cases w/o params as vals in the companion with an anonymous cls
    // pprint.pprintln(symbol.companionClass.tree.show)
    (if (tpe.isSingleton) {
       class CtorTraverser extends TreeAccumulator[Option[Expr[List[U]]]] {
         def foldTree(x: Option[Expr[List[U]]], tree: Tree)(owner: Symbol): Option[Expr[List[U]]] =
           x.orElse(matchCtor(tree, Some(tpe.widenTermRefByName)).orElse(foldOverTree(None, tree)(symbol)))
       }
       CtorTraverser().foldOverTree(None, symbol.companionClass.tree)(Symbol.noSymbol)
     } else {
       symbol.tree match {
         case ClassDef(name, _, headCtorApply :: _, _, _) =>
//           println(s"match!:$name")
//           pprint.pprintln(headCtorApply)
           matchCtor(headCtorApply, None)
         case _ => report.errorAndAbort(s"Not a classdef")
       }
     })
    match {
      case Some(expr) => expr
      case None       => Expr(Nil) // report.errorAndAbort(s"Unrecognised ctor pattern")
    }
  }

  inline def sumTypeCtorParams[T, U, TC[_] <: () => U] =
    ${ sumTypeCtorParamsImpl[T, U, TC] }

  def sumTypeCtorParamsImpl[T: Type, U: Type, TC[_] <: () => U: Type](using
      quotes: Quotes
  ): Expr[List[(String, U)]] = {
    import quotes.reflect.*
    val symbol = TypeRepr.of[T].dealias.simplified.typeSymbol
    if (symbol.primaryConstructor.isNoSymbol)
      report.errorAndAbort(s"No Ctor symbol for ${symbol.name}")
    else if (symbol.caseFields.nonEmpty)
      report.errorAndAbort(s"Sum type required, got product type (${symbol.name}) instead")
    else {
      symbol.primaryConstructor.tree match {
        case DefDef("<init>", head :: Nil, _, _) =>
//          println(s"Non-Sealed Ctor: ${head}")
          Expr.ofList(head.params.collect { case ValDef(name, tpeTree, _) =>
            val instance = Implicits.search(TypeRepr.of[TC].appliedTo(tpeTree.tpe)) match {
              case s: ImplicitSearchSuccess => s.tree
            }
            '{ (${ Expr(name) }, ${ instance.asExpr }.asInstanceOf[TC[Any]]()) }
          })
        case DefDef("<init>", args, _, _) => report.errorAndAbort(s"Ctor has more than one parameter list (${args})")
      }

    }

  }

}
