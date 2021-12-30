package polyregion.data

object compileTime {

  import scala.quoted.*

  enum MirrorKind { case CaseClass, CaseSum, CaseProduct }
  given ToExpr[MirrorKind] with {
    def apply(x: MirrorKind)(using Quotes) =
      import quotes.reflect._
      x match {
        case MirrorKind.CaseClass   => '{ MirrorKind.CaseClass }
        case MirrorKind.CaseSum     => '{ MirrorKind.CaseSum }
        case MirrorKind.CaseProduct => '{ MirrorKind.CaseProduct }
      }
  }

  inline def mirrorMeta[T] =
    ${ mirrorMetaImpl[T] }

  def mirrorMetaImpl[T: Type](using quotes: Quotes): Expr[(List[String], MirrorKind)] = {
    import quotes.reflect.*
    val tpe   = TypeRepr.of[T].dealias.simplified
    val sym   = tpe.typeSymbol
    val flags = Set(Flags.Case, Flags.Enum, Flags.Sealed)

    // case class E             # !singleton  case !enum !sealed
    // enum A /* extends X */ { # !singleton !case  enum  sealed
    //   case B                 #  singleton !case  enum  sealed
    //   case C(s : D)          # !singleton  case  enum !sealed
    // }

    val (parents, kind) =
      (tpe.isSingleton, sym.flags.is(Flags.Case), sym.flags.is(Flags.Enum), sym.flags.is(Flags.Sealed)) match {
        case (false, true, false, false) => (tpe.baseClasses.tail, MirrorKind.CaseClass)
        case (false, false, true, true)  => (tpe.baseClasses, MirrorKind.CaseSum)
        case (false, false, false, true) => (tpe.baseClasses, MirrorKind.CaseSum)
        case (true, false, true, true)   => (tpe.baseClasses, MirrorKind.CaseProduct)
        case (false, true, true, false)  => (tpe.baseClasses.tail, MirrorKind.CaseProduct)
        case (s, c, e, se) =>
          report.errorAndAbort(
            s"Type ${tpe.show} not an enum, enum case, or case class (singleton=$s, case=$c, enum=$e, sealed=$se)"
          )
      }
    val names = parents.filter(s => flags.exists(s.flags.is(_))).map(_.name).reverse
    Expr((names, kind))
  }

  type TermResolver[A, R] = Option[A] => R
  type TypeResolver[R]    = () => R

  case class CtorTermSelect[A](arg: (String, A), rest: List[(String, A)])

  inline def primaryCtorApplyTerms[
      T,
      Val,
      TermRes <: TermResolver[*, Val],
      Tpe,
      TypeRes[_] <: TypeResolver[Tpe]
  ] = ${ primaryCtorApplyTermsImpl[T, Val, TermRes, Tpe, TypeRes] }

  def primaryCtorApplyTermsImpl[
      T: Type,
      Val: Type,
      TermRes <: TermResolver[*, Val]: Type,
      Tpe: Type,
      TypeRes[_] <: TypeResolver[Tpe]: Type
  ](using quotes: Quotes): Expr[List[Val]] = {
    import quotes.reflect.*

    val tpe    = TypeRepr.of[T].dealias.simplified
    val symbol = tpe.typeSymbol

    def summonTermRes[A: Type](f: Expr[TermRes[A]] => Expr[Val]): Expr[Val] = Expr.summon[TermRes[A]] match {
      case Some(imp) => f(imp)
      case _         => report.errorAndAbort(s"No implicit found for ${TypeRepr.of[TermRes[A]].show}")
    }
    def summonTypeRes(r: TypeRepr) = Implicits.search(TypeRepr.of[TypeRes].appliedTo(r)) match {
      case s: ImplicitSearchSuccess => s.tree
      case _ => report.errorAndAbort(s"No implicit found for ${TypeRepr.of[TypeRes].appliedTo(r).show}")
    }

    def const[A: Type: ToExpr](e: A) = summonTermRes[A](tc => '{ ${ tc }(Some(${ Expr(e) })) })
    def select(name: String, tpe: TypeRepr, rest: List[(String, TypeRepr)]): Expr[Val] = {
      def resolveOne(name: String, tpe: TypeRepr) =
        '{ (${ Expr(name) }, ${ summonTypeRes(tpe.widenTermRefByName.dealias).asExpr }.asInstanceOf[TypeRes[Any]]()) }
      val sel = '{
        CtorTermSelect[Tpe](${ resolveOne(name, tpe) }, ${ (Expr.ofList(rest.map(resolveOne(_, _)))) })
      }
      summonTermRes[CtorTermSelect[Tpe]](tc => '{ ${ tc }(Some($sel)) })
    }

    def matchCtor(t: Tree, parent: Option[TypeRepr]): Option[Expr[List[Val]]] = t match {

      case a @ Apply(Select(New(tpeTree), "<init>"), args) if parent.forall(tpeTree.tpe =:= _) =>
        Some(Expr.ofList(args.map {
          case i @ Ident(x) =>
            if (i.symbol.owner == symbol) select(x, i.tpe, Nil)
            else report.errorAndAbort(s"Ident ${i.show} not part of class ${symbol.fullName}")
          case s @ Select(qualifier, name) =>
            if (qualifier.symbol.owner == symbol) {
              // we're selecting from one of the ctor args
              select(qualifier.symbol.name, qualifier.tpe, (name, s.tpe) :: Nil)
            } else {
              // some other constant
              val applied = TypeRepr.of[TermRes].appliedTo(s.tpe.dealias)
              println(s"${applied.show}")
              Implicits.search(applied) match {
                case imp: ImplicitSearchSuccess => '{ ${ imp.tree.asExpr }.asInstanceOf[TermRes[Any]](None) }
                case _                          => report.errorAndAbort(s"No implicit found for ${applied.show}")
              }
            }
          case Literal(BooleanConstant(x)) => const(x)
          case Literal(ByteConstant(x))    => const(x)
          case Literal(ShortConstant(x))   => const(x)
          case Literal(IntConstant(x))     => const(x)
          case Literal(LongConstant(x))    => const(x)
          case Literal(FloatConstant(x))   => const(x)
          case Literal(DoubleConstant(x))  => const(x)
          case Literal(CharConstant(x))    => const(x)
          case Literal(StringConstant(x))  => const(x)
          // case Literal(UnitConstant)       => constTC(())
          // case Literal(NullConstant)       => constTC(null)
          // case Literal(ClassOfConstant(x)) => constTC(x)

        }))
      case _ => None
    }

    // Scala implements enum cases w/o params as vals in the companion with an anonymous cls
    // pprint.pprintln(symbol.companionClass.tree.show)
    (if (tpe.isSingleton) {
       class CtorTraverser extends TreeAccumulator[Option[Expr[List[Val]]]] {
         def foldTree(x: Option[Expr[List[Val]]], tree: Tree)(owner: Symbol): Option[Expr[List[Val]]] =
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

  inline def sumTypeCtorParams[T, U, TypeRes[_] <: TypeResolver[U]] =
    ${ sumTypeCtorParamsImpl[T, U, TypeRes] }

  def sumTypeCtorParamsImpl[T: Type, U: Type, TypeRes[_] <: TypeResolver[U]: Type](using
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
            val instance = Implicits.search(TypeRepr.of[TypeRes].appliedTo(tpeTree.tpe)) match {
              case s: ImplicitSearchSuccess => s.tree
            }
            '{ (${ Expr(name) }, ${ instance.asExpr }.asInstanceOf[TypeRes[Any]]()) }
          })
        case DefDef("<init>", args, _, _) => report.errorAndAbort(s"Ctor has more than one parameter list (${args})")
      }

    }

  }

}
