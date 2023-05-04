package polyregion.ast.mirror

private[polyregion] object compiletime {

  import scala.quoted.*

  enum MirrorKind { case CaseClass, CaseSum, CaseProduct }
  given ToExpr[MirrorKind] with {
    def apply(x: MirrorKind)(using Quotes) =
      x match {
        case MirrorKind.CaseClass   => '{ MirrorKind.CaseClass }
        case MirrorKind.CaseSum     => '{ MirrorKind.CaseSum }
        case MirrorKind.CaseProduct => '{ MirrorKind.CaseProduct }
      }
  }
  inline def symbolNames[T] = ${ symbolNamesImpl[T] }
  def symbolNamesImpl[T: Type](using quotes: Quotes): Expr[List[String]] = {
    import quotes.reflect.*
    val sym  = TypeRepr.of[T].dealias.simplified.typeSymbol
    val name = List.unfold(sym)(s => if (!s.isPackageDef) Some((s.name, s.owner)) else None)
    Expr(name.map(_.replace("$", "")).reverse)
  }

  inline def mirrorMeta[T] = ${ mirrorMetaImpl[T] }
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

    // report.warning(tpe.typeSymbol.toString())

    val names = parents.filter(s => flags.exists(s.flags.is(_))).reverse

    val owners = names.headOption
      .map { s =>

        val prefix = List
          .unfold(s.owner)(s =>
            if (!s.isPackageDef && !s.owner.isPackageDef) Some((s.name.replace("$", ""), s.owner)) else None
          )
          .reverse
          .mkString

        names.map(_.name) match {
          case x :: xs => s"$prefix$x" :: xs
          case xs      => xs
        }
      }
      .getOrElse(names.map(_.name))

    Expr((owners, kind))
  }

  type TermResolver[A, R] = Option[A] => R
  type TypeResolver[R]    = () => R

  enum Value[+A] {
    case Const(value: String)
    case TermSelect(arg: (String, A), rest: List[(String, A)])
    case CtorAp(tpe: A, args: List[Value[A]])
  }

  // case class CtorTermSelect[A](arg: (String, A), rest: List[(String, A)])
  // case class CtorAp[A](tpe: A, args: )

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
    def summonTypeRes(r: TypeRepr): Expr[TypeRes[Any]] = Implicits.search(TypeRepr.of[TypeRes].appliedTo(r)) match {
      case s: ImplicitSearchSuccess => '{ ${ s.tree.asExpr }.asInstanceOf[TypeRes[Any]] }
      case _ => report.errorAndAbort(s"No implicit found for ${TypeRepr.of[TypeRes].appliedTo(r).show}")
    }

    def selectValue(name: String, tpe: TypeRepr, rest: List[(String, TypeRepr)]): Expr[Value[Tpe]] = {
      def resolveOne(name: String, tpe: TypeRepr) = '{
        (${ Expr(name) }, ${ summonTypeRes(tpe.widenTermRefByName.dealias) }())
      }
      '{ Value.TermSelect[Tpe](${ resolveOne(name, tpe) }, ${ Expr.ofList(rest.map(resolveOne(_, _))) }) }
    }

    // def const[A: Type: ToExpr](e: A) = summonTermRes[A](tc => '{ ${ tc }(Some(${ Expr(e) })) })
    // def select(name: String, tpe: TypeRepr, rest: List[(String, TypeRepr)]): Expr[Val] =
    //   summonTermRes[Value[Tpe]](tc => '{ ${ tc }(Some(${ selectValue(name, tpe, rest) })) })

    def liftTermToValue(term: Term): Expr[Value[Tpe]] = term match {
      case Typed(ap @ Apply(_, args), _) =>
        '{
          Value.CtorAp[Tpe](
            ${ summonTypeRes(ap.tpe.widenTermRefByName.dealias) }(),
            ${ Expr.ofList(args.map(liftTermToValue(_))) }
          )
        }
      case i @ Ident(x) =>
        if (i.symbol.owner == symbol) selectValue(x, i.tpe, Nil)
        else report.errorAndAbort(s"Ident ${i.show} not part of class ${symbol.fullName}")
      case s @ Select(qualifier, name) =>
        if (qualifier.symbol.owner == symbol) { // we're selecting from one of the ctor args
          selectValue(qualifier.symbol.name, qualifier.tpe, (name, s.tpe) :: Nil)
        } else { // some other constant
          '{ Value.CtorAp[Tpe](${ summonTypeRes(s.tpe.dealias) }(), Nil) }
        }
      // case Literal(BooleanConstant(x)) => const(x)
      // case Literal(ByteConstant(x))    => const(x)
      // case Literal(ShortConstant(x))   => const(x)
      // case Literal(IntConstant(x))     => const(x)
      // case Literal(LongConstant(x))    => const(x)
      // case Literal(FloatConstant(x))   => const(x)
      // case Literal(DoubleConstant(x))  => const(x)
      // case Literal(CharConstant(x))    => const(x)
      // case Literal(StringConstant(x))  => const(x)

      case x =>
        x.asExpr match {
          case '{ List($xs*) } => ??? // TODO
          case _               => report.errorAndAbort(s"Can't handle expr: ${term.asExpr.show}", term.asExpr)
        }
    }

    def matchCtor(t: Tree, parent: Option[TypeRepr]): Option[Expr[List[Val]]] = t match {
      case a @ Apply(Select(New(tpeTree), "<init>"), args) if parent.forall(tpeTree.tpe =:= _.widenTermRefByName) =>
        Some(
          Expr.ofList(args.map(arg => summonTermRes[Value[Tpe]](tc => '{ ${ tc }(Some(${ liftTermToValue(arg) })) })))
        )
      case _ => None
    }

    // Scala implements enum cases w/o params as vals in the companion with an anonymous cls
    (if (tpe.isSingleton) {
       class CtorTraverser extends TreeAccumulator[Option[Expr[List[Val]]]] {
         def foldTree(x: Option[Expr[List[Val]]], tree: Tree)(owner: Symbol): Option[Expr[List[Val]]] =
           x.orElse(matchCtor(tree, Some(tpe)).orElse(foldOverTree(None, tree)(symbol)))
       }
       CtorTraverser().foldOverTree(None, tpe.termSymbol.tree)(Symbol.noSymbol)
     } else {
       symbol.tree match {
         case ClassDef(name, _, headCtorApply :: _, _, _) => matchCtor(headCtorApply, None)
         case _                                           => report.errorAndAbort(s"Not a classdef")
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
    if (symbol.caseFields.nonEmpty)
      report.errorAndAbort(s"Sum type required, got product type (${symbol.name}) instead")
    else
      symbol.tree match {
        case ClassDef(_, ctor: DefDef, _, _, _) =>
          ctor match {
            case dd @ DefDef("<init>", head :: Nil, _, _) =>
              Expr.ofList(head.params.collect { case d @ ValDef(name, tpeTree, _) =>
                val shape = TypeRepr.of[TypeRes].appliedTo(tpeTree.tpe)
                val instance = Implicits.search(TypeRepr.of[TypeRes].appliedTo(tpeTree.tpe)) match {
                  case s: ImplicitSearchSuccess => s.tree
                  case _                        => report.errorAndAbort(s"Cannot find an instance of ${shape.show}")
                }
                '{ (${ Expr(name) }, ${ instance.asExpr }.asInstanceOf[TypeRes[Any]]()) }
              })
            case DefDef("<init>", args, _, _) =>
              report.errorAndAbort(s"Ctor has more than one parameter list (${args})")
          }
        case bad => report.errorAndAbort(s"Not a ClassDef symbol for ${symbol.name}: ${bad.show}")
      }

  }

}
