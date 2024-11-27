package polyregion.ast

import pprint.pprintln

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

  private inline def findPrimaryCtorParams[T: Type](using
      q: Quotes
  ): Option[(List[q.reflect.Term], List[q.reflect.Term])] = {
    import q.reflect.*
    val tpe = TypeRepr.of[T].dealias.simplified

    def matchCtor(t: Tree, parent: Option[TypeRepr]) = t match {
      case Apply(Select(New(tpeTree), "<init>"), args) //
          if parent.forall(tpeTree.tpe =:= _.widenTermRefByName) =>
        val primaryCtorParamListSize = tpeTree.tpe.typeSymbol.primaryConstructor.paramSymss.headOption.map(_.size)
        Some(primaryCtorParamListSize.fold(args -> Nil)(args.splitAt(_)))
      case Apply(Apply(Select(New(tpeTree), "<init>"), args0), args1)
          if parent.forall(tpeTree.tpe =:= _.widenTermRefByName) =>
        Some(args0 -> args1)
      case _ =>
        None
    }
    if (tpe.isSingleton) {
      val symbol = tpe.typeSymbol
      class CtorTraverser extends TreeAccumulator[Option[(List[Term], List[Term])]] {
        def foldTree(x: Option[(List[Term], List[Term])], tree: Tree)(owner: Symbol) =
          x.orElse(matchCtor(tree, Some(tpe)).orElse(foldOverTree(None, tree)(symbol)))
      }
      CtorTraverser().foldOverTree(None, tpe.termSymbol.tree)(Symbol.noSymbol)
    } else {
      tpe.typeSymbol.tree match {
        case ClassDef(name, _, ctors, _, _) => ctors.collectFirst(Function.unlift(matchCtor(_, None)))
        case _                              => report.errorAndAbort(s"Not a classdef")
      }
    }
  }

  inline def methods[T] = ${ methodsImpl[T] }
  def methodsImpl[T: Type](using quotes: Quotes) = {
    import quotes.reflect.*

    val tpe = TypeRepr.of[T].dealias.simplified

    println(s"${tpe.classSymbol}")

    def traceIt(depth: Int, tree: Tree): Unit = {
      println(s"[$depth]")

      pprintln(tree)
      if (depth > 100) ???
      tree match {

        case s @ Apply(term, args)  => traceIt(depth + 1, term)
        case s @ Select(term, name) => traceIt(depth + 1, term)
        case Typed(term, _)         => traceIt(depth + 1, term)
        case i @ This(x)            => traceIt(depth + 1, i)
        case i @ Ident(_) =>
          println(i.symbol.tree)
        case DefDef(name, args, _, impl) =>
          println(s"> $name $args")
          impl match {
            case Some(value) =>
              println(Expr.betaReduce(value.asExpr).show)

              traceIt(depth + 1, value)
            case None => ()
          }
        case x =>
          ???
          println(x)
      }
    }

    def work(arg1Vals: List[ValDef]) = findPrimaryCtorParams[T].fold(Expr(1)) { case (_, arg1) =>
      if (arg1Vals.size != arg1.size) {
        report.errorAndAbort(s"Base class arg list 1 size mismatch: base is $arg1Vals but sub is $arg1")
      }

      println("~~>" + arg1Vals)
      // arg1Vals.zip(arg1).map { (valDef, impl) =>

      //   println(valDef.name)
      //   println(impl.symbol.tree.show)
      //   // pprintln(impl.symbol.tree)

      //   impl.symbol.tree match {
      //     case DefDef(_, _, _, Some(Block(DefDef(_, _, _, Some(fn)) :: Nil, _))) =>
      //       println(fn)
      //       println(remap(fn))
      //     case _ => ???
      //   }

      // }
      Expr(1)
    }

    // tpe.baseClasses.drop(if (tpe.isSingleton) 0 else 1).headOption.map(_.tree) match {
    //   case Some(ClassDef(name, DefDef(_, _ :: TermParamClause(xs) :: Nil, _, _), _, _, _)) =>
    //     work(xs)
    //   case None => report.errorAndAbort(s"No base class found for $tpe")
    //   case Some(x) if x.symbol.typeRef.widenTermRefByName =:= TypeRepr.of[scala.reflect.Enum] => Expr(1)
    //   case Some(x) => report.errorAndAbort(s"Unexpected base class symbol ${x.show} for ${tpe.show}")
    // }

    val methods = tpe.classSymbol.toList.flatMap(_.declaredMethods).filterNot(_.flags.is(Flags.Private))
    println("~~ " + methods.map(_.tree).mkString("\n"))

    def retype(t: TypeRepr, const: Boolean = false): String = {
      val r = t.widen.widenTermRefByName
      val mapped = r.asType match {
        case '[String] => "std::string"
        case _         => s"${r.show}"
      }
      if (const) s"const ${mapped}&" else mapped
    }

    def remap(tree: Tree, fnDef: Boolean = false, scope: Map[Ident, String] = Map.empty): String = tree match {
      case i @ Ident(name)        => scope.getOrElse(i, name)
      case s @ Select(term, name) => s"${remap(term)}.$name${if (s.tpe.isSingleton) "()" else ""}"
      case Block(Nil, term)       => remap(term)
      case Block(stmts, term)     => s"${stmts.map(remap(_)).mkString("\n")}\n${remap(term)}"
      case ValDef(name, tpe, rhs) =>
        s"${retype(tpe.tpe, const = fnDef)} $name${rhs.map(x => s" = ${remap(x)}").getOrElse("")}"
      case Literal(StringConstant(s)) => s"\"$s\""

      case TypedOrTest(x, _)         => remap(x)
      case Typed(x, _)               => remap(x)
      case Unapply(_, _, pats)       => s"una(${pats.map(remap(_)).mkString(", ")})"
      case If(cond, trueBr, falseBr) => s"if (${remap(cond)}) { ${remap(trueBr)} } else { ${remap(falseBr)} }"

      case Match(term, cases) =>
        // auto _value = _x.get<Type>();

        val mapped = cases.map {
          case CaseDef(TypedOrTest(Unapply(_, _, pats), tpeTree), None, rhs) =>
            val bindTable = pats.collect { case Bind(name, ident) =>
              ident -> s"_y->${name}"
            }.toMap
            s"""if (auto _y = _x.get<${retype(tpeTree.tpe)}>()) { 
                |${remap(rhs)}
                |}""".stripMargin
          case CaseDef(s: Select, None, rhs) => s"if (_x.is<${retype(s.tpe)}>()) { return ${remap(rhs)}; }"
          case x                             => s"/* unhandled case ${x.show} */"
        }
        s"""|[&, _x = ${remap(term)}](){
            |${mapped.map(_.indent(2)).mkString}
            |  throw std::logic_error("Unhandled match case for ${remap(term)} (of type ${retype(
             term.tpe
           )}) at " __FILE__ ":" __LINE__);
            |}();""".stripMargin

      case x =>
        // pprintln(x)

        Expr.betaReduce(x.asExpr) match {
          case '{ StringContext(${Varargs(Exprs(elems))}*) } => s"CANNOT CS ${elems}"
          case '{ ($x : StringContext).s($s)  } => s"CANNOT S ${x.asTerm} ap ${s.asTerm}"
          case _ =>
            s"/* failed ${x}*/"
        }

    }

    methods.foreach { s =>
      s.tree match {
        case f @ DefDef(name, _, _, Some(rhs)) =>
          println(">>>" + name)

          val prog =
            s"""${retype(f.returnTpt.tpe)} $name(${f.termParamss
                .flatMap(_.params)
                .map(p => remap(tree = p, fnDef = true))
                .mkString(", ")}) {  
               |${remap(rhs).indent(2)}}
               |""".stripMargin

          println(prog)

        case unknown => ???
      }

    // pprintln(s.tree)
    }

    // def remapOne(x : Tree) = {
    //   x match {
    //     case _: AnyRef =>

    //   }
    // }

    Expr(1)
  }

  inline def primaryCtorApplyTerms[
      T,
      Val,
      TermRes[x] <: TermResolver[x, Val],
      Tpe,
      TypeRes[_] <: TypeResolver[Tpe]
  ] = ${ primaryCtorApplyTermsImpl[T, Val, TermRes, Tpe, TypeRes] }
  def primaryCtorApplyTermsImpl[
      T: Type,
      Val: Type,
      TermRes[x] <: TermResolver[x, Val]: Type,
      Tpe: Type,
      TypeRes[_] <: TypeResolver[Tpe]: Type
  ](using quotes: Quotes): Expr[List[Val]] = {
    import quotes.reflect.*

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

    val tpe    = TypeRepr.of[T].dealias.simplified
    val symbol = tpe.typeSymbol

    def liftTermToValue(term: Term): Expr[Value[Tpe]] = term match {
      case Typed(x, _)                                                                         => liftTermToValue(x)
      case TypeApply(x, _)                                                                     => liftTermToValue(x)
      case Apply(TypeApply(Select(Ident("List"), "apply"), _), x :: Nil)                       => liftTermToValue(x)
      case Apply(Select(Apply(Ident("List"), List()), "apply"), x :: Nil)                      => liftTermToValue(x)
      case Apply(Select(Ident("ScalaRunTime"), "wrapRefArray" | "genericWrapArray"), x :: Nil) => liftTermToValue(x)
      case Apply(x, Nil)                                                                       => liftTermToValue(x)
      case Repeated(terms, tpt) =>
        '{
          lazy val _x = Value.CtorAp[Tpe](
            ${ summonTypeRes(TypeRepr.of[List].appliedTo(tpt.tpe).widenTermRefByName.dealias) }(),
            ${ Expr.ofList(terms.map(liftTermToValue(_))) }
          )
          _x
        }
      case ap @ Apply(x, args) =>
        '{
          Value.CtorAp[Tpe](
            ${ summonTypeRes(ap.tpe.widenTermRefByName.dealias) }(),
            ${ Expr.ofList(args.map(liftTermToValue(_))) }
          )
        }
      case i @ Ident(x) =>
        if (i.symbol.owner == symbol) selectValue(x, i.tpe, Nil)
        else {
          i.symbol.tree match {
            case ValDef(_, _, Some(x)) if !i.symbol.flags.is(Flags.Case) => liftTermToValue(x)
            case DefDef(_, Nil :: Nil, _, Some(x))                       => liftTermToValue(x)
            case x =>
              '{ Value.CtorAp[Tpe](${ summonTypeRes(i.tpe.dealias) }(), Nil) }
            // report.errorAndAbort(
            //   s"Ident ${i}:${i.tpe.widenTermRefByName.show} with tree${x} doesn't refer to a term tree"
            // )
          }
        }
      case s @ Select(qualifier, "asInstanceOf") => liftTermToValue(qualifier)
      case s @ Select(qualifier, name) =>
        if (qualifier.symbol.owner == symbol) { // we're selecting from one of the ctor args
          selectValue(qualifier.symbol.name, qualifier.tpe, (name, s.tpe) :: Nil)
        } else { // some other constant
          s.symbol.tree match {
            case ValDef(_, _, Some(x)) if !s.symbol.flags.is(Flags.Case) => liftTermToValue(x)
            case DefDef(_, Nil :: Nil | Nil, _, Some(x))                 => liftTermToValue(x)
            case x =>
              '{ Value.CtorAp[Tpe](${ summonTypeRes(s.tpe.dealias) }(), Nil) }
          }
        }
      case x =>
        pprint.pprintln(term)
        report.errorAndAbort(s"Can't handle expr: ${term}\n${term.show}")
    }

    findPrimaryCtorParams[T].fold(Expr(Nil)) { case (primaryCtor, _) =>
      Expr.ofList(
        primaryCtor.map(arg => summonTermRes[Value[Tpe]](tc => '{ ${ tc }(Some(${ liftTermToValue(arg) })) }))
      )
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
            case dd @ DefDef("<init>", head :: _, _, _) =>
              Expr.ofList(head.params.collect { case d @ ValDef(name, tpeTree, _) =>
                val shape = TypeRepr.of[TypeRes].appliedTo(tpeTree.tpe)
                val instance = Implicits.search(TypeRepr.of[TypeRes].appliedTo(tpeTree.tpe)) match {
                  case s: ImplicitSearchSuccess => s.tree
                  case _                        => report.errorAndAbort(s"Cannot find an instance of ${shape.show}")
                }
                '{ (${ Expr(name) }, ${ instance.asExpr }.asInstanceOf[TypeRes[Any]]()) }
              })

          }
        case bad => report.errorAndAbort(s"Not a ClassDef symbol for ${symbol.name}: ${bad.show}")
      }

  }

}
