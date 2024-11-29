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

  inline def generateReprSource[T] = ${ generateReprSourceImpl[T] }
  def generateReprSourceImpl[T: Type](using quotes: Quotes) = {
    import quotes.reflect.*
    val tpe = TypeRepr.of[T].dealias.simplified

    def normalise(s: String) = s.replaceAll("\\$", "_")
    def retype(t: TypeRepr, const: Boolean = false): String = {
      val r = t.dealias.simplified

      val mapped = r.asType match {
        case '[String] => "std::string"
        case '[Int]    => "int32_t"
        case _ =>
          val fqcnTail = List
            .unfold(r.typeSymbol)(s => if (s.isNoSymbol) None else Some((s, s.maybeOwner)))
            .takeWhile(x => x != tpe.typeSymbol)
            .reverse

          val (_, cppName) = fqcnTail
            .map(s => s.flags.is(Flags.Module) -> s.name.replace("$", ""))
            .foldLeft((true, "")) {
              case ((true, acc), (m, x))  => m -> s"$acc$x"
              case ((false, acc), (m, x)) => m -> s"$acc.$x"
            }

          val isBase    = r.widen == r
          val singleton = r.typeSymbol.caseFields.isEmpty

          if (isBase && singleton) s"$cppName::Any" // is base of an enum
          else if (singleton) s"${cppName}::${r.termSymbol.name}"
          else s"${fqcnTail.map(_.name.replace("$", "")).mkString("::")}"

      }
      if (const) s"const ${mapped}&" else mapped
    }

    def remap(tree: Tree, scope: Map[String, String], depth: Int, fnDef: Boolean = false): String = tree match {
      case i @ Ident(name)             => scope.getOrElse(name, normalise(name))
      case s @ Select(term, name)      => s"${remap(term, scope, depth + 1)}.${normalise(name)}"
      case Block(Nil, term)            => remap(term, scope, depth + 1)
      case Block(stmts, Closure(_, _)) => s"${stmts.map(remap(_, scope, depth + 1)).mkString("\n")}"
      case Block(stmts, term) =>
        s"${stmts.map(remap(_, scope, depth + 1)).mkString("\n")}\n${remap(term, scope, depth + 1)}"
      case ValDef(name, tpe, rhs) =>
        s"${retype(tpe.tpe, const = fnDef)} ${normalise(name)}${rhs.map(x => s" = ${remap(x, scope, depth + 1)}").getOrElse("")}"
      case Literal(StringConstant(s)) => s"\"${s.replaceAll("\n", "\\\\n")}\"s"
      case Literal(IntConstant(i))    => s"$i"
      case TypedOrTest(x, _)          => remap(x, scope, depth + 1)
      case Typed(x, _)                => remap(x, scope, depth + 1)
      case If(cond, trueBr, falseBr) =>
        s"if (${remap(cond, scope, depth + 1)}) { ${remap(trueBr, scope, depth + 1)} } else { ${remap(falseBr, scope, depth + 1)} }"
      case f @ DefDef("$anonfun", _, _, Some(rhs)) =>
        val args = f.termParamss.flatMap(_.params).map(p => remap(p, scope, depth + 1, fnDef = true)).mkString(", ")
        s"[&]($args){ return ${remap(rhs, scope, depth + 1)}; }"
      case Match(term, cases) =>
        val names       = List("_x", "_y", "_z")
        val nameAtDepth = names((names.size - 1) % depth)
        val expr        = remap(term, scope, depth + 1)

        val mapped = cases.map {
          case CaseDef(TypedOrTest(Unapply(_, _, pats), tpeTree), None, rhs) if tpeTree.symbol.flags.is(Flags.Case) =>
            val bindScopes = tpeTree.symbol.caseFields
              .zip(pats)
              .collect { case (caseVal, Bind(name, _)) => name -> s"$nameAtDepth->${caseVal.name}" }
              .toMap
            s"if (auto $nameAtDepth = $expr.get<${retype(tpeTree.tpe)}>()) {\n  return ${remap(rhs, scope = bindScopes, depth + 1)};\n}"
          case CaseDef(s: Select, None, rhs) =>
            s"if ($expr.is<${retype(s.tpe)}>()) { return ${remap(rhs, scope, depth + 1)}; }"
          case x => s"/* unhandled case ${x.show} */"
        }

        s"""|[&]{
            |${mapped.map(_.indent(2)).mkString}
            |  throw std::logic_error(fmt::format("Unhandled match case for $expr (of type ${retype(
             term.tpe.widenTermRefByName
           )}) at {}:{})", __FILE__, __LINE__));
            |}()""".stripMargin
      case Apply(Ident("repr"), x :: Nil) => s"repr(${remap(x, scope, depth + 1)})"
      case x =>
        Expr.betaReduce(x.asExpr) match {
          case '{ ($x: Any).toString() } => s"std::to_string(${remap(x.asTerm, scope, depth + 1)})"
          case '{ ($x: Option[t]).getOrElse($v) } =>
            s"${remap(x.asTerm, scope, depth + 1)} ^ get_or_else(${remap(v.asTerm, scope, depth + 1)})"
          case '{ ($x: Option[t]).map($f) } =>
            s"${remap(x.asTerm, scope, depth + 1)} ^ map(${remap(f.asTerm, scope, depth + 1)})"
          case '{ ($init: List[t]) :+ ($last) } =>
            s"${remap(init.asTerm, scope, depth + 1)} | append(${remap(last.asTerm, scope, depth + 1)})"
          case '{ ($x: Iterator[t]).map($f) } =>
            s"${remap(x.asTerm, scope, depth + 1)} | map(${remap(f.asTerm, scope, depth + 1)})"
          case '{ ($x: Set[t]).map($f) } =>
            s"${remap(x.asTerm, scope, depth + 1)} | map(${remap(f.asTerm, scope, depth + 1)})"
          case '{ ($x: List[t]).map($f) } =>
            s"${remap(x.asTerm, scope, depth + 1)} | map(${remap(f.asTerm, scope, depth + 1)})"
          case '{ ($x: List[t]).reduceLeftOption($f) } =>
            s"(${remap(x.asTerm, scope, depth + 1)} | reduce(${remap(f.asTerm, scope, depth + 1)}))"
          case '{ ($x: List[t]).mkString($sep) } =>
            s"(${remap(x.asTerm, scope, depth + 1)} | mk_string(${remap(sep.asTerm, scope, depth + 1)}))"
          case '{ ($x: Set[t]).mkString($sep) } =>
            s"(${remap(x.asTerm, scope, depth + 1)} | mk_string(${remap(sep.asTerm, scope, depth + 1)}))"
          case '{ ($x: String).indent($n) } =>
            s"${remap(x.asTerm, scope, depth + 1)} ^ indent(${remap(n.asTerm, scope, depth + 1)})"
          case '{ wrapString($x: String) } =>
            s"${remap(x.asTerm, scope, depth + 1)}"
          case '{ StringContext.apply(${ Varargs(Exprs(elems)) }*).s(${ Varargs(args) }*) } =>
            if (elems.size == 1) return s"\"${elems(0)}\"s"
            else
              s"fmt::format(\"${elems.mkString("{}")}\", ${args.map(_.asTerm).map(remap(_, scope, depth + 1)).mkString(", ")})"
          case _ => s"/*failed ${x}*/"
        }
    }

    val methods = tpe.classSymbol.toList.flatMap(_.declaredMethods).filterNot(_.flags.is(Flags.Private))

    val (protos, impls) = methods
      .filterNot(_.isNoSymbol)
      .map(_.tree)
      .collect { case f @ DefDef(name, _, _, Some(rhs)) =>
        val args  = f.termParamss.flatMap(_.params).map(p => remap(p, Map.empty, 1, fnDef = true)).mkString(", ")
        val proto = s"${retype(f.returnTpt.tpe)} $name($args)"
        val impl =
          s"""
               |$proto {  
               |${("return " + remap(rhs, Map.empty, 1) + ";").indent(2)}} 
               |""".stripMargin
        s"[[nodiscard]] $proto;" -> impl
      }
      .unzip

    Expr((protos.mkString("\n"), impls.mkString("\n")))
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
