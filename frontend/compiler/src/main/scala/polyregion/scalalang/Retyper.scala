package polyregion.scalalang

import cats.kernel.Monoid
import cats.syntax.all.*
import polyregion.ast.{PolyAst as p, *}

import scala.annotation.tailrec
import scala.util.Try

object Retyper {

//   def lowerClassType[A: scala.quoted.Type](using q: Quoted): Result[p.StructDef] = lowerClassType(q.TypeRepr.of[A])

//   def lowerClassType(using q: Quoted)(repr: q.TypeRepr): Result[p.StructDef] = lowerClassType0(repr.typeSymbol)

//   def lowerClassType0(using q: Quoted)(tpeSym: q.Symbol): Result[p.StructDef] =
//     // if ((tpeSym.flags.is(q.Flags.Module) || tpeSym.flags.is(q.Flags.Abstract)) && tpeSym.fieldMembers.nonEmpty) {
//     //   throw RuntimeException(
//     //     s"Unsupported combination of flags: ${tpeSym.flags.show} for ${tpeSym}, fields=${tpeSym.fieldMembers}"
//     //   )
//     // }

// //    if (tpeSym.typeMembers.exists(_.isTypeParam)) {
// //      throw RuntimeException(
// //        s"Encountered generic class ${tpeSym.fullName}, typeCtorArg=${tpeSym.typeMembers.map(t => s"$t(param=${t.isAbstractType} ${t.isType})")}"
// //      )
// //    }

//     // XXX there appears to be a bug where an assertion error is thrown if we call the start/end (but not the pos itself)
//     // of certain type of trees returned from fieldMembers
//     tpeSym.fieldMembers
//       .filterNot(_.flags.is(q.Flags.Module))
//       .filter(_.maybeOwner == tpeSym) // TODO local members for now, need to workout inherited members
//       .sortBy(_.pos.map(p => (p.startLine, p.startColumn))) // make sure the order follows source code decl. order
//       .traverse(field =>
//         (field.tree match {
//           case d: q.ValDef =>
//             // TODO we need to work out nested structs
//             typer0(d.tpt.tpe).flatMap((_, t) => p.Named(field.name, t).success)
//           case _ => ???
//         })
//       )
//       .map(p.StructDef(p.Sym(tpeSym.fullName), Nil, _))

  def isModuleClass(using q: Quoted)(s: q.Symbol) = !s.isPackageDef && s.flags.is(q.Flags.Module)

  def structName0(using q: Quoted)(clsSym: q.Symbol): p.Sym = p.Sym(clsSym.fullName)

  private def deriveParents(using q: Quoted)(clsSym: q.Symbol): List[p.Sym] = {
    // Find out the parents
    val roots = Set(
      q.defn.AnyClass,
      q.defn.MatchableClass,
      q.defn.AnyValClass,
      q.defn.AnyRefClass,
      q.defn.ObjectClass,
      q.Symbol.requiredClass("java.io.Serializable"),
      q.Symbol.requiredClass("scala.reflect.ClassManifestDeprecatedApis"),
      q.Symbol.requiredClass("scala.reflect.OptManifest"),
      q.Symbol.requiredClass("scala.Equals")
    )
    clsSym.typeRef.baseClasses
      .drop(1)                      // head is the class itself
      .filterNot(roots.contains(_)) // don't want the roots
      .map(structName0(_))
  }

  def structDef0(using q: Quoted)(clsSym: q.Symbol): Result[p.StructDef] = {
    if (clsSym.flags.is(q.Flags.Abstract)) {
      throw RuntimeException(
        s"Unsupported combination of flags: ${clsSym.flags.show} for ${clsSym}, fields=${clsSym.fieldMembers}"
      )
    }

    // We extract all definitions in the class first.
    // XXX Calling `tree` on a field symbol throws a match error on DefDef, possibly a bug
    val definitions = q
      .collectTree(clsSym.tree) {
        case f: q.Definition => (f.symbol, f) :: Nil
        case _               => Nil
      }
      .toMap

    // XXX there appears to be a bug where an assertion error is thrown if we call the start/end (but not the pos itself)
    // of certain type of trees returned from fieldMembers
    clsSym.fieldMembers
      .filterNot(_.flags.is(q.Flags.Module)) // TODO exclude objects for now, need to implement this later
      .filter(_.maybeOwner == clsSym)        // TODO local members for now, need to workout inherited members
      .sortBy(
        _.pos.map(p => Try((p.startLine, p.startColumn)).getOrElse((0, 0)))
      ) // make sure the order follows source code decl. order
      .traverseFilter { field =>
        for {
          fieldTree <- definitions.get(field).failIfEmpty(s"Unexpected field ${field} from `Symbol.fieldMembers`")
          sm <- fieldTree match { // TODO we need to work out nested structs
            case d: q.DefDef => ???
            case d: q.ValDef =>
              val tpe = d.tpt.tpe
              // Skip references to objects as those are singletons and should be identified and lifted by the Remapper.
              if (isModuleClass(tpe.typeSymbol)) None.success
              else
                typer0(tpe).map { case (_ -> t, wit) =>
                  Some(
                    p.StructMember(
                      named = p.Named(field.name, t),
                      isMutable = d.symbol.flags.is(q.Flags.Mutable)
                    )
                  )
                }
            case _ => ???
          }
        } yield sm

      }
      .map(p.StructDef(p.Sym(clsSym.fullName), clsTypeCtorNames(clsSym), _, deriveParents(clsSym)))
  }

  // Ctor rules
  //  isTypeParam  &&  isAbstractType => class C[A]
  // !isTypeParam  &&  isAbstractType => class C{ type A }
  // !isTypeParam  && !isAbstractType => class C{ type A = Concrete }
  private def clsTypeCtorNames(using q: Quoted)(clsSym: q.Symbol): List[String] = clsSym.typeMembers
    .filter(t => t.isAbstractType && t.isTypeParam)
    .map(_.tree)
    .map {
      case q.TypeDef(name, _) => name
      case _                  => ???
    }

  private def resolveClsFromSymbol(using
      q: Quoted
  )(clsSym: q.Symbol): Result[(p.Sym, List[String], q.Symbol, q.ClassKind)] =
    if (clsSym.isClassDef) {
      (
        p.Sym(clsSym.fullName),
        clsTypeCtorNames(clsSym),
        clsSym,
        if (clsSym.flags.is(q.Flags.Module)) q.ClassKind.Object else q.ClassKind.Class
      ).success
    } else if (clsSym.isLocalDummy) {
      // Handle cases like :
      // class X{ { def a = ??? } }
      // Where the owner of `a` is a local dummy that is owned by class X
      resolveClsFromSymbol(clsSym.owner)
    } else {
      s"`$clsSym` is not a class def, the symbol is owned by `${clsSym.maybeOwner}`".fail
    }

  @tailrec private final def resolveClsFromTpeRepr(using
      q: Quoted
  )(r: q.TypeRepr): Result[(p.Sym, List[String], q.Symbol, q.ClassKind)] = {
    println("~ " + r)
    r.dealias.simplified match {
      case q.ThisType(tpe)                => resolveClsFromTpeRepr(tpe)
      case q.SuperType(thisTpe, superTpe) => resolveClsFromTpeRepr(superTpe)
      case tpe: q.NamedType =>
        tpe.classSymbol match {
          case None                              => s"Named type is not a class: ${tpe}".fail
          case Some(sym) if sym.name == "<root>" => resolveClsFromTpeRepr(tpe.qualifier) // discard root package
          case Some(sym)                         => resolveClsFromSymbol(sym)
        }
      case q.AppliedType(x, xs) => resolveClsFromTpeRepr(x)
      case invalid =>
        pprint.pprintln(invalid)
        s"Not a class TypeRepr: ${invalid} (repr=${Try(invalid.show)})".fail
    }
  }

  private def liftClsToTpe(using
      q: Quoted
  )(sym: p.Sym, tpeVars: List[String], clsSym: q.Symbol, kind: q.ClassKind): p.Type =
    (sym, kind) match {
      case (p.Sym(Symbols.Scala :+ "Unit"), q.ClassKind.Class)    => p.Type.Unit0
      case (p.Sym(Symbols.Scala :+ "Boolean"), q.ClassKind.Class) => p.Type.Bool1
      case (p.Sym(Symbols.Scala :+ "Byte"), q.ClassKind.Class)    => p.Type.IntS8
      case (p.Sym(Symbols.Scala :+ "Short"), q.ClassKind.Class)   => p.Type.IntS16
      case (p.Sym(Symbols.Scala :+ "Int"), q.ClassKind.Class)     => p.Type.IntS32
      case (p.Sym(Symbols.Scala :+ "Long"), q.ClassKind.Class)    => p.Type.IntS64
      case (p.Sym(Symbols.Scala :+ "Float"), q.ClassKind.Class)   => p.Type.Float32
      case (p.Sym(Symbols.Scala :+ "Double"), q.ClassKind.Class)  => p.Type.Float64
      case (p.Sym(Symbols.Scala :+ "Char"), q.ClassKind.Class)    => p.Type.IntU16
      // TODO type ctor args for now, need to work out type member refinements
      case (sym, q.ClassKind.Class | q.ClassKind.Object) =>
        p.Type.Struct(sym, tpeVars, tpeVars.map(p.Type.Var(_)), deriveParents(clsSym))
    }

  def clsSymTyper0(using q: Quoted)(clsSym: q.Symbol): Result[p.Type] =
    resolveClsFromSymbol(clsSym).map(liftClsToTpe(_, _, _, _))

  def typer0N(using q: Quoted)(repr: List[q.TypeRepr]): Result[(List[q.Retyped], q.ClsWitnesses)] =
    repr.foldMapM(typer0(_).map((t, c) => (t :: Nil, c)))

  def typer0(using q: Quoted)(repr: q.TypeRepr): Result[(q.Retyped, q.ClsWitnesses)] =
    (repr.dealias.widenTermRefByName.simplified match {
      case ref @ q.TypeRef(_, name) if ref.typeSymbol.isAbstractType => (None -> p.Type.Var(name), Map.empty).success
      case q.ParamRef(q.PolyType(args, _, _), argIdx)          => (None -> p.Type.Var(args(argIdx)), Map.empty).success
      case q.TypeBounds(_, _)                                  => (None -> p.Type.Nothing, Map.empty).success
      case q.PolyType(vars, _, method @ q.MethodType(_, _, _)) =>
        // this shows up from reference to generic methods
        typer0(method).flatMap {
          case (term -> (e @ p.Type.Exec(Nil, args, rtn)), wit) => (term -> p.Type.Exec(vars, args, rtn), wit).success
          case (_ -> bad, wit)                                  => ???
        }
      case poly @ q.PolyType(vars, _, tpe) =>
        // this shows up from reference to generic no-arg methods (i.e. getters)
        typer0(tpe).flatMap { case (term -> rtn, wit) =>
          (term -> p.Type.Exec(vars, Nil, rtn), wit).success
        }

      case m @ q.MethodType(argNames, argTpes, rtn) =>
        for {
          // If this method belongs to a class, the method type needs to prepend the receiver's tpe argTpes.
          // XXX Currently, the owner symbol of the method can be found only *before* widening from a term ref,
          //   so passing an already widened TypeRepr will end up with an NoDenotation error with the
          //   `repr.termSymbol` call
          (_ -> rtnTpe, wit0) <- typer0(rtn)
          (argTpes, wit1)     <- typer0N(argTpes) //  args.traverse(typer0(_))
          (receiverCtorTpes, wit2) <- repr match {
            case tr @ q.TermRef(_: q.NoPrefix, _) => // this is local function so no prefix
              for {
                classSymbol <- Remapper.owningClassSymbol(tr.termSymbol).failIfEmpty(s"No class for type ref ${tr}")
                tpe         <- clsSymTyper0(classSymbol)
              } yield (tpe :: Nil) -> Map.empty
            case q.TermRef(receiverTpe, _) => // we have something concrete
              typer0(receiverTpe).map {
                case (_ -> p.Type.Struct(_, _, args, _), wit) => args         -> wit
                case (_ -> p.Type.Ptr(arg, _), wit)           => (arg :: Nil) -> wit
                case (_ -> _, wit)                            => Nil          -> wit
              }
            case _ => (Nil, Map.empty).success
          }

          receiverCtorTpeVars = receiverCtorTpes.collect { case p.Type.Var(name) => name }

          // receiverTpes = clsTypeCtorNames(repr.termSymbol.maybeOwner) // XXX use the not yet widened `repr`, not `m`!
        } yield {
          println(s"Exec ${repr.termSymbol.maybeOwner}.${repr} ${receiverCtorTpes} => ${receiverCtorTpeVars}")
          (None -> p.Type.Exec(receiverCtorTpeVars, argTpes.map(_._2), rtnTpe), wit0 |+| wit1 |+| wit2)
        }
      case andOr: q.AndOrType =>
        for {
          (leftTerm -> leftTpe, leftWit)    <- typer0(andOr.left)
          (rightTerm -> rightTpe, rightWit) <- typer0(andOr.right)
          term = leftTerm.orElse(rightTerm)
          r <-
            if (leftTpe == rightTpe) (term -> leftTpe, leftWit |+| rightWit).success
//            else if(leftTpe == p.Type.Unit0 || rightTpe == p.Type.Unit0) (term, p.Type.Unit0).success
            else {
              val sym = andOr match {
                case _: q.OrType  => p.Sym("#Union")
                case _: q.AndType => p.Sym("#Intersection")
              }
              (term -> p.Type.Struct(sym, Nil, leftTpe :: rightTpe :: Nil, Nil), leftWit |+| rightWit).success
            }
          //            else s"Left type `$leftTpe` and right type `$rightTpe` did not unify for ${andOr.widenByName.widen.simplified.show}".fail
        } yield r
      case tpe @ q.AppliedType(tpeCtor, args) =>
        for {

          (name, tpeVars, symbol, kind) <- resolveClsFromTpeRepr(tpeCtor) // type ctors must be a class
          (tpeCtorArgs, wit)            <- typer0N(args)

          // _ = println(s"##### $name $kind $tpeCtorArgs ${tpe <:< q.TypeRepr.typeConstructorOf(classOf[scala.collection.mutable.IndexedSeq[_]]).appliedTo(args) }")

          argAppliedSeqLikeTpe = q.TypeRepr.typeConstructorOf(classOf[scala.collection.mutable.Seq[_]]).appliedTo(args)

          retyped <- (name, kind, tpeCtorArgs) match {
            case (Symbols.ArrayMirror, q.ClassKind.Class, (_, comp: p.Type) :: Nil) =>
              // case (Symbols.Array, q.ClassKind.Class, (_, comp: p.Type) :: Nil) =>
              (None -> p.Type.Ptr(comp, p.Type.Space.Global), wit).success
//            case (_, q.ClassKind.Class, (_, comp: p.Type) :: Nil) if tpe <:< argAppliedSeqLikeTpe =>
//              (None -> p.Type.Array(comp), wit).success
            case (_, _, ys) if tpe.isFunctionType => // FunctionN
              // TODO make sure this works
              ys.map(_._2) match {
                case Nil => ???
                case xs :+ rtn =>
                  (None -> p.Type.Exec(Nil, xs, rtn), wit).success
                case _ => ???
              }
            case (name, kind, ctorArgs) =>
              symbol.tree match {
                case clsDef: q.ClassDef =>
                  val appliedTpe: p.Type.Struct =
                    p.Type.Struct(name, tpeVars, ctorArgs.map(_._2), deriveParents(symbol))
                  (None -> appliedTpe, wit |+| Map(symbol -> Set(appliedTpe))).success
                case _ => s"$symbol is not a ClassDef".fail
              }
          }
        } yield retyped
      // widen singletons
      case q.ConstantType(x) =>
        x match {
          case q.BooleanConstant(v) => (Some(p.Term.Bool1Const(v)) -> p.Type.Bool1, Map.empty).success
          case q.ByteConstant(v)    => (Some(p.Term.IntS8Const(v)) -> p.Type.IntS8, Map.empty).success
          case q.ShortConstant(v)   => (Some(p.Term.IntS16Const(v)) -> p.Type.IntS16, Map.empty).success
          case q.IntConstant(v)     => (Some(p.Term.IntS32Const(v)) -> p.Type.IntS32, Map.empty).success
          case q.LongConstant(v)    => (Some(p.Term.IntS64Const(v)) -> p.Type.IntS64, Map.empty).success
          case q.FloatConstant(v)   => (Some(p.Term.Float32Const(v)) -> p.Type.Float32, Map.empty).success
          case q.DoubleConstant(v)  => (Some(p.Term.Float64Const(v)) -> p.Type.Float64, Map.empty).success
          case q.CharConstant(v)    => (Some(p.Term.IntU16Const(v)) -> p.Type.IntU16, Map.empty).success
          case q.StringConstant(v)  => ???
          case q.UnitConstant       => (Some(p.Term.Unit0Const) -> p.Type.Unit0, Map.empty).success
          case q.NullConstant       => ???
          case q.ClassOfConstant(cls) =>
            val reifiedTpe = q.TypeRepr.typeConstructorOf(classOf[Class[_]]).appliedTo(cls)
            typer0(reifiedTpe).map { case (_ -> tpe, wit) =>
              (Some(p.Term.Poison(tpe)) -> tpe, wit)
            }
        }
      case q.ParamRef(r, i) =>
        println(s"C => ${r} ${i}")
        ???
      case expr =>
        // println(s"[fallthrough typer] ${expr} => ${expr.show} ${expr.getClass}")
        resolveClsFromTpeRepr(expr).flatMap { (sym, tpeVars, symbol, kind) =>
          liftClsToTpe(sym, tpeVars, symbol, kind) match {
            case s @ p.Type.Struct(_, _, _, _) =>
              symbol.tree match {
                case clsDef: q.ClassDef => (None -> s, Map(symbol -> Set(s))).success
                case _                  => s"$symbol is not a ClassDef".fail
              }
            case tpe => (None -> tpe, Map.empty).success
          }

        }

    }).recoverWith { case e =>
      new CompilerException(s"Retyper failed while typing `${Try(repr.show)}`", e).asLeft
    }

}
