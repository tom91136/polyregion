package polyregion.scala

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

  def structDef0(using q: Quoted)(clsSym: q.Symbol): Result[p.StructDef] = {
    if (clsSym.flags.is(q.Flags.Abstract)) {
      throw RuntimeException(
        s"Unsupported combination of flags: ${clsSym.flags.show} for ${clsSym}, fields=${clsSym.fieldMembers}"
      )
    }

    // XXX there appears to be a bug where an assertion error is thrown if we call the start/end (but not the pos itself)
    // of certain type of trees returned from fieldMembers
    clsSym.fieldMembers
      .filterNot(_.flags.is(q.Flags.Module)) // TODO exclude objects for now, need to implement this later
      .filter(_.maybeOwner == clsSym)        // TODO local members for now, need to workout inherited members
      .sortBy(
        _.pos.map(p => Try((p.startLine, p.startColumn)).getOrElse((0, 0)))
      ) // make sure the order follows source code decl. order
      .traverseFilter(field =>
        (field.tree match { // TODO we need to work out nested structs
          case d: q.ValDef =>
            val tpe = d.tpt.tpe
            // Skip references to objects as those are singletons and should be identified and lifted by the Remapper.
            if (isModuleClass(tpe.typeSymbol)) {
              None.success
            } else typer0(tpe).map((_, t) => p.Named(field.name, t)).map(Some(_))
          case _ => ???
        })
      )
      .map(p.StructDef(p.Sym(clsSym.fullName), clsTypeCtorNames(clsSym), _))
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
    } else {
      s"$clsSym is not a class def".fail
    }

  @tailrec private final def resolveClsFromTpeRepr(using
      q: Quoted
  )(r: q.TypeRepr): Result[(p.Sym, List[String], q.Symbol, q.ClassKind)] =
    r.dealias.simplified match {
      case q.ThisType(tpe) => resolveClsFromTpeRepr(tpe)
      case tpe: q.NamedType =>
        tpe.classSymbol match {
          case None                              => s"Named type is not a class: ${tpe}".fail
          case Some(sym) if sym.name == "<root>" => resolveClsFromTpeRepr(tpe.qualifier) // discard root package
          case Some(sym)                         => resolveClsFromSymbol(sym)
        }
      case invalid => s"Not a class TypeRepr: ${invalid.show}".fail
    }

  private def liftClsToTpe(using
      q: Quoted
  )(sym: p.Sym, tpeVars: List[String], clsSym: q.Symbol, kind: q.ClassKind): p.Type =
    (sym, kind) match {
      case (p.Sym(Symbols.Scala :+ "Unit"), q.ClassKind.Class)      => p.Type.Unit
      case (p.Sym(Symbols.Scala :+ "Boolean"), q.ClassKind.Class)   => p.Type.Bool
      case (p.Sym(Symbols.Scala :+ "Byte"), q.ClassKind.Class)      => p.Type.Byte
      case (p.Sym(Symbols.Scala :+ "Short"), q.ClassKind.Class)     => p.Type.Short
      case (p.Sym(Symbols.Scala :+ "Int"), q.ClassKind.Class)       => p.Type.Int
      case (p.Sym(Symbols.Scala :+ "Long"), q.ClassKind.Class)      => p.Type.Long
      case (p.Sym(Symbols.Scala :+ "Float"), q.ClassKind.Class)     => p.Type.Float
      case (p.Sym(Symbols.Scala :+ "Double"), q.ClassKind.Class)    => p.Type.Double
      case (p.Sym(Symbols.Scala :+ "Char"), q.ClassKind.Class)      => p.Type.Char
      case (p.Sym(Symbols.JavaLang :+ "String"), q.ClassKind.Class) => p.Type.String
      // TODO type ctor args for now, need to work out type member refinements
      case (sym, q.ClassKind.Class | q.ClassKind.Object) => p.Type.Struct(sym, tpeVars, tpeVars.map(p.Type.Var(_)))
    }

  def clsSymTyper0(using q: Quoted)(clsSym: q.Symbol): Result[p.Type] =
    resolveClsFromSymbol(clsSym).map(liftClsToTpe(_, _, _, _))

  def typer0N(using q: Quoted)(repr: List[q.TypeRepr]): Result[List[(Option[p.Term], p.Type)]] =
    repr.traverse(typer0(_))

  def typer0(using q: Quoted)(repr: q.TypeRepr): Result[(Option[p.Term], p.Type)] =
    repr.dealias.widenTermRefByName.simplified match {
      case ref @ q.TypeRef(_, name) if ref.typeSymbol.isAbstractType => (None, p.Type.Var(name)).success
      case q.ParamRef(q.PolyType(args, _, _), argIdx)                => (None, p.Type.Var(args(argIdx))).success
      case q.TypeBounds(_, _)                                        => (None, p.Type.Nothing).success
      case q.PolyType(vars, _, method @ q.MethodType(_, _, _))       =>
        // this shows up from reference to generic methods
        typer0(method).flatMap {
          case (term, e @ p.Type.Exec(Nil, args, rtn)) => (term, p.Type.Exec(vars, args, rtn)).success
          case (_, bad)                                => ???
        }
      case poly @ q.PolyType(vars, _, tpe) =>
        // this shows up from reference to generic no-arg methods (i.e. getters)
        typer0(tpe).flatMap { (term, rtn) =>
          (term, p.Type.Exec(vars, Nil, rtn)).success
        }

      case m @ q.MethodType(names, args, rtn) =>
        for {
          (_, rtnTpe) <- typer0(rtn)
          argTpes     <- args.traverse(typer0(_))
        } yield (None, p.Type.Exec(Nil, argTpes.map(_._2), rtnTpe))
      case andOr: q.AndOrType =>
        for {
          (leftTerm, leftTpe)   <- typer0(andOr.left)
          (rightTerm, rightTpe) <- typer0(andOr.right)
          term = leftTerm.orElse(rightTerm)
          r <-
            if (leftTpe == rightTpe) (term, leftTpe).success
//            else if(leftTpe == p.Type.Unit || rightTpe == p.Type.Unit) (term, p.Type.Unit).success
            else s"Left type `$leftTpe` and right type `$rightTpe` did not unify for ${andOr.simplified.show}".fail
        } yield r
      case tpe @ q.AppliedType(tpeCtor, args) =>
        for {
          // type ctors must be a class
          (name, tpeVars, symbol, kind) <- resolveClsFromTpeRepr(tpeCtor)
          tpeCtorArgs                   <- args.traverse(typer0(_))
        } yield (name, kind, tpeCtorArgs) match {
          case (Symbols.Buffer, q.ClassKind.Class, (_, comp: p.Type) :: Nil) => (None, p.Type.Array(comp))
          case (Symbols.Array, q.ClassKind.Class, (_, comp: p.Type) :: Nil)  => (None, p.Type.Array(comp))
          case (_, _, ys) if tpe.isFunctionType                              => // FunctionN
            // TODO make sure this works
            (
              None,

//               ys.map(_._2) match {
//                 case Nil      => ???
//                 case x :: Nil => q.ErasedFnTpe(Nil, x)
//                 case xs :+ x  => ???
// //                TODO fn types don't have named args
// //                  q.ErasedFnTpe(xs, x)
//               }
              ???
            )
          case (name, kind, ctorArgs) =>
            // p.Type.Char
            (None, p.Type.Struct(name, tpeVars, ctorArgs.map(_._2)))
          // (None, q.ErasedClsTpe(name, symbol, kind, ctorArgs.map(_._2)))
        }
      // widen singletons
      case q.ConstantType(x) =>
        (x match {
          case q.BooleanConstant(v) => (Some(p.Term.BoolConst(v)), p.Type.Bool)
          case q.ByteConstant(v)    => (Some(p.Term.ByteConst(v)), p.Type.Byte)
          case q.ShortConstant(v)   => (Some(p.Term.ShortConst(v)), p.Type.Short)
          case q.IntConstant(v)     => (Some(p.Term.IntConst(v)), p.Type.Int)
          case q.LongConstant(v)    => (Some(p.Term.LongConst(v)), p.Type.Long)
          case q.FloatConstant(v)   => (Some(p.Term.FloatConst(v)), p.Type.Float)
          case q.DoubleConstant(v)  => (Some(p.Term.DoubleConst(v)), p.Type.Double)
          case q.CharConstant(v)    => (Some(p.Term.CharConst(v)), p.Type.Char)
          case q.StringConstant(v)  => ???
          case q.UnitConstant       => (Some(p.Term.UnitConst), p.Type.Unit)
          case q.NullConstant       => ???
          case q.ClassOfConstant(r) => ???
        }).pure
      case q.ParamRef(r, i) =>
        println(s"C => ${r} ${i}")
        ???
      case expr =>
        // println(s"[fallthrough typer] ${expr} => ${expr.show} ${expr.getClass}")

        resolveClsFromTpeRepr(expr).map(liftClsToTpe(_, _, _, _)).map((None, _))
    }

}
