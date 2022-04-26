package polyregion.scala

import cats.syntax.all.*
import polyregion.ast.{PolyAst as p, *}

import scala.annotation.tailrec
import scala.util.Try

object Retyper {

  def lowerClassType[A: scala.quoted.Type](using q: Quoted): Result[p.StructDef] = lowerClassType(q.TypeRepr.of[A])

  def lowerClassType(using q: Quoted)(repr: q.TypeRepr): Result[p.StructDef] = lowerClassType0(repr.typeSymbol)

  def lowerClassType0(using q: Quoted)(tpeSym: q.Symbol): Result[p.StructDef] = {

    // if ((tpeSym.flags.is(q.Flags.Module) || tpeSym.flags.is(q.Flags.Abstract)) && tpeSym.fieldMembers.nonEmpty) {
    //   throw RuntimeException(
    //     s"Unsupported combination of flags: ${tpeSym.flags.show} for ${tpeSym}, fields=${tpeSym.fieldMembers}"
    //   )
    // }

    //  isTypeParam  &&  isAbstractType => class C[A]
    // !isTypeParam  &&  isAbstractType => class C{ type A }
    // !isTypeParam  && !isAbstractType => class C{ type A = Concrete }
//    if (tpeSym.typeMembers.exists(_.isTypeParam)) {
//      throw RuntimeException(
//        s"Encountered generic class ${tpeSym.fullName}, typeCtorArg=${tpeSym.typeMembers.map(t => s"$t(param=${t.isAbstractType} ${t.isType})")}"
//      )
//    }

    // XXX there appears to be a bug where an assertion error is thrown if we call the start/end (but not the pos itself)
    // of certain type of trees returned from fieldMembers
    tpeSym.fieldMembers
      .filter(_.maybeOwner == tpeSym) // TODO local members for now, need to workout inherited members
      .sortBy(_.pos.map(p => (p.startLine, p.startColumn))) // make sure the order follows source code decl. order
      .traverse(field =>
        (field.tree match {
          case d: q.ValDef =>
            typer0(d.tpt.tpe).flatMap { // TODO we need to work out nested structs
              case (_, t: p.Type) => p.Named(field.name, t).success
              case (_, bad)       => s"bad erased type $bad".fail
            }
          case _ => ???
        })
      )
      .map(p.StructDef(p.Sym(tpeSym.fullName), Nil, _))

  }

  def lowerPolymorphicClassType0(using q: Quoted)(tpeSym: q.Symbol, args: List[p.Type]): Result[p.StructDef] = {
    if ((tpeSym.flags.is(q.Flags.Module) || tpeSym.flags.is(q.Flags.Abstract)) && tpeSym.fieldMembers.nonEmpty) {
      throw RuntimeException(
        s"Unsupported combination of flags: ${tpeSym.flags.show} for ${tpeSym}, fields=${tpeSym.fieldMembers}"
      )
    }

    if (tpeSym.typeMembers.size != args.size) {
      throw RuntimeException(s"Bad monomorphic sizes")
    }

    // TODO make sure Symbol.typeMembers returned is in actual declaration order!
    val typeVarLut = tpeSym.typeMembers.zip(args).toMap // Symbol -> p.Type

    // XXX there appears to be a bug where an assertion error is thrown if we call the start/end (but not the pos itself)
    // of certain type of trees returned from fieldMembers
    tpeSym.fieldMembers
      .filter(_.maybeOwner == tpeSym) // TODO local members for now, need to workout inherited members
      // .sortBy(_.pos.map(p => (p.startLine, p.startColumn))) // make sure the order follows source code decl. order
      .traverse(field =>
        (field.tree match {
          case d: q.ValDef =>
            if (d.tpt.symbol.isTypeParam) {
              p.Named(field.name, typeVarLut(d.tpt.symbol)).success
            } else {
              typer0(d.tpt.tpe).flatMap { // TODO we need to work out nested structs
                case (_, t: p.Type) => p.Named(field.name, t).success
                case (_, bad)       => s"bad erased type $bad".fail
              }
            }
          case _ => ???
        })
      )
      .map(p.StructDef(p.Sym(s"${tpeSym.fullName}[${args.map(_.monomorphicName).mkString(",")}]"), Nil, _))

  }

  private def resolveClsFromSymbol(using q: Quoted)(clsSym: q.Symbol): Result[(p.Sym, q.Symbol, q.ClassKind)] =
    if (clsSym.isClassDef) {
      (
        p.Sym(clsSym.fullName),
        clsSym,
        if (clsSym.flags.is(q.Flags.Module)) q.ClassKind.Object else q.ClassKind.Class
      ).success
    } else {
      s"$clsSym is not a class def".fail
    }

  @tailrec private final def resolveClsFromTpeRepr(using
      q: Quoted
  )(r: q.TypeRepr): Result[(p.Sym, q.Symbol, q.ClassKind)] =
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

  private def liftClsToTpe(using q: Quoted)(sym: p.Sym, clsSym: q.Symbol, kind: q.ClassKind): p.Type =
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
      case (sym, q.ClassKind.Class)                                 =>
        // println("[retyper] witness: " + clsSym.tree.show)
        p.Type.Struct(sym, Nil)
      case (sym, q.ClassKind.Object) =>
        p.Type.Struct(sym, Nil)
      // q.ErasedClsTpe(sym, clsSym, q.ClassKind.Object, Nil)
      // ???
    }

  def clsSymTyper0(using q: Quoted)(clsSym: q.Symbol): Result[p.Type] =
    resolveClsFromSymbol(clsSym).map(liftClsToTpe(_, _, _))

  // def typerN(using
  //     q: Quoted
  // )(xs: List[q.TypeRepr] ): Result[List[(Option[p.Term], p.Type)]] = xs.traverse(typer(_))

  def typer0N(using q: Quoted)(repr: List[q.TypeRepr]): Result[List[(Option[p.Term], p.Type)]] = repr.traverse(typer0(_))

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
          (name, symbol, kind) <- resolveClsFromTpeRepr(tpeCtor)
          tpeCtorArgs          <- args.traverse(typer0(_))
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
            val tpeArgs = ctorArgs.map {
              case (_, t: p.Type) => t
              case _              => ???
            }

            (None, p.Type.Struct(name, tpeArgs))
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
        println(s"[fallthrough typer] ${expr} => ${expr.show} ${expr.getClass}")

        resolveClsFromTpeRepr(expr).map(liftClsToTpe(_, _, _)).map((None, _))
    }

  def typer1(using q: Quoted)(repr: q.TypeRepr): Result[(Option[p.Term], p.Type, q.FnDependencies)] =
    // def flattenCls(value: Option[p.Term], t: p.Type): Result[(Option[p.Term], p.Type, q.FnDependencies)] =
    //   (value, t) match {
    //     case (value, q.ErasedFnTpe(args, rtn)) =>
    //       flattenCls(None, rtn).map((_, tpe, dep) => (value, q.ErasedFnTpe(args, tpe), dep))

    //     case (value, e: q.ErasedClsTpe) if e.ctor.nonEmpty =>
    //       val s = p.Type.Struct(e.name, Nil)
    //       val m = lowerPolymorphicClassType0(
    //         e.symbol,
    //         e.ctor.map {
    //           case t: p.Type => t
    //           case bad       => ???
    //         }
    //       )
    //       m.map(sdef => (value, s, q.FnDependencies(clss = Map(sdef.name -> sdef))))
    //     case (value, s @ p.Type.Struct(sym, _)) =>
    //       Retyper
    //         .lowerClassType(repr)
    //         .map(sdef => (value, s, q.FnDependencies(clss = Map(sym -> sdef))))
    //     case (value, tpe) => (value, tpe, q.FnDependencies()).pure
    //   }

    //   Retyper.typer0(repr).flatMap((value, t) => flattenCls(value, t))
    Retyper.typer0(repr).map((term, tpe) => (term, tpe, q.FnDependencies()))

  def clsSymTyper1(using q: Quoted)(clsSym: q.Symbol): Result[(p.Type, q.FnDependencies)] =
    Retyper.clsSymTyper0(clsSym).flatMap {
      case s @ p.Type.Struct(sym, _) =>
        Retyper
          .lowerClassType0(clsSym)
          .map(sdef => (s, q.FnDependencies(Map(sym -> sdef))))

      case tpe => (tpe, q.FnDependencies()).pure
    }

  // extension (using q: Quoted)(c: q.FnContext) {

  //   def clsSymTyper(clsSym: q.Symbol): Result[(p.Type, q.FnContext)] = Retyper.clsSymTyper0(clsSym).flatMap {
  //     case s @ p.Type.Struct(sym, _) =>
  //       Retyper
  //         .lowerClassType0(clsSym)
  //         .map(sdef => (s, c.copy(clss = c.clss + (sym -> sdef))))

  //     case tpe => (tpe, c).pure
  //   }

  //   def typerN(xs: List[q.TypeRepr]): Result[(List[(Option[p.Term], p.Type)], q.FnContext)] = xs match {
  //     case Nil     => (Nil, c).pure
  //     case x :: xs =>
  //       // TODO make sure we get the right order back!
  //       c.typer(x).flatMap { (v, t, c) =>
  //         xs.foldLeftM(((v, t) :: Nil, c)) { case ((ys, c), x) =>
  //           c.typer(x).map((v, t, c) => (ys :+ (v, t), c))
  //         }
  //       }
  //   }

  //   def typer(repr: q.TypeRepr): Result[(Option[p.Term], p.Type, q.FnContext)] =
  //     Retyper.typer1(repr).map { case (value, t, deps) =>
  //       (value, t, c.copy(clss = c.clss ++ deps.clss)) // TODO use that map and not a full p.FnDependenccies
  //     // case (value, t, None)    => (value, t, c)
  //     }

  // }
}
