package polyregion.scala

import cats.syntax.all.*
import polyregion.ast.{PolyAst as p, *}

import scala.annotation.tailrec

object Retyper {

  def lowerClassType[A: scala.quoted.Type](using q: Quoted): Deferred[p.StructDef] = lowerClassType(q.TypeRepr.of[A])

  def lowerClassType(using q: Quoted)(repr : q.TypeRepr): Deferred[p.StructDef] = lowerClassType0(repr.typeSymbol)

  def lowerClassType0(using q: Quoted)(tpeSym: q.Symbol): Deferred[p.StructDef] = {

    if ((tpeSym.flags.is(q.Flags.Module) || tpeSym.flags.is(q.Flags.Abstract)) && tpeSym.fieldMembers.nonEmpty) {
      throw RuntimeException(
        s"Unsupported combination of flags: ${tpeSym.flags.show} for ${tpeSym}, fields=${tpeSym.fieldMembers}"
      )
    }

    // println(s"Decls = ${tpeSym.declarations.map(_.tree.show)}")

    // TODO workout inherited members
    tpeSym.fieldMembers
      .sortBy(_.pos.map(p => (p.startLine, p.startColumn))) // make sure the order follows source code decl. order
      .traverse(field =>
        (field.tree match {
          case d: q.ValDef =>
            typer0(d.tpt.tpe).flatMap { // TODO we need to work out nested structs
              case (_, t: p.Type ) => p.Named(field.name, t).success
              case (_, bad )       => s"bad erased type $bad".fail
            }
          case _ => ???
        })
      )
      .map(p.StructDef(p.Sym(tpeSym.fullName), _))
      .deferred
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
      case invalid => s"Not a class TypeRepr: ${invalid}".fail
    }

  private def liftClsToTpe(using q: Quoted)(sym: p.Sym, clsSym: q.Symbol, kind: q.ClassKind): q.Tpe =
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
        p.Type.Struct(sym)
      case (sym, q.ClassKind.Object) =>
        q.ErasedClsTpe(sym, q.ClassKind.Object, Nil)
    }

  def clsSymTyper0(using q: Quoted)(clsSym: q.Symbol): Result[q.Tpe] =
    resolveClsFromSymbol(clsSym).map(liftClsToTpe(_, _, _))

  // def typerN(using
  //     q: Quoted
  // )(xs: List[q.TypeRepr] ): Deferred[List[(Option[p.Term], q.Tpe)]] = xs.traverse(typer(_))

  def typer0(using q: Quoted)(repr: q.TypeRepr): Result[(Option[p.Term], q.Tpe)] =
    repr.dealias.widenTermRefByName.simplified match {
      case p @ q.PolyType(_, _, q.MethodType(_, _, _)) =>
        // this shows up from type-unapplied methods:  [x, y] =>> methodTpe(_:x, ...):y
        //  (None, q.ErasedOpaqueTpe(p), c).success.deferred
        ???
      case m @ q.MethodType(names, args, rtn) =>
        for {
          (_, tpe) <- typer0(rtn)
          argTpes  <- args.traverse(typer0(_))
        } yield (None, q.ErasedFnTpe(names.zip(argTpes.map(_._2)), tpe))
      case andOr: q.AndOrType =>
        for {
          (leftTerm, leftTpe)   <- typer0(andOr.left)
          (rightTerm, rightTpe) <- typer0(andOr.right)
        } yield
          if leftTpe == rightTpe then (leftTerm.orElse(rightTerm), leftTpe)
          else ???
      case tpe @ q.AppliedType(tpeCtor, args) =>
        for {
          // type ctors must be a class
          (name, _, kind) <- resolveClsFromTpeRepr(tpeCtor)
          tpeCtorArgs     <- args.traverse(typer0(_))
        } yield (name, kind, tpeCtorArgs) match {
          case (Symbols.Buffer, q.ClassKind.Class, (_, comp: p.Type) :: Nil) => (None, p.Type.Array(comp))
          case (Symbols.Array, q.ClassKind.Class, (_, comp: p.Type) :: Nil)  => (None, p.Type.Array(comp))
          case (_, _, ys) if tpe.isFunctionType                              => // FunctionN
            // TODO make sure this works
            (
              None,
              ys.map(_._2) match {
                case Nil      => ???
                case x :: Nil => q.ErasedFnTpe(Nil, x)
                case xs :+ x  => ???
//                TODO fn types don't have named args
//                  q.ErasedFnTpe(xs, x)
              }
            )
          case (n, m, ys) =>
            // lowerClassType(ctor.classSymbol.get).map( s =>  )
            (None, q.ErasedClsTpe(n, m, ys.map(_._2)))
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
      case expr => resolveClsFromTpeRepr(expr).map(liftClsToTpe(_, _, _)).map((None, _))
    }

  extension (using q: Quoted)(c: q.FnContext) {

    def clsSymTyper(clsSym: q.Symbol): Result[(q.Tpe, q.FnContext)] = Retyper.clsSymTyper0(clsSym).flatMap {
      case s @ p.Type.Struct(sym) =>
        Retyper
          .lowerClassType0(clsSym)
          .map(sdef => (s, c.copy(clss = c.clss + (sym -> sdef))))
          .resolve
      case tpe => (tpe, c).pure
    }

    def typerN(xs: List[q.TypeRepr]): Deferred[(List[(Option[p.Term], q.Tpe)], q.FnContext)] = xs match {
      case Nil     => (Nil, c).pure
      case x :: xs =>
        // TODO make sure we get the right order back!
        c.typer(x).flatMap { (v, t, c) =>
          xs.foldLeftM(((v, t) :: Nil, c)) { case ((ys, c), x) =>
            c.typer(x).map((v, t, c) => (ys :+ (v, t), c))
          }
        }
    }

    def typer(repr: q.TypeRepr): Deferred[(Option[p.Term], q.Tpe, q.FnContext)] =
      Retyper.typer0(repr).deferred.flatMap {
        case (value, s @ p.Type.Struct(sym)) =>
          Retyper
            .lowerClassType(repr)
            .map(sdef => (value, s, c.copy(clss = c.clss + (sym -> sdef))))
        case (value, tpe) => (value, tpe, c).pure
      }

  }
}
