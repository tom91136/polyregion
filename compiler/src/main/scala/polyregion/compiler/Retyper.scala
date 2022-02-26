package polyregion.compiler

import cats.syntax.all.*
import polyregion.ast.PolyAst as p
import polyregion.*

import scala.annotation.tailrec
import scala.quoted.*
import simulacrum.typeclass

object Retyper {

  def lowerClassType[A: Type](using q: Quoted): Deferred[p.StructDef] = lowerClassType(q.TypeRepr.of[A].typeSymbol)
  def lowerClassType(using q: Quoted)(tpeSym: q.Symbol): Deferred[p.StructDef] = {

    if (tpeSym.flags.is(q.Flags.Module) || tpeSym.flags.is(q.Flags.Abstract)) {
      throw RuntimeException(s"Unsupported combination of flags: ${tpeSym.flags.show} for ${tpeSym}")
    }

    println(s"Decls = ${tpeSym.declarations.map(_.tree.show)}")

    // TODO workout inherited members
    tpeSym.fieldMembers
      .sortBy(_.pos.map(p => (p.startLine, p.startColumn))) // make sure the order follows source code decl. order
      .traverse(field =>
        (field.tree match {
          case d: q.ValDef =>
            q.FnContext().typer(d.tpt.tpe).flatMap {
              case (_, t: p.Type, c) => p.Named(field.name, t).success.deferred
              case (_, bad, c)       => s"bad erased type $bad".fail.deferred
            }
          case _ => ???
        })
      )
      .map(p.StructDef(p.Sym(tpeSym.fullName), _))
  }

  extension (using q: Quoted)(c: q.FnContext) {

    private def resolveClsFromSymbol(clsSym: q.Symbol): Result[(q.Symbol, p.Sym, q.ClassKind)] = {
      println(s"[typer] resolveSym ${clsSym.fullName} = ${clsSym.flags.show}")
      if (clsSym.isClassDef) {
        (
          clsSym,
          p.Sym(clsSym.fullName),
          if (clsSym.flags.is(q.Flags.Module)) q.ClassKind.Object else q.ClassKind.Class
        ).success
      } else {
        s"$clsSym is not a class def".fail
      }
    }

    @tailrec private final def resolveClsFromTpeRepr(ref: q.TypeRepr): Result[(q.Symbol, p.Sym, q.ClassKind)] =
      ref.dealias.simplified match {
        case q.ThisType(tpe) => c.resolveClsFromTpeRepr(tpe)
        case tpe: q.NamedType =>
          tpe.classSymbol match {
            case None                              => s"Named type is not a class: ${tpe}".fail
            case Some(sym) if sym.name == "<root>" => c.resolveClsFromTpeRepr(tpe.qualifier) // discard root package
            case Some(sym)                         => c.resolveClsFromSymbol(sym)
          }
        case invalid => s"Invalid type: ${invalid}".fail
      }

    private def liftClsToTpe(clsSym: q.Symbol, sym: p.Sym, kind: q.ClassKind) = {
      val t0: q.Tpe = (sym, kind) match {
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
      t0 match {
        case s @ p.Type.Struct(sym) =>
          lowerClassType(clsSym).map(d => (s, c.copy(clss = c.clss + (sym -> d)))).resolve
        case x => (x, c).success
      }
    }

    def clsSymTyper(clsSym: q.Symbol): Result[(q.Tpe, q.FnContext)] =
      c.resolveClsFromSymbol(clsSym).flatMap(c.liftClsToTpe(_, _, _))

    def typerN(xs: List[q.TypeRepr]): Deferred[(List[(Option[p.Term], q.Tpe)], q.FnContext)] = xs match {
      case Nil     => (Nil, c).pure
      case x :: xs =>
        // TODO make sure we get the right order back!
        c.typer(x).flatMap { (v, t, c) =>
          xs.foldLeftM(((v, t) :: Nil, c)) { case ((ys, c), x) =>
            c.typer(x).map((v, t, c) => ((v, t) :: ys, c))
          }
        }
    }

    def typer(repr: q.TypeRepr): Deferred[(Option[p.Term], q.Tpe, q.FnContext)] =
      repr.dealias.widenTermRefByName.simplified match {
        case p @ q.PolyType(_, _, q.MethodType(_, _, _)) =>
          // this shows up from type-unapplied methods:  [x, y] =>> methodTpe(_:x, ...):y
//          (None, q.ErasedOpaqueTpe(p), c).success.deferred
          ???
        case q.MethodType(_, args, rtn) =>
          for {
            (_, tpe, c) <- c.typer(rtn)
            (xs, c)     <- c.typerN(args)
          } yield (None, q.ErasedFnTpe(xs.map(_._2), tpe), c)
        case andOr: q.AndOrType =>
          for {
            (leftTerm, leftTpe, c)   <- c.typer(andOr.left)
            (rightTerm, rightTpe, c) <- c.typer(andOr.right)
          } yield
            if leftTpe == rightTpe then (leftTerm.orElse(rightTerm), leftTpe, c)
            else ???
        case tpe @ q.AppliedType(ctor, args) =>
          for {
            (_, name, kind) <- c.resolveClsFromTpeRepr(ctor).deferred
            (xs, c)         <- c.typerN(args)
          } yield (name, kind, xs) match {
            case (Symbols.Buffer, q.ClassKind.Class, (_, comp: p.Type) :: Nil) => (None, p.Type.Array(comp), c)
            case (Symbols.Array, q.ClassKind.Class, (_, comp: p.Type) :: Nil)  => (None, p.Type.Array(comp), c)
            case (_, _, ys) if tpe.isFunctionType                              => // FunctionN
              // TODO make sure this works
              (
                None,
                ys.map(_._2) match {
                  case Nil      => ???
                  case x :: Nil => q.ErasedFnTpe(Nil, x)
                  case xs :+ x  => q.ErasedFnTpe(xs, x)
                },
                c
              )
            case (n, m, ys) =>
              // lowerClassType(ctor.classSymbol.get).map( s =>  )
              (None, q.ErasedClsTpe(n, m, ys.map(_._2)), c)
          }
        // widen singletons
        case q.ConstantType(x) =>
          (x match {
            case q.BooleanConstant(v) => (Some(p.Term.BoolConst(v)), p.Type.Bool, c)
            case q.ByteConstant(v)    => (Some(p.Term.ByteConst(v)), p.Type.Byte, c)
            case q.ShortConstant(v)   => (Some(p.Term.ShortConst(v)), p.Type.Short, c)
            case q.IntConstant(v)     => (Some(p.Term.IntConst(v)), p.Type.Int, c)
            case q.LongConstant(v)    => (Some(p.Term.LongConst(v)), p.Type.Long, c)
            case q.FloatConstant(v)   => (Some(p.Term.FloatConst(v)), p.Type.Float, c)
            case q.DoubleConstant(v)  => (Some(p.Term.DoubleConst(v)), p.Type.Double, c)
            case q.CharConstant(v)    => (Some(p.Term.CharConst(v)), p.Type.Char, c)
            case q.StringConstant(v)  => ???
            case q.UnitConstant       => (Some(p.Term.UnitConst), p.Type.Unit, c)
            case q.NullConstant       => ???
            case q.ClassOfConstant(r) => ???
          }).pure

        case expr =>
          c.resolveClsFromTpeRepr(expr).flatMap(c.liftClsToTpe(_, _, _)).map((t, c) => (None, t, c)).deferred
      }
  }
}
