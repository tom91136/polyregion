package polyregion.compiler

import cats.syntax.all.*
import polyregion.ast.PolyAst as p
import polyregion.*

import scala.annotation.tailrec
import scala.quoted.*
import simulacrum.typeclass

object Retyper {

  @tailrec private final def resolveSym(using q: Quoted)(ref: q.TypeRepr): Result[p.Sym] =
    ref.dealias.simplified match {
      case q.ThisType(tpe) => resolveSym(tpe)
      case tpe: q.NamedType =>
        tpe.classSymbol match {
          case None => s"Named type is not a class: ${tpe}".fail
          case Some(sym) if sym.name == "<root>" => // discard root package
            resolveSym(tpe.qualifier)
          case Some(sym) => p.Sym(sym.fullName).success
        }
      // case NoPrefix()    => None.success
      case invalid => s"Invalid type: ${invalid}".fail
    }

  def lowerProductType[A: Type](using q: Quoted): Deferred[p.StructDef] = lowerProductType(q.TypeRepr.of[A].typeSymbol)
  def lowerProductType(using q: Quoted)(tpeSym: q.Symbol): Deferred[p.StructDef] = {

    if (!tpeSym.flags.is(q.Flags.Case)) {
      throw RuntimeException(s"Unsupported combination of flags: ${tpeSym.flags.show} for ${tpeSym}")
    }

    tpeSym.caseFields
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

    def typer(repr: q.TypeRepr): Deferred[(Option[p.Term], q.Tpe, q.FnContext)] =
      repr.dealias.widenTermRefByName.simplified match {
        // case q.MethodType(names, args, rtn) => ???

        case andOr: q.AndOrType =>
          for {
            (leftTerm, leftTpe, c)   <- c.typer(andOr.left)
            (rightTerm, rightTpe, c) <- c.typer(andOr.right)
          } yield
            if leftTpe == rightTpe then (leftTerm.orElse(rightTerm), leftTpe, c)
            else ???
        case tpe @ q.AppliedType(ctor, args) =>
          for {
            name    <- resolveSym(ctor).deferred
            (xs, c) <- args.foldMapM(x => c.typer(x).map((v, t, c) => ((v, t) :: Nil, c)))
          } yield (name, xs) match {
            case (Symbols.Buffer, (_, comp: p.Type) :: Nil) => (None, p.Type.Array(comp, None), c)
            case (n, ys)                                    => (None, q.ErasedTpe(n, ys.map(_._2)), c)
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
          resolveSym(expr)
            .map {
              case p.Sym(Symbols.Scala :+ "Unit")      => p.Type.Unit
              case p.Sym(Symbols.Scala :+ "Boolean")   => p.Type.Bool
              case p.Sym(Symbols.Scala :+ "Byte")      => p.Type.Byte
              case p.Sym(Symbols.Scala :+ "Short")     => p.Type.Short
              case p.Sym(Symbols.Scala :+ "Int")       => p.Type.Int
              case p.Sym(Symbols.Scala :+ "Long")      => p.Type.Long
              case p.Sym(Symbols.Scala :+ "Float")     => p.Type.Float
              case p.Sym(Symbols.Scala :+ "Double")    => p.Type.Double
              case p.Sym(Symbols.Scala :+ "Char")      => p.Type.Char
              case p.Sym(Symbols.JavaLang :+ "String") => p.Type.String
              case sym                                 => p.Type.Struct(sym)
            }
            .flatMap {
              case s @ p.Type.Struct(sym) =>
                lowerProductType(expr.typeSymbol).map(d => (None, s, c.copy(clss = c.clss + (sym -> d)))).resolve
              case x => (None, x, c).success
            }
            .deferred
      }
  }
}
