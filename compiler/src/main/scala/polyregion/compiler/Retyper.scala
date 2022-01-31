package polyregion.compiler

import scala.quoted.*
import scala.annotation.tailrec
import polyregion.ast.PolyAst
import polyregion.internal.*
import cats.syntax.all.toTraverseOps

import polyregion.ast.{PolyAst => p}

object Retyper {

  @tailrec private final def resolveSym(using q: Quoted)(ref: q.TypeRepr): Result[PolyAst.Sym] =
    ref.dealias.simplified match {
      case q.ThisType(tpe) => resolveSym(tpe)
      case tpe: q.NamedType =>
        tpe.classSymbol match {
          case None => s"Named type is not a class: ${tpe}".fail
          case Some(sym) if sym.name == "<root>" => // discard root package
            resolveSym(tpe.qualifier)
          case Some(sym) => PolyAst.Sym(sym.fullName).success
        }
      // case NoPrefix()    => None.success
      case invalid => s"Invalid type: ${invalid}".fail
    }

  def lowerProductType[A: Type](using q: Quoted): Deferred[p.StructDef] = lowerProductType(q.TypeRepr.of[A].typeSymbol)
  def lowerProductType(using q: Quoted)(tpeSym: q.Symbol): Deferred[p.StructDef] = {

    if (!tpeSym.flags.is(q.Flags.Case)) {
      throw RuntimeException(s"Unsupported combination of flags: ${tpeSym.flags.show}")
    }

    tpeSym.caseFields
      .traverse(field =>
        (field.tree match {
          case d: q.ValDef => q.FnContext().typer(d.tpt.tpe).map((_, t, c) => p.Named(field.name, t))
          case _           => ???
        })
      )
      .map(PolyAst.StructDef(PolyAst.Sym(tpeSym.fullName), _))
  }

  extension (using q: Quoted)(c: q.FnContext) {

    def typer(repr: q.TypeRepr): Deferred[(Option[PolyAst.Term], PolyAst.Type, q.FnContext)] = {
      import q.*
      repr.dealias.widenTermRefByName.simplified match {
        case andOr: AndOrType =>
          for {
            (leftTerm, leftTpe, c)   <- c.typer(andOr.left)
            (rightTerm, rightTpe, c) <- c.typer(andOr.right)
          } yield
            if leftTpe == rightTpe then (leftTerm.orElse(rightTerm), leftTpe, c)
            else ???

        case tpe @ AppliedType(ctor, args) =>
          for {
            name <- resolveSym(ctor).deferred
            xs   <- args.traverse(c.typer(_))
          } yield (name, xs) match {
            case (Symbols.Buffer, (_, component, c) :: Nil) => (None, PolyAst.Type.Array(component, None), c)
            case (n, ys) =>
              println(s"Applied = ${n} args=${ys.map(x => (x._1, x._2)).mkString(",")}")

              ???
            // None -> PolyAst.Type.Struct(n, ys)
          }

        // widen singletons
        case ConstantType(x) =>
          (x match {
            case BooleanConstant(v) => (Some(PolyAst.Term.BoolConst(v)), PolyAst.Type.Bool, c)
            case ByteConstant(v)    => (Some(PolyAst.Term.ByteConst(v)), PolyAst.Type.Byte, c)
            case ShortConstant(v)   => (Some(PolyAst.Term.ShortConst(v)), PolyAst.Type.Short, c)
            case IntConstant(v)     => (Some(PolyAst.Term.IntConst(v)), PolyAst.Type.Int, c)
            case LongConstant(v)    => (Some(PolyAst.Term.LongConst(v)), PolyAst.Type.Long, c)
            case FloatConstant(v)   => (Some(PolyAst.Term.FloatConst(v)), PolyAst.Type.Float, c)
            case DoubleConstant(v)  => (Some(PolyAst.Term.DoubleConst(v)), PolyAst.Type.Double, c)
            case CharConstant(v)    => (Some(PolyAst.Term.CharConst(v)), PolyAst.Type.Char, c)
            case StringConstant(v)  => ???
            case UnitConstant       => (Some(PolyAst.Term.UnitConst), PolyAst.Type.Unit, c)
            case NullConstant       => ???
            case ClassOfConstant(r) => ???
          }).pure

        case expr =>
          resolveSym(expr)
            .map {
              case PolyAst.Sym(Symbols.Scala :+ "Unit")      => PolyAst.Type.Unit
              case PolyAst.Sym(Symbols.Scala :+ "Boolean")   => PolyAst.Type.Bool
              case PolyAst.Sym(Symbols.Scala :+ "Byte")      => PolyAst.Type.Byte
              case PolyAst.Sym(Symbols.Scala :+ "Short")     => PolyAst.Type.Short
              case PolyAst.Sym(Symbols.Scala :+ "Int")       => PolyAst.Type.Int
              case PolyAst.Sym(Symbols.Scala :+ "Long")      => PolyAst.Type.Long
              case PolyAst.Sym(Symbols.Scala :+ "Float")     => PolyAst.Type.Float
              case PolyAst.Sym(Symbols.Scala :+ "Double")    => PolyAst.Type.Double
              case PolyAst.Sym(Symbols.Scala :+ "Char")      => PolyAst.Type.Char
              case PolyAst.Sym(Symbols.JavaLang :+ "String") => PolyAst.Type.String
              case sym                                       => PolyAst.Type.Struct(sym)
            }
            .map {
              case s @ PolyAst.Type.Struct(_) => (None, s, c.copy(clss = c.clss + expr))
              case x                          => (None, x, c)
            }
            .deferred
      }
    }
  }
}
