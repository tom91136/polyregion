package polyregion.compiler.pass

import cats.data.EitherT
import cats.syntax.all.*
import polyregion.ast.PolyAst as p
import polyregion.compiler.*

import scala.annotation.tailrec
import scala.quoted.*
import polyregion.compiler.Symbols
import polyregion.compiler.Quoted

object IntrinsifyPass {

  def intrinsify(using q: Quoted)(c: q.FnContext): q.FnContext = {
    val (xs, instanceSyms) = c.stmts.zipWithIndex.foldMapM(intrinsifyInstanceApply(_, _))
    val (ys, moduleSyms)   = xs.zipWithIndex.foldMapM(intrinsifyModuleApply(_, _))
    println(s"Elim : ${c.defs.map(_._1)} -  ${(instanceSyms ++ moduleSyms)} ")
    val eliminated = c.defs -- (instanceSyms ++ moduleSyms)
    c.replaceStmts(ys).copy(defs = eliminated)
  }

  private final inline val DegreesToRadians = 0.017453292519943295

  private final inline val RadiansToDegrees = 57.29577951308232

  private def intrinsifyInstanceApply(s: p.Stmt, idx: Int) = s.mapAccExpr[p.Sym] {
    case inv @ p.Expr.Invoke(sym, Some(recv), args, rtn) =>
      (sym.fqn, recv, args) match {
        case (op :: Nil, x, Nil) if x.tpe.kind == p.TypeKind.Integral || x.tpe.kind == p.TypeKind.Fractional =>
          // xxx bool is integral
          val expr = op match {
            case "toDouble" => p.Expr.Cast(recv, p.Type.Double)
            case "toFloat"  => p.Expr.Cast(recv, p.Type.Float)
            case "toLong"   => p.Expr.Cast(recv, p.Type.Long)
            case "toInt"    => p.Expr.Cast(recv, p.Type.Int)
            case "toShort"  => p.Expr.Cast(recv, p.Type.Short)
            case "toChar"   => p.Expr.Cast(recv, p.Type.Char)
            case "toByte"   => p.Expr.Cast(recv, p.Type.Byte)

            case "toDegrees" =>
              p.Expr.BinaryIntrinsic(x, p.Term.DoubleConst(RadiansToDegrees), p.BinaryIntrinsicKind.Mul, p.Type.Double)
            case "toRadians" =>
              p.Expr.BinaryIntrinsic(x, p.Term.DoubleConst(DegreesToRadians), p.BinaryIntrinsicKind.Mul, p.Type.Double)
            case "unary_!" if x.tpe == p.Type.Bool => p.Expr.UnaryLogicIntrinsic(recv, p.UnaryLogicIntrinsicKind.Not)

            case "unary_~" => p.Expr.UnaryIntrinsic(recv, p.UnaryIntrinsicKind.BNot, x.tpe)

          }
          (expr, Nil, sym :: Nil)
        case (op :: Nil, x, y :: Nil)
            if (x.tpe == y.tpe) && (x.tpe.kind == p.TypeKind.Integral || x.tpe.kind == p.TypeKind.Fractional) =>
          val expr = op match {
            case "+" => p.Expr.BinaryIntrinsic(x, y, p.BinaryIntrinsicKind.Add, rtn)
            case "-" => p.Expr.BinaryIntrinsic(x, y, p.BinaryIntrinsicKind.Sub, rtn)
            case "*" => p.Expr.BinaryIntrinsic(x, y, p.BinaryIntrinsicKind.Mul, rtn)
            case "/" => p.Expr.BinaryIntrinsic(x, y, p.BinaryIntrinsicKind.Div, rtn)
            case "%" => p.Expr.BinaryIntrinsic(x, y, p.BinaryIntrinsicKind.Rem, rtn)

            case "<"  => p.Expr.BinaryLogicIntrinsic(x, y, p.BinaryLogicIntrinsicKind.Lt)
            case "<=" => p.Expr.BinaryLogicIntrinsic(x, y, p.BinaryLogicIntrinsicKind.Lte)
            case ">"  => p.Expr.BinaryLogicIntrinsic(x, y, p.BinaryLogicIntrinsicKind.Gt)
            case ">=" => p.Expr.BinaryLogicIntrinsic(x, y, p.BinaryLogicIntrinsicKind.Gte)
            case "==" => p.Expr.BinaryLogicIntrinsic(x, y, p.BinaryLogicIntrinsicKind.Eq)
            case "!=" => p.Expr.BinaryLogicIntrinsic(x, y, p.BinaryLogicIntrinsicKind.Neq)
            case "&&" => p.Expr.BinaryLogicIntrinsic(x, y, p.BinaryLogicIntrinsicKind.And)
            case "||" => p.Expr.BinaryLogicIntrinsic(x, y, p.BinaryLogicIntrinsicKind.Or)

            case "&"   => p.Expr.BinaryIntrinsic(x, y, p.BinaryIntrinsicKind.BAnd, rtn)
            case "|"   => p.Expr.BinaryIntrinsic(x, y, p.BinaryIntrinsicKind.BOr, rtn)
            case "^"   => p.Expr.BinaryIntrinsic(x, y, p.BinaryIntrinsicKind.BXor, rtn)
            case "<<"  => p.Expr.BinaryIntrinsic(x, y, p.BinaryIntrinsicKind.BSL, rtn)
            case ">>"  => p.Expr.BinaryIntrinsic(x, y, p.BinaryIntrinsicKind.BSR, rtn)
            case ">>>" => p.Expr.BinaryIntrinsic(x, y, p.BinaryIntrinsicKind.BZSR, rtn)

          }
          (expr, Nil, sym :: Nil)
        case ("apply" :: Nil, (xs @ p.Term.Select(_, p.Named(_, p.Type.Array(_)))), idx :: Nil)
            if idx.tpe.kind == p.TypeKind.Integral =>
          (p.Expr.Index(xs, idx, rtn), Nil, sym :: Nil)
        case ("update" :: Nil, (xs @ p.Term.Select(_, p.Named(_, p.Type.Array(_)))), idx :: x :: Nil)
            if idx.tpe.kind == p.TypeKind.Integral =>
          (p.Expr.Alias(p.Term.UnitConst), p.Stmt.Update(xs, idx, x) :: Nil, sym :: Nil)
        case (unknownSym, recv, args) =>
          println(s"No instance intrinsic for call: (($recv) : ${unknownSym.mkString(".")})(${args.mkString(",")}) ")
          (inv, Nil, Nil)
      }
    case x => (x, Nil, Nil)
  }

  private def intrinsifyModuleApply(s: p.Stmt, idx: Int) =
    s.mapAccExpr[p.Sym] {
      case inv @ p.Expr.Invoke(sym, None, args, rtn) =>
        (sym.fqn, args) match {
          case (Symbols.ArrayModule :+ "ofDim", x :: Nil) =>
            rtn match {
              case arr: p.Type.Array =>
                (p.Expr.Alloc(arr, x), Nil, sym :: Nil)
              case _ => ???
            }
          case ((Symbols.ScalaMath | Symbols.JavaMath) :+ op, x :: y :: Nil) => // scala.math binary
            val kind = op match {
              case "pow" => p.BinaryIntrinsicKind.Pow

              case "min" => p.BinaryIntrinsicKind.Min
              case "max" => p.BinaryIntrinsicKind.Max

              case "atan2" => p.BinaryIntrinsicKind.Atan2
              case "hypot" => p.BinaryIntrinsicKind.Hypot
            }
            (p.Expr.BinaryIntrinsic(x, y, kind, rtn), Nil, sym :: Nil)
          case ((Symbols.ScalaMath | Symbols.JavaMath) :+ op, x :: Nil) => // scala.math unary
            val expr = op match {
              case "toDegrees" =>
                p.Expr.BinaryIntrinsic(
                  x,
                  p.Term.DoubleConst(RadiansToDegrees),
                  p.BinaryIntrinsicKind.Mul,
                  p.Type.Double
                )
              case "toRadians" =>
                p.Expr.BinaryIntrinsic(
                  x,
                  p.Term.DoubleConst(DegreesToRadians),
                  p.BinaryIntrinsicKind.Mul,
                  p.Type.Double
                )
              case "sin"  => p.Expr.UnaryIntrinsic(x, p.UnaryIntrinsicKind.Sin, rtn)
              case "cos"  => p.Expr.UnaryIntrinsic(x, p.UnaryIntrinsicKind.Cos, rtn)
              case "tan"  => p.Expr.UnaryIntrinsic(x, p.UnaryIntrinsicKind.Tan, rtn)
              case "asin" => p.Expr.UnaryIntrinsic(x, p.UnaryIntrinsicKind.Asin, rtn)
              case "acos" => p.Expr.UnaryIntrinsic(x, p.UnaryIntrinsicKind.Acos, rtn)
              case "atan" => p.Expr.UnaryIntrinsic(x, p.UnaryIntrinsicKind.Atan, rtn)
              case "sinh" => p.Expr.UnaryIntrinsic(x, p.UnaryIntrinsicKind.Sinh, rtn)
              case "cosh" => p.Expr.UnaryIntrinsic(x, p.UnaryIntrinsicKind.Cosh, rtn)
              case "tanh" => p.Expr.UnaryIntrinsic(x, p.UnaryIntrinsicKind.Tanh, rtn)

              case "signum" => p.Expr.UnaryIntrinsic(x, p.UnaryIntrinsicKind.Signum, rtn)
              case "abs"    => p.Expr.UnaryIntrinsic(x, p.UnaryIntrinsicKind.Abs, rtn)
              case "round"  => p.Expr.UnaryIntrinsic(x, p.UnaryIntrinsicKind.Round, rtn)
              case "ceil"   => p.Expr.UnaryIntrinsic(x, p.UnaryIntrinsicKind.Ceil, rtn)
              case "floor"  => p.Expr.UnaryIntrinsic(x, p.UnaryIntrinsicKind.Floor, rtn)
              case "rint"   => p.Expr.UnaryIntrinsic(x, p.UnaryIntrinsicKind.Rint, rtn)

              case "sqrt"  => p.Expr.UnaryIntrinsic(x, p.UnaryIntrinsicKind.Sqrt, rtn)
              case "cbrt"  => p.Expr.UnaryIntrinsic(x, p.UnaryIntrinsicKind.Cbrt, rtn)
              case "exp"   => p.Expr.UnaryIntrinsic(x, p.UnaryIntrinsicKind.Exp, rtn)
              case "expm1" => p.Expr.UnaryIntrinsic(x, p.UnaryIntrinsicKind.Expm1, rtn)
              case "log"   => p.Expr.UnaryIntrinsic(x, p.UnaryIntrinsicKind.Log, rtn)
              case "log1p" => p.Expr.UnaryIntrinsic(x, p.UnaryIntrinsicKind.Log1p, rtn)
              case "log10" => p.Expr.UnaryIntrinsic(x, p.UnaryIntrinsicKind.Log10, rtn)
            }
            (expr, Nil, sym :: Nil)
          case (unknownSym, args) =>
            println(s"No module intrinsic for: ${unknownSym.mkString(".")}(${args.map(_.repr).mkString(",")}) ")
            (inv, Nil, Nil)
        }
      case x => (x, Nil, Nil)
    }

}
