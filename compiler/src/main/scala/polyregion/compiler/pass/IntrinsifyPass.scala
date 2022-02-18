package polyregion.compiler.pass

import cats.data.EitherT
import cats.syntax.all.*
import polyregion.ast.PolyAst as p
import polyregion.*

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

  private def intrinsifyInstanceApply(s: p.Stmt, idx: Int) = s.mapAccExpr[p.Sym] {
    case inv @ p.Expr.Invoke(sym, Some(recv), args, rtn) =>
      (sym.fqn, recv, args) match {

        case (op :: Nil, x, Nil) if x.tpe.kind == p.TypeKind.Integral || x.tpe.kind == p.TypeKind.Fractional =>
          val expr = op match {
            case "toDouble" => p.Expr.Cast(recv, p.Type.Double)
            case "toFloat"  => p.Expr.Cast(recv, p.Type.Float)
            case "toLong"   => p.Expr.Cast(recv, p.Type.Long)
            case "toInt"    => p.Expr.Cast(recv, p.Type.Int)
            case "toShort"  => p.Expr.Cast(recv, p.Type.Short)
            case "toChar"   => p.Expr.Cast(recv, p.Type.Char)
            case "toByte"   => p.Expr.Cast(recv, p.Type.Byte)
            case _          => ???
          }
          (expr, Nil, sym :: Nil)
        case (op :: Nil, x, y :: Nil)
            if (x.tpe == y.tpe) && (x.tpe.kind == p.TypeKind.Integral || x.tpe.kind == p.TypeKind.Fractional) =>
          val expr = op match {
            case "+"  => p.Expr.BinaryIntrinsic(x, y, p.BinaryIntrinsicKind.Add, rtn)
            case "-"  => p.Expr.BinaryIntrinsic(x, y, p.BinaryIntrinsicKind.Sub, rtn)
            case "*"  => p.Expr.BinaryIntrinsic(x, y, p.BinaryIntrinsicKind.Mul, rtn)
            case "/"  => p.Expr.BinaryIntrinsic(x, y, p.BinaryIntrinsicKind.Div, rtn)
            case "%"  => p.Expr.BinaryIntrinsic(x, y, p.BinaryIntrinsicKind.Rem, rtn)
            case "<"  => p.Expr.BinaryLogicIntrinsic(x, y, p.BinaryLogicIntrinsicKind.Lt)
            case "<=" => p.Expr.BinaryLogicIntrinsic(x, y, p.BinaryLogicIntrinsicKind.Lte)
            case ">"  => p.Expr.BinaryLogicIntrinsic(x, y, p.BinaryLogicIntrinsicKind.Gt)
            case ">=" => p.Expr.BinaryLogicIntrinsic(x, y, p.BinaryLogicIntrinsicKind.Gte)
            case "==" => p.Expr.BinaryLogicIntrinsic(x, y, p.BinaryLogicIntrinsicKind.Eq)
            case "!=" => p.Expr.BinaryLogicIntrinsic(x, y, p.BinaryLogicIntrinsicKind.Neq)
            case "&&" => p.Expr.BinaryLogicIntrinsic(x, y, p.BinaryLogicIntrinsicKind.And)
            case "||" => p.Expr.BinaryLogicIntrinsic(x, y, p.BinaryLogicIntrinsicKind.Or)
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
            ???
          case ((Symbols.ScalaMath | Symbols.JavaMath) :+ op, x :: Nil) => // scala.math unary
            val expr = op match {
              case "sin" => p.Expr.UnaryIntrinsic(x, p.UnaryIntrinsicKind.Sin, rtn)
              case "cos" => p.Expr.UnaryIntrinsic(x, p.UnaryIntrinsicKind.Cos, rtn)
              case "tan" => p.Expr.UnaryIntrinsic(x, p.UnaryIntrinsicKind.Tan, rtn)
              case "abs" => p.Expr.UnaryIntrinsic(x, p.UnaryIntrinsicKind.Abs, rtn)
            }
            (expr, Nil, sym :: Nil)
          case (unknownSym, args) =>
            println(s"No module intrinsic for: ${unknownSym.mkString(".")}(${args.map(_.repr).mkString(",")}) ")
            (inv, Nil, Nil)
        }
      case x => (x, Nil, Nil)
    }

}
