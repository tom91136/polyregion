package polyregion.compiler.pass

import cats.data.EitherT
import cats.syntax.all.*
import polyregion.ast.PolyAst as p
import polyregion.*

import scala.annotation.tailrec
import scala.quoted.*
import polyregion.compiler.Symbols

object IntrinsifyPass {

  def intrinsify(xs: List[p.Stmt]): List[p.Stmt] = for {
    x <- xs
    x <- intrinsifyInstanceApply(x)
    x <- intrinsifyModuleApply(x)
  } yield x

  private def intrinsifyInstanceApply(s: p.Stmt) = s.mapExpr {
    case inv @ p.Expr.Invoke(sym, Some(recv), args, rtn) =>
      (sym.fqn, recv, args) match {
        case (
              Symbols.Scala :+ ("Byte" | "Short" | "Int" | "Long" | "Float" | "Double" | "Char") :+ op,
              x,
              y :: Nil
            ) =>
          val expr = op match {
            case "+"  => p.Expr.Add(x, y, rtn)
            case "-"  => p.Expr.Sub(x, y, rtn)
            case "*"  => p.Expr.Mul(x, y, rtn)
            case "/"  => p.Expr.Div(x, y, rtn)
            case "%"  => p.Expr.Rem(x, y, rtn)
            case "<"  => p.Expr.Lt(x, y)
            case "<=" => p.Expr.Lte(x, y)
            case ">"  => p.Expr.Gt(x, y)
            case ">=" => p.Expr.Gte(x, y)
            case "==" => p.Expr.Eq(x, y)
            case "!=" => p.Expr.Neq(x, y)
            case "&&" => p.Expr.And(x, y)
            case "||" => p.Expr.Or(x, y)
          }
          (expr, Nil)
        case ((Symbols.SeqOps | Symbols.SeqMutableOps) :+ "apply", (xs: p.Term.Select), idx :: Nil) =>
          (p.Expr.Index(xs, idx, rtn), Nil)
        case (
              (Symbols.SeqOps | Symbols.SeqMutableOps) :+ "update",
              (xs: p.Term.Select),
              idx :: x :: Nil
            ) =>
          (p.Expr.Alias(p.Term.UnitConst), p.Stmt.Update(xs, idx, x) :: Nil)
        case (sym, recv, args) =>
          println(s"No instance intrinsic for call: (($recv) : ${sym.mkString(".")})(${args.mkString(",")}) ")
          (inv, Nil)
      }
    case x => (x, Nil)
  }

  private def intrinsifyModuleApply(s: p.Stmt) = s.mapExpr {
    case inv @ p.Expr.Invoke(sym, None, args, rtn) =>
      (sym.fqn, args) match {
        case ((Symbols.ScalaMath | Symbols.JavaMath) :+ op, x :: y :: Nil) => // scala.math binary
          ???
        case ((Symbols.ScalaMath | Symbols.JavaMath) :+ op, x :: Nil) => // scala.math unary
          val expr = op match {
            case "sin" => p.Expr.Sin(x, rtn)
            case "cos" => p.Expr.Cos(x, rtn)
            case "tan" => p.Expr.Tan(x, rtn)
            case "abs" => p.Expr.Abs(x, rtn)
          }
          (expr, Nil)
        case (sym, args) =>
          println(s"No module intrinsic for: ${sym.mkString(".")}(${args.mkString(",")}) ")
          (inv, Nil)
      }

    case x => (x, Nil)
  }

}
