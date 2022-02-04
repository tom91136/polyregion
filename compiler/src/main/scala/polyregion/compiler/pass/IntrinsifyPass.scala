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
    val (xs, instanceSyms)   = c.stmts.foldMapM(intrinsifyInstanceApply(_))
    val (ys, moduleSyms) = xs.foldMapM(intrinsifyModuleApply(_))
    val eliminated       = c.defs -- (instanceSyms ++ moduleSyms)
    c.replaceStmts(ys).copy(defs = eliminated)
  }

  private def intrinsifyInstanceApply(s: p.Stmt) = s.mapAccExpr[p.Sym] {
    case inv @ p.Expr.Invoke(sym, Some(recv), args, rtn) =>
      (sym.fqn, recv, args) match {
        case (
              Symbols.Scala :+ ("Byte" | "Short" | "Int" | "Long" | "Float" | "Double" | "Char") :+ op,
              x,
              y :: Nil
            ) =>
          val expr = op match {
            case "+"  => p.Expr.BinaryIntrinsic(x, y, p.BinaryIntrinsicKind.Add, rtn)
            case "-"  => p.Expr.BinaryIntrinsic(x, y, p.BinaryIntrinsicKind.Sub, rtn)
            case "*"  => p.Expr.BinaryIntrinsic(x, y, p.BinaryIntrinsicKind.Mul, rtn)
            case "/"  => p.Expr.BinaryIntrinsic(x, y, p.BinaryIntrinsicKind.Div, rtn)
            case "%"  => p.Expr.BinaryIntrinsic(x, y, p.BinaryIntrinsicKind.Rem, rtn)
            case "<"  => p.Expr.Lt(x, y)
            case "<=" => p.Expr.Lte(x, y)
            case ">"  => p.Expr.Gt(x, y)
            case ">=" => p.Expr.Gte(x, y)
            case "==" => p.Expr.Eq(x, y)
            case "!=" => p.Expr.Neq(x, y)
            case "&&" => p.Expr.And(x, y)
            case "||" => p.Expr.Or(x, y)
          }
          (expr, Nil, sym :: Nil)
        case ((Symbols.SeqOps | Symbols.SeqMutableOps) :+ "apply", (xs: p.Term.Select), idx :: Nil) =>
          (p.Expr.Index(xs, idx, rtn), Nil, sym :: Nil)
        case (
              (Symbols.SeqOps | Symbols.SeqMutableOps) :+ "update",
              (xs: p.Term.Select),
              idx :: x :: Nil
            ) =>
          (p.Expr.Alias(p.Term.UnitConst), p.Stmt.Update(xs, idx, x) :: Nil, sym :: Nil)
        case (unknownSym, recv, args) =>
          println(s"No instance intrinsic for call: (($recv) : ${unknownSym.mkString(".")})(${args.mkString(",")}) ")
          (inv, Nil, Nil)
      }
    case x => (x, Nil, Nil)
  }

  private def intrinsifyModuleApply(s: p.Stmt) = s.mapAccExpr[p.Sym] {
    case inv @ p.Expr.Invoke(sym, None, args, rtn) =>
      (sym.fqn, args) match {
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
          println(s"No module intrinsic for: ${unknownSym.mkString(".")}(${args.mkString(",")}) ")
          (inv, Nil, Nil)
      }
    case x => (x, Nil, Nil)
  }

}
