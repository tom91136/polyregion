package polyregion.ast.pass

import cats.data.EitherT
import cats.syntax.all.*
import polyregion.ast.{PolyAst as p, *}

import scala.annotation.tailrec

object VerifyPass {

  def transform(fs: List[p.Function]): List[(p.Function, List[String])] = fs.map { f =>
    f -> (f.body match {
      case Nil =>
        List("Function does not contain any statement")
      case xs =>
        xs.flatMap(x =>
          x.acc[p.Type] {
            case p.Stmt.Return(e) => e.tpe :: Nil
            case x                => Nil
          }
        ) match {
          case Nil => List("Function contains no return statements")
          case ts if ts.exists(_ != f.rtn) =>
            List(
              s"Not all return stmts return the function return type, expected ${f.rtn.repr}, got ${ts.map(_.repr).mkString(",")}"
            )
          case _ => Nil
        }
    })
  }
}
