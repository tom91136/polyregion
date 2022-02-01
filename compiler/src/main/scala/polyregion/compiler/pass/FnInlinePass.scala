package polyregion.compiler.pass

import cats.data.EitherT
import cats.syntax.all.*
import polyregion.ast.PolyAst as p
import polyregion.*

import scala.annotation.tailrec
import scala.quoted.*
import scala.collection.immutable.VectorMap

object FnInlinePass {

  def inlineAll(fs: List[p.Function]): List[p.Function] = {
    val lut = fs.map(f => f.name -> f).to(VectorMap)
    fs.map { f =>
      f.copy(body = f.body.flatMap { x =>
        x.mapExpr {
          case ivk @ p.Expr.Invoke(name, recv, args, tpe) =>
            lut.get(name) match {
              case None => (ivk, Nil) // can't find fn, keep it for now
              case Some(f) =>
                val argSubstituted =
                  f.args.map(p.Term.Select(Nil, _)).zip(args).foldLeft(f.body) { case (xs, (target, replacement)) =>
                    xs.flatMap(
                      _.mapTerm(
                        original => if (original == target) replacement.asInstanceOf else original,
                        original => if (original == target) replacement else original
                      )
                    )
                  }

                val returnName               = p.Named("r", tpe)
                val returnRef: p.Term.Select = p.Term.Select(Nil, returnName)

                val returnRebound = argSubstituted.flatMap(_.map {
                  case p.Stmt.Return(e) => p.Stmt.Mut(returnRef, e) :: Nil
                  case x                => x :: Nil
                })

                val inlineStmts = p.Stmt.Var(returnName, None) :: returnRebound

                (p.Expr.Alias(returnRef), inlineStmts)
            }
          case x => (x, Nil)
        }
      })
    }
  }

}
