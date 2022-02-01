package polyregion.compiler.pass

import cats.data.EitherT
import cats.syntax.all.*
import polyregion.ast.PolyAst as p
import polyregion.*

import scala.annotation.tailrec
import scala.quoted.*

object FnAllocElisionPass {

  def transform(xs: List[p.Function]): List[p.Function] = {

    // def fn(...): S {
    //   var x: S = ...
    //   (x.a: S.x) := ...
    //   return x
    // }
    //
    // var s: S := fn(...) // out = s

    // def fn(x: S, ...): Unit {
    //   (x.a: S.x) := ...
    //   return ()
    // }
    //
    // var s': S := ...
    // var _: Unit := fn(s', ...)
    // var s: S := s'

    // rewrite all call sites for all functions
    val bodyRewritten = xs.map { f =>
      f.copy(body = f.body.flatMap { x =>
        x.mapExpr {
          case p.Expr.Invoke(name, recv, args, tpe) if tpe.kind == p.TypeKind.Ref && tpe != p.Type.Unit =>
            val alloc  = p.Named("rewrite_alloc", tpe)
            val select = p.Term.Select(Nil, alloc)
            val inv    = p.Expr.Invoke(name, recv, select :: args, p.Type.Unit)
            val stmts = List(
              p.Stmt.Var(alloc, None),
              p.Stmt.Var(p.Named("rewrite_unit", p.Type.Unit), Some(inv))
            )
            (p.Expr.Alias(select), stmts)
          case x => (x, Nil)
        }
      })
    }

    // rewrite all functions with a struct return type
    bodyRewritten
      .filter(f => f.rtn.kind == p.TypeKind.Ref && f.rtn != p.Type.Unit)
      .map { f =>

        // * delete struct alloc stmt
        // * replace return with Unit
        // * add struct var


        val (stmts, allocNames) = f.body.foldMap(_.mapAcc[p.Named] {
          case p.Stmt.Var(n @ p.Named(_, tpe), None) if tpe.kind == f.rtn => (Nil, n :: Nil)
          case p.Stmt.Return(_) => (p.Stmt.Return(p.Expr.Alias(p.Term.UnitConst)) :: Nil, Nil)
          case x                => (x :: Nil, Nil)
        })

        // TODO 
        val incoming = p.Named("rewrite_alloc_incoming", f.rtn)


        val body2 = f.body.flatMap { x =>
          x.mapExpr {
            // case p.Stmt.Return(e) =>  ???
            case p.Expr.Invoke(name, recv, args, tpe) if tpe.kind == p.TypeKind.Ref && tpe != p.Type.Unit =>
              val alloc  = p.Named("rewrite_alloc", tpe)
              val select = p.Term.Select(Nil, alloc)
              val inv    = p.Expr.Invoke(name, recv, select :: args, p.Type.Unit)
              val stmts = List(
                p.Stmt.Var(alloc, None),
                p.Stmt.Var(p.Named("rewrite_unit", p.Type.Unit), Some(inv))
              )
              (p.Expr.Alias(select), stmts)
            case x => (x, Nil)
          }
        }

        f.copy(rtn = p.Type.Unit, body = body2, args = incoming :: f.args)

      }
  }
}
