//package polyregion.scala
//
//import cats.syntax.all.*
//import polyregion.ast.{PolyAst as p, *}
//import polyregion.scala.{Quoted, Symbols}
//
//import scala.annotation.tailrec
//
//object MirrorPass {
//
//  def mirror(using q: Quoted)(mirror: Map[p.Signature, (p.Function, p.StructDef)])(c: q.FnContext): q.FnContext = {
//
//    def replace(t: p.Term, tpe: p.Type) = (t, tpe) match {
//      case (term, tpe) if term.tpe == tpe               => term
//      case (p.Term.Select(init, p.Named(last, _)), tpe) => p.Term.Select(init, p.Named(last, tpe))
//      case (original, replacement) =>
//        println(s"$original -> $replacement")
//        ???
//    }
//
//    val (xs, acc) = c.stmts.foldMapM(_.mapAccExpr[(p.Signature, p.Function, p.StructDef)] {
//      case inv @ p.Expr.Invoke(sym, recv, args, rtn) =>
//        val s = p.Signature(sym, recv.map(_.tpe), args.map(_.tpe), rtn)
//        mirror.get(s) match {
//          case None => (inv, Nil, Nil)
//          case Some((f, sdef)) =>
//            (
//              p.Expr.Invoke(
//                name = f.name,
//                receiver = (recv, f.receiver) match {
//                  case (Some(term), Some(p.Named(_, tpe))) => Some(replace(term, tpe))
//                  case (None, None)                        => None
//                  case (original, replacement) =>
//                    println(s"Receiver mismatch: $original -> $replacement")
//                    ???
//                },
//                args.zip(f.args).map((t, n) => replace(t, n.tpe)),
//                rtn = f.rtn
//              ),
//              Nil,
//              (s, f, sdef) :: Nil
//            )
//        }
//      case x => (x, Nil, Nil)
//    })
//
//    val (replacedSigs, mirrored, sdefs) = acc.unzip3
//    c.replaceStmts(xs).copy(defs = c.defs -- replacedSigs, mirrored = mirrored, clss = c.clss ++ (sdefs.map(d => d.name -> d).toMap))
//  }
//
//}
