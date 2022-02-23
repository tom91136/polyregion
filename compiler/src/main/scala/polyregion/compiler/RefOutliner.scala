package polyregion.compiler

import scala.annotation.tailrec
import cats.syntax.all.*
import polyregion.*
import polyregion.ast.PolyAst as p
import Retyper.*

object RefOutliner {

  // pull all Ident out of Select/Indent or None
  @tailrec private final def idents(using
      q: Quoted
  )(t: q.Term, xs: List[String] = Nil): Option[(q.Ident, List[String])] = t match {
    case q.Select(x, n) => idents(x, n :: xs) // outside first, no need to reverse at the end
    case i @ q.Ident(_) => Some(i, xs)
    case _              => None               // we got a non ref node, give up
  }

  def outline(using q: Quoted)(term: q.Term): Result[(Vector[(q.Ref, q.Reference)], q.FnContext)] = {

    val foreignRefs = q.collectTree[q.Ref](term) {
      // FIXME TODO make sure the owner is actually this macro, and not any other macro
      case ref: q.Ref if !ref.symbol.maybeOwner.flags.is(q.Flags.Macro) => ref :: Nil
      case _                                                            => Nil
    }

    // drop anything that isn't a val ref and resolve root ident
    val normalisedForeignValRefs = (for {
      s <- foreignRefs.distinctBy(_.symbol)
      if !s.symbol.flags.is(q.Flags.Module) && (
        s.symbol.isValDef ||
          (s.symbol.isDefDef && s.symbol.flags.is(q.Flags.FieldAccessor))
      )

      (root, path) <- idents(s)

      // the entire path is not foreign if the root is not foreign
      // TODO see above, need robust owner validation
      if !root.symbol.maybeOwner.flags.is(q.Flags.Macro)

    } yield (root, path.toVector, s)).sortBy(_._2.length)

    // remove refs if already covered by the same root
    val sharedValRefs = normalisedForeignValRefs
      .foldLeft(Vector.empty[(q.Ident, Vector[String], q.Ref)]) {
        case (acc, x @ (_, _, q.Ident(_))) => acc :+ x
        case (acc, x @ (root, path, q.Select(i, _))) =>
          println(s"$root->${path}")
          acc.filterNot((root0, path0, _) => root.symbol == root0.symbol && path.startsWith(path0)) :+ x
        case (_, (_, _, r)) =>
          // not recoverable, the API shape is different
          q.report.errorAndAbort(s"Unexpected val (reference) kind while outlining", r.asExpr)
      }
      .map((_, _, ref) => ref)

    // final vals in stdlib  :  Flags.{FieldAccessor, Final, Method, StableRealizable}
    // free methods          :  Flags.{Method}
    // free vals             :  Flags.{}

    println(
      s" -> foreign refs:${" " * 9}\n${foreignRefs.map(x => s"${x.show} (${x.symbol}) ~> $x, tpe=${x.tpe.widenTermRefByName.show}").mkString("\n").indent(4)}"
    )
    println(
      s" -> filtered  (found):${" " * 9}\n${normalisedForeignValRefs.map(x => s"${x._3.show} (${x._3.symbol}) ~> $x").mkString("\n").indent(4)}"
    )
    println(
      s" -> collapse  (found):${" " * 9}\n${sharedValRefs.map(x => s"${x.symbol} ~> $x").mkString("\n").indent(4)}"
    )

    val typedRefs = sharedValRefs.foldLeftM((Vector.empty[(q.Ref, q.Reference)], q.FnContext())) {
      case ((xs, c), i @ q.Ident(_)) =>
        c.typer(i.tpe).map {
          case (Some(x), tpe, c) => (xs :+ (i, q.Reference(x, tpe)), c)
          case (None, tpe, c)    => (xs :+ (i, q.Reference(i.symbol.name, tpe)), c)
        }
      case ((xs, c), s @ q.Select(term, name)) =>
        c.typer(s.tpe).map {
          case (Some(x), tpe, c) => (xs :+ (s, q.Reference(x, tpe)), c)
          case (None, tpe, c)    => (xs :+ (s, q.Reference("_ref_" + name + "_" + s.pos.startLine + "_", tpe)), c)
        }
      case (_, r) =>
        q.report.errorAndAbort(s"Unexpected val (reference) kind while outlining", r.asExpr)
    }

    // remove anything we can't use, like ClassTag
    val filteredTypedRefs = typedRefs.map { (xs, c) =>
      val ys = xs.filter {
        case (_, q.Reference(_, q.ErasedClsTpe(Symbols.ClassTag, q.ClassKind.Class, Nil))) => false
        case _                                                                             => true
      }
      (ys, c)
    }

    val resolved = filteredTypedRefs.resolve
    println(resolved match {
      case Right((xs, c)) =>
        s" -> typer   (found):${" " * 9}\n${xs.map((r, ref) => s"${r.symbol} ~> $ref").mkString("\n").indent(4)}"
      case Left(e) => s" -> typer   (found):${e.getMessage}"
    })

    resolved
  }

}
