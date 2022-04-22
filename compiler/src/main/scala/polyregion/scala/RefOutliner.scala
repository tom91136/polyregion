package polyregion.scala

import cats.syntax.all.*
import polyregion.ast.{PolyAst as p, *}
import polyregion.scala.Retyper.*

import scala.annotation.tailrec

object RefOutliner {

  // pull all Ident out of Select/Indent or None
  @tailrec private final def idents(using
      q: Quoted
  )(t: q.Term, xs: List[String] = Nil): Option[(q.Ident, List[String])] = t match {
    case q.Select(x, n) => idents(x, n :: xs) // outside first, no need to reverse at the end
    case i @ q.Ident(_) => Some(i, xs)
    case _              => None               // we got a non ref node, give up
  }

  def outline       //
  (using q: Quoted) //
  (term: q.Term)    //
  (log: Log)        //
      : Result[((Vector[(q.Ident, q.Ref, Option[p.Term], p.Type)], q.FnDependencies), Log)] = log.mark(s"Outline") {
    log =>
      for {

        _ <- ().success

        localDefs = q.collectTree(term) {
          case b: q.ValDef => b :: Nil
          case _           => Nil
        }
        localDefTable = localDefs.map(_.symbol).toSet

        log <- log.info("Local ValDefs", localDefs.map(_.toString)*)

        foreignRefs = q.collectTree[q.Ref](term) {
//      case ref: q.Ref if !ref.symbol.maybeOwner.flags.is(q.Flags.Macro) => ref :: Nil
          case ref: q.Ref if !localDefTable.contains(ref.symbol) => ref :: Nil
          case _                                                 => Nil
        }

        // drop anything that isn't a val ref and resolve root ident
        normalisedForeignValRefs = (for {
          s <- foreignRefs // .distinctBy(_.symbol)
          if !s.symbol.flags.is(q.Flags.Module) && (
            s.symbol.isValDef ||
              (s.symbol.isDefDef && s.symbol.flags.is(q.Flags.FieldAccessor))
          )

          (root, path) <- idents(s) // s === root.$path

          // the entire path is not foreign if the root is not foreign
          if !localDefTable.contains(root.symbol)
//      if !root.symbol.maybeOwner.flags.is(q.Flags.Macro)

        } yield (root, path.toVector, s)).sortBy(_._2.length)

        // remove refs if already covered by the same root
        sharedValRefs = normalisedForeignValRefs
          .foldLeft(Vector.empty[(q.Ident, Vector[String], q.Ref)]) {
            case (acc, x @ (_, _, q.Ident(_))) => acc :+ x
            case (acc, x @ (root, path, q.Select(i, _))) =>
              acc.filterNot((root0, path0, _) => root.symbol == root0.symbol && path.startsWith(path0)) :+ x
            case (_, (_, _, r)) =>
              // not recoverable, the API shape is different
              q.report.errorAndAbort(s"Unexpected val (reference) kind while outlining", r.asExpr)
          }
          .map((root, _, ref) => (root, ref))

        // final vals in stdlib  :  Flags.{FieldAccessor, Final, Method, StableRealizable}
        // free methods          :  Flags.{Method}
        // free vals             :  Flags.{}

        log <- log.info(
          "Foreign Refs",
          foreignRefs.map(x =>
            s"${x.show}(symbol=${x.symbol}, owner=${x.symbol.owner}) ~> $x, tpe=${x.tpe.widenTermRefByName.show}"
          )*
        )
        log <- log.info(
          "Normalised",
          normalisedForeignValRefs.map((root, path, x) =>
            s"${x.show}(symbol=${x.symbol}, root=$root, path=${path.mkString})"
          )*
        )
        log <- log.info(
          "Root collapsed",
          sharedValRefs.map((root, x) => s"${x.show}(symbol=${x.symbol}, owner=${x.symbol.owner}, root=$root)")*
        )

        // Set[(q.Ident, q.Ref, Option[p.Term])]

        (typedRefs, c) <- sharedValRefs.foldLeftM(
          (Vector.empty[(q.Ident, q.Ref, Option[p.Term], p.Type)], q.FnDependencies())
        ) { case ((xs, c), (root, i: q.Ref)) =>
          Retyper.typer1(i.tpe).map { case (term, tpe, c0) =>
            (xs :+ (root, i, term, tpe), c0 |+| c)
          }
        }

        // (typedRefs, c) <- sharedValRefs.foldLeftM((Vector.empty[(q.Ident, q.Ref, q.Reference)], c)) {
        //   case ((xs, c), (root, i @ q.Ident(_))) =>
        //     c.typer(i.tpe).map {
        //       case (Some(x), tpe, c) => (xs :+ (root, i, q.Reference(x, tpe)), c)
        //       case (None, tpe, c)    => (xs :+ (root, i, q.Reference(i.name, tpe)), c)
        //     }
        //   case ((xs, c), (root, s @ q.Select(term, name))) =>
        //     c.typer(s.tpe).map {
        //       case (Some(x), tpe, c) => (xs :+ (root, s, q.Reference(x, tpe)), c)
        //       case (None, tpe, c) => (xs :+ (root, s, q.Reference("_ref_" + name + "_" + s.pos.startLine + "_", tpe)), c)
        //     }
        //   case (_, r) => s"Unexpected val (${r}) kind while outlining".fail
        // }

        // remove anything we can't use, like ClassTag
        filteredTypedRefs = typedRefs.filter {
          // case (_, _, _, q.ErasedClsTpe(Symbols.ClassTag,_, q.ClassKind.Class, Nil)) => false
          case (_, _, _, p.Type.Struct(Symbols.ClassTag, p.Type.Var(_) :: Nil)) => false
          case _                                                                   => true
        }

        log <- log.info(
          s"Typed",
          filteredTypedRefs.map((root, ref, value, tpe) =>
            s"${ref.show}(symbol=${ref.symbol}, owner=${ref.symbol.owner}, root=${root}${if (root == ref) ";self"
            else ""}) : ${tpe} = ${value}"
          )*
        )

      } yield ((filteredTypedRefs, c), log)
  }
}
