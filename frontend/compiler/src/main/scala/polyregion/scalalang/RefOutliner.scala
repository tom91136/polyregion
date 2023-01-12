package polyregion.scalalang

import cats.syntax.all.*
import polyregion.ast.{PolyAst as p, *}
import polyregion.scalalang.Retyper.*

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

  def outline(using q: Quoted)(sink: Log, term: q.Term) //
      : Result[(List[(q.Ident, q.Ref, q.Retyped)], q.ClsWitnesses)] = for {

    log <- sink.subLog(s"Outline").success

    localDefs = q.collectTree(term) {
      case b: q.ValDef => b :: Nil
      case _           => Nil
    }
    localDefTable = localDefs.map(_.symbol).toSet

    _ = log.info("Local ValDefs", localDefs.map(_.toString)*)

    foreignRefs = q.collectTree[q.Ref](term) {
      // case ref: q.Ref if !ref.symbol.maybeOwner.flags.is(q.Flags.Macro) => ref :: Nil
      case ref: q.Ref if !localDefTable.contains(ref.symbol) => ref :: Nil
      case _                                                 => Nil
    }

    // Drop anything that isn't a val ref and resolve root ident.
    normalisedForeignValRefs = (for {
      s <- foreignRefs // .distinctBy(_.symbol)
      if !s.symbol.flags.is(q.Flags.Module) && (
        s.symbol.isValDef ||
          (s.symbol.isDefDef && s.symbol.flags.is(q.Flags.FieldAccessor))
      )

      (root, path) <- idents(s) // s === root.$path

      // The entire path is not foreign if the root is not foreign.
      if !localDefTable.contains(root.symbol)
      // if !root.symbol.maybeOwner.flags.is(q.Flags.Macro)

    } yield (root, ("this" :: path).toVector, s)).distinctBy(_._3.symbol).sortBy(_._2.length)

    // remove refs if already covered by the same root
    sharedValRefs = normalisedForeignValRefs
      .foldLeft(Vector.empty[(q.Ident, Vector[String], q.Ref)]) {
        case (acc, x @ (_, _, q.Ident(_))) =>
          println(s">>>! $x ${acc}")
          acc :+ x
        case (acc, x @ (root, path, s @ q.Select(i, _))) =>
          println(s">>>  $root ~ $path $i ${acc} ${s.symbol.flags.is(q.Flags.Mutable)}")
          if (s.symbol.flags.is(q.Flags.Mutable)) acc
          else {
            if (acc.exists((root0, path0, _) => root.symbol == root0.symbol && path.startsWith(path0))) acc
            else acc :+ x
          }
        case (_, (_, _, r)) =>
          // not recoverable, the API shape is different
          q.report.errorAndAbort(s"Unexpected val (reference) kind while outlining", r.asExpr)
      }
      .map((root, _, ref) => (root, ref))

    // final vals in stdlib  :  Flags.{FieldAccessor, Final, Method, StableRealizable}
    // free methods          :  Flags.{Method}
    // free vals             :  Flags.{}

    _ = log.info(
      "Foreign Refs",
      foreignRefs.map(x =>
        s"${x.show}(symbol=${x.symbol}, owner=${x.symbol.owner}) ~> $x, tpe=${x.tpe.widenTermRefByName.show}"
      )*
    )
    _ = log.info(
      "Normalised",
      normalisedForeignValRefs
        .distinctBy(_._3.symbol)
        .map((root, path, x) => s"${x.show}(symbol=${x.symbol}, root=$root, path=${path.mkString(".")})")*
    )
    _ = log.info(
      "Root collapsed",
      sharedValRefs.map((root, x) => s"${x.show}(symbol=${x.symbol}, owner=${x.symbol.owner}, root=$root)")*
    )
//    _ = println(log.render().mkString("\n"))
//
//    _ = ???

    // Set[(q.Ident, q.Ref, Option[p.Term])]

    (typedRefs, wit) <- sharedValRefs.foldMapM { (root, i: q.Ref) =>
      Retyper.typer0(i.tpe).map { case (t, wit) => ((root, i, t) :: Nil, wit) }
    }

    uniqueTypedRefs = typedRefs.distinctBy { case (root, ref, _) => root.symbol -> ref.symbol }

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
    filteredTypedRefs = uniqueTypedRefs.filter {
      // case (_, _, _, q.ErasedClsTpe(Symbols.ClassTag,_, q.ClassKind.Class, Nil)) => false
      case (_, _, _ -> p.Type.Struct(Symbols.ClassTag, _, p.Type.Var(_) :: Nil)) => false
      case _                                                                     => true
    }

    _ = log.info(
      s"Typed",
      filteredTypedRefs.map { case (root, ref, value -> tpe) =>
        s"${ref.show}(symbol=${ref.symbol}, owner=${ref.symbol.owner}, root=${root}${
            if (root == ref) ";self"
            else ""
          }) : ${tpe} = ${value}"
      }*
    )

  } yield (filteredTypedRefs, wit)
}
