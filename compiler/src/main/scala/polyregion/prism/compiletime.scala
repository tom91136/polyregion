package polyregion.prism

import scala.annotation.{compileTimeOnly, tailrec}
import scala.quoted.*
import polyregion.scala.*
import cats.syntax.all.*

import scala.compiletime.{erasedValue, summonInline}
import polyregion.ast.{PolyAst as p, *}
import polyregion.scala.Retyper.{lowerClassType, typer}

import scala.reflect.ClassTag

@compileTimeOnly("This class only exists at compile-time to for internal use")
object compiletime {

  private def simplifyTpe(using q: Quoted)(t: q.TypeRepr) = t.dealias.simplified.widenTermRefByName

  inline def derivePackedMirrors1[T <: Tuple]: Map[p.Sym, p.Mirror] = ${ derivePackedMirrors1Impl[T] }

  private def derivePackedMirrors1Impl[T <: Tuple](using Q: Quotes, t: Type[T]): Expr[Map[p.Sym, p.Mirror]] = {
    implicit val q   = Quoted(Q)
    val witnesses    = collectWitnesses[T]().map((a, b) => (simplifyTpe(a), simplifyTpe(b)))
    val witnessTable = witnesses.toMap
    witnesses.traverse(derivePackedMirrorsImpl(_, _)(witnessTable)) match {
      case Left(e) => throw e
      case Right(xs: List[p.Mirror]) =>
        val enc = Expr(MsgPack.encode(xs.map(m => m.source -> m).toMap))
        '{ MsgPack.decode[Map[p.Sym, p.Mirror]]($enc).fold(throw _, x => x) }
    }
  }

  @tailrec private def collectWitnesses[T <: Tuple](using q: Quoted, t: Type[T])(
      xs: List[(q.TypeRepr, q.TypeRepr)] = Nil
  ): List[(q.TypeRepr, q.TypeRepr)] = {
    given Quotes = q.underlying
    t match {
      case '[EmptyTuple]   => xs
      case '[(s, m) *: ts] => collectWitnesses[ts]((q.TypeRepr.of[s], q.TypeRepr.of[m]) :: xs)
    }
  }

  private def derivePackedMirrorsImpl(using q: Quoted) //
  (source: q.TypeRepr, mirror: q.TypeRepr)             //
  (witness: Map[q.TypeRepr, q.TypeRepr]): Result[p.Mirror] = {

    def extractMethodTypes(clsTpe: q.TypeRepr): Result[(q.Symbol, List[(q.Symbol, List[q.TypeRepr], q.TypeRepr)])] =
      clsTpe.classSymbol.failIfEmpty(s"No class symbol for ${clsTpe}").map { sym =>
        sym -> sym.memberMethods
          .filter { m =>
            val flags = m.flags
            flags.is(q.Flags.Method) && !(
              flags.is(q.Flags.Private) || flags.is(q.Flags.Protected) || flags.is(q.Flags.Synthetic)
            )
          }
          .map(m => m -> simplifyTpe(clsTpe.memberType(m)))
          .collect { case (m, q.MethodType(_, argTpes, rtnTpe)) =>
            (m, argTpes.map(simplifyTpe(_)), simplifyTpe(rtnTpe))
          }
      }

    def tpeMatch(source: q.TypeRepr, mirror: q.TypeRepr): Boolean = (source =:= mirror) ||
      (witness.toList.filter((k, v) => k =:= source && v =:= mirror) match {
        case _ :: Nil => true
        case _        => false
      })

    def mkMirroredMethods(
        sourceMethodSym: q.Symbol,
        sourceClassKind: q.ClassKind,
        mirrorMethodSym: q.Symbol,
        expectedStructDef: p.StructDef
    ): Result[List[p.Function]] = for {
      sourceSignature <- sourceMethodSym.tree match {
        case d: q.DefDef => polyregion.scala.Compiler.deriveSignature(d, sourceClassKind)
        case unsupported => s"Unsupported source tree: ${unsupported.show} ".fail
      }

      mirrorMethods <- mirrorMethodSym.tree match {
        case d: q.DefDef =>
          for {
            (fns, sdefs) <- polyregion.scala.Compiler.compileFnAndDependencies(d)

            // make sure our fn matches sourceSignature

            // sdefs

            _ <- sdefs match {
              case `expectedStructDef` :: Nil => ().success
              case bad                        => s"Unexpected struct dependencies for method ${d} : ${bad}".fail
            }
          } yield fns.map{ ( f : p.Function) => f.copy(body = f.body.flatMap(_.mapType {
            case p.Type.Struct(sym)  if sym == ???   => ???
            case x => x
          }))}

        case unsupported => s"Unsupported mirror tree: ${unsupported.show} ".fail
      }
    } yield mirrorMethods

    val m = for {
      (sourceSym, sourceMethods) <- extractMethodTypes(source)
      (mirrorSym, mirrorMethods) <- extractMethodTypes(mirror)

      sourceClassKind = if (sourceSym.flags.is(q.Flags.Module)) q.ClassKind.Object else q.ClassKind.Class

      _ = println(s">>## ${sourceSym.fullName} -> ${mirrorSym.fullName} ")
      mirrorStruct <- Retyper.lowerClassType0(mirrorSym).resolve
      sourceMethodTable = sourceMethods.groupBy(_._1.name)
      _                 = println(s"${sourceMethodTable.mkString("\n\t")}")
      mirrors <- mirrorMethods.traverseFilter { (mirror, mirrorArgs, mirrorRtn) =>
        def fmtName(source: String) =
          s"<$source>${mirror.name}(${mirrorArgs.map(_.show).mkString(",")}):${mirrorRtn.show}"

        println(s"need ${mirror.name}")
        sourceMethodTable.get(mirror.name) match {
          case None => None.success // extra method on mirror
          case Some(xs) => // we got overloads, resolve them
            xs.filter { (_, sourceArgs, sourceRtn) =>
              sourceArgs.lengthCompare(mirrorArgs) == 0 &&
              mirrorArgs.zip(sourceArgs).forall(tpeMatch(_, _)) &&
              tpeMatch(sourceRtn, mirrorRtn)
            } match {
              case (source, _, _) :: Nil =>
                mkMirroredMethods(source, sourceClassKind, mirror, mirrorStruct).map(Some(_)) // single match
              case Nil => s"Overload resolution for ${fmtName(mirrorSym.fullName)} resulted in no match".fail
              case xs =>
                s"Overload resolution for ${fmtName(mirrorSym.fullName)} resulted in multiple matches: $xs".fail
            }
        }
      }
    } yield p.Mirror(p.Sym(sourceSym.fullName), mirrorStruct, mirrors.flatten)

    println(">>>" + m)
    m
  }

}
