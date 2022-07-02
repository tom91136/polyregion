package polyregion.prism

import cats.syntax.all.*
import polyregion.ast.{PolyAst as p, *}
import polyregion.scala.*

import _root_.scala.annotation.{compileTimeOnly, tailrec}
import _root_.scala.quoted.*

@compileTimeOnly("This class only exists at compile-time for internal use")
object compiletime {

  private final val TopRefs =
    Set("scala.Any", classOf[java.lang.Object].getName) // classOf[Any].getName gives "java.lang.Object"

  private def simplifyTpe(using q: Quoted)(t: q.TypeRepr) = t.dealias.simplified.widenTermRefByName

  @tailrec private def collectWitnesses[T <: Tuple](using q: Quoted, t: Type[T])(
      xs: List[(q.TypeRepr, q.TypeRepr)] = Nil
  ): List[(q.TypeRepr, q.TypeRepr)] = {
    given Quotes = q.underlying
    t match {
      case '[EmptyTuple]   => xs
      case '[(s, m) *: ts] => collectWitnesses[ts]((q.TypeRepr.of[s], q.TypeRepr.of[m]) :: xs)
    }
  }

  inline def derivePackedMirrors1[T <: Tuple]: Map[p.Sym, p.Mirror] = ${ derivePackedMirrors1Impl[T] }
  private def derivePackedMirrors1Impl[T <: Tuple](using Q: Quotes, t: Type[T]): Expr[Map[p.Sym, p.Mirror]] = {
    implicit val q = Quoted(Q)
    val witnesses  = collectWitnesses[T]().map((a, b) => (simplifyTpe(a), simplifyTpe(b)))
    (for {
      typeLUT <- witnesses.traverse { (s, m) =>
        for {
          (_ -> st, _) <- Retyper.typer0(s)
          (_ -> mt, _) <- Retyper.typer0(m)
          st <- st match {
            case p.Type.Struct(sym, _, _) => sym.success
            case bad                      => s"source class $s is not a class type, got $bad".fail
          }
          mt <- mt match {
            case p.Type.Struct(sym, _, _) => sym.success
            case bad                      => s"mirror class $m is not a class type, got $bad".fail
          }
        } yield mt -> st
      }
      mirrorToSourceTable = typeLUT.toMap
      data <- witnesses.traverse(derivePackedMirrorsImpl(_, _)(mirrorToSourceTable))
    } yield data) match {
      case Left(e)                   => throw e
      case Right(xs: List[p.Mirror]) =>
        // XXX JVM bytecode limit is 64k per method, make sure we're under that by using 1k per constant,
        // we also generate lazy vals (could be defs if this becomes a problem) which transforms to a separate synthetic method.
        val packs = MsgPack.encode(xs).grouped(1024).toList
        println(s"packs = ${packs.size}")
        type PickleType = Array[Byte]
        val (vals, refs) = packs.zipWithIndex.map { (pack, i) =>
          val symbol =
            q.Symbol.newVal(q.Symbol.spliceOwner, s"pack$i", q.TypeRepr.of[PickleType], q.Flags.Lazy, q.Symbol.noSymbol)
          (
            q.ValDef(symbol, Some(Expr(pack).asTerm)),
            q.Ref(symbol).asExprOf[PickleType]
          )
        }.unzip
        val decodeExpr = '{
          val data = Array.concat(${ Varargs(refs) }*)
          MsgPack.decode[List[p.Mirror]](data).fold(throw _, x => x).map(m => m.source -> m).toMap
        }
        q.Block(vals, decodeExpr.asTerm).asExprOf[Map[p.Sym, p.Mirror]]
    }
  }

  private def extractMethodTypes(using q: Quoted) //
  (clsTpe: q.TypeRepr): Result[(q.Symbol, List[(q.Symbol, q.DefDef)])] =
    clsTpe.classSymbol.failIfEmpty(s"No class symbol for ${clsTpe}").flatMap { sym =>
      println(s"$sym ${sym.methodMembers.toList}")
      sym.methodMembers
        .filter { m =>
          val flags = m.flags
          flags.is(q.Flags.Method) && !(
            flags.is(q.Flags.Private) || flags.is(q.Flags.Protected) || flags.is(q.Flags.Synthetic)
          )
        }
        .filterNot(x => TopRefs.contains(x.owner.fullName))
        .traverseFilter { s =>
          s.tree match {
            case d: q.DefDef => Some(s -> d).success
            case _           => None.success
          }
        }
        .map(sym -> _)
    }

  private def mirrorMethod(using q: Quoted)              //
  (sourceMethodSym: q.Symbol, mirrorMethodSym: q.Symbol) //
  (mirrorToSourceTable: Map[p.Sym, p.Sym]): Result[(p.Function, q.Dependencies)] = for {

    log <- Log(s"Mirror for ${sourceMethodSym} -> ${mirrorMethodSym}")
    _ = println(s"Do ${sourceMethodSym} -> ${mirrorMethodSym}")

    sourceSignature <- sourceMethodSym.tree match {
      case d: q.DefDef => polyregion.scala.Compiler.deriveSignature(d)
      case unsupported => s"Unsupported source tree: ${unsupported.show} ".fail
    }
    _ = println(s"SIG=$sourceSignature")

    mirrorMethods <- mirrorMethodSym.tree match {
      case d: q.DefDef =>
        println(d.show)
        for {

          ((fn, fnDeps), fnLog) <- polyregion.scala.Compiler.compileFn(d)

          rewrittenMirror =
            fn.copy(name = sourceSignature.name, receiver = sourceSignature.receiver.map(p.Named("this", _)))
              .mapType(_.map {
                case p.Type.Struct(sym, tpeVars, args) =>
                  p.Type.Struct(mirrorToSourceTable.getOrElse(sym, sym), tpeVars, args)
                case x => x
              })
        } yield rewrittenMirror -> fnDeps

      case unsupported => s"Unsupported mirror tree: ${unsupported.show} ".fail
    }
  } yield mirrorMethods

  private def derivePackedMirrorsImpl(using q: Quoted) //
  (source: q.TypeRepr, mirror: q.TypeRepr)             //
  (mirrorToSourceTable: Map[p.Sym, p.Sym]): Result[p.Mirror] = {

    val m = for {
      (sourceSym, sourceMethods) <- extractMethodTypes(source)
      (mirrorSym, mirrorMethods) <- extractMethodTypes(mirror)

      _ = println(
        s">>## ${sourceSym.fullName}(m=${sourceMethods.size})  -> ${mirrorSym.fullName}(m=${mirrorMethods.size}) "
      )
      mirrorStruct <- Retyper.structDef0(mirrorSym)
      sourceMethodTable = sourceMethods.groupBy(_._1.name)
      _                 = println(s"Source Symbols:\n${sourceMethodTable.mkString("\n\t")}")
      mirroredMethods <- mirrorMethods.flatTraverse { case (reflectedSym, reflectedDef) =>
        sourceMethodTable.get(reflectedSym.name) match {
          case None => // Extra method on mirror, check that it's private.
            if (reflectedSym.flags.is(q.Flags.Private)) Nil.success
            else
              s"Method ${reflectedSym} from ${mirrorSym} does not exists in source (${sourceSym}) and is not marked private".fail
          case Some(sources) => // We got matches with potential overloads, resolve them.
            for {
              sourceName <- sources
                .map(_._1.fullName)
                .distinct
                .failIfNotSingleton(
                  s"Expected all method name matching ${reflectedSym.name} from ${sourceSym} to be the same"
                )
              reflectedSig <- Compiler.deriveSignature(reflectedDef)
              // Map the mirrored method to use the source types first, otherwise we're comparing different classes entirely.
              reflectedSigWithSourceTpes = reflectedSig
                .copy(name = p.Sym(sourceName)) // We also replace the name to use the source ones.
                .mapType(_.map {
                  case p.Type.Struct(sym, tpeVars, args) =>
                    p.Type.Struct(mirrorToSourceTable.getOrElse(sym, sym), tpeVars, args)
                  case x => x
                })

              sourceSigs <- sources.traverse((sourceSym, sourceDef) =>
                Compiler.deriveSignature(sourceDef).map(sourceSym -> _)
              )

              r <- sourceSigs.filter { case (_, sourceSig) =>
                // FIXME Removing the receiver for now because it may be impossible to mirror a private type.
                // Doing this may make the overload resolution less accurate.
                sourceSig.copy(receiver = None) == reflectedSigWithSourceTpes.copy(receiver = None)
              } match {
                case sourceMirror :: Nil => // single match
                  mirrorMethod(sourceMirror._1, reflectedSym)(mirrorToSourceTable).map((f, deps) =>
                    (f :: Nil, deps) :: Nil
                  )
                case Nil =>
                  val considered = sourceSigs.map("\t(failed)    " + _._2.repr).mkString("\n")
                  (s"Overload resolution for ${reflectedSym} with resulted in no match, the following signatures were considered (mirror is the requirement):" +
                    s"\n\t(mirror)    ${reflectedSig.repr}" +
                    s"\n\t(reflected) ${reflectedSigWithSourceTpes.repr}" +
                    s"\n${considered}").fail
                case xs =>
                  s"Overload resolution for ${reflectedSym} resulted in multiple matches (the program should not compile): $xs".fail
              }
            } yield r

        }
      }
      // We reject any mirror that require dependent functions as those would have to go through mirror substitution first,
      // thus making this process recursive, which is quite error prone in a macro context.
      // The workaround is to mark dependent functions as inline in the mirror.
      (functions, deps) = mirroredMethods.combineAll
      _ <-
        if (deps.functions.nonEmpty)
          (s"${sourceSym.fullName} -> ${mirrorSym.fullName} contains call to dependent functions: ${deps.functions.map(_._1.symbol)}, " +
            s"this is not allowed and dependent methods should be marked inline to avoid this").fail
        else ().success
      // deps.classes
      // deps.modules
      _ = println("Dependent Classes = " + deps.classes.keys.toList)
      _ = println("Dependent Modules = " + deps.modules.keys.map(_.fullName).toList)
      (sdefs, sdefLogs) <- Compiler.deriveAllStructs(deps)()
    } yield p.Mirror(source = p.Sym(sourceSym.fullName), struct = mirrorStruct, functions = functions, sdefs)

    println(">>>" + m)
    m
  }

}
