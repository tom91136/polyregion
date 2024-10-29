package polyregion.prism

import cats.syntax.all.*
import polyregion.ast.{PolyAST as p, *, given}
import polyregion.scalalang.*
import polyregion.ast.Traversal.*

import scala.annotation.{compileTimeOnly, tailrec}
import scala.quoted.*

@compileTimeOnly("This class only exists at compile-time for internal use")
object compiletime {

  inline def derivePackedMirrors[T <: Tuple](xs: T) = {
    // XXX We are not in a macro context yet  so we don't have a quote context.
    //  Everything we do here must be either a proper term/type and not a quoted q.Expr/q.Type.

    // We first extract the term propositions from the witness.
    val termPrisms = xs.toList.map { case (w: WitnessK[_, _]) => w.unsafePrism }

    // We then extract the type witnesses of the source and mirror type and map them into a tuple again at the type level
    type Witnesses[Xs <: Tuple, Ys <: Tuple] <: Tuple = Xs match {
      case (WitnessK[x, y] *: ts) => Witnesses[ts, (x, y) *: Ys]
      case EmptyTuple             => Ys
    }

    derivePackedTypeMirrors[Witnesses[T, EmptyTuple]](termPrisms)
  }

  private final val TopRefs =
    Set("scala.Any", classOf[java.lang.Object].getName) // classOf[Any].getName gives "java.lang.Object"

  private final val IntrinsicName = polyregion.scalalang.intrinsics.getClass.getName

  private inline def simplifyTpe(using q: Quoted)(t: q.TypeRepr) = t.dealias.simplified.widenTermRefByName

  private inline def derivePackedTypeMirrors[T <: Tuple](
      inline termPrisms: List[TermPrism[Any, Any]]
  ): List[Prism] = ${ derivePackedTypeMirrorsImpl[T]('termPrisms) }

  inline def showTree(inline f: Any): Any =
    ${ showTreeImpl('{ 2 }) }

  def showTreeImpl(f: Expr[Int])(using Q: Quotes): Expr[Int] = {
    import Q.reflect.*
    // f.asTerm.as
    // Expr(f)
    // ???
    f
  }

  private def derivePackedTypeMirrorsImpl[T <: Tuple](using Q: Quotes, t: Type[T])(
      termPrisms: Expr[List[TermPrism[Any, Any]]]
  ): Expr[List[Prism]] = {
    implicit val q = Quoted(Q)

    @tailrec def collectWitnesses[T <: Tuple: Type](
        xs: List[(q.TypeRepr, q.TypeRepr)] = Nil
    ): List[(q.TypeRepr, q.TypeRepr)] =
      Type.of[T] match {
        case '[EmptyTuple]   => xs
        case '[(s, m) *: ts] => collectWitnesses[ts]((q.TypeRepr.of[s], q.TypeRepr.of[m]) :: xs)
      }

    val witnesses = collectWitnesses[T]().map((a, b) => (simplifyTpe(a), simplifyTpe(b)))

    given l: Log = Log("Derive packed mirrors")
    (for {
      typeLUT <- witnesses.traverse { (s, m) =>
        for {
          (_ -> st, _) <- Retyper.typer0(s)
          (_ -> mt, _) <- Retyper.typer0(m)
          st <- st match {
            case p.Type.Struct(sym, _, _, _) =>
              println(s"???> Go ${sym} ==  ${m.show}")
              sym.success
            case bad => s"source class ${s.show} (mirror is ${m.show}) is not a class type, got repr: ${bad.repr}".fail
          }
          mt <- mt match {
            case p.Type.Struct(sym, _, _, _) => sym.success
            case bad => s"mirror class ${m.show} (source is ${s.show}) is not a class type, got repr: ${bad.repr}".fail
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
          val data    = Array.concat(${ Varargs(refs) }*)
          val mirrors = MsgPack.decode[List[p.Mirror]](data).fold(throw _, x => x)
          val terms   = $termPrisms
          assert(terms.size == mirrors.size, "Term prism size and mirror size mismatch")
          mirrors.zip(terms)
        }
        q.Block(vals, decodeExpr.asTerm).asExprOf[List[Prism]]
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

  private def replaceTypes(mirrorToSourceTable: Map[p.Sym, p.Sym])(t: p.Type) = t match {
    case p.Type.Struct(sym, tpeVars, args, parents) =>
      p.Type.Struct(mirrorToSourceTable.getOrElse(sym, sym), tpeVars, args, parents) match {
        case p.Type.Struct(Symbols.ArrayMirror, _, x :: Nil, _) =>
          // XXX restore @scala.Array back to the proper array type if needed
          p.Type.Ptr(x, None, p.Type.Space.Global)
        case x => x
      }
    case x => x
  }

  private def mirrorMethod(using q: Quoted, sink: Log)   //
  (sourceMethodSym: q.Symbol, mirrorMethodSym: q.Symbol) //
  (mirrorToSourceTable: Map[p.Sym, p.Sym]): Result[(p.Function, q.Dependencies)] = for {

    log <- sink.subLog(s"Mirror for ${sourceMethodSym} -> ${mirrorMethodSym}").success
    _ = println(s"Do ${sourceMethodSym} -> ${mirrorMethodSym}")

    sourceSignature <- sourceMethodSym.tree match {
      case d: q.DefDef => polyregion.scalalang.Compiler.deriveSignature(d)
      case unsupported => s"Unsupported source tree: ${unsupported.show} ".fail
    }
    _ = println(s"SIG=${sourceSignature.repr}")

    mirrorMethods <- mirrorMethodSym.tree match {
      case d: q.DefDef =>
        println(d.show)
        for {

          (fn, fnDeps) <- polyregion.scalalang.Compiler.compileFn(log, d, Map.empty, intrinsify = true)

          rewrittenMirror =
            fn.copy(
              name = sourceSignature.name,
              // Get rid of the intrinsic$ capture introduced by calling stubs in that object.
              moduleCaptures = fn.moduleCaptures.filter(_.named.tpe match {
                case p.Type.Struct(sym, _, _, _) => sym != p.Sym(IntrinsicName)
                case _                           => true
              })
//              receiver = sourceSignature.receiver.map(p.Named("this", _)),
//              tpeVars = fn.tpeVars
//                        ++ fn.receiver.map(_.tpe).fold(Nil){
//                case p.Type.Struct(_, vars, _) => vars
//                case _ => Nil
//              }

            ).modifyAll[p.Type](_.mapNode(replaceTypes(mirrorToSourceTable)(_)))

//          _ = pprint.pprintln(rewrittenMirror)

//          rewrittenMirror2 = rewrittenMirror.copy(
//              body = rewrittenMirror.body.flatMap(_.mapTerm({
//
//                case s @ p.Term.Select(Nil, p.Named("this", tpe))    if fn.receiver.exists(_.tpe == tpe) =>
//
//                  sourceSignature.receiver match {
//                    case Some(x) => p.Term.Select(Nil, p.Named("this", x))
//                    case None    => ??? // replacement doesn't have a receiver!?
//                  }
//
//                case s @ p.Term.Select(p.Named("this", tpe) :: xs, y)  => // if fn.receiver.exists(_.tpe == tpe) =>
//
////                  if fn.receiver.exists(_.tpe == tpe)
//
//
//                  // println(fn.receiver)
//                  // println(tpe)
//                  // if(sourceSignature.name.toString.contains("length"))
//                  //   ???
//
//                  sourceSignature.receiver match {
//                    case Some(x) => p.Term.Select(p.Named("this", x) :: xs, y)
//                    case None    => ??? // replacement doesn't have a receiver!?
//                  }
//                case x =>                   x
//              }))
//            )

        } yield rewrittenMirror -> fnDeps

      case unsupported => s"Unsupported mirror tree: ${unsupported.show} ".fail
    }
  } yield mirrorMethods

  private def derivePackedMirrorsImpl(using q: Quoted, sink: Log) //
  (source: q.TypeRepr, mirror: q.TypeRepr)                        //
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
                .modifyAll[p.Type](_.mapNode(replaceTypes(mirrorToSourceTable)(_)))
                .modifyAll[p.Type] {
                  case s: p.Type.Struct => s.copy(parents = Nil)
                  case x                => x
                }

              sourceSigs <- sources.traverse((sourceSym, sourceDef) =>
                Compiler.deriveSignature(sourceDef).map(sourceSym -> _)
              )

              r <- sourceSigs.filter { case (_, sourceSig) =>
                // FIXME Removing the receiver for now because it may be impossible to mirror a private type.
                //  Doing this may make the overload resolution less accurate.
                //  We also discard the generic types here, to handle cases like Seq[A] =:= SeqOp[A, CC[_], C]

                sourceSig
                  .copy(receiver = None, tpeVars = Nil)
                  .modifyAll[p.Type] {
                    case s: p.Type.Struct => s.copy(parents = Nil)
                    case x                => x
                  } ==
                  reflectedSigWithSourceTpes.copy(receiver = None, tpeVars = Nil)
              } match {
                case sourceMirror :: Nil => // single match
                  mirrorMethod(sourceMirror._1, reflectedSym)(mirrorToSourceTable).map((f, deps) =>
                    (f :: Nil, deps) :: Nil
                  )
                case Nil =>
                  val considered = sourceSigs.map("\t(no-match)=>" + _._2.repr).mkString("\n")

                  println(sourceSigs(0)._2.copy(receiver = None))
                  println(reflectedSigWithSourceTpes.copy(receiver = None).copy(receiver = None))

                  (s"Overload resolution for ${reflectedSym} with resulted in no match, the following signatures were considered (mirror is the requirement):" +
                    s"\n\t(mirror)    ${reflectedSig.repr}" +
                    s"\n\t(reflected) ${reflectedSigWithSourceTpes.repr}" +
                    s"\n${considered}" +
                    s"\n == Available struct mappings == " +
                    s"\n${mirrorToSourceTable.map((m, s) => s"${m.repr} => ${s.repr}").mkString("\n")}").fail
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
      // _ <-
      //   if (deps.functions.nonEmpty)
      //     (s"${sourceSym.fullName} -> ${mirrorSym.fullName} contains call to dependent functions: ${deps.functions.map(_._1.symbol)}, " +
      //       s"this is not allowed and dependent methods should be marked inline to avoid this").fail
      //   else ().success
      // deps.classes
      // deps.modules
      _ = println("Dependent Classes = " + deps.classes.keys.toList)
      _ = println("Dependent Modules = " + deps.modules.keys.map(_.fullName).toList)

      // FIXME we're creating a dummy function so that the replacement works,
      //  shouldn't have to do this really.
      (_, _, dependentStructs, _) <-
        Compiler.compileAndReplaceStructDependencies(
          sink,
          p.Function(p.Sym("_dummy_"), Nil, None, Nil, Nil, Nil, p.Type.Nothing, Nil, p.Function.Kind.Exported),
          deps
        )(Map.empty)

      _ = println(sink.render(1).mkString("\n"))

      parents = source.baseClasses.map(c => p.Sym(c.fullName))

    } yield p.Mirror(
      source = p.Sym(sourceSym.fullName),
      sourceParents = parents,
      struct = mirrorStruct,
      functions = functions,
      dependencies = dependentStructs.toList
    )

    println(">>>" + m)
    m
  }

}
