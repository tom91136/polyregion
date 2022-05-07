package polyregion.prism

import cats.syntax.all.*
import polyregion.ast.{PolyAst as p, *}
import polyregion.scala.*

import _root_.scala.annotation.{compileTimeOnly, tailrec}
import _root_.scala.compiletime.{erasedValue, summonInline}
import _root_.scala.quoted.*
import _root_.scala.reflect.ClassTag

@compileTimeOnly("This class only exists at compile-time to for internal use")
object compiletime {

  private final val TopRefs =
    Set("scala.Any", classOf[java.lang.Object].getName) // classOf[Any].getName gives "java.lang.Object"

  private def simplifyTpe(using q: Quoted)(t: q.TypeRepr) = t.dealias.simplified.widenTermRefByName

  inline def derivePackedMirrors1[T <: Tuple]: Map[p.Sym, p.Mirror] = ${ derivePackedMirrors1Impl[T] }

  private def derivePackedMirrors1Impl[T <: Tuple](using Q: Quotes, t: Type[T]): Expr[Map[p.Sym, p.Mirror]] = {
    implicit val q = Quoted(Q)
    val witnesses  = collectWitnesses[T]().map((a, b) => (simplifyTpe(a), simplifyTpe(b)))

    (for {
      typeLUT <- witnesses.traverse { (s, m) =>
        for {
          (_, st) <- Retyper.typer0(s)
          (_, mt) <- Retyper.typer0(m)
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
      typeTable = typeLUT.toMap
      data <- witnesses.traverse(derivePackedMirrorsImpl(_, _) { (source, mirror) =>
        (source =:= mirror) ||
        (witnesses.filter((k, v) => k =:= source && v =:= mirror) match {
          case _ :: Nil => true
          case _        => false
        })
      }(typeTable))
    } yield data) match {
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
  (typeEq: (q.TypeRepr, q.TypeRepr) => Boolean)        //
  (typeLUT: Map[p.Sym, p.Sym]): Result[p.Mirror] = {

    case class ReflectedMethodTpe(symbol: q.Symbol, args: List[(String, q.TypeRepr)], rtn: q.TypeRepr) {
      def =:=(that: ReflectedMethodTpe): Boolean =
        args.lengthCompare(that.args) == 0 &&
          args.zip(that.args).forall { case ((lName, lTpe), (rName, rTpe)) =>
            lName == rName && typeEq(lTpe, rTpe)
          } &&
          typeEq(rtn, that.rtn)
      def show: String = s"${symbol.name}(${args.map((name, tpe) => s"$name:${tpe.show}").mkString(",")}):${rtn.show}"
    }

    def extractMethodTypes(clsTpe: q.TypeRepr): Result[(q.Symbol, List[ReflectedMethodTpe])] =
      clsTpe.classSymbol.failIfEmpty(s"No class symbol for ${clsTpe}").map { sym =>
        sym -> sym.methodMembers
          .filter { m =>
            val flags = m.flags
            flags.is(q.Flags.Method) && !(
              flags.is(q.Flags.Private) || flags.is(q.Flags.Protected) || flags.is(q.Flags.Synthetic)
            )
          }
          .filterNot(x => TopRefs.contains(x.owner.fullName))
          .map(m => m -> simplifyTpe(clsTpe.memberType(m)))
          .collect { case (m, q.MethodType(argNames, argTpes, rtnTpe)) =>
            ReflectedMethodTpe(m, argNames.zip(argTpes.map(simplifyTpe(_))), simplifyTpe(rtnTpe))
          }
      }

    def mirrorMethod(sourceMethodSym: q.Symbol, mirrorMethodSym: q.Symbol): Result[(p.Function, q.Dependencies)] = for {

      log <- Log(s"Mirror for ${sourceMethodSym} -> ${mirrorMethodSym}")
      _ = println(s"Do ${sourceMethodSym}")

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
                  case p.Type.Struct(sym, tpeVars, args) => p.Type.Struct(typeLUT.getOrElse(sym, sym), tpeVars, args)
                  case x                                 => x
                })
          } yield rewrittenMirror -> fnDeps

        case unsupported => s"Unsupported mirror tree: ${unsupported.show} ".fail
      }
    } yield mirrorMethods

    val m = for {
      (sourceSym, sourceMethods) <- extractMethodTypes(source)
      (mirrorSym, mirrorMethods) <- extractMethodTypes(mirror)

      _ = println(s">>## ${sourceSym.fullName} -> ${mirrorSym.fullName} ")
      mirrorStruct <- Retyper.structDef0(mirrorSym)
      sourceMethodTable = sourceMethods.groupBy(_.symbol.name)
      _                 = println(s"${sourceMethodTable.mkString("\n\t")}")
      mirroredMethods <- mirrorMethods.flatTraverse { reflectedMirror =>
//        def fmtName(source: String) =
//          s"<$source>${mirror.name}(${mirrorArgs.map((name, tpe) => s"$name:${tpe.show}").mkString(",")}):${mirrorRtn.show}"

        sourceMethodTable.get(reflectedMirror.symbol.name) match {
          case None => Nil.success // extra method on mirror
          case Some(xs) => // we got overloads, resolve them
            xs.filter(_ =:= reflectedMirror) match {
              case sourceMirror :: Nil => // single match
                mirrorMethod(sourceMirror.symbol, reflectedMirror.symbol).map((f, deps) => (f :: Nil, deps) :: Nil)
              case Nil =>
                s"Overload resolution for <${mirrorSym.fullName}>${reflectedMirror.show} with named arguments resulted in no match, the following methods were considered:\n\t${xs
                  .map("\t" + _.show)
                  .mkString("\n")}".fail
              case xs =>
                s"Overload resolution for <${mirrorSym.fullName}>${reflectedMirror.show} resulted in multiple matches: $xs".fail
            }
        }
      }
      // We reject any mirror that require dependent functions as those would have to go through mirror substitution first,
      // thus making this process recursive, which is quite error prone for in a macro context.
      // The workaround is to mark dependent functions as inline in the mirror.
      (functions, deps) = mirroredMethods.combineAll
      _ <-
        if (deps.functions.nonEmpty)
          (s"$sourceSym -> $mirrorSym contains call to dependent functions: ${deps.functions.map(_._1.symbol)}, " +
            s"this is not allowed and dependent methods should be marked inline to avoid this").fail
        else ().success
      // deps.classes
      // deps.modules
      _ = println(deps.classes.keys.toList)
      _ = println(deps.modules.keys.map(_.fullName).toList)
      sdefs <- Compiler.deriveStructDefs(deps)
    } yield p.Mirror(source = p.Sym(sourceSym.fullName), struct = mirrorStruct, functions = functions, sdefs.map(_._1))

    println(">>>" + m)
    m
  }

}
