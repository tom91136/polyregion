package polyregion.prism

import scala.annotation.compileTimeOnly
import scala.quoted.*
import polyregion.scala.*
import cats.syntax.all.*
import scala.compiletime.erasedValue
import polyregion.ast.{PolyAst as p, *}
import polyregion.scala.Retyper.typer

@compileTimeOnly("This class only exists at compile-time to for internal use")
object compiletime {

  inline def unsupported: Nothing = ???

  trait SignatureWitness[S, M]

  inline def reflectAndDerivePackedMirrors[T <: Tuple]: Map[p.Sym, p.Mirror] = inline erasedValue[T] match {
    case _: EmptyTuple => Map.empty
    case _: ((s, m) *: ts) =>
      given SignatureWitness[s, m] = validateSignature[s, m]
      val mirror                   = derivePackedMirror[s, m]
      reflectAndDerivePackedMirrors[ts] + (mirror.source -> mirror)
  }

  inline def derivePackedMirror[S, M]: p.Mirror = ${ derivePackedMirrorImpl[S, M] }

  def derivePackedMirrorImpl[S: Type, M: Type](using q: Quotes): Expr[p.Mirror] = {
    implicit val Q = Quoted(q)
    val mirrorTpe  = Q.TypeRepr.of[M]
    val sourceTpe  = Q.TypeRepr.of[S]
    (for {
      sym  <- mirrorTpe.classSymbol.failIfEmpty(s"No class symbol for ${mirrorTpe}")
      sdef <- Retyper.lowerClassType(sym).resolve
      fns <- sym.declaredMethods.map(_.tree).traverse {
        case d: Q.DefDef => polyregion.scala.Compiler.compileFn(d)
        case unsupported => s"Unsupported mirror tree: ${unsupported.show} ".fail
      }
    } yield p.Mirror(p.Sym(sourceTpe.typeSymbol.fullName), sdef, fns.map(_._2))) match {
      case Left(e)   => throw e
      case Right(xs) => '{ MsgPack.decode[p.Mirror](${ Expr(MsgPack.encode(xs)) }).fold(throw _, identity) }
    }
  }

  inline def validateSignature[S, M]: SignatureWitness[S, M] = ${ validateSignatureImpl[S, M] }

  def validateSignatureImpl[S: Type, M: Type](using q: Quotes): Expr[SignatureWitness[S, M]] = {
    import quotes.reflect.*
    given Printer[Tree] = Printer.TreeStructure
//    pprint.pprintln(x.asTerm) // term => AST

    val sourceTpe = TypeRepr.of[S]
    val mirrorTpe = TypeRepr.of[M]

    def simplifyTpe(t: TypeRepr) = t.dealias.simplified.widenTermRefByName

    def extractMethodTypes(clsTpe: TypeRepr): Result[List[(String, List[TypeRepr], TypeRepr)]] =
      clsTpe.classSymbol.failIfEmpty(s"No class symbol for ${clsTpe}").map { sym =>
        sym.declaredMethods
          .filter { m =>
            val flags = m.flags
          flags.is(Flags.Method) &&
          !(flags.is(Flags.Private) || flags.is(Flags.Protected) || flags.is(Flags.Synthetic))
          }
          .map { m =>
            val tpe = clsTpe.memberType(m).dealias.simplified.widenTermRefByName
            m -> tpe
          }
          .collect { case (s, MethodType(_, argTpes, rtnTpe)) =>
            (s.name, argTpes.map(simplifyTpe(_)), simplifyTpe(rtnTpe))
          }
      }

    for {
      sourceMethods <- extractMethodTypes(sourceTpe)
      mirrorMethods <- extractMethodTypes(mirrorTpe)

      mirrorMethodTable = mirrorMethods.groupBy(_._1)

    } yield sourceMethods.map { (name, args, rtn) =>

      def hasMatch(source: TypeRepr, mirror: TypeRepr) =
        source =:= mirror || (Implicits.search(TypeRepr.of[SignatureWitness].appliedTo(source :: mirror :: Nil)) match {
          case cc: ImplicitSearchSuccess =>
            println(s"\tC=${source} == ${mirror} t=${cc.tree}")
            true
          case _ => false
        })

      val found = mirrorMethodTable.get(name) match {
        case None => false
        case Some(xs) =>
          xs.exists { (_, mirrorArgs, mirrorRtn) =>
            mirrorArgs.lengthCompare(args) == 0 &&
            args.zip(mirrorArgs).forall(hasMatch(_, _)) &&
            hasMatch(rtn, mirrorRtn)
          }
      }
      println(s"def ${name}(${args.map(_.show).mkString(", ")}) : ${rtn.show} // =  found=${found}")

    }

    '{ new SignatureWitness {} }
  }

}
