package polyregion.prism

import _root_.scala.annotation.compileTimeOnly
import _root_.scala.quoted.*

@compileTimeOnly("This class only exists at compile-time to for internal use")
object compiletime {

  inline def unsupported : Nothing = ???

  inline def checkPrism[S, M]: Unit = ${ checkPrismImpl[S, M] }

  def checkPrismImpl[S: Type, M: Type](using q: Quotes): Expr[Unit] = {
    import quotes.reflect.*
    given Printer[Tree] = Printer.TreeStructure
//    pprint.pprintln(x.asTerm) // term => AST

    val sourceTpe = TypeRepr.of[S]
    val mirrorTpe = TypeRepr.of[M]

    (sourceTpe.classSymbol, mirrorTpe.classSymbol) match {
      case (None, None)    => report.error(s"No Class symbol for both $sourceTpe and $mirrorTpe ")
      case (Some(_), None) => report.error(s"No Class symbol for $sourceTpe")
      case (None, Some(_)) => report.error(s"No Class symbol for $mirrorTpe")
      case (Some(sourceSym), Some(mirrorSym)) =>
        println(sourceSym.declaredMethods)
        println(mirrorSym.declaredMethods)

        println(mirrorSym.allOverriddenSymbols.toList)
        println(mirrorSym.declaredTypes)
    }

    // imp[Range, scala.Range]
    // imp[RichInt, scala.RichInt]
    // class RichInt{
      // def rr : Range  = ???
    //}

    // impl
    // def prism
    // 
    // case class Mirrored[S, M](struct: p.StructDef, methods : List[p.Function])
    // def makeMirrored[S, M] : Mirrored[S, M]
    // def findMirrorOfType(s : Symbol) = summon[...] 

    '{ () }
  }

}
