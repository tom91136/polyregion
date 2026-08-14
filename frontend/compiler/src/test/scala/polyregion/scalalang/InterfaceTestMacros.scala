package polyregion.scalalang

import scala.quoted.*

object InterfaceTestMacros {
  inline def interfaceIdentityOf[T]: Option[(String, String)] = ${ identityOfImpl[T] }

  private def identityOfImpl[T: Type](using quotes: Quotes): Expr[Option[(String, String)]] = {
    val q = Quoted(quotes)
    import q.underlying.reflect.*
    val method = TypeRepr.of[T].typeSymbol.methodMember("increment").head
    Expr(Compiler.interfaceIdentity(using q)(method))
  }
}
