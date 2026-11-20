package polyregion

import scala.quoted.*

object InterfaceSurfaceCheck {
  inline def erase(inline value: Any): Unit = ${ eraseImpl('value) }

  private def eraseImpl(value: Expr[Any])(using Quotes): Expr[Unit] = '{ () }
}
