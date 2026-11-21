package polyregion.scalalang

import polyregion.jvm.runtime as rt

object launch {
  inline def call[R](inline identity: String, inline context: rt.Context, inline args: Any*): R =
    ${ LaunchMacro.generate[R]('identity, 'context, 'args) }
}
