package polyregion.scalalang

import polyregion.intrinsic.Key

object JvmDescriptor {

  private val Primitives = Map(
    "scala.Boolean" -> "Z",
    "scala.Byte"    -> "B",
    "scala.Short"   -> "S",
    "scala.Char"    -> "C",
    "scala.Int"     -> "I",
    "scala.Long"    -> "J",
    "scala.Float"   -> "F",
    "scala.Double"  -> "D",
    "scala.Unit"    -> "V"
  )

  private def erased(sig: String): String =
    if (sig.endsWith("[]")) "[" + erased(sig.dropRight(2))
    else Primitives.getOrElse(sig, s"L${sig.replace('.', '/')};")

  private def internalName(using q: Quoted)(cls: q.Symbol): String = {
    val owner = cls.owner
    if (owner.isPackageDef) {
      val pkg = owner.fullName
      if (pkg == "<root>" || pkg == "<empty>") cls.name else s"${pkg.replace('.', '/')}/${cls.name}"
    } else internalName(owner) + "$" + cls.name
  }

  def keyOf(using q: Quoted)(sym: q.Symbol): Option[Key] = {
    val cls = sym.owner
    Option
      .when(sym.isDefDef && !cls.isNoSymbol && cls.isClassDef) {
        val sig    = sym.signature
        val sigs   = sig.paramSigs
        val params = sigs.collect { case s: String => s }
        Option.when(params.size == sigs.size) {
          val owner = internalName(cls)
          Key(
            if (sym.flags.is(q.Flags.JavaStatic)) owner.stripSuffix("$") else owner,
            sym.name,
            s"(${params.map(erased).mkString})${erased(sig.resultSig)}"
          )
        }
      }
      .flatten
  }
}
