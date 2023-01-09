package polyregion.scalalang
import polyregion.ast.{PolyAst as p, *}

object SymbolCache {

  val StructDefTable = collection.mutable.Map.empty[p.Sym, (p.StructDef, polyregion.jvm.compiler.Layout)]

}
