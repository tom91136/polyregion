package polyregion.javalang

import polyregion.ast.{PolyAST as p, *}
import polyregion.intrinsic.{IntrinsicTable, Key}

// the descriptor-keyed table answers first, the bytecode tier only sees what the table does not own
object Natives {

  def keyOf(ref: MethodRef): Key = Key(ref.owner.replace('.', '/'), ref.name, ref.desc)

  def covered(ref: MethodRef): Boolean = IntrinsicTable.CoveredOwners.contains(keyOf(ref).owner)

  // None sends the callee on to the bytecode tier, a miss in a covered owner is refused rather than lowered from a native body
  def resolve(ref: MethodRef, args: List[p.Term]): Option[Result[IntrinsicTable.Resolved]] = {
    val key = keyOf(ref)
    IntrinsicTable.resolve(key, args) match {
      case Some(resolved) => Some(Right(resolved))
      case None if IntrinsicTable.CoveredOwners.contains(key.owner) =>
        Some(s"unsupported JVM intrinsic ${key.repr}".fail)
      case None => None
    }
  }
}
