package polyregion.javalang

import polyregion.ast.PolyAST as p
import polyregion.intrinsic.{IntrinsicTable, Key}

object Natives {

  def keyOf(ref: MethodRef): Key = Key(ref.owner.replace('.', '/'), ref.name, ref.desc)

  def covered(ref: MethodRef): Boolean = IntrinsicTable.CoveredOwners.contains(keyOf(ref).owner)

  // the table answers first, a miss goes on to the bytecode tier
  def resolve(ref: MethodRef, args: List[p.Term]): Option[IntrinsicTable.Resolved] =
    IntrinsicTable.resolve(keyOf(ref), args)
}
