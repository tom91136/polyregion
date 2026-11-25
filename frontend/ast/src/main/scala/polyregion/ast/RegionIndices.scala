package polyregion.ast

object RegionIndices {

  final case class Axis(quotient: List[Int], modulus: Option[Int])

  def extents(rank: Int): List[String] = List.tabulate(rank)(axis => s"n$axis")

  def axes(rank: Int): List[Axis] =
    List.tabulate(rank) { axis =>
      Axis((axis + 1 until rank).toList, Option.when(axis > 0)(axis))
    }
}
