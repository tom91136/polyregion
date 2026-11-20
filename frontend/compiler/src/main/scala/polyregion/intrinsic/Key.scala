package polyregion.intrinsic

import cats.syntax.all.*
import polyregion.ast.PolyAST as p

final case class Key(owner: String, name: String, descriptor: String) {
  def repr: String = s"$owner.$name$descriptor"

  def parameters: Option[List[p.Type]] = Descriptor.parse(descriptor).map(_._1)
  def result: Option[p.Type]           = Descriptor.parse(descriptor).map(_._2)

  def functionName: p.Sym = {
    val mangled = descriptor.collect { case c if c.isLetterOrDigit => c; case ')' => '_' }
    p.Sym(owner.split('/').toList :+ s"${name}_$mangled")
  }
}

object Key {
  def parse(repr: String): Key = {
    val open = repr.indexOf('(')
    val dot  = repr.lastIndexOf('.', open)
    Key(repr.substring(0, dot), repr.substring(dot + 1, open), repr.substring(open))
  }
}

object Descriptor {

  private val Primitives: List[(Char, p.Type)] = List(
    'Z' -> p.Type.Bool1,
    'B' -> p.Type.IntS8,
    'S' -> p.Type.IntS16,
    'C' -> p.Type.IntU16,
    'I' -> p.Type.IntS32,
    'J' -> p.Type.IntS64,
    'F' -> p.Type.Float32,
    'D' -> p.Type.Float64,
    'V' -> p.Type.Unit0
  )

  def letter(t: p.Type): Option[Char] = Primitives.collectFirst { case (c, `t`) => c }
  def tpe(c: Char): Option[p.Type]    = Primitives.collectFirst { case (`c`, t) => t }

  def parse(descriptor: String): Option[(List[p.Type], p.Type)] =
    descriptor.indexOf(')') match {
      case close if descriptor.startsWith("(") && close > 0 && close == descriptor.length - 2 =>
        val args = descriptor.substring(1, close).toList.map(tpe)
        tpe(descriptor.last).filter(_ => args.forall(_.exists(_ != p.Type.Unit0))).map(args.flatten -> _)
      case _ => None
    }

  def render(args: List[p.Type], rtn: p.Type): Option[String] =
    for {
      as <- args.traverse(letter)
      r  <- letter(rtn)
    } yield s"(${as.mkString})$r"
}
