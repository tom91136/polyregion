package hps

import org.openjdk.jol.info.{ClassLayout, GraphLayout}

object LayoutExperiments {

  def main(args: Array[String]): Unit = {

    case class Atom(x: Float, y: Float, z: Float, tpe: Int)
    case class Atom2(x: Float)

    println(ClassLayout.parseInstance(Atom(42, 43, 44, 120)).toPrintable)

    println(GraphLayout.parseInstance(Atom(42, 43, 44, 120)).toPrintable)
    println(GraphLayout.parseInstance(Atom2(42)).toPrintable)

  }

}
