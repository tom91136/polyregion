package polyregion.ast

package object pass {

  trait ProgramPass     extends ((PolyAst.Program, Log) => PolyAst.Program)
  trait BoundaryPass[A] extends ((PolyAst.Program, Log) => (A, PolyAst.Program))

  def printPass(pass: ProgramPass): ProgramPass = { (p: PolyAst.Program, l: Log) =>
    val r  = pass(p, l)
    val sl = l.subLog(s"[${pass.getClass.getName}]")
    sl.info("Structs", r.defs.map(_.repr)*)
    sl.info("Fns", r.functions.map(_.repr)*)
    r
  }

}
