package polyregion.ast

package object pass {

  trait ProgramPass     extends ((PolyAST.Program, Log) => PolyAST.Program)
  trait BoundaryPass[A] extends ((PolyAST.Program, Log) => (A, PolyAST.Program))

  def printPass(pass: ProgramPass): ProgramPass = { (p: PolyAST.Program, l: Log) =>
    val r  = pass(p, l)
    val sl = l.subLog(s"[${pass.getClass.getName}]")
    sl.info("Structs", r.defs.map(_.repr)*)
    sl.info("Fns", r.functions.map(_.repr)*)
    sl.info("Entry", r.entry.repr)
    r
  }

}
