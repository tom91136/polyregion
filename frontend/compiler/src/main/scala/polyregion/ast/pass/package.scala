package polyregion.ast

package object pass {

  trait ProgramPass     extends ((ScalaSRR.Program, Log) => ScalaSRR.Program)
  trait BoundaryPass[A] extends ((ScalaSRR.Program, Log) => (A, ScalaSRR.Program))

  def printPass(pass: ProgramPass): ProgramPass = { (p: ScalaSRR.Program, l: Log) =>
    val r  = pass(p, l)
    val sl = l.subLog(s"[${pass.getClass.getName}]")
    sl.info("Structs", r.defs.map(_.repr)*)
    sl.info("Fns", r.functions.map(_.repr)*)
    sl.info("Entry", r.entry.repr)
    r
  }

}
