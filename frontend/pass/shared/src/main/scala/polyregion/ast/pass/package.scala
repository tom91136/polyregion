package polyregion.ast

package object pass {

  trait ProgramPass extends ((PolyAST.Program, Log) => PolyAST.Program) {
    def phase: PolyAST.PassPhase = PolyAST.PassPhase.Initial
  }
  trait BoundaryPass[A] extends ((PolyAST.Program, Log) => (A, PolyAST.Program)) {
    def phase: PolyAST.PassPhase = PolyAST.PassPhase.Initial
  }

  def printPass(pass: ProgramPass): ProgramPass = new ProgramPass {
    override def phase = pass.phase
    def apply(p: PolyAST.Program, l: Log): PolyAST.Program = {
      val r  = pass(p, l)
      val sl = l.subLog(s"[${pass.getClass.getName}]")
      sl.info("Structs", r.defs.map(_.repr)*)
      sl.info("Fns", r.functions.map(_.repr)*)
      sl.info("Entry", r.entry.repr)
      r
    }
  }

}
