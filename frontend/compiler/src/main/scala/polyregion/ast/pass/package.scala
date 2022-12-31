package polyregion.ast

package object pass {

  trait ProgramPass extends ((PolyAst.Program, Log) => (PolyAst.Program, Log))
  trait BoundaryPass[A] extends ((PolyAst.Program, Log) => (A, PolyAst.Program, Log))

}
