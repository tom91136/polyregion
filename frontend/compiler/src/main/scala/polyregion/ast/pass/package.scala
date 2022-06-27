package polyregion.ast

package object pass {

  trait ProgramPass extends ((PolyAst.Program, Log) => (PolyAst.Program, Log))

}
