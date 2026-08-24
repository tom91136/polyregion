package polyregion.ast

import polyregion.ast.PolyAST as p

private[polyregion] object InterfaceBinding {

  final case class BoundCall(types: Map[String, p.Type], callables: Map[Int, p.Sym])

  enum ReturnConvention {
    case Direct
    case TrailingOutput(index: Int)
  }

  final case class BoundImplementation(
      types: Map[String, p.Type],
      callables: Map[String, Int],
      result: ReturnConvention,
      systemArguments: Int
  )

  final case class ResolvedCall(
      publicDeclaration: p.FunctionDecl,
      signature: BoundCall,
      implementation: p.Function,
      implementationBinding: BoundImplementation
  )
}
