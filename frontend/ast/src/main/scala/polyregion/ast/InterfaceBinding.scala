package polyregion.ast

import polyregion.ast.PolyAST as p

object InterfaceBinding {

  case class Binding(types: Map[String, p.Type], callables: Map[Int, p.Sym])

  enum ResultBinding {
    case Direct
    case TrailingOutput(index: Int)
  }

  case class ImplementationBinding(types: Map[String, p.Type], callables: Map[String, Int], result: ResultBinding)

  case class Resolution(
      publicDecl: p.FunctionDecl,
      callBinding: Binding,
      candidate: p.ImplementationCandidate,
      implementationBinding: ImplementationBinding
  )

  enum ArgumentKind {
    case Buffer(access: p.Arg.Access, extent: p.Arg.Extent)
    case ExtentScalar
    case Scalar
    case Callable(signature: p.Type.Exec)
  }
}
