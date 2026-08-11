package polyregion.ast

import polyregion.ast.PolyAST as p

object LibraryBinding {

  case class Binding(types: Map[String, p.Type], callables: Map[Int, p.Sym])

  enum ResultBinding {
    case Direct
    case TrailingOutput(index: Int)
  }

  case class ImplementationBinding(types: Map[String, p.Type], result: ResultBinding)

  case class TypeSizeConstraint(typeVariable: String, sizeInBytes: Int) derives MsgPack.Codec

  case class ImplementationCandidate(
      publicName: p.Sym,
      implementation: p.FunctionDecl,
      requiredCapabilities: List[String],
      typeSizes: List[TypeSizeConstraint]
  ) derives MsgPack.Codec

  case class PackageIndex(interface: p.LibraryDef, candidates: List[ImplementationCandidate]) derives MsgPack.Codec

  case class Resolution(
      publicDecl: p.FunctionDecl,
      callBinding: Binding,
      candidate: ImplementationCandidate,
      implementationBinding: ImplementationBinding
  )

  enum ArgumentKind {
    case Buffer(access: p.Arg.Access, extent: p.Arg.Extent)
    case ExtentScalar
    case Scalar
    case Callable(signature: p.Type.Exec)
  }
}
