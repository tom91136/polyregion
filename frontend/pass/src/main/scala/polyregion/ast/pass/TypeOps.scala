package polyregion.ast.pass

import polyregion.ast.PolyAST as p

private[pass] object TypeOps {

  def launchShaped(candidate: p.Function, tpeArgCount: Int, parameters: List[?], argCount: Int): Boolean =
    candidate.affinity == p.Function.Affinity.Offload &&
      candidate.receiver.isEmpty &&
      candidate.rtn == p.Type.Unit0 &&
      candidate.tpeVars.size == tpeArgCount &&
      parameters.size == argCount
  def sameType(left: p.Type, right: p.Type): Boolean = {
    def loop(
        x: p.Type,
        y: p.Type,
        xBound: Map[String, Int],
        yBound: Map[String, Int],
        depth: Int
    ): Boolean = (x, y) match {
      case (p.Type.Nothing, _) | (_, p.Type.Nothing) => true
      case (p.Type.Var(xName, _), p.Type.Var(yName, _)) =>
        (xBound.get(xName), yBound.get(yName)) match {
          case (Some(xId), Some(yId)) => xId == yId
          case (None, None)           => xName == yName
          case _                      => false
        }
      case (p.Type.Struct(xName, xArgs), p.Type.Struct(yName, yArgs)) =>
        xName == yName && xArgs.size == yArgs.size &&
        xArgs.zip(yArgs).forall((a, b) => loop(a, b, xBound, yBound, depth))
      case (p.Type.Ptr(xComp, xSpace), p.Type.Ptr(yComp, ySpace)) =>
        xSpace == ySpace && loop(xComp, yComp, xBound, yBound, depth)
      case (p.Type.Arr(xComp, xLength, xSpace), p.Type.Arr(yComp, yLength, ySpace)) =>
        xLength == yLength && xSpace == ySpace && loop(xComp, yComp, xBound, yBound, depth)
      case (p.Type.Exec(xVars, xArgs, xRtn), p.Type.Exec(yVars, yArgs, yRtn)) =>
        val ids       = xVars.indices.map(depth + _)
        val nestedX   = xBound ++ xVars.map(_.name).zip(ids)
        val nestedY   = yBound ++ yVars.map(_.name).zip(ids)
        val nestedEnd = depth + xVars.size
        xVars.size == yVars.size && xArgs.size == yArgs.size &&
        xArgs.zip(yArgs).forall((a, b) => loop(a, b, nestedX, nestedY, nestedEnd)) &&
        loop(xRtn, yRtn, nestedX, nestedY, nestedEnd)
      case _ => x == y
    }
    loop(left, right, Map.empty, Map.empty, 0)
  }

  def width(tpe: p.Type): Option[Int] = tpe match {
    case p.Type.Bool1 | p.Type.IntU8 | p.Type.IntS8     => Some(1)
    case p.Type.IntU16 | p.Type.IntS16 | p.Type.Float16 => Some(2)
    case p.Type.IntU32 | p.Type.IntS32 | p.Type.Float32 => Some(4)
    case p.Type.IntU64 | p.Type.IntS64 | p.Type.Float64 => Some(8)
    case _                                              => None
  }

  def fieldType(defs: Map[p.Sym, p.StructDef], owner: p.Type, field: String): p.Type = {
    val structure = owner match {
      case value: p.Type.Struct                => value
      case p.Type.Ptr(value: p.Type.Struct, _) => value
      case other => throw IllegalArgumentException(s"field $field selected on non-struct type ${other.repr}")
    }
    defs
      .get(structure.name)
      .flatMap(_.members.find(_.symbol == field))
      .map(_.tpe)
      .getOrElse(throw IllegalArgumentException(s"field $field not found on ${structure.name.fqcn}"))
  }

  def pathType(defs: Map[p.Sym, p.StructDef], root: p.Type, steps: List[p.PathStep]): p.Type = steps.foldLeft(root) {
    case (current, p.PathStep.Field(field)) => fieldType(defs, current, field)
    case (p.Type.Ptr(component, _), p.PathStep.Deref | p.PathStep.Index(_) | p.PathStep.IndexDyn(_)) => component
    case (p.Type.Arr(component, _, _), p.PathStep.Index(_) | p.PathStep.IndexDyn(_))                 => component
    case (current, step) => throw IllegalArgumentException(s"cannot apply ${step.repr} to ${current.repr}")
  }
}
