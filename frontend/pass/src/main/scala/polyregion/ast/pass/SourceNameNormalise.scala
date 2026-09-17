package polyregion.ast.pass

import polyregion.ast.Traversal.*
import polyregion.ast.{Log, PolyAST as p, *, given}

// normalises names that cross the source-backend boundary. struct and function symbols share the C ordinary
// identifier namespace; fields are allocated within their owner. references are rewritten with their definitions,
// leaving ForeignCall names and local diagnostic names untouched
// examples:
//   ns.Box / ns.make  ->  ns_Box / ns_make
//   operator=         ->  operator_
//   a-b, a_b          ->  a_b, a_b_1
// edge cases:
//   OpenCL vector spellings and address-space words  ->  prefixed with `_`
//   Metal address-space words                        ->  prefixed only when metalKeywords is true
//   source-position and Origin.source text           ->  retained verbatim for diagnostics
final case class SourceNameNormalise(metalKeywords: Boolean = false) extends ProgramPass derives PassArgCodec {

  override def phase: p.Pass.Phase = p.Pass.Phase.PostMono

  private val OpenCLReserved = {
    val vectorBases =
      List("char", "uchar", "short", "ushort", "int", "uint", "long", "ulong", "float", "double", "half")
    val vectorWidths = List("2", "3", "4", "8", "16")
    Set("global", "local", "kernel", "constant", "private") ++
      vectorBases.flatMap(base => vectorWidths.map(base + _))
  }
  private val MetalReserved = Set("device", "threadgroup", "thread")

  private def identifier(raw: String): String = {
    val replaced = raw.map { c =>
      if (c.isLetterOrDigit || c == '_') c else '_'
    }
    val nonEmpty = Option.when(replaced.nonEmpty)(replaced).getOrElse("_")
    val headed   = if (nonEmpty.head.isDigit) "_" + nonEmpty else nonEmpty
    if (OpenCLReserved(headed) || (metalKeywords && MetalReserved(headed))) "_" + headed else headed
  }

  private final class Names {
    private val used = scala.collection.mutable.Set.empty[String]

    def allocate(raw: String): String = {
      val base = identifier(raw)
      (Iterator.single(base) ++ Iterator.from(1).map(index => s"${base}_$index")).find(used.add).get
    }
  }

  override def apply(program: p.Program, log: Log): p.Program = {
    val ordinary = new Names
    val structNames =
      program.defs.map(definition => definition.name -> p.Sym(ordinary.allocate(definition.name.fqcn))).toMap
    val functions     = (program.entry.toList ::: program.functions).map(_.name).distinct
    val functionNames = functions.map(name => name -> p.Sym(ordinary.allocate(name.fqcn))).toMap
    val definitions   = program.defs.map(definition => definition.name -> definition).toMap
    val fieldNames = program.defs.map { definition =>
      val names = new Names
      definition.name -> definition.members.map(member => member.symbol -> names.allocate(member.symbol)).toMap
    }.toMap

    def renameType(tpe: p.Type): p.Type = tpe match {
      case structure @ p.Type.Struct(name, _) => structure.copy(name = structNames.getOrElse(name, name))
      case callable @ p.Type.FnRef(name)      => callable.copy(name = functionNames.getOrElse(name, name))
      case other                              => other
    }

    def struct(tpe: p.Type): Option[p.Type.Struct] = tpe match {
      case value: p.Type.Struct                => Some(value)
      case p.Type.Ptr(value: p.Type.Struct, _) => Some(value)
      case _                                   => None
    }

    def element(tpe: p.Type): p.Type = tpe match {
      case p.Type.Ptr(component, _)    => component
      case p.Type.Arr(component, _, _) => component
      case other                       => other
    }

    def renameSteps(root: p.Type, steps: List[p.PathStep]): List[p.PathStep] =
      steps
        .foldLeft(root -> List.empty[p.PathStep]) {
          case ((current, renamedSteps), p.PathStep.Field(name)) =>
            val owner = struct(current).map(_.name)
            val next = owner
              .flatMap(definitions.get)
              .flatMap(_.members.find(_.symbol == name))
              .map(_.tpe)
              .getOrElse(p.Type.Nothing)
            val renamed = owner.flatMap(fieldNames.get).flatMap(_.get(name)).getOrElse(identifier(name))
            next -> (p.PathStep.Field(renamed) :: renamedSteps)
          case ((current, renamedSteps), step @ (p.PathStep.Deref | _: p.PathStep.Index | _: p.PathStep.IndexDyn)) =>
            element(current) -> (step :: renamedSteps)
        }
        ._2
        .reverse

    def renameFunction(function: p.Function): p.Function = {
      val paths = function
        .modifyAll[p.Term] {
          case select @ p.Term.Select(root, steps, _) => select.copy(steps = renameSteps(root.tpe, steps))
          case other                                  => other
        }
        .modifyAll[p.Expr] {
          case offset @ p.Expr.OffsetOf(structTpe, field) =>
            val renamed =
              struct(structTpe).map(_.name).flatMap(fieldNames.get).flatMap(_.get(field)).getOrElse(identifier(field))
            offset.copy(field = renamed)
          case other => other
        }
        .modifyAll[p.Type](renameType)
      paths.copy(
        decl = paths.decl.copy(name = functionNames.getOrElse(function.name, function.name)),
        implements = paths.implements.map(name => functionNames.getOrElse(name, name))
      )
    }

    program.copy(
      entry = program.entry.map(renameFunction),
      functions = program.functions.map(renameFunction),
      defs = program.defs.map { definition =>
        val fields = fieldNames(definition.name)
        definition
          .modifyAll[p.Type](renameType)
          .copy(
            name = structNames(definition.name),
            members = definition.members.map(member =>
              member.modifyAll[p.Type](renameType).copy(symbol = fields(member.symbol))
            )
          )
      }
    )
  }
}
