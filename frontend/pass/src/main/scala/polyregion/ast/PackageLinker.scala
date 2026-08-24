package polyregion.ast

import polyregion.ast.Traversal.*
import polyregion.ast.{PolyAST as p, given}

import scala.collection.mutable

private[polyregion] object PackageLinker {

  private def symbol(name: p.Sym): String = name.fqn.mkString(".")

  private def shiftSize(size: p.Arg.SizeExpr, by: Int): p.Arg.SizeExpr = size match {
    case p.Arg.SizeExpr.Param(index)  => p.Arg.SizeExpr.Param(index + by)
    case value: p.Arg.SizeExpr.Const  => value
    case p.Arg.SizeExpr.Add(lhs, rhs) => p.Arg.SizeExpr.Add(shiftSize(lhs, by), shiftSize(rhs, by))
    case p.Arg.SizeExpr.Mul(lhs, rhs) => p.Arg.SizeExpr.Mul(shiftSize(lhs, by), shiftSize(rhs, by))
    case p.Arg.SizeExpr.Min(lhs, rhs) => p.Arg.SizeExpr.Min(shiftSize(lhs, by), shiftSize(rhs, by))
  }

  private def shiftBoundary(boundary: Option[p.Arg.Boundary], by: Int): Option[p.Arg.Boundary] =
    boundary.map { value =>
      val extent = value.extent match {
        case p.Arg.Extent.Elements(size) => p.Arg.Extent.Elements(shiftSize(size, by))
        case p.Arg.Extent.Bytes(size)    => p.Arg.Extent.Bytes(shiftSize(size, by))
      }
      value.copy(extent = extent)
    }

  private def composeImplementationDecl(
      harvested: p.FunctionDecl,
      publicDecl: p.FunctionDecl
  ): Either[List[String], p.FunctionDecl] = {
    val systemArguments = harvested.args.headOption.count(_.named.symbol == "#context")
    val publicArguments = harvested.args.size - systemArguments
    val trailingResult = publicArguments == publicDecl.args.size + 1 &&
      harvested.rtn == p.Type.Unit0 && publicDecl.rtn != p.Type.Unit0
    if (publicArguments != publicDecl.args.size && !trailingResult)
      Left(List("argument/result shape differs"))
    else {
      val args = harvested.args.zipWithIndex.map { case (argument, index) =>
        if (index >= systemArguments && index < systemArguments + publicDecl.args.size)
          argument.copy(boundary = shiftBoundary(publicDecl.args(index - systemArguments).boundary, systemArguments))
        else argument
      }.toArray
      if (trailingResult) {
        args(args.length - 1) = args.last.copy(boundary =
          Some(
            p.Arg.Boundary(
              p.Arg.Access.Write,
              p.Arg.Extent.Elements(p.Arg.SizeExpr.Const(1))
            )
          )
        )
      }
      val initiallyComposed = harvested.copy(args = args.toList)
      val usedNames         = initiallyComposed.collectAll[p.Type].collect { case p.Type.Var(name, _) => name }.toSet
      val composed          = initiallyComposed.copy(tpeVars = initiallyComposed.tpeVars.filter(v => usedNames(v.name)))
      PackageSymResolver.bindImplementation(composed, publicDecl).map(_ => composed)
    }
  }

  private def implementationClosure(root: p.Sym, program: p.Program): Either[List[String], List[p.Function]] = {
    val errors = List.newBuilder[String]
    val out    = List.newBuilder[p.Function]
    @annotation.tailrec
    def loop(frontier: List[p.Sym], reached: Set[p.Sym]): Unit = frontier match {
      case Nil                           => ()
      case name :: rest if reached(name) => loop(rest, reached)
      case name :: rest =>
        val matches = program.functions.filter(_.name == name)
        if (matches.size != 1) {
          errors +=
            s"implementation closure references ${if (matches.isEmpty) "absent" else "ambiguous"} function `${symbol(name)}`"
          loop(rest, reached + name)
        } else {
          val function = matches.head
          function.decl.validate.foreach(error =>
            errors += s"implementation closure function `${symbol(name)}`: $error"
          )
          out += function
          val next = function.collectAll[p.Type].collect { case p.Type.FnRef(target) => target }
          loop(next ::: rest, reached + name)
        }
    }
    loop(List(root), Set.empty)
    val distinct = errors.result().distinct
    Either.cond(distinct.isEmpty, out.result(), distinct)
  }

  private def validateStructClosure(functions: List[p.Function], program: p.Program): List[String] = {
    val errors = List.newBuilder[String]
    @annotation.tailrec
    def loop(frontier: List[p.Type.Struct], reached: Set[p.Sym]): Unit = frontier match {
      case Nil => ()
      case applied :: rest =>
        val matches = program.defs.filter(_.name == applied.name)
        if (matches.size != 1) {
          errors += s"struct definition `${symbol(applied.name)}` is ${if (matches.isEmpty) "absent" else "ambiguous"}"
          loop(rest, reached + applied.name)
        } else {
          val definition = matches.head
          if (definition.tpeVars.size != applied.args.size)
            errors +=
              s"struct `${symbol(applied.name)}` type-argument count differs: expected ${definition.tpeVars.size}, got ${applied.args.size}"
          if (reached(applied.name)) loop(rest, reached)
          else {
            definition.validate.foreach(error => errors += s"struct definition `${symbol(applied.name)}`: $error")
            val nested = definition.collectAll[p.Type].collect { case value: p.Type.Struct => value }
            loop(nested ::: rest, reached + applied.name)
          }
        }
    }
    val roots = functions.flatMap(_.collectAll[p.Type].collect { case value: p.Type.Struct => value })
    loop(roots, Set.empty)
    errors.result().distinct
  }

  def validate(pkg: p.Package): List[String] = {
    val errors = List.newBuilder[String]
    if (pkg.interface.name.fqn.exists(_.trim.isEmpty)) errors += "package identity contains an empty component"
    if (pkg.program.entry.nonEmpty) errors += "package program must be entryless"
    pkg.program.defs.foreach { definition =>
      definition.validate.foreach(error => errors += s"struct definition `${symbol(definition.name)}`: $error")
    }
    pkg.interface.declarations.foreach { declaration =>
      declaration.validate.foreach(error => errors += s"public declaration `${symbol(declaration.name)}`: $error")
    }
    pkg.program.functions.filter(_.implements.nonEmpty).foreach { implementation =>
      val name = symbol(implementation.name)
      if (implementation.visibility != p.Function.Visibility.Exported)
        errors += s"implementation `$name` is not exported"
      if (implementation.requiredCapabilities.distinct.size != implementation.requiredCapabilities.size)
        errors += s"implementation `$name` has duplicate capabilities"
      implementationClosure(implementation.name, pkg.program) match {
        case Left(messages)   => errors ++= messages
        case Right(functions) => errors ++= validateStructClosure(functions, pkg.program)
      }
      val declarations = pkg.interface.declarations.filter(decl => implementation.implements.contains(decl.name))
      val compatible =
        declarations.flatMap(decl => PackageSymResolver.bindImplementation(implementation.decl, decl).toOption)
      if (compatible.size != 1)
        errors += s"implementation `$name` matches ${compatible.size} public declarations"
      compatible.headOption.foreach { binding =>
        val constraints = implementation.tpeVars.flatMap(v => v.exactSizeInBytes.map(v.name -> _))
        val names       = constraints.map(_._1)
        if (names.distinct.size != names.size)
          errors += s"implementation `$name` has duplicate type-size constraints"
        val sizeable = binding.types.keySet -- binding.callables.keySet
        if (names.nonEmpty && names.toSet != sizeable)
          errors += s"implementation `$name` type-size constraints must cover all type variables"
      }
    }
    pkg.interface.declarations.foreach { declaration =>
      val covered = pkg.program.functions.exists { implementation =>
        implementation.implements.contains(declaration.name) &&
        PackageSymResolver.bindImplementation(implementation.decl, declaration).isRight
      }
      if (!covered) errors += s"public declaration `${symbol(declaration.name)}` has no compatible implementation"
    }
    errors.result().distinct
  }

  private def localName(fragment: Int, name: p.Sym): p.Sym = p.Sym("#fragment" :: fragment.toString :: name.fqn)

  def link(request: p.Package.LinkRequest): Either[List[String], p.Package] = {
    val capabilities = request.capabilities.toSet
    val selected = request.programFragments.map { fragment =>
      if (capabilities.isEmpty) fragment
      else
        fragment.copy(functions = fragment.functions.filter { function =>
          function.implements.isEmpty || function.requiredCapabilities.forall(capabilities)
        })
    }
    val implementationNames = selected.flatMap(_.functions).filter(_.implements.nonEmpty).map(_.name).toSet
    val functionGroups      = selected.flatMap(_.functions).groupBy(_.name)
    val definitionGroups    = selected.flatMap(_.defs).groupBy(_.name)
    val localFunctions = mutable.Set.from(functionGroups.collect {
      case (name, values) if !values.forall(_ == values.head) => name
    })
    val localDefinitions = mutable.Set.from(definitionGroups.collect {
      case (name, values) if !values.forall(_ == values.head) => name
    })
    var changed = true
    while (changed) {
      changed = false
      definitionGroups.foreach { case (name, definitions) =>
        if (definitions.size >= 2 && !localDefinitions(name)) {
          val depends = definitions.exists(_.collectAll[p.Type].exists {
            case p.Type.Struct(target, _) => localDefinitions(target)
            case _                        => false
          })
          if (depends) changed = localDefinitions.add(name) || changed
        }
      }
      functionGroups.foreach { case (name, functions) =>
        if (functions.size >= 2 && !localFunctions(name) && !implementationNames(name)) {
          val depends = functions.exists { function =>
            function.collectAll[p.Type].exists {
              case p.Type.FnRef(target)     => localFunctions(target)
              case p.Type.Struct(target, _) => localDefinitions(target)
              case _                        => false
            }
          }
          if (depends) changed = localFunctions.add(name) || changed
        }
      }
    }

    val isolated = selected.zipWithIndex.map { case (fragment, index) =>
      val functionNames = fragment.functions.collect {
        case function if localFunctions(function.name) => function.name -> localName(index, function.name)
      }.toMap
      val definitionNames = fragment.defs.collect {
        case definition if localDefinitions(definition.name) => definition.name -> localName(index, definition.name)
      }.toMap
      val functions = fragment.functions.map { function =>
        val rewritten = function
          .modifyAll[p.Type] {
            case value @ p.Type.FnRef(name)        => functionNames.get(name).fold(value)(p.Type.FnRef(_))
            case value @ p.Type.Struct(name, args) => definitionNames.get(name).fold(value)(p.Type.Struct(_, args))
            case value                             => value
          }
        functionNames
          .get(rewritten.name)
          .fold(rewritten)(name => rewritten.copy(decl = rewritten.decl.copy(name = name)))
      }
      val definitions = fragment.defs.map { definition =>
        val rewritten = definition.modifyAll[p.Type] {
          case value @ p.Type.Struct(name, args) => definitionNames.get(name).fold(value)(p.Type.Struct(_, args))
          case value                             => value
        }
        definitionNames.get(rewritten.name).fold(rewritten)(name => rewritten.copy(name = name))
      }
      p.Program(None, functions, definitions)
    }
    val errors    = List.newBuilder[String]
    val functions = isolated.flatMap(_.functions).toArray
    functions.indices.foreach { index =>
      val function = functions(index)
      function.implements.foreach { publicName =>
        val declarations = request.interface.declarations.filter(_.name == publicName)
        val attempts     = declarations.map(composeImplementationDecl(function.decl, _))
        val matches      = attempts.flatMap(_.toOption)
        if (matches.size != 1) {
          errors +=
            s"implementation `${symbol(function.name)}` matches ${matches.size} public declarations for `${symbol(publicName)}`"
          if (matches.isEmpty) attempts.flatMap(_.left.toOption.toList.flatten).distinct.foreach(errors += _)
        } else
          functions(index) = function.copy(
            decl = matches.head,
            visibility = p.Function.Visibility.Exported
          )
      }
    }
    val mergedFunctions = functions.toList.groupBy(_.name)
    mergedFunctions.foreach { case (name, values) =>
      if (!values.forall(_ == values.head)) errors += s"package program contains conflicting function `${symbol(name)}`"
    }
    val definitions       = isolated.flatMap(_.defs)
    val mergedDefinitions = definitions.groupBy(_.name)
    mergedDefinitions.foreach { case (name, values) =>
      if (!values.forall(_ == values.head))
        errors += s"package program contains conflicting struct definition `${symbol(name)}`"
    }
    val program = p.Program(
      None,
      mergedFunctions.values.map(_.head).toList.sortBy(_.name.fqn.mkString(".")),
      mergedDefinitions.values.map(_.head).toList.sortBy(_.name.fqn.mkString("."))
    )
    val pkg       = p.Package(request.interface, program)
    val allErrors = (errors.result() ::: validate(pkg)).distinct
    Either.cond(allErrors.isEmpty, pkg, allErrors)
  }
}
