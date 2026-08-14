package polyregion.spectra

import polyregion.ast.PolyAST as p

private object SpectraImplementations {

  private val variantName = "[A-Za-z_][A-Za-z0-9_]*".r
  private val capability  = "[A-Za-z0-9][A-Za-z0-9_.-]*".r

  private def combinations(widths: List[Int], arity: Int): List[List[Int]] =
    if arity == 0 then List(Nil)
    else
      for
        width <- widths
        tail  <- combinations(widths, arity - 1)
      yield width :: tail

  private def substitute(tpe: p.Type, bindings: Map[String, p.Type]): p.Type = tpe match {
    case p.Type.Var(name)                     => bindings.getOrElse(name, tpe)
    case p.Type.Ptr(component, space)         => p.Type.Ptr(substitute(component, bindings), space)
    case p.Type.Arr(component, length, space) => p.Type.Arr(substitute(component, bindings), length, space)
    case p.Type.Struct(name, args)            => p.Type.Struct(name, args.map(substitute(_, bindings)))
    case p.Type.Exec(tpeVars, args, rtn) =>
      val nested = bindings -- tpeVars
      p.Type.Exec(tpeVars, args.map(substitute(_, nested)), substitute(rtn, nested))
    case _ => tpe
  }

  def candidates(
      interfaceDef: p.InterfaceDef,
      variants: Spectra.ImplementationVariants
  ): List[p.ImplementationCandidate] = {
    val widths       = variants.widths.sorted
    val capabilities = variants.requiredCapabilities.sorted
    require(variantName.matches(variants.name), s"invalid implementation variant name `${variants.name}`")
    require(
      widths.nonEmpty || variants.includeFallback,
      "at least one implementation width or a fallback implementation is required"
    )
    require(widths.forall(_ > 0), "implementation widths must be positive")
    require(widths.distinct.size == widths.size, "implementation widths must be distinct")
    require(
      variants.requiredCapabilities.forall(capability.matches),
      "implementation capabilities must contain only letters, digits, dots, underscores and hyphens"
    )
    require(
      variants.requiredCapabilities.distinct.size == variants.requiredCapabilities.size,
      "implementation capabilities must be distinct"
    )
    require(
      interfaceDef.decls.groupBy(_.name).forall(_._2.size == 1),
      "implementation generation does not support overloaded public declarations"
    )

    interfaceDef.decls.flatMap { publicDecl =>
      val specialised =
        if publicDecl.tpeVars.isEmpty then Nil else combinations(widths, publicDecl.tpeVars.size).map(Some.apply)
      val sizes = specialised ::: Option.when(variants.includeFallback || publicDecl.tpeVars.isEmpty)(None).toList
      sizes.map { sizes =>
        val implementationVariables = sizes match {
          case Some(widths) => publicDecl.tpeVars.zip(widths).map((name, width) => s"$name$width")
          case None         => publicDecl.tpeVars
        }
        val bindings      = publicDecl.tpeVars.zip(implementationVariables.map(p.Type.Var.apply)).toMap
        var callableIndex = 0
        val implementationArgs = publicDecl.args.map { arg =>
          val tpe = arg.named.tpe match {
            case _: p.Type.Exec =>
              val name = s"Callable$callableIndex"
              callableIndex += 1
              p.Type.Var(name)
            case other => substitute(other, bindings)
          }
          arg.copy(named = arg.named.copy(tpe = tpe))
        }
        val callableVariables = List.tabulate(callableIndex)(index => s"Callable$index")
        val suffix = sizes match {
          case Some(widths) => widths.map(width => s"w$width").mkString("_")
          case None         => "fallback"
        }
        val implementationName = s"${publicDecl.name.last}_$suffix"
        val implementation = p.FunctionDecl(
          p.Sym(List("spectra", "implementation", variants.name, implementationName)),
          implementationVariables ::: callableVariables,
          publicDecl.receiver.map(arg => arg.copy(named = arg.named.copy(tpe = substitute(arg.named.tpe, bindings)))),
          implementationArgs,
          publicDecl.moduleCaptures.map(arg =>
            arg.copy(named = arg.named.copy(tpe = substitute(arg.named.tpe, bindings)))
          ),
          publicDecl.termCaptures.map(arg =>
            arg.copy(named = arg.named.copy(tpe = substitute(arg.named.tpe, bindings)))
          ),
          substitute(publicDecl.rtn, bindings),
          publicDecl.affinity
        )
        p.ImplementationCandidate(
          publicDecl.name,
          implementation,
          capabilities,
          sizes.toList.flatMap(widths => implementationVariables.zip(widths).map(p.TypeSizeConstraint.apply))
        )
      }
    }
  }

  def index(interfaceDef: p.InterfaceDef, variants: List[Spectra.ImplementationVariants]): p.PackageIndex = {
    require(variants.nonEmpty, "at least one implementation variant set is required")
    require(variants.map(_.name).distinct.size == variants.size, "implementation variant names must be distinct")
    p.PackageIndex(interfaceDef, variants.sortBy(_.name).flatMap(candidates(interfaceDef, _)))
  }
}
