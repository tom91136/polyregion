package polyregion.ast.pass

import polyregion.ast.{Log, PolyAST as p}

final case class OffloadEntryInline(maxDepth: Int = 1024) extends ProgramPass derives PassArgCodec {

  override def phase: p.Pass.Phase = p.Pass.Phase.PostMono

  override def apply(program: p.Program, log: Log): p.Program = {
    val all    = program.entry.toList ::: program.functions
    val frames = List.newBuilder[p.StructDef]

    def inlineEntry(entry: p.Function): p.Function = {
      val entryLog = log.subLog(entry.name.repr)
      // Each offload entry is independent.  Do not make FnInline build an overload table for, and repeatedly
      // traverse, every other exported package root.  Internalising the candidates lets the ordinary reachability
      // pass retain precisely this entry's transitive callees (including every overload of a reached name).
      val candidates = all.filterNot(_.decl == entry.decl).map(_.copy(visibility = p.Function.Visibility.Internal))
      val reachable = DeadFunctionElimination(
        program.copy(entry = Some(entry), functions = candidates),
        entryLog.subLog("reachable")
      )
      val lowered = RecursionLower(maxDepth)(reachable, entryLog.subLog("recursion"))
      frames ++= lowered.defs.drop(reachable.defs.size)
      FnInline(lowered, entryLog).entry
        .getOrElse(entry)
    }

    def transform(function: p.Function): p.Function =
      if (function.convention == p.CallConvention.OffloadEntry && function.affinity == p.Function.Affinity.Offload)
        inlineEntry(function)
      else function

    val out = program.copy(entry = program.entry.map(transform), functions = program.functions.map(transform))
    frames.result() match {
      case Nil   => out
      case added => out.copy(defs = (out.defs ::: added).distinct)
    }
  }
}
