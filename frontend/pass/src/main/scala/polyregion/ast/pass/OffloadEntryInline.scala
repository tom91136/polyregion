package polyregion.ast.pass

import polyregion.ast.{Log, PolyAST as p}

object OffloadEntryInline extends ProgramPass {

  override def phase: p.Pass.Phase = p.Pass.Phase.PostMono

  override def apply(program: p.Program, log: Log): p.Program = {
    val all = program.entry.toList ::: program.functions

    def inlineEntry(entry: p.Function): p.Function = {
      val dependencies = all.filterNot(_.decl == entry.decl)
      FnInline(program.copy(entry = Some(entry), functions = dependencies), log.subLog(entry.name.repr)).entry
        .getOrElse(entry)
    }

    def transform(function: p.Function): p.Function =
      if (function.convention == p.CallConvention.OffloadEntry && function.affinity == p.Function.Affinity.Offload)
        inlineEntry(function)
      else function

    program.copy(entry = program.entry.map(transform), functions = program.functions.map(transform))
  }
}
