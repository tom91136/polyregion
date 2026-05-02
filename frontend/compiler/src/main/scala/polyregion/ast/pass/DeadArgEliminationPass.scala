package polyregion.ast.pass

import cats.syntax.all.*
import polyregion.ast.Traversal.*
import polyregion.ast.{PolyAST as p, *, given}

object DeadArgEliminationPass extends ProgramPass {

  private def cleanModuleCaptures(f: p.Function): p.Function = {
    val topLevelRefs = f.body.flatMap { s =>
      s.collectWhere[p.Expr] {
        case p.Expr.Select(Nil, x)    => x
        case p.Expr.Select(x :: _, _) => x
      }
    }.toSet
    // Only sweep module captures since they're the ones IntrinsifyPass can render unused
    // (e.g., the `intrinsics$` module after `intrinsics$.sin(x)` becomes `MathOp(Sin)`).
    // Receiver, args, termCaptures are part of the calling convention and stay regardless.
    f.copy(moduleCaptures = f.moduleCaptures.filter(arg => topLevelRefs.contains(arg.named)))
  }

  // Cleans the entry function fully (including unused term captures, which the offload binding
  // path doesn't pass) — entry has no callers, so the asymmetric treatment is safe.
  private def cleanEntryFully(f: p.Function): p.Function = {
    val topLevelRefs = f.body.flatMap { s =>
      s.collectWhere[p.Expr] {
        case p.Expr.Select(Nil, x)    => x
        case p.Expr.Select(x :: _, _) => x
      }
    }.toSet
    f.copy(
      receiver = f.receiver.filter(arg => topLevelRefs.contains(arg.named)),
      args = f.args.filter(arg => topLevelRefs.contains(arg.named)),
      moduleCaptures = f.moduleCaptures.filter(arg => topLevelRefs.contains(arg.named)),
      termCaptures = f.termCaptures.filter(arg => topLevelRefs.contains(arg.named))
    )
  }

  override def apply(program: p.Program, log: Log): (p.Program) =
    program.copy(
      entry = cleanEntryFully(program.entry),
      // Helper functions: only sweep module captures so call-site signatures still align on
      // receiver/args/termCaptures. Module captures are unique because IntrinsifyPass can make
      // them statically unused mid-pipeline.
      functions = program.functions.map(cleanModuleCaptures)
    )

}
