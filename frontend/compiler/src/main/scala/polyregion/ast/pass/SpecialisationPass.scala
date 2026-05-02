package polyregion.ast.pass

import cats.syntax.all.*
import polyregion.ast.Traversal.*
import polyregion.ast.{PolyAST as p, *, given}
import polyregion.prism.Prism

// This pass copies all generic functions with the applied types at callsite:
//   def foo[A](a: A) = a
//   foo[Int]
//   foo[Long]
// becomes
//   def foo_Long(a : Long) = a
//   def foo_Int(a : Int) = a
//   foo_Int
//   foo_Long
object SpecialisationPass extends ProgramPass {

  def monomorphicName(ivk: p.Expr.Invoke): p.Sym = {
    val monomorphicToken = ivk.tpeArgs.map(_.monomorphicName).mkString("_")
    ivk.name.fqn match {
      case xs :+ x => p.Sym(xs :+ monomorphicToken :+ x)
      case xs      => p.Sym(monomorphicToken :: xs)
    }
  }

  def recursiveSpecialise(
      fnLUT: Map[p.Sym, p.Function],
      entry: p.Function,
      done: Map[p.Sym, p.Function] = Map.empty
  ): Map[p.Sym, p.Function] = entry
    .collectWhere[p.Expr] { case ivk: p.Expr.Invoke => ivk }
    .filter(_.tpeArgs.nonEmpty)
    .distinct
    .foldLeft(done) { case (acc, ivk) =>
      val newName = monomorphicName(ivk)
      if (done.contains(newName)) acc
      else if (!fnLUT.contains(ivk.name)) acc // missing impl — verifier will surface a useful error
      else {
        val fnImpl = fnLUT(ivk.name)
        // Inherited-trait method calls (e.g. `xs.size` resolving to `SeqOps.size`) sometimes
        // arrive here with `ivk.tpeArgs` carrying duplicated type-var refs accumulated up the
        // hierarchy — `[Var(A), Var(CC), Var(C), Var(A), Var(CC), Var(C)]` for a 3-tparam method.
        // Take only the prefix that aligns with the function's own tpeVars; the trailing copies
        // are redundant. Also tolerate Vars whose name isn't in our lut (passing them through
        // unchanged) so a missing substitution doesn't crash specialisation.
        val tpeLut = fnImpl.tpeVars.zip(ivk.tpeArgs.take(fnImpl.tpeVars.size)).toMap
        val specialisedFnImpl = fnImpl
          .copy(name = newName, tpeVars = Nil)
          .modifyAll[p.Type] {
            case v @ p.Type.Var(name) => tpeLut.getOrElse(name, v)
            case x                    => x
          }
        recursiveSpecialise(fnLUT, specialisedFnImpl, acc + (specialisedFnImpl.name -> specialisedFnImpl))
      }
    }

  override def apply(program: p.Program, log: Log): p.Program = {

    val callsites = (program.entry :: program.functions)
      .collectWhere[p.Expr] { case ivk: p.Expr.Invoke => ivk }
      .distinct

    val fnLUT = program.functions.map(f => f.name -> f).toMap

    println("--")
    println(fnLUT.keySet.mkString("\n"))
    println("--")
    println(callsites.mkString("\n"))

    // Tracing specialisation
    // 1. For entry fn, find all callsites
    // 2. Specialise fn for each callsite, cache results
    // 3. Walk all callsites again, replace with monomorphic names

    val specialisations = recursiveSpecialise(fnLUT, program.entry)

    log.info("Specialisations", specialisations.values.map(_.signatureRepr).toList.sorted*)

    def doReplace(f: p.Function) = f.modifyAll[p.Expr] {
      case ivk: p.Expr.Invoke =>
        if (ivk.tpeArgs.isEmpty) ivk
        else ivk.copy(name = monomorphicName(ivk), tpeArgs = Nil)
      case x => x
    }

    program.copy(
      entry = doReplace(program.entry),
      functions = (program.functions.filter(_.tpeVars.isEmpty) ++ specialisations.values).map(doReplace(_))
    )

  }

}
