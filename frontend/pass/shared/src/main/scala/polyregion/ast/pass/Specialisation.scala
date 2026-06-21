package polyregion.ast.pass

import polyregion.ast.Traversal.*
import polyregion.ast.{PolyAST as p, *, given}

// monomorphises generic functions: one specialised copy per distinct tpeArg set reached from entry,
// rewrites each Invoke to the monomorphic name + drops tpeArgs, then removes generic templates
// examples:
//   id[Int](3); id[Float](1.0)  ->  Int_id(3); Float_id(1.0)   (+ two specialised copies, template dropped)
//   id[Int](3); id[Int](7)      ->  Int_id(3); Int_id(7)       (+ one specialised copy, deduped)
//   pair[Int,Float](3, 1.0)     ->  Int_Float_pair(3, 1.0)     (+ one specialised copy, template dropped)
// edge cases:
//   specialisation that invokes further generics -> recursiveSpecialise recurses into the new body
//   tpeArg set already specialised               -> deduped by monomorphic name (no second copy)
//   Invoke of an unknown / non-generic name      -> left untouched (no tpeArgs -> not rewritten)
object Specialisation extends ProgramPass {

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
      else if (!fnLUT.contains(ivk.name)) acc
      else {
        val fnImpl = fnLUT(ivk.name)
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

    log.info("functions", fnLUT.keys.toSeq.map(_.repr)*)
    log.info("callsites", callsites.map(_.repr)*)

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
