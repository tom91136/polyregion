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

  private case class Callsite(calleeName: p.Sym, tpeArgs: List[p.Type])

  private def callsites(fn: p.Function): List[Callsite] =
    fn.collectWhere[p.Expr] {
      case ivk: p.Expr.Invoke => Callsite(ivk.calleeName, ivk.tpeArgs) :: Nil
      case p.Expr.SpecOp(launch: p.Spec.RemoteLaunch) =>
        launch.kernel.tpe match {
          case p.Type.FnRef(name) => Callsite(name, launch.tpeArgs) :: Nil
          case _                  => Nil
        }
    }.flatten

  def monomorphicName(calleeName: p.Sym, tpeArgs: List[p.Type]): p.Sym = {
    val monomorphicToken = tpeArgs.map(_.monomorphicName).mkString("_")
    calleeName.fqn match {
      case xs :+ x => p.Sym(xs :+ monomorphicToken :+ x)
      case xs      => p.Sym(monomorphicToken :: xs)
    }
  }

  def monomorphicName(ivk: p.Expr.Invoke): p.Sym = monomorphicName(ivk.calleeName, ivk.tpeArgs)

  def recursiveSpecialise(
      fnLUT: Map[p.Sym, p.Function],
      entry: p.Function,
      done: Map[p.Sym, p.Function] = Map.empty
  ): Map[p.Sym, p.Function] = callsites(entry)
    .filter(_.tpeArgs.nonEmpty)
    .distinct
    .foldLeft(done) { case (acc, callsite) =>
      val newName = monomorphicName(callsite.calleeName, callsite.tpeArgs)
      if (acc.contains(newName)) acc
      else if (!fnLUT.contains(callsite.calleeName)) acc
      else {
        val fnImpl = fnLUT(callsite.calleeName)
        val tpeLut = fnImpl.tpeVars.zip(callsite.tpeArgs.take(fnImpl.tpeVars.size)).toMap
        val specialisedFnImpl = fnImpl
          .copy(decl = fnImpl.decl.copy(name = newName, tpeVars = Nil))
          .modifyAll[p.Type] {
            case v @ p.Type.Var(name) => tpeLut.getOrElse(name, v)
            case x                    => x
          }
        recursiveSpecialise(fnLUT, specialisedFnImpl, acc + (specialisedFnImpl.name -> specialisedFnImpl))
      }
    }

  override def apply(program: p.Program, log: Log): p.Program = {

    val allCallsites = (program.entry :: program.functions).flatMap(callsites).distinct

    val fnLUT = program.functions.map(f => f.name -> f).toMap

    log.info("functions", fnLUT.keys.toSeq.map(_.repr)*)
    log.info(
      "callsites",
      allCallsites.map(x => s"${x.calleeName.repr}[${x.tpeArgs.map(_.repr).mkString(", ")}]")*
    )

    val specialisations =
      (program.entry :: program.functions.filter(_.tpeVars.isEmpty)).foldLeft(Map.empty[p.Sym, p.Function]) {
        case (done, root) => recursiveSpecialise(fnLUT, root, done)
      }

    log.info("Specialisations", specialisations.values.map(_.signatureRepr).toList.sorted*)

    def doReplace(f: p.Function) = f.modifyAll[p.Expr] {
      case ivk: p.Expr.Invoke =>
        if (ivk.tpeArgs.isEmpty) ivk
        else ivk.copy(callee = p.Type.FnRef(monomorphicName(ivk)), tpeArgs = Nil)
      case p.Expr.SpecOp(launch: p.Spec.RemoteLaunch) if launch.tpeArgs.nonEmpty =>
        launch.kernel.tpe match {
          case p.Type.FnRef(name) =>
            val newName = monomorphicName(name, launch.tpeArgs)
            val kernel = launch.kernel.modifyAll[p.Type] {
              case p.Type.FnRef(`name`) => p.Type.FnRef(newName)
              case x                    => x
            }
            p.Expr.SpecOp(launch.copy(kernel = kernel, tpeArgs = Nil))
          case _ => p.Expr.SpecOp(launch)
        }
      case x => x
    }

    program.copy(
      entry = doReplace(program.entry),
      functions = (program.functions.filter(_.tpeVars.isEmpty) ++ specialisations.values).map(doReplace(_))
    )

  }

}
