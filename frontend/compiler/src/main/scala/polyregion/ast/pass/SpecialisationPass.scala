package polyregion.ast.pass

import cats.syntax.all.*
import polyregion.ast.Traversal.*
import polyregion.ast.{PolyAst as p, *, given}
import polyregion.prism.Prism

object SpecialisationPass extends ProgramPass {

  override def apply(program: p.Program, log: Log): p.Program = {

    val callsites = (program.entry :: program.functions)
      .collectWhere[p.Expr] { case ivk: p.Expr.Invoke => ivk }
      .distinct

    val fnLUT = program.functions.map(f => f.name -> f).toMap

    val callsiteWithImpl = callsites
      .filter(_.tpeArgs.nonEmpty)
      .map { ivk =>
        val monomorphicName = ivk.name :+ ivk.tpeArgs.map(_.monomorphicName).mkString("_")
        val fnImpl          = fnLUT(ivk.name)
        val tpeLut          = fnImpl.tpeVars.zip(ivk.tpeArgs).toMap
        val specialisedFnImpl = fnImpl
          .copy(name = monomorphicName, tpeVars = Nil)
          .modifyAll[p.Type] {
            case p.Type.Var(name) => tpeLut(name)
            case x                => x
          }
        ivk -> (ivk.copy(name = monomorphicName, tpeArgs = Nil), specialisedFnImpl)
      }
      .toMap

    val fnSpecialisationLUT = callsiteWithImpl.map { case (v, (_, f)) => v.name -> f }

    val programFnsWithSpecialisation = program.functions.map(f => fnSpecialisationLUT.getOrElse(f.name, f))

    def doReplace(f: p.Function) = f.modifyAll[p.Expr] {
      case ivk: p.Expr.Invoke => callsiteWithImpl.get(ivk).fold(ivk)((rewritten, _) => rewritten)
      case x                  => x
    }

    program.copy(entry = doReplace(program.entry), functions = programFnsWithSpecialisation.map(doReplace(_)))

  }

}
