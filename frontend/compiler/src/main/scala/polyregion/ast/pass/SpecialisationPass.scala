package polyregion.ast.pass

import cats.syntax.all.*
import polyregion.ast.Traversal.*
import polyregion.ast.{PolyAst as p, *, given}
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

  override def apply(program: p.Program, log: Log): p.Program = {

    val callsites = (program.entry :: program.functions)
      .collectWhere[p.Expr] { case ivk: p.Expr.Invoke => ivk }
      .distinct

    val fnLUT = program.functions.map(f => f.name -> f).toMap

    println("--")
    println(fnLUT.keySet.mkString("\n"))
    println("--")
    println(callsites.mkString("\n"))

    val callsiteWithImpl = callsites
      .filter(_.tpeArgs.nonEmpty)
      .map { ivk =>

        log.info("ivk", ivk.repr)

        val monomorphicToken = ivk.tpeArgs.map(_.monomorphicName).mkString("_")
        val monomorphicName = ivk.name.fqn match {
          case xs :+ x => p.Sym(xs :+ monomorphicToken :+ x)
          case xs      => p.Sym(monomorphicToken :: xs)
        }

        val fnImpl = fnLUT(ivk.name)
        val tpeLut = fnImpl.tpeVars.zip(ivk.tpeArgs).toMap
        val specialisedFnImpl = fnImpl
          .copy(name = monomorphicName, tpeVars = Nil)
          .modifyAll[p.Type] {
            case p.Type.Var(name) => tpeLut(name)
            case x                => x
          }
        ivk -> (ivk.copy(name = monomorphicName, tpeArgs = Nil), specialisedFnImpl, fnImpl)
      }
      .toMap

    val originalToSpecialisedLUT = callsiteWithImpl.toList.groupMap { case (_, (_, specialisedFn, originalFn)) =>
      originalFn
    } { case (_, (_, specialisedFn, _)) => specialisedFn }

    log.info(
      "Specialisations",
      originalToSpecialisedLUT
        .map((k, vs) => s"${k.signatureRepr} -> ${vs.map(_.signatureRepr).mkString(" ; ")}")
        .toList*
    )

    val programFnsWithSpecialisation = program.functions.flatMap(f => originalToSpecialisedLUT.getOrElse(f, f:: Nil))

    def doReplace(f: p.Function) = f.modifyAll[p.Expr] {
      case ivk: p.Expr.Invoke => callsiteWithImpl.get(ivk).fold(ivk)((rewritten, _, _) => rewritten)
      case x                  => x
    }

    program.copy(entry = doReplace(program.entry), functions = programFnsWithSpecialisation.map(doReplace(_)))

  }

}
