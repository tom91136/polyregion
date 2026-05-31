package polyregion.ast.pass

import polyregion.ast.{Log, PolyAST as p}

case class FullOpt(level: Int = 3) extends ProgramPass derives PassArgCodec {
  def apply(program: p.Program, log: Log): p.Program =
    run(program, PassContext(log, PassRunner()))

  override def run(program: p.Program, ctx: PassContext): p.Program =
    FullOpt.children(level).foldLeft(program)((acc, pass) => ctx.run(pass, acc))
}

object FullOpt {
  private val baseline: Vector[ProgramPass] =
    Vector(ConstantFold, VarReduce, UnitExprElision, DeadArgElimination)

  // XXX Intrinsify runs once to recognise direct `intrinsics.X` calls, then again after
  // FnInline so calls revealed by inlining a prism replacement (e.g. `StdLib.math.abs`
  // body is `intrinsics.abs(x)`) also get folded into Intr/Math nodes.
  def children(level: Int): Vector[ProgramPass] =
    if (level <= 0) Vector.empty
    else if (level == 1) baseline
    else
      Vector(printPass(Intrinsify), printPass(Specialisation), printPass(FnInline), printPass(Intrinsify)) ++ baseline
}

object PassRegistry {
  val definitions: Vector[PassDef] = Vector(
    PassDef.configured(FullOpt()),
    PassDef.singleton(ConstantFold),
    PassDef.singleton(DeadArgElimination),
    PassDef.singleton(DeadStructElimination),
    PassDef.singleton(FnInline),
    PassDef.singleton(Intrinsify),
    PassDef.singleton(MonoStruct),
    PassDef.singleton(Specialisation),
    PassDef.singleton(UnitExprElision),
    PassDef.singleton(VarReduce)
  )

  private val byName = definitions.map(d => d.name -> d).toMap

  def build(spec: p.PassSpec): Either[String, ProgramPass | BoundaryPass[?]] =
    byName.get(spec.name) match {
      case Some(defn) => PassArgs.from(spec).flatMap(defn.build).left.map(e => s"${spec.name}: $e")
      case None       => Left(s"unknown pass: ${spec.name}")
    }

  def build(pipeline: p.PassPipeline): Either[String, Vector[ProgramPass | BoundaryPass[?]]] =
    pipeline.steps.foldLeft[Either[String, Vector[ProgramPass | BoundaryPass[?]]]](Right(Vector.empty)) { (acc, spec) =>
      acc.flatMap(xs => build(spec).map(xs :+ _))
    }
}
