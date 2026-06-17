package polyregion.ast.pass

import polyregion.ast.{Log, PolyAST as p}

import scala.collection.mutable
import scala.compiletime.{constValue, erasedValue, summonInline}
import scala.deriving.Mirror

final case class PassArgs(values: Map[String, String]) {
  def expectKnown(keys: Set[String]): Either[String, Unit] = {
    val unknown = values.keySet.diff(keys)
    if (unknown.isEmpty) Right(()) else Left(s"unexpected argument(s): ${unknown.toVector.sorted.mkString(", ")}")
  }
}

object PassArgs {
  def from(spec: p.PassSpec): Either[String, PassArgs] = {
    val duplicates = spec.args.groupBy(_.name).collect { case (k, xs) if xs.size > 1 => k }.toVector.sorted
    if (duplicates.nonEmpty) Left(s"${spec.name}: duplicate argument(s): ${duplicates.mkString(", ")}")
    else Right(PassArgs(spec.args.map(a => a.name -> a.value).toMap))
  }
}

trait PassArgValue[A] {
  def parse(key: String, value: String): Either[String, A]
}

object PassArgValue {
  given PassArgValue[String] = (_, value) => Right(value)
  given PassArgValue[Int]    = (key, value) => value.toIntOption.toRight(s"$key: expected Int")
  given PassArgValue[Long]   = (key, value) => value.toLongOption.toRight(s"$key: expected Long")
  given PassArgValue[Double] = (key, value) => value.toDoubleOption.toRight(s"$key: expected Double")
  given PassArgValue[Boolean] = (key, value) =>
    value match {
      case "true" | "1"  => Right(true)
      case "false" | "0" => Right(false)
      case _             => Left(s"$key: expected Boolean")
    }
}

trait PassArgCodec[A] {
  def parse(args: PassArgs, default: A): Either[String, A]
}

object PassArgCodec {

  inline def derived[A <: Product](using m: Mirror.ProductOf[A]): PassArgCodec[A] =
    mkCodec[A](m, labels[m.MirroredElemLabels], summonArgValues[m.MirroredElemTypes])

  private def mkCodec[A](
      m: Mirror.ProductOf[A],
      fieldNames: List[String],
      argValues: Array[PassArgValue[?]]
  ): PassArgCodec[A] = new PassArgCodec[A] {
    private val fieldNamesArr = fieldNames.toArray
    private val fieldSet      = fieldNames.toSet
    def parse(args: PassArgs, default: A): Either[String, A] =
      args.expectKnown(fieldSet).flatMap { _ =>
        val defProd     = default.asInstanceOf[Product]
        val n           = argValues.length
        val arr         = new Array[Any](n)
        var i           = 0
        var err: String = null
        while (i < n && err == null) {
          val key = fieldNamesArr(i)
          val head: Either[String, Any] = args.values.get(key) match {
            case Some(value) => argValues(i).asInstanceOf[PassArgValue[Any]].parse(key, value)
            case None        => Right(defProd.productElement(i))
          }
          head match {
            case Right(v) => arr(i) = v
            case Left(e)  => err = e
          }
          i += 1
        }
        if (err == null) Right(m.fromProduct(Tuple.fromArray(arr)))
        else Left(err)
      }
  }

  private inline def labels[T <: Tuple]: List[String] =
    inline erasedValue[T] match {
      case _: EmptyTuple => Nil
      case _: (h *: t)   => constValue[h].asInstanceOf[String] :: labels[t]
    }

  private inline def summonArgValues[T <: Tuple]: Array[PassArgValue[?]] =
    inline erasedValue[T] match {
      case _: EmptyTuple => Array.empty
      case _: (h *: t) =>
        val head = summonInline[PassArgValue[h]]
        val tail = summonArgValues[t]
        val out  = new Array[PassArgValue[?]](tail.length + 1)
        out(0) = head
        Array.copy(tail, 0, out, 1, tail.length)
        out
    }
}

object PassName {
  def derive(cls: Class[?]): String = {
    // getSimpleName returns "" for anonymous classes
    val raw = cls.getName.split('.').last.stripSuffix("$")
    raw.takeWhile(_ != '$')
  }

  def eventName(name: String): String =
    "polypass_" + name
      .flatMap(c => if (c.isUpper) "_" + c.toLower else if (c.isLetterOrDigit) c.toString else "_")
      .replaceAll("_+", "_")
      .stripPrefix("_")
      .stripSuffix("_")
}

trait ProgramPass extends ((p.Program, Log) => p.Program) {
  def name: String                                         = PassName.derive(getClass)
  def phase: p.PassPhase                                   = p.PassPhase.Initial
  def run(program: p.Program, ctx: PassContext): p.Program = apply(program, ctx.log)
}

trait BoundaryPass[A] extends ((p.Program, Log) => (A, p.Program)) {
  def name: String                                              = PassName.derive(getClass)
  def phase: p.PassPhase                                        = p.PassPhase.Initial
  def run(program: p.Program, ctx: PassContext): (A, p.Program) = apply(program, ctx.log)
}

final case class PassContext(log: Log, runner: PassRunner) {
  def run(pass: ProgramPass, program: p.Program): p.Program =
    runner.run(pass, program, this)

  def run[A](pass: BoundaryPass[A], program: p.Program): (A, p.Program) =
    runner.run(pass, program, this)
}

trait PassClock {
  def epochMillis(): Long
  def nanoTime(): Long
}

object PassClock {
  val system: PassClock = new PassClock {
    def epochMillis(): Long = System.currentTimeMillis()
    def nanoTime(): Long    = System.nanoTime()
  }
}

final class PassRunner(clock: PassClock = PassClock.system) {
  private val eventStack = mutable.ArrayBuffer.empty[mutable.ArrayBuffer[p.CompileEvent]]

  def run(pass: ProgramPass, program: p.Program, parent: PassContext): p.Program = {
    val log = parent.log.subLog(pass.name)
    timed(pass.name) {
      pass.run(program, PassContext(log, this))
    }
  }

  def run[A](pass: BoundaryPass[A], program: p.Program, parent: PassContext): (A, p.Program) = {
    val log = parent.log.subLog(pass.name)
    timed(pass.name) {
      pass.run(program, PassContext(log, this))
    }
  }

  def runPipeline(steps: Vector[ProgramPass | BoundaryPass[?]], program: p.Program, log: Log): p.PassRunResult = {
    val startEpoch = clock.epochMillis()
    val startNanos = clock.nanoTime()
    val rootItems  = mutable.ArrayBuffer.empty[p.CompileEvent]
    eventStack += rootItems
    val out =
      try {
        val ctx = PassContext(log, this)
        steps.foldLeft(program) {
          case (acc, pass: ProgramPass)     => run(pass, acc, ctx)
          case (acc, pass: BoundaryPass[?]) => run(pass, acc, ctx)._2
        }
      } catch {
        case t: Throwable =>
          eventStack.remove(eventStack.size - 1)
          throw t
      }
    val items = eventStack.remove(eventStack.size - 1).toList
    p.PassRunResult(out, p.CompileEvent(startEpoch, math.max(0L, clock.nanoTime() - startNanos), "polypass", "", items))
  }

  private def timed[A](name: String)(f: => A): A = {
    val startEpoch = clock.epochMillis()
    val startNanos = clock.nanoTime()
    val items      = mutable.ArrayBuffer.empty[p.CompileEvent]
    eventStack += items
    try f
    finally {
      eventStack.remove(eventStack.size - 1)
      if (eventStack.nonEmpty)
        eventStack.last += p.CompileEvent(
          startEpoch,
          math.max(0L, clock.nanoTime() - startNanos),
          PassName.eventName(name),
          "",
          items.toList
        )
    }
  }
}

final case class PassDef(
    name: String,
    phase: p.PassPhase,
    build: PassArgs => Either[String, ProgramPass | BoundaryPass[?]]
)

object PassDef {
  private type AnyPass = ProgramPass | BoundaryPass[?]

  private def head(pass: AnyPass): (String, p.PassPhase) = pass match {
    case x: ProgramPass     => (x.name, x.phase)
    case x: BoundaryPass[?] => (x.name, x.phase)
  }

  def singleton(pass: AnyPass): PassDef = {
    val (n, ph) = head(pass)
    PassDef(n, ph, _.expectKnown(Set.empty).map(_ => pass))
  }

  def configured[A <: Product & AnyPass](default: A)(using codec: PassArgCodec[A]): PassDef = {
    val (n, ph) = head(default)
    PassDef(n, ph, args => codec.parse(args, default))
  }
}

object PassPipelineParser {
  def parseStep(step: String): Either[String, p.PassSpec] = step match {
    case "" => Left("empty step")
    case s"$name($body)" =>
      if (body.contains('(') || body.contains(')')) Left(s"nested parentheses in args: '$step'")
      else
        validateName(name.trim).flatMap { n =>
          val args = body.trim
          if (args.isEmpty) Right(p.PassSpec(n, Nil))
          else
            args
              .split(',')
              .toList
              .foldRight(Right(Nil): Either[String, List[p.PassArg]])((raw, acc) =>
                parseArg(raw.trim).flatMap(a => acc.map(a :: _))
              )
              .map(p.PassSpec(n, _))
        }
    case n if n.contains('(') || n.contains(')') => Left(s"unbalanced parentheses: '$n'")
    case n                                       => validateName(n).map(p.PassSpec(_, Nil))
  }

  private val ReservedArgChars = Set(',', ';', '(', ')')

  private def parseArg(arg: String): Either[String, p.PassArg] = arg match {
    case s"$key=$value" =>
      val v = value.trim
      v.find(ReservedArgChars.contains) match {
        case Some(c) => Left(s"reserved character '$c' in arg value '$v'")
        case None    => validateName(key.trim).map(k => p.PassArg(k, v))
      }
    case _ => Left(s"expected key=value argument, got '$arg'")
  }

  private def validateName(name: String): Either[String, String] =
    if (name.matches("[A-Za-z_][A-Za-z0-9_]*")) Right(name) else Left(s"invalid identifier '$name'")
}

def printPass(pass: ProgramPass): ProgramPass = new ProgramPass {
  override def name  = pass.name
  override def phase = pass.phase
  // XXX `apply` is the Function2 contract; the active path is `run` (called via PassContext) which
  // threads the live runner. Direct `apply` only fires if someone calls the pass outside a runner.
  def apply(program: p.Program, l: Log): p.Program = run(program, PassContext(l, PassRunner()))
  override def run(program: p.Program, ctx: PassContext): p.Program = {
    val r  = pass.run(program, ctx)
    val sl = ctx.log.subLog(s"[${pass.name}]")
    sl.info("Structs", r.defs.map(_.repr)*)
    sl.info("Fns", r.functions.map(_.repr)*)
    sl.info("Entry", r.entry.repr)
    r
  }
}
