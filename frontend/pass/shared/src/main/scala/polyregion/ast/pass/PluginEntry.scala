package polyregion.ast.pass

import polyregion.ast.{Log, MsgPack, PolyAST as p}

trait PluginEntry {
  def passNames: Vector[String]
  def passDescr(name: String): Option[String] = None

  def runSteps(steps: Vector[String], program: p.Program, log: Log): p.PassRunResult = {
    val specs = steps.zipWithIndex
      .foldRight(Right(Nil): Either[String, List[p.PassSpec]]) { case ((s, idx), acc) =>
        PassPipelineParser.parseStep(s.trim).left.map(e => s"step ${idx + 1}: $e").flatMap(spec => acc.map(spec :: _))
      }
      .fold(e => throw IllegalArgumentException(e), identity)
    val passes = PassRegistry.build(p.PassPipeline(specs)).fold(e => throw IllegalArgumentException(e), identity)
    PassRunner(PluginEntry.clock).runPipeline(passes, program, log)
  }

  def runStepsMsgpack(steps: Vector[String], inBytes: Array[Byte]): Array[Byte] = {
    val clock     = PluginEntry.clock
    val rootEpoch = clock.epochMillis()
    val rootNanos = clock.nanoTime()
    val (prog, deserialiseEvent) =
      PluginEntry.timed(clock, "polyast_msgpack_deserialise", s"bytes=${inBytes.length}") {
        MsgPack.decodeInput[p.Program](MsgPack.ArrayByteInput(inBytes)).fold(throw _, identity)
      }
    val passRun = runSteps(steps, prog, PluginEntry.NoopLog)
    val rootEvent = p.CompileEvent(
      rootEpoch,
      math.max(0L, clock.nanoTime() - rootNanos),
      "PolyPass",
      steps.mkString(","),
      deserialiseEvent :: passRun.event.items
    )
    val out = MsgPack.ArrayByteOutput()
    MsgPack.encodeTo(p.PassRunResult(passRun.program, rootEvent), out)
    out.toByteArray
  }
}

object PluginEntry {
  val clock: PassClock = PassClock.system

  object NoopLog extends Log {
    def info(message: String, details: String*): Unit = ()
    def subLog(name: String): Log                     = this
  }

  def timed[A](clock: PassClock, name: String, data: String)(f: => A): (A, p.CompileEvent) = {
    val epoch = clock.epochMillis()
    val start = clock.nanoTime()
    val out   = f
    out -> p.CompileEvent(epoch, math.max(0L, clock.nanoTime() - start), name, data, Nil)
  }
}

object DefaultPlugin extends PluginEntry {
  val passNames: Vector[String] = PassRegistry.definitions.map(_.name)
}
