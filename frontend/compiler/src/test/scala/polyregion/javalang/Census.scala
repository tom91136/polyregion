package polyregion.javalang

import net.bytebuddy.jar.asm.{ClassReader, ClassVisitor, MethodVisitor, Opcodes}
import polyregion.ast.*
import polyregion.javalang.Interop.absent

import java.io.File
import java.nio.file.Paths
import java.util.concurrent.atomic.{AtomicInteger, AtomicReference}
import java.util.concurrent.{ConcurrentHashMap, CountDownLatch, TimeUnit}
import java.util.jar.JarFile
import scala.annotation.tailrec
import scala.collection.mutable
import scala.jdk.CollectionConverters.*
import scala.util.Random
import scala.util.Using

object Census {

  case class Corpus(name: String, jar: File)

  case class Target(owner: String, name: String, desc: String) {
    override def toString: String = s"$owner.$name$desc"
  }

  case class Report(
      corpus: Corpus,
      classes: Int,
      attempted: Int,
      lowered: Int,
      refusals: List[(String, Int)],
      crashes: List[(Target, Throwable)],
      stalled: Option[Target],
      elapsedMs: Long
  ) {

    def render: String = {
      val head = List(
        s"census: ${corpus.name} (${corpus.jar.getName})",
        f"  classes    ${classes}%7d",
        f"  methods    ${attempted}%7d",
        f"  lowered    ${lowered}%7d  (${lowered * 100.0 / math.max(attempted, 1)}%.2f%%)",
        f"  refused    ${attempted - lowered - crashes.size}%7d",
        f"  crashed    ${crashes.size}%7d",
        f"  elapsed    ${elapsedMs}%7d ms"
      )
      val reasons = "  refusal reasons, most frequent first:" ::
        refusals.map((reason, n) => f"    $n%7d  $reason")
      val stall = stalled.toList.map(t => s"  STALLED on $t")
      val bad   = crashes.map((t, e) => s"  CRASHED $t: ${describe(e)}")
      (head ::: reasons ::: stall ::: bad).mkString("\n")
    }
  }

  private val Quoted   = "`[^`]*`".r
  private val Digits   = "[0-9]+".r
  private val Specific = ".()/[];,<> ".toSet

  private def isSymbol(s: String): Boolean = s.isEmpty || s.exists(Specific)

  def bucket(e: Throwable): String = e match {
    case c: CompilerException =>
      val message = Option(c.getMessage).getOrElse("<no message>").linesIterator.next()
      val quoted  = Quoted.findAllMatchIn(message).toList
      val ends    = 0 :: quoted.map(_.end)
      val leading = quoted.zip(ends).map { (m, last) =>
        Digits.replaceAllIn(message.substring(last, m.start), "#") +
          (if (isSymbol(message.substring(m.start + 1, m.end - 1))) "`_`" else m.matched)
      }
      leading.mkString + Digits.replaceAllIn(message.substring(ends.last), "#")
    case other => describe(other)
  }

  def describe(e: Throwable): String = {
    val frame = e.getStackTrace.headOption.fold("")(f => s" at ${f.getClassName}.${f.getMethodName}:${f.getLineNumber}")
    s"${e.getClass.getName}: ${Option(e.getMessage).getOrElse("")}$frame"
  }

  def isRefusal(e: Throwable): Boolean = e.isInstanceOf[CompilerException]

  def locate(marker: Class[?]): File =
    Paths.get(marker.getProtectionDomain.getCodeSource.getLocation.toURI).toFile

  private def classEntries(jar: File): List[String] =
    Using.resource(JarFile(jar)) { f =>
      f.entries.asScala
        .map(_.getName)
        .filter(n => n.endsWith(".class") && !n.endsWith("module-info.class") && !n.startsWith("META-INF/"))
        .toList
        .sorted
    }

  // fixed seed over a sorted input keeps the sample identical across machines
  val Seed: Long = 0x50414c59L

  def sample(entries: List[String], limit: Int): List[String] =
    if (entries.sizeIs <= limit) entries else Random(Seed).shuffle(entries).take(limit).sorted

  private class MethodCollector extends ClassVisitor(Opcodes.ASM9) {
    val methods = mutable.ListBuffer.empty[(String, String, Int)]

    override def visitMethod(a: Int, n: String, d: String, s: String, e: Array[String]): MethodVisitor = {
      if ((a & (Opcodes.ACC_ABSTRACT | Opcodes.ACC_NATIVE)) == 0) methods += ((n, d, a))
      absent
    }
  }

  def methodsOf(bytes: Array[Byte]): List[(String, String, Int)] = {
    val collector = MethodCollector()
    ClassReader(bytes).accept(collector, ClassReader.SKIP_CODE | ClassReader.SKIP_DEBUG | ClassReader.SKIP_FRAMES)
    collector.methods.toList
  }

  def ownerOf(bytes: Array[Byte]): String = ClassReader(bytes).getClassName.replace('/', '.')

  def eachClass[A](jar: File, entries: List[String])(f: (String, Array[Byte]) => Unit): Unit =
    Using.resource(JarFile(jar)) { j =>
      entries.foreach { entry =>
        val bytes =
          try Using.resource(j.getInputStream(j.getEntry(entry)))(_.readAllBytes())
          catch { case _: Throwable => Array.emptyByteArray }
        if (bytes.nonEmpty) f(entry, bytes)
      }
    }

  def entriesOf(jar: File): List[String] = classEntries(jar)

  def bounded[A](millis: Long)(f: => A): Option[A] = {
    val box  = AtomicReference[Either[Throwable, A]]()
    val done = CountDownLatch(1)
    val body: Runnable = () =>
      try box.set(Right(f))
      catch { case e: Throwable => box.set(Left(e)) }
      finally done.countDown()
    val worker = Thread(body, "census-bounded")
    worker.setDaemon(true)
    worker.start()
    if (!done.await(millis, TimeUnit.MILLISECONDS)) None
    else
      box.get match {
        case Right(a) => Some(a)
        case Left(e)  => throw e
      }
  }

  private case class Heartbeat(target: Option[Target], stamp: Long)

  private class Progress {
    val heartbeat = AtomicReference(Heartbeat(None, System.nanoTime))
    val classes   = AtomicInteger()
    val attempted = AtomicInteger()
    val lowered   = AtomicInteger()
    val refusals  = ConcurrentHashMap[String, AtomicInteger]()
    val crashes   = java.util.Collections.synchronizedList(java.util.ArrayList[(Target, Throwable)]())
  }

  // Vineflower decompiles under a process-wide monitor, so a stuck method cannot be abandoned and retried
  def scan(corpus: Corpus, limit: Int, stallBudgetMs: Long = 120_000L): Report = {
    val entries  = sample(classEntries(corpus.jar), limit)
    val progress = Progress()
    val done     = CountDownLatch(1)
    val started  = System.nanoTime

    def one(owner: String, bytes: Array[Byte], name: String, desc: String): Unit = {
      val target = Target(owner, name, desc)
      progress.heartbeat.set(Heartbeat(Some(target), System.nanoTime))
      progress.attempted.incrementAndGet()
      val outcome =
        try BytecodeRemapper.lower(bytes, name, desc)
        catch { case e: Throwable => Left(e) }
      outcome match {
        case Right(_) => progress.lowered.incrementAndGet()
        case Left(e) if isRefusal(e) =>
          progress.refusals.computeIfAbsent(bucket(e), _ => AtomicInteger()).incrementAndGet()
        case Left(e) => progress.crashes.add((target, e))
      }
    }

    val body: Runnable = () =>
      try
        eachClass(corpus.jar, entries) { (entry, bytes) =>
          progress.classes.incrementAndGet()
          val owner =
            try ownerOf(bytes)
            catch { case _: Throwable => entry }
          val methods =
            try methodsOf(bytes)
            catch { case _: Throwable => Nil }
          methods.foreach((name, desc, _) => one(owner, bytes, name, desc))
        }
      catch { case e: Throwable => progress.crashes.add((Target(corpus.name, "<scan>", ""), e)) }
      finally done.countDown()
    val worker = Thread(body, s"census-${corpus.name}")
    worker.setDaemon(true)
    worker.start()

    @tailrec def watch(): Option[Target] =
      if (done.getCount == 0) None
      else if (
        !done.await(500, TimeUnit.MILLISECONDS) &&
        (System.nanoTime - progress.heartbeat.get.stamp) / 1000000L > stallBudgetMs
      )
        progress.heartbeat.get.target match {
          case None  => watch()
          case stuck => stuck
        }
      else watch()
    val stalled = watch()

    Report(
      corpus = corpus,
      classes = progress.classes.get,
      attempted = progress.attempted.get,
      lowered = progress.lowered.get,
      refusals = progress.refusals.asScala.map((k, v) => (k, v.get)).toList.sortBy((k, n) => (-n, k)),
      crashes = progress.crashes.asScala.toList,
      stalled = stalled,
      elapsedMs = (System.nanoTime - started) / 1000000L
    )
  }
}
