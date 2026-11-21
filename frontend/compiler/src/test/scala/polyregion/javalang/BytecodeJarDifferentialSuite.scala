package polyregion.javalang

import net.bytebuddy.jar.asm.{Opcodes, Type as AsmType}
import polyregion.ast.PolyAST as p

import java.util.jar.JarFile
import scala.util.{Random, Using}

class BytecodeJarDifferentialSuite extends munit.FunSuite {

  override def munitTimeout: scala.concurrent.duration.Duration =
    scala.concurrent.duration.Duration(20, "min")

  // no float (interpreted at double width) and no arrays (arraylength is refused)
  private val Scalars = Set('Z', 'B', 'S', 'C', 'I', 'J', 'D')

  private def scalar(t: AsmType): Boolean = t.getDescriptor.length == 1 && Scalars(t.getDescriptor.charAt(0))

  private def eligible(name: String, desc: String): Boolean =
    name != "<init>" && name != "<clinit>" && {
      val args = AsmType.getArgumentTypes(desc)
      args.nonEmpty && args.length <= 4 && args.forall(scalar(_)) && scalar(AsmType.getReturnType(desc))
    }

  private case class Candidate(
      owner: String,
      name: String,
      desc: String,
      fn: p.Function,
      callees: List[p.Function],
      method: java.lang.reflect.Method,
      receiver: Option[Object]
  ) {
    override def toString: String = s"$owner.$name$desc"
  }

  private def resolve(
      owner: String,
      name: String,
      desc: String,
      static: Boolean
  ): Either[String, (java.lang.reflect.Method, Option[Object])] =
    try {
      val cls = Class.forName(owner, false, getClass.getClassLoader)
      cls.getDeclaredMethods.find(m => m.getName == name && AsmType.getMethodDescriptor(m) == desc) match {
        case None              => Left("no reflective method")
        case Some(m) if static => Right(m -> None)
        case Some(m) =>
          try Right(m -> Some(cls.getField("MODULE$").get(cls)))
          catch { case _: Throwable => Left(NoReceiver) }
      }
    } catch { case t: Throwable => Left(s"class not loadable: $t") }

  private def classesOf(corpus: Census.Corpus): BytecodeResolver.ClassBytes = owner =>
    Using(JarFile(corpus.jar)) { jar =>
      Option(jar.getEntry(BytecodeResolver.classFile(owner))).map(e => jar.getInputStream(e).readAllBytes())
    }.toOption.flatten.orElse(BytecodeResolver.runtimeClasses(owner))

  // a method the lowerer refuses for one of these reasons is a known gap of the tier, any other failure is a defect
  private val KnownRefusals: List[(String, String)] = List(
    "Unsupported call"                    -> "instance, interface, dynamic and object-typed calls",
    "Unsupported field access"            -> "fields need receiver structs",
    "Unsupported allocation"              -> "no device heap",
    "Unsupported array allocation"        -> "no device heap",
    "Unsupported `String` constant"       -> "no device-side String",
    "Unsupported `null`"                  -> "no device heap",
    "Unsupported `length`"                -> "array lengths travel as separate arguments",
    "Unsupported `instanceof`"            -> "no device heap",
    "Unsupported string concatenation"    -> "no device-side String",
    "Unsupported `throw`"                 -> "kernels cannot throw",
    "Unsupported `try`/`catch`"           -> "kernels cannot throw",
    "Unsupported `finally`"               -> "kernels cannot throw",
    "Unsupported `synchronized`"          -> "kernels have no monitors",
    "Unsupported `switch`"                -> "not lowered yet",
    "Unsupported expression"              -> "decompiler expression with no lowering",
    "Unsupported statement"               -> "decompiler statement with no lowering",
    "Unsupported constant"                -> "constant with no device form",
    "Vineflower could not recover"        -> "unstructured control flow",
    "the method keeps an unstructured"    -> "unstructured control flow",
    "Vineflower reported"                 -> "the decompiler gave up on the body",
    "Vineflower gave up"                  -> "the decompiler gave up on the body",
    "Cannot lower the JVM type"           -> "object types do not travel to a device",
    "is abstract or native"               -> "no body to lower",
    "is not static and uses its receiver" -> "instance methods need a receiver struct"
  )

  private def knownRefusal(message: String): Boolean = KnownRefusals.exists((prefix, _) => message.contains(prefix))

  private val NoReceiver = "an instance method of a class with no singleton receiver"

  private case class Discovery(
      eligible: Int,
      candidates: List[Candidate],
      refused: List[(String, String)],
      unexecutable: List[(String, String)]
  )

  private def discover(corpus: Census.Corpus): Discovery = {
    val out          = List.newBuilder[Candidate]
    val refused      = List.newBuilder[(String, String)]
    val unexecutable = List.newBuilder[(String, String)]
    val eligibleRefs = List.newBuilder[MethodRef]
    val classes      = classesOf(corpus)
    Census.eachClass(corpus.jar, Census.entriesOf(corpus.jar)) { (_, bytes) =>
      val owner =
        try Census.ownerOf(bytes)
        catch { case _: Throwable => "" }
      val ms =
        try Census.methodsOf(bytes)
        catch { case _: Throwable => Nil }
      ms.filter((n, d, _) => eligible(n, d)).foreach { (name, desc, access) =>
        val ref = MethodRef(owner, name, desc)
        eligibleRefs += ref
        BytecodeResolver.resolve(List(ref), classes) match {
          case Left(e) => refused += ref.repr -> e.getMessage
          case Right(BytecodeResolver.Resolved(fn :: callees, _)) =>
            resolve(owner, name, desc, (access & Opcodes.ACC_STATIC) != 0) match {
              case Right((m, r)) => out += Candidate(owner, name, desc, fn, callees, m, r)
              case Left(why)     => unexecutable += ref.repr -> why
            }
          case Right(_) => refused += ref.repr -> "the resolver returned no function"
        }
      }
    }
    Discovery(eligibleRefs.result().size, out.result().sortBy(_.toString), refused.result(), unexecutable.result())
  }

  private val rng = Random(Census.Seed)

  // XXX narrow pools are written as ints because RedundantParens turns `(-1).toByte` into an Int
  private val ints: Vector[Int]   = Vector(0, 1, -1, 2, 3, 7, -13, 16, 31, 32, 63, 127, 255, 1000, -1000)
  private val longs: Vector[Long] = Vector(0, 1, -1, 2, 7, -13, 31, 63, 255, 1L << 31, 1L << 40, -(1L << 40))

  private def pool(d: Char): Vector[Any] = d match {
    case 'Z' => Vector(false, true)
    case 'B' => Vector(0, 1, -1, 42, -128, 127).map(_.toByte)
    case 'S' => Vector(0, 1, -1, 300, -300, -32768, 32767).map(_.toShort)
    case 'C' => Vector(0, 1, 97, 128, 32768, 65535).map(_.toChar)
    case 'I' => ints ++ Vector(Int.MaxValue, Int.MinValue, Int.MaxValue - 1, Int.MinValue + 1)
    case 'J' => longs ++ Vector(Long.MaxValue, Long.MinValue, Long.MaxValue - 1, Long.MinValue + 1)
    case 'D' =>
      Vector(
        0d,
        -0d,
        1d,
        -1d,
        0.5d,
        -2.25d,
        3d,
        1024d,
        -1024d,
        1e6d,
        -1e6d,
        1d / 3,
        3e9d,
        -3e9d,
        1e19d,
        Double.PositiveInfinity,
        Double.NegativeInfinity,
        Double.NaN
      )
    case _ => Vector.empty
  }

  private def argSets(desc: String, count: Int): List[List[Any]] = {
    val pools = AsmType.getArgumentTypes(desc).toList.map(t => pool(t.getDescriptor.charAt(0)))
    val exhaustive =
      if (pools.map(_.size.toLong).product <= count)
        pools
          .foldLeft(List(List.empty[Any])) { (acc, ps) =>
            val values = ps.toList
            for {
              a <- acc
              v <- values
            } yield v :: a
          }
          .map(_.reverse)
      else Nil
    if (exhaustive.nonEmpty) exhaustive
    else List.fill(count)(pools.map(ps => ps(rng.nextInt(ps.size))))
  }

  private case class Outcome(checked: Int, skipped: Int, disagreed: List[String])

  private enum Verdict {
    case Skipped
    case Agreed
    case Disagreed(report: String)
  }

  private def compare(c: Candidate, args: List[Any]): Verdict = {
    val expected =
      try Census.bounded(2000L)(Differential.invoke(c.method, c.receiver, args).result).toRight(())
      catch { case e: Throwable => Left(e) }
    expected match {
      // the JVM throws, or never returns, where PolyRegion leaves the result undefined, so the input says nothing
      case Left(_) => Verdict.Skipped
      case Right(e) =>
        val where = s"$c(${args.map(Differential.render(_)).mkString(", ")})"
        val actual =
          try Right(Differential.interpret(c.fn, args, c.callees).result)
          catch { case t: Throwable => Left(t) }
        actual match {
          case Left(t) => Verdict.Disagreed(s"$where: jvm=${Differential.render(e)} interpreter threw $t")
          case Right(a) =>
            val same = (e, a) match {
              case (x: Double, y: Double) => java.lang.Double.compare(x, y) == 0
              case (x, y)                 => Differential.render(x) == Differential.render(y)
            }
            if (same) Verdict.Agreed
            else Verdict.Disagreed(s"$where: jvm=${Differential.render(e)} interpreter=${Differential.render(a)}")
        }
    }
  }

  private def exercise(c: Candidate): Outcome = {
    val verdicts = argSets(c.desc, 512).map(compare(c, _))
    Outcome(
      verdicts.count(_ != Verdict.Skipped),
      verdicts.count(_ == Verdict.Skipped),
      verdicts.collect { case Verdict.Disagreed(report) => report }
    )
  }

  private def differential(name: String, marker: Class[?]): Unit =
    test(s"executable methods discovered in $name agree with the JVM") {
      val corpus                                                      = Census.Corpus(name, Census.locate(marker))
      val Discovery(eligibleCount, candidates, refused, unexecutable) = discover(corpus)
      val results   = candidates.map(c => c -> Census.bounded(30_000L)(exercise(c)))
      val outcomes  = results.collect { case (_, Some(o)) => o }
      val abandoned = results.collect { case (c, None) => c.toString }.reverse
      val checked   = outcomes.map(_.checked).sum
      val skipped   = outcomes.map(_.skipped).sum
      val bad       = outcomes.flatMap(_.disagreed)
      println(
        s"differential: $name, $eligibleCount eligible, ${candidates.size} executable methods discovered, " +
          s"${refused.size} refused, ${unexecutable.size} not executable, " +
          s"$checked inputs compared, $skipped skipped as undefined, ${abandoned.size} abandoned, ${bad.size} disagreed"
      )
      refused
        .groupMapReduce((_, why) => KnownRefusals.find((prefix, _) => why.contains(prefix)).fold("unknown")(_._1))(_ =>
          1
        )(_ + _)
        .toList
        .sortBy(-_._2)
        .foreach((why, n) => println(s"  refused x$n: $why"))
      candidates.take(40).foreach(c => println(s"  $c"))
      if (candidates.sizeIs > 40) println(s"  ... and ${candidates.size - 40} more")
      val unknown = refused.filterNot((_, why) => knownRefusal(why))
      assert(
        unknown.isEmpty,
        s"${unknown.size} lowering failures outside the known refusals:\n" + unknown.take(20).mkString("\n")
      )
      val unexpected = unexecutable.filterNot((_, why) => why == NoReceiver)
      assert(
        unexpected.isEmpty,
        s"${unexpected.size} lowered methods the JVM cannot run:\n" + unexpected.take(20).mkString("\n")
      )
      assertEquals(candidates.size + refused.size + unexecutable.size, eligibleCount)
      assert(bad.isEmpty, bad.take(20).mkString("\n"))
      assert(abandoned.isEmpty, s"did not terminate: ${abandoned.mkString(", ")}")
      assert(candidates.nonEmpty, s"no method of $name lowered")
      assert(checked > 0, s"no input of $name was compared")
    }

  differential("scala-library", classOf[scala.collection.immutable.List[?]])
  differential("scala3-compiler", classOf[dotty.tools.dotc.Driver])
  differential("guava", classOf[com.google.common.math.IntMath])
}
