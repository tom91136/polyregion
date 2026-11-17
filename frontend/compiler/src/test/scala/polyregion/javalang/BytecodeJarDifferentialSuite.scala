package polyregion.javalang

import net.bytebuddy.jar.asm.{Opcodes, Type as AsmType}
import polyregion.ast.{PolyAST as p, *}

import scala.util.Random

class BytecodeJarDifferentialSuite extends munit.FunSuite {

  override def munitTimeout: scala.concurrent.duration.Duration =
    scala.concurrent.duration.Duration(20, "min")

  // no float (interpreted at double width) and no arrays (arraylength is refused)
  private val Scalars = Set('Z', 'B', 'S', 'C', 'I', 'J', 'D')

  private def scalar(t: AsmType): Boolean = t.getDescriptor.length == 1 && Scalars(t.getDescriptor.charAt(0))

  private def eligible(name: String, desc: String): Boolean =
    name != "<init>" && name != "<clinit>" && {
      val args = AsmType.getArgumentTypes(desc)
      args.nonEmpty && args.length <= 4 && args.forall(scalar) && scalar(AsmType.getReturnType(desc))
    }

  private final case class Candidate(
      owner: String,
      name: String,
      desc: String,
      fn: p.Function,
      method: java.lang.reflect.Method,
      receiver: Object
  ) {
    override def toString: String = s"$owner.$name$desc"
  }

  private def resolve(
      owner: String,
      name: String,
      desc: String,
      static: Boolean
  ): Option[(java.lang.reflect.Method, Object)] =
    try {
      val cls = Class.forName(owner, false, getClass.getClassLoader)
      val m   = cls.getDeclaredMethods.find(m => m.getName == name && AsmType.getMethodDescriptor(m) == desc)
      val receiver =
        if (static) Some(null: Object)
        else
          try Some(cls.getField("MODULE$").get(null))
          catch { case _: Throwable => None }
      m.zip(receiver)
    } catch { case _: Throwable => None }

  private def discover(corpus: Census.Corpus): List[Candidate] = {
    val out = List.newBuilder[Candidate]
    Census.eachClass(corpus.jar, Census.entriesOf(corpus.jar)) { (_, bytes) =>
      val owner =
        try Census.ownerOf(bytes)
        catch { case _: Throwable => "" }
      val ms =
        try Census.methodsOf(bytes)
        catch { case _: Throwable => Nil }
      ms.filter((n, d, _) => eligible(n, d)).foreach { (name, desc, access) =>
        BytecodeRemapper.lower(bytes, name, desc) match {
          case Right(fn) =>
            resolve(owner, name, desc, (access & Opcodes.ACC_STATIC) != 0).foreach { (m, r) =>
              out += Candidate(owner, name, desc, fn, m, r)
            }
          case Left(_) => ()
        }
      }
    }
    out.result().sortBy(_.toString)
  }

  private val rng = new Random(Census.Seed)

  // no infinity: PolyRegion takes the target's d2i, not Java's saturating one
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
    case 'D' => Vector(0d, -0d, 1d, -1d, 0.5d, -2.25d, 3d, 1024d, -1024d, 1e6d, -1e6d, 1d / 3, Double.NaN)
    case _   => Vector.empty
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

  private final case class Outcome(checked: Int, skipped: Int, disagreed: List[String])

  private def exercise(c: Candidate): Outcome = {
    var checked   = 0
    var skipped   = 0
    val disagreed = List.newBuilder[String]
    argSets(c.desc, 512).foreach { args =>
      val expected =
        try Right(Differential.invoke(c.method, c.receiver, args)._1)
        catch { case e: Throwable => Left(e) }
      expected match {
        // the JVM throws where PolyRegion leaves the result undefined, so the input says nothing
        case Left(_) => skipped += 1
        case Right(e) =>
          checked += 1
          val where = s"$c(${args.map(Differential.render).mkString(", ")})"
          val actual =
            try Right(Differential.interpret(c.fn, args)._1)
            catch { case t: Throwable => Left(t) }
          actual match {
            case Left(t) => disagreed += s"$where: jvm=${Differential.render(e)} interpreter threw $t"
            case Right(a) =>
              val same = (e, a) match {
                case (x: java.lang.Double, y: Double) => java.lang.Double.compare(x.doubleValue, y) == 0
                case (x, y)                           => Differential.render(x) == Differential.render(y)
              }
              if (!same) disagreed += s"$where: jvm=${Differential.render(e)} interpreter=${Differential.render(a)}"
          }
      }
    }
    Outcome(checked, skipped, disagreed.result())
  }

  private def differential(name: String, marker: Class[?], floor: Int): Unit =
    test(s"executable methods discovered in $name agree with the JVM") {
      val corpus     = Census.Corpus(name, Census.locate(marker))
      val candidates = discover(corpus)
      var checked    = 0
      var skipped    = 0
      var abandoned  = List.empty[String]
      val disagreed  = List.newBuilder[String]
      candidates.foreach { c =>
        Census.bounded(30_000L)(exercise(c)) match {
          case None => abandoned = c.toString :: abandoned
          case Some(o) =>
            checked += o.checked
            skipped += o.skipped
            disagreed ++= o.disagreed
        }
      }
      val bad = disagreed.result()
      println(
        s"differential: $name, ${candidates.size} executable methods discovered, " +
          s"$checked inputs compared, $skipped skipped as undefined, ${abandoned.size} abandoned, ${bad.size} disagreed"
      )
      candidates.take(40).foreach(c => println(s"  $c"))
      if (candidates.sizeIs > 40) println(s"  ... and ${candidates.size - 40} more")
      assert(bad.isEmpty, bad.take(20).mkString("\n"))
      assert(abandoned.isEmpty, s"did not terminate: ${abandoned.mkString(", ")}")
      assert(candidates.sizeIs >= floor, s"only ${candidates.size} candidates, floor $floor")
    }

  differential("scala-library", classOf[scala.collection.immutable.List[?]], floor = 80)
  differential("scala3-compiler", classOf[dotty.tools.dotc.Driver], floor = 80)
  differential("guava", classOf[com.google.common.math.IntMath], floor = 20)
}
