package polyregion.javalang

import net.bytebuddy.jar.asm.{ClassReader, ClassVisitor, Handle, Label, MethodVisitor, Opcodes}
import polyregion.javalang.Interop.absent

import scala.collection.mutable

class BytecodeOpcodeCensusSuite extends munit.FunSuite {

  private enum Use {
    case GetField, PutField, GetStatic, PutStatic
    case New, NewArray, MultiArray
    case Lambda, Concat, OtherDynamic
    case Virtual, Interface, Constructor, Special, Static
    case Blocker
  }

  import Use.*

  private val Allocation = Set(New, NewArray, MultiArray, Constructor)

  private val Tiers: List[(String, Set[Use])] = List(
    "static calls only" -> Set(Static),
    "+ instance fields" -> Set(Static, GetField, PutField),
    "+ static fields"   -> Set(Static, GetField, PutField, GetStatic, PutStatic),
    "+ allocation"      -> (Set(Static, GetField, PutField, GetStatic, PutStatic) ++ Allocation),
    "+ lambdas"         -> (Set(Static, GetField, PutField, GetStatic, PutStatic, Lambda) ++ Allocation),
    "+ virtual, interface, special calls" -> (Use.values.toSet - Concat - OtherDynamic - Blocker)
  )

  private class MethodUses extends MethodVisitor(Opcodes.ASM9) {
    val seen = mutable.Set.empty[Use]

    override def visitFieldInsn(op: Int, o: String, n: String, d: String): Unit = seen += (op match {
      case Opcodes.GETFIELD  => GetField
      case Opcodes.PUTFIELD  => PutField
      case Opcodes.GETSTATIC => GetStatic
      case _                 => PutStatic
    })

    override def visitTypeInsn(op: Int, t: String): Unit =
      if (op == Opcodes.NEW) seen += New
      else if (op == Opcodes.ANEWARRAY) seen += NewArray
      else if (op == Opcodes.INSTANCEOF) seen += Blocker

    override def visitIntInsn(op: Int, operand: Int): Unit = if (op == Opcodes.NEWARRAY) seen += NewArray

    override def visitMultiANewArrayInsn(d: String, dims: Int): Unit = seen += MultiArray

    override def visitInsn(op: Int): Unit =
      if (op == Opcodes.ATHROW || op == Opcodes.MONITORENTER) seen += Blocker

    override def visitTableSwitchInsn(min: Int, max: Int, dflt: Label, labels: Label*): Unit = seen += Blocker

    override def visitLookupSwitchInsn(dflt: Label, keys: Array[Int], labels: Array[Label]): Unit = seen += Blocker

    override def visitTryCatchBlock(s: Label, e: Label, h: Label, t: String): Unit = seen += Blocker

    override def visitInvokeDynamicInsn(n: String, d: String, bsm: Handle, args: Object*): Unit =
      seen += (bsm.getOwner match {
        case "java/lang/invoke/LambdaMetafactory"   => Lambda
        case "java/lang/invoke/StringConcatFactory" => Concat
        case _                                      => OtherDynamic
      })

    override def visitMethodInsn(op: Int, o: String, n: String, d: String, itf: Boolean): Unit =
      seen += (op match {
        case Opcodes.INVOKESTATIC    => Static
        case Opcodes.INVOKEVIRTUAL   => Virtual
        case Opcodes.INVOKEINTERFACE => Interface
        case _ if n == "<init>"      => Constructor
        case _                       => Special
      })
  }

  private class ClassUses extends ClassVisitor(Opcodes.ASM9) {
    val methods = mutable.ListBuffer.empty[MethodUses]

    override def visitMethod(a: Int, n: String, d: String, s: String, e: Array[String]): MethodVisitor =
      if ((a & (Opcodes.ACC_ABSTRACT | Opcodes.ACC_NATIVE)) != 0) absent
      else {
        val uses = MethodUses()
        methods += uses
        uses
      }
  }

  private def usesOf(bytes: Array[Byte]): List[Set[Use]] = {
    val classUses = ClassUses()
    ClassReader(bytes).accept(classUses, ClassReader.SKIP_DEBUG | ClassReader.SKIP_FRAMES)
    classUses.methods.toList.map(_.seen.toSet)
  }

  private def jdk(name: String): Option[Array[Byte]] = BytecodeResolver.runtimeClasses(name)

  private val JdkTargets = List(
    "java/lang/Math",
    "java/lang/StrictMath",
    "java/lang/Integer",
    "java/lang/Long",
    "java/lang/Double",
    "java/lang/Character",
    "java/util/Arrays",
    "java/util/ArrayList",
    "java/util/Objects",
    "java/util/Optional",
    "java/util/OptionalInt",
    "java/util/BitSet",
    "java/util/SplittableRandom",
    "java/util/stream/IntPipeline",
    "java/util/stream/Streams",
    "java/util/concurrent/atomic/AtomicInteger"
  )

  private val ScalaTargets = List(
    "scala/runtime/RichInt",
    "scala/runtime/RichDouble",
    "scala/runtime/RichLong",
    "scala/Tuple2",
    "scala/Option",
    "scala/Some",
    "scala/collection/immutable/Range",
    "scala/collection/immutable/Range$Inclusive",
    "scala/runtime/ScalaRunTime$",
    "scala/math/package$"
  )

  private def scalaClass(name: String): Option[Array[Byte]] =
    Option(classOf[scala.Option[?]].getClassLoader.getResourceAsStream(s"$name.class")).map { in =>
      try in.readAllBytes()
      finally in.close()
    }

  private def render(title: String, classes: List[Array[Byte]]): String = {
    val methods     = classes.flatMap(usesOf(_))
    val total       = math.max(methods.size, 1)
    def pct(n: Int) = f"${n * 100.0 / total}%5.1f%%"
    val perUse      = methods.flatten.groupMapReduce(identity)(_ => 1)(_ + _)
    val per = Use.values.toList.map { u =>
      val n = perUse.getOrElse(u, 0)
      f"    $n%7d  ${pct(n)}  uses ${u.toString}"
    }
    val cumulative = Tiers.map { (label, allowed) =>
      val n = methods.count(_.subsetOf(allowed))
      f"    $n%7d  ${pct(n)}  $label"
    }
    (s"opcode census: $title, ${methods.size} methods" :: "  methods using" :: per :::
      "  methods whose body needs at most" :: cumulative).mkString("\n")
  }

  private def scan(title: String, load: => List[Array[Byte]])(implicit loc: munit.Location): Unit =
    test(s"opcode census of $title") {
      val classes = load
      assert(classes.nonEmpty, s"no class of $title was found")
      println(render(title, classes))
    }

  scan("JDK stdlib targets", JdkTargets.flatMap(jdk(_)))
  scan("Scala stdlib targets", ScalaTargets.flatMap(scalaClass(_)))
  scan("scala-library", jarClasses(Census.locate(classOf[scala.collection.immutable.List[?]])))
  scan("guava", jarClasses(Census.locate(classOf[com.google.common.math.IntMath])))

  private def jarClasses(jar: java.io.File): List[Array[Byte]] = {
    val out = mutable.ListBuffer.empty[Array[Byte]]
    Census.eachClass(jar, Census.sample(Census.entriesOf(jar), 600))((_, bytes) => out += bytes)
    out.toList
  }
}
