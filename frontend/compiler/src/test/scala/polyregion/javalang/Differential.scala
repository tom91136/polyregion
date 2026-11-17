package polyregion.javalang

import polyregion.ast.{Interpreter, PolyAST as p, *}

trait Differential extends munit.FunSuite {

  protected def subject: Differential.Subject

  protected def lower(name: String): p.Function =
    BytecodeRemapper.lower(subject.bytes, name, descriptorOf(name)).fold(throw _, identity)

  private def methodOf(name: String): java.lang.reflect.Method =
    subject.clazz.getDeclaredMethods
      .find(_.getName == name)
      .getOrElse(sys.error(s"no method `$name` on the fixture class"))

  protected def descriptorOf(name: String): String =
    net.bytebuddy.jar.asm.Type.getMethodDescriptor(methodOf(name))

  private def viaReflection(name: String, args: List[Any]): (Any, List[Any]) =
    Differential.invoke(methodOf(name), subject.receiver, args)

  protected def check(name: String, argSets: List[List[Any]])(implicit loc: munit.Location): Unit = {
    val fn = lower(name)
    argSets.foreach { args =>
      val (expected, expectedArgs) = viaReflection(name, args)
      val (actual, actualArgs)     = Differential.interpret(fn, args)
      val label                    = s"$name(${args.map(render).mkString(", ")})"
      (expected, actual) match {
        case (_, _) if fn.rtn == p.Type.Unit0 => ()
        case (e: java.lang.Double, a: Double) => assertEqualsDouble(a, e.doubleValue, 0d, label)
        case (e, a)                           => assertEquals(render(a), render(e), label)
      }
      expectedArgs.zip(actualArgs).foreach((e, a) => assertEquals(render(a), render(e), s"$label argument"))
    }
  }

  protected def render(a: Any): String = Differential.render(a)
}

object Differential {
  final case class Subject(bytes: Array[Byte], clazz: Class[?], receiver: Object)

  def staticsOf(loaded: JavaInMemory.Loaded, name: String): Subject =
    Subject(loaded.bytesOf(name), loaded.loadClass(name), null)

  def moduleOf(loaded: JavaInMemory.Loaded, name: String): Subject = {
    val cls = loaded.loadClass(s"$name$$")
    Subject(loaded.bytesOf(s"$name$$"), cls, cls.getField("MODULE$").get(null))
  }

  def invoke(m: java.lang.reflect.Method, receiver: Object, args: List[Any]): (Any, List[Any]) = {
    m.setAccessible(true)
    val fresh = args.map {
      case a if a.getClass.isArray =>
        val length = java.lang.reflect.Array.getLength(a)
        val copy   = java.lang.reflect.Array.newInstance(a.getClass.getComponentType, length)
        System.arraycopy(a, 0, copy, 0, length)
        copy
      case other => other
    }
    (m.invoke(receiver, fresh.map(_.asInstanceOf[Object])*), fresh)
  }

  def interpret(fn: p.Function, args: List[Any]): (Any, List[Any]) = {
    val vm = new Interpreter.Vm(p.Program(Some(fn), Nil, Nil))
    def buffer(len: Int, comp: p.Type, write: (Long, Int) => Unit): Long = {
      val addr = vm.allocOf(comp, len.toLong)
      (0 until len).foreach(i => write(addr + i * vm.sizeOf(comp), i))
      addr
    }
    inline def integral(t: p.Type, v: Long, inline back: Long => Any): (p.Type, Interpreter.V, () => Any) =
      (t, Interpreter.V.I(v), () => back(v))
    val bound = args.map {
      case x: Boolean => integral(p.Type.Bool1, if (x) 1L else 0L, _ != 0L)
      case x: Byte    => integral(p.Type.IntS8, x.toLong, _.toByte)
      case x: Short   => integral(p.Type.IntS16, x.toLong, _.toShort)
      case x: Char    => integral(p.Type.IntU16, x.toLong, _.toChar)
      case x: Int     => integral(p.Type.IntS32, x.toLong, _.toInt)
      case x: Long    => integral(p.Type.IntS64, x, identity)
      case x: Double  => (p.Type.Float64, Interpreter.V.D(x), () => x)
      case a: Array[Int] =>
        val comp = p.Type.IntS32
        val addr = buffer(a.length, comp, (at, i) => vm.store(at, Interpreter.V.I(a(i).toLong), comp))
        val read = () =>
          Array.tabulate(a.length) { i =>
            vm.load(addr + i * vm.sizeOf(comp), comp) match { case Interpreter.V.I(v) => v.toInt; case _ => 0 }
          }
        (p.Type.Ptr(comp, p.Type.Space.Global), Interpreter.V.I(addr), read)
      case a: Array[Long] =>
        val comp = p.Type.IntS64
        val addr = buffer(a.length, comp, (at, i) => vm.store(at, Interpreter.V.I(a(i)), comp))
        val read = () =>
          Array.tabulate(a.length) { i =>
            vm.load(addr + i * vm.sizeOf(comp), comp) match { case Interpreter.V.I(v) => v; case _ => 0L }
          }
        (p.Type.Ptr(comp, p.Type.Space.Global), Interpreter.V.I(addr), read)
      case a: Array[Double] =>
        val comp = p.Type.Float64
        val addr = buffer(a.length, comp, (at, i) => vm.store(at, Interpreter.V.D(a(i)), comp))
        val read = () =>
          Array.tabulate(a.length) { i =>
            vm.load(addr + i * vm.sizeOf(comp), comp) match { case Interpreter.V.D(v) => v; case _ => 0d }
          }
        (p.Type.Ptr(comp, p.Type.Space.Global), Interpreter.V.I(addr), read)
      case other => sys.error(s"unsupported test argument $other")
    }
    val out = vm.call(fn.name, bound.map((t, v, _) => (t, v)))
    val decoded = (fn.rtn, out) match {
      case (p.Type.Unit0, _)                    => ()
      case (p.Type.Bool1, Interpreter.V.I(v))   => v != 0
      case (p.Type.IntS8, Interpreter.V.I(v))   => v.toByte
      case (p.Type.IntS16, Interpreter.V.I(v))  => v.toShort
      case (p.Type.IntU16, Interpreter.V.I(v))  => v.toChar
      case (p.Type.IntS32, Interpreter.V.I(v))  => v.toInt
      case (p.Type.IntS64, Interpreter.V.I(v))  => v
      case (p.Type.Float64, Interpreter.V.D(v)) => v
      case (t, v)                               => sys.error(s"cannot decode ${t.repr} from $v")
    }
    (decoded, bound.map((_, _, read) => read()))
  }

  def render(a: Any): String = a match {
    case x: Array[Int]    => x.mkString("[", ",", "]")
    case x: Array[Long]   => x.mkString("[", ",", "]")
    case x: Array[Double] => x.mkString("[", ",", "]")
    case x: Char          => x.toInt.toString
    case x                => String.valueOf(x)
  }
}
