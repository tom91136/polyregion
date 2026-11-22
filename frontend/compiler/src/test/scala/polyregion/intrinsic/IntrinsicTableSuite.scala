package polyregion.intrinsic

import polyregion.ast.Interpreter.{V, Vm}
import polyregion.ast.pass.{PluginEntry, Verify}
import polyregion.ast.Traversal.*
import polyregion.ast.{Interpreter, PolyAST as p, *, given}

import java.lang.reflect.{InvocationTargetException, Method}

class IntrinsicTableSuite extends munit.FunSuite {

  private def keyOf(s: String) = Key.parse(s)

  private val ints: List[Int] =
    List(Int.MinValue, Int.MaxValue, 0, 1, -1, 2, -2, 7, -7, 42, 0x55555555, 0x0f0f0f0f, 65536, -65536, 1234567)
  private val longs: List[Long] =
    List(Long.MinValue, Long.MaxValue, 0L, 1L, -1L, 2L, -2L, 7L, -7L, 42L, 0x5555555555555555L, 1L << 40, -(1L << 40))
  private val floats: List[Float] = List(
    0f,
    -0f,
    1f,
    -1f,
    0.5f,
    -0.5f,
    1.5f,
    -1.5f,
    2.5f,
    -2.5f,
    3.25f,
    100f,
    0.49999997f,
    8388609f,
    3.4028235e38f,
    Float.NaN,
    Float.PositiveInfinity,
    Float.NegativeInfinity,
    1.4e-45f,
    1.17549435e-38f,
    1e10f,
    -3e9f
  )
  private val doubles: List[Double] = List(
    0d,
    -0d,
    1d,
    -1d,
    0.5d,
    -0.5d,
    1.5d,
    -1.5d,
    2.5d,
    -2.5d,
    3.25d,
    100d,
    0.49999999999999994,
    4503599627370497d,
    1.7976931348623157e308,
    Double.NaN,
    Double.PositiveInfinity,
    Double.NegativeInfinity,
    4.9e-324,
    2.2250738585072014e-308,
    1e19,
    -1e19,
    1.23,
    -0.12
  )

  private def samples(t: p.Type): List[Any] = t match {
    case p.Type.IntS32  => ints
    case p.Type.IntS64  => longs
    case p.Type.Float32 => floats
    case p.Type.Float64 => doubles
    case other          => fail(s"no samples for ${other.repr}")
  }

  private val intInputs    = ints.map(List(_))
  private val longInputs   = longs.map(List(_))
  private val rotateAmount = List[Any](0, 1, 5, 31, 32, 33, -1, -31, 64)

  private def cross(xs: List[List[Any]]): List[List[Any]] = xs match {
    case Nil => List(Nil)
    case head :: rest =>
      for {
        h <- head
        t <- cross(rest)
      } yield h :: t
  }

  private def interpreted(a: Any, t: p.Type): (p.Type, V) = (
    t,
    a match {
      case x: Int    => V.I(x.toLong)
      case x: Long   => V.I(x)
      case x: Float  => V.D(x.toDouble)
      case x: Double => V.D(x)
    }
  )

  private def javaClass(t: p.Type): Class[?] = t match {
    case p.Type.IntS32  => classOf[Int]
    case p.Type.IntS64  => classOf[Long]
    case p.Type.Float32 => classOf[Float]
    case p.Type.Float64 => classOf[Double]
    case other          => fail(s"no class for ${other.repr}")
  }

  private def jvm(key: Key, params: List[p.Type]): Method =
    Class.forName(key.owner.replace('/', '.')).getMethod(key.name, params.map(javaClass)*)

  private def normalise(a: Any): Any = a match {
    case x: Float  => java.lang.Float.floatToIntBits(x)
    case x: Double => java.lang.Double.doubleToLongBits(x)
    case x         => x
  }

  private def observed(v: V, rtn: p.Type): Any = rtn match {
    case p.Type.IntS32  => v.asI.toInt
    case p.Type.IntS64  => v.asI
    case p.Type.Float32 => java.lang.Float.floatToIntBits(v.asD.toFloat)
    case p.Type.Float64 => java.lang.Double.doubleToLongBits(v.asD)
    case p.Type.Bool1   => v.asI != 0
    case other          => fail(s"cannot observe ${other.repr}")
  }

  test("look up by owner, name and descriptor") {
    assert(IntrinsicTable.lookup(keyOf("java/lang/Math.sqrt(D)D")).isDefined)
    assert(IntrinsicTable.lookup(keyOf("java/lang/StrictMath.sqrt(D)D")).isDefined)
    assert(IntrinsicTable.lookup(keyOf("java/lang/Long.numberOfLeadingZeros(J)I")).isDefined)
    assert(IntrinsicTable.lookup(keyOf("java/lang/Math.sqrt(F)F")).isEmpty)
    assert(IntrinsicTable.lookup(keyOf("java/lang/Math.noSuchMethod(D)D")).isEmpty)
    assert(IntrinsicTable.lookup(keyOf("scala/math/package$.nope(D)D")).isEmpty)
  }

  test("key parsing round trips") {
    val key = keyOf("java/lang/Math.floorMod(JI)I")
    assertEquals(key, Key("java/lang/Math", "floorMod", "(JI)I"))
    assertEquals(key.repr, "java/lang/Math.floorMod(JI)I")
    assertEquals(key.parameters, Some(List(p.Type.IntS64, p.Type.IntS32)))
    assertEquals(key.result, Some(p.Type.IntS32))
  }

  test("no duplicate keys") {
    val duplicated = IntrinsicTable.entries.map(_._1).groupBy(identity).collect { case (k, v) if v.size > 1 => k.repr }
    assertEquals(duplicated.toList, Nil)
    assert(IntrinsicTable.entries.nonEmpty)
  }

  test("every key names a method that exists with that descriptor") {
    val missing = IntrinsicTable.entries.map(_._1).flatMap { key =>
      val (params, _) = Descriptor.parse(key.descriptor).get
      try { jvm(key, params); None }
      catch { case _: NoSuchMethodException => Some(key.repr) }
    }
    assertEquals(missing.toList, Nil)
  }

  test("every entry lowers to verifiable PolyAST") {
    val failures = IntrinsicTable.entries.map(_._1).flatMap { key =>
      val fn      = IntrinsicTable.function(key).getOrElse(fail(s"no function for ${key.repr}"))
      val program = p.Program(None, List(fn), Nil)
      Verify(program, PluginEntry.NoopLog, verifyFunction = true)
        .flatMap((_, errors) => errors.map(e => s"${key.repr}: $e"))
    }
    assertEquals(failures.toList, Nil)
  }

  test("every entry agrees with the JVM on sampled inputs") {
    val mismatches = IntrinsicTable.entries.map(_._1).flatMap { key =>
      val (params, rtn) = Descriptor.parse(key.descriptor).get
      val fn            = IntrinsicTable.function(key).get
      val vm            = Vm(p.Program(None, List(fn), Nil))
      val method = jvm(if (key.owner == "java/lang/StrictMath") key.copy(owner = "java/lang/Math") else key, params)
      val inputs =
        if (key.name.startsWith("floor") && key.name.endsWith("Mod") || key.name == "floorDiv")
          cross(params.map(samples)).filterNot(_.last.asInstanceOf[Number].longValue == 0)
        else if (key.name == "intBitsToFloat") intInputs
        else if (key.name == "longBitsToDouble") longInputs
        else if (key.name.startsWith("rotate"))
          cross(List(samples(params.head), rotateAmount))
        else cross(params.map(samples))
      inputs.flatMap { args =>
        val expected =
          try Right(normalise(method.invoke(null, args.map(_.asInstanceOf[AnyRef])*)))
          catch { case e: InvocationTargetException => Left(e.getCause.toString) }
        val actual =
          try Right(observed(vm.call(fn.name, args.zip(params).map((a, t) => interpreted(a, t))), rtn))
          catch { case e: Interpreter.AssertionRaised => Left(s"java.lang.ArithmeticException: ${e.message}") }
        Option.when(expected != actual)(s"${key.repr}(${args.mkString(", ")}): expected $expected, got $actual")
      }
    }
    assertEquals(mismatches.take(40).toList, Nil)
  }

  test("StrictMath keeps only the exactly specified entries") {
    val strict = IntrinsicTable.entries.map(_._1).filter(_.owner == "java/lang/StrictMath").map(_.name).toSet
    assert(strict.contains("sqrt") && strict.contains("floor") && strict.contains("min"), strict.toString)
    val inexact =
      Set("sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh", "exp", "expm1", "log", "log1p", "log10")
    assertEquals(strict.intersect(inexact ++ Set("cbrt", "pow", "atan2", "hypot")), Set.empty[String])
  }

  test("NaN and infinity checks do not rely on floating point comparisons") {
    val names = Set("isNaN", "isInfinite", "isFinite", "floatToIntBits", "doubleToLongBits")
    val offending = IntrinsicTable.entries.map(_._1).filter(k => names.contains(k.name)).flatMap { key =>
      val fn = IntrinsicTable.function(key).get
      val compares = fn.collectWhere[p.Expr] {
        case p.Expr.IntrOp(i) if i.overloads == p.Intr.BinaryUniformLogic && i.terms.head.tpe.isFractional => i
      }
      compares.map(_ => key.repr)
    }
    assertEquals(offending.toList, Nil)
  }

  test("floorDiv and floorMod leave division by zero undefined, like the / operator") {
    val raising = IntrinsicTable.entries.map(_._1).filter(k => k.name.startsWith("floor")).filter { key =>
      IntrinsicTable.function(key).get.collectWhere[p.Expr] { case p.Expr.SpecOp(p.Spec.Assert(_, _)) => () }.nonEmpty
    }
    assertEquals(raising.toList, Nil)
  }

  test("intrinsic helpers deduplicate only when identical") {
    val key = Key.parse("java/lang/Math.floorMod(II)I")
    val fn  = IntrinsicTable.function(key).get
    assertEquals(IntrinsicTable.dedupe(List(fn, fn)), List(fn))
    val other = fn.copy(body = fn.body.take(1))
    intercept[IllegalStateException](IntrinsicTable.dedupe(List(fn, other)))
    val user = fn.copy(decl = fn.decl.copy(name = p.Sym(List("user", "f"))))
    assertEquals(IntrinsicTable.dedupe(List(user, user)), List(user, user))
  }
}
