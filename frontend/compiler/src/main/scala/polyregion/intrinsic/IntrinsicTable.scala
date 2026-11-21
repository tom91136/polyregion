package polyregion.intrinsic

import polyregion.ast.PolyAST as p
import polyregion.ast.signature

object IntrinsicTable {

  enum Lowering {
    case Op(build: (List[p.Term], p.Type) => p.Expr)
    case Body(build: (Emit, p.Type) => p.Expr)
  }

  enum Resolved {
    case Inline(expr: p.Expr)
    case Call(fn: p.Function, invoke: p.Expr.Invoke)
  }

  private val I = p.Type.IntS32
  private val J = p.Type.IntS64
  private val F = p.Type.Float32
  private val D = p.Type.Float64

  private val MathOwner   = "java/lang/Math"
  private val StrictOwner = "java/lang/StrictMath"

  val CoveredOwners: Set[String] = Set(MathOwner, StrictOwner)

  private def op(key: String)(f: (List[p.Term], p.Type) => p.Expr): (Key, Lowering) =
    Key.parse(key) -> Lowering.Op(f)

  private def body(key: String)(f: (Emit, p.Type) => p.Expr): (Key, Lowering) =
    Key.parse(key) -> Lowering.Body(f)

  private def intr(i: p.Intr): p.Expr = p.Expr.IntrOp(i)
  private def math(m: p.Math): p.Expr = p.Expr.MathOp(m)

  private def tpeLetter(t: p.Type): String = Descriptor.letter(t).get.toString

  private val UnaryFractional: List[(String, (p.Term, p.Type) => p.Math)] = List(
    "sin"   -> (p.Math.Sin(_, _)),
    "cos"   -> (p.Math.Cos(_, _)),
    "tan"   -> (p.Math.Tan(_, _)),
    "asin"  -> (p.Math.Asin(_, _)),
    "acos"  -> (p.Math.Acos(_, _)),
    "atan"  -> (p.Math.Atan(_, _)),
    "sinh"  -> (p.Math.Sinh(_, _)),
    "cosh"  -> (p.Math.Cosh(_, _)),
    "tanh"  -> (p.Math.Tanh(_, _)),
    "sqrt"  -> (p.Math.Sqrt(_, _)),
    "cbrt"  -> (p.Math.Cbrt(_, _)),
    "exp"   -> (p.Math.Exp(_, _)),
    "expm1" -> (p.Math.Expm1(_, _)),
    "log"   -> (p.Math.Log(_, _)),
    "log1p" -> (p.Math.Log1p(_, _)),
    "log10" -> (p.Math.Log10(_, _)),
    "ceil"  -> (p.Math.Ceil(_, _)),
    "floor" -> (p.Math.Floor(_, _)),
    "rint"  -> (p.Math.Rint(_, _))
  )

  private val BinaryFractional: List[(String, (p.Term, p.Term, p.Type) => p.Math)] = List(
    "pow"   -> (p.Math.Pow(_, _, _)),
    "atan2" -> (p.Math.Atan2(_, _, _)),
    "hypot" -> (p.Math.Hypot(_, _, _))
  )

  private val DegreesToRadians = 0.017453292519943295
  private val RadiansToDegrees = 57.29577951308232

  private def sign(b: Emit, x: p.Term, out: p.Type): p.Term.Select = {
    val zero   = b.const(x.tpe, 0)
    val result = b.variable(p.Expr.Alias(b.const(out, 0)))
    b.when(b.gt(x, zero))(b.set(result, p.Expr.Alias(b.const(out, 1))))
    b.when(b.lt(x, zero))(b.set(result, p.Expr.Alias(b.const(out, -1))))
    result
  }

  private def compare(b: Emit, x: p.Term, y: p.Term): p.Term.Select = {
    val result = b.variable(p.Expr.Alias(b.const(I, 0)))
    b.when(b.lt(x, y))(b.set(result, p.Expr.Alias(b.const(I, -1))))
    b.when(b.gt(x, y))(b.set(result, p.Expr.Alias(b.const(I, 1))))
    result
  }

  private def divide(b: Emit, x: p.Term, y: p.Term, remainder: Boolean): p.Term.Select = {
    val t      = x.tpe
    val zero   = b.const(t, 0)
    val result = b.variable(p.Expr.Alias(zero))
    if (remainder) {
      val r      = b.rem(x, y)
      val adjust = b.both(b.neq(r, zero), b.lt(b.xor(r, y), zero))
      b.set(result, p.Expr.Alias(r))
      b.when(adjust)(b.set(result, intr(p.Intr.Add(r, y, t))))
    } else {
      val q      = b.div(x, y)
      val adjust = b.both(b.lt(b.xor(x, y), zero), b.neq(b.mul(q, y), x))
      b.set(result, p.Expr.Alias(q))
      b.when(adjust)(b.set(result, intr(p.Intr.Sub(q, b.const(t, 1), t))))
    }
    result
  }

  private def floorMod(b: Emit, x: p.Term, y: p.Term): p.Term.Select = divide(b, x, y, remainder = true)
  private def floorDiv(b: Emit, x: p.Term, y: p.Term): p.Term.Select = divide(b, x, y, remainder = false)

  private def abs(b: Emit, x: p.Term): p.Term.Select = {
    val result = b.variable(p.Expr.Alias(x))
    b.when(b.lt(x, b.const(x.tpe, 0)))(b.set(result, p.Expr.Alias(b.wrapNeg(x))))
    result
  }

  private def round(b: Emit, x: p.Term, to: p.Type): p.Term.Select = {
    val fp      = x.tpe
    val limit   = if (to == I) 2147483648.0 else 9.223372036854775807e18
    val max     = if (to == I) Int.MaxValue.toLong else Long.MaxValue
    val min     = if (to == I) Int.MinValue.toLong else Long.MinValue
    val floor   = b.math(p.Math.Floor(x, fp))
    val up      = b.gte(b.sub(x, floor), b.real(fp, 0.5))
    val shifted = b.variable(p.Expr.Alias(floor))
    b.when(up)(b.set(shifted, intr(p.Intr.Add(floor, b.real(fp, 1), fp))))
    val result = b.variable(p.Expr.Alias(b.const(to, 0)))
    b.when(b.not(b.isNaN(x))) {
      b.branch(b.gte(shifted, b.real(fp, limit)))(b.set(result, p.Expr.Alias(b.const(to, max)))) {
        b.branch(b.lte(shifted, b.real(fp, -limit)))(b.set(result, p.Expr.Alias(b.const(to, min))))(
          b.set(result, p.Expr.Cast(shifted, to))
        )
      }
    }
    result
  }

  private def copySign(b: Emit, magnitude: p.Term, sign: p.Term): p.Expr = {
    val fp      = magnitude.tpe
    val bits    = if (fp == F) I else J
    val signBit = if (fp == F) Int.MinValue.toLong else Long.MinValue
    val mag     = b.and(b.bitCast(magnitude, bits), b.const(bits, ~signBit))
    val sgn     = b.and(b.bitCast(sign, bits), b.const(bits, signBit))
    p.Expr.BitCast(b.or(mag, sgn), fp)
  }

  private def minMax(b: Emit, x: p.Term, y: p.Term, isMin: Boolean): p.Term.Select = {
    val fp     = x.tpe
    val bits   = if (fp == F) I else J
    val result = b.variable(intr(if (isMin) p.Intr.Min(x, y, fp) else p.Intr.Max(x, y, fp)))
    b.when(b.eq(x, y)) {
      val xb = b.bitCast(x, bits)
      val yb = b.bitCast(y, bits)
      b.set(result, p.Expr.BitCast(if (isMin) b.or(xb, yb) else b.and(xb, yb), fp))
    }
    b.when(b.isNaN(y))(b.set(result, p.Expr.Alias(y)))
    b.when(b.isNaN(x))(b.set(result, p.Expr.Alias(x)))
    result
  }

  private def leadingZeros(b: Emit, x: p.Term): p.Term = {
    val wide  = x.tpe == J
    val u     = if (wide) p.Type.IntU64 else p.Type.IntU32
    val steps = if (wide) List(1, 2, 4, 8, 16, 32) else List(1, 2, 4, 8, 16)
    val smear = steps.foldLeft(b.cast(x, u))((v, s) => b.or(v, b.ushr(v, b.const(u, s))))
    b.sub(b.const(I, if (wide) 64 else 32), b.cast(b.popCount(smear), I))
  }

  private def trailingZeros(b: Emit, x: p.Term): p.Term = {
    val u   = if (x.tpe == J) p.Type.IntU64 else p.Type.IntU32
    val v   = b.cast(x, u)
    val low = b.and(v, b.sub(b.const(u, 0), v))
    b.cast(b.popCount(b.sub(low, b.const(u, 1))), I)
  }

  private def rotate(b: Emit, x: p.Term, distance: p.Term, left: Boolean): p.Term = {
    val wide   = x.tpe == J
    val u      = if (wide) p.Type.IntU64 else p.Type.IntU32
    val mask   = b.const(u, if (wide) 63 else 31)
    val v      = b.cast(x, u)
    val near   = b.and(b.cast(distance, u), mask)
    val far    = b.and(b.sub(b.const(u, if (wide) 64 else 32), near), mask)
    val (l, r) = if (left) (near, far) else (far, near)
    b.cast(b.or(b.shl(v, l), b.ushr(v, r)), x.tpe)
  }

  private def canonicalBits(b: Emit, x: p.Term): p.Term.Select = {
    val fp     = x.tpe
    val bits   = if (fp == F) I else J
    val result = b.variable(p.Expr.BitCast(x, bits))
    val nan    = if (fp == F) 0x7fc00000L else 0x7ff8000000000000L
    b.when(b.isNaN(x))(b.set(result, p.Expr.Alias(b.const(bits, nan))))
    result
  }

  private val ExactUnary = Set("sqrt", "ceil", "floor", "rint")

  private def mathEntries(owner: String, exactOnly: Boolean): List[(Key, Lowering)] = {
    val unary = UnaryFractional.filter((name, _) => !exactOnly || ExactUnary.contains(name)).map { (name, f) =>
      op(s"$owner.$name(D)D")((as, rtn) => math(f(as.head, rtn)))
    }
    val binary = BinaryFractional.filter(_ => !exactOnly).map { (name, f) =>
      op(s"$owner.$name(DD)D")((as, rtn) => math(f(as.head, as(1), rtn)))
    }
    val numeric = List(I, J, F, D).flatMap { t =>
      val l = tpeLetter(t)
      (if (t == I || t == J) List(body(s"$owner.abs($l)$l")((b, _) => p.Expr.Alias(abs(b, b.params.head))))
       else
         List(
           op(s"$owner.abs($l)$l")((as, rtn) => math(p.Math.Abs(as.head, rtn)))
         )) ++ (if (t == I || t == J)
                  List(
                    op(s"$owner.max($l$l)$l")((as, rtn) => intr(p.Intr.Max(as.head, as(1), rtn))),
                    op(s"$owner.min($l$l)$l")((as, rtn) => intr(p.Intr.Min(as.head, as(1), rtn)))
                  )
                else
                  List(
                    body(s"$owner.max($l$l)$l")((b, _) =>
                      p.Expr.Alias(minMax(b, b.params.head, b.params(1), isMin = false))
                    ),
                    body(s"$owner.min($l$l)$l")((b, _) =>
                      p.Expr.Alias(minMax(b, b.params.head, b.params(1), isMin = true))
                    ),
                    op(s"$owner.signum($l)$l")((as, rtn) => math(p.Math.Signum(as.head, rtn))),
                    body(s"$owner.copySign($l$l)$l")((b, _) => copySign(b, b.params.head, b.params(1)))
                  ))
    }
    val rounding = List(
      body(s"$owner.round(F)I")((b, _) => p.Expr.Alias(round(b, b.params.head, I))),
      body(s"$owner.round(D)J")((b, _) => p.Expr.Alias(round(b, b.params.head, J)))
    )
    val angles = List(
      op(s"$owner.toRadians(D)D")((as, rtn) => intr(p.Intr.Mul(as.head, p.Term.Float64Const(DegreesToRadians), rtn))),
      op(s"$owner.toDegrees(D)D")((as, rtn) => intr(p.Intr.Mul(as.head, p.Term.Float64Const(RadiansToDegrees), rtn)))
    )
    val integral = List(
      body(s"$owner.floorMod(II)I")((b, _) => p.Expr.Alias(floorMod(b, b.params.head, b.params(1)))),
      body(s"$owner.floorMod(JJ)J")((b, _) => p.Expr.Alias(floorMod(b, b.params.head, b.params(1)))),
      body(s"$owner.floorMod(JI)I")((b, _) =>
        p.Expr.Alias(b.cast(floorMod(b, b.params.head, b.cast(b.params(1), J)), I))
      ),
      body(s"$owner.floorDiv(II)I")((b, _) => p.Expr.Alias(floorDiv(b, b.params.head, b.params(1)))),
      body(s"$owner.floorDiv(JJ)J")((b, _) => p.Expr.Alias(floorDiv(b, b.params.head, b.params(1)))),
      body(s"$owner.floorDiv(JI)J")((b, _) => p.Expr.Alias(floorDiv(b, b.params.head, b.cast(b.params(1), J))))
    )
    unary ++ binary ++ numeric ++ rounding ++ angles ++ integral
  }

  private def integralEntries(owner: String, t: p.Type): List[(Key, Lowering)] = {
    val l = tpeLetter(t)
    List(
      body(s"$owner.bitCount($l)I")((b, _) => p.Expr.Alias(b.cast(b.popCount(b.params.head), I))),
      body(s"$owner.signum($l)I")((b, _) => p.Expr.Alias(sign(b, b.params.head, I))),
      body(s"$owner.numberOfLeadingZeros($l)I")((b, _) => p.Expr.Alias(leadingZeros(b, b.params.head))),
      body(s"$owner.numberOfTrailingZeros($l)I")((b, _) => p.Expr.Alias(trailingZeros(b, b.params.head))),
      body(s"$owner.lowestOneBit($l)$l")((b, _) => p.Expr.Alias(b.and(b.params.head, b.wrapNeg(b.params.head)))),
      body(s"$owner.rotateLeft(${l}I)$l")((b, _) => p.Expr.Alias(rotate(b, b.params.head, b.params(1), left = true))),
      body(s"$owner.rotateRight(${l}I)$l")((b, _) => p.Expr.Alias(rotate(b, b.params.head, b.params(1), left = false))),
      op(s"$owner.max($l$l)$l")((as, rtn) => intr(p.Intr.Max(as.head, as(1), rtn))),
      op(s"$owner.min($l$l)$l")((as, rtn) => intr(p.Intr.Min(as.head, as(1), rtn))),
      op(s"$owner.sum($l$l)$l")((as, rtn) => intr(p.Intr.Add(as.head, as(1), rtn))),
      body(s"$owner.compare($l$l)I")((b, _) => p.Expr.Alias(compare(b, b.params.head, b.params(1))))
    )
  }

  private def floatingEntries(owner: String, t: p.Type): List[(Key, Lowering)] = {
    val l         = tpeLetter(t)
    val bits      = if (t == F) I else J
    val b         = tpeLetter(bits)
    val raw       = if (t == F) "floatToRawIntBits" else "doubleToRawLongBits"
    val canonical = if (t == F) "floatToIntBits" else "doubleToLongBits"
    val from      = if (t == F) "intBitsToFloat" else "longBitsToDouble"
    List(
      op(s"$owner.$raw($l)$b")((as, _) => p.Expr.BitCast(as.head, bits)),
      body(s"$owner.$canonical($l)$b")((bd, _) => p.Expr.Alias(canonicalBits(bd, bd.params.head))),
      op(s"$owner.$from($b)$l")((as, _) => p.Expr.BitCast(as.head, t)),
      body(s"$owner.isNaN($l)Z")((bd, _) => p.Expr.Alias(bd.isNaN(bd.params.head))),
      body(s"$owner.isInfinite($l)Z")((bd, _) => p.Expr.Alias(bd.isInfinite(bd.params.head))),
      body(s"$owner.isFinite($l)Z")((bd, _) => p.Expr.Alias(bd.isFinite(bd.params.head))),
      body(s"$owner.max($l$l)$l")((bd, _) => p.Expr.Alias(minMax(bd, bd.params.head, bd.params(1), isMin = false))),
      body(s"$owner.min($l$l)$l")((bd, _) => p.Expr.Alias(minMax(bd, bd.params.head, bd.params(1), isMin = true))),
      op(s"$owner.sum($l$l)$l")((as, rtn) => intr(p.Intr.Add(as.head, as(1), rtn)))
    )
  }

  val entries: Vector[(Key, Lowering)] = (
    mathEntries(MathOwner, exactOnly = false) ++
      mathEntries(StrictOwner, exactOnly = true) ++
      integralEntries("java/lang/Integer", I) ++
      integralEntries("java/lang/Long", J) ++
      floatingEntries("java/lang/Float", F) ++
      floatingEntries("java/lang/Double", D)
  ).toVector

  private val table: Map[Key, Lowering] = entries.toMap

  def lookup(key: Key): Option[Lowering] = table.get(key)

  private val helperNames: Set[p.Sym] = entries.map(_._1.functionName).toSet

  def dedupe(fns: List[p.Function]): List[p.Function] = {
    val helpers = fns.filter(fn => helperNames.contains(fn.name))
    helpers.groupBy(_.signature).foreach { (signature, group) =>
      if (group.distinct.size != 1)
        throw IllegalStateException(s"Intrinsic helpers share the signature ${signature.repr} but differ in body")
    }
    val seen = scala.collection.mutable.Set.empty[p.Signature]
    fns.filter(fn => !helperNames.contains(fn.name) || seen.add(fn.signature))
  }

  def function(key: Key): Option[p.Function] = for {
    lowering      <- table.get(key)
    (params, rtn) <- Descriptor.parse(key.descriptor)
  } yield {
    val named   = params.zipWithIndex.map((t, i) => p.Named(s"a$i", t))
    val builder = Emit(named.map(builderSelect))
    val result = lowering match {
      case Lowering.Op(f)   => f(builder.params, rtn)
      case Lowering.Body(f) => f(builder, rtn)
    }
    p.Function(
      decl = p.FunctionDecl(
        name = key.functionName,
        tpeVars = Nil,
        receiver = None,
        args = named.map(p.Arg(_)),
        moduleCaptures = Nil,
        termCaptures = Nil,
        rtn = rtn,
        affinity = p.Function.Affinity.Offload
      ),
      body = builder.statements :+ p.Stmt.Return(result),
      visibility = p.Function.Visibility.Exported,
      fpMode = p.Function.FpMode.Relaxed,
      convention = p.CallConvention.RegularCall
    )
  }

  private def builderSelect(n: p.Named): p.Term.Select = p.Term.Select(n, Nil, n.tpe)

  def resolve(key: Key, args: List[p.Term]): Option[Resolved] = for {
    lowering      <- table.get(key)
    (params, rtn) <- Descriptor.parse(key.descriptor)
    if args.map(_.tpe) == params
    resolved <- lowering match {
      case Lowering.Op(f) => Some(Resolved.Inline(f(args, rtn)))
      case Lowering.Body(_) =>
        function(key).map(fn => Resolved.Call(fn, p.Expr.Invoke(p.Type.FnRef(fn.name), Nil, None, args, rtn)))
    }
  } yield resolved
}
