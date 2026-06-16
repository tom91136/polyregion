package polyregion.ast

import polyregion.ast.PolyAST as p

import scala.collection.mutable

object Interpreter {

  enum V { case I(v: Long); case D(v: Double); case U }

  private final class Ret(val v: V) extends scala.util.control.ControlThrowable
  private object Brk                extends scala.util.control.ControlThrowable
  private object Cnt                extends scala.util.control.ControlThrowable

  type Foreign = (name: String, args: List[(p.Type, V)], rtn: p.Type) => V

  object Vm {
    val noForeign: Foreign = (n, _, _) => sys.error(s"ForeignCall $n not supported (phase 1)")
  }

  final class Vm(val program: p.Program, val globalId: Long = 0, val globalSize: Long = 1) {

    var foreign: Foreign = Vm.noForeign

    private val defs = program.defs.map(d => d.name -> d).toMap
    private val fns  = (program.entry :: program.functions).map(f => f.name -> f).toMap

    // device allocations live in a disjoint range so the kernel cannot reach host memory by accident;
    // an unmirrored or unpatched pointer surviving into the kernel trips the deviceMode guard below
    private val DeviceBase = 1L << 40
    private var hbuf       = new Array[Byte](1 << 16)
    private var dbuf       = new Array[Byte](1 << 16)
    private var htop       = 8L
    private var dtop       = DeviceBase + 8L
    private val allocs     = mutable.TreeMap.empty[Long, Long]

    var deviceMode = false

    def alloc(n: Long): Long = if (deviceMode) allocDevice(n)
    else {
      val a = htop
      htop += (math.max(n, 0) + 7) / 8 * 8
      while (htop > hbuf.length) hbuf = java.util.Arrays.copyOf(hbuf, hbuf.length * 2)
      allocs(a) = math.max(n, 0)
      a
    }
    def allocDevice(n: Long): Long = {
      val a = dtop
      dtop += (math.max(n, 0) + 7) / 8 * 8
      while (dtop - DeviceBase > dbuf.length) dbuf = java.util.Arrays.copyOf(dbuf, dbuf.length * 2)
      allocs(a) = math.max(n, 0)
      a
    }
    def allocOf(t: p.Type, count: Long = 1): Long = alloc(sizeOf(t) * count)

    def findAlloc(addr: Long): (Long, Long) = allocs.maxBefore(addr + 1) match {
      case Some((b, sz)) if addr < b + sz => (b, sz)
      case _                              => sys.error(s"no allocation for $addr")
    }
    private def bufFor(addr: Long): (Array[Byte], Int) =
      if (addr >= DeviceBase) (dbuf, (addr - DeviceBase).toInt) else (hbuf, addr.toInt)
    def copy(dst: Long, src: Long, n: Long): Unit = {
      val (sa, so)  = bufFor(src)
      val (da, dof) = bufFor(dst)
      System.arraycopy(sa, so, da, dof, n.toInt)
    }
    def loadPtr(addr: Long): Long           = loadBits(addr, 8)
    def storePtr(addr: Long, v: Long): Unit = storeBits(addr, 8, v)

    def sizeOf(t: p.Type): Long = t match {
      case p.Type.Bool1 | p.Type.IntU8 | p.Type.IntS8     => 1
      case p.Type.IntU16 | p.Type.IntS16 | p.Type.Float16 => 2
      case p.Type.IntU32 | p.Type.IntS32 | p.Type.Float32 => 4
      case p.Type.IntU64 | p.Type.IntS64 | p.Type.Float64 => 8
      case p.Type.Unit0                                   => 0
      case p.Type.Ptr(_, _)                               => 8
      case p.Type.Arr(c, n, _)                            => stride(c) * n
      case p.Type.Struct(s, _)                            => layout(s)._2
      case _                                              => sys.error(s"no size for $t")
    }

    def alignOf(t: p.Type): Long = t match {
      case p.Type.Ptr(_, _)    => 8
      case p.Type.Arr(c, _, _) => alignOf(c)
      case p.Type.Struct(s, _) => defs(s).members.foldLeft(1L)((m, x) => math.max(m, alignOf(x.tpe)))
      case _                   => math.max(1, sizeOf(t))
    }

    private def stride(c: p.Type): Long         = alignUp(sizeOf(c), alignOf(c))
    private def alignUp(x: Long, a: Long): Long = if (a <= 1) x else (x + a - 1) / a * a

    private val layoutMemo = mutable.Map.empty[p.Sym, (Map[String, Long], Long)]
    private def layout(s: p.Sym): (Map[String, Long], Long) = layoutMemo.getOrElseUpdate(
      s, {
        var off  = 0L
        var maxA = 1L
        val offs = defs(s).members.map { m =>
          val a = alignOf(m.tpe)
          off = alignUp(off, a)
          val o = off
          off += sizeOf(m.tpe)
          maxA = math.max(maxA, a)
          m.symbol -> o
        }.toMap
        (offs, alignUp(off, maxA))
      }
    )
    def offsetOf(s: p.Sym, field: String): Long            = layout(s)._1(field)
    private def memberTpe(s: p.Sym, field: String): p.Type = defs(s).members.find(_.symbol == field).get.tpe
    private def structSym(t: p.Type): p.Sym = t match {
      case p.Type.Ptr(p.Type.Struct(s, _), _) => s
      case p.Type.Struct(s, _)                => s
      case _                                  => sys.error(s"not a struct $t")
    }

    private def storeBits(addr: Long, n: Int, bits: Long): Unit = {
      if (deviceMode && addr < DeviceBase) sys.error(s"device kernel wrote host address $addr (unmirrored pointer)")
      val (buf, off) = bufFor(addr)
      var i          = 0
      var x          = bits
      while (i < n) { buf(off + i) = (x & 0xff).toByte; x >>>= 8; i += 1 }
    }
    private def loadBits(addr: Long, n: Int): Long = {
      if (deviceMode && addr < DeviceBase) sys.error(s"device kernel read host address $addr (unmirrored pointer)")
      val (buf, off) = bufFor(addr)
      var i          = n - 1
      var x          = 0L
      while (i >= 0) { x = (x << 8) | (buf(off + i) & 0xffL); i -= 1 }
      x
    }

    def store(addr: Long, v: V, t: p.Type): Unit = storeBits(addr, sizeOf(t).toInt, encode(v, t))
    def load(addr: Long, t: p.Type): V = t match {
      case p.Type.Struct(_, _) | p.Type.Arr(_, _, _) => sys.error(s"cannot load aggregate $t as a value")
      case _                                         => decode(loadBits(addr, sizeOf(t).toInt), t)
    }
    def decodeBits(bits: Long, t: p.Type): V = decode(bits, t)

    private def isFloat(t: p.Type) = t match {
      case p.Type.Float16 | p.Type.Float32 | p.Type.Float64 => true; case _ => false
    }
    private def isSigned(t: p.Type) = t match {
      case p.Type.IntS8 | p.Type.IntS16 | p.Type.IntS32 | p.Type.IntS64 => true; case _ => false
    }
    private def asI(v: V): Long   = v match { case V.I(x) => x; case V.D(d) => d.toLong; case V.U => 0 }
    private def asD(v: V): Double = v match { case V.D(d) => d; case V.I(x) => x.toDouble; case V.U => 0 }

    private def encode(v: V, t: p.Type): Long = t match {
      case p.Type.Float32 => java.lang.Float.floatToIntBits(asD(v).toFloat) & 0xffffffffL
      case p.Type.Float64 => java.lang.Double.doubleToLongBits(asD(v))
      case _              => asI(v)
    }
    private def decode(bits: Long, t: p.Type): V = t match {
      case p.Type.Float32 => V.D(java.lang.Float.intBitsToFloat((bits & 0xffffffffL).toInt).toDouble)
      case p.Type.Float64 => V.D(java.lang.Double.longBitsToDouble(bits))
      case p.Type.Unit0   => V.U
      case _              => V.I(narrow(bits, t))
    }
    private def narrow(x: Long, t: p.Type): Long = {
      val n = sizeOf(t).toInt
      if (n >= 8) x
      else if (isSigned(t)) x << (64 - n * 8) >> (64 - n * 8)
      else x & ((1L << (n * 8)) - 1)
    }

    private final class Frame { val slots = mutable.Map.empty[String, Long] }

    def call(name: p.Sym, args: List[(p.Type, V)]): V = {
      val f      = fns.getOrElse(name, sys.error(s"no function $name"))
      val fr     = new Frame
      val params = f.receiver.toList ::: f.args ::: f.moduleCaptures ::: f.termCaptures
      params.zip(args).foreach { case (a, (_, v)) =>
        val slot = allocOf(a.named.tpe)
        fr.slots(a.named.symbol) = slot
        store(slot, v, a.named.tpe)
      }
      try { f.body.foreach(exec(_, fr)); V.U }
      catch { case r: Ret => r.v }
    }
    def call(name: String, args: List[(p.Type, V)]): V = call(p.Sym(List(name)), args)

    private def exec(s: p.Stmt, fr: Frame): Unit = s match {
      case p.Stmt.Var(n, e, _) =>
        val slot = allocOf(n.tpe)
        fr.slots(n.symbol) = slot
        e.foreach(ex => store(slot, eval(ex, fr), n.tpe))
      case p.Stmt.Mut(sel, e) =>
        val (addr, t) = resolve(sel.root, sel.steps, fr)
        store(addr, eval(e, fr), t)
      case p.Stmt.Update(sel, idx, value) =>
        val (addr, t) = resolve(sel.root, sel.steps, fr)
        val (b, c) = t match {
          case p.Type.Ptr(c, _)    => (loadBits(addr, 8), c)
          case p.Type.Arr(c, _, _) => (addr, c)
          case x                   => (addr, x)
        }
        store(b + asI(evalT(idx, fr)) * sizeOf(c), evalT(value, fr), c)
      case p.Stmt.While(cond, body) =>
        try
          while (asI(evalT(cond, fr)) != 0)
            try body.foreach(exec(_, fr))
            catch { case Cnt => () }
        catch { case Brk => () }
      case p.Stmt.ForRange(ind, lb, ub, step, body) =>
        val slot = allocOf(ind.tpe)
        fr.slots(ind.symbol) = slot
        var i  = asI(evalT(lb, fr))
        val u  = asI(evalT(ub, fr))
        val st = asI(evalT(step, fr))
        try
          while (i < u) {
            store(slot, V.I(i), ind.tpe);
            try body.foreach(exec(_, fr))
            catch { case Cnt => () }; i += st
          }
        catch { case Brk => () }
      case p.Stmt.Cond(c, t, e) => if (asI(evalT(c, fr)) != 0) t.foreach(exec(_, fr)) else e.foreach(exec(_, fr))
      case p.Stmt.Return(e)     => throw new Ret(eval(e, fr))
      case p.Stmt.Break         => throw Brk
      case p.Stmt.Cont          => throw Cnt
      case p.Stmt.Annotated(inner, _, _) => exec(inner, fr)
    }

    private def eval(e: p.Expr, fr: Frame): V = e match {
      case p.Expr.Alias(t)              => evalT(t, fr)
      case p.Expr.IntrOp(op)            => intr(op, fr)
      case p.Expr.MathOp(op)            => mathOp(op, fr)
      case p.Expr.SpecOp(op)            => spec(op)
      case p.Expr.Cast(from, as)        => cast(evalT(from, fr), from.tpe, as)
      case p.Expr.Index(lhs, idx, comp) => load(base(lhs, fr) + asI(evalT(idx, fr)) * sizeOf(comp), comp)
      case p.Expr.RefTo(lhs, idx, comp, _, _) =>
        val b = idx match { case Some(_) => base(lhs, fr); case None => addressOf(lhs, fr) }
        V.I(idx.fold(b)(k => b + asI(evalT(k, fr)) * sizeOf(comp)))
      case p.Expr.Alloc(comp, size, _, _) => V.I(alloc(sizeOf(comp) * asI(evalT(size, fr))))
      case p.Expr.Invoke(name, _, recv, args, _) =>
        call(name, recv.toList.map(t => t.tpe -> evalT(t, fr)) ::: args.map(t => t.tpe -> evalT(t, fr)))
      case p.Expr.ForeignCall(name, args, rtn) => foreign(name, args.map(t => t.tpe -> evalT(t, fr)), rtn)
      case p.Expr.OffsetOf(st, field)          => V.I(offsetOf(structSym(st), field))
    }

    private def evalT(t: p.Term, fr: Frame): V = t match {
      case p.Term.Float16Const(v)       => V.D(v.toDouble)
      case p.Term.Float32Const(v)       => V.D(v.toDouble)
      case p.Term.Float64Const(v)       => V.D(v)
      case p.Term.IntU8Const(v)         => V.I(v & 0xffL)
      case p.Term.IntU16Const(v)        => V.I(v.toLong)
      case p.Term.IntU32Const(v)        => V.I(v.toLong & 0xffffffffL)
      case p.Term.IntU64Const(v)        => V.I(v)
      case p.Term.IntS8Const(v)         => V.I(v.toLong)
      case p.Term.IntS16Const(v)        => V.I(v.toLong)
      case p.Term.IntS32Const(v)        => V.I(v.toLong)
      case p.Term.IntS64Const(v)        => V.I(v)
      case p.Term.Unit0Const            => V.U
      case p.Term.Bool1Const(v)         => V.I(if (v) 1 else 0)
      case p.Term.NullPtrConst(_, _, _) => V.I(0)
      case p.Term.Poison(_)             => sys.error("poison evaluated")
      case s: p.Term.Select             => val (a, lt) = resolve(s.root, s.steps, fr); load(a, lt)
    }

    private def base(lhs: p.Term, fr: Frame): Long = lhs.tpe match {
      case p.Type.Ptr(_, _) => asI(evalT(lhs, fr))
      case _                => addressOf(lhs, fr)
    }
    private def addressOf(t: p.Term, fr: Frame): Long = t match {
      case s: p.Term.Select => resolve(s.root, s.steps, fr)._1
      case _                => sys.error("no address for term")
    }

    private def resolve(root: p.Named, steps: List[p.PathStep], fr: Frame): (Long, p.Type) =
      steps.foldLeft((fr.slots(root.symbol), root.tpe)) { case ((addr, t), s) => step(addr, t, s, fr) }
    private def step(addr: Long, t: p.Type, s: p.PathStep, fr: Frame): (Long, p.Type) = s match {
      case p.PathStep.Field(name) =>
        t match {
          case p.Type.Ptr(p.Type.Struct(sn, _), _) => (loadBits(addr, 8) + offsetOf(sn, name), memberTpe(sn, name))
          case p.Type.Struct(sn, _)                => (addr + offsetOf(sn, name), memberTpe(sn, name))
          case _                                   => sys.error(s"field on $t")
        }
      case p.PathStep.Deref        => indexStep(addr, t, 0)
      case p.PathStep.Index(k)     => indexStep(addr, t, k.toLong)
      case p.PathStep.IndexDyn(it) => indexStep(addr, t, asI(evalT(it, fr)))
    }
    private def indexStep(addr: Long, t: p.Type, i: Long): (Long, p.Type) = t match {
      case p.Type.Ptr(c, _)    => (loadBits(addr, 8) + i * sizeOf(c), c)
      case p.Type.Arr(c, _, _) => (addr + i * sizeOf(c), c)
      case _                   => sys.error(s"index on $t")
    }

    private def binN(a: p.Term, b: p.Term, r: p.Type, fr: Frame)(
        fi: (Long, Long) => Long
    )(fd: (Double, Double) => Double): V =
      if (isFloat(r)) V.D(fd(asD(evalT(a, fr)), asD(evalT(b, fr))))
      else V.I(narrow(fi(asI(evalT(a, fr)), asI(evalT(b, fr))), r))
    private def cmp(a: p.Term, b: p.Term, fr: Frame)(
        fi: (Long, Long) => Boolean
    )(fd: (Double, Double) => Boolean): V = {
      val r = if (isFloat(a.tpe)) fd(asD(evalT(a, fr)), asD(evalT(b, fr))) else fi(asI(evalT(a, fr)), asI(evalT(b, fr)))
      V.I(if (r) 1 else 0)
    }

    private def intr(op: p.Intr, fr: Frame): V = op match {
      case p.Intr.Add(a, b, r) => binN(a, b, r, fr)(_ + _)(_ + _)
      case p.Intr.Sub(a, b, r) => binN(a, b, r, fr)(_ - _)(_ - _)
      case p.Intr.Mul(a, b, r) => binN(a, b, r, fr)(_ * _)(_ * _)
      case p.Intr.Div(a, b, r) =>
        binN(a, b, r, fr)((x, y) => if (isSigned(r)) x / y else java.lang.Long.divideUnsigned(x, y))(_ / _)
      case p.Intr.Rem(a, b, r) =>
        binN(a, b, r, fr)((x, y) => if (isSigned(r)) x % y else java.lang.Long.remainderUnsigned(x, y))(_ % _)
      case p.Intr.Min(a, b, r)  => binN(a, b, r, fr)((x, y) => math.min(x, y))((x, y) => math.min(x, y))
      case p.Intr.Max(a, b, r)  => binN(a, b, r, fr)((x, y) => math.max(x, y))((x, y) => math.max(x, y))
      case p.Intr.BAnd(a, b, r) => V.I(narrow(asI(evalT(a, fr)) & asI(evalT(b, fr)), r))
      case p.Intr.BOr(a, b, r)  => V.I(narrow(asI(evalT(a, fr)) | asI(evalT(b, fr)), r))
      case p.Intr.BXor(a, b, r) => V.I(narrow(asI(evalT(a, fr)) ^ asI(evalT(b, fr)), r))
      case p.Intr.BSL(a, b, r)  => V.I(narrow(asI(evalT(a, fr)) << asI(evalT(b, fr)), r))
      case p.Intr.BSR(a, b, r) =>
        V.I(
          narrow(
            if (isSigned(r)) asI(evalT(a, fr)) >> asI(evalT(b, fr)) else asI(evalT(a, fr)) >>> asI(evalT(b, fr)),
            r
          )
        )
      case p.Intr.BZSR(a, b, r)  => V.I(narrow(asI(evalT(a, fr)) >>> asI(evalT(b, fr)), r))
      case p.Intr.BNot(a, r)     => V.I(narrow(~asI(evalT(a, fr)), r))
      case p.Intr.Pos(a, _)      => evalT(a, fr)
      case p.Intr.Neg(a, r)      => if (isFloat(r)) V.D(-asD(evalT(a, fr))) else V.I(narrow(-asI(evalT(a, fr)), r))
      case p.Intr.LogicNot(a)    => V.I(if (asI(evalT(a, fr)) == 0) 1 else 0)
      case p.Intr.LogicAnd(a, b) => V.I(if (asI(evalT(a, fr)) != 0 && asI(evalT(b, fr)) != 0) 1 else 0)
      case p.Intr.LogicOr(a, b)  => V.I(if (asI(evalT(a, fr)) != 0 || asI(evalT(b, fr)) != 0) 1 else 0)
      case p.Intr.LogicEq(a, b)  => cmp(a, b, fr)(_ == _)(_ == _)
      case p.Intr.LogicNeq(a, b) => cmp(a, b, fr)(_ != _)(_ != _)
      case p.Intr.LogicLt(a, b)  => cmp(a, b, fr)(_ < _)(_ < _)
      case p.Intr.LogicLte(a, b) => cmp(a, b, fr)(_ <= _)(_ <= _)
      case p.Intr.LogicGt(a, b)  => cmp(a, b, fr)(_ > _)(_ > _)
      case p.Intr.LogicGte(a, b) => cmp(a, b, fr)(_ >= _)(_ >= _)
    }

    private def mathOp(op: p.Math, fr: Frame): V = {
      def u(x: p.Term)(f: Double => Double) = V.D(f(asD(evalT(x, fr))))
      op match {
        case p.Math.Abs(x, r) =>
          if (isFloat(r)) V.D(math.abs(asD(evalT(x, fr)))) else V.I(narrow(math.abs(asI(evalT(x, fr))), r))
        case p.Math.Sin(x, _)      => u(x)(math.sin)
        case p.Math.Cos(x, _)      => u(x)(math.cos)
        case p.Math.Tan(x, _)      => u(x)(math.tan)
        case p.Math.Asin(x, _)     => u(x)(math.asin)
        case p.Math.Acos(x, _)     => u(x)(math.acos)
        case p.Math.Atan(x, _)     => u(x)(math.atan)
        case p.Math.Sinh(x, _)     => u(x)(math.sinh)
        case p.Math.Cosh(x, _)     => u(x)(math.cosh)
        case p.Math.Tanh(x, _)     => u(x)(math.tanh)
        case p.Math.Signum(x, _)   => u(x)(math.signum)
        case p.Math.Round(x, _)    => u(x)(d => math.round(d).toDouble)
        case p.Math.Ceil(x, _)     => u(x)(math.ceil)
        case p.Math.Floor(x, _)    => u(x)(math.floor)
        case p.Math.Rint(x, _)     => u(x)(math.rint)
        case p.Math.Sqrt(x, _)     => u(x)(math.sqrt)
        case p.Math.Cbrt(x, _)     => u(x)(math.cbrt)
        case p.Math.Exp(x, _)      => u(x)(math.exp)
        case p.Math.Expm1(x, _)    => u(x)(math.expm1)
        case p.Math.Log(x, _)      => u(x)(math.log)
        case p.Math.Log1p(x, _)    => u(x)(math.log1p)
        case p.Math.Log10(x, _)    => u(x)(math.log10)
        case p.Math.Pow(x, y, _)   => V.D(math.pow(asD(evalT(x, fr)), asD(evalT(y, fr))))
        case p.Math.Atan2(x, y, _) => V.D(math.atan2(asD(evalT(x, fr)), asD(evalT(y, fr))))
        case p.Math.Hypot(x, y, _) => V.D(math.hypot(asD(evalT(x, fr)), asD(evalT(y, fr))))
      }
    }

    private def spec(op: p.Spec): V = op match {
      case _: p.Spec.GpuGlobalIdx  => V.I(globalId)
      case _: p.Spec.GpuGlobalSize => V.I(globalSize)
      case _: p.Spec.GpuGroupIdx   => V.I(0)
      case _: p.Spec.GpuGroupSize  => V.I(globalSize)
      case _: p.Spec.GpuLocalIdx   => V.I(0)
      case _: p.Spec.GpuLocalSize  => V.I(1)
      case _                       => V.U
    }

    private def cast(v: V, from: p.Type, to: p.Type): V =
      if (isFloat(to) && !isFloat(from)) V.D(asI(v).toDouble)
      else if (!isFloat(to) && isFloat(from)) V.I(narrow(asD(v).toLong, to))
      else if (isFloat(to)) V.D(asD(v))
      else V.I(narrow(asI(v), to))
  }

  final class Sma(vm: Vm) {
    private val table   = mutable.Map.empty[Long, (Long, Long)]
    private val visited = mutable.Set.empty[Long]

    private val argBuf      = mutable.ArrayBuffer.empty[(Long, Long)]
    private val poolOffsets = mutable.ArrayBuffer.empty[Long]
    private var poolNodes   = Vector.empty[Long]
    private var poolBase    = 0L
    private var poolNodeSz  = 0L

    private def ensureBase(local: Long, minSize: Long): Long = {
      val (b, sz) = vm.findAlloc(local)
      val want    = math.max(sz, minSize)
      table.get(b) match {
        case Some((d, ds)) if ds >= want => d
        case _                           => val d = vm.allocDevice(want); vm.copy(d, b, sz); table(b) = (d, want); d
      }
    }
    private def ensure(local: Long, minSize: Long): Long = {
      val (b, _) = vm.findAlloc(local); ensureBase(local, minSize) + (local - b)
    }

    // mirror `depth` levels of pointer indirection, patching each device copy's pointer slots to point
    // at the mirrored child buffers (a single-heap model would leave host pointers and never fault)
    private def ensureDeep(local: Long, depth: Long): Long = {
      val (b, sz) = vm.findAlloc(local)
      val dev     = ensure(local, 0)
      if (depth > 0) {
        var i = 0L
        while (local + (i + 1) * 8 <= b + sz) {
          val host = vm.loadPtr(local + i * 8)
          if (host != 0) vm.storePtr(dev + i * 8, ensureDeep(host, depth - 1))
          i += 1
        }
      }
      dev
    }
    private def readAlloc(local: Long): Unit = {
      val (b, sz) = vm.findAlloc(local)
      if (!visited(b)) { visited += b; table.get(b).foreach { case (d, _) => vm.copy(b, d, sz) } }
    }
    // mirror of ensureDeep: read leaf data back, recursing through host pointers (never copy the device
    // pointer slots back - they hold device addresses)
    private def readDeep(local: Long, depth: Long): Unit =
      if (depth == 0) readAlloc(local)
      else {
        val (b, sz) = vm.findAlloc(local)
        var i       = 0L
        while (local + (i + 1) * 8 <= b + sz) {
          val host = vm.loadPtr(local + i * 8)
          if (host != 0) readDeep(host, depth - 1)
          i += 1
        }
      }

    private def poolGraph(root: Long, nodeSz: Long): Long = {
      val nodes = mutable.ArrayBuffer.empty[Long]
      val idx   = mutable.LinkedHashMap.empty[Long, Long]
      def intern(n: Long): Long =
        if (n == 0) -1L else idx.getOrElseUpdate(n, { val i = nodes.size.toLong; nodes += n; i })
      intern(root)
      var i = 0
      while (i < nodes.size) { poolOffsets.foreach(off => intern(vm.loadPtr(nodes(i) + off))); i += 1 }
      val pool = vm.allocDevice(nodeSz * nodes.size)
      nodes.zipWithIndex.foreach { case (n, k) =>
        vm.copy(pool + k * nodeSz, n, nodeSz)
        poolOffsets.foreach(off => vm.storePtr(pool + k * nodeSz + off, intern(vm.loadPtr(n + off))))
      }
      poolNodes = nodes.toVector; poolBase = pool; poolNodeSz = nodeSz
      pool
    }
    private def readPool(): Unit = poolNodes.zipWithIndex.foreach { case (host, k) =>
      val saved = poolOffsets.map(off => vm.loadPtr(host + off))
      vm.copy(host, poolBase + k * poolNodeSz, poolNodeSz)
      poolOffsets.zip(saved).foreach { case (off, ptr) => vm.storePtr(host + off, ptr) }
    }

    private var graphNodes = Vector.empty[Long]
    private def discover(root: Long): Vector[Long] = {
      val nodes = mutable.ArrayBuffer.empty[Long]
      val seen  = mutable.HashSet.empty[Long]
      def go(n: Long): Unit =
        if (n != 0 && seen.add(n)) { nodes += n; poolOffsets.foreach(off => go(vm.loadPtr(n + off))) }
      go(root); nodes.toVector
    }
    // Mirror's graph path keeps the pointer shape (unlike pool_graph's index form): deep-copy each node
    // and patch its self-pointers to the device copies
    private def mirrorGraph(root: Long): Long = if (root == 0) 0L
    else {
      val nodes = discover(root)
      val dmap  = nodes.map(n => n -> ensureBase(n, 0)).toMap
      nodes.foreach { n =>
        val d = dmap(n)
        poolOffsets.foreach { off =>
          val child = vm.loadPtr(n + off)
          vm.storePtr(d + off, if (child == 0) 0L else dmap(child))
        }
      }
      graphNodes = nodes
      dmap(root)
    }
    private def readGraph(): Unit = graphNodes.foreach { host =>
      val (b, sz) = vm.findAlloc(host)
      if (!visited(b)) {
        visited += b
        val saved = poolOffsets.map(off => vm.loadPtr(host + off))
        table.get(b).foreach { case (d, _) => vm.copy(b, d, sz) }
        poolOffsets.zip(saved).foreach { case (off, ptr) => vm.storePtr(host + off, ptr) }
      }
    }

    def boundArgs(entry: p.Function): List[(p.Type, V)] = {
      val params = entry.receiver.toList ::: entry.args ::: entry.moduleCaptures ::: entry.termCaptures
      params.zip(argBuf.toList).map { case (arg, (_, bits)) => arg.named.tpe -> vm.decodeBits(bits, arg.named.tpe) }
    }

    val handler: Foreign = (name, args, _) => {
      def a(i: Int): Long = args(i)._2 match { case V.I(x) => x; case V.D(d) => d.toLong; case V.U => 0L }
      name match {
        case "polyrt_sma_alloc" =>
          val (local, size) = (a(0), a(1))
          V.I(table.get(local) match {
            case Some((d, _)) => d
            case _            => val d = vm.allocDevice(size); vm.copy(d, local, size); table(local) = (d, size); d
          })
        case "polyrt_sma_ensure"          => V.I(ensure(a(0), 0))
        case "polyrt_sma_ensure_min"      => V.I(ensure(a(0), a(1)))
        case "polyrt_sma_ensure_base_min" => V.I(ensureBase(a(0), a(1)))
        case "polyrt_sma_ensure_deep"     => V.I(ensureDeep(a(0), a(1)))
        case "polyrt_sma_offset_bytes"    => val (b, _) = vm.findAlloc(a(0)); V.I(a(0) - b)
        case "polyrt_sma_pointee_size"    => val (b, sz) = vm.findAlloc(a(0)); V.I(b + sz - a(0))
        case "polyrt_sma_patch"           => vm.storePtr(a(0) + a(1), a(2)); V.U
        case "polyrt_sma_read_alloc"      => readAlloc(a(0)); V.U
        case "polyrt_sma_read_deep"       => readDeep(a(0), a(1)); V.U
        case "polyrt_sma_visit_clear"     => visited.clear(); V.U
        case "polyrt_sma_release"         => V.U
        case "polyrt_sma_mirror_graph"    => V.I(mirrorGraph(a(0)))
        case "polyrt_sma_read_graph"      => readGraph(); V.U
        case "polyrt_args_reset"          => argBuf.clear(); V.U
        case "polyrt_args_put"            => argBuf += ((a(0), a(1))); V.U
        case "polyrt_sma_pool_reset"      => poolOffsets.clear(); V.U
        case "polyrt_sma_pool_ptr"        => poolOffsets += a(0); V.U
        case "polyrt_sma_pool_graph"      => V.I(poolGraph(a(0), a(1)))
        case "polyrt_sma_pool_root_index" => V.I(0)
        case "polyrt_sma_read_pool"       => readPool(); V.U
        case other                        => sys.error(s"unsupported foreign $other")
      }
    }
  }
}
