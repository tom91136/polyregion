package polyregion.javalang

import cats.syntax.all.*
import org.jetbrains.java.decompiler.code.CodeConstants
import org.jetbrains.java.decompiler.modules.decompiler.StatEdge
import org.jetbrains.java.decompiler.modules.decompiler.exps.FunctionExprent.FunctionType
import org.jetbrains.java.decompiler.modules.decompiler.exps.*
import org.jetbrains.java.decompiler.modules.decompiler.stats.*
import org.jetbrains.java.decompiler.struct.gen.{CodeType, MethodDescriptor, VarType}
import polyregion.ast.{PolyAST as p, *}

import scala.collection.mutable
import scala.jdk.CollectionConverters.*

// lowers a Vineflower Statement/Exprent tree to PolyAST, see BYTECODE.md
object BytecodeRemapper {

  private val Global = p.Type.Space.Global

  case class Ref(owner: String, name: String, desc: String) {
    def repr: String = s"$owner.$name$desc"
  }

  def symbolOf(owner: String, name: String): p.Sym = p.Sym(s"$owner.$name")

  private def nameOf(t: VarType): String =
    Option(t.value).filter(_ => t.`type` == CodeType.OBJECT).fold(String.valueOf(t))(_.replace('/', '.'))

  def typeOf(t: VarType): Result[p.Type] = {
    val base = t.`type` match {
      case CodeType.BOOLEAN                    => p.Type.Bool1.success
      case CodeType.BYTE | CodeType.BYTECHAR   => p.Type.IntS8.success
      case CodeType.SHORT | CodeType.SHORTCHAR => p.Type.IntS16.success
      case CodeType.CHAR                       => p.Type.IntU16.success
      case CodeType.INT                        => p.Type.IntS32.success
      case CodeType.LONG                       => p.Type.IntS64.success
      case CodeType.FLOAT                      => p.Type.Float32.success
      case CodeType.DOUBLE                     => p.Type.Float64.success
      case CodeType.VOID                       => p.Type.Unit0.success
      case CodeType.NULL                       => "Unsupported `null`; PolyRegion has no device heap".fail
      case _ =>
        s"Cannot lower the JVM type `${nameOf(t)}`; only primitives and their arrays travel to a device".fail
    }
    base.map(b => (1 to t.arrayDim).foldLeft(b)((c, _) => p.Type.Ptr(c, Global)))
  }

  // the JVM computes on int, long, float and double only, narrower integrals are truncated back by i2b/i2c/i2s
  private def widen(t: p.Type): p.Type = t match {
    case p.Type.Bool1 | p.Type.IntS8 | p.Type.IntS16 | p.Type.IntU16 => p.Type.IntS32
    case other                                                       => other
  }

  private def promote(a: p.Type, b: p.Type): Result[p.Type] = (widen(a), widen(b)) match {
    case (x, y) if x == y                          => x.success
    case (p.Type.Float64, _) | (_, p.Type.Float64) => p.Type.Float64.success
    case (p.Type.Float32, _) | (_, p.Type.Float32) => p.Type.Float32.success
    case (p.Type.IntS64, _) | (_, p.Type.IntS64)   => p.Type.IntS64.success
    case (x, y) => s"No common type for ${x.repr} and ${y.repr} at a binary operator".fail
  }

  private def sanitise(s: String): String = s.map(c => if (c.isLetterOrDigit) c else '_')

  private def oneOf(t: p.Type): Result[p.Term] = t match {
    case p.Type.IntS32  => p.Term.IntS32Const(1).success
    case p.Type.IntS64  => p.Term.IntS64Const(1L).success
    case p.Type.Float32 => p.Term.Float32Const(1f).success
    case p.Type.Float64 => p.Term.Float64Const(1d).success
    case other          => s"No unit constant for ${other.repr}".fail
  }

  private enum Wrap { case Loop(node: Int); case Blk(node: Int) }

  private def blkId(n: Int): Int  = 2 * n + 1
  private def contId(n: Int): Int = 2 * n + 2

  private def stats(s: Statement): List[Statement] = s.getStats.asScala.toList

  private def descend(s: Statement): List[Statement] = s :: stats(s).flatMap(descend(_))

  private def isJump(e: StatEdge): Boolean =
    e.explicit && e.getType != StatEdge.TYPE_REGULAR && !e.getDestination.isInstanceOf[DummyExitStatement]

  private def exitEdge(s: Statement): Option[StatEdge] =
    s.getSuccessorEdges(Statement.STATEDGE_DIRECT_ALL).asScala.toList match {
      case e :: Nil => Some(e)
      case _        => None
    }

  // PolyAST's Break only leaves the innermost loop, so a non-loop statement that jumps close on becomes a run-once block
  private def needsBlock(s: Statement): Boolean =
    !s.isInstanceOf[DoStatement] && s.getLabelEdges.asScala.exists(isJump(_))

  // for and do..while run their condition and increment after the body, so continues route through the jump register
  private def hasTail(d: DoStatement): Boolean =
    d.getLooptype == DoStatement.Type.FOR || d.getLooptype == DoStatement.Type.DO_WHILE

  private def fits(e: StatEdge, ctx: List[Wrap]): Boolean = Option(e.closure).exists { c =>
    if (e.getType == StatEdge.TYPE_CONTINUE) ctx.headOption.contains(Wrap.Loop(c.id))
    else ctx.headOption.exists(w => w == Wrap.Blk(c.id) || w == Wrap.Loop(c.id))
  }

  // a jump past the innermost wrapper cannot use an unlabelled Break, so the function routes jumps through `#jmp`
  private def needsRegister(root: Statement): Boolean = {
    def edges(s: Statement): List[StatEdge] = {
      val ifEdges = s match {
        case f: IfStatement => List(Option(f.getIfEdge), Option(f.getElseEdge)).flatten
        case _              => Nil
      }
      (exitEdge(s).toList ::: ifEdges).filter(isJump(_))
    }
    def go(s: Statement, ctx: List[Wrap]): Boolean = {
      val ctxInner = s match {
        case d: DoStatement     => Wrap.Loop(d.id) :: ctx
        case _ if needsBlock(s) => Wrap.Blk(s.id) :: ctx
        case _                  => ctx
      }
      val tailContinue = s match {
        case d: DoStatement if hasTail(d) =>
          descend(d).exists(x => edges(x).exists(e => e.getType == StatEdge.TYPE_CONTINUE && e.closure == d))
        case _ => false
      }
      tailContinue ||
      stats(s).exists(go(_, ctxInner)) ||
      (s match {
        case f: IfStatement =>
          List(Option(f.getIfEdge), Option(f.getElseEdge)).flatten.filter(isJump(_)).exists(!fits(_, ctxInner))
        case _ => false
      }) ||
      exitEdge(s).filter(isJump(_)).exists(!fits(_, ctx))
    }
    go(root, Nil)
  }

  private class Lowerer(rtn: p.Type, register: Boolean) {

    private val ids = Iterator.from(0)
    val declared    = mutable.LinkedHashMap.empty[String, p.Named]
    val calls       = mutable.LinkedHashSet.empty[Ref]

    private val jmp: p.Term.Select = p.Term.Select(p.Named("#jmp", p.Type.IntS32), Nil, p.Type.IntS32)

    def fresh(t: p.Type): p.Named = p.Named(s"#v${ids.next()}", t)

    private def declare(symbol: String, t: p.Type): p.Named = declared.getOrElseUpdate(symbol, p.Named(symbol, t))

    def slot(index: Int, version: Int, t: p.Type): p.Term.Select = {
      val n = declare(s"l${index}_${version}_${sanitise(t.repr)}", t)
      p.Term.Select(n, Nil, t)
    }

    def bind(buf: mutable.ListBuffer[p.Stmt], e: p.Expr): p.Term = e match {
      case p.Expr.Alias(t) => t
      case other =>
        val n = fresh(other.tpe)
        buf += p.Stmt.Var(n, Some(other))
        p.Term.Select(n, Nil, n.tpe)
    }

    def coerce(buf: mutable.ListBuffer[p.Stmt], t: p.Term, target: p.Type): p.Term =
      if (t.tpe == target) t else bind(buf, p.Expr.Cast(t, target))

    // bind folds an alias away, which is wrong for a value read out of a slot about to be written
    private def snapshot(buf: mutable.ListBuffer[p.Stmt], t: p.Term): p.Term = t match {
      case p.Term.Select(_, _, _) =>
        val n = fresh(t.tpe)
        buf += p.Stmt.Var(n, Some(p.Expr.Alias(t)))
        p.Term.Select(n, Nil, n.tpe)
      case other => other
    }

    private def varOf(v: VarExprent): Result[p.Term.Select] =
      typeOf(v.getVarType).map(t => slot(v.getIndex, v.getVersion, t))

    private def constOf(c: ConstExprent): Result[p.Term] = {
      def num: Result[Scalar] = Option(c.getValue: Any) match {
        case Some(b: Boolean) => Scalar.I(if (b) 1 else 0).success
        case Some(_: String)  => "Unsupported `String` constant; PolyRegion has no device-side String".fail
        case None             => "Unsupported `null` constant; PolyRegion has no device heap".fail
        case Some(other) =>
          Scalar.of(other) match {
            case Some(n) => n.success
            case None    => s"Unsupported constant `${other.getClass.getName}`".fail
          }
      }
      typeOf(c.getConstType).flatMap { t =>
        num.flatMap(v =>
          Scalar.constant(t) match {
            case Some(lower) => lower(v).success
            case None        => s"Unsupported constant of type `${t.repr}`".fail
          }
        )
      }
    }

    private def invoke(i: InvocationExprent, buf: mutable.ListBuffer[p.Stmt]): Result[p.Term] =
      if (i.getInvocationType != InvocationExprent.InvocationType.STATIC) unsupported(i)
      else {
        val owner = i.getClassname.replace('/', '.')
        val md    = MethodDescriptor.parseDescriptor(i.getStringDescriptor)
        for {
          params <- md.params.toList.traverse(typeOf(_))
          rtn    <- typeOf(md.ret)
          args   <- i.getLstParameters.asScala.toList.traverse(expr(_, buf))
          lowered = args.zip(params).map((a, t) => coerce(buf, a, t))
          call = Intrinsics.lookup(owner, i.getName, i.getStringDescriptor, lowered, rtn).getOrElse {
            calls += Ref(owner, i.getName, i.getStringDescriptor)
            p.Expr.Invoke(p.Type.FnRef(symbolOf(owner, i.getName)), Nil, None, lowered, rtn)
          }
        } yield bind(buf, call)
      }

    private def unsupported(e: Exprent): Result[Nothing] = e match {
      case i: InvocationExprent =>
        s"Unsupported call `${i.getClassname.replace('/', '.')}.${i.getName}${i.getStringDescriptor}`".fail
      case f: FieldExprent =>
        s"Unsupported field access `${f.getClassname.replace('/', '.')}.${f.getName}`".fail
      case n: NewExprent =>
        val what = if (n.getNewType.arrayDim > 0) "array allocation" else "allocation"
        s"Unsupported $what of `${nameOf(n.getNewType)}`".fail
      case c: ConstExprent if c.getConstType == VarType.VARTYPE_STRING =>
        "Unsupported `String` constant; PolyRegion has no device-side String".fail
      case other => s"Unsupported expression `${other.getClass.getSimpleName}`".fail
    }

    private def arrayParts(a: ArrayExprent, buf: mutable.ListBuffer[p.Stmt]): Result[(p.Term.Select, p.Term, p.Type)] =
      for {
        arr <- expr(a.getArray, buf)
        idx <- expr(a.getIndex, buf)
        comp <- arr.tpe match {
          case p.Type.Ptr(c, _) => c.success
          case other            => s"Array access needs an array, got `${other.repr}`".fail
        }
        sel <- arr match {
          case s: p.Term.Select => s.success
          case other            => s"Array `${other.repr}` is not addressable".fail
        }
      } yield (sel, coerce(buf, idx, p.Type.IntS32), comp)

    // `unordered` is what the l and g compare forms report for a NaN operand
    private def threeWay(a: p.Term, b: p.Term, unordered: Int, buf: mutable.ListBuffer[p.Stmt]): p.Term = {
      val r                  = fresh(p.Type.IntS32)
      val sel: p.Term.Select = p.Term.Select(r, Nil, p.Type.IntS32)
      buf += p.Stmt.Var(r, Some(p.Expr.Alias(p.Term.IntS32Const(unordered))), isMutable = true)
      buf += p.Stmt.Cond(
        bind(buf, p.Expr.IntrOp(p.Intr.LogicEq(a, b))),
        List(p.Stmt.Mut(sel, p.Expr.Alias(p.Term.IntS32Const(0)))),
        List(
          p.Stmt.Cond(
            bind(buf, p.Expr.IntrOp(p.Intr.LogicLt(a, b))),
            List(p.Stmt.Mut(sel, p.Expr.Alias(p.Term.IntS32Const(-1)))),
            List(
              p.Stmt.Cond(
                bind(buf, p.Expr.IntrOp(p.Intr.LogicGt(a, b))),
                List(p.Stmt.Mut(sel, p.Expr.Alias(p.Term.IntS32Const(1)))),
                Nil
              )
            )
          )
        )
      )
      sel
    }

    private def binary(f: FunctionExprent, buf: mutable.ListBuffer[p.Stmt]): Result[p.Term] = {
      val ops = f.getLstOperands
      for {
        a <- expr(ops.get(0), buf)
        b <- expr(ops.get(1), buf)
        r <- apply(f.getFuncType, a, b, buf)
      } yield r
    }

    // the JVM masks a shift distance to the operand width, C and LLVM leave the wider case undefined
    private def shift(t: p.Type, by: p.Term, logical: Boolean, a: p.Term, buf: mutable.ListBuffer[p.Stmt])(
        op: (p.Term, p.Term, p.Type) => p.Intr
    ): p.Term = {
      val mask   = if (t == p.Type.IntS64) 63 else 31
      val amount = coerce(buf, by, p.Type.IntS32)
      val masked = bind(buf, p.Expr.IntrOp(p.Intr.BAnd(amount, p.Term.IntS32Const(mask), p.Type.IntS32)))
      // an unsigned shift on a signed type would drag the sign bit in, so it runs on the unsigned twin
      val opTpe = if (!logical) t else if (t == p.Type.IntS64) p.Type.IntU64 else p.Type.IntU32
      val r     = p.Expr.IntrOp(op(coerce(buf, a, opTpe), coerce(buf, masked, opTpe), opTpe))
      coerce(buf, bind(buf, r), t)
    }

    private def apply(ft: FunctionType, a: p.Term, b: p.Term, buf: mutable.ListBuffer[p.Stmt]): Result[p.Term] = {
      def arith(op: (p.Term, p.Term, p.Type) => p.Intr): Result[p.Term] = promote(a.tpe, b.tpe).map { t =>
        bind(buf, p.Expr.IntrOp(op(coerce(buf, a, t), coerce(buf, b, t), t)))
      }
      def compare(op: (p.Term, p.Term) => p.Intr): Result[p.Term] = promote(a.tpe, b.tpe).map { t =>
        bind(buf, p.Expr.IntrOp(op(coerce(buf, a, t), coerce(buf, b, t))))
      }
      def bitwise(bool: (p.Term, p.Term) => p.Intr, int: (p.Term, p.Term, p.Type) => p.Intr): Result[p.Term] =
        if (a.tpe == p.Type.Bool1 && b.tpe == p.Type.Bool1) bind(buf, p.Expr.IntrOp(bool(a, b))).success
        else arith(int)
      ft match {
        case FunctionType.ADD => arith(p.Intr.Add(_, _, _))
        case FunctionType.SUB => arith(p.Intr.Sub(_, _, _))
        case FunctionType.MUL => arith(p.Intr.Mul(_, _, _))
        case FunctionType.DIV => arith(p.Intr.Div(_, _, _))
        case FunctionType.REM => arith(p.Intr.Rem(_, _, _))
        case FunctionType.AND => bitwise(p.Intr.LogicAnd(_, _), p.Intr.BAnd(_, _, _))
        case FunctionType.OR  => bitwise(p.Intr.LogicOr(_, _), p.Intr.BOr(_, _, _))
        case FunctionType.XOR => bitwise(p.Intr.LogicNeq(_, _), p.Intr.BXor(_, _, _))

        case FunctionType.SHL  => shift(widen(a.tpe), b, logical = false, a, buf)(p.Intr.BSL(_, _, _)).success
        case FunctionType.SHR  => shift(widen(a.tpe), b, logical = false, a, buf)(p.Intr.BSR(_, _, _)).success
        case FunctionType.USHR => shift(widen(a.tpe), b, logical = true, a, buf)(p.Intr.BZSR(_, _, _)).success

        case FunctionType.EQ => compare(p.Intr.LogicEq(_, _))
        case FunctionType.NE => compare(p.Intr.LogicNeq(_, _))
        case FunctionType.LT => compare(p.Intr.LogicLt(_, _))
        case FunctionType.LE => compare(p.Intr.LogicLte(_, _))
        case FunctionType.GT => compare(p.Intr.LogicGt(_, _))
        case FunctionType.GE => compare(p.Intr.LogicGte(_, _))

        case FunctionType.LCMP => promote(a.tpe, b.tpe).map(t => threeWay(coerce(buf, a, t), coerce(buf, b, t), 0, buf))
        case FunctionType.FCMPL =>
          promote(a.tpe, b.tpe).map(t => threeWay(coerce(buf, a, t), coerce(buf, b, t), -1, buf))
        case FunctionType.DCMPL =>
          promote(a.tpe, b.tpe).map(t => threeWay(coerce(buf, a, t), coerce(buf, b, t), -1, buf))
        case FunctionType.FCMPG =>
          promote(a.tpe, b.tpe).map(t => threeWay(coerce(buf, a, t), coerce(buf, b, t), 1, buf))
        case FunctionType.DCMPG =>
          promote(a.tpe, b.tpe).map(t => threeWay(coerce(buf, a, t), coerce(buf, b, t), 1, buf))

        case other => s"Unsupported binary operator `$other`".fail
      }
    }

    private def unify(a: p.Type, b: p.Type): Result[p.Type] =
      if (a == b) a.success else promote(a, b)

    private def shortCircuit(f: FunctionExprent, buf: mutable.ListBuffer[p.Stmt]): Result[p.Term] = {
      val ops = f.getLstOperands
      val rhs = mutable.ListBuffer.empty[p.Stmt]
      for {
        a <- expr(ops.get(0), buf)
        b <- expr(ops.get(1), rhs)
      } yield {
        val l                  = coerce(buf, a, p.Type.Bool1)
        val n                  = fresh(p.Type.Bool1)
        val sel: p.Term.Select = p.Term.Select(n, Nil, p.Type.Bool1)
        buf += p.Stmt.Var(n, Some(p.Expr.Alias(l)), isMutable = true)
        rhs += p.Stmt.Mut(sel, p.Expr.Alias(coerce(rhs, b, p.Type.Bool1)))
        if (f.getFuncType == FunctionType.BOOLEAN_AND) buf += p.Stmt.Cond(l, rhs.toList, Nil)
        else buf += p.Stmt.Cond(bind(buf, p.Expr.IntrOp(p.Intr.LogicNot(l))), rhs.toList, Nil)
        sel
      }
    }

    private def ternary(f: FunctionExprent, buf: mutable.ListBuffer[p.Stmt]): Result[p.Term] = {
      val ops = f.getLstOperands
      val tb  = mutable.ListBuffer.empty[p.Stmt]
      val fb  = mutable.ListBuffer.empty[p.Stmt]
      for {
        c <- expr(ops.get(0), buf)
        t <- expr(ops.get(1), tb)
        e <- expr(ops.get(2), fb)
        r <- unify(t.tpe, e.tpe)
      } yield {
        val n                  = fresh(r)
        val sel: p.Term.Select = p.Term.Select(n, Nil, r)
        buf += p.Stmt.Var(n, None, isMutable = true)
        tb += p.Stmt.Mut(sel, p.Expr.Alias(coerce(tb, t, r)))
        fb += p.Stmt.Mut(sel, p.Expr.Alias(coerce(fb, e, r)))
        buf += p.Stmt.Cond(coerce(buf, c, p.Type.Bool1), tb.toList, fb.toList)
        sel
      }
    }

    // `i++` in value position keeps the old value, so every intermediate lands in a temporary
    private def step(f: FunctionExprent, buf: mutable.ListBuffer[p.Stmt], wantValue: Boolean): Result[p.Term] = {
      val target    = f.getLstOperands.get(0)
      val increment = f.getFuncType == FunctionType.IPP || f.getFuncType == FunctionType.PPI
      val post      = f.getFuncType == FunctionType.IPP || f.getFuncType == FunctionType.IMM
      def update(read: p.Term, write: (p.Term, mutable.ListBuffer[p.Stmt]) => Unit): Result[p.Term] = {
        val t = widen(read.tpe)
        oneOf(t).map { one =>
          val before  = if (wantValue && post) snapshot(buf, read) else read
          val current = coerce(buf, before, t)
          val next    = p.Expr.IntrOp(if (increment) p.Intr.Add(current, one, t) else p.Intr.Sub(current, one, t))
          val stored  = coerce(buf, bind(buf, next), read.tpe)
          write(stored, buf)
          if (!wantValue) p.Term.Unit0Const else if (post) before else snapshot(buf, stored)
        }
      }
      target match {
        case v: VarExprent =>
          varOf(v).flatMap(sel => update(sel, (value, b) => b += p.Stmt.Mut(sel, p.Expr.Alias(value))))
        case a: ArrayExprent =>
          arrayParts(a, buf).flatMap { (sel, idx, comp) =>
            val read = bind(buf, p.Expr.Index(sel, idx, comp))
            update(read, (value, b) => b += p.Stmt.Update(sel, idx, coerce(b, value, comp)))
          }
        case other => s"Unsupported increment target `${other.getClass.getSimpleName}`".fail
      }
    }

    private def function(f: FunctionExprent, buf: mutable.ListBuffer[p.Stmt]): Result[p.Term] = f.getFuncType match {
      case FunctionType.BOOLEAN_AND | FunctionType.BOOLEAN_OR => shortCircuit(f, buf)
      case FunctionType.TERNARY                               => ternary(f, buf)
      case ft if ft.isPPMM                                    => step(f, buf, wantValue = true)
      case FunctionType.BOOL_NOT =>
        expr(f.getLstOperands.get(0), buf).map(a =>
          bind(buf, p.Expr.IntrOp(p.Intr.LogicNot(coerce(buf, a, p.Type.Bool1))))
        )
      case FunctionType.BIT_NOT =>
        expr(f.getLstOperands.get(0), buf).map { a =>
          val t = widen(a.tpe)
          bind(buf, p.Expr.IntrOp(p.Intr.BNot(coerce(buf, a, t), t)))
        }
      case FunctionType.NEG =>
        expr(f.getLstOperands.get(0), buf).map { a =>
          val t = widen(a.tpe)
          bind(buf, p.Expr.IntrOp(p.Intr.Neg(coerce(buf, a, t), t)))
        }
      case FunctionType.CAST =>
        for {
          a <- expr(f.getLstOperands.get(0), buf)
          t <- typeOf(f.getLstOperands.get(1).getExprType)
        } yield coerce(buf, a, t)
      case ft if Option(ft.castType).isDefined =>
        for {
          a <- expr(f.getLstOperands.get(0), buf)
          t <- typeOf(ft.castType)
        } yield coerce(buf, a, t)
      case FunctionType.ARRAY_LENGTH =>
        "Unsupported `length` of an array; PolyRegion carries array lengths as separate arguments".fail
      case FunctionType.INSTANCEOF => "Unsupported `instanceof`; PolyRegion has no device heap".fail
      case FunctionType.STR_CONCAT => "Unsupported string concatenation; PolyRegion has no device-side String".fail
      case _                       => binary(f, buf)
    }

    def expr(e: Exprent, buf: mutable.ListBuffer[p.Stmt]): Result[p.Term] = e match {
      case v: VarExprent      => varOf(v)
      case c: ConstExprent    => constOf(c)
      case f: FunctionExprent => function(f, buf)
      case a: ArrayExprent =>
        arrayParts(a, buf).map((sel, idx, comp) => bind(buf, p.Expr.Index(sel, idx, comp)))
      case a: AssignmentExprent => assign(a, buf)
      case i: InvocationExprent => invoke(i, buf)
      case other                => unsupported(other)
    }

    private def assign(a: AssignmentExprent, buf: mutable.ListBuffer[p.Stmt]): Result[p.Term] = {
      def value(current: p.Term): Result[p.Term] =
        expr(a.getRight, buf).flatMap { r =>
          Option(a.getCondType).fold(r.success)(ft => apply(ft, current, r, buf))
        }
      a.getLeft match {
        case v: VarExprent =>
          for {
            sel <- varOf(v)
            // Java evaluates the target of a compound assignment before the right hand side
            r <- value(if (Option(a.getCondType).isEmpty) sel else snapshot(buf, sel))
          } yield {
            val stored = coerce(buf, r, sel.tpe)
            buf += p.Stmt.Mut(sel, p.Expr.Alias(stored))
            stored
          }
        case l: ArrayExprent =>
          for {
            parts <- arrayParts(l, buf)
            (sel, idx, comp) = parts
            r <- value(bind(buf, p.Expr.Index(sel, idx, comp)))
          } yield {
            val stored = coerce(buf, r, comp)
            buf += p.Stmt.Update(sel, idx, stored)
            stored
          }
        case other => unsupported(other)
      }
    }

    def effect(e: Exprent, buf: mutable.ListBuffer[p.Stmt]): Result[Unit] = e match {
      case a: AssignmentExprent                       => assign(a, buf).map(_ => ())
      case f: FunctionExprent if f.getFuncType.isPPMM => step(f, buf, wantValue = false).map(_ => ())
      case _: VarExprent                              => ().success
      case i: InvocationExprent                       => invoke(i, buf).map(_ => ())
      case x: ExitExprent =>
        x.getExitType match {
          case ExitExprent.Type.RETURN =>
            Option(x.getValue) match {
              case None => (buf += p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))).success
              case Some(v) =>
                expr(v, buf).map(t => buf += p.Stmt.Return(p.Expr.Alias(coerce(buf, t, rtn))))
            }
          case ExitExprent.Type.THROW => "Unsupported `throw`; PolyRegion kernels cannot throw".fail
        }
      case other => unsupported(other)
    }

    private def set(v: Int): p.Stmt = p.Stmt.Mut(jmp, p.Expr.Alias(p.Term.IntS32Const(v)))

    // a labelled block is spelled `while (once) { once = false; ... }`, the only thing Break can leave
    private def runOnce(buf: mutable.ListBuffer[p.Stmt], body: List[p.Stmt]): Unit = {
      val once               = fresh(p.Type.Bool1)
      val sel: p.Term.Select = p.Term.Select(once, Nil, p.Type.Bool1)
      buf += p.Stmt.Var(once, Some(p.Expr.Alias(p.Term.Bool1Const(true))), isMutable = true)
      buf += p.Stmt.While(sel, p.Stmt.Mut(sel, p.Expr.Alias(p.Term.Bool1Const(false))) :: body)
    }

    private def arrive(buf: mutable.ListBuffer[p.Stmt], id: Int, outer: Boolean): Unit = {
      val inFlight = bind(buf, p.Expr.IntrOp(p.Intr.LogicNeq(jmp, p.Term.IntS32Const(0))))
      val mine     = bind(buf, p.Expr.IntrOp(p.Intr.LogicEq(jmp, p.Term.IntS32Const(id))))
      buf += p.Stmt.Cond(inFlight, List(p.Stmt.Cond(mine, List(set(0)), if (outer) List(p.Stmt.Break) else Nil)), Nil)
    }

    private def jump(e: StatEdge, ctx: List[Wrap]): Result[List[p.Stmt]] =
      Option(e.closure).failIfEmpty("Vineflower produced a break or continue with no enclosing statement").flatMap {
        c =>
          val n = c.id
          if (e.getType == StatEdge.TYPE_CONTINUE) {
            if (ctx.headOption.contains(Wrap.Loop(n))) List(if (register) p.Stmt.Break else p.Stmt.Cont).success
            else if (register) List(set(contId(n)), p.Stmt.Break).success
            else s"Internal error: `continue` to loop $n is not innermost".fail
          } else if (!register && ctx.headOption.exists(w => w == Wrap.Blk(n) || w == Wrap.Loop(n)))
            List(p.Stmt.Break).success
          else if (register) List(set(blkId(n)), p.Stmt.Break).success
          else s"Internal error: `break` to statement $n is not innermost".fail
      }

    private def after(s: Statement, ctx: List[Wrap]): Result[List[p.Stmt]] =
      exitEdge(s).filter(isJump(_)).fold(List.empty[p.Stmt].success)(jump(_, ctx))

    private def sequence(xs: List[Statement], ctx: List[Wrap]): Result[List[p.Stmt]] =
      xs.traverse(x => statement(x, ctx).map2(after(x, ctx))(_ ++ _)).map(_.flatten)

    private def loop(d: DoStatement, ctx: List[Wrap]): Result[List[p.Stmt]] = {
      val head = mutable.ListBuffer.empty[p.Stmt]
      val tail = mutable.ListBuffer.empty[p.Stmt]
      val exit = mutable.ListBuffer.empty[p.Stmt]
      val kind = d.getLooptype
      def test(buf: mutable.ListBuffer[p.Stmt]): Result[Unit] = Option(d.getConditionExprent) match {
        case None => ().success
        case Some(c) =>
          expr(c, buf).map { t =>
            val neg = bind(buf, p.Expr.IntrOp(p.Intr.LogicNot(coerce(buf, t, p.Type.Bool1))))
            buf += p.Stmt.Cond(neg, List(p.Stmt.Break), Nil)
          }
      }
      for {
        _    <- Option(d.getInitExprent).fold(().success)(effect(_, exit))
        _    <- if (kind == DoStatement.Type.DO_WHILE) ().success else test(head)
        body <- sequence(stats(d), Wrap.Loop(d.id) :: ctx)
        _    <- if (kind == DoStatement.Type.DO_WHILE) test(tail) else ().success
        _    <- Option(d.getIncExprent).fold(().success)(effect(_, tail))
      } yield {
        val inner = mutable.ListBuffer.empty[p.Stmt]
        inner ++= head
        if (!register) inner ++= body
        else { runOnce(inner, body); arrive(inner, contId(d.id), outer = true) }
        inner ++= tail
        exit += p.Stmt.While(p.Term.Bool1Const(true), inner.toList)
        if (register) arrive(exit, blkId(d.id), ctx.nonEmpty)
        exit.toList
      }
    }

    private def branch(f: IfStatement, ctx: List[Wrap]): Result[List[p.Stmt]] = {
      val buf = mutable.ListBuffer.empty[p.Stmt]
      def side(body: Statement, edge: StatEdge): Result[List[p.Stmt]] =
        (Option(body), Option(edge)) match {
          case (Some(b), _)                 => statement(b, ctx).map2(after(b, ctx))(_ ++ _)
          case (None, Some(e)) if isJump(e) => jump(e, ctx)
          case _                            => Nil.success
        }
      for {
        head <- statement(f.getFirst, ctx)
        cond <- Option(f.getHeadexprent).failIfEmpty("Vineflower left an `if` without a condition")
        c    <- expr(cond.getCondition, buf)
        t    <- side(f.getIfstat, f.getIfEdge)
        e    <- side(f.getElsestat, f.getElseEdge)
      } yield head ++ (buf += p.Stmt.Cond(coerce(buf, c, p.Type.Bool1), t, e)).toList
    }

    private def block(s: Statement, buf: mutable.ListBuffer[p.Stmt]): Result[Unit] =
      Option(s.getExprents).fold(().success)(_.asScala.toList.traverse_(effect(_, buf)))

    private def body(s: Statement, ctx: List[Wrap]): Result[List[p.Stmt]] = s match {
      case r: RootStatement     => statement(r.getFirst, ctx).map2(after(r.getFirst, ctx))(_ ++ _)
      case d: DoStatement       => loop(d, ctx)
      case f: IfStatement       => branch(f, ctx)
      case q: SequenceStatement => sequence(stats(q), ctx)
      case b: BasicBlockStatement =>
        val buf = mutable.ListBuffer.empty[p.Stmt]
        block(b, buf).map(_ => buf.toList)
      case _: SwitchStatement       => "Unsupported `switch`".fail
      case _: CatchStatement        => "Unsupported `try`/`catch`; PolyRegion kernels cannot throw".fail
      case _: CatchAllStatement     => "Unsupported `finally`; PolyRegion kernels cannot throw".fail
      case _: SynchronizedStatement => "Unsupported `synchronized`; PolyRegion kernels have no monitors".fail
      case _: GeneralStatement =>
        "Vineflower could not recover structured control flow for this region".fail
      case other => s"Unsupported statement `${other.getClass.getSimpleName}`".fail
    }

    def statement(s: Statement, ctx: List[Wrap]): Result[List[p.Stmt]] =
      if (!needsBlock(s)) body(s, ctx)
      else
        body(s, Wrap.Blk(s.id) :: ctx).map { inner =>
          val buf = mutable.ListBuffer.empty[p.Stmt]
          runOnce(buf, inner)
          if (register) arrive(buf, blkId(s.id), ctx.nonEmpty)
          buf.toList
        }
  }

  private def varsOf(root: Statement): List[VarExprent] =
    for {
      s <- descend(root)
      e <- s.getVarDefinitions.asScala.toList ::: Option(s.getStatExprents).toList.flatMap(_.asScala)
      v <- (e :: e.getAllExprents(true).asScala.toList).collect { case v: VarExprent => v }
    } yield v

  def lower(classBytes: Array[Byte], name: String, desc: String): Result[p.Function] =
    lowerWithCalls(classBytes, name, desc).map(_._1)

  // the second component is every static callee the body invokes that no intrinsic covers
  def lowerWithCalls(classBytes: Array[Byte], name: String, desc: String): Result[(p.Function, List[Ref])] =
    Vineflower.structure(classBytes, name, desc)(lowerMethod(_, name))

  private def lowerMethod(m: Vineflower.Method, name: String): Result[(p.Function, List[Ref])] = for {
    owner    <- m.cl.qualifiedName.replace('/', '.').success
    isStati  <- m.mt.hasModifier(CodeConstants.ACC_STATIC).success
    slotVars <- varsOf(m.root).success
    _        <-
      // a `this`-free instance method is a static one with a slot reserved, the shape of every Scala `object` method
      if (!isStati && slotVars.exists(_.getIndex == 0))
        s"`$owner.$name` is not static and uses its receiver; instance methods need a receiver struct".fail
      else ().success
    rtn      <- typeOf(m.md.ret)
    argTypes <- m.md.params.toList.traverse(typeOf(_))
    register = needsRegister(m.root)
    lowerer  = Lowerer(rtn, register)
    base     = if (isStati) 0 else 1
    slots = argTypes
      .foldLeft((List.empty[(Int, p.Type)], base)) { case ((acc, at), t) => ((at, t) :: acc, at + words(t)) }
      ._1
      .reverse
    args = slots.map((at, t) => p.Arg(p.Named(s"a$at", t)))
    // Vineflower narrows a parameter to what it inferred, so the copy lands in the variable the body reads
    entries <- slots.traverse { (at, declaredTpe) =>
      slotVars.filter(_.getIndex == at).minByOption(_.getVersion) match {
        case None    => (at, 0, declaredTpe).success
        case Some(v) => typeOf(v.getVarType).map(t => (at, v.getVersion, t))
      }
    }
    body <- lowerer.statement(m.root, Nil)
  } yield {
    val copies = entries.zip(args).map { (entry, arg) =>
      val (at, version, tpe) = entry
      val target             = lowerer.slot(at, version, tpe)
      val src                = p.Term.Select(arg.named, Nil, arg.named.tpe)
      p.Stmt.Mut(target, if (target.tpe == src.tpe) p.Expr.Alias(src) else p.Expr.Cast(src, target.tpe))
    }
    val prologue = mutable.ListBuffer.empty[p.Stmt]
    if (register)
      prologue += p.Stmt.Var(
        p.Named("#jmp", p.Type.IntS32),
        Some(p.Expr.Alias(p.Term.IntS32Const(0))),
        isMutable = true
      )
    // PolyAST scopes a Var to its block and a slot written in a loop is read after it, so slots are hoisted
    lowerer.declared.values.foreach(n => prologue += p.Stmt.Var(n, None, isMutable = true))
    val fn = p.Function(
      decl = p.FunctionDecl(
        name = symbolOf(owner, name),
        tpeVars = Nil,
        receiver = None,
        args = args,
        moduleCaptures = Nil,
        termCaptures = Nil,
        rtn = rtn,
        affinity = p.Function.Affinity.Offload
      ),
      body = prologue.toList ++ copies ++ body ++ Option.when(
        rtn == p.Type.Unit0 && !body.lastOption.exists(_.isInstanceOf[p.Stmt.Return])
      )(
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      ),
      visibility = p.Function.Visibility.Exported,
      fpMode = p.Function.FpMode.Relaxed,
      convention = p.CallConvention.RegularCall
    )
    (fn, lowerer.calls.toList)
  }

  private def words(t: p.Type): Int = t match {
    case p.Type.IntS64 | p.Type.Float64 => 2
    case _                              => 1
  }

  def lowerToProgram(classBytes: Array[Byte], name: String, desc: String): Result[p.Program] =
    lower(classBytes, name, desc).map(f => p.Program(Some(f), Nil, Nil))
}
