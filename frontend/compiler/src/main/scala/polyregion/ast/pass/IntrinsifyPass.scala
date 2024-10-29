package polyregion.ast.pass

import cats.syntax.all.*
import polyregion.ast.{PolyAST as p, *, given}
import polyregion.scalalang.Symbols
import polyregion.ast.Traversal.*
import polyregion.scalalang.Symbols.Intrinsics

object IntrinsifyPass extends ProgramPass {

  override def apply(program: p.Program, log: Log): p.Program = {
    val subLog = log.subLog("Intrinsify")
    program.copy(
      entry = intrinsifyOne(program.entry, subLog),
      functions = program.functions.map(intrinsifyOne(_, subLog))
    )
  }

  private def intrinsifyOne(f: p.Function, log: Log): (p.Function) = {
    val (xs, instanceInvokes) = f.body.zipWithIndex.foldMapM(intrinsifyInstanceApply(_, _))
    val (ys, moduleInvokes)   = xs.zipWithIndex.foldMapM(intrinsifyModuleApply(_, _))
    log.info(s"${f.signatureRepr}: ", (instanceInvokes ++ moduleInvokes).map(_.repr)*)
    // val eliminated = dep.functions.flatMap { (fn, ivks) =>
    //   val xs = ivks.filterNot(intrinsified.contains(_))
    //   if (xs.isEmpty) Map() else Map(fn -> xs)
    // }
    // (ys, dep.copy(functions = eliminated))
    f.copy(body = ys)
  }

  private final inline val DegreesToRadians = 0.017453292519943295

  private final inline val RadiansToDegrees = 57.29577951308232

  // 5.6.1. Unary Numeric Promotion
  private def unaryPromote(t: p.Type) = t match {
    case p.Type.IntS8 | p.Type.IntS16 | p.Type.IntU16 => p.Type.IntS32
    case x                                            => x
  }

  // 5.6.2. Binary Numeric Promotion
  private def binaryPromote(t: p.Type, u: p.Type) = Set(t, u) match {
    case xs if xs contains p.Type.Float64 => p.Type.Float64
    case xs if xs contains p.Type.Float32 => p.Type.Float32
    case xs if xs contains p.Type.IntS64  => p.Type.IntS64
    case _                                => p.Type.IntS32
  }

  private def binaryPromoteIntr[A](x: p.Term, y: p.Term, upper: p.Type, idx: Int)(f: (p.Term, p.Term) => A) = {
    val (xVal, xStmts) = castOrId(x, upper, s"intr_l${idx}")
    val (yVal, yStmts) = castOrId(y, upper, s"intr_r${idx}")
    (f(xVal, yVal), xStmts ++ yStmts)
  }

  private def unaryPromoteIntr[A](x: p.Term, upper: p.Type, idx: Int)(f: (p.Term) => A) = {
    val (xVal, xStmts) = castOrId(x, upper, s"intr_${idx}")
    (f(xVal), xStmts)
  }

  private def castOrId(x: p.Term, to: p.Type, name: String): (p.Term, List[p.Stmt]) =
    if (x.tpe == to) {
      (x, Nil)
    } else {
      val named = p.Named(name, to)
      (p.Term.Select(Nil, named), p.Stmt.Var(named, Some(p.Expr.Cast(x, to))) :: Nil)
    }

  // private def unaryNumericIntrinsic(x: p.Term, idx: Int, kind: p.UnaryIntrinsicKind) = {
  //   val (xVal, xStmts) = castOrId(x, unaryPromote(x.tpe), s"intr_${idx}")
  //   (p.Expr.UnaryIntrinsic(xVal, kind, xVal.tpe), xStmts)
  // }

  // def binaryNumericIntrinsic(x: p.Term, y: p.Term, upper: p.Type, idx: Int, kind: p.BinaryIntrinsicKind) = {
  //   val (xVal, xStmts) = castOrId(x, upper, s"intr_l${idx}")
  //   val (yVal, yStmts) = castOrId(y, upper, s"intr_r${idx}")
  //   val tpe = kind match {
  //     case p.BinaryIntrinsicKind.LogicEq | p.BinaryIntrinsicKind.LogicNeq | p.BinaryIntrinsicKind.LogicAnd |
  //         p.BinaryIntrinsicKind.LogicOr | p.BinaryIntrinsicKind.LogicLte | p.BinaryIntrinsicKind.LogicGte |
  //         p.BinaryIntrinsicKind.LogicLt | p.BinaryIntrinsicKind.LogicGt =>
  //       p.Type.Bool1
  //     case _ => upper
  //   }
  //   (p.Expr.BinaryIntrinsic(xVal, yVal, kind, tpe), xStmts ++ yStmts)
  // }

  private def intrinsifyNamed(
      op: String,
      args: List[p.Term],
      tpeArgs: List[p.Type],
      rtn: p.Type
  ): (p.Expr, List[p.Stmt]) = {

    def intr(x: p.Intr) = p.Expr.IntrOp(x) -> List.empty[p.Stmt]
    def math(x: p.Math) = p.Expr.MathOp(x) -> List.empty[p.Stmt]
    def spec(x: p.Spec) = p.Expr.SpecOp(x) -> List.empty[p.Stmt]

    (op, args) match {
      case "assert" -> Nil => spec(p.Spec.Assert)

      case "gpuGlobalIdx" -> (x :: Nil)  => spec(p.Spec.GpuGlobalIdx(x))
      case "gpuGlobalSize" -> (x :: Nil) => spec(p.Spec.GpuGlobalSize(x))
      case "gpuGroupIdx" -> (x :: Nil)   => spec(p.Spec.GpuGroupIdx(x))
      case "gpuGroupSize" -> (x :: Nil)  => spec(p.Spec.GpuGroupSize(x))
      case "gpuLocalIdx" -> (x :: Nil)   => spec(p.Spec.GpuLocalIdx(x))
      case "gpuLocalSize" -> (x :: Nil)  => spec(p.Spec.GpuLocalSize(x))

      case "gpuBarrierGlobal" -> Nil => spec(p.Spec.GpuBarrierGlobal)
      case "gpuFenceGlobal" -> Nil   => spec(p.Spec.GpuFenceGlobal)
      case "gpuBarrierLocal" -> Nil  => spec(p.Spec.GpuBarrierLocal)
      case "gpuFenceLocal" -> Nil    => spec(p.Spec.GpuFenceLocal)
      case "gpuBarrierAll" -> Nil    => spec(p.Spec.GpuBarrierAll)
      case "gpuFenceAll" -> Nil      => spec(p.Spec.GpuFenceAll)

      case "sin" -> (x :: Nil)  => math(p.Math.Sin(x, rtn))
      case "cos" -> (x :: Nil)  => math(p.Math.Cos(x, rtn))
      case "tan" -> (x :: Nil)  => math(p.Math.Tan(x, rtn))
      case "asin" -> (x :: Nil) => math(p.Math.Asin(x, rtn))
      case "acos" -> (x :: Nil) => math(p.Math.Acos(x, rtn))
      case "atan" -> (x :: Nil) => math(p.Math.Atan(x, rtn))
      case "sinh" -> (x :: Nil) => math(p.Math.Sinh(x, rtn))
      case "cosh" -> (x :: Nil) => math(p.Math.Cosh(x, rtn))
      case "tanh" -> (x :: Nil) => math(p.Math.Tanh(x, rtn))

      case "signum" -> (x :: Nil) => math(p.Math.Signum(x, rtn))
      case "abs" -> (x :: Nil)    => math(p.Math.Abs(x, rtn))
      case "round" -> (x :: Nil)  => math(p.Math.Round(x, rtn))
      case "ceil" -> (x :: Nil)   => math(p.Math.Ceil(x, rtn))
      case "floor" -> (x :: Nil)  => math(p.Math.Floor(x, rtn))
      case "rint" -> (x :: Nil)   => math(p.Math.Rint(x, rtn))

      case "sqrt" -> (x :: Nil)  => math(p.Math.Sqrt(x, rtn))
      case "cbrt" -> (x :: Nil)  => math(p.Math.Cbrt(x, rtn))
      case "exp" -> (x :: Nil)   => math(p.Math.Exp(x, rtn))
      case "expm1" -> (x :: Nil) => math(p.Math.Expm1(x, rtn))
      case "log" -> (x :: Nil)   => math(p.Math.Log(x, rtn))
      case "log1p" -> (x :: Nil) => math(p.Math.Log1p(x, rtn))
      case "log10" -> (x :: Nil) => math(p.Math.Log10(x, rtn))

      case "pow" -> (x :: y :: Nil)   => math(p.Math.Pow(x, y, rtn))
      case "min" -> (x :: y :: Nil)   => intr(p.Intr.Min(x, y, rtn))
      case "max" -> (x :: y :: Nil)   => intr(p.Intr.Max(x, y, rtn))
      case "atan2" -> (x :: y :: Nil) => math(p.Math.Atan2(x, y, rtn))
      case "hypot" -> (x :: y :: Nil) => math(p.Math.Hypot(x, y, rtn))

      case "array" -> (x :: Nil) //
          if x.tpe == p.Type.IntS32 =>
        p.Expr.Alloc(p.Type.Ptr(tpeArgs.head, None, p.Type.Space.Global), x) -> Nil
      case "apply" -> ((s @ p.Term.Select(_, p.Named(_, p.Type.Ptr(`rtn`, _, _)))) :: i :: Nil) //
          if i.tpe == p.Type.IntS32 =>
        p.Expr.Index(s, i, rtn) -> Nil
      case "update" -> ((s @ p.Term.Select(_, p.Named(_, p.Type.Ptr(c, _, _)))) :: i :: x :: Nil) //
          if i.tpe == p.Type.IntS32 && x.tpe == c && rtn == p.Type.Unit0 =>
        p.Expr.Alias(p.Term.Unit0Const) -> (p.Stmt.Update(s, i, x) :: Nil)
      case _ => ???
    }

  }

  private def intrinsifyInstanceApply(s: p.Stmt, idx: Int): (List[p.Stmt], List[p.Expr.Invoke]) = {
    val (stmt, cs) = s.modifyCollect[p.Expr, (List[p.Stmt], List[p.Expr.Invoke])] {
      case inv @ p.Expr.Invoke(sym, tpeArgs, Some(recv), args, captures, rtn) =>
        (sym.fqn, recv, args) match {
          case (
                "polyregion" :: "scalalang" :: "intrinsics$" :: op :: Nil,
                p.Term
                  .Select(
                    Nil,
                    p.Named(_, p.Type.Struct(p.Sym("polyregion" :: "scalalang" :: "intrinsics$" :: Nil), _, _, _))
                  ),
                xs
              ) =>
            println(s">>> ${recv} $op[${tpeArgs.map(_.repr)}](${xs.map(_.repr)}) : $rtn")

            val (expr, stmts) = intrinsifyNamed(op, args, tpeArgs, rtn)
            (expr, (stmts, inv :: Nil))

          case (
                "polyregion" :: "scalalang" :: "intrinsics$" :: "TypedBuffer" :: op :: Nil,
                xs @ p.Term
                  .Select(
                    _,
                    _
                  ),
                args
              ) =>
            (op, args) match {
              case "update" -> (i :: x :: Nil) if i.tpe.kind == p.TypeKind.Integral =>
                (p.Expr.Alias(p.Term.Unit0Const), (p.Stmt.Update(xs, i, x) :: Nil, inv :: Nil))
              case "apply" -> (i :: Nil) if i.tpe.kind == p.TypeKind.Integral =>
                (p.Expr.Index(xs, i, rtn), (Nil, inv :: Nil))
              case (op, args) =>
                println(s"$op $$args")
                ???
            }
          case (_ :+ op, x, y :: Nil) if x.tpe == p.Type.Bool1 && y.tpe == p.Type.Bool1 && rtn == p.Type.Bool1 =>
            val (expr, stmts) = op match {
              case "&&" => (p.Expr.IntrOp(p.Intr.LogicAnd(x, y)), Nil)
              case "||" => (p.Expr.IntrOp(p.Intr.LogicOr(x, y)), Nil)
              case "==" => (p.Expr.IntrOp(p.Intr.LogicEq(x, y)), Nil)
              case "!=" => (p.Expr.IntrOp(p.Intr.LogicNeq(x, y)), Nil)
            }
            (expr, (stmts, inv :: Nil))
          case (_ :+ op, x, y :: Nil) if x.tpe.isNumeric && y.tpe.isNumeric && rtn == p.Type.Bool1 =>
            val (intr, stmts) = op match {
              case "<"  => binaryPromoteIntr(x, y, binaryPromote(x.tpe, y.tpe), idx)(p.Intr.LogicLt(_, _))
              case "<=" => binaryPromoteIntr(x, y, binaryPromote(x.tpe, y.tpe), idx)(p.Intr.LogicLte(_, _))
              case ">"  => binaryPromoteIntr(x, y, binaryPromote(x.tpe, y.tpe), idx)(p.Intr.LogicGt(_, _))
              case ">=" => binaryPromoteIntr(x, y, binaryPromote(x.tpe, y.tpe), idx)(p.Intr.LogicGte(_, _))
              // rules for eq and neq is different from the general ref equality so we handle them here
              case "==" => binaryPromoteIntr(x, y, binaryPromote(x.tpe, y.tpe), idx)(p.Intr.LogicEq(_, _))
              case "!=" => binaryPromoteIntr(x, y, binaryPromote(x.tpe, y.tpe), idx)(p.Intr.LogicNeq(_, _))
            }
            (p.Expr.IntrOp(intr), (stmts, inv :: Nil))
          case (op :: Nil, x, y :: Nil) if x.tpe == y.tpe && rtn == p.Type.Bool1 =>
            val (expr, stmts) = op match {
              case "==" => (p.Expr.IntrOp(p.Intr.LogicEq(x, y)), Nil)
              case "!=" => (p.Expr.IntrOp(p.Intr.LogicNeq(x, y)), Nil)
            }
            (expr, (stmts, inv :: Nil))
          case ("scala" :: "Boolean" :: op :: Nil, x, Nil) =>
            val (expr, stmts) = op match {
              case "unary_!" if x.tpe == p.Type.Bool1 =>
                (p.Expr.IntrOp(p.Intr.LogicNot(recv)), Nil)
            }
            (expr, (stmts, inv :: Nil))
          case ("scala" :: ("Double" | "Float" | "Long" | "Int" | "Short" | "Char" | "Byte") :: op :: Nil, x, Nil)
              if x.tpe.isNumeric =>
            // xxx bool is integral
            val (expr, stmts) = op match {
              case "toDouble" => (p.Expr.Cast(recv, p.Type.Float64), Nil)
              case "toFloat"  => (p.Expr.Cast(recv, p.Type.Float32), Nil)
              case "toLong"   => (p.Expr.Cast(recv, p.Type.IntS64), Nil)
              case "toInt"    => (p.Expr.Cast(recv, p.Type.IntS32), Nil)
              case "toShort"  => (p.Expr.Cast(recv, p.Type.IntS16), Nil)
              case "toChar"   => (p.Expr.Cast(recv, p.Type.IntU16), Nil)
              case "toByte"   => (p.Expr.Cast(recv, p.Type.IntS8), Nil)

              case "toDegrees" =>
                (
                  p.Expr.IntrOp(p.Intr.Mul(x, p.Term.Float64Const(RadiansToDegrees), p.Type.Float64)),
                  Nil
                )
              case "toRadians" =>
                (
                  p.Expr.IntrOp(p.Intr.Mul(x, p.Term.Float64Const(DegreesToRadians), p.Type.Float64)),
                  Nil
                )

              // JLS 5.6.1. Unary Numeric Promotion
              case "unary_~" =>
                val (intr, stmts) = unaryPromoteIntr(x, rtn, idx)(p.Intr.BNot(_, rtn))
                (p.Expr.IntrOp(intr), stmts)
              case "unary_+" =>
                val (intr, stmts) = unaryPromoteIntr(x, rtn, idx)(p.Intr.Pos(_, rtn))
                (p.Expr.IntrOp(intr), stmts)
              case "unary_-" =>
                val (intr, stmts) = unaryPromoteIntr(x, rtn, idx)(p.Intr.Neg(_, rtn))
                (p.Expr.IntrOp(intr), stmts)

            }
            (expr, (Nil, inv :: Nil))
          case ("scala" :: ("Double" | "Float" | "Long" | "Int" | "Short" | "Char" | "Byte") :: op :: Nil, x, y :: Nil)
              if x.tpe.isNumeric && y.tpe.isNumeric && rtn.isNumeric =>
            val (expr, stmts) = op match {
              // JLS 5.6.2. Binary Numeric Promotion
              case "+" => binaryPromoteIntr(x, y, rtn, idx)(p.Intr.Add(_, _, rtn))
              case "-" => binaryPromoteIntr(x, y, rtn, idx)(p.Intr.Sub(_, _, rtn))
              case "*" => binaryPromoteIntr(x, y, rtn, idx)(p.Intr.Mul(_, _, rtn))
              case "/" => binaryPromoteIntr(x, y, rtn, idx)(p.Intr.Div(_, _, rtn))
              case "%" => binaryPromoteIntr(x, y, rtn, idx)(p.Intr.Rem(_, _, rtn))
              case "&" => binaryPromoteIntr(x, y, rtn, idx)(p.Intr.BAnd(_, _, rtn))
              case "|" => binaryPromoteIntr(x, y, rtn, idx)(p.Intr.BOr(_, _, rtn))
              case "^" => binaryPromoteIntr(x, y, rtn, idx)(p.Intr.BXor(_, _, rtn))

              // JLS 5.6.1. Unary Numeric Promotion
              case "<<"  => binaryPromoteIntr(x, y, rtn, idx)(p.Intr.BSL(_, _, rtn))
              case ">>"  => binaryPromoteIntr(x, y, rtn, idx)(p.Intr.BSR(_, _, rtn))
              case ">>>" => binaryPromoteIntr(x, y, rtn, idx)(p.Intr.BZSR(_, _, rtn))
            }
            (p.Expr.IntrOp(expr), (stmts, inv :: Nil))
          case (
                ("scala" :: "Array" :: "apply" :: Nil) |                              //
                ("polyregion" :: "scalalang" :: "Buffer" :: "apply" :: Nil) |         //
                ("scala" :: "collection" :: "SeqOps" :: "apply" :: Nil) |             //
                ("scala" :: "collection" :: "mutable" :: "SeqOps" :: "apply" :: Nil), //
                xs @ p.Term.Select(_, p.Named(_, p.Type.Ptr(_, _, _))),
                idx :: Nil
              ) if idx.tpe.kind == p.TypeKind.Integral =>
            (p.Expr.Index(xs, idx, rtn), (Nil, inv :: Nil))
          case (
                ("scala" :: "Array" :: "update" :: Nil) |                             //
                ("polyregion" :: "scalalang" :: "Buffer" :: "update" :: Nil) |        //
                ("scala" :: "collection" :: "mutable" :: "SeqOps" :: "update" :: Nil) //
                ,
                xs @ p.Term.Select(_, p.Named(_, p.Type.Ptr(_, _, _))),
                idx :: x :: Nil
              ) if idx.tpe.kind == p.TypeKind.Integral =>
            (p.Expr.Alias(p.Term.Unit0Const), (p.Stmt.Update(xs, idx, x) :: Nil, inv :: Nil))
          case (unknownSym, recv, args) =>
            println(s"No instance intrinsic for call: $recv.`${unknownSym.mkString(".")}`(${args
                .mkString(",")}), rtn=${rtn}, argn=${args.size}")
            (inv, (Nil, Nil))
        }
      case x => (x, (Nil, Nil))
    }
    val (stmts, ivks) = cs.combineAll
    (stmts :+ stmt, ivks)
  }

  private def intrinsifyModuleApply(s: p.Stmt, idx: Int) = {
    val (stmt, cs) = s.modifyCollect[p.Expr, (List[p.Stmt], List[p.Expr.Invoke])] {
      case inv @ p.Expr.Invoke(sym, tpeArgs, Some(_), args, captures, rtn) =>
        (sym.fqn, args) match {

          case ("scala" :: "Int$" :: "int2double" :: Nil, x :: Nil) if x.tpe == p.Type.IntS32 =>
            (p.Expr.Cast(x, p.Type.Float64), (Nil, inv :: Nil))

          case ("scala" :: "Short$" :: "short2int" :: Nil, x :: Nil) if x.tpe == p.Type.IntS16 =>
            (p.Expr.Cast(x, p.Type.IntS32), (Nil, inv :: Nil))

          case ("scala" :: "Char$" :: "char2int" :: Nil, x :: Nil) if x.tpe == p.Type.IntU16 =>
            (p.Expr.Cast(x, p.Type.IntS32), (Nil, inv :: Nil))

          case ("scala" :: "Byte$" :: "byte2int" :: Nil, x :: Nil) if x.tpe == p.Type.IntS8 =>
            (p.Expr.Cast(x, p.Type.IntS32), (Nil, inv :: Nil))

          case (Symbols.ArrayModule :+ "ofDim", x :: Nil) =>
            rtn match {
              case arr: p.Type.Ptr =>
                (p.Expr.Alloc(arr, x), (Nil, inv :: Nil))
              case _ => ???
            }
          case ((Symbols.ScalaMath | Symbols.JavaMath) :+ op, x :: y :: Nil) => // scala.math binary
            val expr = op match {
              case "pow" => p.Expr.MathOp(p.Math.Pow(x, y, rtn))

              case "min" => p.Expr.IntrOp(p.Intr.Min(x, y, rtn))
              case "max" => p.Expr.IntrOp(p.Intr.Max(x, y, rtn))

              case "atan2" => p.Expr.MathOp(p.Math.Atan2(x, y, rtn))
              case "hypot" => p.Expr.MathOp(p.Math.Hypot(x, y, rtn))
            }
            (expr, (Nil, inv :: Nil))
          case ((Symbols.ScalaMath | Symbols.JavaMath) :+ op, x :: Nil) => // scala.math unary
            val expr = op match {
              case "toDegrees" =>
                p.Expr.IntrOp(p.Intr.Mul(x, p.Term.Float64Const(RadiansToDegrees), p.Type.Float64))
              case "toRadians" =>
                p.Expr.IntrOp(p.Intr.Mul(x, p.Term.Float64Const(DegreesToRadians), p.Type.Float64))
              case "sin"  => p.Expr.MathOp(p.Math.Sin(x, rtn))
              case "cos"  => p.Expr.MathOp(p.Math.Cos(x, rtn))
              case "tan"  => p.Expr.MathOp(p.Math.Tan(x, rtn))
              case "asin" => p.Expr.MathOp(p.Math.Asin(x, rtn))
              case "acos" => p.Expr.MathOp(p.Math.Acos(x, rtn))
              case "atan" => p.Expr.MathOp(p.Math.Atan(x, rtn))
              case "sinh" => p.Expr.MathOp(p.Math.Sinh(x, rtn))
              case "cosh" => p.Expr.MathOp(p.Math.Cosh(x, rtn))
              case "tanh" => p.Expr.MathOp(p.Math.Tanh(x, rtn))

              case "signum" => p.Expr.MathOp(p.Math.Signum(x, rtn))
              case "abs"    => p.Expr.MathOp(p.Math.Abs(x, rtn))
              case "round"  => p.Expr.MathOp(p.Math.Round(x, rtn))
              case "ceil"   => p.Expr.MathOp(p.Math.Ceil(x, rtn))
              case "floor"  => p.Expr.MathOp(p.Math.Floor(x, rtn))
              case "rint"   => p.Expr.MathOp(p.Math.Rint(x, rtn))

              case "sqrt"  => p.Expr.MathOp(p.Math.Sqrt(x, rtn))
              case "cbrt"  => p.Expr.MathOp(p.Math.Cbrt(x, rtn))
              case "exp"   => p.Expr.MathOp(p.Math.Exp(x, rtn))
              case "expm1" => p.Expr.MathOp(p.Math.Expm1(x, rtn))
              case "log"   => p.Expr.MathOp(p.Math.Log(x, rtn))
              case "log1p" => p.Expr.MathOp(p.Math.Log1p(x, rtn))
              case "log10" => p.Expr.MathOp(p.Math.Log10(x, rtn))
            }
            (expr, (Nil, inv :: Nil))
          case (unknownSym, args) =>
            println(s"No module intrinsic for: ${unknownSym.mkString(".")}(${args.map(_.repr).mkString(",")}) ")
            (inv, (Nil, Nil))
        }
      case x => (x, (Nil, Nil))
    }
    val (stmts, ivks) = cs.combineAll
    (stmts :+ stmt, ivks)
  }

}
