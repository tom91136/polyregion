package polyregion.ast.pass

import cats.syntax.all.*
import polyregion.ast.{PolyAst as p, *, given}
import polyregion.scalalang.Symbols
import polyregion.ast.Traversal.*

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
    case p.Type.Byte | p.Type.Short | p.Type.Char => p.Type.Int
    case x                                        => x
  }

  // 5.6.2. Binary Numeric Promotion
  private def binaryPromote(t: p.Type, u: p.Type) = Set(t, u) match {
    case xs if xs contains p.Type.Double => p.Type.Double
    case xs if xs contains p.Type.Float  => p.Type.Float
    case xs if xs contains p.Type.Long   => p.Type.Long
    case _                               => p.Type.Int
  }

  private def castOrId(x: p.Term, to: p.Type, name: String): (p.Term, List[p.Stmt]) =
    if (x.tpe == to) {
      (x, Nil)
    } else {
      val named = p.Named(name, to)
      (p.Term.Select(Nil, named), p.Stmt.Var(named, Some(p.Expr.Cast(x, to))) :: Nil)
    }

  private def unaryNumericIntrinsic(x: p.Term, idx: Int, kind: p.UnaryIntrinsicKind) = {
    val (xVal, xStmts) = castOrId(x, unaryPromote(x.tpe), s"intr_${idx}")
    (p.Expr.UnaryIntrinsic(xVal, kind, xVal.tpe), xStmts)
  }

  def binaryNumericIntrinsic(x: p.Term, y: p.Term, upper: p.Type, idx: Int, kind: p.BinaryIntrinsicKind) = {
    val (xVal, xStmts) = castOrId(x, upper, s"intr_l${idx}")
    val (yVal, yStmts) = castOrId(y, upper, s"intr_r${idx}")
    val tpe = kind match {
      case p.BinaryIntrinsicKind.LogicEq | p.BinaryIntrinsicKind.LogicNeq | p.BinaryIntrinsicKind.LogicAnd |
          p.BinaryIntrinsicKind.LogicOr | p.BinaryIntrinsicKind.LogicLte | p.BinaryIntrinsicKind.LogicGte |
          p.BinaryIntrinsicKind.LogicLt | p.BinaryIntrinsicKind.LogicGt =>
        p.Type.Bool
      case _ => upper
    }
    (p.Expr.BinaryIntrinsic(xVal, yVal, kind, tpe), xStmts ++ yStmts)
  }

  private def intrinsifyNamed(
      op: String,
      args: List[p.Term],
      tpeArgs: List[p.Type],
      rtn: p.Type
  ): (p.Expr, List[p.Stmt]) = {

    def nullary(k: p.NullaryIntrinsicKind)        = p.Expr.NullaryIntrinsic(k, rtn)  -> List.empty[p.Stmt]
    def unary(x: p.Term, k: p.UnaryIntrinsicKind) = p.Expr.UnaryIntrinsic(x, k, rtn) -> List.empty[p.Stmt]
    def binary(x: p.Term, y: p.Term, k: p.BinaryIntrinsicKind) =
      p.Expr.BinaryIntrinsic(x, y, k, rtn) -> List.empty[p.Stmt]

    (op, args) match {
      case "gpuGlobalIdxX" -> Nil   => nullary(p.NullaryIntrinsicKind.GpuGlobalIdxX)
      case "gpuGlobalIdxY" -> Nil   => nullary(p.NullaryIntrinsicKind.GpuGlobalIdxY)
      case "gpuGlobalIdxZ" -> Nil   => nullary(p.NullaryIntrinsicKind.GpuGlobalIdxZ)
      case "gpuGlobalSizeX" -> Nil  => nullary(p.NullaryIntrinsicKind.GpuGlobalSizeX)
      case "gpuGlobalSizeY" -> Nil  => nullary(p.NullaryIntrinsicKind.GpuGlobalSizeY)
      case "gpuGlobalSizeZ" -> Nil  => nullary(p.NullaryIntrinsicKind.GpuGlobalSizeZ)
      case "gpuGroupIdxX" -> Nil    => nullary(p.NullaryIntrinsicKind.GpuGroupIdxX)
      case "gpuGroupIdxY" -> Nil    => nullary(p.NullaryIntrinsicKind.GpuGroupIdxY)
      case "gpuGroupIdxZ" -> Nil    => nullary(p.NullaryIntrinsicKind.GpuGroupIdxZ)
      case "gpuGroupSizeX" -> Nil   => nullary(p.NullaryIntrinsicKind.GpuGroupSizeX)
      case "gpuGroupSizeY" -> Nil   => nullary(p.NullaryIntrinsicKind.GpuGroupSizeY)
      case "gpuGroupSizeZ" -> Nil   => nullary(p.NullaryIntrinsicKind.GpuGroupSizeZ)
      case "gpuLocalIdxX" -> Nil    => nullary(p.NullaryIntrinsicKind.GpuLocalIdxX)
      case "gpuLocalIdxY" -> Nil    => nullary(p.NullaryIntrinsicKind.GpuLocalIdxY)
      case "gpuLocalIdxZ" -> Nil    => nullary(p.NullaryIntrinsicKind.GpuLocalIdxZ)
      case "gpuLocalSizeX" -> Nil   => nullary(p.NullaryIntrinsicKind.GpuLocalSizeX)
      case "gpuLocalSizeY" -> Nil   => nullary(p.NullaryIntrinsicKind.GpuLocalSizeY)
      case "gpuLocalSizeZ" -> Nil   => nullary(p.NullaryIntrinsicKind.GpuLocalSizeZ)
      case "gpuGroupBarrier" -> Nil => nullary(p.NullaryIntrinsicKind.GpuGroupBarrier)
      case "gpuGroupFence" -> Nil   => nullary(p.NullaryIntrinsicKind.GpuGroupFence)

      case "sin" -> (x :: Nil)  => unary(x, p.UnaryIntrinsicKind.Sin)
      case "cos" -> (x :: Nil)  => unary(x, p.UnaryIntrinsicKind.Cos)
      case "tan" -> (x :: Nil)  => unary(x, p.UnaryIntrinsicKind.Tan)
      case "asin" -> (x :: Nil) => unary(x, p.UnaryIntrinsicKind.Asin)
      case "acos" -> (x :: Nil) => unary(x, p.UnaryIntrinsicKind.Acos)
      case "atan" -> (x :: Nil) => unary(x, p.UnaryIntrinsicKind.Atan)
      case "sinh" -> (x :: Nil) => unary(x, p.UnaryIntrinsicKind.Sinh)
      case "cosh" -> (x :: Nil) => unary(x, p.UnaryIntrinsicKind.Cosh)
      case "tanh" -> (x :: Nil) => unary(x, p.UnaryIntrinsicKind.Tanh)

      case "signum" -> (x :: Nil) => unary(x, p.UnaryIntrinsicKind.Signum)
      case "abs" -> (x :: Nil)    => unary(x, p.UnaryIntrinsicKind.Abs)
      case "round" -> (x :: Nil)  => unary(x, p.UnaryIntrinsicKind.Round)
      case "ceil" -> (x :: Nil)   => unary(x, p.UnaryIntrinsicKind.Ceil)
      case "floor" -> (x :: Nil)  => unary(x, p.UnaryIntrinsicKind.Floor)
      case "rint" -> (x :: Nil)   => unary(x, p.UnaryIntrinsicKind.Rint)

      case "sqrt" -> (x :: Nil)  => unary(x, p.UnaryIntrinsicKind.Sqrt)
      case "cbrt" -> (x :: Nil)  => unary(x, p.UnaryIntrinsicKind.Cbrt)
      case "exp" -> (x :: Nil)   => unary(x, p.UnaryIntrinsicKind.Exp)
      case "expm1" -> (x :: Nil) => unary(x, p.UnaryIntrinsicKind.Expm1)
      case "log" -> (x :: Nil)   => unary(x, p.UnaryIntrinsicKind.Log)
      case "log1p" -> (x :: Nil) => unary(x, p.UnaryIntrinsicKind.Log1p)
      case "log10" -> (x :: Nil) => unary(x, p.UnaryIntrinsicKind.Log10)

      case "pow" -> (x :: y :: Nil)   => binary(x, y, p.BinaryIntrinsicKind.Pow)
      case "min" -> (x :: y :: Nil)   => binary(x, y, p.BinaryIntrinsicKind.Min)
      case "max" -> (x :: y :: Nil)   => binary(x, y, p.BinaryIntrinsicKind.Max)
      case "atan2" -> (x :: y :: Nil) => binary(x, y, p.BinaryIntrinsicKind.Atan2)
      case "hypot" -> (x :: y :: Nil) => binary(x, y, p.BinaryIntrinsicKind.Hypot)

      case "array" -> (x :: Nil) //
          if x.tpe == p.Type.Int =>
        p.Expr.Alloc(p.Type.Array(tpeArgs.head), x) -> Nil
      case "apply" -> ((s @ p.Term.Select(_, p.Named(_, p.Type.Array(`rtn`)))) :: i :: Nil) //
          if i.tpe == p.Type.Int =>
        p.Expr.Index(s, i, rtn) -> Nil
      case "update" -> ((s @ p.Term.Select(_, p.Named(_, p.Type.Array(c)))) :: i :: x :: Nil) //
          if i.tpe == p.Type.Int && x.tpe == c && rtn == p.Type.Unit =>
        p.Expr.Alias(p.Term.UnitConst) -> (p.Stmt.Update(s, i, x) :: Nil)
      case _ => ???
    }

  }

  private def intrinsifyInstanceApply(s: p.Stmt, idx: Int): (List[p.Stmt], List[p.Expr.Invoke]) = {
    val (stmt, cs) = s.modifyCollect[p.Expr, (List[p.Stmt], List[p.Expr.Invoke])] {
      case inv @ p.Expr.Invoke(sym, tpeArgs, Some(recv), args, captures, rtn) =>
        (sym.fqn, recv, args) match {
          case (
                "polyregion" :: "scala" :: "intrinsics$" :: op :: Nil,
                p.Term
                  .Select(
                    Nil,
                    p.Named(_, p.Type.Struct(p.Sym("polyregion" :: "scala" :: "intrinsics$" :: Nil), _, _, _))
                  ),
                xs
              ) =>
            println(s">>> ${recv} $op[${tpeArgs.map(_.repr)}](${xs.map(_.repr)}) : $rtn")

            val (expr, stmts) = intrinsifyNamed(op, args, tpeArgs, rtn)
            (expr, (stmts, inv :: Nil))

          case (
                "polyregion" :: "scala" :: "intrinsics$" :: "TypedBuffer" :: op :: Nil,
                xs @ p.Term
                  .Select(
                    _,
                    _
                  ),
                args
              ) =>
            (op, args) match {
              case "update" -> (i :: x :: Nil) if i.tpe.kind == p.TypeKind.Integral =>
                (p.Expr.Alias(p.Term.UnitConst), (p.Stmt.Update(xs, i, x) :: Nil, inv :: Nil))
              case "apply" -> (i :: Nil) if i.tpe.kind == p.TypeKind.Integral =>
                (p.Expr.Index(xs, i, rtn), (Nil, inv :: Nil))
              case (op, args) =>
                println(s"$op $$args")
                ???
            }
          case (_ :+ op, x, y :: Nil) if x.tpe == p.Type.Bool && y.tpe == p.Type.Bool && rtn == p.Type.Bool =>
            val (expr, stmts) = op match {
              case "&&" => (p.Expr.BinaryIntrinsic(x, y, p.BinaryIntrinsicKind.LogicAnd, p.Type.Bool), Nil)
              case "||" => (p.Expr.BinaryIntrinsic(x, y, p.BinaryIntrinsicKind.LogicOr, p.Type.Bool), Nil)
              case "==" => (p.Expr.BinaryIntrinsic(x, y, p.BinaryIntrinsicKind.LogicEq, p.Type.Bool), Nil)
              case "!=" => (p.Expr.BinaryIntrinsic(x, y, p.BinaryIntrinsicKind.LogicNeq, p.Type.Bool), Nil)
            }
            (expr, (stmts, inv :: Nil))
          case (_ :+ op, x, y :: Nil) if x.tpe.isNumeric && y.tpe.isNumeric && rtn == p.Type.Bool =>
            val (expr, stmts) = op match {
              case "<" =>
                binaryNumericIntrinsic(x, y, binaryPromote(x.tpe, y.tpe), idx, p.BinaryIntrinsicKind.LogicLt)
              case "<=" =>
                binaryNumericIntrinsic(x, y, binaryPromote(x.tpe, y.tpe), idx, p.BinaryIntrinsicKind.LogicLte)
              case ">" =>
                binaryNumericIntrinsic(x, y, binaryPromote(x.tpe, y.tpe), idx, p.BinaryIntrinsicKind.LogicGt)
              case ">=" =>
                binaryNumericIntrinsic(x, y, binaryPromote(x.tpe, y.tpe), idx, p.BinaryIntrinsicKind.LogicGte)
              // rules for eq and neq is different from the general ref equality so we handle them here
              case "==" => binaryNumericIntrinsic(x, y, binaryPromote(x.tpe, y.tpe), idx, p.BinaryIntrinsicKind.LogicEq)
              case "!=" =>
                binaryNumericIntrinsic(x, y, binaryPromote(x.tpe, y.tpe), idx, p.BinaryIntrinsicKind.LogicNeq)
            }
            (expr, (stmts, inv :: Nil))
          case (op :: Nil, x, y :: Nil) if x.tpe == y.tpe && rtn == p.Type.Bool =>
            val (expr, stmts) = op match {
              case "==" => (p.Expr.BinaryIntrinsic(x, y, p.BinaryIntrinsicKind.LogicEq, p.Type.Bool), Nil)
              case "!=" => (p.Expr.BinaryIntrinsic(x, y, p.BinaryIntrinsicKind.LogicNeq, p.Type.Bool), Nil)
            }
            (expr, (stmts, inv :: Nil))
          case ("scala" :: "Boolean" :: op :: Nil, x, Nil) =>
            val (expr, stmts) = op match {
              case "unary_!" if x.tpe == p.Type.Bool =>
                (p.Expr.UnaryIntrinsic(recv, p.UnaryIntrinsicKind.LogicNot, p.Type.Bool), Nil)
            }
            (expr, (stmts, inv :: Nil))
          case ("scala" :: ("Double" | "Float" | "Long" | "Int" | "Short" | "Char" | "Byte") :: op :: Nil, x, Nil)
              if x.tpe.isNumeric =>
            // xxx bool is integral
            val (expr, stmts) = op match {
              case "toDouble" => (p.Expr.Cast(recv, p.Type.Double), Nil)
              case "toFloat"  => (p.Expr.Cast(recv, p.Type.Float), Nil)
              case "toLong"   => (p.Expr.Cast(recv, p.Type.Long), Nil)
              case "toInt"    => (p.Expr.Cast(recv, p.Type.Int), Nil)
              case "toShort"  => (p.Expr.Cast(recv, p.Type.Short), Nil)
              case "toChar"   => (p.Expr.Cast(recv, p.Type.Char), Nil)
              case "toByte"   => (p.Expr.Cast(recv, p.Type.Byte), Nil)

              case "toDegrees" =>
                (
                  p.Expr.BinaryIntrinsic(
                    x,
                    p.Term.DoubleConst(RadiansToDegrees),
                    p.BinaryIntrinsicKind.Mul,
                    p.Type.Double
                  ),
                  Nil
                )
              case "toRadians" =>
                (
                  p.Expr.BinaryIntrinsic(
                    x,
                    p.Term.DoubleConst(DegreesToRadians),
                    p.BinaryIntrinsicKind.Mul,
                    p.Type.Double
                  ),
                  Nil
                )

              // JLS 5.6.1. Unary Numeric Promotion
              case "unary_~" => unaryNumericIntrinsic(x, idx, p.UnaryIntrinsicKind.BNot)
              case "unary_+" => unaryNumericIntrinsic(x, idx, p.UnaryIntrinsicKind.Pos)
              case "unary_-" => unaryNumericIntrinsic(x, idx, p.UnaryIntrinsicKind.Neg)

            }
            (expr, (Nil, inv :: Nil))
          case ("scala" :: ("Double" | "Float" | "Long" | "Int" | "Short" | "Char" | "Byte") :: op :: Nil, x, y :: Nil)
              if x.tpe.isNumeric && y.tpe.isNumeric && rtn.isNumeric =>
            val (expr, stmts) = op match {
              // JLS 5.6.2. Binary Numeric Promotion
              case "+" => binaryNumericIntrinsic(x, y, rtn, idx, p.BinaryIntrinsicKind.Add)
              case "-" => binaryNumericIntrinsic(x, y, rtn, idx, p.BinaryIntrinsicKind.Sub)
              case "*" => binaryNumericIntrinsic(x, y, rtn, idx, p.BinaryIntrinsicKind.Mul)
              case "/" => binaryNumericIntrinsic(x, y, rtn, idx, p.BinaryIntrinsicKind.Div)
              case "%" => binaryNumericIntrinsic(x, y, rtn, idx, p.BinaryIntrinsicKind.Rem)
              case "&" => binaryNumericIntrinsic(x, y, rtn, idx, p.BinaryIntrinsicKind.BAnd)
              case "|" => binaryNumericIntrinsic(x, y, rtn, idx, p.BinaryIntrinsicKind.BOr)
              case "^" => binaryNumericIntrinsic(x, y, rtn, idx, p.BinaryIntrinsicKind.BXor)

              // JLS 5.6.1. Unary Numeric Promotion
              case "<<"  => binaryNumericIntrinsic(x, y, rtn, idx, p.BinaryIntrinsicKind.BSL)
              case ">>"  => binaryNumericIntrinsic(x, y, rtn, idx, p.BinaryIntrinsicKind.BSR)
              case ">>>" => binaryNumericIntrinsic(x, y, rtn, idx, p.BinaryIntrinsicKind.BZSR)
            }
            (expr, (stmts, inv :: Nil))
          case (
                ("scala" :: "Array" :: "apply" :: Nil) |                              //
                ("polyregion" :: "scala" :: "Buffer" :: "apply" :: Nil) |             //
                ("scala" :: "collection" :: "SeqOps" :: "apply" :: Nil) |             //
                ("scala" :: "collection" :: "mutable" :: "SeqOps" :: "apply" :: Nil), //
                xs @ p.Term.Select(_, p.Named(_, p.Type.Array(_))),
                idx :: Nil
              ) if idx.tpe.kind == p.TypeKind.Integral =>
            (p.Expr.Index(xs, idx, rtn), (Nil, inv :: Nil))
          case (
                ("scala" :: "Array" :: "update" :: Nil) |                             //
                ("polyregion" :: "scala" :: "Buffer" :: "update" :: Nil) |            //
                ("scala" :: "collection" :: "mutable" :: "SeqOps" :: "update" :: Nil) //
                ,
                xs @ p.Term.Select(_, p.Named(_, p.Type.Array(_))),
                idx :: x :: Nil
              ) if idx.tpe.kind == p.TypeKind.Integral =>
            (p.Expr.Alias(p.Term.UnitConst), (p.Stmt.Update(xs, idx, x) :: Nil, inv :: Nil))
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

          case ("scala" :: "Int$" :: "int2double" :: Nil, x :: Nil) if x.tpe == p.Type.Int =>
            (p.Expr.Cast(x, p.Type.Double), (Nil, inv :: Nil))

          case ("scala" :: "Short$" :: "short2int" :: Nil, x :: Nil) if x.tpe == p.Type.Short =>
            (p.Expr.Cast(x, p.Type.Int), (Nil, inv :: Nil))

          case ("scala" :: "Char$" :: "char2int" :: Nil, x :: Nil) if x.tpe == p.Type.Char =>
            (p.Expr.Cast(x, p.Type.Int), (Nil, inv :: Nil))

          case ("scala" :: "Byte$" :: "byte2int" :: Nil, x :: Nil) if x.tpe == p.Type.Byte =>
            (p.Expr.Cast(x, p.Type.Int), (Nil, inv :: Nil))

          case (Symbols.ArrayModule :+ "ofDim", x :: Nil) =>
            rtn match {
              case arr: p.Type.Array =>
                (p.Expr.Alloc(arr, x), (Nil, inv :: Nil))
              case _ => ???
            }
          case ((Symbols.ScalaMath | Symbols.JavaMath) :+ op, x :: y :: Nil) => // scala.math binary
            val kind = op match {
              case "pow" => p.BinaryIntrinsicKind.Pow

              case "min" => p.BinaryIntrinsicKind.Min
              case "max" => p.BinaryIntrinsicKind.Max

              case "atan2" => p.BinaryIntrinsicKind.Atan2
              case "hypot" => p.BinaryIntrinsicKind.Hypot
            }
            (p.Expr.BinaryIntrinsic(x, y, kind, rtn), (Nil, inv :: Nil))
          case ((Symbols.ScalaMath | Symbols.JavaMath) :+ op, x :: Nil) => // scala.math unary
            val expr = op match {
              case "toDegrees" =>
                p.Expr.BinaryIntrinsic(
                  x,
                  p.Term.DoubleConst(RadiansToDegrees),
                  p.BinaryIntrinsicKind.Mul,
                  p.Type.Double
                )
              case "toRadians" =>
                p.Expr.BinaryIntrinsic(
                  x,
                  p.Term.DoubleConst(DegreesToRadians),
                  p.BinaryIntrinsicKind.Mul,
                  p.Type.Double
                )
              case "sin"  => p.Expr.UnaryIntrinsic(x, p.UnaryIntrinsicKind.Sin, rtn)
              case "cos"  => p.Expr.UnaryIntrinsic(x, p.UnaryIntrinsicKind.Cos, rtn)
              case "tan"  => p.Expr.UnaryIntrinsic(x, p.UnaryIntrinsicKind.Tan, rtn)
              case "asin" => p.Expr.UnaryIntrinsic(x, p.UnaryIntrinsicKind.Asin, rtn)
              case "acos" => p.Expr.UnaryIntrinsic(x, p.UnaryIntrinsicKind.Acos, rtn)
              case "atan" => p.Expr.UnaryIntrinsic(x, p.UnaryIntrinsicKind.Atan, rtn)
              case "sinh" => p.Expr.UnaryIntrinsic(x, p.UnaryIntrinsicKind.Sinh, rtn)
              case "cosh" => p.Expr.UnaryIntrinsic(x, p.UnaryIntrinsicKind.Cosh, rtn)
              case "tanh" => p.Expr.UnaryIntrinsic(x, p.UnaryIntrinsicKind.Tanh, rtn)

              case "signum" => p.Expr.UnaryIntrinsic(x, p.UnaryIntrinsicKind.Signum, rtn)
              case "abs"    => p.Expr.UnaryIntrinsic(x, p.UnaryIntrinsicKind.Abs, rtn)
              case "round"  => p.Expr.UnaryIntrinsic(x, p.UnaryIntrinsicKind.Round, rtn)
              case "ceil"   => p.Expr.UnaryIntrinsic(x, p.UnaryIntrinsicKind.Ceil, rtn)
              case "floor"  => p.Expr.UnaryIntrinsic(x, p.UnaryIntrinsicKind.Floor, rtn)
              case "rint"   => p.Expr.UnaryIntrinsic(x, p.UnaryIntrinsicKind.Rint, rtn)

              case "sqrt"  => p.Expr.UnaryIntrinsic(x, p.UnaryIntrinsicKind.Sqrt, rtn)
              case "cbrt"  => p.Expr.UnaryIntrinsic(x, p.UnaryIntrinsicKind.Cbrt, rtn)
              case "exp"   => p.Expr.UnaryIntrinsic(x, p.UnaryIntrinsicKind.Exp, rtn)
              case "expm1" => p.Expr.UnaryIntrinsic(x, p.UnaryIntrinsicKind.Expm1, rtn)
              case "log"   => p.Expr.UnaryIntrinsic(x, p.UnaryIntrinsicKind.Log, rtn)
              case "log1p" => p.Expr.UnaryIntrinsic(x, p.UnaryIntrinsicKind.Log1p, rtn)
              case "log10" => p.Expr.UnaryIntrinsic(x, p.UnaryIntrinsicKind.Log10, rtn)
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