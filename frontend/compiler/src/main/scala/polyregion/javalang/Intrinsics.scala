package polyregion.javalang

import org.jetbrains.java.decompiler.struct.gen.MethodDescriptor
import polyregion.ast.PolyAST as p

object Intrinsics {

  private val MathOwners = Set("java.lang.Math", "java.lang.StrictMath")

  private val Unary = Set(
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "sinh",
    "cosh",
    "tanh",
    "signum",
    "abs",
    "round",
    "ceil",
    "floor",
    "rint",
    "sqrt",
    "cbrt",
    "exp",
    "expm1",
    "log",
    "log1p",
    "log10"
  )

  private val Binary = Set("pow", "atan2", "hypot", "min", "max")

  // the one seam shared by the tree and bytecode tiers, a miss sends the callee on to the bytecode resolver
  def covers(callee: MethodRef): Boolean =
    MathOwners.contains(callee.owner) && (MethodDescriptor.parseDescriptor(callee.desc).params.length match {
      case 1 => Unary.contains(callee.name)
      case 2 => Binary.contains(callee.name)
      case _ => false
    })

  def lookup(callee: MethodRef, args: List[p.Term], rtn: p.Type): Option[p.Expr] =
    if (covers(callee)) math(callee.name, args, rtn) else None

  private def math(name: String, args: List[p.Term], rtn: p.Type): Option[p.Expr] = {
    def m(x: p.Math): Option[p.Expr]    = Some(p.Expr.MathOp(x))
    def intr(i: p.Intr): Option[p.Expr] = Some(p.Expr.IntrOp(i))
    (name, args) match {
      case ("sin", x :: Nil)        => m(p.Math.Sin(x, rtn))
      case ("cos", x :: Nil)        => m(p.Math.Cos(x, rtn))
      case ("tan", x :: Nil)        => m(p.Math.Tan(x, rtn))
      case ("asin", x :: Nil)       => m(p.Math.Asin(x, rtn))
      case ("acos", x :: Nil)       => m(p.Math.Acos(x, rtn))
      case ("atan", x :: Nil)       => m(p.Math.Atan(x, rtn))
      case ("sinh", x :: Nil)       => m(p.Math.Sinh(x, rtn))
      case ("cosh", x :: Nil)       => m(p.Math.Cosh(x, rtn))
      case ("tanh", x :: Nil)       => m(p.Math.Tanh(x, rtn))
      case ("signum", x :: Nil)     => m(p.Math.Signum(x, rtn))
      case ("abs", x :: Nil)        => m(p.Math.Abs(x, rtn))
      case ("round", x :: Nil)      => m(p.Math.Round(x, rtn))
      case ("ceil", x :: Nil)       => m(p.Math.Ceil(x, rtn))
      case ("floor", x :: Nil)      => m(p.Math.Floor(x, rtn))
      case ("rint", x :: Nil)       => m(p.Math.Rint(x, rtn))
      case ("sqrt", x :: Nil)       => m(p.Math.Sqrt(x, rtn))
      case ("cbrt", x :: Nil)       => m(p.Math.Cbrt(x, rtn))
      case ("exp", x :: Nil)        => m(p.Math.Exp(x, rtn))
      case ("expm1", x :: Nil)      => m(p.Math.Expm1(x, rtn))
      case ("log", x :: Nil)        => m(p.Math.Log(x, rtn))
      case ("log1p", x :: Nil)      => m(p.Math.Log1p(x, rtn))
      case ("log10", x :: Nil)      => m(p.Math.Log10(x, rtn))
      case ("pow", x :: y :: Nil)   => m(p.Math.Pow(x, y, rtn))
      case ("atan2", x :: y :: Nil) => m(p.Math.Atan2(x, y, rtn))
      case ("hypot", x :: y :: Nil) => m(p.Math.Hypot(x, y, rtn))
      case ("min", x :: y :: Nil)   => intr(p.Intr.Min(x, y, rtn))
      case ("max", x :: y :: Nil)   => intr(p.Intr.Max(x, y, rtn))
      case _                        => None
    }
  }
}
