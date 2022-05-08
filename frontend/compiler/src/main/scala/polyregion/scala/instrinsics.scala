package polyregion.scala

object intrinsics {

  private def intrinsic: Nothing = throw new AssertionError("illegal")

  def apply[A](a: A, index: Int)           = intrinsic
  def update[A, B](a: A, index: Int, b: B) = intrinsic

  def array(size: Int) = intrinsic

  def pow[A](a: A, b: A): A   = intrinsic
  def min[A](a: A, b: A): A   = intrinsic
  def max[A](a: A, b: A): A   = intrinsic
  def atan2[A](a: A, b: A): A = intrinsic
  def hypot[A](a: A, b: A): A = intrinsic

  def sin[A](a: A): A      = intrinsic
  def cos[A](a: A): A      = intrinsic
  def tan[A](a: A): A      = intrinsic
  def asin[A](a: A): A     = intrinsic
  def acos[A](a: A): A     = intrinsic
  def atan[A](a: A): A     = intrinsic
  def sinh[A](a: A): A     = intrinsic
  def cosh[A](a: A): A     = intrinsic
  def tanh[A](a: A): A     = intrinsic
  def signum[A](a: A): A   = intrinsic
  def abs[A](a: A): A      = intrinsic
  def round[A, B](a: A): B = intrinsic
  def ceil[A](a: A): A     = intrinsic
  def floor[A](a: A): A    = intrinsic
  def rint[A](a: A): A     = intrinsic
  def sqrt[A](a: A): A     = intrinsic
  def cbrt[A](a: A): A     = intrinsic
  def exp[A](a: A): A      = intrinsic
  def expm1[A](a: A): A    = intrinsic
  def log[A](a: A): A      = intrinsic
  def log1p[A](a: A): A    = intrinsic
  def log10[A](a: A): A    = intrinsic

}
