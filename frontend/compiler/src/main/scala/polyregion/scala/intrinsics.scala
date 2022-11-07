package polyregion.scala

object intrinsics {

  private def intrinsic: Nothing = throw new AssertionError("illegal")

  trait Arr[A] {
//    def length: Int
    def apply(i: Int): A
    def update(i: Int, x: A): Unit
  }
  
  trait MutArr{
    
  }

  def array[A](size: Int): Arr[A]                   = intrinsic
  def apply[A](xs: Arr[A], index: Int): A           = intrinsic
  def update[A](xs: Arr[A], index: Int, x: A): Unit = intrinsic

  def gpuGlobalIdxX: Int  = intrinsic
  def gpuGlobalIdxY: Int  = intrinsic
  def gpuGlobalIdxZ: Int  = intrinsic
  def gpuGlobalSizeX: Int = intrinsic
  def gpuGlobalSizeY: Int = intrinsic
  def gpuGlobalSizeZ: Int = intrinsic
  def gpuGroupIdxX: Int   = intrinsic
  def gpuGroupIdxY: Int   = intrinsic
  def gpuGroupIdxZ: Int   = intrinsic
  def gpuGroupSizeX: Int  = intrinsic
  def gpuGroupSizeY: Int  = intrinsic
  def gpuGroupSizeZ: Int  = intrinsic
  def gpuLocalIdxX: Int   = intrinsic
  def gpuLocalIdxY: Int   = intrinsic
  def gpuLocalIdxZ: Int   = intrinsic
  def gpuLocalSizeX: Int  = intrinsic
  def gpuLocalSizeY: Int  = intrinsic
  def gpuLocalSizeZ: Int  = intrinsic

  def gpuGroupBarrier(): Unit = intrinsic
  def gpuGroupFence(): Unit   = intrinsic

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
