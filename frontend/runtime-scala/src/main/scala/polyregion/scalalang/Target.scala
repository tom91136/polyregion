package polyregion.scalalang

import polyregion.jvm.{compiler as cp, runtime as rt}

import scala.compiletime.constValue

type Callback[A] = Either[Throwable, A] => Unit

enum Opt(val value: cp.Opt) {
  case O0    extends Opt(cp.Opt.O0)
  case O1    extends Opt(cp.Opt.O1)
  case O2    extends Opt(cp.Opt.O2)
  case O3    extends Opt(cp.Opt.O3)
  case Ofast extends Opt(cp.Opt.Ofast)
}
object Opt {
  type O0    = Opt.O0.type
  type O1    = Opt.O1.type
  type O2    = Opt.O2.type
  type O3    = Opt.O3.type
  type Ofast = Opt.Ofast.type
}

sealed trait Target {
  type Arch <: cp.Target & Singleton
  type UArch <: String & Singleton
  def arch: Arch
  def uarch: String
}

object Target {

  import cp.Target as cpt
  private transparent inline def valueOf[T <: Singleton: ValueOf] = scala.compiletime.summonInline[ValueOf[T]].value

  type OpenCL_C = OpenCL_C.type
  val OpenCL_C = new Target {
    type Arch  = cp.Target.C_OpenCL1_1.type;
    type UArch = "auto"
    val arch  = valueOf[Arch]
    val uarch = constValue[UArch]
  }

  sealed trait CPU extends Target
  val Host = new CPU {
    type Arch = cp.Target.LLVM_HOST.type; type UArch = "native"
    val arch = valueOf[Arch]; val uarch = constValue[UArch]
  }

  private type SString = String & Singleton

  case class X86[A <: SString](uarch: A) extends CPU {
    type UArch = A; type Arch = cpt.LLVM_X86_64.type; val arch = valueOf[Arch]
  }
  case class AArch64[A <: SString](uarch: A) extends CPU {
    type UArch = A; type Arch = cpt.LLVM_AARCH64.type; val arch = valueOf[Arch]
  }
  case class ARM[A <: SString](uarch: A) extends CPU {
    type UArch = A; type Arch = cpt.LLVM_ARM.type; val arch = valueOf[Arch]
  }
  case class NVPTX64[A <: SString](uarch: A) extends Target {
    type UArch = A; type Arch = cpt.LLVM_NVPTX64.type; val arch = valueOf[Arch]
  }
  case class AMDGCN[A <: SString](uarch: A) extends Target {
    type UArch = A; type Arch = cpt.LLVM_AMDGCN.type; val arch = valueOf[Arch]
  }
  case class SPIRV64[A <: SString](uarch: A) extends Target {
    type UArch = A; type Arch = cpt.LLVM_SPIRV64_KERNEL.type; val arch = valueOf[Arch]
  }

  object X86 {

    final val Znver2 = X86("znver2")
  }
  object AArch64 {

    final val AppleM1 = AArch64("apple-m1")
    final val A64fx   = AArch64("a64fx")
  }
  object ARM {}

  object NVPTX64 {

    final val SM52 = NVPTX64("sm_52")
    final val SM61 = NVPTX64("sm_61")
    final val SM80 = NVPTX64("sm_80")
  }
  object AMDGCN {

    final val gfx906 = AMDGCN("gfx906")
    final val gfx803 = AMDGCN("gfx803")
  }
  object SPIRV64 {}

}

case class Config[T <: Target, O <: Opt](targets: List[(T, O)]) {
  type Target = T
  type Opt    = O

  infix def ++(that: Config[T, O]) = Config(targets ++ that.targets)
}
object Config {
  def apply[T <: Target, O <: Opt](opt: O, target: T*): Config[T, O] = Config(target.map(_ -> opt).toList)
  def apply[T <: Target](target: T*): Config[T, Opt]                 = Config(Opt.O3, target*)
}

object Platforms {
  val platforms = rt.Platforms.create()
  sys.addShutdownHook(platforms.close())
  export platforms.*
}
