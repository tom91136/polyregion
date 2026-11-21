package polyregion.scalalang

import polyregion.jvm.compiler.Opt

import scala.quoted.Quotes

object LaunchConfig {

  case class DeviceTarget(name: String, arch: String)

  val TargetsKey: String      = "polyregion.targets"
  val OptKey: String          = "polyregion.opt"
  val JitKey: String          = "polyregion.jit"
  val RootsKey: String        = "polyregion.library.path"
  val CapabilitiesKey: String = "polyregion.library.capabilities"

  val Default: LaunchConfig = LaunchConfig(List(DeviceTarget("host", "native")), Opt.O2, jit = false, "", Nil)

  def parseTarget(entry: String): DeviceTarget = {
    val at = entry.indexOf('@')
    if (at < 0) DeviceTarget(entry, "native") else DeviceTarget(entry.take(at), entry.drop(at + 1))
  }

  // XXX CompilationInfo.XmacroSettings is @experimental and -experimental would taint every consumer
  def macroSettings(using q: Quotes): List[String] = {
    val info   = q.reflect.CompilationInfo
    val method = info.getClass.getMethods.find(_.getName == "XmacroSettings").get
    method.setAccessible(true)
    method.invoke(info) match {
      case settings: List[?] => settings.map(_.toString)
      case other             => throw IllegalStateException(s"unexpected macro settings $other")
    }
  }

  def fromSettings(settings: List[String]): Either[String, LaunchConfig] = {
    val entries = settings.collect {
      case setting if setting.startsWith("polyregion.") && setting.contains('=') =>
        setting.takeWhile(_ != '=') -> setting.dropWhile(_ != '=').drop(1).trim
    }
    def values(key: String): List[String] = entries.collect { case (`key`, value) if value.nonEmpty => value }
    for {
      opt <- values(OptKey).lastOption.fold(Right(Default.opt): Either[String, Opt])(name =>
        Opt.VALUES.find(_.name == name).toRight(s"unknown $OptKey `$name`")
      )
      jit <- values(JitKey).lastOption.fold(Right(Default.jit): Either[String, Boolean])(value =>
        value.toBooleanOption.toRight(s"$JitKey must be true or false, got `$value`")
      )
    } yield LaunchConfig(
      values(TargetsKey) match {
        case Nil     => Default.targets
        case configs => configs.map(parseTarget(_))
      },
      opt,
      jit,
      values(RootsKey).lastOption.getOrElse(""),
      values(CapabilitiesKey)
    )
  }
}

case class LaunchConfig(
    targets: List[LaunchConfig.DeviceTarget],
    opt: Opt,
    jit: Boolean,
    roots: String,
    capabilities: List[String]
) {
  def targetNames: List[String] = targets.map(_.name).distinct
  def targetList: String        = targets.map(target => s"${target.name}@${target.arch}").mkString(",")
  def capabilityList: String    = capabilities.mkString(",")
}
