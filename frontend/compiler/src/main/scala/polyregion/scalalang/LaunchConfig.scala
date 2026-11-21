package polyregion.scalalang

import polyregion.jvm.compiler.Opt

import scala.quoted.Quotes

final case class LaunchConfig(
    targets: List[(String, String)],
    opt: Opt,
    jit: Boolean,
    roots: String,
    capabilities: List[String]
) {
  def targetNames: List[String] = targets.map(_._1).distinct
  def targetList: String        = targets.map((name, arch) => s"$name@$arch").mkString(",")
  def capabilityList: String    = capabilities.mkString(",")
}

object LaunchConfig {

  val TargetsKey      = "polyregion.targets"
  val OptKey          = "polyregion.opt"
  val JitKey          = "polyregion.jit"
  val RootsKey        = "polyregion.library.path"
  val CapabilitiesKey = "polyregion.library.capabilities"

  val Default: LaunchConfig = LaunchConfig(List("host" -> "native"), Opt.O2, jit = false, "", Nil)

  def parseTarget(entry: String): (String, String) = {
    val at = entry.indexOf('@')
    if (at < 0) entry -> "native" else entry.take(at) -> entry.drop(at + 1)
  }

  // XXX CompilationInfo.XmacroSettings is @experimental and -experimental would taint every consumer
  def macroSettings(using q: Quotes): List[String] = {
    val info   = q.reflect.CompilationInfo
    val method = info.getClass.getMethods.find(_.getName == "XmacroSettings").get
    method.setAccessible(true)
    method.invoke(info).asInstanceOf[List[String]]
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
        case configs => configs.map(parseTarget)
      },
      opt,
      jit,
      values(RootsKey).lastOption.getOrElse(""),
      values(CapabilitiesKey)
    )
  }
}
