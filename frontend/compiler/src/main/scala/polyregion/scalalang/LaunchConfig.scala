package polyregion.scalalang

private[scalalang] object LaunchConfig {

  def targets: String =
    Option(System.getProperty("polyregion.targets"))
      .orElse(Option(System.getenv("POLYREGION_TARGETS")))
      .map(_.trim)
      .filter(_.nonEmpty)
      .getOrElse("host@native")

  def deviceTargets(value: String): List[(String, String)] =
    value
      .split(',')
      .iterator
      .map(_.trim)
      .filter(_.nonEmpty)
      .map { entry =>
        val at = entry.indexOf('@')
        if (at < 0) entry -> "native" else entry.take(at) -> entry.drop(at + 1)
      }
      .toList

  def opt: polyregion.jvm.compiler.Opt =
    Option(System.getProperty("polyregion.opt"))
      .orElse(Option(System.getenv("POLYREGION_OPT")))
      .map(_.trim)
      .filter(_.nonEmpty)
      .fold(polyregion.jvm.compiler.Opt.O2)(polyregion.jvm.compiler.Opt.valueOf)

  def jit: Boolean =
    Option(System.getProperty("polyregion.jit"))
      .orElse(Option(System.getenv("POLYREGION_JIT")))
      .exists(_.trim.toBooleanOption.contains(true))

  def targetNames: List[String] = targets.split(',').iterator.map(_.trim.takeWhile(_ != '@')).filter(_.nonEmpty).toList

  def capabilitiesFor(capabilities: Set[String], target: String): Set[String] =
    capabilities.collect {
      case capability if !capability.contains('@')                        => capability
      case capability if capability.dropWhile(_ != '@').drop(1) == target => capability.takeWhile(_ != '@')
    }
}
