package ewgen

object Config {

  case class Source(
      url: String,
      stripComponents: Int = 0,
      rootSubdir: Option[String] = None
  )

  case class Group(
      name: String,
      source: Source,
      headers: Vector[String],
      includes: Vector[String],
      defines: Vector[String] = Vector.empty,
      symPrefixes: Vector[String],
      versioning: Option[VersioningStrategy] = None
  )

  enum VersioningStrategy { case opencl }

  case class Wrangler(name: String, groups: Vector[Group])
}
