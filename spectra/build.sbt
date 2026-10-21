ThisBuild / scalaVersion := "3.8.4"

lazy val genSpectra          = taskKey[Unit]("Generate the Spectra interfaces and C++ API tests.")
lazy val genSpectraInterface = taskKey[Unit]("Generate only the Spectra interface package.")

lazy val commonSettings = Seq(
  scalafmtDetailedError := true,
  scalafmtFailOnErrors  := true,
  scalacOptions ++= Seq(
    "-release:17",
    "-no-indent",
    "-Wconf:cat=unchecked:error",
    "-Wconf:name=MatchCaseUnreachable:error",
    "-Wconf:name=PatternMatchExhaustivity:error"
  ),
  publish / skip := true
)

lazy val polyregionMaven: File = {
  val dist = sys.props
    .get("polyregion.dist")
    .orElse(sys.env.get("POLYREGION_DIST"))
    .getOrElse(sys.error("Set POLYREGION_DIST (or -Dpolyregion.dist) to a Polyregion dist"))
  val repo = file(dist) / "share" / "polyregion" / "maven"
  if (!repo.isDirectory) sys.error(s"Polyregion dist has no JVM SDK at $repo")
  repo
}

lazy val polyregionSdk = settingKey[String]("Maven repository of the Polyregion JVM SDK.")

lazy val polyregionVersion: String =
  Option((polyregionMaven / "uk" / "ac" / "bristol" / "uob-hpc" / "interface-codegen_3").listFiles())
    .map(_.filter(_.isDirectory).map(_.getName).toList) match {
    case Some(version :: Nil) => version
    case other                => sys.error(s"Expected one interface-codegen version in $polyregionMaven, found $other")
  }

lazy val api = Project("spectraApi", file("test/scala"))
  .settings(
    commonSettings,
    name := "spectra-api",
    Compile / unmanagedSources ++= Seq(
      file("generated/scala/polyregion/spectra.scala"),
      file("test/scala/polyregion/InterfaceSurfaceCheck.scala"),
      file("test/scala/polyregion/check_surface.scala")
    )
  )

lazy val spectra = Project("spectra", file("."))
  .settings(
    commonSettings,
    name          := "spectra",
    polyregionSdk := polyregionMaven.getAbsolutePath,
    csrResolvers := {
      val sdk = MavenCache("polyregion-sdk", file(polyregionSdk.value))
      sdk +: csrResolvers.value.filterNot(_.name == "local")
    },
    updateCacheName := s"${updateCacheName.value}_${Hash.toHex(Hash(polyregionSdk.value)).take(8)}",
    libraryDependencies ++= Seq(
      "uk.ac.bristol.uob-hpc" %% "interface-codegen" % polyregionVersion,
      "org.scalameta"         %% "munit"             % "1.3.4" % Test
    ),
    genSpectra := Def.uncached {
      Def.taskDyn {
        val root = baseDirectory.value.getAbsolutePath
        (Compile / runMain).toTask(s" polyregion.spectra.SpectraCodeGen $root")
      }.value
    },
    genSpectraInterface := Def.uncached {
      Def.taskDyn {
        val root = baseDirectory.value.getAbsolutePath
        (Compile / runMain).toTask(s" polyregion.spectra.SpectraCodeGen $root --interface")
      }.value
    }
  )
  .aggregate(api)
