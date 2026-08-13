ThisBuild / scalaVersion := "3.8.4"

lazy val genSpectra = taskKey[Unit]("Generate the C++, Fortran and Scala Spectra APIs.")

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

lazy val libraryCodegen = ProjectRef(file("../frontend").toURI, "library-codegen")

lazy val api = Project("spectraApi", file("test/scala"))
  .settings(
    commonSettings,
    name := "spectra-api",
    Compile / unmanagedSources ++= Seq(
      file("generated/scala/polyregion/spectra/SpectraApi.scala"),
      file("test/scala/polyregion/spectra/check_surface.scala")
    )
  )

lazy val spectra = Project("spectra", file("."))
  .settings(
    commonSettings,
    name                                   := "spectra",
    libraryDependencies += "org.scalameta" %% "munit" % "1.3.4" % Test,
    genSpectra := Def.uncached {
      Def.taskDyn {
        val root = baseDirectory.value.getAbsolutePath
        (Compile / runMain).toTask(s" polyregion.spectra.SpectraCodeGen $root")
      }.value
    }
  )
  .dependsOn(libraryCodegen)
  .aggregate(api)
