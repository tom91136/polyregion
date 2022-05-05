Global / onChangedBuildSource := ReloadOnSourceChanges

val scala3Version = "3.1.3-RC2"

lazy val commonSettings = Seq(
  scalaVersion     := scala3Version,
  version          := "0.0.1-SNAPSHOT",
  organization     := "uk.ac.bristol.uob-hpc",
  organizationName := "University of Bristol",
  scalacOptions ~= filterConsoleScalacOptions,
  javacOptions ++=
    Seq(
      "-Xlint:all",
      "-XprintProcessorInfo",
      "-XprintRounds"
    ) ++
      Seq("-source", "1.8") ++
      Seq("-target", "1.8"),
  scalacOptions ~= { options: Seq[String] =>
    options.filterNot(
      Set("-explain-types", "-explain")
    )
  },
  scalacOptions ++= Seq(                     //
    "-no-indent",                            //
    "-Wconf:cat=unchecked:error",            //
    "-Wconf:name=MatchCaseUnreachable:error" //
    // "-Wconf:name=PatternMatchExhaustivity:error" // TODO enable later
    // "-language:strictEquality"
  ),
  scalafmtDetailedError := true,
  scalafmtFailOnErrors  := true
)

lazy val catsVersion  = "2.7.0"
lazy val munitVersion = "1.0.0-M3"

lazy val `loader-jvm` = project.settings(
  commonSettings,
  name             := "loader-jvm",
  autoScalaLibrary := false
)

lazy val bindingsDir      = file(".") / "native" / "bindings"
lazy val loaderShadeRules = Seq(ShadeRule.rename("polyregion.loader.**" -> "polyregion.shaded.loader.@1").inProject)

lazy val `runtime-java` = project
  .settings(
    commonSettings,
    name             := "runtime-java",
    javah / target   := bindingsDir / "java-runtime",
    autoScalaLibrary := false,
//    assemblyShadeRules  := loaderShadeRules,
    assembly / artifact := (assembly / artifact).value.withClassifier(Some("assembly"))
  )
  .dependsOn(`loader-jvm`)

lazy val `runtime-scala` = project
  .settings(
    commonSettings,
    name := "runtime-scala",
    libraryDependencies ++= Seq(
      "org.scalameta" %% "munit" % munitVersion % Test
    ),
    assemblyShadeRules  := loaderShadeRules,
    assembly / artifact := (assembly / artifact).value.withClassifier(Some("assembly"))
  )
  .dependsOn(`runtime-java`)

lazy val compiler = project
  .settings(
    commonSettings,
    name                := "compiler",
    javah / target      := bindingsDir / "java-compiler",
    assemblyShadeRules  := loaderShadeRules,
    assembly / artifact := (assembly / artifact).value.withClassifier(Some("assembly")),
    javacOptions ++= Seq("-proc:none"),
    scalacOptions ++=
      Seq("-Yretain-trees") ++     // XXX we need this so that the AST -> C++ conversion with partial ctors work
        Seq("-Xmax-inlines", "64") // the AST has lots of leaf nodes and we use inline so bump the limit
    ,
    libraryDependencies ++= Seq(
      "net.bytebuddy"  % "byte-buddy" % "1.12.10",
      "com.lihaoyi"   %% "pprint"     % "0.7.3",
      "com.lihaoyi"   %% "fansi"      % "0.3.1",
      "com.lihaoyi"   %% "upickle"    % "2.0.0",
      "org.typelevel" %% "cats-core"  % catsVersion
    ),
    (Compile / unmanagedJars) := {
      val xs       = (Compile / unmanagedJars).value
      val log      = streams.value.log
      val toolsJar = file(sys.props("java.home")).getParentFile / "lib" / "tools.jar"
      if (!toolsJar.exists()) xs
      else {
        log.info(s"Found tools.jar at $toolsJar")
        Attributed.blank(toolsJar) +: xs
      }
    }
  )
  .dependsOn(`runtime-scala`)

lazy val `compiler-testsuite-scala` = project
  .settings(
    commonSettings,
    name := "compiler-testsuite-scala",
    scalacOptions ++= Seq(
      "-Yretain-trees" // XXX for the test kernels
    ),
    libraryDependencies ++= Seq(
      "org.scalameta" %% "munit" % munitVersion % Test
    )
  )
  .dependsOn(compiler)

lazy val `compiler-testsuite-java` = project
  .settings(
    commonSettings,
    autoScalaLibrary := false,
    javacOptions ++= Seq("-XprintProcessorInfo", "-XprintRounds"),
    name := "compiler-testsuite-java",
    (Test / javaHome) := {
      // make sure we get the JDK dir and not the JRE
      val javaHome = file(sys.props("java.home"))
      if (!(javaHome.getParentFile / "lib" / "tools.jar").exists()) Some(javaHome)
      else Some(javaHome.getParentFile)
    },
    libraryDependencies ++= Seq(
      "com.github.sbt" % "junit-interface" % "0.13.3" % Test
    )
  )
  .dependsOn(compiler)

lazy val mainCls = Some("polyregion.examples.Stage")

lazy val `examples-scala` = project
  .settings(
    commonSettings,
    name                 := "examples-scala",
    fork                 := true,
    Compile / mainClass  := mainCls,
    assembly / mainClass := mainCls,
    scalacOptions ++= Seq("-Yretain-trees"),
    libraryDependencies ++= Seq(
      ("com.github.pathikrit"  %% "better-files"               % "3.9.1").cross(CrossVersion.for3Use2_13),
      "org.scala-lang.modules" %% "scala-parallel-collections" % "1.0.4"
    )
  )
  .dependsOn(compiler % Provided, `runtime-scala`)

lazy val `benchmarks-scala` = project
  .enablePlugins(JmhPlugin)
  .settings(
    commonSettings,
    name                 := "benchmarks-scala",
    fork                 := true,
    Compile / mainClass  := Some("polyregion.benchmarks.Main"),
    assembly / mainClass := Some("polyregion.benchmarks.Main"),
    scalacOptions ++= Seq("-Yretain-trees"),
    libraryDependencies ++= Seq(
      "net.openhft" % "affinity" % "3.20.0"
      // "org.scala-lang.modules" %% "scala-parallel-collections" % "1.0.4"
    )
  )
  .dependsOn(compiler % Provided, `runtime-scala`)

lazy val root = project
  .in(file("."))
  .settings(commonSettings)
  .aggregate(
    `loader-jvm`,
    `runtime-scala`,
    `runtime-java`,
    compiler,
    `compiler-testsuite-scala`,
    `compiler-testsuite-java`,
    `examples-scala`,
    `benchmarks-scala`
  )
