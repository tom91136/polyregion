val scala2Version = "2.13.6"
val scala3Version = "3.1.1"

lazy val commonSettings = Seq(
  scalaVersion     := scala3Version,
  version          := "0.0.1-SNAPSHOT",
  organization     := "uk.ac.bristol.uob-hpc",
  organizationName := "University of Bristol",
  scalacOptions ~= filterConsoleScalacOptions,
  scalacOptions ~= { options: Seq[String] =>
    options.filterNot(
      Set(
        "-explain-types",
        "-explain"
      )
    )
  },
  scalacOptions ++= Seq(                    //
    "-no-indent",                           //
    "-Wconf:cat=other-match-analysis:error" //
    // "-language:strictEquality"
  ),
  scalafmtDetailedError := true,
  scalafmtFailOnErrors  := true
)

lazy val catsVersion  = "2.7.0"
lazy val munitVersion = "1.0.0-M1"

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
    name                := "runtime-java",
    javah / target      := bindingsDir / "java-runtime",
    autoScalaLibrary    := false,
    assemblyShadeRules  := loaderShadeRules,
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
    javacOptions ++= Seq(
      "-Xlint:all",
      "--add-exports=jdk.compiler/com.sun.tools.javac.code=ALL-UNNAMED",
      "--add-exports=jdk.compiler/com.sun.tools.javac.comp=ALL-UNNAMED",
      "--add-exports=jdk.compiler/com.sun.tools.javac.file=ALL-UNNAMED",
      "--add-exports=jdk.compiler/com.sun.tools.javac.main=ALL-UNNAMED",
      "--add-exports=jdk.compiler/com.sun.tools.javac.model=ALL-UNNAMED",
      "--add-exports=jdk.compiler/com.sun.tools.javac.parser=ALL-UNNAMED",
      "--add-exports=jdk.compiler/com.sun.tools.javac.processing=ALL-UNNAMED",
      "--add-exports=jdk.compiler/com.sun.tools.javac.tree=ALL-UNNAMED",
      "--add-exports=jdk.compiler/com.sun.tools.javac.util=ALL-UNNAMED",
      "--add-exports=jdk.compiler/com.sun.tools.javac.api=ALL-UNNAMED"

    ),
    scalacOptions ++= Seq(
      "-Xmax-inlines",
      "64",            // the AST has lots of leaf nodes and we use inline so bump the limit
      "-Yretain-trees" // XXX we need this so that the AST -> C++ conversion with partial ctors work
    ),
    libraryDependencies ++= Seq(
      "com.lihaoyi"   %% "pprint"    % "0.7.1",
      "com.lihaoyi"   %% "upickle"   % "1.4.4",
      "org.typelevel" %% "cats-core" % catsVersion
    )
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
Global / onChangedBuildSource := ReloadOnSourceChanges
lazy val `compiler-testsuite-java` = project
  .settings(
    commonSettings,
    autoScalaLibrary := false,
    javacOptions ++= Seq("-XprintProcessorInfo", "-XprintRounds"),
    name              := "compiler-testsuite-java",
    (Test / javaHome) := Some(file(sys.props("java.home"))),
    libraryDependencies ++= Seq(
      "junit"          % "junit"           % "4.13.1" % Test,
      "com.github.sbt" % "junit-interface" % "0.13.2" % Test
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
