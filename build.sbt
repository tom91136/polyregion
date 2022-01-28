lazy val commonSettings = Seq(
  scalaVersion     := "3.1.0",
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
  scalacOptions ++= Seq("-no-indent"),
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
    scalacOptions ++= Seq(
      "-Xmax-inlines", "64", // the AST has lots of leaf nodes and we use inline so bump the limit
      "-Yretain-trees" // XXX we need this so that the AST -> C++ conversion with partial ctors work
    ),
    libraryDependencies ++= Seq(
      "com.lihaoyi"   %% "pprint"    % "0.7.1",
      "com.lihaoyi"   %% "upickle"   % "1.4.4",
      "org.typelevel" %% "cats-core" % catsVersion,
      "org.scalameta" %% "munit"     % munitVersion % Test
    )
  )
  .dependsOn(`runtime-scala`)

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
      // ("com.github.jnr"         % "jffi"                       % "1.3.8").classifier("native"),
      // "com.github.jnr"          % "jffi"                       % "1.3.8",
      // "org.bytedeco"            % "llvm-platform"              % "12.0.1-1.5.6",
      // "org.bytedeco"            % "libffi-platform"            % "3.4.2-1.5.6",
      // "org.openjdk.jol"         % "jol-core"                   % "0.16",
      // "net.openhft"             % "affinity"                   % "3.20.0",
      // "org.typelevel"          %% "cats-core"                  % catsVersion,
      // "io.github.iltotore"     %% "iron"                       % "1.1.2",
      // "io.github.iltotore"     %% "iron-numeric"               % "1.1-1.0.1",
      // "io.github.iltotore"     %% "iron-string"                % "1.1-0.1.0",
      // "io.github.iltotore"     %% "iron-iterable"              % "1.1-0.1.0",
      // "org.scala-lang.modules" %% "scala-parallel-collections" % "1.0.4"
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
      "net.openhft"             % "affinity"                   % "3.20.0",
      // "org.scala-lang.modules" %% "scala-parallel-collections" % "1.0.4"
    )
  )
  .dependsOn(compiler % Provided, `runtime-scala`)

lazy val root = project
  .in(file("."))
  .settings(commonSettings)
  .aggregate(`loader-jvm`, `runtime-scala`, `runtime-java`, compiler, `examples-scala`, `benchmarks-scala`)
