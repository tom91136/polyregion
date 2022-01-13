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
  scalacOptions ++= Seq("-no-indent")
)

lazy val catsVersion = "2.7.0"

lazy val `runtime-java` = project.settings(
  commonSettings,
  name           := "runtime-java",
  fork           := true,
  Compile / fork := true,
  javah / target := file(".") / "native" / "bindings" / "java-runtime",
  libraryDependencies ++= Seq(
  )
)

lazy val `runtime-scala` = project
  .settings(
    commonSettings,
    name           := "runtime-scala",
    fork           := true,
    Compile / fork := true,
    libraryDependencies ++= Seq(
      ("com.github.jnr" % "jffi"            % "1.3.8").classifier("native"),
      "com.github.jnr"  % "jffi"            % "1.3.8",
      "org.bytedeco"    % "llvm-platform"   % "12.0.1-1.5.6",
      "org.bytedeco"    % "libffi-platform" % "3.4.2-1.5.6",
      "org.openjdk.jol" % "jol-core"        % "0.16"
    )
  )
  .dependsOn(`runtime-java`)

lazy val compiler = project
  .settings(
    commonSettings,
    name := "compiler",
    scalacOptions ++= Seq("-Yretain-trees"),
    fork           := true,
    Compile / fork := true,
    javah / target := file(".") / "native" / "bindings" / "java-compiler",
    libraryDependencies ++= Seq(
      "org.typelevel" %% "cats-core" % catsVersion,
      "com.lihaoyi"   %% "pprint"    % "0.7.1",
      "com.lihaoyi"   %% "upickle"   % "1.4.3"
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
      "net.openhft"             % "affinity"                   % "3.20.0",
      "org.typelevel"          %% "cats-core"                  % catsVersion,
      "io.github.iltotore"     %% "iron"                       % "1.1.2",
      "io.github.iltotore"     %% "iron-numeric"               % "1.1-1.0.1",
      "io.github.iltotore"     %% "iron-string"                % "1.1-0.1.0",
      "io.github.iltotore"     %% "iron-iterable"              % "1.1-0.1.0",
      "org.scala-lang.modules" %% "scala-parallel-collections" % "1.0.4"
    )
  )
  .dependsOn(compiler, `runtime-scala`)

lazy val root = project
  .in(file("."))
  .settings(commonSettings)
  .aggregate(compiler, `runtime-scala`, `runtime-java`, `examples-scala`)
