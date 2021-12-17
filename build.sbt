lazy val commonSettings = Seq(
  scalaVersion     := "3.1.0",
  version          := "0.0.1-SNAPSHOT",
  organization     := "uk.ac.bristol.uob-hpc",
  organizationName := "University of Bristol",
  scalacOptions ~= filterConsoleScalacOptions
)

lazy val catsVersion = "2.7.0"

lazy val runtime = project.settings(
  commonSettings,
  name := "runtime",
  libraryDependencies ++= Seq(
    "org.bytedeco"    % "llvm-platform"   % "12.0.1-1.5.6",
    "org.bytedeco"    % "libffi-platform" % "3.4.2-1.5.6",
    "org.openjdk.jol" % "jol-core"        % "0.16"
  )
)

lazy val compiler = project
  .settings(
    commonSettings,
    name := "compiler",
    Compile / PB.targets := Seq(
      scalapb.gen() -> (Compile / sourceManaged).value / "scalapb"
    ),
    libraryDependencies ++= Seq(
      "com.thesamet.scalapb" %% "scalapb-runtime" % scalapb.compiler.Version.scalapbVersion % "protobuf",
      "org.typelevel"        %% "cats-core"       % catsVersion,
      "com.lihaoyi"          %% "pprint"          % "0.7.1"
    )
  )
  .dependsOn(runtime)

lazy val mainCls = Some("polyregion.examples.Stage")

lazy val examples = project
  .settings(
    commonSettings,
    name                 := "examples",
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
  .dependsOn(compiler, runtime)

lazy val root = project
  .in(file("."))
  .settings(commonSettings)
  .aggregate(runtime, compiler, examples)
