lazy val mainCls = Some("polyregion.Main")

lazy val root = (project in file(".")).settings(
  scalaVersion := "3.1.0",
  version := "0.0.1-SNAPSHOT",
  organization := "uk.ac.bristol.uob-hpc",
  organizationName := "University of Bristol",
  Compile / mainClass := mainCls,
  assembly / mainClass := mainCls,
  scalacOptions ~= filterConsoleScalacOptions,
  scalacOptions ++= Seq("-Yretain-trees"),
  // resolvers += Resolver.sonatypeRepo("snapshots"),
  name := "polyregion",
  libraryDependencies ++= Seq(
    "org.bytedeco"            % "llvm-platform"              % "12.0.1-1.5.6",
    "org.bytedeco"            % "libffi-platform"            % "3.4.2-1.5.6",
    "org.openjdk.jol"         % "jol-core"                   % "0.16",
    "net.openhft"             % "affinity"                   % "3.21ea82",
    "com.lihaoyi"            %% "pprint"                     % "0.6.6",
    "org.scala-lang"         %% "scala3-staging"             % scalaVersion.value,
    "org.scala-lang"         %% "scala3-tasty-inspector"     % scalaVersion.value,
    "io.github.iltotore"     %% "iron"                       % "1.1.2",
    "io.github.iltotore"     %% "iron-numeric"               % "1.1-1.0.1",
    "io.github.iltotore"     %% "iron-string"                % "1.1-0.1.0",
    "io.github.iltotore"     %% "iron-iterable"              % "1.1-0.1.0",
    "com.github.scopt"       %% "scopt"                      % "4.0.1",
    "org.scala-lang.modules" %% "scala-parallel-collections" % "1.0.4"
  )
)
