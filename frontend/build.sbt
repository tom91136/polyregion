// import org.typelevel.scalacoptions.ScalacOptions

import org.scalajs.linker.interface.{ESVersion, ModuleKind => SJSModuleKind, OutputPatterns}

Global / onChangedBuildSource := ReloadOnSourceChanges

lazy val nativeDir   = (file(".") / ".." / "native").getAbsoluteFile
lazy val bindingsDir = (nativeDir / "bindings" / "jvm").getAbsoluteFile

lazy val passBundleDest   = settingKey[File]("Destination of the JS pass bundle in the native tree.")
lazy val exportPassBundle = taskKey[File]("Build pass.js (fullLinkJS) and copy it into the native tree.")
lazy val genCodegen       = taskKey[Unit]("Run polyregion.ast.CodeGen to (re)generate native C++/JNI sources.")

// /home/tom/polyregion/native/cmake-build-debug-clang/bindings/jvm/libpolyregion-compiler-jvm.so

lazy val scala3Version = "3.8.3"
lazy val catsVersion   = "2.12.0"
lazy val munitVersion  = "1.0.2"

lazy val commonSettings = Seq(
  scalaVersion     := scala3Version,
  version          := "0.0.1-SNAPSHOT",
  organization     := "uk.ac.bristol.uob-hpc",
  organizationName := "University of Bristol",
  // compile / tpolecatExcludeOptions ++= ScalacOptions.defaultConsoleExclude,
  javacOptions ++=
    Seq(
      "-parameters",
      "-Xlint:all",
      "-XprintProcessorInfo",
      "-XprintRounds"
    ) ++
      Seq("-source", "1.8") ++
      Seq("-target", "1.8"),
//  scalacOptions ~= { options: Seq[String] =>
//    options.filterNot(
//      Set("-explain-types", "-explain")
//    )
//  },
  scalacOptions ++= Seq(                         //
    "-no-indent",                                //
    "-Wconf:cat=unchecked:error",                //
    "-Wconf:name=MatchCaseUnreachable:error",    //
    "-Wconf:name=PatternMatchExhaustivity:error" //
    // "-language:strictEquality"
  ),
  scalafmtDetailedError := true,
  scalafmtFailOnErrors  := true
)

lazy val nativeLibSettings =
  commonSettings ++ Seq(autoScalaLibrary := false, unmanagedResources / includeFilter := "*.so" || "*.dll" || "*.dylib")

lazy val `binding-jvm` = project.settings(
  commonSettings,
  name             := "binding-jvm",
  autoScalaLibrary := false
)

lazy val `runtime-java` = project
  .settings(
    commonSettings,
    name := "runtime-java",
    javacOptions ++= Seq("-proc:none"),
    autoScalaLibrary    := false,
    assembly / artifact := (assembly / artifact).value.withClassifier(Some("assembly"))
  )
  .dependsOn(`binding-jvm`, compiler % Compile)

lazy val `runtime-scala` = project
  .settings(
    commonSettings,
    name := "runtime-scala",
    javacOptions ++= Seq("-proc:none"),
    libraryDependencies ++= Seq(
      "org.scalameta" %% "munit" % munitVersion % Test
    ),
    assembly / artifact := (assembly / artifact).value.withClassifier(Some("assembly"))
  )
  .dependsOn(`binding-jvm`, compiler % Compile, codegen % Compile)

lazy val ast = crossProject(JVMPlatform, JSPlatform)
  .in(file("ast"))
  .settings(
    commonSettings,
    name := "ast",
    scalacOptions ++=
      Seq("-Yretain-trees") ++
        Seq("-Xmax-inlines", "80"),
    libraryDependencies ++= Seq(
      "com.lihaoyi"   %%% "upickle"   % "4.0.2",
      "org.typelevel" %%% "cats-core" % catsVersion
    )
  )

lazy val codegen = project
  .in(file("codegen"))
  .settings(
    commonSettings,
    name                := "codegen",
    assembly / artifact := (assembly / artifact).value.withClassifier(Some("assembly")),
    scalacOptions ++=
      Seq("-Yretain-trees") ++
        Seq("-Xmax-inlines", "80"),
    Compile / mainClass := Some("polyregion.ast.CodeGen"),
    libraryDependencies ++= Seq(
      "com.lihaoyi" %% "pprint" % "0.9.0"
    ),
    genCodegen := (Compile / runMain).toTask(" polyregion.ast.CodeGen").value
  )
  .dependsOn(ast.jvm, `binding-jvm`)

lazy val pass = crossProject(JVMPlatform, JSPlatform)
  .in(file("pass"))
  .settings(
    commonSettings,
    name := "pass",
    scalacOptions ++=
      Seq("-Yretain-trees") ++
        Seq("-Xmax-inlines", "80"),
    libraryDependencies ++= Seq(
      "org.scalameta" %%% "munit" % munitVersion % Test
    )
  )
  .jsSettings(
    scalaJSUseMainModuleInitializer := false,
    scalaJSLinkerConfig ~= { _.withModuleKind(SJSModuleKind.NoModule) },
    passBundleDest := nativeDir / "polyc" / "polypass.js",
    exportPassBundle := {
      // fastLinkJS does IR-level DCE (tree-shaking) without Closure's name mangling, leaving the
      // bundle readable for debugging from C++/QuickJS. fullLinkJS would minify everything.
      val _      = (Compile / fastLinkJS).value
      val srcDir = (Compile / fastLinkJS / scalaJSLinkerOutputDirectory).value
      val src    = srcDir / "main.js"
      val dst    = passBundleDest.value
      IO.copyFile(src, dst)
      streams.value.log.info(s"pass bundle: $src -> $dst (${dst.length() / 1024} KB)")
      dst
    }
  )
  .dependsOn(ast)

lazy val compiler = project
  .settings(
    commonSettings,
    name                := "compiler",
    assembly / artifact := (assembly / artifact).value.withClassifier(Some("assembly")),
    javacOptions ++= Seq("-proc:none"),
    scalacOptions ++=
      Seq("-Yretain-trees") ++
        Seq("-Xmax-inlines", "80"),
    libraryDependencies ++= Seq(
      "net.bytebuddy"  % "byte-buddy" % "1.15.10",
      "com.lihaoyi"   %% "pprint"     % "0.9.0",
      "org.scalameta" %% "munit"      % munitVersion % Test
    ),
    (Compile / unmanagedJars) := {
      val xs       = (Compile / unmanagedJars).value
      val log      = streams.value.log
      val toolsJar = file(sys.props("java.home")).getParentFile / "lib" / "tools.jar"
      // XXX This is only for Java 8's classpath, Java 9+ uses JPMS so it's OK to not find anything.
      if (!toolsJar.exists()) xs
      else {
        log.info(s"Found tools.jar at $toolsJar")
        Attributed.blank(toolsJar) +: xs
      }
    }
  )
  .dependsOn(ast.jvm, pass.jvm, `binding-jvm`)

lazy val `compiler-testsuite-scala` = project
  .settings(
    commonSettings,
    fork                      := true,
    Test / parallelExecution  := false,
    Test / testForkedParallel := false,
    javacOptions ++= Seq("-proc:none"),
    commands += Command.command("testUntilFailed") { state =>
      "test" :: "testUntilFailed" :: state
    },
    name := "compiler-testsuite-scala",
    scalacOptions ++= Seq(
      "-Yretain-trees" // XXX for the test kernels
    ),
    libraryDependencies ++= Seq(
      "org.scalameta" %% "munit" % munitVersion % Test
    )
  )
  .dependsOn(`runtime-scala`)

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
  .dependsOn(`runtime-java`)

lazy val mainCls = Some("polyregion.examples.CheckApi")

lazy val `examples-scala` = project
  .settings(
    commonSettings,
    name                 := "examples-scala",
    fork                 := true,
    Compile / mainClass  := mainCls,
    assembly / mainClass := mainCls,
    scalacOptions ++= Seq("-Yretain-trees"),
    libraryDependencies ++= Seq(
      "com.github.pathikrit"   %% "better-files"               % "3.9.2",
      "org.scala-lang.modules" %% "scala-parallel-collections" % "1.0.4"
    )
  )
  .dependsOn(`runtime-scala`)

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
      "net.openhft" % "affinity" % "3.23.3"
      // "org.scala-lang.modules" %% "scala-parallel-collections" % "1.0.4"
    )
  )
  .dependsOn(`runtime-scala`)

lazy val root = project
  .in(file("."))
  .settings(commonSettings)
  .aggregate(
    `binding-jvm`,
    ast.jvm,
    ast.js,
    codegen,
    pass.jvm,
    pass.js,
    `runtime-scala`,
    `runtime-java`,
    compiler,
    `compiler-testsuite-scala`,
    `compiler-testsuite-java`,
    `examples-scala`,
    `benchmarks-scala`
  )
