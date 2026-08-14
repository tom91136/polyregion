// import org.typelevel.scalacoptions.ScalacOptions

import org.scalajs.linker.interface.{ESVersion, ModuleKind => SJSModuleKind, OutputPatterns}

Global / concurrentRestrictions := Seq(
  Tags.limit(Tags.CPU, java.lang.Runtime.getRuntime.availableProcessors),
  Tags.limit(Tags.Network, java.lang.Runtime.getRuntime.availableProcessors),
  Tags.limitAll(java.lang.Runtime.getRuntime.availableProcessors * 2)
)

ThisBuild / javafmtFormatterCompatibleJavaVersion := 17

Global / excludeLintKeys += assembly / artifact

lazy val nativeDir   = (file(".") / ".." / "native").getAbsoluteFile
lazy val bindingsDir = (nativeDir / "bindings" / "jvm").getAbsoluteFile

lazy val passJsDest  = settingKey[File]("Destination of the JS PolyPass source in the native tree.")
lazy val passDsoDest = settingKey[File]("Destination of the SN PolyPass DSO in the native tree.")

lazy val exportPassBundle = taskKey[File]("Build pass.js (fullLinkJS) and copy it into the native tree.")
lazy val exportPassDso    = taskKey[File]("Build the SN pass DSO (nativeLink) and copy it into the native tree.")
lazy val genCodegen       = taskKey[Unit]("Run polyregion.ast.CodeGen to (re)generate native C++/JNI sources.")
lazy val genEw            = taskKey[Unit]("Run ewgen.Main to (re)generate the polyinvoke wrangler sources.")

def findBdwgcPrefix: File = {
  val nativeDir = (file(".") / ".." / "native").getAbsoluteFile
  Option(nativeDir.listFiles())
    .map(_.toSeq)
    .getOrElse(Nil)
    .filter(_.isDirectory)
    .flatMap(b => Option((b / "vcpkg_installed").listFiles()).map(_.toSeq).getOrElse(Nil))
    .filter(d => d.isDirectory && !d.getName.startsWith("vcpkg"))
    .find(d => (d / "include" / "gc.h").exists || (d / "include" / "gc" / "gc.h").exists)
    .getOrElse(
      sys.error(
        "bdw-gc not found in any native/*/vcpkg_installed/<triplet>/. " +
          "Configure the native build first: cmake -S native -B native/build-<config>."
      )
    )
}

lazy val scala3Version = "3.8.4"
lazy val catsVersion   = "2.13.0"
lazy val munitVersion  = "1.3.4"

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
    ) ++ Seq("--release", "17"),
//  scalacOptions ~= { options: Seq[String] =>
//    options.filterNot(
//      Set("-explain-types", "-explain")
//    )
//  },
  scalacOptions ++= Seq(                         //
    "-release:17",                               //
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

def nativeCompilerInputFiles: Seq[File] =
  val nativeLibraries = sys.env
    .get("POLYREGION_NATIVE_LIB_DIR")
    .toSeq
    .flatMap(_.split(java.util.regex.Pattern.quote(java.io.File.pathSeparator)))
    .map(_.trim)
    .filter(_.nonEmpty)
    .flatMap(dir => Option(file(dir).listFiles()).toSeq.flatten)
    .filter(file => file.isFile && file.getName.contains("polyc-JNI"))
  val passPlugins = sys.env
    .get("POLYPASS_PLUGINS")
    .toSeq
    .flatMap(_.split(java.util.regex.Pattern.quote(java.io.File.pathSeparator)))
    .map(path => file(path.trim))
    .filter(file => file.isFile)
  (nativeLibraries ++ passPlugins).distinct

def nativeCompilerInputs(converter: xsbti.FileConverter): Seq[Attributed[xsbti.HashedVirtualFileRef]] =
  nativeCompilerInputFiles.map(file => Attributed.blank(converter.toVirtualFile(file.toPath)))

lazy val nativeCompilerInputsTask =
  taskKey[Seq[Attributed[xsbti.HashedVirtualFileRef]]]("Native compiler libraries that participate in compile caching")

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
  .dependsOn(`binding-jvm`)

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

lazy val ast = (projectMatrix in file("ast"))
  .settings(
    commonSettings,
    name                      := "ast",
    Compile / sourceDirectory := (ThisBuild / baseDirectory).value / "ast" / "src" / "main",
    scalacOptions ++=
      Seq("-Yretain-trees") ++
        Seq("-Xmax-inlines", "80"),
    libraryDependencies ++= Seq(
      "org.typelevel" %% "cats-core" % catsVersion
    )
  )
  .jvmPlatform(scalaVersions = Seq(scala3Version))
  .jsPlatform(scalaVersions = Seq(scala3Version))
  .nativePlatform(scalaVersions = Seq(scala3Version))

lazy val ewgen = project
  .in(file("ewgen"))
  .settings(
    commonSettings,
    name                := "ewgen",
    fork                := true,
    Compile / mainClass := Some("ewgen.Main"),
    libraryDependencies ++= Seq(
      "com.lihaoyi"                   %% "upickle"          % "4.4.3",
      "com.lihaoyi"                   %% "mainargs"         % "0.7.8",
      "com.softwaremill.sttp.client4" %% "core"             % "4.0.26",
      "org.apache.commons"             % "commons-compress" % "1.28.0",
      "org.scalameta"                 %% "munit"            % munitVersion % Test
    ),
    genEw := Def.uncached {
      Def.taskDyn {
        val outRoot = (nativeDir / "polyinvoke" / "thirdparty").getAbsoluteFile
        val cache   = target.value / "ewgen-cache"
        val work    = target.value / "ewgen-work"
        (Compile / runMain).toTask(s" ewgen.Main --out $outRoot --cache $cache --work $work")
      }.value
    }
  )

lazy val `interface-codegen` = project
  .in(file("interface-codegen"))
  .settings(
    commonSettings,
    name                                   := "interface-codegen",
    libraryDependencies += "org.scalameta" %% "munit" % munitVersion % Test
  )
  .dependsOn(ast.jvm(scala3Version))

lazy val codegen = project
  .in(file("codegen"))
  .settings(
    commonSettings,
    name                := "codegen",
    assembly / artifact := (assembly / artifact).value.withClassifier(Some("assembly")),
    scalacOptions ++=
      Seq("-Yretain-trees") ++
        Seq("-Xmax-inlines", "80"),
    Compile / mainClass                  := Some("polyregion.ast.CodeGen"),
    libraryDependencies += "com.lihaoyi" %% "pprint" % "0.9.6",
    genCodegen := Def.uncached {
      (Compile / runMain).toTask(" polyregion.ast.CodeGen").value
    }
  )
  .dependsOn(ast.jvm(scala3Version), `binding-jvm`, `interface-codegen`)

lazy val pass = (projectMatrix in file("pass"))
  .settings(
    commonSettings,
    name := "pass",
    scalacOptions ++=
      Seq("-Yretain-trees") ++
        Seq("-Xmax-inlines", "80"),
    libraryDependencies ++= Seq(
      "org.scalameta" %% "munit" % munitVersion % Test
    )
  )
  .jvmPlatform(scalaVersions = Seq(scala3Version))
  .jsPlatform(
    scalaVersions = Seq(scala3Version),
    settings = Seq(
      scalaJSUseMainModuleInitializer := false,
      scalaJSLinkerConfig ~= { _.withModuleKind(SJSModuleKind.CommonJSModule) },
      passJsDest := nativeDir / "polyc" / "polypass.js",
      exportPassBundle := Def.uncached {
        import com.google.javascript.jscomp.{
          AbstractCommandLineRunner,
          CheckLevel,
          CompilationLevel,
          CompilerOptions,
          DiagnosticGroups,
          JSError,
          SortingErrorManager,
          WarningLevel,
          Compiler => ClosureCompiler,
          SourceFile => ClosureSourceFile
        }

        val _      = (Compile / fastLinkJS).value
        val srcDir = (Compile / fastLinkJS / scalaJSLinkerOutputDirectory).value
        val src    = srcDir / "main.js"
        val dst    = passJsDest.value
        val log    = streams.value.log

        // XXX Scala.js encodes `_` in JVM names as U+FF3F to avoid collisions with its own `_` usage;
        // Closure's parser then NPEs on the escape (`Cannot read field "features"`).
        val raw = IO.read(src).replace("\\uff3f", "_FF3F_")

        val defaultExterns = AbstractCommandLineRunner.getBuiltinExterns(CompilerOptions.Environment.BROWSER)
        val inputs         = java.util.Collections.singletonList(ClosureSourceFile.fromCode(src.getName, raw))

        def quietErrorManager(onReport: (CheckLevel, JSError) => Unit) = {
          val gen: SortingErrorManager.ErrorReportGenerator = (manager: SortingErrorManager) => {
            manager.getErrors.forEach(error => onReport(CheckLevel.ERROR, error))
            manager.getWarnings.forEach(error => onReport(CheckLevel.WARNING, error))
          }
          new SortingErrorManager(java.util.Collections.singleton(gen))
        }

        // XXX `exports.X = ...` are Scala.js @JSExportTopLevel surfaces
        val exportNames =
          """(?m)^exports\.([A-Za-z_$][\w$]*)\s*=""".r.findAllMatchIn(raw).map(_.group(1)).toSet
        val undeclared = scala.collection.mutable.Set.empty[String]
        val undefRe    = """variable (\S+) is undeclared""".r
        locally {
          val pre = new ClosureCompiler(System.err)
          pre.setErrorManager(quietErrorManager { (_, error) =>
            if (error.getType.key == "JSC_UNDEFINED_VARIABLE")
              undefRe.findFirstMatchIn(error.getDescription).foreach(m => undeclared += m.group(1))
          })
          val preOpts = new CompilerOptions
          preOpts.setLanguageIn(CompilerOptions.LanguageMode.ECMASCRIPT_NEXT)
          preOpts.setChecksOnly(true)
          preOpts.setWarningLevel(DiagnosticGroups.UNDEFINED_VARIABLES, CheckLevel.WARNING)
          pre.compile(defaultExterns, inputs, preOpts)
        }
        // XXX `exports` is the CJS module slot we always seed; everything else is a host-supplied global.
        val hostGlobals = (undeclared.toSet - "exports").toSeq.sorted
        val externsBody =
          (Seq("/** @externs */", "var exports = {};") ++
            exportNames.toSeq.sorted.map(n => s"exports.$n = function() {};") ++
            hostGlobals.map(n => s"var $n;")).mkString("\n")

        val opts = new CompilerOptions
        CompilationLevel.ADVANCED_OPTIMIZATIONS.setOptionsForCompilationLevel(opts)
        WarningLevel.QUIET.setOptionsForWarningLevel(opts)
        opts.setLanguageIn(CompilerOptions.LanguageMode.ECMASCRIPT_NEXT)
        opts.setLanguageOut(CompilerOptions.LanguageMode.ECMASCRIPT_2015)

        val gcc = new ClosureCompiler(System.err)
        gcc.setErrorManager(quietErrorManager { (level, error) =>
          if (level == CheckLevel.ERROR) log.error(error.toString)
        })
        val externs = new java.util.ArrayList[ClosureSourceFile]
        externs.addAll(defaultExterns)
        externs.add(ClosureSourceFile.fromCode("polyregion-externs.js", externsBody))
        val result = gcc.compile(externs, inputs, opts)
        if (!result.success || result.errors.size > 0)
          sys.error(s"Closure ADVANCED failed (${result.errors.size} errors)")
        IO.write(dst, gcc.toSource)
        log.info(
          s"pass bundle: $dst (${dst.length() / 1024} KB; ${exportNames.size} exports, ${hostGlobals.size} host bindings)"
        )
        dst
      }
    )
  )
  .nativePlatform(
    scalaVersions = Seq(scala3Version),
    settings = Seq(
      nativeConfig := Def.uncached {
        val cfg      = nativeConfig.value
        val gcPrefix = findBdwgcPrefix
        val isMac    = scala.util.Properties.isMac
        val isWin    = scala.util.Properties.isWin

        // XXX macOS uses ld64.lld, no ICF so releaseFull, others have ICF so releaseFast for max folding
        val mode =
          if (isMac) scala.scalanative.build.Mode.releaseFull
          else scala.scalanative.build.Mode.releaseFast

        val gcLpath = "-L" + (gcPrefix / "lib").getAbsolutePath
        val linkOpts: Seq[String] = if (isWin) {
          // Windows hides non-exported symbols via @exported / __declspec(dllexport).
          Seq(gcLpath, "-Wl,/opt:icf,/opt:ref", "-Wl,/debug:none")
        } else {
          val exportsList = {
            val f = nativeDir / "polyc" / "generated" / "polypass-exports.txt"
            if (!f.exists)
              sys.error(s"PolyPass exports list missing: $f - run `sbt genCodegen` to regenerate")
            IO.readLines(f).map(_.trim).filter(_.nonEmpty)
          }
          val exportsDir = (Compile / target).value / "polypass-exports"
          IO.createDirectory(exportsDir)
          if (isMac) {
            val f = exportsDir / "polypass-exports.txt"
            IO.write(f, exportsList.map("_" + _).mkString("", "\n", "\n"))
            Seq(
              gcLpath,
              "-Wl,-exported_symbols_list," + f.getAbsolutePath,
              "-Wl,-dead_strip",
              "-Wl,-dead_strip_dylibs",
              "-Wl,-x",
              "-Wl,-no_function_starts"
            )
          } else {
            val f = exportsDir / "polypass-exports.ver"
            IO.write(f, "{\n  global:\n    polypass_*;\n  local:\n    *;\n};\n")
            // --undefined roots survive SN's --start-lib/--end-lib lazy archives.
            val keepRoots = exportsList.map("-Wl,--undefined=" + _)
            val stripFlag =
              if (Option(System.getenv("POLYREGION_POLYPASS_NO_STRIP")).exists(_.trim.nonEmpty)) Nil
              else Seq("-Wl,-s")
            Seq(
              gcLpath,
              "-fuse-ld=lld",
              "-Wl,--icf=all",
              "-static-libstdc++",
              "-static-libgcc",
              "-Wl,--version-script=" + f.getAbsolutePath
            ) ++ keepRoots ++
              Seq("-Wl,--gc-sections") ++ stripFlag
          }
        }

        val sysrootFlag = Option(System.getenv("CMAKE_SYSROOT"))
          .map(_.trim)
          .filter(_.nonEmpty)
          .map(s => Seq(s"--sysroot=$s"))
          .getOrElse(Nil)

        val polyregionArch = Option(System.getenv("POLYREGION_ARCH")).map(_.trim).filter(_.nonEmpty)
        val hostArch       = System.getProperty("os.arch")
        def norm(a: String) = a.toLowerCase match {
          case "amd64" | "x86_64"  => "x86_64"
          case "arm64" | "aarch64" => "arm64"
          case other               => other
        }
        val targetTriple: Option[String] = polyregionArch
          .filter(a => norm(a) != norm(hostArch))
          .flatMap { a =>
            val n = norm(a)
            if (isMac) Some(s"$n-apple-darwin")
            else if (isWin) Some(if (n == "x86_64") "x86_64-pc-windows-msvc" else "aarch64-pc-windows-msvc")
            else Some(if (n == "x86_64") "x86_64-unknown-linux-gnu" else "aarch64-unknown-linux-gnu")
          }
        val crossFlag = targetTriple.map(t => Seq("-target", t)).getOrElse(Nil)

        val withTriple = targetTriple.fold(cfg)(cfg.withTargetTriple(_))

        val nproc       = java.lang.Runtime.getRuntime.availableProcessors
        val thinLtoFlag = if (!isMac && !isWin) Seq(s"-Wl,--threads=$nproc", "-Wl,--lto-O2") else Nil

        withTriple
          .withMode(mode)
          .withLTO(scala.scalanative.build.LTO.thin)
          .withGC(scala.scalanative.build.GC.boehm)
          .withMultithreading(false)
          .withCheckFeatures(false)
          .withBuildTarget(scala.scalanative.build.BuildTarget.libraryDynamic)
          .withCompileOptions(
            // XXX scala-native's dylib_init.c calls getenv; MSVC ucrt deprecates it so silence it on Windows.
            // empty value (not =1) to match SN's own `#define _CRT_SECURE_NO_WARNINGS`, else -Wmacro-redefined
            sysrootFlag ++ crossFlag ++ cfg.compileOptions ++ Seq("-I", (gcPrefix / "include").getAbsolutePath) ++
              (if (isWin) Seq("-D_CRT_SECURE_NO_WARNINGS=") else Nil)
          )
          // XXX macOS ld64 picks the first match so we must prepend
          .withLinkingOptions(sysrootFlag ++ crossFlag ++ thinLtoFlag ++ linkOpts ++ cfg.linkingOptions)
      },
      passDsoDest := nativeDir / "polyc" / (
        if (scala.util.Properties.isWin) "libpolypass.dll"
        else if (scala.util.Properties.isMac) "libpolypass.dylib"
        else "libpolypass.so"
      ),
      exportPassDso := Def.uncached {
        val src = fileConverter.value.toPath((Compile / nativeLink).value).toFile
        val dst = passDsoDest.value
        IO.copyFile(src, dst)
        streams.value.log.info(s"pass DSO: $dst (${dst.length() / 1024} KB)")
        dst
      }
    )
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
      "net.bytebuddy"  % "byte-buddy" % "1.18.11",
      "com.lihaoyi"   %% "pprint"     % "0.9.6",
      "org.scalameta" %% "munit"      % munitVersion % Test
    )
  )
  .dependsOn(ast.jvm(scala3Version), pass.jvm(scala3Version), `binding-jvm`)

lazy val `compiler-testsuite-scala` = project
  .settings(
    commonSettings,
    fork                      := true,
    Test / parallelExecution  := false,
    Test / testForkedParallel := false,
    nativeCompilerInputsTask := Def.uncached {
      nativeCompilerInputs(fileConverter.value)
    },
    Test / unmanagedJars ++= nativeCompilerInputsTask.value,
    javacOptions ++= Seq("-proc:none"),
    commands += Command.command("testUntilFailed") { state =>
      "testFull" :: "testUntilFailed" :: state
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
    name := "examples-scala",
    fork := true,
    nativeCompilerInputsTask := Def.uncached {
      nativeCompilerInputs(fileConverter.value)
    },
    Compile / unmanagedJars ++= nativeCompilerInputsTask.value,
    Compile / mainClass  := mainCls,
    assembly / mainClass := mainCls,
    scalacOptions ++= Seq("-Yretain-trees"),
    libraryDependencies ++= Seq(
      "com.github.pathikrit"   %% "better-files"               % "3.9.2",
      "org.scala-lang.modules" %% "scala-parallel-collections" % "1.2.0"
    )
  )
  .dependsOn(`runtime-scala`)

lazy val `benchmarks-scala` = project
  .enablePlugins(JmhPlugin)
  .settings(
    commonSettings,
    name := "benchmarks-scala",
    fork := true,
    nativeCompilerInputsTask := Def.uncached {
      nativeCompilerInputs(fileConverter.value)
    },
    Compile / unmanagedJars ++= nativeCompilerInputsTask.value,
    Compile / mainClass  := Some("polyregion.benchmarks.Main"),
    assembly / mainClass := Some("polyregion.benchmarks.Main"),
    scalacOptions ++= Seq("-Yretain-trees"),
    libraryDependencies ++= Seq(
      "net.openhft" % "affinity" % "2026.2"
      // "org.scala-lang.modules" %% "scala-parallel-collections" % "1.0.4"
    )
  )
  .dependsOn(`runtime-scala`)

lazy val root = project
  .in(file("."))
  .settings(commonSettings)
  .aggregate(
    `binding-jvm`,
    `interface-codegen`,
    codegen,
    ewgen,
    `runtime-scala`,
    `runtime-java`,
    compiler,
    `compiler-testsuite-scala`,
    `compiler-testsuite-java`,
    `examples-scala`,
    `benchmarks-scala`
  )
  .aggregate(ast.projectRefs*)
  .aggregate(pass.projectRefs*)
