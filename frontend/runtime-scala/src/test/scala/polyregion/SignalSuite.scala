package polyregion

import polyregion.ast.{CodeGen, MsgPack, PolyAST as p}
import polyregion.jvm.compiler.{Compiler, Options, Target}
import polyregion.jvm.runtime.Context

import java.nio.file.{Files, Path}
import java.util.Comparator
import scala.jdk.CollectionConverters.*
import scala.util.{Try, Using}

object NullChecks {

  private final class Box(val value: Int)

  private def read(box: Box): Int =
    try box.value
    catch case _: NullPointerException => -1

  def survive(): Boolean = {
    val box = new Box(1)
    var sum = 0L
    var i   = 0
    while (i < 1_000_000) {
      sum += read(box)
      i += 1
    }
    var caught = 0
    i = 0
    while (i < 2_000_000) {
      if (read(if (i % 64 == 0) null else box) < 0) caught += 1
      i += 1
    }
    println(s"survived caught=$caught")
    caught == (2_000_000 + 63) / 64 && sum == 1_000_000L
  }
}

object SignalProbe {
  def main(args: Array[String]): Unit = {
    Context.platforms()
    val loaded = Files.readString(Path.of("/proc/self/maps")).contains("libpocl")
    println(s"pocl=$loaded")
    sys.exit(if (NullChecks.survive()) 0 else 1)
  }
}

object LinkerProbe {
  def main(args: Array[String]): Unit = {
    val program = Files.readAllBytes(Path.of(sys.env("LINKER_PROBE_PROGRAM")))
    Compiler.create().compile(program, false, Options.of(Target.LLVM_AMDGCN, "gfx1036"), 12)
    sys.exit(if (NullChecks.survive()) 0 else 1)
  }
}

object GuardProbe {

  def main(args: Array[String]): Unit = {
    System.load(sys.env("SIGNAL_PROBE_LIB"))
    println(s"libjsig=${Context.libjsigLoaded()}")
    val bytes  = 32L * 4096
    val base   = SignalNatives.map(bytes)
    val buffer = SignalNatives.buffer(base, bytes)
    sys.env("SIGNAL_PROBE_MODE") match {
      case "guard" =>
        SignalNatives.guard(base, bytes)
        buffer.putInt(0, 42)
        assert(buffer.getInt(0) == 42 && SignalNatives.resolutions() == 1, "first write was not resolved")
        SignalNatives.rearm(base, bytes)
        buffer.putInt(4096, 7)
        assert(buffer.getInt(4096) == 7 && SignalNatives.resolutions() == 2, "second write was not resolved")
        println("resolved=2")
      case "guard-foreign" =>
        SignalNatives.guardThenForeign(base, bytes)
        buffer.putInt(0, 42)
        assert(buffer.getInt(0) == 42 && SignalNatives.resolutions() == 1, "write was not resolved")
        println("resolved=1")
      case "foreign-guard" =>
        SignalNatives.foreignThenGuard(base, bytes)
        buffer.putInt(0, 42)
        assert(buffer.getInt(0) == 42 && SignalNatives.resolutions() == 1, "write was not resolved")
        println("resolved=1")
      case "guard-foreign-install" =>
        SignalNatives.guard(base, bytes)
        SignalNatives.foreignThenInstall()
        SignalNatives.rearm(base, bytes)
        buffer.putInt(0, 42)
        assert(buffer.getInt(0) == 42 && SignalNatives.resolutions() == 1, "write was not resolved")
        println("resolved=1")
      case "foreign" =>
        SignalNatives.foreign()
    }
    sys.exit(if (NullChecks.survive()) 0 else 1)
  }
}

class SignalSuite extends munit.FunSuite {

  test("a JVM that enumerates OpenCL keeps its own SIGSEGV handler") {
    val (code, output) = ChildJvm.run(
      "polyregion.SignalProbe",
      ChildJvm.pocl() + ("POLYINVOKE_DISABLE_BACKENDS" -> "HSA,HIP,LevelZero")
    )
    assume(output.contains("pocl=true") || code != 0, s"pocl is not loadable here:\n$output")
    assertEquals(code, 0, output)
    assert(output.contains("survived"), output)
  }

  test("a JVM that links an AMDGCN kernel keeps its own SIGSEGV handler") {
    val entry = p.Function(
      p.FunctionDecl(p.Sym("probe"), Nil, None, Nil, Nil, Nil, p.Type.Unit0, p.Function.Affinity.Offload),
      p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const)) :: Nil,
      p.Function.Visibility.Exported,
      p.Function.FpMode.Relaxed,
      p.CallConvention.OffloadEntry
    )
    val program = Files.createTempFile("probe", ".polyast")
    try {
      Files.write(program, MsgPack.encode(CodeGen.programVersioned(p.Program(Some(entry), List(entry), Nil))))
      val (code, output) = ChildJvm.run("polyregion.LinkerProbe", Map("LINKER_PROBE_PROGRAM" -> program.toString))
      assertEquals(code, 0, output)
      assert(output.contains("survived"), output)
    } finally Files.deleteIfExists(program)
  }

  private var probeDir: Option[Path] = None

  override def afterAll(): Unit = probeDir.foreach { dir =>
    Using.resource(Files.walk(dir))(_.sorted(Comparator.reverseOrder).forEach(Files.delete(_)))
  }

  private lazy val probeLibrary: Path = {
    def root(from: Path): Option[Path] =
      Option(from).flatMap(p =>
        if (Files.exists(p.resolve("native/bindings/jvm/jni_utils.h"))) Some(p) else root(p.getParent)
      )
    val os       = System.getProperty("os.name").toLowerCase
    val javaHome = Path.of(System.getProperty("java.home"))
    val repo     = root(Path.of("").toAbsolutePath)
    assume(os.contains("linux") || os.contains("mac"), s"the probe library is not built on $os")
    assume(Files.exists(javaHome.resolve("include/jni.h")), s"no JNI headers under $javaHome")
    assume(repo.isDefined, "the repository sources are not reachable from the test directory")
    val compiler = sys.env.getOrElse("CXX", "c++")
    val dir      = Files.createTempDirectory("signal-probe")
    probeDir = Some(dir)
    val source  = dir.resolve("signalprobe.cpp")
    val library = dir.resolve(System.mapLibraryName("signalprobe"))
    Files.copy(getClass.getResourceAsStream("/signalprobe.cpp"), source)
    val command = Seq(
      compiler,
      "-std=c++20",
      "-shared",
      "-fPIC",
      "-pthread",
      "-w",
      s"-I${javaHome.resolve("include")}",
      s"-I${javaHome.resolve("include").resolve(if (os.contains("mac")) "darwin" else "linux")}",
      s"-I${repo.get.resolve("native/polyfront/rt/include")}",
      s"-I${repo.get.resolve("native/bindings/jvm")}",
      "-o",
      library.toString,
      source.toString
    )
    val process = Try(new ProcessBuilder(command.asJava).redirectErrorStream(true).start()).toOption
    assume(process.isDefined, s"no C++ compiler `$compiler`")
    val output = new String(process.get.getInputStream.readAllBytes())
    assertEquals(process.get.waitFor(), 0, s"cannot build the probe library:\n$output")
    library
  }

  private def libjsig: Option[Path] =
    Some(Path.of(System.getProperty("java.home"), "lib", "libjsig.so")).filter(Files.exists(_))

  private def probe(mode: String, preload: Option[Path]): (Int, String) =
    ChildJvm.run(
      "polyregion.GuardProbe",
      Map("SIGNAL_PROBE_LIB" -> probeLibrary.toString, "SIGNAL_PROBE_MODE" -> mode)
        ++ preload.map("LD_PRELOAD" -> _.toString)
    )

  for ((name, jsig) <- List("without libjsig" -> false, "with libjsig" -> true)) {
    def preload         = if (jsig) libjsig else None
    def prepare(): Unit = if (jsig) assume(libjsig.isDefined, "this JDK has no libjsig")

    test(s"a page guard installed inside a backend call survives it ($name)") {
      prepare()
      val (code, output) = probe("guard", preload)
      assertEquals(code, 0, output)
      assert(output.contains("resolved=2") && output.contains(s"libjsig=$jsig"), output)
    }

    test(s"a foreign handler installed after the page guard in one call is undone ($name)") {
      prepare()
      val (code, output) = probe("guard-foreign", preload)
      assertEquals(code, 0, output)
      assert(output.contains("resolved=1") && output.contains("survived"), output)
    }

    test(s"a page guard installed over a foreign handler in one call does not chain to it ($name)") {
      prepare()
      val (code, output) = probe("foreign-guard", preload)
      assertEquals(code, 0, output)
      assert(output.contains("resolved=1") && output.contains("survived"), output)
    }

    test(s"a page guard installed again over a foreign handler does not chain to it ($name)") {
      prepare()
      val (code, output) = probe("guard-foreign-install", preload)
      assertEquals(code, 0, output)
      assert(output.contains("resolved=1") && output.contains("survived"), output)
    }

    test(s"a foreign handler installed in a backend call is undone ($name)") {
      prepare()
      val (code, output) = probe("foreign", preload)
      assertEquals(code, 0, output)
      assert(output.contains("survived"), output)
    }
  }
}
