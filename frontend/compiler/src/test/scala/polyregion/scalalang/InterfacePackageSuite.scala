package polyregion.scalalang

import polyregion.ast.PolyAST as p
import polyregion.ast.given

import java.nio.file.{Files, Path, Paths}
import java.util.Comparator

class InterfacePackageSuite extends munit.FunSuite {

  private val libraryPath = Option(System.getProperty("polyregion.library.path")).map(_.trim).filter(_.nonEmpty)

  override def beforeEach(context: BeforeEach): Unit =
    assume(libraryPath.isDefined, "no package directory is available")

  private def load(name: String, capabilities: Set[String] = Set.empty): p.Package =
    InterfaceResolver
      .loadPackage(name, libraryPath.get, capabilities)
      .fold(errors => fail(errors.mkString("\n")), identity)

  private def deleteTree(root: Path): Unit = {
    val paths = Files.walk(root)
    try paths.sorted(Comparator.reverseOrder[Path]()).forEach(path => Files.delete(path))
    finally paths.close()
  }

  private def copySpectra(root: Path): Unit = {
    Files.createDirectories(root.resolve("spectra"))
    Files.copy(
      Paths.get(libraryPath.get).resolve("spectra").resolve("lib.polyast"),
      root.resolve("spectra").resolve("lib.polyast")
    )
  }

  override def munitTimeout: scala.concurrent.duration.Duration = scala.concurrent.duration.Duration(10, "min")

  private def importCopy(targets: List[String]) = {
    val pointer = p.Type.Ptr(p.Type.IntS32, p.Type.Space.Global)
    val invoke: p.Expr.Invoke = p.Expr.Invoke(
      p.Type.FnRef(p.Sym("consumer.copy")),
      Nil,
      None,
      List(
        p.Term.Select(p.Named("in", pointer), Nil, pointer),
        p.Term.IntS32Const(4),
        p.Term.Select(p.Named("out", pointer), Nil, pointer)
      ),
      p.Type.Unit0
    )
    InterfaceResolver.importAll(
      List(InterfaceResolver.Import("spectra", "spectra.copy", invoke)),
      Nil,
      libraryPath.get,
      Set("spectra.thrust@cuda", "spectra.dpl@host"),
      Map(p.Type.IntS32 -> 4),
      targets
    )
  }

  test("the dist writes POLYPKG4 archives") {
    val magic = Files.newInputStream(Paths.get(libraryPath.get).resolve("spectra").resolve("lib.polyast"))
    try assertEquals(new String(magic.readNBytes(8), "US-ASCII"), "POLYPKG4")
    finally magic.close()
  }

  test("each configured target resolves the implementation its capability names") {
    val resolved              = importCopy(List("host", "cuda")).fold(errors => fail(errors.mkString("\n")), identity)
    def names(target: String) = resolved(target)._1.map(_.name.fqcn)
    val host                  = names("host")
    assert(host.exists(_.contains("dpl")), host.toString)
    assert(!host.exists(_.contains("thrust")), host.toString)
    val cuda = names("cuda")
    assert(cuda.exists(_.contains("thrust")), cuda.toString)
    assert(!cuda.exists(_.contains("dpl")), cuda.toString)
  }

  test("reject a target nothing serves") {
    val errors = importCopy(List("host", "opencl")).left.getOrElse(fail("expected an error"))
    assert(errors.exists(_.contains("spectra.copy")), errors.toString)
  }

  test("load the spectra package with the implementations its capabilities select") {
    val pkg = load("spectra")
    assertEquals(pkg.interface.name.fqcn, "spectra")
    assert(pkg.interface.declarations.nonEmpty)
    val implementations = pkg.program.functions.filter(_.implements.nonEmpty)
    val implemented     = implementations.flatMap(_.implements).toSet
    pkg.interface.declarations.foreach(decl =>
      assert(implemented(decl.name), s"no implementation of ${decl.name.fqcn}")
    )
    assert(implementations.filter(_.requiredCapabilities.isEmpty).forall(_.body.nonEmpty))
    assert(implementations.filter(_.requiredCapabilities.nonEmpty).forall(_.body.isEmpty))
  }

  test("load capability-gated spectra implementations when the capability is configured") {
    val gated = load("spectra", Set("spectra.thrust@cuda")).program.functions.filter(_.requiredCapabilities.nonEmpty)
    assert(gated.exists(_.body.nonEmpty))
  }

  test("omit capability-gated spectra implementations when the capability is not configured") {
    val gated = load("spectra").program.functions.filter(f => f.implements.nonEmpty && f.requiredCapabilities.nonEmpty)
    assert(gated.nonEmpty)
    assert(gated.forall(_.body.isEmpty))
  }

  test("load the polyregion.core package") {
    val pkg   = load("polyregion.core")
    val names = pkg.interface.declarations.map(_.name.fqcn).toSet
    assert(names("polyregion.core.for_each"), names.toString)
    assert(names("polyregion.core.transform_reduce"), names.toString)
    val implementations = pkg.program.functions.filter(_.implements.nonEmpty)
    val implemented     = implementations.flatMap(_.implements).toSet
    pkg.interface.declarations.foreach(decl =>
      assert(implemented(decl.name), s"no implementation of ${decl.name.fqcn}")
    )
    assert(implementations.filter(_.requiredCapabilities.isEmpty).forall(_.body.nonEmpty))
    assert(implementations.filter(_.requiredCapabilities.nonEmpty).forall(_.body.isEmpty))
  }

  test("reject a package that two configured roots both provide") {
    val roots = List(Files.createTempDirectory("polyregion-roots"), Files.createTempDirectory("polyregion-roots"))
    try {
      roots.foreach(copySpectra)
      val error = InterfaceResolver
        .loadPackage("spectra", roots.mkString(java.io.File.pathSeparator), Set.empty)
        .left
        .getOrElse(fail("expected an error"))
      assert(error.exists(_.contains("ambiguous across 2 package roots")), error.toString)
    } finally roots.foreach(deleteTree)
  }

  test("report a missing package") {
    val error =
      InterfaceResolver
        .loadPackage("spectra.missing", libraryPath.get, Set.empty)
        .left
        .getOrElse(fail("expected an error"))
    assert(error.exists(_.contains("spectra.missing")), error.toString)
  }
}
