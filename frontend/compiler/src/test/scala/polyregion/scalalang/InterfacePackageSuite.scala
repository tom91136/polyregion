package polyregion.scalalang

import polyregion.ast.PolyAST as p
import polyregion.ast.given

import java.nio.file.{Files, Path, Paths}
import java.util.Comparator

class InterfacePackageSuite extends munit.FunSuite {

  private val libraryPath = Option(System.getenv("POLYFRONT_LIBRARY_PATH")).map(_.trim).filter(_.nonEmpty)

  override def beforeEach(context: BeforeEach): Unit =
    assume(libraryPath.isDefined, "POLYFRONT_LIBRARY_PATH does not point at a dist package directory")

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

  test("the dist writes POLYPKG4 archives") {
    val magic = Files.newInputStream(Paths.get(libraryPath.get).resolve("spectra").resolve("lib.polyast"))
    try assertEquals(new String(magic.readNBytes(8), "US-ASCII"), "POLYPKG4")
    finally magic.close()
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
      InterfaceResolver.loadPackage("spectra.missing", libraryPath.get).left.getOrElse(fail("expected an error"))
    assert(error.exists(_.contains("spectra.missing")), error.toString)
  }
}
