package polyregion.spectra

import java.nio.file.Paths

import polyregion.ast.*
import polyregion.ast.PolyAST as p

class SpectraSuite extends munit.FunSuite {

  private val root                                        = Paths.get(sys.props("user.dir")).toAbsolutePath.normalize
  private def exactSizes(function: p.Function): List[Int] = function.decl.tpeVars.flatMap(_.exactSizeInBytes)

  test("declarations are structurally valid") {
    val interfaceDef = Spectra.interfaceDef

    assertEquals(interfaceDef.name, p.Sym("spectra"))
    assertEquals(interfaceDef.declarations.flatMap(_.validate), Nil)
    assertEquals(interfaceDef.declarations.flatMap(_.validateInterfaceDeclaration), Nil)
  }

  test("generated APIs are current") {
    assertEquals(SpectraCodeGen.checkGenerated(root), Nil)
  }

  test("implementation candidates conform to every public declaration") {
    val widths     = List(4, 8, 12)
    val variants   = Spectra.ImplementationVariants("portable", List("portable"), widths)
    val candidates = Spectra.implementationCandidates(variants)
    val byName     = Spectra.interfaceDef.declarations.map(decl => decl.name -> decl).toMap

    assertEquals(candidates.flatMap(_.implements).toSet, byName.keySet)
    candidates.foreach { candidate =>
      val publicDecl = byName(candidate.implements.get)
      assertEquals(candidate.decl.validate, Nil)
      assertEquals(exactSizes(candidate).size, publicDecl.tpeVars.size)
      assert(exactSizes(candidate).forall(widths.contains))
      assertEquals(candidate.requiredCapabilities, variants.requiredCapabilities)
    }
  }

  test("implementation candidates retain independent input and output element types") {
    val variants  = Spectra.ImplementationVariants("portable", Nil, List(4, 8))
    val transform = Spectra.implementationCandidates(variants).filter(_.implements.contains(p.Sym("spectra.transform")))

    assertEquals(transform.size, 4)
    assertEquals(
      transform.map(exactSizes).toSet,
      Set(List(4, 4), List(4, 8), List(8, 4), List(8, 8))
    )
    assertEquals(transform.map(_.decl.name).toSet.size, transform.size)

    val mixed = transform.find(exactSizes(_) == List(4, 8)).get
    assertEquals(mixed.decl.tpeVars.map(_.name), List("T4", "U8", "Callable0"))
    assertEquals(mixed.decl.args.head.named.tpe, p.Type.Ptr(p.Type.Var("T4", Some(4)), p.Type.Space.Global))
    assertEquals(mixed.decl.args(1).named.tpe, p.Type.Ptr(p.Type.Var("U8", Some(8)), p.Type.Space.Global))
    assertEquals(mixed.decl.args.last.named.tpe, p.Type.Var("Callable0"))
    assertEquals(mixed.decl.rtn, p.Type.Unit0)
  }

  test("implementation variants coexist in one package") {
    val portable = Spectra.ImplementationVariants("portable", List("portable"), List(4, 8), includeFallback = true)
    val native   = Spectra.ImplementationVariants("native", List("native"), List(4))
    val pack     = Spectra.implementationPackage(List(portable, native))

    assertEquals(pack.interface, Spectra.interfaceDef)
    assertEquals(pack.program.functions.map(_.decl.name).distinct.size, pack.program.functions.size)
    assertEquals(pack.program.functions.map(_.requiredCapabilities).toSet, Set(List("portable"), List("native")))

    val transform = pack.program.functions.filter(_.implements.contains(p.Sym("spectra.transform")))
    assert(transform.exists(_.decl.name.last.endsWith("_w4_w4")))
    assert(transform.exists(_.decl.name.last.endsWith("_w4_w8")))
    assert(transform.exists(_.decl.name.last.endsWith("_w8_w4")))
    assert(transform.exists(_.decl.name.last.endsWith("_w8_w8")))
    assert(transform.exists(_.decl.name.last.endsWith("_fallback")))
    assert(
      transform.exists(candidate =>
        candidate.requiredCapabilities == List("native") && exactSizes(candidate) == List(4, 4)
      )
    )
  }

  test("implementation variant configuration is validated and canonical") {
    val portable =
      Spectra.ImplementationVariants("portable", List("spirv", "fallback"), List(8, 4), includeFallback = true)
    val native = Spectra.ImplementationVariants("native", List("sm_89"), List(8, 4))
    val reorderedPortable = portable.copy(
      requiredCapabilities = portable.requiredCapabilities.reverse,
      widths = portable.widths.reverse
    )

    assertEquals(
      Spectra.implementationPackage(List(portable, native)),
      Spectra.implementationPackage(List(native, reorderedPortable))
    )
    intercept[IllegalArgumentException](Spectra.implementationCandidates(portable.copy(name = "../portable")))
    intercept[IllegalArgumentException](
      Spectra.implementationCandidates(portable.copy(requiredCapabilities = List("spirv", " ")))
    )
    intercept[IllegalArgumentException](
      Spectra.implementationCandidates(portable.copy(requiredCapabilities = List("spirv", "native\ncode")))
    )
    intercept[IllegalArgumentException](
      Spectra.implementationCandidates(portable.copy(requiredCapabilities = List("spirv", "spirv")))
    )
    intercept[IllegalArgumentException](
      Spectra.implementationCandidates(portable.copy(widths = Nil, includeFallback = false))
    )
    assert(
      Spectra
        .implementationCandidates(portable.copy(widths = Nil, includeFallback = true))
        .forall(exactSizes(_).isEmpty)
    )
  }
}
