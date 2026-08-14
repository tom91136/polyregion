package polyregion.spectra

import java.nio.file.Paths

import polyregion.ast.*
import polyregion.ast.PolyAST as p

class SpectraSuite extends munit.FunSuite {

  private val root = Paths.get(sys.props("user.dir")).toAbsolutePath.normalize

  test("declarations are structurally valid") {
    val interfaceDef = Spectra.interfaceDef

    assertEquals(interfaceDef.name, p.Sym("spectra"))
    assertEquals(interfaceDef.decls.flatMap(_.validate), Nil)
    assertEquals(interfaceDef.decls.flatMap(_.classifyArguments.left.toOption.toList.flatten), Nil)
  }

  test("generated APIs are current") {
    assertEquals(SpectraCodeGen.checkGenerated(root), Nil)
  }

  test("implementation candidates conform to every public declaration") {
    val widths     = List(4, 8, 12)
    val variants   = Spectra.ImplementationVariants("portable", List("portable"), widths)
    val candidates = Spectra.implementationCandidates(variants)
    val byName     = Spectra.interfaceDef.decls.map(decl => decl.name -> decl).toMap

    assertEquals(candidates.map(_.publicName).toSet, byName.keySet)
    candidates.foreach { candidate =>
      val publicDecl = byName(candidate.publicName)
      assert(candidate.implementation.conformsTo(publicDecl).isRight)
      assertEquals(candidate.typeSizes.size, publicDecl.tpeVars.size)
      assert(candidate.typeSizes.forall(constraint => widths.contains(constraint.sizeInBytes)))
      assertEquals(candidate.requiredCapabilities, variants.requiredCapabilities)
    }
  }

  test("implementation candidates retain independent input and output element types") {
    val variants  = Spectra.ImplementationVariants("portable", Nil, List(4, 8))
    val transform = Spectra.implementationCandidates(variants).filter(_.publicName == p.Sym("spectra.transform"))

    assertEquals(transform.size, 4)
    assertEquals(
      transform.map(_.typeSizes.map(_.sizeInBytes)).toSet,
      Set(List(4, 4), List(4, 8), List(8, 4), List(8, 8))
    )
    assertEquals(transform.map(_.implementation.name).toSet.size, transform.size)

    val mixed = transform.find(_.typeSizes.map(_.sizeInBytes) == List(4, 8)).get
    assertEquals(mixed.implementation.tpeVars, List("T4", "U8", "Callable0"))
    assertEquals(mixed.implementation.args.head.named.tpe, p.Type.Ptr(p.Type.Var("T4"), p.Type.Space.Global))
    assertEquals(mixed.implementation.args(1).named.tpe, p.Type.Ptr(p.Type.Var("U8"), p.Type.Space.Global))
    assertEquals(mixed.implementation.args.last.named.tpe, p.Type.Var("Callable0"))
    assertEquals(mixed.implementation.rtn, p.Type.Unit0)
  }

  test("implementation variants coexist in one package index") {
    val portable = Spectra.ImplementationVariants("portable", List("portable"), List(4, 8), includeFallback = true)
    val native   = Spectra.ImplementationVariants("native", List("native"), List(4))
    val index    = Spectra.packageIndex(List(portable, native))

    assertEquals(index.interface, Spectra.interfaceDef)
    assertEquals(index.candidates.map(_.implementation.name).distinct.size, index.candidates.size)
    assertEquals(index.candidates.map(_.requiredCapabilities).toSet, Set(List("portable"), List("native")))

    val transform = Spectra.interfaceDef.decls.find(_.name == p.Sym("spectra.transform")).get
    val f32Ptr    = p.Type.Ptr(p.Type.Float32, p.Type.Space.Global)
    val callable = p.FunctionDecl(
      p.Sym("caller.transform"),
      Nil,
      None,
      List(p.Arg(p.Named("x", p.Type.Float32))),
      Nil,
      Nil,
      p.Type.Float32,
      p.Function.Affinity.Host
    )
    val call = p.InvokeSignature(
      transform.name,
      List(p.Type.Float32, p.Type.Float32),
      None,
      List(f32Ptr, f32Ptr, p.Type.IntS32, p.Type.FnRef(callable.name)),
      p.Type.Unit0
    )
    val resolved = index.resolve(
      call,
      List(callable),
      Set("portable"),
      Map(p.Type.Float32 -> 4)
    )
    assert(resolved.exists(_.candidate.typeSizes.nonEmpty))
    assert(resolved.exists(_.candidate.implementation.name.last.endsWith("_w4_w4")))

    val fallback = index.resolve(
      call,
      List(callable),
      Set("portable"),
      Map(p.Type.Float32 -> 2)
    )
    assert(fallback.exists(_.candidate.typeSizes.isEmpty))
    assert(fallback.exists(_.candidate.implementation.name.last.endsWith("_fallback")))

    val f64Ptr = p.Type.Ptr(p.Type.Float64, p.Type.Space.Global)
    val mixedCallable = callable.copy(
      name = p.Sym("caller.transform_mixed"),
      rtn = p.Type.Float64
    )
    val mixedCall = call.copy(
      tpeArgs = List(p.Type.Float32, p.Type.Float64),
      args = List(f32Ptr, f64Ptr, p.Type.IntS32, p.Type.FnRef(mixedCallable.name))
    )
    val mixed = index.resolve(
      mixedCall,
      List(mixedCallable),
      Set("portable"),
      Map(p.Type.Float32 -> 4, p.Type.Float64 -> 8)
    )
    assert(mixed.exists(_.candidate.implementation.name.last.endsWith("_w4_w8")))
    val partial = mixed.toOption.get.candidate.copy(typeSizes = mixed.toOption.get.candidate.typeSizes.take(1))
    assert(
      index
        .copy(candidates = List(partial))
        .resolve(
          mixedCall,
          List(mixedCallable),
          Set("portable"),
          Map(p.Type.Float32 -> 4, p.Type.Float64 -> 8)
        )
        .left
        .exists(_.exists(_.contains("must cover")))
    )

    val mixedFallback = index.resolve(
      mixedCall,
      List(mixedCallable),
      Set("portable"),
      Map(p.Type.Float32 -> 4, p.Type.Float64 -> 16)
    )
    assert(mixedFallback.exists(_.candidate.implementation.name.last.endsWith("_fallback")))
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
      Spectra.packageIndex(List(portable, native)),
      Spectra.packageIndex(List(native, reorderedPortable))
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
      Spectra.implementationCandidates(portable.copy(widths = Nil, includeFallback = true)).forall(_.typeSizes.isEmpty)
    )
  }
}
