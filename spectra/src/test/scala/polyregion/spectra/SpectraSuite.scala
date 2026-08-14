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
}
