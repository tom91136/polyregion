package polyregion.scalalang

import polyregion.ast.PolyAST as p

class LaunchLayoutSuite extends munit.FunSuite {

  private val environment = p.StructDef(
    p.Sym("environment"),
    Nil,
    List(
      p.Named("scale", p.Type.IntS32),
      p.Named("values", p.Type.Ptr(p.Type.IntS32, p.Type.Space.Global)),
      p.Named("offset", p.Type.IntS64)
    ),
    Nil
  )

  private def layout(targets: (String, String)*) =
    LaunchMacro.layoutOf(
      environment,
      LaunchConfig.Default.copy(targets = targets.toList.map((name, arch) => LaunchConfig.DeviceTarget(name, arch)))
    )

  test("the environment layout is shared by the host and every configured device target") {
    val host = layout("host" -> "native").fold(throw _, identity)
    val all  = layout("host" -> "native", "cuda" -> "sm_80").fold(throw _, identity)
    assertEquals(all.sizeInBytes, host.sizeInBytes)
    assertEquals(all.members.map(_.offsetInBytes).toList, host.members.map(_.offsetInBytes).toList)
  }

  test("a layout that differs between targets is rejected") {
    val error = layout("host" -> "native", "spirv32_kernel" -> "native").left.getOrElse(fail("expected an error"))
    assert(error.getMessage.contains("differs between the targets"), error.getMessage)
  }
}
