package polyregion.scalalang

import polyregion.jvm.compiler.Opt

class LaunchConfigSuite extends munit.FunSuite {

  test("settings default to a host-only ahead-of-time build") {
    assertEquals(LaunchConfig.fromSettings(Nil), Right(LaunchConfig.Default))
  }

  test("repeated settings accumulate targets and capabilities") {
    val config = LaunchConfig
      .fromSettings(
        List(
          "polyregion.targets=host@native",
          "polyregion.targets=cuda@sm_80",
          "polyregion.opt=O3",
          "polyregion.jit=true",
          "polyregion.library.path=/a:/b",
          "polyregion.library.capabilities=spectra.dpl@host",
          "polyregion.library.capabilities=spectra.thrust@cuda",
          "unrelated=value"
        )
      )
      .fold(error => fail(error), identity)
    assertEquals(config.targets, List("host" -> "native", "cuda" -> "sm_80"))
    assertEquals(config.targetList, "host@native,cuda@sm_80")
    assertEquals(config.targetNames, List("host", "cuda"))
    assertEquals(config.opt, Opt.O3)
    assert(config.jit)
    assertEquals(config.roots, "/a:/b")
    assertEquals(config.capabilityList, "spectra.dpl@host,spectra.thrust@cuda")
  }

  test("a target without an architecture is native") {
    assertEquals(LaunchConfig.parseTarget("host"), "host" -> "native")
  }

  test("reject an unknown optimisation level and a malformed jit flag") {
    assert(LaunchConfig.fromSettings(List("polyregion.opt=O9")).isLeft)
    assert(LaunchConfig.fromSettings(List("polyregion.jit=maybe")).isLeft)
  }
}
