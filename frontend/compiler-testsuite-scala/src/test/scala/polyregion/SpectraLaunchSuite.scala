package polyregion

class SpectraLaunchSuite extends SpectraLaunchBase {
  private def cuda =
    device(d => selectedBackend.forall(_ == "cuda") && d.platformKind == 2 && d.cudaArchitectureMajor > 0)
  deviceTests("CUDA", cuda)
  jvmSemanticsTests("CUDA", cuda)
  selectedTests()
}
