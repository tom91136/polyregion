package polyregion

class SpectraLaunchSuite extends SpectraLaunchBase {
  deviceTests(
    "CUDA",
    device(d => selectedBackend.forall(_ == "cuda") && d.platformKind == 2 && d.cudaArchitectureMajor > 0)
  )
  selectedTests()
}
