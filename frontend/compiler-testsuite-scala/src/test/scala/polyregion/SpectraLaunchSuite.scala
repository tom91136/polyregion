package polyregion

class SpectraLaunchSuite extends SpectraLaunchBase {
  deviceTests("CUDA", device(d => d.platformKind == 2 && d.cudaArchitectureMajor > 0))
}
