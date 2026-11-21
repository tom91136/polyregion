package polyregion

class SpectraHipLaunchSuite extends SpectraLaunchBase {
  private val hipBackend = 1

  deviceTests("HIP", device(_.backend == hipBackend), defaultMayRefuse = true)
}
