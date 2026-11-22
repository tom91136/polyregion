package polyregion

import polyregion.jvm.runtime as rt
import polyregion.scalalang.blocking

object CudaProbe {

  lazy val hasDevice: Boolean =
    try blocking.CUDA.devices.nonEmpty
    catch {
      case e: rt.PolyregionRuntimeException if absent(e.getMessage) => false
    }

  private def absent(message: String): Boolean =
    message.contains("failed to open libcuda dynamic library") || message.contains("CUDA_ERROR_NO_DEVICE")
}
