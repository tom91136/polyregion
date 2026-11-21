package polyregion

class ParCudaSuite extends ParCases {
  protected def backends: List[Backend] = List(hostBackend, cudaBackend)
}
