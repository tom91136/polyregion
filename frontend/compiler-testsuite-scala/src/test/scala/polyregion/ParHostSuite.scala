package polyregion

class ParHostSuite extends ParCases {
  protected def backends: List[Backend] = List(hostBackend)
}
