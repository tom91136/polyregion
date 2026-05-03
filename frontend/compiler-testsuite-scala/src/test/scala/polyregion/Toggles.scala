package polyregion

object Toggles {

  final inline val NoOffload = false

  // Suites still hitting macro/backend errors (deferred):
  final inline val CaptureSuite      = true // complex-captures test using Option[Node] disabled in source; rest enabled
  final inline val CompoundOpsSuite  = true
  final inline val ExtensionSuite    = true
  final inline val FunctionCallSuite = true
  final inline val GivenSuite        = true
  final inline val InlineArraySuite  = true
  final inline val IntrinsicSuite    = true
  final inline val MathSuite         = true
  final inline val CompoundCaptureSuite = true

  // Suites without macro errors (should compile; runtime status unknown):
  final inline val ApiSuite              = true
  final inline val BufferSuite           = true
  final inline val CastSuite             = true
  final inline val CollectionLengthSuite = true
  final inline val LengthSuite           = true
  final inline val ControlFlowSuite      = true
  final inline val LogicSuite            = true
  final inline val StructSuite           = true
  final inline val ValueReturnSuite      = true

}
