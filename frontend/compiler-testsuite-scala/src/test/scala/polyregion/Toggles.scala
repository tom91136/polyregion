package polyregion

object Toggles {

  final inline val NoOffload = false

  // need impl
  final inline val ApiSuite = false

  // ok
  final inline val InlineArraySuite = false

  final inline val StructSuite    = false
  final inline val ExtensionSuite = false

  // works
  final inline val BufferSuite       = false
  final inline val CaptureSuite      = false
  final inline val CastSuite         = false
  final inline val FunctionCallSuite = false
  final inline val MathSuite         = false
  final inline val IntrinsicSuite    = true
  final inline val ValueReturnSuite  = false
  final inline val ControlFlowSuite  = false
  final inline val LogicSuite        = false

}
