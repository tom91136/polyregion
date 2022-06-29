package polyregion

object Toggles {

  final inline val NoOffload = false

  // need impl
  final inline val ApiSuite = false

  // ok
  final inline val InlineArraySuite = false

  final inline val StructSuite    = true
  final inline val ExtensionSuite = false

  // works
  final inline val BufferSuite       = true
  final inline val CaptureSuite      = true
  final inline val CastSuite         = true
  final inline val FunctionCallSuite = true
  final inline val MathSuite         = true
  final inline val IntrinsicSuite    = true
  final inline val ValueReturnSuite  = true
  final inline val ControlFlowSuite  = true
  final inline val LogicSuite        = true

}
