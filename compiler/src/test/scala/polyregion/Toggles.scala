package polyregion

object Toggles {
  final inline val NoOffload = false

  // need impl
  final inline val FunctionCallSuite = false
  final inline val InlineArraySuite  = true

  // ok

  final inline val BufferSuite      = false
  final inline val CaptureSuite     = false
  final inline val ControlFlowSuite = false
  final inline val IntrinsicSuite   = false
  final inline val StructSuite      = false
  final inline val MathSuite        = false
  final inline val ValueReturnSuite = false
}
