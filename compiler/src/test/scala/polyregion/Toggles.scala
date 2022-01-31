package polyregion

object Toggles {
  final inline val NoOffload = true

  final inline val BufferSuite       = true
  final inline val ControlFlowSuite  = true
  final inline val MathSuite         = true
  final inline val ValueReturnSuite  = true
  final inline val CaptureSuite      = true
  final inline val IntrinsicSuite    = true

  final inline val FunctionCallSuite = false

  final inline val StructSuite      = false
  final inline val InlineArraySuite = false
}
