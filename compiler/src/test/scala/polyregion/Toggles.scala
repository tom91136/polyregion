package polyregion

object Toggles {
  final inline val NoOffload = false

  final inline val BufferSuite       = false
  final inline val ControlFlowSuite  = false
  final inline val MathSuite         = false
  final inline val ValueReturnSuite  = false
  final inline val CaptureSuite      = false
  final inline val IntrinsicSuite    = false


  final inline val FunctionCallSuite = false

  final inline val StructSuite      = true
  final inline val InlineArraySuite = false
}
