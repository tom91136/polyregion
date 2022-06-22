package polyregion

object Toggles {
  final inline val NoOffload = false

  // need impl
  final inline val FunctionCallSuite = false
  final inline val CaptureSuite      = false

  final inline val ApiSuite = false

  // ok
  final inline val InlineArraySuite = false
  final inline val LogicSuite       = false // Compiler SEGV
  final inline val ControlFlowSuite = false
  final inline val ValueReturnSuite = false // Compiler SEGV

  final inline val StructSuite      = false

  final inline val CastSuite        = true
  final inline val BufferSuite      = true
  final inline val MathSuite        = true
  final inline val IntrinsicSuite   = true

  final inline val ExtensionSuite = false

}
