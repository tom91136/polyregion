package polyregion

object Toggles {
  final inline val NoOffload = false

  // need impl
  final inline val FunctionCallSuite = false

  // ok
  final inline val InlineArraySuite = true
  final inline val BufferSuite      = true
  final inline val CaptureSuite     = true
  final inline val ControlFlowSuite = true
  final inline val MathSuite        = true
  final inline val ValueReturnSuite = true
  final inline val IntrinsicSuite   = true

  // `new` needs fixing
  final inline val StructSuite = false

}
