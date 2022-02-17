package polyregion

object Toggles {
  final inline val NoOffload = false

  // need impl
  final inline val FunctionCallSuite = false

  // ok
  final inline val InlineArraySuite = false
  final inline val BufferSuite      = false
  final inline val CaptureSuite     = false
  final inline val ControlFlowSuite = false
  final inline val MathSuite        = false
  final inline val ValueReturnSuite = false
  final inline val IntrinsicSuite   = false

  // `new` needs fixing
  final inline val StructSuite = true

}
