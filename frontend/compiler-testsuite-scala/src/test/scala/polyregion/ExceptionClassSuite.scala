package polyregion

import polyregion.scalalang.*

class TopLevelFailure(message: String) extends RuntimeException(message)

object Failures {
  class Inner(message: String) extends Exception(message)
  object Deeper {
    class Deep(message: String) extends IllegalStateException(message)
  }
  class NoStringConstructor                extends RuntimeException("fixed")
  class IntConstructor(val code: Int)      extends RuntimeException("code")
  abstract class Abstract(message: String) extends RuntimeException(message)
}

class ExceptionClassSuite extends RangeSuiteBase {

  class SuiteFailure(message: String) extends RuntimeException(message)

  raisesInstance[java.util.NoSuchElementException]("jdk-class", "gone")(
    intrinsics.raise[java.util.NoSuchElementException]("gone")
  )
  raisesInstance[TopLevelFailure]("top-level-class", "top")(intrinsics.raise[TopLevelFailure]("top"))
  raisesInstance[Failures.Inner]("object-nested-class", "inner")(intrinsics.raise[Failures.Inner]("inner"))
  raisesInstance[Failures.Deeper.Deep]("doubly-nested-class", "deep")(intrinsics.raise[Failures.Deeper.Deep]("deep"))
  raisesInstance[IllegalArgumentException]("conditional", "odd") {
    val x = 3
    if (x % 2 == 1) intrinsics.raise[IllegalArgumentException]("odd")
  }

  raisesInstance[java.lang.IllegalStateException]("throw-new-illegal-state", "x") {
    throw new java.lang.IllegalStateException("x")
    ()
  }
  raisesInstance[java.lang.ArrayIndexOutOfBoundsException]("throw-new-index", "idx") {
    throw new java.lang.ArrayIndexOutOfBoundsException("idx")
    ()
  }
  raisesInstance[java.lang.ArithmeticException]("divide-by-zero", "/ by zero") {
    val zero    = 0
    val one     = 1
    val ignored = one / zero
    ()
  }
  raisesInstance[java.lang.IllegalStateException]("throw-as-call-argument", "arg")(
    throw new java.lang.IllegalStateException("arg")
  )
  raisesInstance[java.lang.IllegalStateException]("bmp-message", "caf\u00e9 \u2603") {
    throw new java.lang.IllegalStateException("caf\u00e9 \u2603")
    ()
  }
  raisesInstance[java.lang.IllegalStateException]("non-bmp-message", "smile \uD83D\uDE00") {
    throw new java.lang.IllegalStateException("smile \uD83D\uDE00")
    ()
  }

  test("a type that is not a Throwable is refused") {
    val errors = compileErrors("intrinsics.raise[String](\"x\")")
    assert(errors.contains("Throwable"), errors)
  }

  test("an exception class without a String constructor is refused") {
    val errors = compileErrors("""
      compiletime.offload0(blocking.Host.context, _ => ())(intrinsics.raise[Failures.NoStringConstructor]("x"))
    """)
    assert(errors.contains("String"), errors)
    assert(errors.contains("NoStringConstructor"), errors)
  }

  test("an exception class whose constructor takes another type is refused") {
    val errors = compileErrors("""
      compiletime.offload0(blocking.Host.context, _ => ())(intrinsics.raise[Failures.IntConstructor]("x"))
    """)
    assert(errors.contains("IntConstructor"), errors)
  }

  test("an abstract exception class is refused") {
    val errors = compileErrors("""
      compiletime.offload0(blocking.Host.context, _ => ())(intrinsics.raise[Failures.Abstract]("x"))
    """)
    assert(errors.contains("Abstract"), errors)
  }

  test("an exception class nested in a class is refused") {
    val errors = compileErrors("""
      compiletime.offload0(blocking.Host.context, _ => ())(intrinsics.raise[SuiteFailure]("x"))
    """)
    assert(errors.contains("SuiteFailure"), errors)
  }

  test("a local exception class is refused") {
    val errors = compileErrors("""
      class Local(message: String) extends RuntimeException(message)
      compiletime.offload0(blocking.Host.context, _ => ())(intrinsics.raise[Local]("x"))
    """)
    assert(errors.contains("Local"), errors)
  }

  test("a throw of a variable is refused") {
    val errors = compileErrors("""
      val e = new java.lang.IllegalStateException("x")
      compiletime.offload0(blocking.Host.context, _ => ()) { throw e }
    """)
    assert(errors.contains("Unsupported `throw`"), errors)
  }

  test("a throw of a class without a message is refused") {
    val errors = compileErrors("""
      compiletime.offload0(blocking.Host.context, _ => ()) { throw new java.lang.IllegalStateException() }
    """)
    assert(errors.contains("Unsupported `throw`"), errors)
  }

  test("a throw with a null message is refused") {
    val errors = compileErrors("""
      compiletime.offload0(blocking.Host.context, _ => ()) { throw new java.lang.IllegalStateException(null: String) }
    """)
    assert(errors.contains("Unsupported `throw`"), errors)
  }
}
