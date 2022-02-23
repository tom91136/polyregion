package polyregion

import polyregion.compiletime._
import scala.compiletime._
import scala.reflect.ClassTag

class MathSuite extends BaseSuite {

  inline def testExpr[A](inline r: A)(using C: ClassTag[A]) = if (Toggles.MathSuite) {
    test(s"${C.runtimeClass}=${codeOf(r)}=${r}")(assertOffload[A](r))
  }

  // XXX Short, Byte, Char ops promote to Int

  inline def mk4[A](inline xs: Array[A], inline c: A, inline d: A)(inline f: (A, A, A, A) => Unit): Unit = for {
    a <- xs
    b <- xs.reverse
  } yield f(a, b, c, d)

  // TODO  handle % or / by 0
  mk4[Byte](Bytes.filter(_ != 0), 1, 2) { (a, b, c, d) =>
    testExpr(+a)
    testExpr(-a)
    testExpr(~a)
    testExpr(a + b)
    testExpr(a - b)
    testExpr(a * b)
    testExpr(a / b)
    testExpr(a + b + c + d)
    testExpr(a - b - c - d)
    testExpr(a * b * c * d)
    testExpr(a / b / c / d)
    testExpr(a % b % c % d)
    testExpr(a << b << c << d)
    testExpr(a >> b >> c >> d)
    testExpr(a >>> b >>> c >>> d)
    testExpr(a ^ b ^ c ^ d)
    testExpr(a | b | c | d)
    testExpr(a & b & c & d)
    testExpr(a + b - c * d / a % b)
    testExpr(a << b >> c >>> d ^ a | b & ~a)
  }

  // TODO  handle % or / by 0
  mk4[Char](Chars.filter(_ != 0), 1, 2) { (a, b, c, d) =>
    testExpr(+a)
    testExpr(-a)
    testExpr(~a)
    testExpr(a + b)
    testExpr(a - b)
    testExpr(a * b)
    testExpr(a / b)
    testExpr(a + b + c + d)
    testExpr(a - b - c - d)
    testExpr(a * b * c * d)
    testExpr(a / b / c / d)
    testExpr(a % b % c % d)
    testExpr(a << b << c << d)
    testExpr(a >> b >> c >> d)
    testExpr(a >>> b >>> c >>> d)
    testExpr(a ^ b ^ c ^ d)
    testExpr(a | b | c | d)
    testExpr(a & b & c & d)
    testExpr(a + b - c * d / a % b)
    testExpr(a << b >> c >>> d ^ a | b & ~a)
  }

  // TODO  handle % or / by 0
  mk4[Short](Shorts.filter(_ != 0), 1, 2) { (a, b, c, d) =>
    testExpr(+a)
    testExpr(-a)
    testExpr(~a)
    testExpr(a + b)
    testExpr(a - b)
    testExpr(a * b)
    testExpr(a / b)
    testExpr(a + b + c + d)
    testExpr(a - b - c - d)
    testExpr(a * b * c * d)
    testExpr(a / b / c / d)
    testExpr(a % b % c % d)
    testExpr(a << b << c << d)
    testExpr(a >> b >> c >> d)
    testExpr(a >>> b >>> c >>> d)
    testExpr(a ^ b ^ c ^ d)
    testExpr(a | b | c | d)
    testExpr(a & b & c & d)
    testExpr(a + b - c * d / a % b)
    testExpr(a << b >> c >>> d ^ a | b & ~a)
  }

  // TODO replace with Ints and handle % or / by 0
  mk4[Int](Array(3, 4), 1, 2) { (a, b, c, d) =>
    testExpr(+a)
    testExpr(-a)
    testExpr(~a)
    testExpr(a + b)
    testExpr(a - b)
    testExpr(a * b)
    testExpr(a / b)
    testExpr(a + b + c + d)
    testExpr(a - b - c - d)
    testExpr(a * b * c * d)
    testExpr(a / b / c / d)
    testExpr(a % b % c % d)
    testExpr(a << b << c << d)
    testExpr(a >> b >> c >> d)
    testExpr(a >>> b >>> c >>> d)
    testExpr(a ^ b ^ c ^ d)
    testExpr(a | b | c | d)
    testExpr(a & b & c & d)
    testExpr(a + b - c * d / a % b)
    testExpr(a << b >> c >>> d ^ a | b & ~a)
  }

  // TODO replace with Longs and handle % or / by 0
  mk4[Long](Array(3L, 4L), 1, 2) { (a, b, c, d) =>
    testExpr(+a)
    testExpr(-a)
    testExpr(~a)
    testExpr(a + b)
    testExpr(a - b)
    testExpr(a * b)
    testExpr(a / b)
    testExpr(a + b + c + d)
    testExpr(a - b - c - d)
    testExpr(a * b * c * d)
    testExpr(a / b / c / d)
    testExpr(a % b % c % d)
    testExpr(a << b << c << d)
    testExpr(a >> b >> c >> d)
    testExpr(a >>> b >>> c >>> d)
    testExpr(a ^ b ^ c ^ d)
    testExpr(a | b | c | d)
    testExpr(a & b & c & d)
    testExpr(a + b - c * d / a % b)
    testExpr(a << b >> c >>> d ^ a | b & ~a)
  }

  mk4[Float](Floats, 1f, 2f) { (a, b, c, d) =>
    testExpr(+a)
    testExpr(-a)
    testExpr(a + b)
    testExpr(a - b)
    testExpr(a * b)
    testExpr(a / b)
    testExpr(a + b + c + d)
    testExpr(a - b - c - d)
    testExpr(a * b * c * d)
    testExpr(a / b / c / d)
    testExpr(a + b - c * d / a)
  }

  // TODO replace with Double
  mk4[Double](Doubles, 1d, 2d) { (a, b, c, d) =>
    testExpr(+a)
    testExpr(-a)
    testExpr(a + b)
    testExpr(a - b)
    testExpr(a * b)
    testExpr(a / b)
    testExpr(a + b + c + d)
    testExpr(a - b - c - d)
    testExpr(a * b * c * d)
    testExpr(a / b / c / d)
    testExpr(a + b - c * d / a)
  }

}
