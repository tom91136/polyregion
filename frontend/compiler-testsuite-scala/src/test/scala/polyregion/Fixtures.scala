package polyregion

object Fixtures {

  val Booleans: Array[Boolean] = Array[Boolean](true, false)
  val Bytes: Array[Byte]       = Array[Byte](-128, 127, 1, -1, 0, 42)
  val Chars: Array[Char]       = Array[Char]('\u0000', '\uFFFF', 1, 0, 42)
  val Shorts: Array[Short]     = Array[Short](-32768, 32767, 1, -1, 0, 42)
  val Ints: Array[Int]         = Array[Int](0x80000000, 0x7fffffff, 1, -1, 0, 42)
  val Longs: Array[Long]       = Array[Long](0x8000000000000000L, 0x7fffffffffffffffL, 1, -1, 0, 42)
  val Floats: Array[Float] = Array[Float](
    1.4e-45f,
    1.17549435e-38f,
    3.4028235e+38, // XXX 3.4028235e+38f appears to not fit!
    0.0f / 0.0f,
    1,
    -1,
    0,
    42,
    3.14159265358979323846
  )
  val Doubles: Array[Double] = Array[Double](
    4.9e-324d,
    2.2250738585072014e-308d,
    1.7976931348623157e+308d,
    0.0f / 0.0d,
    1,
    -1,
    0,
    42,
    3.14159265358979323846d
  )

}
