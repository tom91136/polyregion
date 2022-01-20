package polyregion


class BufferSpec extends munit.FunSuite {

  test("buffer is index-able"){
    assertEquals(Buffer(1, 2, 3).toList, List(1, 2, 3))
  }
  


}
