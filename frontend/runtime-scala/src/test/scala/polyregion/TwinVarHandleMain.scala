package polyregion

import polyregion.jvm.twin.FieldAccess

object TwinVarHandleMain {
  def main(args: Array[String]): Unit = {
    val xs = java.util.ArrayList[Integer]()
    (1 to 3).foreach(i => xs.add(i))
    val data  = FieldAccess.of(classOf[java.util.ArrayList[?]], "elementData", "[Ljava/lang/Object;")
    val size  = FieldAccess.of(classOf[java.util.ArrayList[?]], "size", "I")
    val elems = data.get(xs).asInstanceOf[Array[AnyRef]]
    assertEq(elems.take(3).toList.map(x => x: Any), List[Any](1, 2, 3))
    assertEq(size.get(xs), 3)
    val replaced = Array[AnyRef](Integer.valueOf(7), Integer.valueOf(8), Integer.valueOf(9), null)
    data.set(xs, replaced)
    size.set(xs, 3)
    assertEq(xs.toArray.toList.map(x => x: Any), List[Any](7, 8, 9))
    val words = FieldAccess.of(classOf[java.util.BitSet], "words", "[J")
    val bits  = java.util.BitSet(); bits.set(3)
    assertEq(words.get(bits).asInstanceOf[Array[Long]](0), 8L)
    println("twin varhandle ok")
  }
  private def assertEq(a: Any, b: Any): Unit = if (a != b) throw AssertionError(s"$a != $b")
}
