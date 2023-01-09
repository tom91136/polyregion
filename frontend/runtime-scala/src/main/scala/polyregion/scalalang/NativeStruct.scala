package polyregion.scalalang

import scala.compiletime.*
import scala.deriving.*
import scala.reflect.ClassTag

trait NativeStruct[A] {
  // def name: String
  // def sizeInBytes(a: Option[A]): Int
  def sizeInBytes: Int
  def encode(buffer: java.nio.ByteBuffer, index: Int, a: A): Unit
  def decode(buffer: java.nio.ByteBuffer, index: Int): A
}
object NativeStruct {

//  inline given [A: ClassTag](using ns: NativeStruct[A]): NativeStruct[Array[A]] with {
//    // override def name: String                          = ""
////    override def sizeInBytes =   ns.sizeInBytes
//    override def encode(buffer: java.nio.ByteBuffer, index: Int, xs: Array[A]): Unit = {
//      buffer.putInt(xs.length)
//      var i = 0
//      while (i < xs.size) { ns.encode(buffer, i, xs(i)); i += 1 }
//    }
//
//    override def decode(buffer: java.nio.ByteBuffer, index: Int): Array[A] = {
//      val len = buffer.getInt
//      val xs = Array.ofDim[A](len)
//      var i = 0
//      while (i < len) { xs(i) = ns.decode(buffer, i); i += 1 }
//      xs
//    }
//  }

}
