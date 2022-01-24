package polyregion

import polyregion.NativeStruct.Type
import scala.compiletime.*
import scala.deriving.*

trait NativeStruct[A] {
  def name: String
  def sizeInBytes: Int
//   def members: IndexedSeq[(String, Type)]
  def encode(buffer: java.nio.ByteBuffer, a: A): Unit
  def decode(buffer: java.nio.ByteBuffer): A
}

object NativeStruct {

  enum Type {
    case Double, Float, Long, Int, Short, Char, Byte, Boolean // String
  }

 
}
