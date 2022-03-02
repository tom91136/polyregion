package polyregion.scala

import scala.compiletime.*
import scala.deriving.*

trait NativeStruct[A] {
  def name: String
  def sizeInBytes: Int
//   def members: IndexedSeq[(String, Type)]
  def encode(buffer: java.nio.ByteBuffer, index: Int, a: A): Unit
  def decode(buffer: java.nio.ByteBuffer, index: Int): A
}
