package polyregion.ast

trait Log {
  def info(message: String, details: String*): Unit
  def subLog(name: String): Log
}
