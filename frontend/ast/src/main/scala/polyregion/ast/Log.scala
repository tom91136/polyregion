package polyregion.ast

trait Log {
  def enabled: Boolean = true
  def info(message: String, details: String*): Unit
  def subLog(name: String): Log
}
