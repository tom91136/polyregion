package polyregion.javalang

case class MethodRef(owner: String, name: String, desc: String, instance: Boolean = false) {
  def repr: String = s"$owner.$name$desc"
}
