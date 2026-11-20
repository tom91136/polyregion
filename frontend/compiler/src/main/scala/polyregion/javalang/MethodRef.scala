package polyregion.javalang

case class MethodRef(owner: String, name: String, desc: String) {
  def repr: String = s"$owner.$name$desc"
}
