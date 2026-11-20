package polyregion.javalang

object Interop {

  // Java APIs take null for an absent argument
  def absent[A >: Null]: A = Option.empty[A].orNull
}
