package polyregion.javalang

import net.bytebuddy.jar.asm.Type
import polyregion.ast.PolyAST as p

// a virtual or interface call whose receiver class is only known to be a subtype of `owner`
case class OpenSite(owner: String, name: String, desc: String) {
  def repr: String  = s"$owner.$name$desc"
  def callee: p.Sym = p.Sym(repr)
  def arity: Int    = Type.getArgumentTypes(desc).length
}
