package polyregion.javalang

import polyregion.ast.PolyAST as p

// `args` picks the call's arguments an implementation takes; `implReceiver` is the class that declares an inherited one
case class Arm(cls: p.Sym, impl: p.Sym, args: List[Int], implReceiver: Option[p.Sym] = None)
