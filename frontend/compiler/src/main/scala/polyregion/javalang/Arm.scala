package polyregion.javalang

import polyregion.ast.PolyAST as p

// `args` picks the call's arguments an implementation takes
case class Arm(cls: p.Sym, impl: p.Sym, args: List[Int])
