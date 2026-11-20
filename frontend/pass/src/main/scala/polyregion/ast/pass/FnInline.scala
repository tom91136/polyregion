package polyregion.ast.pass

// inlines every Invoke reachable from entry to a fixed point (the result Program has no functions):
// the callee body is type-substituted, alpha-renamed, its params bound to the call args, and spliced
// in at the callsite scope
// examples:
//   f(x){ ret x+1 }; y=f(2)            ->  y=2+1   // body spliced, x bound to 2, return becomes the value
//   f[T](x:T){ ret x }; y=f[Int](a)    ->  y=a     // T -> Int before splicing
//   f(x){ if c {ret x} else {ret 0} }  ->  var _phi; if c {_phi=x} else {_phi=0}; ..; _phi  (multi-return phi)
//   g(){ ret f(1) }; z=g()             ->  z=1+1   // nested Invoke inlined first, to a fixed point
// edge cases:
//   single Return        -> the return Expr becomes the callsite value; Return stmts stripped
//   multiple Returns     -> a mutable phi var, each Return rebound to a Mut of the phi
//   moduleCaptures       -> NOT alpha-renamed (shared across inlinings); only locals get the per-call id
//   nested Invoke in body -> recursed before splicing so inlining reaches a fixed point
// inlined stmts stay in the original Invoke's enclosing scope - hoisting to the body root would let
// references to loop-/branch-local vars escape their scope
object FnInline extends FnInliner(selective = false, prefix = "_inline_")
