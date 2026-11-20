package polyregion.ast.pass

// inlines only callees that take a callable and drops them, every other call is left alone
object CallableInline extends FnInliner(selective = true, prefix = "_callable_")
