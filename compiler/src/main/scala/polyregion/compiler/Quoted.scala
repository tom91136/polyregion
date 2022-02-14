package polyregion.compiler

import cats.kernel.Monoid
import polyregion.ast.PolyAst as p

import scala.quoted.*
import polyregion.*

class Quoted(val q: Quotes) {

  import q.reflect.*
  export q.reflect.*

  case class Reference(value: String | p.Term, tpe: Tpe)

  case class FnDependencies(
      clss: Map[p.Sym, p.StructDef] = Map.empty, // external class defs
      defs: Map[p.Sym, DefDef] = Map.empty       // external def defs
  )

  given Monoid[FnDependencies] = Monoid.instance(
    FnDependencies(),                                            //
    (x, y) => FnDependencies(x.clss ++ y.clss, x.defs ++ y.defs) //
  )

  type Val = p.Term | ErasedMethodVal | ErasedModuleSelect
  type Tpe = p.Type | ErasedTpe | ErasedOpaqueTpe | ErasedFnTpe

  case class ErasedModuleSelect(sym: p.Sym)

  case class ErasedMethodVal(receiver: p.Sym | p.Term, sym: p.Sym, tpe: ErasedFnTpe)

  case class ErasedOpaqueTpe(underlying : q.reflect.TypeRepr)

  case class ErasedTpe(name: p.Sym, module: Boolean, args: List[Tpe]) {
    override def toString =
      s"<!${if (module) "<module>." else ""}${name.repr}${if (args.isEmpty) "" else args.mkString("[", ", ", "]")}>"
  }
  case class ErasedFnTpe(  args: List[Tpe], rtn: Tpe) {
    override def toString =
      s"!{ (${args.mkString(",")}) => ${rtn} }"
  }

  case class FnContext(
      depth: Int = 0,                  // ref depth
      traces: List[Tree] = List.empty, // debug

      refs: Map[Symbol, Reference] = Map.empty, // ident/select table

      clss: Map[p.Sym, p.StructDef] = Map.empty, // external class defs
      defs: Map[p.Sym, DefDef] = Map.empty,      // external def defs

      suspended: Map[(String, ErasedFnTpe), ErasedMethodVal] = Map.empty,
      stmts: List[p.Stmt] = List.empty // fn statements
  ) {
    infix def !!(t: Tree)                                     = copy(traces = t :: traces)
    def down(t: Tree)                                         = !!(t).copy(depth = depth + 1)
    def named(tpe: p.Type)                                    = p.Named(s"v${depth}", tpe)
    def suspend(k: (String, ErasedFnTpe))(v: ErasedMethodVal) = copy(suspended = suspended + (k -> v))

    def noStmts                                 = copy(stmts = Nil)
    def inject(refs: Map[Symbol, Reference])    = copy(refs = refs ++ refs)
    def mark(s: p.Sym, d: DefDef)               = copy(defs = defs + (s -> d))
    infix def ::=(xs: p.Stmt*)                  = copy(stmts = stmts ++ xs)
    def replaceStmts(xs: Seq[p.Stmt])           = copy(stmts = xs.toList)
    def mapStmts(f: Seq[p.Stmt] => Seq[p.Stmt]) = copy(stmts = f(stmts).toList)

    def fail[A](reason: String) =
      s"""[depth=$depth] $reason
		 |[Refs]:
		 |  -> ${refs.mkString("\n  -> ")}
		 |[Trace]
		 |  ->${traces.map(x => s"${x.show} -> $x".indent(1)).mkString("---\n  ->")}
		 |[Stmts]
		 |  ->${stmts.map(_.repr.indent(1)).mkString("  ->")}""".stripMargin.fail[A].deferred

    def deps = FnDependencies(clss, defs)

  }

  given Monoid[FnContext] = Monoid.instance(
    FnContext(),
    (x, y) =>
      FnContext(
        x.depth + y.depth,
        x.traces ::: y.traces,
        x.refs ++ y.refs,
        x.clss ++ y.clss,
        x.defs ++ y.defs,
        x.suspended ++ y.suspended,
        x.stmts ::: y.stmts
      )
  )

  def collectTree[A](in: Tree)(f: Tree => List[A]) = {
    val acc = new TreeAccumulator[List[A]] {
      def foldTree(xs: List[A], tree: Tree)(owner: Symbol): List[A] =
        foldOverTree(f(tree) ::: xs, tree)(owner)
    }
    f(in) ::: acc.foldOverTree(Nil, in)(Symbol.noSymbol)
  }

}
