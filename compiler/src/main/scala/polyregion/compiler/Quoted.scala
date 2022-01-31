package polyregion.compiler

import cats.kernel.Monoid
import polyregion.ast.PolyAst as p

import scala.quoted.*

class Quoted(val q: Quotes) {

  import q.reflect.*
  export q.reflect.*

  case class Reference(value: String | p.Term, tpe: p.Type)

  case class FnContext(
      depth: Int = 0,                  // ref depth
      traces: List[Tree] = List.empty, // debug

      refs: Map[Symbol, Reference] = Map.empty, // ident/select table

      clss: Map[p.Sym, p.StructDef] = Map.empty, // external class defs
      defs: Set[DefDef] = Set.empty,             // external def defs

      stmts: List[p.Stmt] = List.empty // fn statements
  ) {
    infix def !!(t: Tree)  = copy(traces = t :: traces)
    def down(t: Tree)      = !!(t).copy(depth = depth + 1)
    def named(tpe: p.Type) = p.Named(s"v${depth}", tpe)

    def noStmts                              = copy(stmts = Nil)
    def inject(refs: Map[Symbol, Reference]) = copy(refs = refs ++ refs)
    def mark(d: DefDef)                      = copy(defs = defs + d)
    infix def ::=(xs: p.Stmt*)               = copy(stmts = stmts ++ xs)
    infix def replaceStmts(xs: Seq[p.Stmt])  = copy(stmts = xs.toList)

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
