package polyregion.scala

import cats.kernel.Monoid
import polyregion.ast.{PolyAst as p, *}

class Quoted(val underlying: scala.quoted.Quotes) {

  import underlying.reflect.*
  export underlying.reflect.*
  given quotes : scala.quoted.Quotes  = underlying

  // Reference = CaptureVarName | DefaultTermValue
  case class Reference(value: String | p.Term, tpe: p.Type)

  // // TODO rename to FnScope
  // case class FnDependencies(
  //     clss: Map[p.Sym, p.StructDef] = Map.empty,  // external class defs
  //     defs: Map[p.Signature, DefDef] = Map.empty, // external def defs
  //     vars: Map[p.Named, Ref] = Map.empty         // external val defs
  // )

  enum ClassKind {
    case Object, Class
  }

  case class Dependencies(
      classes: Map[ClassDef, Set[p.Type.Struct]] = Map.empty,
      functions: Map[DefDef, Set[p.Expr.Invoke]] = Map.empty
  ) {
    def witness(x: ClassDef, application: p.Type.Struct) =
      copy(classes = classes.updatedWith(x)(x => Some(x.getOrElse(Set.empty) + application)))
    def witness(x: DefDef, application: p.Expr.Invoke) =
      copy(functions = functions.updatedWith(x)(x => Some(x.getOrElse(Set.empty) + application)))
  }

  given Monoid[Dependencies] = Monoid.instance(
    Dependencies(),
    (x, y) => Dependencies(x.classes ++ y.classes, x.functions ++ y.functions)
  )

  // TODO rename to RemapContext
  case class RemapContext(
      root : Symbol,
      depth: Int = 0,                  // ref depth
      traces: List[Tree] = List.empty, // debug

      refs: Map[Ref, p.Term] = Map.empty, // ident/select table

      // clss: Map[p.Sym, p.StructDef] = Map.empty,  // external class defs
      // defs: Map[p.Signature, DefDef] = Map.empty, // external def defs

      deps: Dependencies = Dependencies(),
      stmts: List[p.Stmt] = List.empty // fn statements
  ) {
    infix def !!(t: Tree)  = copy(traces = t :: traces)
    def down(t: Tree)      = !!(t).copy(depth = depth + 1)
    def named(tpe: p.Type) = p.Named(s"v${depth}", tpe)

    def noStmts = copy(stmts = Nil)
    // def mark(s: p.Signature, d: DefDef) = copy(defs = defs + (s -> d))
    infix def ::=(xs: p.Stmt*)                      = copy(stmts = stmts ++ xs)
    def replaceStmts(xs: Seq[p.Stmt])               = copy(stmts = xs.toList)
    def updateDeps(f: Dependencies => Dependencies) = copy(deps = f(deps))

    def fail[A](reason: String) =
      s"""[depth=$depth] $reason
		 |[Refs]:
		 |  -> ${refs.mkString("\n  -> ")}
		 |[Trace]
		 |  ->${traces.map(x => s"${x.show} -> $x".indent_(1)).mkString("---\n  ->")}
		 |[Stmts]
		 |  ->${stmts.map(_.repr.indent_(1)).mkString("  ->")}""".stripMargin.fail[A]

    // def deps = FnDependencies( )

  }

  // given Monoid[RemapContext] = Monoid.instance(
  //   RemapContext(),
  //   (x, y) =>
  //     RemapContext(
  //       x.depth + y.depth,
  //       x.traces ::: y.traces,
  //       x.refs ++ y.refs,
  //       x.clss ++ y.clss,
  //       x.defs ++ y.defs,
  //       x.stmts ::: y.stmts
  //     )
  // )

  def collectTree[A](in: Tree)(f: Tree => List[A]) = {
    val acc = new TreeAccumulator[List[A]] {
      def foldTree(xs: List[A], tree: Tree)(owner: Symbol): List[A] =
        foldOverTree(f(tree) ::: xs, tree)(owner)
    }
    f(in) ::: acc.foldOverTree(Nil, in)(Symbol.noSymbol)
  }

}