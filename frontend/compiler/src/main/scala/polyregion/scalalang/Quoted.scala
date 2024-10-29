package polyregion.scalalang

import cats.kernel.Monoid
import polyregion.ast.{PolyAST as p, *}

import scala.annotation.targetName

class Quoted(val underlying: scala.quoted.Quotes) {

  import underlying.reflect.*
  export underlying.reflect.*
  given quotes: scala.quoted.Quotes = underlying

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

  type ClsWitnesses = Map[Symbol, Set[p.Type.Struct]]
  type FnWitnesses  = Map[Symbol, Set[p.Expr.Invoke]]
  type Retyped      = (Option[p.Term], p.Type)

  // TODO everything here can be a Set as we don't need the rhs
  case class Dependencies(
      modules: Map[Symbol, p.Type.Struct] = Map.empty,
      classes: ClsWitnesses = Map.empty,
      functions: FnWitnesses = Map.empty,
      resolvedFunctions: List[p.Function] = Nil
  ) {
    @targetName("witness_module")
    def witness(x: Symbol, tpe: p.Type.Struct) = copy(modules = modules + (x -> tpe))
    def witness(x: ClassDef, application: p.Type.Struct) =
      if (TypeRepr.of[polyregion.scalalang.intrinsics.type].typeSymbol.fullName == x.symbol.fullName) {
        this
      } else if (x.symbol.flags.is(Flags.Module)) {
        report.errorAndAbort(s"Witness illegal module ClassDef ${x.symbol} (${x.symbol.fullName})")
      } else {
        copy(classes = classes.updatedWith(x.symbol) {
          case Some(ivks) => Some(ivks + application)
          case None       => Some(Set(application))
        })
      }

    def witness(x: DefDef, application: p.Expr.Invoke) =
      copy(functions = functions.updatedWith(x.symbol) {
        case Some(ivks) => Some(ivks + application)
        case None       => Some(Set(application))
      })

    def witness(f: p.Function) = copy(resolvedFunctions = f :: resolvedFunctions)
  }

  given Monoid[Dependencies] = Monoid.instance(
    Dependencies(),
    (x, y) =>
      Dependencies(
        x.modules ++ y.modules,
        x.classes ++ y.classes,
        x.functions ++ y.functions,
        x.resolvedFunctions ++ y.resolvedFunctions
      )
  )

  // TODO rename to RemapContext
  case class RemapContext(
      root: Symbol,
      depth: Int = 0,                  // ref depth
      traces: List[Tree] = List.empty, // debug

      refs: Map[Symbol, p.Term] = Map.empty, // ident/select table
      names: Map[Symbol, Int] = Map.empty,
      invokeCaptures: Map[Symbol, List[p.Term]] = Map.empty,
      deps: Dependencies = Dependencies(),
      stmts: List[p.Stmt] = List.empty, // fn statements
      thisCls: Option[(ClassDef, p.Type.Struct)] = None
//      symbolDefMap: Map[Symbol, Definition] = Map.empty
  ) {
    infix def !!(t: Tree): RemapContext = copy(traces = t :: traces)
    def down(t: Tree): RemapContext     = !!(t).copy(depth = depth + 1)
    def named(tpe: p.Type): p.Named     = p.Named(s"_v${depth}", tpe)

    def mkName(s: Symbol): (String, RemapContext) = {
      def indexedName(ord: Int) = ord match {
        case 0 => s.name
        case n => s"${s.name}_$n"
      }
      names.get(s) match {
        case Some(n) => (indexedName(n), this)
        case None    =>
          // New symbol, look for name collisions if any and generate new name with a higher ordinal
          val newOrdinal = names
            .collect { case (k, n) if k.name == s.name => n }
            .maxOption
            .map(_ + 1)
            .getOrElse(0)
          (indexedName(newOrdinal), copy(names = names + (s -> newOrdinal)))
      }
    }

    // FIXME symbolDefMap should not required any more, it's for solving the non-existent issue
    // where definitions appear in implementation of a macro which can't happen user code.
//    def findDefTree(s: Symbol): Option[Definition] = symbolDefMap.get(s)
//    def withDefs(x: Tree): RemapContext            = withDefs(x :: Nil)
//    def withDefs(xs: List[Tree]): RemapContext = copy(symbolDefMap =
//      symbolDefMap ++
//        xs.flatMap(collectTree(_) {
//          case d: Definition => (d.symbol -> d) :: Nil
//          case _             => Nil
//        }).toMap
//    )

    def withInvokeCapture(s: Symbol, xs: List[p.Term]): RemapContext =
      copy(invokeCaptures = invokeCaptures + (s -> xs))

    def noStmts: RemapContext = copy(stmts = Nil)
    // def mark(s: p.Signature, d: DefDef) = copy(defs = defs + (s -> d))
    infix def ::=(xs: p.Stmt*): RemapContext = copy(stmts = stmts ++ xs)
//    infix def withFunction(xs: p.Function*): RemapContext                  = copy(functions = functions ++ xs)
    def replaceStmts(xs: Seq[p.Stmt]): RemapContext               = copy(stmts = xs.toList)
    def updateDeps(f: Dependencies => Dependencies): RemapContext = copy(deps = f(deps))

    def bindThis(x: ClassDef, tpe: p.Type.Struct): Result[RemapContext] = thisCls match {
      case None => updateDeps(_.witness(x, tpe)).copy(thisCls = Some(x -> tpe)).success
      case Some((oldSym, oldTpe)) if oldSym == x && oldTpe == tpe => this.success
      case Some((oldSym, oldTpe)) => s"Cannot witness different this type: $oldSym != $x ($oldTpe != $tpe)".fail
    }

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
