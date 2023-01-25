package polyregion.ast.pass

import cats.syntax.all.*
import polyregion.ast.Traversal.*
import polyregion.ast.{PolyAst as p, *, given}

object DynamicDispatchPass extends ProgramPass {

  inline def run(f: p.Function): p.Function =
    // TODO
    f

  override def apply(program: p.Program, log: Log): p.Program = {

    val (monomorphic, polymorphic) = program.defs.partition(_.parents.isEmpty)

    val withClassTag = polymorphic.map(s =>
      s.copy(members = p.StructMember(p.Named("_#cls", p.Type.Int), isMutable = false) :: s.members)
    )

    val polymorphicSyms = withClassTag.map(_.name).toSet

    def clsTag(x: p.Type.Struct) = x.name.fqn.mkString(".").hashCode

    // 1. Ensure synthetic class tag fields are initialised to the correct constant
    program.functions.map(f =>
      f.body.flatMap {
        case stmt @ p.Stmt.Var(local @ p.Named(_, s @ p.Type.Struct(name, _, _, _)), None)
            if polymorphicSyms.contains(name) =>
          val rhs        = p.Term.IntConst(clsTag(s))
          val setClsHash = p.Stmt.Mut(p.Term.Select(Nil, local), p.Expr.Alias(rhs), false)
          stmt :: setClsHash :: Nil
        case x => x :: Nil
      }
    )

    def clsFns(tpe: p.Type.Struct): List[p.Function] = program.functions.filter(_.receiver.exists(_.tpe == tpe))

    // 2. Synthesize the dynamic dispatch method
    val fs = program.defs.flatMap { c =>
      // Find all subclasses first
      val children = program.defs.filter(_.parents.contains(c.name))
      // Then, for each method in the base class, see if it has overrides from any subclass
      clsFns(c.tpe).map { baseFn =>
        val simpleName = baseFn.name.last
        val overridingFns = children.flatMap { c =>
          val recvTpe = c.tpe
          clsFns(recvTpe).filter(_.name.last == simpleName).map(recvTpe -> _)
        }
        // If we do find any, synthesise the dynamic dispatch function
        val clsTagArg = p.Named("cls", p.Type.Int)
        val objArg    = p.Named("obj", c.tpe)

        val branches = Function.chain(((c.tpe, baseFn) :: overridingFns).map { (tpe, fn) => (elseBr: p.Stmt) =>
          p.Stmt.Cond(
            cond = p.Expr.BinaryIntrinsic(
              p.Term.Select(Nil, clsTagArg),
              p.Term.IntConst(clsTag(tpe)),
              p.BinaryIntrinsicKind.LogicEq,
              p.Type.Bool
            ),
            trueBr = p.Stmt.Var(p.Named("recv", tpe), Some(p.Expr.Cast(p.Term.Select(Nil, objArg), tpe))) ::
              p.Stmt.Return(
                p.Expr.Invoke(
                  name = fn.name,
                  tpeArgs = baseFn.tpeVars.map(p.Type.Var(_)),
                  receiver = Some(p.Term.Select(Nil, p.Named("recv", tpe))),
                  args = baseFn.args.map(p.Term.Select(Nil, _)),
                  captures = Nil,
                  rtn = baseFn.rtn
                )
              ) :: Nil,
            falseBr = elseBr :: Nil
          )
        })(p.Stmt.Comment("Assert"))

        p.Function(
          name = p.Sym(s"$simpleName^"),
          tpeVars = baseFn.tpeVars,
          receiver = None,
          args = clsTagArg :: objArg :: baseFn.args,
          moduleCaptures = Nil,
          termCaptures = Nil,
          rtn = baseFn.rtn,
          body = branches :: Nil
        )
      }
    }

    println(fs.map(_.repr).mkString("\n======\n"))

    // Base, A, B

    program.copy(entry = run(program.entry))
  }

}
