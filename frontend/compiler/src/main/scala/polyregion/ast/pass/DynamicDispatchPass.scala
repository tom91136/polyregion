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
      clsFns(c.tpe).flatMap { baseFn =>
        val simpleName = baseFn.name.last
        val overridingFns = children.flatMap { c =>
          val recvTpe = c.tpe
          clsFns(recvTpe).filter(_.name.last == simpleName).map(recvTpe -> _)
        }
        // If we do find any, synthesise the dynamic dispatch function
        val clsTagArg = p.Named("cls", p.Type.Int)
        val objArg    = p.Named("obj", c.tpe)

        if(overridingFns.isEmpty) Nil
        else {

          val branches = Function.chain(((c.tpe, baseFn) :: overridingFns).zipWithIndex.map { case ( (tpe, fn), i) => (elseBr: List[p.Stmt]) =>
            p.Stmt.Cond(
              cond = p.Expr.BinaryIntrinsic(
                p.Term.Select(Nil, clsTagArg),
                p.Term.IntConst(clsTag(tpe)),
                p.BinaryIntrinsicKind.LogicEq,
                p.Type.Bool
              ),
              trueBr = p.Stmt.Var(p.Named(s"recv_$i", tpe), Some(p.Expr.Cast(p.Term.Select(Nil, objArg), tpe))) ::
                       p.Stmt.Return(
                         p.Expr.Invoke(
                           name = fn.name,
                           tpeArgs = baseFn.tpeVars.map(p.Type.Var(_)),
                           receiver = Some(p.Term.Select(Nil, p.Named(s"recv_$i", tpe))),
                           args = baseFn.args.map(p.Term.Select(Nil, _)),
                           captures = Nil,
                           rtn = baseFn.rtn
                         )
                       ) :: Nil,
              falseBr = elseBr
            ) :: Nil
          })

          val assertRtn = baseFn.rtn match {
            case p.Type.Float                                => p.Term.FloatConst(0)
            case p.Type.Double                               => p.Term.DoubleConst(0)
            case p.Type.Bool                                 => p.Term.BoolConst(true)
            case p.Type.Byte                                 => p.Term.ByteConst(0)
            case p.Type.Char                                 => p.Term.CharConst(0)
            case p.Type.Short                                => p.Term.ShortConst(0)
            case p.Type.Int                                  => p.Term.IntConst(0)
            case p.Type.Long                                 => p.Term.LongConst(0)
            case p.Type.Unit                                 => p.Term.UnitConst
            case p.Type.Nothing                              => ???
            case p.Type.Struct(name, tpeVars, args, parents) => ???
            case p.Type.Array(component)                     => ???
            case p.Type.Var(name)                            => ???
            case p.Type.Exec(tpeVars, args, rtn)             => ???
          }

          p.Function(
            name = p.Sym(s"$simpleName^"),
            tpeVars = baseFn.tpeVars,
            receiver = None,
            args = clsTagArg :: objArg :: baseFn.args,
            moduleCaptures = Nil,
            termCaptures = Nil,
            rtn = baseFn.rtn,
            body = branches(p.Stmt.Comment("Assert") :: p.Stmt.Return(p.Expr.Alias(assertRtn)) :: Nil)
          ) :: Nil
        }
      }
    }



    // Base, A, B

    program.copy(entry = run(program.entry), functions = program.functions ::: fs)
    // program
  }

}
