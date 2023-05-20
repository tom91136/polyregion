package polyregion.ast.pass

import cats.syntax.all.*
import polyregion.ast.Traversal.*
import polyregion.ast.{PolyAst as p, *, given}

object DynamicDispatchPass extends ProgramPass {

  override def apply(program: p.Program, log: Log): p.Program = {

    val (polymorphic, monomorphic) =
      program.defs.partition(c => c.parents.nonEmpty || program.defs.exists(_.parents.contains(c.name)))

    val withClassTag = polymorphic.map(s =>
      s.copy(members = p.StructMember(p.Named("_#cls", p.Type.IntS32), isMutable = false) :: s.members)
    )

    def clsTag(x: p.Type.Struct) = x.name.fqn.mkString(".").hashCode

    def clsFns(tpe: p.Type.Struct): List[p.Function] = {
      val erasedTpe = tpe.erased
      program.functions.filter(_.receiver.exists(_.named.tpe.erased == erasedTpe))
    }

    // 2. Synthesize the dynamic dispatch method
    val fs = program.defs.flatMap { c =>
      // Find all subclasses first
      val children = program.defs.filter(_.parents.contains(c.name))
      log.info(s"Children for ${c.repr}", children.map(_.repr)*)
      log.info(s"Fns for ${c.repr}", clsFns(c.tpe).map(_.repr)*)
      log.info(
        s"overriding fns for ${c.repr}",
        clsFns(c.tpe).map { f =>

          val simpleName = f.name.last
          val overridingFns = children.flatMap { c =>
            val recvTpe = c.tpe
            clsFns(recvTpe) // .filter(_.name.last == simpleName).map(recvTpe -> _)
          }

          s"${f.repr} => ${overridingFns.map(_.repr).mkString(" ; ")}"

        }*
      )

      // Then, for each method in the base class, see if it has overrides from any subclass
      clsFns(c.tpe).flatMap { baseFn =>

        val simpleName = baseFn.name.last
        val overridingFns = children.flatMap { c =>
          val recvTpe = c.tpe
          clsFns(recvTpe).filter(_.name.last == simpleName).map(recvTpe -> _)
        }
        // If we do find any, synthesise the dynamic dispatch function
        val clsTagArg = p.Arg(p.Named("cls", p.Type.IntS32))
        val objArg    = p.Arg(p.Named("obj", baseFn.receiver.get.named.tpe))

        if (overridingFns.isEmpty) Nil
        else {

          val branches = Function.chain(((c.tpe, baseFn) :: overridingFns).zipWithIndex.map { case ((_, fn), i) =>
            val tpe = fn.receiver.get.named.tpe.asInstanceOf[p.Type.Struct]
            (elseBr: List[p.Stmt]) =>
              p.Stmt.Cond(
                cond = p.Expr.IntrOp(
                  p.Intr.LogicEq(p.Term.Select(Nil, clsTagArg.named), p.Term.IntS32Const(clsTag(tpe)))
                ),
                trueBr =
                  p.Stmt.Var(p.Named(s"recv_$i", tpe), Some(p.Expr.Cast(p.Term.Select(Nil, objArg.named), tpe))) ::
                    p.Stmt.Return(
                      p.Expr.Invoke(
                        name = fn.name,
                        tpeArgs = baseFn.tpeVars.map(p.Type.Var(_)),
                        receiver = Some(p.Term.Select(Nil, p.Named(s"recv_$i", tpe))),
                        args = baseFn.args.map(arg => p.Term.Select(Nil, arg.named)),
                        captures = Nil,
                        rtn = baseFn.rtn
                      )
                    ) :: Nil,
                falseBr = elseBr
              ) :: Nil
          })

          (
            c.tpe,
            baseFn.name,
            p.Function(
              name = p.Sym(s"$simpleName^"),
              tpeVars = baseFn.tpeVars,
              receiver = None,
              args = clsTagArg :: objArg :: baseFn.args,
              moduleCaptures = Nil,
              termCaptures = Nil,
              rtn = baseFn.rtn,
              body = branches(
                p.Stmt.Comment("unseen class, assert")
                  :: p.Stmt.Return(p.Expr.SpecOp(p.Spec.Assert))
                  :: Nil
              )
            )
          ) :: Nil
        }
      }
    }

    val lut             = fs.map((tpe, name, fn) => (tpe.erased: p.Type, name.last) -> fn).toMap
    val polymorphicSyms = withClassTag.map(_.name).toSet

    // Ensure synthetic class tag fields are initialised to the correct constant
    def insertClassTags(f: p.Function) = f.modifyAll[p.Stmt] {
      case stmt @ p.Stmt.Var(local @ p.Named(_, s @ p.Type.Struct(name, _, _, _)), None)
          if polymorphicSyms.contains(name) =>
        val rhs        = p.Term.IntS32Const(clsTag(s))
        val clsSelect  = p.Term.Select(local :: Nil, p.Named("_#cls", p.Type.IntS32))
        val setClsHash = p.Stmt.Mut(clsSelect, p.Expr.Alias(rhs), false)
        p.Stmt.Block(stmt :: setClsHash :: Nil)
      case x => x
    }

    log.info(s"LUT: ", lut.map { case ((t, s), v) => s"$t($s) => ${v.signatureRepr}" }.toList*)

    def replaceDispatches(f: p.Function) = f.modifyAll[p.Expr] {
      case ivk @ p.Expr.Invoke(name, tpeArgs, Some(recv: p.Term.Select), args, captures, rtn) =>
        log.info(s"Replace: ${ivk.repr}", lut.get((recv.tpe, name.last)).toString)

        def isSuperCall(t: p.Type) = (t, f.receiver) match {
          case (p.Type.Struct(clsName, _, _, _), Some(p.Arg(p.Named(_, p.Type.Struct(_, _, _, parents)), _))) =>
            parents.contains(clsName) && f.name.last == name.last
          case _ => false
        }

        lut.get((recv.tpe.erased, name.last)) match {
          case Some(dynamicDispatchFn) =>
            if (isSuperCall(recv.tpe.erased)) ivk
            else {
              val clsTagSelect = p.Term.Select(recv.init :+ recv.last, p.Named("_#cls", p.Type.IntS32))
              p.Expr.Invoke(dynamicDispatchFn.name, tpeArgs, None, clsTagSelect :: recv :: args, captures, rtn)
            }
          case _ => ivk
        }
      case x => x
    }

    // Base, A, B

    program.copy(
      entry = replaceDispatches(insertClassTags(program.entry)),
      functions = program.functions.map(f => replaceDispatches(insertClassTags(f))) ::: fs.map(_._3),
      defs = monomorphic ::: withClassTag
    )
    // program
  }

}
