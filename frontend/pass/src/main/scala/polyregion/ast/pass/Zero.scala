package polyregion.ast.pass

import polyregion.ast.PolyAST as p

private[pass] object Zero {

  def term(tpe: p.Type): Option[p.Term] = tpe match {
    case p.Type.Float16          => Some(p.Term.Float16Const(0f))
    case p.Type.Float32          => Some(p.Term.Float32Const(0f))
    case p.Type.Float64          => Some(p.Term.Float64Const(0d))
    case p.Type.IntU8            => Some(p.Term.IntU8Const(0))
    case p.Type.IntU16           => Some(p.Term.IntU16Const(0))
    case p.Type.IntU32           => Some(p.Term.IntU32Const(0))
    case p.Type.IntU64           => Some(p.Term.IntU64Const(0L))
    case p.Type.IntS8            => Some(p.Term.IntS8Const(0))
    case p.Type.IntS16           => Some(p.Term.IntS16Const(0))
    case p.Type.IntS32           => Some(p.Term.IntS32Const(0))
    case p.Type.IntS64           => Some(p.Term.IntS64Const(0L))
    case p.Type.Bool1            => Some(p.Term.Bool1Const(false))
    case p.Type.Ptr(comp, space) => Some(p.Term.NullPtrConst(comp, space, p.Region.Opaque))
    case _                       => None
  }

  def one(tpe: p.Type): Option[p.Term] = tpe match {
    case p.Type.IntU8  => Some(p.Term.IntU8Const(1))
    case p.Type.IntU16 => Some(p.Term.IntU16Const(1))
    case p.Type.IntU32 => Some(p.Term.IntU32Const(1))
    case p.Type.IntU64 => Some(p.Term.IntU64Const(1L))
    case p.Type.IntS8  => Some(p.Term.IntS8Const(1))
    case p.Type.IntS16 => Some(p.Term.IntS16Const(1))
    case p.Type.IntS32 => Some(p.Term.IntS32Const(1))
    case p.Type.IntS64 => Some(p.Term.IntS64Const(1L))
    case _             => None
  }

  def members(defs: Map[p.Sym, p.StructDef], tpe: p.Type): Option[List[(String, p.Type)]] = tpe match {
    case p.Type.Struct(name, args) =>
      defs.get(name).filter(d => !d.isUnion && d.tpeVars.size == args.size).map { d =>
        val env = d.tpeVars.map(_.name).zip(args).toMap
        d.members.map(m => m.symbol -> Binders.substType(m.tpe, env))
      }
    case _ => None
  }

  def into(defs: Map[p.Sym, p.StructDef], target: p.Term.Select): Option[List[p.Stmt]] =
    term(target.tpe) match {
      case Some(zero) => Some(List(p.Stmt.Mut(target, p.Expr.Alias(zero))))
      case None =>
        members(defs, target.tpe).flatMap { ms =>
          ms.foldLeft(Option(List.empty[p.Stmt])) { case (acc, (name, tpe)) =>
            for {
              done <- acc
              next <- into(defs, p.Term.Select(target.root, target.steps :+ p.PathStep.Field(name), tpe))
            } yield done ::: next
          }
        }
    }

  def declare(defs: Map[p.Sym, p.StructDef], named: p.Named, mutable: Boolean): Option[List[p.Stmt]] =
    term(named.tpe) match {
      case Some(zero) => Some(List(p.Stmt.Var(named, Some(p.Expr.Alias(zero)), mutable)))
      case None       => into(defs, p.Term.Select(named, Nil, named.tpe)).map(p.Stmt.Var(named, None) :: _)
    }
}
