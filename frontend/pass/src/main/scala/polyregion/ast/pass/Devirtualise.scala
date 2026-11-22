package polyregion.ast.pass

import polyregion.ast.{Log, PolyAST as p, *}

import scala.annotation.tailrec

object Devirtualise {

  case class Candidate(
      receiver: p.Type.Struct,
      impl: p.Sym,
      tag: Int,
      args: Option[List[Int]] = None,
      implReceiver: Option[p.Type.Struct] = None
  )

  def tagsOf(classes: Iterable[p.Sym]): Map[p.Sym, Int] =
    classes.toList.distinct.sortBy(_.fqcn).zipWithIndex.map((c, i) => c -> (i + 1)).toMap

  private def word(n: Int): Int = (n + 7) / 8 * 8

  private class Sizes(defs: List[p.StructDef], candidates: Map[p.Sym, List[p.Sym]], tags: Map[p.Sym, Int]) {
    val byName           = defs.map(d => d.name -> d).toMap
    private val memo     = scala.collection.mutable.Map.empty[p.Sym, Int]
    private val visiting = scala.collection.mutable.Set.empty[p.Sym]

    def of(tpe: p.Type): Int = tpe match {
      case p.Type.Struct(name, _) =>
        memo.getOrElse(
          name, {
            if (!visiting.add(name)) throw IllegalArgumentException(s"${name.repr} holds itself by value")
            val size = structSize(name)
            visiting -= name
            memo(name) = size
            size
          }
        )
      case p.Type.Arr(comp, n, _) => word(of(comp) * n)
      case p.Type.Ptr(_, _)       => 8
      case other                  => word(TypeOps.width(other).getOrElse(8))
    }

    def membersOf(name: p.Sym): Int = byName.get(name).fold(0)(_.members.map(m => of(m.tpe)).sum)

    // the words an owner needs beyond its tag to hold its widest candidate
    def words(owner: p.Sym): Int =
      candidates(owner).filter(_ != owner).map(membersOf(_)).maxOption.getOrElse(0) / 8

    private def structSize(name: p.Sym): Int = (candidates.contains(name), tags.contains(name)) match {
      case (true, false) => 8 + 8 * words(name)
      case (true, true)  => 8 + math.max(word(membersOf(name)), 8 * words(name))
      case (false, true) => word(membersOf(name) + 8)
      case _             => word(membersOf(name))
    }
  }

  def decorate(
      defs: List[p.StructDef],
      candidates: Map[p.Sym, List[p.Sym]],
      tags: Map[p.Sym, Int]
  ): List[p.StructDef] = {
    val tag             = p.Named(p.Conventions.DispatchTag, p.Type.IntS32)
    val slack           = p.Named("#slack", p.Type.IntS32)
    val sizes           = Sizes(defs, candidates, tags)
    def pad(words: Int) = Option.when(words > 0)(p.Named("#pad", p.Type.Arr(p.Type.IntS64, words, p.Type.Space.Global)))
    defs.map {
      case d if d.members.contains(tag) => d
      case d if tags.contains(d.name) && candidates.contains(d.name) =>
        val spare = (sizes.words(d.name) * 8 - word(sizes.membersOf(d.name))) / 8
        d.copy(members = tag :: d.members ::: pad(spare).toList)
      case d if tags.contains(d.name)       => d.copy(members = tag :: d.members)
      case d if candidates.contains(d.name) => d.copy(members = tag :: slack :: pad(sizes.words(d.name)).toList)
      case d                                => d
    }
  }

  // the bytes a by-value struct occupies once every owner is decorated
  def sizeOf(defs: List[p.StructDef], tpe: p.Type): Int = Sizes(defs, Map.empty, Map.empty).of(tpe)

  def hostMembers(sdef: p.StructDef): List[p.Named] = sdef.members.filterNot(_.symbol.startsWith("#"))

  def noImplementation(callee: p.Sym): String =
    s"receiver has no implementation of ${callee.repr}: its class was not a known candidate"

  private def receiverSelect(ivk: p.Expr.Invoke): Option[p.Term.Select] = ivk.receiver.collect {
    case s: p.Term.Select =>
      s
  }
}

class Devirtualise(candidates: Map[p.Sym, List[Devirtualise.Candidate]]) extends ProgramPass {
  import Devirtualise.*

  override def apply(program: p.Program, log: Log): p.Program =
    if (candidates.isEmpty) program
    else
      program.copy(
        entry = program.entry.map(rewrite(_, program.defs)),
        functions = program.functions.map(rewrite(_, program.defs))
      )

  private val tagged: Map[p.Sym, Int] = candidates.values.flatten.map(c => c.receiver.name -> c.tag).toMap

  private object Tagged {
    def unapply(name: p.Named): Option[Int] = name.tpe match {
      case p.Type.Struct(cls, _) => tagged.get(cls)
      case _                     => None
    }
  }

  private def dispatchable(e: p.Expr): Option[(p.Expr.Invoke, p.Term.Select, List[Candidate])] = e match {
    case ivk: p.Expr.Invoke =>
      for {
        callee <- ivk.calleeSym
        cs     <- candidates.get(callee)
        recv   <- receiverSelect(ivk)
      } yield (ivk, recv, cs)
    case _ => None
  }

  private def rewrite(f: p.Function, defs: List[p.StructDef]): p.Function = {
    val fresh = Binders.freshNamer(f, "_devirt_")
    f.copy(body = Binders.mapStmts(f.body) {
      case v @ p.Stmt.Var(name @ Tagged(tag), None, _) =>
        val select: p.Term.Select =
          p.Term.Select(name, List(p.PathStep.Field(p.Conventions.DispatchTag)), p.Type.IntS32)
        List(v, p.Stmt.Mut(select, p.Expr.Alias(p.Term.IntS32Const(tag))))
      case p.Stmt.Var(name, Some(e), isMutable) if dispatchable(e).isDefined =>
        val (ivk, recv, cs) = dispatchable(e).get
        if (ivk.rtn == p.Type.Unit0)
          p.Stmt
            .Var(name, Some(p.Expr.Alias(p.Term.Unit0Const)), isMutable) :: dispatch(ivk, recv, cs, fresh, None, defs)
        else {
          val result: p.Term.Select = p.Term.Select(name, Nil, name.tpe)
          p.Stmt.Var(name, None, isMutable = true) :: dispatch(ivk, recv, cs, fresh, Some(result), defs)
        }
      case p.Stmt.Mut(target, e) if dispatchable(e).isDefined =>
        val (ivk, recv, cs) = dispatchable(e).get
        dispatch(ivk, recv, cs, fresh, Some(target), defs)
      case p.Stmt.Return(e) if dispatchable(e).isDefined =>
        val (ivk, recv, cs) = dispatchable(e).get
        if (ivk.rtn == p.Type.Unit0)
          dispatch(ivk, recv, cs, fresh, None, defs) :+ p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
        else {
          val result                = fresh(ivk.rtn)
          val select: p.Term.Select = p.Term.Select(result, Nil, result.tpe)
          p.Stmt.Var(result, None, isMutable = true) :: dispatch(ivk, recv, cs, fresh, Some(select), defs) :::
            List(p.Stmt.Return(p.Expr.Alias(select)))
        }
      case other => List(other)
    })
  }

  private def dispatch(
      ivk: p.Expr.Invoke,
      recv: p.Term.Select,
      cs: List[Candidate],
      fresh: p.Type => p.Named,
      into: Option[p.Term.Select],
      defs: List[p.StructDef]
  ): List[p.Stmt] = {
    val through = recv.tpe match {
      case p.Type.Ptr(p.Type.Struct(_, _), _) => recv.steps :+ p.PathStep.Deref
      case _                                  => recv.steps
    }
    val tag = fresh(p.Type.IntS32)
    val read = p.Stmt.Var(
      tag,
      Some(p.Expr.Alias(p.Term.Select(recv.root, through :+ p.PathStep.Field(p.Conventions.DispatchTag), tag.tpe)))
    )
    val tagSelect: p.Term.Select = p.Term.Select(tag, Nil, tag.tpe)

    def arm(c: Candidate): List[p.Stmt] = {
      recv.tpe match {
        case held: p.Type.Struct if sizeOf(defs, c.receiver) > sizeOf(defs, held) =>
          throw IllegalStateException(
            s"cannot dispatch ${ivk.calleeName.repr} on a ${held.name.repr} held by value: " +
              s"${c.receiver.name.repr} is larger, so copying it into the receiver would drop its state"
          )
        case _ =>
      }
      def viewAs(struct: p.Type.Struct): p.Type = recv.tpe match {
        case p.Type.Ptr(_, space) => p.Type.Ptr(struct, space)
        case _                    => struct
      }
      val (cast, target) = (c.receiver :: c.implReceiver.filter(_ != c.receiver).toList)
        .map(viewAs(_))
        .foldLeft((List.empty[p.Stmt], recv: p.Term)) {
          case (acc @ (_, from), as) if as == from.tpe => acc
          case ((stmts, from), as) =>
            val n = fresh(as)
            (stmts :+ p.Stmt.Var(n, Some(p.Expr.Cast(from, as))), p.Term.Select(n, Nil, as))
        }
      val call =
        p.Expr.Invoke(p.Type.FnRef(c.impl), ivk.tpeArgs, Some(target), c.args.fold(ivk.args)(_.map(ivk.args)), ivk.rtn)
      cast ::: List(into.fold(p.Stmt.Var(fresh(ivk.rtn), Some(call)))(p.Stmt.Mut(_, call)))
    }

    val raise = p.Stmt.Var(
      fresh(p.Type.Unit0),
      Some(
        p.Expr.SpecOp(
          p.Spec.Assert(
            p.Term.IntU32Const(p.Enums.AssertCode.Error.value),
            p.Term.StringConst(noImplementation(ivk.calleeName))
          )
        )
      )
    )

    @tailrec def chain(remaining: List[Candidate], tail: List[p.Stmt]): List[p.Stmt] = remaining match {
      case Nil => tail
      case c :: rest =>
        val hit = fresh(p.Type.Bool1)
        chain(
          rest,
          List(
            p.Stmt.Var(hit, Some(p.Expr.IntrOp(p.Intr.LogicEq(tagSelect, p.Term.IntS32Const(c.tag))))),
            p.Stmt.Cond(p.Term.Select(hit, Nil, hit.tpe), arm(c), tail)
          )
        )
    }

    read :: chain(cs.reverse, List(raise))
  }
}
