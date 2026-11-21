package polyregion.ast.pass

import polyregion.ast.{Log, PolyAST as p, *}

import scala.annotation.tailrec

// replaces a call to an abstract method with a dispatch on the receiver's host-written tag
// examples:
//   Shape.area(s)  with  Circle(tag 0), Rect(tag 1)
//     ->  if (s.#tag == 0) Circle.area(cast s) else if (s.#tag == 1) Rect.area(cast s) else assert
// edge cases:
//   candidate sets are lower bounds  ->  a class the frontend never saw carries a tag no arm tests, so the
//                                        trailing assert fires instead of a wrong arm running
//   callee absent from the table     ->  the invoke is untouched
//   receiver not a Select            ->  the invoke is untouched, there is no tag to read
//   Unit0 callee                     ->  the arms discard the result and the bound name is the unit constant
//   candidate with `args`            ->  the arm passes only those positions of the call's arguments, in that order
object Devirtualise {

  case class Candidate(receiver: p.Type.Struct, impl: p.Sym, tag: Int, args: Option[List[Int]] = None)

  def tagsOf(classes: Iterable[p.Sym]): Map[p.Sym, Int] = classes.toList.distinct.sortBy(_.fqcn).zipWithIndex.toMap

  // an abstract struct gets the tag, a word of slack and enough words to hold the largest candidate, so a by-value copy
  // of it keeps every byte of whichever candidate sits in it; each candidate gets the tag as its first member
  def decorate(
      defs: List[p.StructDef],
      candidates: Map[p.Sym, List[p.Sym]],
      tags: Map[p.Sym, Int]
  ): List[p.StructDef] = {
    val tag          = p.Named(p.Conventions.DispatchTag, p.Type.IntS32)
    val slack        = p.Named("#slack", p.Type.IntS32)
    val byName       = defs.map(d => d.name -> d).toMap
    def word(n: Int) = (n + 7) / 8 * 8
    def bytes(tpe: p.Type): Int = tpe match {
      case p.Type.Struct(name, _) =>
        word(byName.get(name).fold(8)(_.members.map(m => bytes(m.tpe)).sum) + (if (tags.contains(name)) 8 else 0))
      case p.Type.Arr(comp, n, _) => word(bytes(comp) * n)
      case p.Type.Ptr(_, _)       => 8
      case other                  => word(TypeOps.width(other).getOrElse(8))
    }
    def words(owner: p.Sym): Int =
      candidates(owner).flatMap(byName.get(_)).map(_.members.map(m => bytes(m.tpe)).sum).maxOption.getOrElse(0) / 8
    defs.map {
      case d if tags.contains(d.name) && !d.members.contains(tag) => d.copy(members = tag :: d.members)
      case d if candidates.contains(d.name) && !d.members.contains(tag) =>
        val pad =
          Option.when(words(d.name) > 0)(p.Named("#pad", p.Type.Arr(p.Type.IntS64, words(d.name), p.Type.Space.Global)))
        d.copy(members = tag :: slack :: pad.toList)
      case d => d
    }
  }

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
    else program.copy(entry = program.entry.map(rewrite(_)), functions = program.functions.map(rewrite(_)))

  private def dispatchable(e: p.Expr): Option[(p.Expr.Invoke, p.Term.Select, List[Candidate])] = e match {
    case ivk: p.Expr.Invoke =>
      for {
        callee <- ivk.calleeSym
        cs     <- candidates.get(callee)
        recv   <- receiverSelect(ivk)
      } yield (ivk, recv, cs)
    case _ => None
  }

  private def rewrite(f: p.Function): p.Function = {
    val fresh = Binders.freshNamer(f, "_devirt_")
    f.copy(body = Binders.mapStmts(f.body) {
      case p.Stmt.Var(name, Some(e), isMutable) if dispatchable(e).isDefined =>
        val (ivk, recv, cs) = dispatchable(e).get
        if (ivk.rtn == p.Type.Unit0)
          p.Stmt.Var(name, Some(p.Expr.Alias(p.Term.Unit0Const)), isMutable) :: dispatch(ivk, recv, cs, fresh, None)
        else {
          val result: p.Term.Select = p.Term.Select(name, Nil, name.tpe)
          p.Stmt.Var(name, None, isMutable = true) :: dispatch(ivk, recv, cs, fresh, Some(result))
        }
      case p.Stmt.Mut(target, e) if dispatchable(e).isDefined =>
        val (ivk, recv, cs) = dispatchable(e).get
        dispatch(ivk, recv, cs, fresh, Some(target))
      case p.Stmt.Return(e) if dispatchable(e).isDefined =>
        val (ivk, recv, cs) = dispatchable(e).get
        if (ivk.rtn == p.Type.Unit0)
          dispatch(ivk, recv, cs, fresh, None) :+ p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
        else {
          val result                = fresh(ivk.rtn)
          val select: p.Term.Select = p.Term.Select(result, Nil, result.tpe)
          p.Stmt.Var(result, None, isMutable = true) :: dispatch(ivk, recv, cs, fresh, Some(select)) :::
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
      into: Option[p.Term.Select]
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
      val as = recv.tpe match {
        case p.Type.Ptr(_, space) => p.Type.Ptr(c.receiver, space)
        case _                    => c.receiver
      }
      val (cast, target) =
        if (as == recv.tpe) (Nil, recv: p.Term)
        else {
          val n = fresh(as)
          (List(p.Stmt.Var(n, Some(p.Expr.Cast(recv, as)))), p.Term.Select(n, Nil, as): p.Term)
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
