package polyregion.ast.pass

import scala.collection.mutable.ListBuffer

import polyregion.ast.Traversal.*
import polyregion.ast.{Log, PolyAST as p, *, given}

// makes the physical type of every source lvalue explicit. a Select is first typed as its declared storage;
// consumers that intentionally view equal-width scalar bits through another type receive an explicit BitCast.
// Boolean views use numeric Cast because their source representation is a truth value, not a bit pattern
// examples:
//   struct S { i32 bits }; select(s.bits): f32  ->  bitcast(select(s.bits): i32, f32)
//   select(s.value): i32 = rhs, value: f32     ->  select(s.value): f32 = bitcast(rhs, f32)
// edge cases:
//   differently-sized or aggregate views  ->  rejected instead of guessed by the source renderer
//   reinterpretation in a loop condition  ->  recomputed on every iteration
object SourceSelectionLegalise extends ProgramPass {

  override def phase: p.Pass.Phase = p.Pass.Phase.PostMono

  private def width(tpe: p.Type): Option[Int] = tpe match {
    case p.Type.Bool1 | p.Type.IntU8 | p.Type.IntS8     => Some(1)
    case p.Type.IntU16 | p.Type.IntS16 | p.Type.Float16 => Some(2)
    case p.Type.IntU32 | p.Type.IntS32 | p.Type.Float32 => Some(4)
    case p.Type.IntU64 | p.Type.IntS64 | p.Type.Float64 => Some(8)
    case _                                              => None
  }

  override def apply(program: p.Program, log: Log): p.Program = {
    val definitions = program.defs.map(definition => definition.name -> definition).toMap

    def memberType(owner: p.Type, field: String): p.Type = {
      val structure = owner match {
        case value: p.Type.Struct                => value
        case p.Type.Ptr(value: p.Type.Struct, _) => value
        case other => throw IllegalArgumentException(s"field $field selected on non-struct type ${other.repr}")
      }
      definitions
        .get(structure.name)
        .flatMap(_.members.find(_.symbol == field))
        .map(_.tpe)
        .getOrElse(throw IllegalArgumentException(s"field $field not found on ${structure.name.fqcn}"))
    }

    def storageType(select: p.Term.Select): p.Type = select.steps.foldLeft(select.root.tpe) {
      case (current, p.PathStep.Field(field)) => memberType(current, field)
      case (p.Type.Ptr(component, _), p.PathStep.Deref | _: p.PathStep.Index | _: p.PathStep.IndexDyn) => component
      case (p.Type.Arr(component, _, _), _: p.PathStep.Index | _: p.PathStep.IndexDyn)                 => component
      case (current, step) => throw IllegalArgumentException(s"cannot apply ${step.repr} to ${current.repr}")
    }

    def legalise(function: p.Function): p.Function = {
      val occupied = scala.collection.mutable.Set.from(
        (function.receiver.toList ::: function.args ::: function.moduleCaptures ::: function.termCaptures)
          .map(_.named.symbol) :::
          function.collectAll[p.Stmt].flatMap {
            case p.Stmt.Var(name, _, _)                 => List(name.symbol)
            case p.Stmt.ForRange(induction, _, _, _, _) => List(induction.symbol)
            case _                                      => Nil
          }
      )
      val candidates = Iterator.from(0).map(index => s"#source_view_$index")
      def fresh(tpe: p.Type): p.Named = {
        val symbol = candidates.find(occupied.add).get
        p.Named(symbol, tpe)
      }

      def conversion(value: p.Term, from: p.Type, to: p.Type): p.Expr =
        if (from == p.Type.Bool1 || to == p.Type.Bool1) p.Expr.Cast(value, to)
        else if (width(from).nonEmpty && width(from) == width(to)) p.Expr.BitCast(value, to)
        else throw IllegalArgumentException(s"source selection cannot view ${from.repr} as ${to.repr}")

      def lowerTerm(term: p.Term, prefix: ListBuffer[p.Stmt]): p.Term = term.modifyAll[p.Term] {
        case select: p.Term.Select =>
          val storage = storageType(select)
          if (storage == select.tpe) select
          else {
            val result = fresh(select.tpe)
            prefix += p.Stmt.Var(result, Some(conversion(select.copy(tpe = storage), storage, select.tpe)))
            p.Term.Select(result, Nil, result.tpe)
          }
        case other => other
      }

      def lowerExpr(expression: p.Expr, prefix: ListBuffer[p.Stmt]): p.Expr =
        expression.modifyAll[p.Term](lowerTerm(_, prefix))

      def storeExpr(expression: p.Expr, from: p.Type, to: p.Type, prefix: ListBuffer[p.Stmt]): p.Expr = {
        val lowered = lowerExpr(expression, prefix)
        if (from == to) lowered
        else {
          val value = lowered match {
            case p.Expr.Alias(term) => term
            case other =>
              val temporary = fresh(from)
              prefix += p.Stmt.Var(temporary, Some(other))
              p.Term.Select(temporary, Nil, from)
          }
          conversion(value, from, to)
        }
      }

      def lowerBlock(body: List[p.Stmt]): List[p.Stmt] = body.flatMap {
        case variable @ p.Stmt.Var(_, expression, _) =>
          val prefix  = ListBuffer.empty[p.Stmt]
          val lowered = expression.map(lowerExpr(_, prefix))
          prefix.toList :+ variable.copy(expr = lowered)
        case mutation @ p.Stmt.Mut(name, expression) =>
          val prefix  = ListBuffer.empty[p.Stmt]
          val storage = storageType(name)
          val lowered = storeExpr(expression, name.tpe, storage, prefix)
          prefix.toList :+ mutation.copy(name = name.copy(tpe = storage), expr = lowered)
        case update @ p.Stmt.Update(lhs, index, value) =>
          val prefix  = ListBuffer.empty[p.Stmt]
          val storage = storageType(lhs)
          if (storage != lhs.tpe)
            throw IllegalArgumentException(s"source update target cannot view ${storage.repr} as ${lhs.tpe.repr}")
          val loweredIndex = lowerTerm(index, prefix)
          val loweredValue = lowerTerm(value, prefix)
          prefix.toList :+ update.copy(lhs = lhs.copy(tpe = storage), idx = loweredIndex, value = loweredValue)
        case loop @ p.Stmt.While(condition, body) =>
          val prefix  = ListBuffer.empty[p.Stmt]
          val lowered = lowerTerm(condition, prefix)
          val inner   = lowerBlock(body)
          if (prefix.isEmpty) List(loop.copy(cond = lowered, body = inner))
          else
            List(
              loop.copy(
                cond = p.Term.Bool1Const(true),
                body = prefix.toList :+ p.Stmt.Cond(lowered, inner, List(p.Stmt.Break))
              )
            )
        case loop @ p.Stmt.ForRange(induction, lb, ub, step, body) =>
          val prefix      = ListBuffer.empty[p.Stmt]
          val loweredLb   = lowerTerm(lb, prefix)
          val loweredUb   = lowerTerm(ub, prefix)
          val loweredStep = lowerTerm(step, prefix)
          if (prefix.nonEmpty)
            throw IllegalArgumentException("source selection reinterpretation in a for-range bound is unsupported")
          List(loop.copy(lbIncl = loweredLb, ubExcl = loweredUb, step = loweredStep, body = lowerBlock(body)))
        case branch @ p.Stmt.Cond(condition, trueBr, falseBr) =>
          val prefix  = ListBuffer.empty[p.Stmt]
          val lowered = lowerTerm(condition, prefix)
          prefix.toList :+ branch.copy(cond = lowered, trueBr = lowerBlock(trueBr), falseBr = lowerBlock(falseBr))
        case returned @ p.Stmt.Return(value) =>
          val prefix  = ListBuffer.empty[p.Stmt]
          val lowered = lowerExpr(value, prefix)
          prefix.toList :+ returned.copy(value = lowered)
        case annotated @ p.Stmt.Annotated(inner, _, _) =>
          lowerBlock(List(inner)) match {
            case init :+ last => init :+ annotated.copy(inner = last)
            case Nil          => Nil
            case other        => other
          }
        case tried @ p.Stmt.Try(body, handlers, fin) =>
          List(
            tried.copy(
              body = lowerBlock(body),
              handlers = handlers.map(h => h.copy(body = lowerBlock(h.body))),
              fin = lowerBlock(fin)
            )
          )
        case raised @ p.Stmt.Raise(value, _, cleanup) =>
          val prefix  = ListBuffer.empty[p.Stmt]
          val lowered = lowerTerm(value, prefix)
          prefix.toList :+ raised.copy(value = lowered, cleanup = lowerBlock(cleanup))
        case other => List(other)
      }

      function.copy(body = lowerBlock(function.body))
    }

    program.copy(entry = program.entry.map(legalise), functions = program.functions.map(legalise))
  }
}
