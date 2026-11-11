package polyregion.ast.pass

import scala.collection.mutable.ListBuffer

import polyregion.ast.Traversal.*
import polyregion.ast.{Log, PolyAST as p, *, given}
import polyregion.ast.pass.TypeOps.pathType
import polyregion.ast.pass.TypeOps.width

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

  override def apply(program: p.Program, log: Log): p.Program = {
    val definitions = program.defs.map(definition => definition.name -> definition).toMap

    def storageType(select: p.Term.Select): p.Type = pathType(definitions, select.root.tpe, select.steps)

    def legalise(function: p.Function): p.Function = {
      val fresh = Binders.freshNamer(function, "#source_view_")

      def sameStorage(storage: p.Type, view: p.Type): Boolean = {
        def erase(t: p.Type): p.Type = t match {
          case p.Type.Ptr(component, _)    => p.Type.Ptr(erase(component), p.Type.Space.Private)
          case p.Type.Arr(component, n, _) => p.Type.Arr(erase(component), n, p.Type.Space.Private)
          case other                       => other
        }
        erase(storage) == erase(view)
      }

      def conversion(value: p.Term, from: p.Type, to: p.Type): p.Expr = (from, to) match {
        case _ if from == p.Type.Bool1 || to == p.Type.Bool1       => p.Expr.Cast(value, to)
        case (p.Type.Ptr(_, _), _) if sameStorage(from, to)        => p.Expr.Cast(value, to)
        case (p.Type.Arr(_, _, _), _) if sameStorage(from, to)     => p.Expr.Alias(value)
        case _ if width(from).nonEmpty && width(from) == width(to) => p.Expr.BitCast(value, to)
        case _ => throw IllegalArgumentException(s"source selection cannot view ${from.repr} as ${to.repr}")
      }

      def lowerTerm(term: p.Term, prefix: ListBuffer[p.Stmt]): p.Term = term.modifyAll[p.Term] {
        // SourceStorageLegalise reserves a byte for callable members in physical aggregates, while calls retain the
        // logical FnRef solely to select their statically known callee. The byte is never read as a runtime value.
        case select: p.Term.Select if select.tpe.isInstanceOf[p.Type.FnRef] => select
        case select: p.Term.Select =>
          val storage = storageType(select)
          if (storage == select.tpe) select
          else if (sameStorage(storage, select.tpe)) select.copy(tpe = storage)
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
        case mutation @ p.Stmt.Mut(name, expression) if name.tpe.isInstanceOf[p.Type.FnRef] =>
          val prefix  = ListBuffer.empty[p.Stmt]
          val lowered = lowerExpr(expression, prefix)
          prefix.toList :+ mutation.copy(expr = lowered)
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
        case nested @ (p.Stmt.Annotated(_, _, _) | p.Stmt.Try(_, _, _)) => Binders.lowerNested(lowerBlock)(nested)
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
