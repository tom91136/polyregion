package polyregion.ast.pass

import cats.data.EitherT
import cats.syntax.all.*
import polyregion.ast.{PolyAst as p, *, given}
import polyregion.ast.Traversal.*

import scala.annotation.tailrec
import cats.syntax.validated

object VerifyPass {

  case class Context(declared: Set[p.Named] = Set.empty, errors: List[String] = Nil) {
    def |+|(y: Context): Context = copy(declared ++ y.declared, errors ::: y.errors)
    def +(n: p.Named): Context   = copy(declared + n)

    def ~(error: String): Context = copy(errors = error :: errors)
    def !!(n: p.Named, tree: String): Context =
      if (declared.contains(n)) this
      else {
        // ok, see if the same var is defined with another type
        (declared.find(_.symbol == n.symbol), n.tpe) match {
          // case (Some(nn @ p.Named(_, p.Type.Struct(_, _, _, parents))), p.Type.Struct(name, _, _, _))
          //     if parents.contains(name) =>
          //   this // Subtypes are ok
          case (Some(existing), _) =>
            copy(errors =
              s"$tree uses the variable ${n.repr}, " +
                s"but its type was already defined as ${existing.tpe.repr}, " +
                s"variables defined up to point: \n\t${declared.mkString("\n\t")}" :: errors
            )
          case (None, _) =>
            copy(errors =
              s"$tree uses the unseen variable ${n.repr}, " +
                s"variables defined up to point: \n\t${declared.mkString("\n\t")}" :: errors
            )
        }
      }
  }

  def validateSingle(
      f: p.Function,
      defs: List[p.StructDef],
      fs: List[p.Function],
      verifyFunction: Boolean
  ): List[String] = {

    val sdefLUT = defs.map(x => x.name -> x).toMap

    def validateTerm(c: Context, t: p.Term): Context = t match {
      case p.Term.Select(Nil, local) => c !! (local, t.repr)
      case p.Term.Select(local :: rest, last) =>
        (rest :+ last)
          .foldLeft((c !! (local, t.repr), local.tpe)) { case ((acc, tpe), n) =>
            val c0 = tpe match {
              case s @ p.Type.Struct(name, tpeVars, args, _) =>
                sdefLUT.get(name) match {
                  case None =>
                    acc ~ s"Unknown struct type ${name.repr} in `${t.repr}`, known structs: \n${sdefLUT.map(_._2).map(_.repr).mkString("\n")}"
                  case Some(sdef) =>
                    if (sdef.tpeVars != tpeVars) {
                      acc ~ s"Struct def ${sdef.repr} and struct type ${s.repr} has different type variables"
                    } else {

                      val apTable = tpeVars.zip(args).toMap

                      println(apTable.toString + " " + sdef.repr + " " + s.repr)

                      sdef.members
                        .map(x =>
                          x.modifyAll[p.Type](_.mapLeaf {
                            case p.Type.Var(name) => apTable(name)
                            case x                => x
                          })
                        )
                        .filter(_.named == n) match {
                        case _ :: Nil => acc
                        case Nil =>
                          acc ~ s"Struct type ${sdef.repr} does not contain member ${n.repr} in `${t.repr} r={${defs}}`"
                        case _ => acc ~ s"Struct type ${sdef.repr} contains multiple members of $n in `${t.repr}`"
                      }
                    }
                }
              case illegal => acc ~ s"Cannot select member ${n} from a non-struct type $illegal in `${t.repr}`"
            }
            (c0, n.tpe)
          }
          ._1
      case _ => c
    }

    def validateExpr(c: Context, e: p.Expr): Context = e match {
      // case p.Expr.NullaryIntrinsic(kind, rtn)    => c
      // case p.Expr.UnaryIntrinsic(lhs, kind, rtn) => validateTerm(c, lhs)
      // case p.Expr.BinaryIntrinsic(lhs, rhs, kind, rtn) =>
      //   (validateTerm(_: Context, lhs)).andThen(validateTerm(_, rhs))(c)
      case p.Expr.SpecOp(op) =>
        val c0 = op.terms.foldLeft(c)(validateTerm(_, _))
        op.overloads.filter { o =>
          op.terms.map(_.tpe).zip(o.args).forall(_ == _) && op.tpe == o.rtn
        } match {
          case Nil      => c0 ~ s"No matching overload for intrinsic ${op}"
          case x :: Nil => c0
          case xs       => c0 ~ s"Multiple matching overload for intrinsic ${op}:\n${xs}"
        }
      case p.Expr.IntrOp(op) =>
        val c0 = op.terms.foldLeft(c)(validateTerm(_, _))
        op.overloads.filter { o =>
          op.terms.map(_.tpe).zip(o.args).forall(_ == _) && op.tpe == o.rtn
        } match {
          case Nil      => c0 ~ s"No matching overload for intrinsic ${op}"
          case x :: Nil => c0
          case xs       => c0 ~ s"Multiple matching overload for intrinsic ${op}:\n${xs}"
        }
      case p.Expr.MathOp(op) =>
        val c0 = op.terms.foldLeft(c)(validateTerm(_, _))
        op.overloads.filter { o =>
          op.terms.map(_.tpe).zip(o.args).forall(_ == _) && op.tpe == o.rtn
        } match {
          case Nil      => c0 ~ s"No matching overload for intrinsic ${op}"
          case x :: Nil => c0
          case xs       => c0 ~ s"Multiple matching overload for intrinsic ${op}:\n${xs}"
        }
      case p.Expr.Cast(from, as) =>
        val c0 = validateTerm(c, from)
        (from.tpe, as) match {
          case (from, as) if from == as => c0
          case (p.Type.Struct(_, _, _, parents), p.Type.Struct(name, _, _, _)) if parents.contains(name) =>
            c0 // safe upcase
          case (p.Type.Struct(name, _, _, _), p.Type.Struct(_, _, _, parents)) if parents.contains(name) =>
            c0 // unsafe downcast
          case (from, as) => c0 ~ s"Cannot cast unrelated type ${from.repr} to ${as.repr}: ${e.repr}"
        }
      case p.Expr.Alias(ref) => validateTerm(c, ref)
      case p.Expr.Invoke(name, tpeArgs, receiver, args, captures, rtn) =>
        val c0 = receiver.map(validateTerm(c, _)).getOrElse(c)
        val c1 = args.foldLeft(c0)(validateTerm(_, _))
        val c2 = captures.foldLeft(c1)(validateTerm(_, _))
        c2
      case p.Expr.Index(lhs, idx, component) => (validateTerm(_: Context, lhs)).andThen(validateTerm(_, idx))(c)
      case p.Expr.RefTo(lhs, idx, component) =>
        idx match {
          case Some(idx) => (validateTerm(_: Context, lhs)).andThen(validateTerm(_, idx))(c)
          case None      => (validateTerm(_: Context, lhs))(c)
        }
      case p.Expr.Alloc(witness, size) => validateTerm(c, size)
    }

    def validateStmt(c: Context, s: p.Stmt): Context = s match {
      case p.Stmt.Block(xs)  => xs.foldLeft(c)(validateStmt(_, _))
      case p.Stmt.Comment(_) => c
      case p.Stmt.Var(name, expr) =>
        val c0 = expr.map(e => validateExpr(_: Context, e)).getOrElse(identity[Context])(c + name)
        expr match {
          case Some(rhs) if rhs.tpe != name.tpe =>
            (name.tpe, rhs.tpe) match {
              case (p.Type.Struct(name, _, _, _), p.Type.Struct(_, _, _, parents)) if parents.contains(name) => c0
              case _ => c0 ~ s"Var declaration of incompatible type ${rhs.tpe.repr} != ${name.tpe.repr}: ${s.repr}"
            }
          case _ => c0
        }
      case p.Stmt.Mut(name, expr, copy) =>
        val c0 = (validateTerm(_: Context, name)).andThen(validateExpr(_, expr))(c)
        if (name.tpe == expr.tpe) c0
        else c0 ~ s"Assignment of incompatible type ${expr.tpe.repr} != ${name.tpe.repr}: ${s.repr}"
      case p.Stmt.Update(lhs, idx, value) =>
        (validateTerm(_, lhs)).andThen(validateTerm(_, idx)).andThen(validateTerm(_, value))(c)
      case p.Stmt.While(tests, cond, body) =>
        (tests
          .foldLeft(_: Context)(validateStmt(_, _)))
          .andThen(validateTerm(_, cond))
          .andThen(body.foldLeft(_)(validateStmt(_, _)))(c)
      case p.Stmt.Break => c
      case p.Stmt.Cont  => c
      case p.Stmt.Cond(cond, trueBr, falseBr) =>
        (validateExpr(_, cond))
          .andThen(trueBr.foldLeft(_)(validateStmt(_, _)))
          .andThen(falseBr.foldLeft(_)(validateStmt(_, _)))(c)
      case p.Stmt.Return(value) => validateExpr(c, value)
    }

    // Check for general bad (missing) reference and type mismatch errors
    val referenceAndTypeErrors = f.body match {
      case Nil => List("Function does not contain any statement") // Not legal even for a unit function
      case xs  =>
        // Use the function args as starting names
        val initialNames = Context(
          (f.receiver.toList ++ f.args ++ f.moduleCaptures ++ f.termCaptures).map(_.named).toSet
        )
        xs.foldLeft(initialNames)(validateStmt(_, _)).errors
    }

    // Check for var name collisions
    val varCollisionErrors = f
      .collectWhere[p.Stmt] { case s: p.Stmt.Var => s }
      .groupMap(_.name.symbol)(m => m.name.tpe -> m.expr)
      .collect {
        case (name, xs) if xs.size > 1 =>
          s"Variable $name was defined ${xs.size} times, RHSs=${xs.map((tpe, rhs) => s"${rhs.fold("_")(_.repr)} :$tpe").mkString(";")}"
      }
      .toList

    val badReturnErrors = f.collectWhere[p.Stmt] { case p.Stmt.Return(e) => e.tpe } match {
      case Nil => List("Function contains no return statements")
      case ts if ts.exists(x => x != f.rtn && !(x == p.Type.Nothing || f.rtn == p.Type.Nothing)) =>
        List(
          s"Not all return stmts return the function return type, expected ${f.rtn.repr}, got ${ts.map(_.repr).mkString(",")}"
        )
      case _ => Nil
    }

    val allFnLUT = fs.map(f => f.name -> f).toMap
    val badFnInvokeErrors =
      if (!verifyFunction) Nil
      else
        f.collectWhere[p.Expr] { case ivk: p.Expr.Invoke =>
          allFnLUT.get(ivk.name) match {
            case None => s"Callsite `${ivk.repr}` invokes an undefined function" :: Nil
            case Some(f) =>
              val sig = f.signature
              (None ::
                Option.when(ivk.rtn != sig.rtn)(
                  s"Callsite return type mismatch for `${ivk.repr}`, function signature: ${sig.repr}"
                ) :: Option.when(ivk.receiver.map(_.tpe) != sig.receiver)(
                  s"Callsite receiver mismatch for `${ivk.repr}`, function signature: ${sig.repr}"
                ) :: Option.when(ivk.args.map(_.tpe) != sig.args)(
                  s"Callsite args mismatch for `${ivk.repr}`, function signature: ${sig.repr}"
                ) :: Option.when(ivk.captures.map(_.tpe) != (sig.moduleCaptures ++ sig.termCaptures))(
                  s"Callsite capture mismatch for `${ivk.repr}`, function signature: ${sig.repr}"
                ) :: Option.when(ivk.tpeArgs.size != sig.tpeVars.size)(
                  s"Callsite type arg arity mismatch for `${ivk.repr}`, function signature: ${sig.repr}"
                ) ::
                Nil).flatten
          }
        }.flatten

    varCollisionErrors ++ referenceAndTypeErrors ++ badReturnErrors ++ badFnInvokeErrors
  }

  def apply(program: p.Program, log: Log, verifyFunction: Boolean): (List[(p.Function, List[String])]) = {
    val allFunctions = program.entry :: program.functions
    allFunctions.map(f => f -> validateSingle(f, program.defs, allFunctions, verifyFunction))
  }
}
