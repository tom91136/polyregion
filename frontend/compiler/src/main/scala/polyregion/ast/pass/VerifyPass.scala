package polyregion.ast.pass

import cats.data.EitherT
import cats.syntax.all.*
import polyregion.ast.{PolyAST as p, *, given}
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

    def validateTerm(c: Context, t: p.Expr): Context = t match {
      case p.Expr.Select(Nil, local) => c !! (local, t.repr)
      case p.Expr.Select(local :: rest, last) =>
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
                        .filter(_ == n) match {
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
      case p.Expr.Cast(from, as) =>
        val c0 = validateTerm(c, from)
        def isNumeric(t: p.Type) = t.kind == p.Type.Kind.Integral || t.kind == p.Type.Kind.Fractional
        (from.tpe, as) match {
          case (from, as) if from == as => c0
          case (p.Type.Struct(_, _, _, parents), p.Type.Struct(name, _, _, _)) if parents.contains(name) =>
            c0 // safe upcase
          case (p.Type.Struct(name, _, _, _), p.Type.Struct(_, _, _, parents)) if parents.contains(name) =>
            c0 // unsafe downcast
          case (from, as) if isNumeric(from) && isNumeric(as) =>
            c0 // numeric primitive cast (signed/unsigned/widening/narrowing)
          case (from, as) => c0 ~ s"Cannot cast unrelated type ${from.repr} to ${as.repr}: ${e.repr}"
        }
      case p.Expr.Invoke(name, tpeArgs, receiver, args, captures, rtn) =>
        val c0 = receiver.map(validateTerm(c, _)).getOrElse(c)
        val c1 = args.foldLeft(c0)(validateTerm(_, _))
        val c2 = captures.foldLeft(c1)(validateTerm(_, _))
        c2
      case p.Expr.Index(lhs, idx, component)         => (validateTerm(_: Context, lhs)).andThen(validateTerm(_, idx))(c)
      case p.Expr.RefTo(lhs, idx, _, _)              =>
        val c0 = validateTerm(c, lhs); idx.fold(c0)(validateTerm(c0, _))
      case p.Expr.Alloc(witness, size, space)        => validateTerm(c, size)
      case p.Expr.Annotated(inner, _, _)             => validateExpr(c, inner)
      case p.Expr.SpecOp(_) | p.Expr.MathOp(_) | p.Expr.IntrOp(_) => c
      case ref                                       => validateTerm(c, ref)
    }

    def validateStmt(c: Context, s: p.Stmt): Context = s match {
      case p.Stmt.Block(xs)  => xs.foldLeft(c)(validateStmt(_, _))
      case p.Stmt.Comment(_) => c
      case p.Stmt.Var(name, expr) =>
        val c0 = expr.map(e => validateExpr(_: Context, e)).getOrElse(identity[Context])(c + name)
        // Treat two types as compatible when they only differ in `Type.Var` positions getting
        // substituted by concrete types (call-site specialisation may leave a polymorphic return
        // type that the caller's var has already concretised).
        def varCompatible(t: p.Type, u: p.Type): Boolean = (t, u) match {
          case (a, b) if a == b                                   => true
          case (p.Type.Var(_), _) | (_, p.Type.Var(_))            => true
          case (p.Type.Ptr(c1, l1, s1), p.Type.Ptr(c2, l2, s2))   => l1 == l2 && s1 == s2 && varCompatible(c1, c2)
          case (p.Type.Struct(n1, v1, a1, _), p.Type.Struct(n2, v2, a2, _)) =>
            n1 == n2 && v1 == v2 && a1.size == a2.size && a1.zip(a2).forall((l, r) => varCompatible(l, r))
          case _ => false
        }
        expr match {
          case Some(rhs) if rhs.tpe != name.tpe =>
            (name.tpe, rhs.tpe) match {
              case (p.Type.Struct(name, _, _, _), p.Type.Struct(_, _, _, parents)) if parents.contains(name) => c0
              case (l, r) if varCompatible(l, r) => c0
              case _ => c0 ~ s"Var declaration of incompatible type ${rhs.tpe.repr} != ${name.tpe.repr}: ${s.repr}"
            }
          case _ => c0
        }
      case p.Stmt.Mut(name, expr ) =>
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
      case p.Stmt.ForRange(induction, lbIncl, ubExcl, step, body) =>
        val c0 = validateTerm(c, induction)
        val c1 = validateTerm(c0, lbIncl)
        val c2 = validateTerm(c1, ubExcl)
        val c3 = validateTerm(c2, step)
        body.foldLeft(c3)(validateStmt(_, _))
      case p.Stmt.Annotated(inner, _, _) => validateStmt(c, inner)
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
      // Abstract definitions (e.g. typeclass methods like `Monoid.mempty`) have empty/comment-only
      // bodies; they're dispatched via vtable in DynamicDispatchPass and never invoked directly.
      case Nil if f.body.forall(_.isInstanceOf[p.Stmt.Comment]) => Nil
      case Nil => List("Function contains no return statements")
      case ts if ts.exists(x => x != f.rtn && !(x == p.Type.Nothing || f.rtn == p.Type.Nothing)) =>
        List(
          s"Not all return stmts return the function return type, expected ${f.rtn.repr}, got ${ts.map(_.repr).mkString(",")}"
        )
      case _ => Nil
    }

    // Group functions by name so we can disambiguate overloads (`fn(Int)` vs `fn(Double)`).
    val allFnLUT: Map[p.Sym, List[p.Function]] = fs.groupBy(_.name)
    val badFnInvokeErrors =
      if (!verifyFunction) Nil
      else
        f.collectWhere[p.Expr] { case ivk: p.Expr.Invoke =>
          allFnLUT.get(ivk.name) match {
            case None | Some(Nil) => s"Callsite `${ivk.repr}` invokes an undefined function" :: Nil
            case Some(candidates) =>
              // Pick the candidate whose signature exactly matches the call site (overload resolution).
              val matching = candidates.find { f =>
                val sig = f.signature
                ivk.rtn == sig.rtn &&
                ivk.receiver.map(_.tpe) == sig.receiver &&
                ivk.args.map(_.tpe) == sig.args &&
                ivk.captures.map(_.tpe) == (sig.moduleCaptures ++ sig.termCaptures) &&
                ivk.tpeArgs.size == sig.tpeVars.size
              }
              matching match {
                case Some(_) => Nil
                case None    =>
                  // Surface a comprehensive error: list every candidate's signature so it's easy to see why none matched.
                  val sigs = candidates.map(c => s"  - ${c.signature.repr}").mkString("\n")
                  s"Callsite `${ivk.repr}` does not match any overload of `${ivk.name.repr}`:\n$sigs" :: Nil
              }
          }
        }.flatten

    varCollisionErrors ++ referenceAndTypeErrors ++ badReturnErrors ++ badFnInvokeErrors
  }

  def apply(program: p.Program, log: Log, verifyFunction: Boolean): (List[(p.Function, List[String])]) = {
    val allFunctions = program.entry :: program.functions
    allFunctions.map(f => f -> validateSingle(f, program.defs, allFunctions, verifyFunction))
  }
}
