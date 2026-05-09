package polyregion.ast.pass

import polyregion.ast.{PolyAST as p, *, given}
import polyregion.ast.Traversal.*

import scala.annotation.tailrec

object VerifyPass {

  case class Context(declared: Set[p.Named] = Set.empty, errors: List[String] = Nil) {
    def |+|(y: Context): Context = copy(declared ++ y.declared, errors ::: y.errors)
    def +(n: p.Named): Context   = copy(declared + n)

    def ~(error: String): Context = copy(errors = error :: errors)
    def !!(n: p.Named, tree: String): Context =
      if (declared.contains(n)) this
      else {
        (declared.find(_.symbol == n.symbol), n.tpe) match {
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

  // Validate per-address-space Alloc invariants.
  def validateAlloc(program: p.Program): List[String] = {
    // Local allocs need a constant size (lowered to a stack slot); Global allocs go to malloc on
    // the host but need a backend-side host-vs-shader check, which the IR layer can't make.
    def isConstSize(t: p.Term): Boolean = t match {
      case _: p.Term.IntS32Const | _: p.Term.IntS64Const | _: p.Term.IntU32Const | _: p.Term.IntU64Const => true
      case _                                                                                             => false
    }
    (program.entry :: program.functions)
      .flatMap { f =>
        f.collectWhere[p.Expr] { case a: p.Expr.Alloc => (f, a) }
      }
      .collect {
        case (f, p.Expr.Alloc(_, size, p.Type.Space.Local)) if !isConstSize(size) =>
          s"Alloc Local in ${f.name.repr} requires a constant size; got ${size.repr}"
      }
  }

  // Reject Type.Var / Type.Exec when the program is in the PostMono phase.
  def validatePostMono(program: p.Program): List[String] =
    if (program.phase != p.PassPhase.PostMono) Nil
    else {
      val fns = program.entry :: program.functions
      val typeViolations = fns.flatMap { f =>
        f.collectWhere[p.Type] {
          case t @ p.Type.Var(_)        => s"PostMono: ${f.name.repr} contains Type.Var ${t.repr}"
          case t @ p.Type.Exec(_, _, _) => s"PostMono: ${f.name.repr} contains Type.Exec ${t.repr}"
        }
      }
      val tpeVarViolations = fns.collect {
        case f if f.tpeVars.nonEmpty => s"PostMono: ${f.name.repr} still has tpeVars ${f.tpeVars.mkString(",")}"
      }
      typeViolations ::: tpeVarViolations
    }

  def validateSingle(
      f: p.Function,
      defs: List[p.StructDef],
      fs: List[p.Function],
      verifyFunction: Boolean,
      sdefLUT: Map[p.Sym, p.StructDef],
      allFnLUT: Map[p.Sym, List[p.Function]]
  ): List[String] = {

    // Walk a Term.Select's path from root through steps; validate at each step that the
    // surrounding type lets us take that step (Field requires Struct, Deref requires Ptr).
    def validatePath(c: Context, term: p.Term.Select): Context = {
      val termRepr = term.repr
      val initial  = c !! (term.root, termRepr)
      term.steps
        .foldLeft((initial, term.root.tpe: p.Type)) { case ((acc, currTpe), step) =>
          step match {
            case p.PathStep.Deref =>
              currTpe match {
                case p.Type.Ptr(comp, _) => (acc, comp)
                case other =>
                  (acc ~ s"Deref step on non-pointer type ${other.repr} in `$termRepr`", other)
              }
            case p.PathStep.Field(name) =>
              currTpe match {
                case s @ p.Type.Struct(sym, args) =>
                  sdefLUT.get(sym) match {
                    case None =>
                      (acc ~ s"Unknown struct type ${sym.repr} in `$termRepr`", currTpe)
                    case Some(sdef) =>
                      val apTable = sdef.tpeVars.zip(args).toMap
                      val members = sdef.members.map(m =>
                        m.modifyAll[p.Type](_.mapLeaf {
                          case p.Type.Var(n) => apTable.getOrElse(n, p.Type.Var(n))
                          case x             => x
                        })
                      )
                      members.find(_.symbol == name) match {
                        case Some(m) => (acc, m.tpe)
                        case None =>
                          (acc ~ s"Struct ${sdef.repr} does not contain field $name in `$termRepr`", currTpe)
                      }
                  }
                case other =>
                  (acc ~ s"Field step on non-struct type ${other.repr} in `$termRepr`", other)
              }
          }
        }
        ._1
    }

    def validateTerm(c: Context, t: p.Term): Context = t match {
      case s: p.Term.Select => validatePath(c, s)
      case _                => c
    }

    def validateExpr(c: Context, e: p.Expr): Context = e match {
      case p.Expr.Alias(t)  => validateTerm(c, t)
      case p.Expr.SpecOp(_) => c
      case p.Expr.MathOp(_) => c
      case p.Expr.IntrOp(_) => c
      case p.Expr.Cast(from, as) =>
        val c0                   = validateTerm(c, from)
        def isNumeric(t: p.Type) = t.kind == p.Type.Kind.Integral || t.kind == p.Type.Kind.Fractional
        (from.tpe, as) match {
          case (a, b) if a == b                              => c0
          case (p.Type.Struct(_, _), p.Type.Struct(name, _)) =>
            // upcast/downcast permission requires a parents lookup against StructDefs.
            sdefLUT.get(name) match {
              case Some(sdef)
                  if sdef.parents.exists(_.name == name) ||
                    sdefLUT.values.exists(_.parents.exists(_.name == name)) =>
                c0
              case _ => c0
            }
          case (a, b) if isNumeric(a) && isNumeric(b) => c0
          case (a, b) => c0 ~ s"Cannot cast unrelated type ${a.repr} to ${b.repr}: ${e.repr}"
        }
      case p.Expr.Invoke(name, tpeArgs, receiver, args, rtn) =>
        val c0 = receiver.map(validateTerm(c, _)).getOrElse(c)
        args.foldLeft(c0)(validateTerm(_, _))
      case p.Expr.Index(lhs, idx, _) => validateTerm(validateTerm(c, lhs), idx)
      case p.Expr.RefTo(lhs, idx, _, _) =>
        val c0 = validateTerm(c, lhs); idx.fold(c0)(validateTerm(c0, _))
      case p.Expr.Alloc(_, size, _) => validateTerm(c, size)
    }

    def validateStmt(c: Context, s: p.Stmt): Context = s match {
      case p.Stmt.Var(name, expr, _) =>
        val c0 = expr.map(e => validateExpr(_: Context, e)).getOrElse(identity[Context])(c + name)
        def varCompatible(t: p.Type, u: p.Type): Boolean = (t, u) match {
          case (a, b) if a == b                                 => true
          case (p.Type.Var(_), _) | (_, p.Type.Var(_))          => true
          case (p.Type.Ptr(c1, s1), p.Type.Ptr(c2, s2))         => s1 == s2 && varCompatible(c1, c2)
          case (p.Type.Arr(c1, l1, s1), p.Type.Arr(c2, l2, s2)) => l1 == l2 && s1 == s2 && varCompatible(c1, c2)
          case (p.Type.Struct(n1, a1), p.Type.Struct(n2, a2)) =>
            n1 == n2 && a1.size == a2.size && a1.zip(a2).forall((l, r) => varCompatible(l, r))
          case _ => false
        }
        expr match {
          case Some(rhs) if rhs.tpe != name.tpe =>
            if (varCompatible(name.tpe, rhs.tpe)) c0
            else c0 ~ s"Var declaration of incompatible type ${rhs.tpe.repr} != ${name.tpe.repr}: ${s.repr}"
          case _ => c0
        }
      case p.Stmt.Mut(name, expr) =>
        val c0 = validateExpr(validatePath(c, name), expr)
        if (name.tpe == expr.tpe) c0
        else c0 ~ s"Assignment of incompatible type ${expr.tpe.repr} != ${name.tpe.repr}: ${s.repr}"
      case p.Stmt.Update(lhs, idx, value) =>
        validateTerm(validateTerm(validatePath(c, lhs), idx), value)
      case p.Stmt.While(cond, body) =>
        val c0 = validateTerm(c, cond)
        body.foldLeft(c0)(validateStmt(_, _))
      case p.Stmt.Break => c
      case p.Stmt.Cont  => c
      case p.Stmt.Cond(cond, trueBr, falseBr) =>
        val c0 = validateTerm(c, cond)
        falseBr.foldLeft(trueBr.foldLeft(c0)(validateStmt(_, _)))(validateStmt(_, _))
      case p.Stmt.Return(value) => validateExpr(c, value)
      case p.Stmt.ForRange(induction, lbIncl, ubExcl, step, body) =>
        val c0 = c + induction
        val c1 = validateTerm(c0, lbIncl)
        val c2 = validateTerm(c1, ubExcl)
        val c3 = validateTerm(c2, step)
        body.foldLeft(c3)(validateStmt(_, _))
      case p.Stmt.Annotated(inner, _, _) => validateStmt(c, inner)
    }

    val referenceAndTypeErrors = f.body match {
      case Nil => List("Function does not contain any statement")
      case xs =>
        val initialNames = Context(
          (f.receiver.toList ++ f.args ++ f.moduleCaptures ++ f.termCaptures).map(_.named).toSet
        )
        xs.foldLeft(initialNames)(validateStmt(_, _)).errors
    }

    val varCollisionErrors = f
      .collectWhere[p.Stmt] { case s: p.Stmt.Var => s }
      .groupMap(_.name.symbol)(m => m.name.tpe -> m.expr)
      .collect {
        case (name, xs) if xs.size > 1 =>
          s"Variable $name was defined ${xs.size} times, RHSs=${xs.map((tpe, rhs) => s"${rhs.fold("_")(_.repr)} :$tpe").mkString(";")}"
      }
      .toList

    val badReturnErrors = f.collectWhere[p.Stmt] { case p.Stmt.Return(e) => e.tpe } match {
      case Nil if f.body.isEmpty => Nil
      case Nil                   => List("Function contains no return statements")
      case ts if ts.exists(x => x != f.rtn && !(x == p.Type.Nothing || f.rtn == p.Type.Nothing)) =>
        List(
          s"Not all return stmts return the function return type, expected ${f.rtn.repr}, got ${ts.map(_.repr).mkString(",")}"
        )
      case _ => Nil
    }

    val badFnInvokeErrors =
      if (!verifyFunction) Nil
      else
        f.collectWhere[p.Expr] { case ivk: p.Expr.Invoke =>
          allFnLUT.get(ivk.name) match {
            case None | Some(Nil) => s"Callsite `${ivk.repr}` invokes an undefined function" :: Nil
            case Some(candidates) =>
              val matching = candidates.find { f =>
                val sig = f.signature
                // The call-site args carry `moduleCaptures ::: termCaptures ::: args` (see
                // Compiler.patchFn). The actual call-site arity may be lower than the callee's
                // expected arity if patchFn hasn't yet reached the call (e.g. mid-pass) - the
                // backend tolerates the under-shoot via the same lookup. Accept either exact match
                // or under-shoot here.
                val expected = sig.args.size + sig.moduleCaptures.size + sig.termCaptures.size
                ivk.rtn == sig.rtn &&
                ivk.receiver.map(_.tpe) == sig.receiver &&
                ivk.args.size <= expected &&
                ivk.tpeArgs.size == sig.tpeVars.size
              }
              matching match {
                case Some(_) => Nil
                case None =>
                  val sigs = candidates.map(c => s"  - ${c.signature.repr}").mkString("\n")
                  s"Callsite `${ivk.repr}` does not match any overload of `${ivk.name.repr}`:\n$sigs" :: Nil
              }
          }
        }.flatten

    varCollisionErrors ++ referenceAndTypeErrors ++ badReturnErrors ++ badFnInvokeErrors
  }

  def apply(program: p.Program, log: Log, verifyFunction: Boolean): List[(p.Function, List[String])] = {
    val allFunctions = program.entry :: program.functions
    val sdefLUT      = program.defs.iterator.map(d => d.name -> d).toMap
    val allFnLUT     = allFunctions.groupBy(_.name)
    val perFn =
      allFunctions.map(f => f -> validateSingle(f, program.defs, allFunctions, verifyFunction, sdefLUT, allFnLUT))
    val global = validateAlloc(program) ::: validatePostMono(program)
    if (global.isEmpty) perFn
    else
      perFn match {
        case (entry, errs) :: rest => (entry, errs ::: global) :: rest
        case Nil                   => (program.entry, global) :: Nil
      }
  }
}
