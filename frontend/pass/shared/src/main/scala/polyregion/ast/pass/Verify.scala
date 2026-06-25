package polyregion.ast.pass

import polyregion.ast.{PolyAST as p, *, given}
import polyregion.ast.Traversal.*

// well-formedness checks over a program
// examples:
//   p rooted, p[i]                      ->  ok
//   q opaque, q[i] / q[i] = v / q.f     ->  "read/write/deref through opaque-origin pointer q"
//   alloc Local of a runtime size       ->  "Alloc Local ... requires a constant size"
//   PostMono fn still holds a Type.Var  ->  "PostMono: ... contains Type.Var"
//   Global p rooted at a Local a        ->  "p declared Global but rooted at a declared Local"
//   r = (T*) intExpr                    ->  "Cannot cast unrelated type ..."
//   use of an undeclared / retyped name ->  "uses the unseen/already-defined variable ..."
object Verify {

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

  def validateAlloc(program: p.Program): List[String] = {
    def isConstSize(t: p.Term): Boolean = t match {
      case _: p.Term.IntS32Const | _: p.Term.IntS64Const | _: p.Term.IntU32Const | _: p.Term.IntU64Const => true
      case _                                                                                             => false
    }
    (program.entry :: program.functions)
      .flatMap { f =>
        f.collectWhere[p.Expr] { case a: p.Expr.Alloc => (f, a) }
      }
      .collect {
        case (f, p.Expr.Alloc(_, size, p.Type.Space.Local, _)) if !isConstSize(size) =>
          s"Alloc Local in ${f.name.repr} requires a constant size; got ${size.repr}"
      }
  }

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

  def validatePoison(program: p.Program): List[String] =
    (program.entry :: program.functions).flatMap { f =>
      f.collectWhere[p.Term] { case x: p.Term.Poison =>
        s"${f.name.repr}: unlowered poison value of type ${x.tpe.repr}"
      }
    }

  def validateRegions(program: p.Program): List[String] =
    (program.entry :: program.functions).flatMap { f =>
      val derived = Provenance.derivedIn(f)
      f.collectWhere[p.Expr] {
        case p.Expr.Index(ptr, _, _) if Provenance.at(derived, ptr) == p.Region.Opaque =>
          s"${f.name.repr}: read through opaque-origin pointer ${ptr.repr}"
      } ::: f.collectWhere[p.Stmt] {
        case p.Stmt.Update(lhs, _, _) if Provenance.at(derived, lhs) == p.Region.Opaque =>
          s"${f.name.repr}: write through opaque-origin pointer ${lhs.repr}"
      } ::: f.collectWhere[p.Term] {
        case s @ p.Term.Select(root, steps, _)
            if steps.nonEmpty && Provenance.isPtr(root.tpe) && Provenance.at(derived, s) == p.Region.Opaque =>
          s"${f.name.repr}: deref through opaque-origin pointer ${s.repr}"
      }
    }

  def validateRegionSpaces(program: p.Program): List[String] = {
    def spaceOf(t: p.Type): Option[p.Type.Space] = t match {
      case p.Type.Ptr(_, s)    => Some(s)
      case p.Type.Arr(_, _, s) => Some(s)
      case _                   => None
    }
    (program.entry :: program.functions).flatMap { f =>
      Provenance.derivedIn(f).toList.sortBy(_._1.symbol).flatMap {
        case (n, p.Region.Rooted(r)) if r != n =>
          (for {
            sn <- spaceOf(n.tpe)
            sr <- spaceOf(r.tpe)
            if sn != sr
          } yield s"${f.name.repr}: ${n.symbol} declared $sn but rooted at ${r.symbol} declared $sr").toList
        case _ => Nil
      }
    }
  }

  def validateSingle(
      f: p.Function,
      defs: List[p.StructDef],
      fs: List[p.Function],
      verifyFunction: Boolean,
      sdefLUT: Map[p.Sym, p.StructDef],
      allFnLUT: Map[p.Sym, List[p.Function]]
  ): List[String] = {

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
            case p.PathStep.Index(_) =>
              currTpe match {
                case p.Type.Ptr(comp, _) => (acc, comp)
                case other =>
                  (acc ~ s"Index step on non-pointer type ${other.repr} in `$termRepr`", other)
              }
            case p.PathStep.IndexDyn(_) =>
              currTpe match {
                case p.Type.Arr(comp, _, _) => (acc, comp)
                case p.Type.Ptr(comp, _)    => (acc, comp)
                case other =>
                  (acc ~ s"IndexDyn step on non-indexable type ${other.repr} in `$termRepr`", other)
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
          case (a, b) if a == b                       => c0
          case (_: p.Type.Struct, _: p.Type.Struct)   => c0
          case (a, b) if isNumeric(a) && isNumeric(b) => c0
          case (a, b) => c0 ~ s"Cannot cast unrelated type ${a.repr} to ${b.repr}: ${e.repr}"
        }
      case p.Expr.Invoke(name, tpeArgs, receiver, args, rtn) =>
        val c0 = receiver.map(validateTerm(c, _)).getOrElse(c)
        args.foldLeft(c0)(validateTerm(_, _))
      case p.Expr.ForeignCall(_, args, _) => args.foldLeft(c)(validateTerm(_, _))
      case p.Expr.OffsetOf(_, _)          => c
      case p.Expr.SizeOf(_)               => c
      case p.Expr.Index(lhs, idx, _)      => validateTerm(validateTerm(c, lhs), idx)
      case p.Expr.RefTo(lhs, idx, _, _, _) =>
        val c0 = validateTerm(c, lhs); idx.fold(c0)(validateTerm(c0, _))
      case p.Expr.Alloc(_, size, _, _) => validateTerm(c, size)
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
                val sig      = f.signature
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
