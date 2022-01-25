package polyregion

import scala.quoted.{Expr, Quotes}
import scala.quoted.*
import scala.annotation.tailrec
import scala.reflect.ClassTag
import fansi.ErrorMode.Throw

import java.nio.file.{Files, Paths, StandardOpenOption}
import cats.data.EitherT
import cats.Eval
import cats.data.NonEmptyList
import polyregion.ast.PolyAst
import polyregion.internal.*

import java.lang.reflect.Modifier
import polyregion.data.MsgPack
import polyregion.ast.PolyAst.StructDef

class AstTransformer(using val q: Quotes) {

  import quotes.reflect.*
  import cats.syntax.all.*

  def extractClosureBody(term: Term): Result[DefDef] = term match {
    case Block(List(defStmt: DefDef), Closure(Ident(name), _)) =>
      if (defStmt.name == name) defStmt.success
      else s"Closure name mismatch: def block was `${defStmt.name}` but closure is named `${name}`".fail

    case bad => s"Illegal closure tree:\n${pprint(bad)}".fail
  }

  def extractInlineBlock(term: Term): Result[Term] = term match {
    case Inlined(None, Nil, term) => term.success
    case x                        => s"Illegal top-level inline tree: ${x}".fail
  }

  object Symbols {
    val JavaLang      = "java" :: "lang" :: Nil
    val Scala         = "scala" :: Nil
    val ScalaMath     = "scala" :: "math" :: "package$" :: Nil
    val JavaMath      = "java" :: "lang" :: "Math$" :: Nil
    val SeqMutableOps = "scala" :: "collection" :: "mutable" :: "SeqOps" :: Nil
    val SeqOps        = "scala" :: "collection" :: "SeqOps" :: Nil
    val Buffer        = PolyAst.Sym[Buffer[_]]

  }

  def resolveIntrinsics(c: Context)(ap: Apply): Deferred[TermRes] = {

    def resolveModuleApply(receiverSym: PolyAst.Sym, tpe: PolyAst.Type)(args: List[Option[PolyAst.Term]]) = {

      val outName = PolyAst.Named(s"v${c.depth}", tpe)
      val outRef  = PolyAst.Term.Select(Nil, outName)

      (receiverSym.fqn, args) match {
        case ((Symbols.ScalaMath | Symbols.JavaMath) :+ op, Some(x) :: Some(y) :: Nil) => // scala.math binary
          ???
        case ((Symbols.ScalaMath | Symbols.JavaMath) :+ op, Some(x) :: Nil) => // scala.math unary
          val expr = op match {
            case "sin" => PolyAst.Expr.Sin(x, tpe)
            case "cos" => PolyAst.Expr.Cos(x, tpe)
            case "tan" => PolyAst.Expr.Tan(x, tpe)
            case "abs" => PolyAst.Expr.Abs(x, tpe)
          }
          (Some(outRef), PolyAst.Stmt.Var(outName, Some(expr)) :: Nil)
        case (sym, args) =>
          println(s"${sym.mkString(".")}(${args.mkString(",")}) ")
          ???
      }
    }

    def resolveInstanceApply(
        receiverSym: PolyAst.Sym,
        tpe: PolyAst.Type
    )(receiver: Option[PolyAst.Term], args: List[Option[PolyAst.Term]]) = {

      val outName = PolyAst.Named(s"v${c.depth}", tpe)
      val outRef  = PolyAst.Term.Select(Nil, outName)

      (receiverSym.fqn, receiver, args) match {
        case (
              Symbols.Scala :+ (
                "Byte" | "Short" | "Int" | "Long" | "Float" | "Double" | "Char"
              ) :+ op,
              Some(x),
              Some(y) :: Nil
            ) =>
          val expr = op match {
            case "+"  => PolyAst.Expr.Add(x, y, tpe)
            case "-"  => PolyAst.Expr.Sub(x, y, tpe)
            case "*"  => PolyAst.Expr.Mul(x, y, tpe)
            case "/"  => PolyAst.Expr.Div(x, y, tpe)
            case "%"  => PolyAst.Expr.Rem(x, y, tpe)
            case "<"  => PolyAst.Expr.Lt(x, y)
            case "<=" => PolyAst.Expr.Lte(x, y)
            case ">"  => PolyAst.Expr.Gt(x, y)
            case ">=" => PolyAst.Expr.Gte(x, y)
            case "==" => PolyAst.Expr.Eq(x, y)
            case "!=" => PolyAst.Expr.Neq(x, y)
            case "&&" => PolyAst.Expr.And(x, y)
            case "||" => PolyAst.Expr.Or(x, y)
          }
          (Some(outRef), PolyAst.Stmt.Var(outName, Some(expr)) :: Nil)
        case ((Symbols.SeqOps | Symbols.SeqMutableOps) :+ "apply", Some(xs: PolyAst.Term.Select), Some(idx) :: Nil) =>
          (Some(outRef), PolyAst.Stmt.Var(outName, Some(PolyAst.Expr.Index(xs, idx, tpe))) :: Nil)
        case (
              (Symbols.SeqOps | Symbols.SeqMutableOps) :+ "update",
              Some(xs: PolyAst.Term.Select),
              Some(idx) :: Some(x) :: Nil
            ) =>
          (None, PolyAst.Stmt.Update(xs, idx, x) :: Nil)
        case (sym, recv, args) =>
          println(s" (($recv) : ${sym.mkString(".")})(${args.mkString(",")}) ")
          ???
      }
    }

    for {
      (_, tpe, c) <- resolveTpe(c.log(ap))(ap.tpe)

      receiverOwnerFlags = ap.fun.symbol.maybeOwner.flags
      receiverSym        = PolyAst.Sym(ap.fun.symbol.fullName)

      _ = println(s"receiverFlags:${receiverSym} = ${receiverOwnerFlags.show}")

      (c, argRefs, argTrees) <- resolveTerms(c.down(ap))(ap.args)

      r <-
        if (receiverOwnerFlags.is(Flags.Module)) { // receiver is an object/package object
          val (outRef, tree) = resolveModuleApply(receiverSym, tpe)(argRefs) //
          (c, outRef, argTrees ::: tree).success.deferred
        } else {

          for {
            (receiverDepth, receiverRef, receiverTrees) <- ap.fun match {
              case Select(q, n) => resolveTerm(c.log(ap))(q)
              case _            => ??? // (ctx.depth, None, Nil).success.deferred
            }

            (outRef, tree) = resolveInstanceApply(receiverSym, tpe)(receiverRef, argRefs)
          } yield (receiverDepth, outRef, receiverTrees ::: argTrees ::: tree)

        }

    } yield r
  }

  @tailrec final def resolveSym(ref: TypeRepr): Result[PolyAst.Sym] = ref.dealias.simplified match {
    case ThisType(tpe) => resolveSym(tpe)
    case tpe: NamedType =>
      tpe.classSymbol match {
        case None => s"Named type is not a class: ${tpe}".fail
        case Some(sym) if sym.name == "<root>" => // discard root package
          resolveSym(tpe.qualifier)
        case Some(sym) => PolyAst.Sym(sym.fullName).success
      }
    // case NoPrefix()    => None.success
    case invalid => s"Invalid type: ${invalid}".fail
  }

  // we may encounter singleton types hence maybe term
  def resolveTpe(c: Context)(repr: TypeRepr): Deferred[(Option[PolyAst.Term], PolyAst.Type, Context)] =
    repr.dealias.widenTermRefByName.simplified match {
      case andOr: AndOrType =>
        for {
          (leftTerm, leftTpe, c)   <- resolveTpe(c)(andOr.left)
          (rightTerm, rightTpe, c) <- resolveTpe(c)(andOr.right)
        } yield
          if (leftTpe == rightTpe) (leftTerm.orElse(rightTerm), leftTpe, c)
          else ???

      case tpe @ AppliedType(ctor, args) =>
        for {
          name <- resolveSym(ctor).deferred
          xs   <- args.traverse(resolveTpe(c)(_))
        } yield (name, xs) match {
          case (Symbols.Buffer, (_, component, c) :: Nil) => (None, PolyAst.Type.Array(component, None), c)
          case (n, ys)                                    => ??? // None -> PolyAst.Type.Struct(n, ys)
        }

      // widen singletons
      case ConstantType(x) =>
        (x match {
          case BooleanConstant(v) => (Some(PolyAst.Term.BoolConst(v)), PolyAst.Type.Bool, c)
          case ByteConstant(v)    => (Some(PolyAst.Term.ByteConst(v)), PolyAst.Type.Byte, c)
          case ShortConstant(v)   => (Some(PolyAst.Term.ShortConst(v)), PolyAst.Type.Short, c)
          case IntConstant(v)     => (Some(PolyAst.Term.IntConst(v)), PolyAst.Type.Int, c)
          case LongConstant(v)    => (Some(PolyAst.Term.LongConst(v)), PolyAst.Type.Long, c)
          case FloatConstant(v)   => (Some(PolyAst.Term.FloatConst(v)), PolyAst.Type.Float, c)
          case DoubleConstant(v)  => (Some(PolyAst.Term.DoubleConst(v)), PolyAst.Type.Double, c)
          case CharConstant(v)    => (Some(PolyAst.Term.CharConst(v)), PolyAst.Type.Char, c)
          case StringConstant(v)  => ???
          case UnitConstant       => (Some(PolyAst.Term.UnitConst), PolyAst.Type.Unit, c)
          case NullConstant       => ???
          case ClassOfConstant(r) => ???
        }).pure

      case expr =>
        resolveSym(expr)
          .map {
            case PolyAst.Sym(Symbols.Scala :+ "Unit")      => PolyAst.Type.Unit
            case PolyAst.Sym(Symbols.Scala :+ "Boolean")   => PolyAst.Type.Bool
            case PolyAst.Sym(Symbols.Scala :+ "Byte")      => PolyAst.Type.Byte
            case PolyAst.Sym(Symbols.Scala :+ "Short")     => PolyAst.Type.Short
            case PolyAst.Sym(Symbols.Scala :+ "Int")       => PolyAst.Type.Int
            case PolyAst.Sym(Symbols.Scala :+ "Long")      => PolyAst.Type.Long
            case PolyAst.Sym(Symbols.Scala :+ "Float")     => PolyAst.Type.Float
            case PolyAst.Sym(Symbols.Scala :+ "Double")    => PolyAst.Type.Double
            case PolyAst.Sym(Symbols.Scala :+ "Char")      => PolyAst.Type.Char
            case PolyAst.Sym(Symbols.JavaLang :+ "String") => PolyAst.Type.String
            case sym                                       => PolyAst.Type.Struct(sym)
          }
          .map {
            case s @ PolyAst.Type.Struct(_) => (None, s, c.copy(defs = c.defs + expr))
            case x                          => (None, x, c)
          }
          .deferred
    }

  case class Context(depth: Int, trace: List[Tree], refs: Map[Ref, Reference], defs: Set[TypeRepr]) {
    def log(t: Tree)  = copy(trace = t :: trace)
    def down(t: Tree) = log(t).copy(depth = depth + 1)

//    def name(tpe: PolyAst.Type) = PolyAst.Named(s"v$depth", tpe)

  }

  type TermRes = (Context, Option[PolyAst.Term], List[PolyAst.Stmt])

  def resolveTrees(c: Context)(args: List[Tree]): Deferred[TermRes] =
    args match {
      case Nil => (c, None, Nil).pure
      case x :: xs =>
        resolveTree(c)(x).flatMap(xs.foldLeftM(_) { case ((prev, _, stmts0), term) =>
          resolveTree(prev)(term).map((curr, ref, stmts) =>
            (curr, ref, stmts0 ::: (PolyAst.Stmt.Comment(term.show) :: stmts))
          )
        })
    }

  def resolveTree(c: Context)(tree: Tree): Deferred[TermRes] = tree match {
    case ValDef(name, tpe, Some(rhs)) =>
      // if tpe is singleton, substitute with constant directly
      for {
        (term, t, c)      <- resolveTpe(c.log(tree))(tpe.tpe)
        (c, refOpt, tree) <- term.fold(resolveTerm(c.log(tree))(rhs))(x => (c, Some(x), Nil).pure)
        ref               <- refOpt.failIfEmpty("term res did not end up with a ref").deferred
      } yield (c, None, tree :+ PolyAst.Stmt.Var(PolyAst.Named(name, t), Some(PolyAst.Expr.Alias(ref))))
    case ValDef(name, tpe, None) => s"Unexpected variable $name:$tpe".fail.deferred
    case t: Term                 => resolveTerm(c.log(tree))(t)
  }

  def resolveTerms(c: Context)(args: List[Term]) = args match {
    case Nil => (c, Nil, Nil).pure
    case x :: xs =>
      resolveTerm(c)(x)
        .map((c, ref, xs) => (c, ref :: Nil, xs))
        .flatMap(xs.foldLeftM(_) { case ((prev, rs, tss), term) =>
          resolveTerm(prev)(term).map((curr, r, ts) => (curr, rs :+ r, ts ++ tss))
        })
  }

  def resolveTerm(ctx: Context)(term: Term): Deferred[TermRes] = term match {
    case Typed(x, _)                        => resolveTerm(ctx.log(term))(x)
    case Inlined(call, bindings, expansion) => resolveTerm(ctx.log(term))(expansion) // simple-inline
    case c @ Literal(BooleanConstant(v))    => ((ctx, Some(PolyAst.Term.BoolConst(v)), Nil)).pure
    case c @ Literal(IntConstant(v))        => ((ctx, Some(PolyAst.Term.IntConst(v)), Nil)).pure
    case c @ Literal(FloatConstant(v))      => ((ctx, Some(PolyAst.Term.FloatConst(v)), Nil)).pure
    case c @ Literal(DoubleConstant(v))     => ((ctx, Some(PolyAst.Term.DoubleConst(v)), Nil)).pure
    case c @ Literal(LongConstant(v))       => ((ctx, Some(PolyAst.Term.LongConst(v)), Nil)).pure
    case c @ Literal(ShortConstant(v))      => ((ctx, Some(PolyAst.Term.ShortConst(v)), Nil)).pure
    case c @ Literal(ByteConstant(v))       => ((ctx, Some(PolyAst.Term.ByteConst(v)), Nil)).pure
    case c @ Literal(CharConstant(v))       => ((ctx, Some(PolyAst.Term.CharConst(v)), Nil)).pure
    case c @ Literal(UnitConstant())        => ((ctx, Some(PolyAst.Term.UnitConst), Nil)).pure
    case r: Ref =>
      (ctx.refs.get(r), r) match {
        case (Some(Reference(value, tpe)), _) =>
          val term = value match {
            case name: String       => PolyAst.Term.Select(Nil, PolyAst.Named(name, tpe))
            case term: PolyAst.Term => term
          }
          ((ctx, Some(term), Nil)).pure
        case (None, i @ Ident(s)) =>
          val name = i.tpe match {
            // we've encountered a case where the ident's name is different from the TermRef's name
            // this is likely a result of inline where we end up with synthetic names
            // we use the TermRefs name in this case
            case TermRef(_, name) if name != s => name
            case _                             => s
          }
          resolveTpe(ctx.log(term))(i.tpe).map { (_, tpe, c) =>
            (c, Some(PolyAst.Term.Select(Nil, PolyAst.Named(name, tpe))), Nil)
          }
        case (None, x) =>
          s"[depth=${ctx.depth}] Ref ${x} with tpe=${x.tpe} was not identified at closure args stage ".fail.deferred

      }

    case ap @ Apply(_, _) => resolveIntrinsics(ctx)(ap)
    case Block(stat, expr) => // stat : List[Statement]
      for {
        (c, _, statTrees) <- resolveTrees(ctx.log(term))(stat)
        (c, ref, tree)    <- resolveTerm(c.log(term))(expr)
      } yield (c, ref, statTrees ++ tree)

    case Assign(lhs, rhs) =>
      for {
        (c, lhsRef, lhsTrees) <- resolveTerm(ctx.down(term))(lhs) // go down here
        (c, rhsRef, rhsTrees) <- resolveTerm(c.log(term))(rhs)
        r <- (lhsRef, rhsRef) match {
          case (Some(s @ PolyAst.Term.Select(Nil, _)), Some(rhs)) =>
            (
              c,
              None,
              lhsTrees ++ rhsTrees :+ PolyAst.Stmt.Mut(s, PolyAst.Expr.Alias(rhs))
            ).pure
          case bad => s"Illegal assign LHS,RHS: ${bad}".fail.deferred
        }
      } yield r
    case If(cond, thenTerm, elseTerm) =>
      for {
        //
        (_, tpe, c) <- resolveTpe(ctx)(term.tpe)

        (c, condRefOpt, condTrees) <- resolveTerm(c.down(term))(cond)
        (c, thenRefOpt, thenTrees) <- resolveTerm(c.log(term))(thenTerm)
        (c, elseRefOpt, elseTrees) <- resolveTerm(c.log(term))(elseTerm)
        condRef                    <- condRefOpt.failIfEmpty("Cond must have a ref").deferred

        cond <- (thenRefOpt, elseRefOpt) match {
          case (Some(thenRef), Some(elseRef)) if thenRef.tpe == tpe && elseRef.tpe == tpe =>
            val name   = PolyAst.Named(s"v${ctx.depth}", tpe)
            val result = PolyAst.Stmt.Var(name, None)
            val cond = PolyAst.Stmt
              .Cond(
                PolyAst.Expr.Alias(condRef),
                thenTrees :+ PolyAst.Stmt.Mut(PolyAst.Term.Select(Nil, name), PolyAst.Expr.Alias(thenRef)),
                elseTrees :+ PolyAst.Stmt.Mut(PolyAst.Term.Select(Nil, name), PolyAst.Expr.Alias(elseRef))
              )

            (c, Some(PolyAst.Term.Select(Nil, name)), result :: cond :: Nil).success.deferred
          case _ =>
            s"condition unification failure, then=${thenRefOpt} else=${elseRefOpt}, expr tpe=${tpe}".fail.deferred
        }
      } yield cond
    case While(cond, body) =>
      for {
        (c, condRef, condTrees) <- resolveTerm(ctx.down(term))(cond)
        (c, _, bodyTrees)       <- resolveTerm(c.log(term))(body)
      } yield {
        val block = condTrees match {
          case Nil                                  => ??? // this is illegal, while needs a bool predicate
          case PolyAst.Stmt.Var(_, Some(iv)) :: Nil =>
            // simple condition:
            // while(cond)
            PolyAst.Stmt.While(iv, bodyTrees)
          case xs =>
            // complex condition:
            // while(true) {  stmts...; if(!condRef) break;  }
            println(xs)
            ???
            val body = (xs :+ PolyAst.Stmt.Cond(
              PolyAst.Expr.Alias(condRef.get),
              Nil,
              PolyAst.Stmt.Break :: Nil
            )) ++ bodyTrees

            PolyAst.Stmt.While(PolyAst.Expr.Alias(PolyAst.Term.BoolConst(true)), body)
        }
        (c, None, block :: Nil)
      }
    case _ =>
      s"[depth=${ctx.depth}] Unhandled: $term\nTrace was:\n${(term :: ctx.trace).map(_.show).mkString("\n---\n")}".fail.deferred
  }

  def collectTree[A](in: Tree)(f: Tree => List[A]) = {
    val acc = new TreeAccumulator[List[A]] {
      def foldTree(xs: List[A], tree: Tree)(owner: Symbol): List[A] =
        foldOverTree(f(tree) ::: xs, tree)(owner)
    }
    acc.foldOverTree(Nil, in)(Symbol.noSymbol)
  }

  // case class Reference(name: String, tpe: TypeRepr)
  case class Reference(value: String | PolyAst.Term, tpe: PolyAst.Type)

  def lowerProductType[A: Type]: Deferred[StructDef] = lowerProductType(TypeRepr.of[A].typeSymbol)
  def lowerProductType(tpeSym: Symbol): Deferred[StructDef] = {

    if (!tpeSym.flags.is(Flags.Case)) {
      throw RuntimeException(s"Unsupported combination of flags: ${tpeSym.flags.show}")
    }

    tpeSym.caseFields
      .traverse(field =>
        resolveTpe(Context(0, tpeSym.tree :: Nil, Map(), Set()))(field.tree.asInstanceOf[ValDef].tpt.tpe)
          .map((_, t, c) => PolyAst.Named(field.name, t))
      )
      .map(PolyAst.StructDef(PolyAst.Sym(tpeSym.fullName), _))
  }

  @tailrec final def flattenRefs(r: Term, xs: List[Ref] = Nil): List[Ref] = r match {
    case s @ Select(nested @ Select(x, _), _) =>
      println(s"got ${s}")
      flattenRefs(x, xs)
    case s @ Select(_, _) =>
      println(s"got! ${s}")
      s :: xs
    case id @ Ident(_) => id :: xs
  }

  def lower(x: Expr[Any]): Result[(List[(Ref, PolyAst.Type)], PolyAst.Function)] = for {
    term <- extractInlineBlock(x.asTerm)

    //  _ = pprint.pprintln(block)
    //        closure <- extractClosureBody(block)

    pos         = term.pos
    closureName = s"${pos.sourceFile.name}:${pos.startLine}-${pos.endLine}"

    _ = println(s"========${closureName}=========")
    _ = println(Paths.get(".").toAbsolutePath)
    _ = println(s" -> name:               ${closureName}")
    _ = println(s" -> body(Quotes):\n${x.asTerm.toString.indent(4)}")

    externalRefs = collectTree[Ref](term) {
      // FIXME TODO make sure the owner is actually this macro, and not any other macro
      case ref: Ref
          if !ref.symbol.maybeOwner.flags.is(Flags.Macro)
//             && !ref.symbol.maybeOwner.flags.is(Flags.Module)
//             && !ref.symbol.flags.is(Flags.Method)
          =>
        ref :: Nil
      case _ => Nil
    }

    valExternalRefs = externalRefs.filter {s =>
      s.symbol.isValDef && !s.symbol.flags.is(Flags.Module)
    }.distinctBy(_.symbol)

    collapsed = valExternalRefs.foldLeft(List.empty[Ref]){
      case (acc, i @Ident(_)) => i::acc
      case (acc, s@Select(i, _)) =>

        // for this select, find all nested idents
        //  if (acc contains ident) then
        //    // the parent was brought in as foreign, keep this

        //


        acc
//        s ::acc.filterNot(x => x.symbol == i.symbol)

    }

    _ = println(s" -> foreign refs:         \n${externalRefs.map(_.show).mkString("\n").indent(4)}")
    _ = println(s" -> filtered  (found):         \n${valExternalRefs.map(x =>    s"${x.show} (${x.symbol}) ~> $x").mkString("\n").indent(4)}")
    _ = println(s" -> collapse  (found):         \n${collapsed.map(x =>   s"${x.symbol} ~> $x").mkString("\n").indent(4)}")


    c = Context(0, term :: Nil, Map(), Set())

    typedExternalRefs <- valExternalRefs.traverseFilter {
      case i @ Ident(_) =>
        resolveTpe(c)(i.tpe).map {
          case (Some(x), tpe, c) => Some((i, Reference(x, tpe), c))
          case (None, tpe, c)    => Some((i, Reference(i.symbol.name, tpe), c))
        }
      case s @ Select(term, name) =>
        // final vals in stdlib  :  Flags.{FieldAccessor, Final, Method, StableRealizable}
        // free methods          :  Flags.{Method}
        // free vals             :  Flags.{}
        // val ident = Ident.apply(TermRef.apply(s.tpe.dealias.simplified, "_ref_" + name + "_" + s.pos.startLine + "_"))
        // val alias = '{ val ${ident.asExpr} = ${s.asExpr} }

        println(s">>>!${s.symbol.flags.is(Flags.Method)}")

        if (s.symbol.flags.is(Flags.Method)) {}

        println(
          s"Select -> ${term} => `${name}` where: ${s.show} \n  select=${s.tpe}\n  tpe=   ${term.tpe}\n  sym=   ${s.symbol.tree}"
        )

//        println(s.)

        resolveTpe(c)(s.tpe).map {
          case (Some(x), tpe, c) => Some((s, Reference(x, tpe), c))
          case (None, tpe, c) =>
            Some((s, Reference("_ref_" + name + "_" + s.pos.startLine + "_", tpe), c))
        }
    }.resolve

    // Ref, Reference, Context

    capturedNames = typedExternalRefs
      .collect {
        case (r, Reference(name: String, tpe), c) if tpe != PolyAst.Type.Unit =>
          r -> PolyAst.Named(name, tpe)
      }
      .distinctBy(_._1.symbol.pos)

    // captuerdVars   = capturedIdents.traverse(i => resolveTpe(Nil)(i.tpe).map(i -> _._2)).resolve

    _ = println(
      s" -> all refs (typed):         \n${typedExternalRefs.map((a, b, _) => s"$b = ${a}").mkString("\n").indent(4)}"
    )
    _ = println(s" -> captured refs:    \n${capturedNames.map(_._2.repr).mkString("\n").indent(4)}")
    _ = println(s" -> body(long):\n${x.asTerm.show(using Printer.TreeAnsiCode).indent(4)}")

    // block <- term match {
    //   case Block(xs, x) => (xs :+ x).success // de-block
    //   case Return(x, sym) => List(x).success // can't really happen because return in lambda is illegal in Scala
    //   // Ref and Literal are the most common ones for the last
    //   case x => List(x).success
    //   // case bad             => s"block required, got $bad".fail
    // }
    (c, ref, stmts) <- resolveTrees(
      Context(0, Nil, typedExternalRefs.map((a, b, c) => (a, b)).toMap, typedExternalRefs.map(_._3.defs).flatten.toSet)
    )(term :: Nil).resolve

    (_, fnTpe, c) <- resolveTpe(c)(term.tpe).resolve

    //    _ = println(c)

    v <- c.defs.toList.traverse(s => lowerProductType(s.typeSymbol)).resolve
    _ = println(s"sdd=${v}")

    returnTerm = ref.getOrElse(PolyAst.Term.UnitConst)
    _ <-
      if (fnTpe != returnTerm.tpe) {
        s"lambda tpe ($fnTpe) != last term tpe (${returnTerm.tpe}), term was $returnTerm".fail
      } else ().success
    fnReturn = PolyAst.Stmt.Return(PolyAst.Expr.Alias(returnTerm))
    fnStmts  = stmts :+ fnReturn
    _        = println(s" -> PolyAsT:\n${fnStmts.map(_.repr).mkString("\n")}")

    args     = capturedNames.map(_._2)
    captures = capturedNames.map((r, n) => r -> n.tpe)

  } yield (captures, PolyAst.Function(closureName, args, fnTpe, stmts :+ fnReturn, v))

}
