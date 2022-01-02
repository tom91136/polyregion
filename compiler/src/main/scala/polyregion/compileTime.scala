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

object compileTime {

  extension [A](xs: Array[A]) {
    inline def foreach(inline r: Range)(inline f: Int => Unit) =
      ${ offloadImpl('f) }
  }
  inline def foreachJVM(inline range: Range)(inline x: Int => Unit): Any =
    range.foreach(x)

  inline def foreach(range: Range)(inline x: Int => Unit): Any = {
    val start = range.start
    val bound = if (range.isInclusive) range.end - 1 else range.end
    val step  = range.step
    offload {
      var i = start
      while (i < bound) {
        x(i)
        i += step
      }
    }
  }

  inline def offload(inline x: Any): Any =
    ${ offloadImpl('x) }

  inline def showExpr(inline x: Any): Any =
    ${ showExprImpl('x) }

  inline def showTpe[B]: Unit = {
    ${ showTpeImpl[B] }
    ()
  }

  def showTpeImpl[B: Type](using Quotes): Expr[Unit] = {
    import quotes.reflect.*
    println(TypeRepr.of[B].typeSymbol.tree.show)
    import pprint.*
    pprint.pprintln(TypeRepr.of[B].typeSymbol)
    '{}
  }

  def showExprImpl(x: Expr[Any])(using q: Quotes): Expr[Any] = {
    import quotes.reflect.*
    given Printer[Tree] = Printer.TreeStructure
    pprint.pprintln(x.asTerm) // term => AST
    x
  }

  class Resolver(using val q: Quotes) {

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

    val BufferTpe = PolyAst.Sym[Buffer[_]]

    def resolveTpe(repr: TypeRepr): Deferred[PolyAst.Type] = {

      @tailrec def resolveSym(ref: TypeRepr): Result[PolyAst.Sym] = ref match {
        case ThisType(tpe) => resolveSym(tpe)
        case tpe: NamedType =>
          tpe.classSymbol match {
            case None => s"Named type is not a class: ${tpe}".fail
            case Some(sym) if sym.name == "<root>" => // discard root package
              resolveSym(tpe.qualifier)
            case Some(sym) => PolyAst.Sym(sym.fullName).success
          }
        // case NoPrefix()    => None.success
        case invalid => s"Invalid type ${invalid}".fail
      }

      repr.dealias.widenTermRefByName.simplified match {
        case tpe @ AppliedType(ctor, args) =>
          for {
            name <- resolveSym(ctor).deferred
            xs   <- args.traverse(resolveTpe(_))
          } yield (name, xs) match {
            case (BufferTpe, component :: Nil) => PolyAst.Type.Array(component)
            case (n, ys)                       => PolyAst.Type.Struct(n, ys)
          }
        case expr =>
          resolveSym(expr).map {
            case PolyAst.Sym("scala" :: "Unit" :: Nil)            => PolyAst.Type.Unit
            case PolyAst.Sym("scala" :: "Boolean" :: Nil)         => PolyAst.Type.Bool
            case PolyAst.Sym("scala" :: "Byte" :: Nil)            => PolyAst.Type.Byte
            case PolyAst.Sym("scala" :: "Short" :: Nil)           => PolyAst.Type.Short
            case PolyAst.Sym("scala" :: "Int" :: Nil)             => PolyAst.Type.Int
            case PolyAst.Sym("scala" :: "Long" :: Nil)            => PolyAst.Type.Long
            case PolyAst.Sym("scala" :: "Float" :: Nil)           => PolyAst.Type.Float
            case PolyAst.Sym("scala" :: "Double" :: Nil)          => PolyAst.Type.Double
            case PolyAst.Sym("scala" :: "Char" :: Nil)            => PolyAst.Type.Char
            case PolyAst.Sym("java" :: "lang" :: "String" :: Nil) => PolyAst.Type.String
            case sym                                              => PolyAst.Type.Struct(sym, Nil)
          }.deferred
      }
    }

    def collectTree[A](in: Tree)(f: PartialFunction[Tree, List[A]]) = {
      val acc = new TreeAccumulator[List[A]] {
        def foldTree(xs: List[A], tree: Tree)(owner: Symbol): List[A] =
          f.andThen(_ ::: xs).applyOrElse(tree, foldOverTree(xs, _)(owner))
      }
      acc.foldOverTree(Nil, in)(Symbol.noSymbol)
    }

    def lower(x: Expr[Any]): (Array[Byte], List[(Ident, PolyAst.Type)], PolyAst.Function) = {

      val terms = x.asTerm
      println(s"foreach@${terms.pos}")

      (for {
        block <- extractInlineBlock(terms)
//        _ = pprint.pprintln(terms)
//        closure <- extractClosureBody(block)
      } yield {

        val closureName = block.pos.toString

        val captures = collectTree[Ident](block) { case ident: Ident =>
          val owner = ident.symbol.owner
          //FIXME TODO make sure the owner is actually this macro, and not any other macro
          if (owner.flags.is(Flags.Macro)) Nil else ident :: Nil

        }

        val capturedIdents = captures
          .distinctBy(_.symbol.pos)
        val captuerdVars = capturedIdents.traverse(i => resolveTpe(i.tpe).map(i -> _))

//        val indexArgument = closure.termParamss match {
//          case TermParamClause(ValDef(name, tree, None) :: Nil) :: Nil =>
//            resolveTpe(tree.tpe).subflatMap { tpe =>
//              if (tpe == PolyAst.Type.Int) (name -> tpe).success
//              else s"${tpe} != ${PolyAst.Type.Int}".fail
//            }
//          case bad => s"Illegal index parameter pattern ${bad}".fail.deferred
//        }

        println(s"========${closureName}=========")
        println(Paths.get(".").toAbsolutePath)
        println(s"-> JVM args:" + java.lang.management.ManagementFactory.getRuntimeMXBean().getInputArguments)
        println(s"-> JVM name:" + java.lang.management.ManagementFactory.getRuntimeMXBean().getName)
        println(s" -> name:    ${closureName}")
//        println(s" -> arg:     ${indexArgument.map((l, r) => l -> r.repr).resolve}")
        println(s" -> captures (names):   ${capturedIdents.map(_.name)}")
        println(
          s"    ${captuerdVars.map(xs => xs.map((n, t) => s"${n.name}: ${t.repr}").mkString("\n    ")).resolve}"
        )
        println(Printer.TreeShortCode.getClass)
        println(s" -> body(short):\n${block.show(using Printer.TreeShortCode).indent(4)}")
        println(s" -> body(long):\n${block.show(using Printer.TreeAnsiCode).indent(4)}")

        val idents = collectTree[Ident](block) { case i @ Ident(name) => i :: Nil }
        println(idents)
        // pprint.pprintln(closure)

        def resolveTerms(depth: Int, args: List[Term]) = args match {
          case Nil => (depth, Nil, Nil).success.deferred
          case x :: xs =>
            resolveTerm(x, depth)
              .map((n, ref, xs) => (n, ref :: Nil, xs))
              .flatMap(xs.foldLeftM(_) { case ((prev, rs, tss), term) =>
                resolveTerm(term, prev).map((curr, r, ts) => (curr, rs :+ r, ts ++ tss))
              })
        }

        def resolveTrees(depth: Int, args: List[Tree]): Deferred[(Int, List[PolyAst.Stmt])] = args match {
          case Nil => (depth, Nil).success.deferred
          case x :: xs =>
            resolveTree(x, depth).flatMap(xs.foldLeftM(_) { case ((prev, tss), term) =>
              resolveTree(term, prev).map((curr, ts) => (curr, tss ++ ts))
            })
        }

        def resolveTerm(term: Term, depth: Int = 0): Deferred[(Int, Option[PolyAst.Term], List[PolyAst.Stmt])] =
          term match {
            case i @ Ident(name) =>
              resolveTpe(i.tpe).map { tpe =>
                val named = PolyAst.Named(i.symbol.name, tpe)

                val tree = if (i.symbol.name != name) {
                  println(s"Name aliased = Seen name `${name}` = ${i.symbol.name}")
                  // return new name, create alias back to original:

                  // Vector(
                  //   PolyAst.Tree
                  //     .Var(named, PolyAst.Tree.Alias(PolyAst.Term.Select(PolyAst.Named(i.symbol.name, tpe), VNil())))
                  // )
                  Nil
                } else Nil

                (depth, Some(PolyAst.Term.Select(Nil, named)), tree)
              }
            case c @ Literal(BooleanConstant(v)) => ((depth, Some(PolyAst.Term.BoolConst(v)), Nil)).success.deferred
            case c @ Literal(IntConstant(v))     => ((depth, Some(PolyAst.Term.IntConst(v)), Nil)).success.deferred
            case c @ Literal(FloatConstant(v))   => ((depth, Some(PolyAst.Term.FloatConst(v)), Nil)).success.deferred
            case c @ Literal(DoubleConstant(v))  => ((depth, Some(PolyAst.Term.DoubleConst(v)), Nil)).success.deferred
            case c @ Literal(LongConstant(v))    => ((depth, Some(PolyAst.Term.LongConst(v)), Nil)).success.deferred
            case c @ Literal(CharConstant(v))    => ((depth, Some(PolyAst.Term.CharConst(v)), Nil)).success.deferred
            case c @ Literal(UnitConstant())     => ???
            case ap @ Apply(Select(qualifier, name), args) =>
              for {
                tpe <- resolveTpe(ap.tpe)
                (lhsDepth, lhsRef, lhsTrees) <- resolveTerm(qualifier, depth + 1) // go down here
                (maxDepth, argRefs, argTrees) <- resolveTerms(lhsDepth, args)

                path = PolyAst.Named(s"v${depth}", tpe)
                // tree = // don't save a ref for unit methods
                //   if (tpe == PolyAst.Type.Empty) PolyAst.Tree.Effect(lhsRef, name, argRefs)
                //   else PolyAst.Stmt.Var(path, PolyAst.Tree.Invoke(lhsRef, name, argRefs, tpe))
                existingTree = (argTrees ++ lhsTrees)
                lhsSelect <- lhsRef match {
                  case Some(s @ PolyAst.Term.Select(_, _)) => s.success.deferred
                  case bad                                 => s"Illegal LHS for apply: ${bad}".fail.deferred
                }

              } yield (name, lhsSelect, argRefs, tpe) match {
                case (
                      "apply",
                      select,
                      Some(idx) :: Nil,
                      tpe
                    ) => //TODO check idx.tpe =:= Int
                  val tree = PolyAst.Stmt.Var(path, PolyAst.Expr.Index(select, idx, tpe))
                  val ref  = PolyAst.Term.Select(Nil, path)
                  (maxDepth, Some(ref), existingTree :+ tree)
                case (
                      "update",
                      select,
                      Some(idx) :: Some(value) :: Nil,
                      tpe
                    ) => //TODO check idx.tpe =:= Int && value.tpe =:= lhs.tpe
                  val tree = PolyAst.Stmt.Update(select, idx, value)
                  (maxDepth, None, existingTree :+ tree)
                case (_, lhs, _, PolyAst.Type.Unit) =>
                  val tree = PolyAst.Stmt.Effect(lhsSelect, name, argRefs.flatten)
                  (maxDepth, None, existingTree :+ tree)
                case (name, lhs, Nil, tpe) => // unary op
                  val term = name match {
                    case "!" => PolyAst.Expr.Inv(lhs)
                  }
                  val tree = PolyAst.Stmt.Var(path, term)
                  val ref  = PolyAst.Term.Select(Nil, path)
                  (maxDepth, Some(ref), existingTree :+ tree)
                case (name, lhs, Some(rhs) :: Nil, tpe) => // binary op
                  val term = name match {
                    case "+" => PolyAst.Expr.Add(lhs, rhs, tpe)
                    case "-" => PolyAst.Expr.Sub(lhs, rhs, tpe)
                    case "*" => PolyAst.Expr.Mul(lhs, rhs, tpe)
                    case "/" => PolyAst.Expr.Div(lhs, rhs, tpe)
                    case "%" => PolyAst.Expr.Mod(lhs, rhs, tpe)
                    case "<" => PolyAst.Expr.Lt(lhs, rhs)
                    case ">" => PolyAst.Expr.Gt(lhs, rhs)
                  }
                  val tree = PolyAst.Stmt.Var(path, term)
                  val ref  = PolyAst.Term.Select(Nil, path)
                  (maxDepth, Some(ref), existingTree :+ tree)
                case _ =>
                  val tree =
                    PolyAst.Stmt.Var(path, PolyAst.Expr.Invoke(lhsSelect, name, argRefs.flatten, tpe))
                  val ref = PolyAst.Term.Select(Nil, path)
                  (maxDepth, Some(ref), existingTree :+ tree)
              }
            case Inlined(None, Nil, expansion) => resolveTerm(expansion, depth) // simple-inline
            case Block(stat, expr) => // stat : List[Statement]
              for {
                (statDepth, statTrees) <- resolveTrees(depth, stat)
                (exprDepth, ref, tree) <- resolveTerm(expr, statDepth)
              } yield (exprDepth, ref, statTrees ++ tree)

            case Assign(lhs, rhs) =>
              for {
                (lhsDepth, lhsRef, lhsTrees) <- resolveTerm(lhs, depth + 1) // go down here
                (maxDepth, rhsRef, rhsTrees) <- resolveTerm(rhs, lhsDepth)
                r <- (lhsRef, rhsRef) match {
                  case (Some(s @ PolyAst.Term.Select(Nil, _)), Some(rhs)) =>
                    (
                      maxDepth,
                      None,
                      lhsTrees ++ rhsTrees :+ PolyAst.Stmt.Mut(s, PolyAst.Expr.Alias(rhs))
                    ).success.deferred
                  case bad => s"Illegal assign LHS,RHS: ${bad}".fail.deferred
                }
              } yield r
            case wd @ While(cond, body) =>
              for {
                (condDepth, condRef, condTrees) <- resolveTerm(cond, depth + 1)
                (maxDepth, _, bodyTrees)        <- resolveTerm(body, condDepth)
              } yield {
                val block = condTrees match {
                  case Nil                            => ??? // this is illegal, while needs a bool predicate
                  case PolyAst.Stmt.Var(_, iv) :: Nil =>
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
                (maxDepth, None, block :: Nil)
              }
            case _ => s"[$depth] Unhandled: $term".fail.deferred
          }

        def resolveTree(c: Tree, depth: Int = 0): Deferred[(Int, List[PolyAst.Stmt])] = c match {
          case ValDef(name, tpe, Some(rhs)) =>
            for {
              t                  <- resolveTpe(tpe.tpe)
              (depth, ref, tree) <- resolveTerm(rhs, depth)
            } yield (depth, tree :+ PolyAst.Stmt.Var(PolyAst.Named(name, t), PolyAst.Expr.Alias(ref.get)))
          case ValDef(name, tpe, None) => s"Unexpected variable $name:$tpe".fail.deferred
          case t: Term                 => resolveTerm(t, depth).map((depth, ref, tree) => (depth, tree)) // discard ref
        }

        block match {
          case Block(stat, expr) =>
            pprint.pprintln(stat :+ expr)

            val out = (stat :+ expr).foldLeftM((0, List.empty[PolyAst.Stmt])) { case ((prev, ts), stmt) =>
              resolveTree(stmt, prev).map((curr, tss) =>
                (curr, (ts :+ PolyAst.Stmt.Comment(stmt.show.replaceAll("\n", ""))) ++ tss)
              )
            }

            println(">>>")
            println(out.map(x => (x._2.map(_.repr)).repr.mkString("\n")).resolve.fold(_.getMessage, identity))
            println("<<<")

            val x = for {
              stmts     <- out.map(_._2)
              args      <- captuerdVars.map(_.map((id, tpe) => PolyAst.Named(id.name, tpe)))
              identArgs <- captuerdVars.map(_.map((id, tpe) => (id, tpe)))
//              idx       <- indexArgument.map(_._1)
            } yield (
              LLVMBackend.codegen(stmts.toVector, args*),
              identArgs,
              PolyAst.Function(closureName, args, PolyAst.Type.Unit, stmts)
            )
            x.resolve match {
              case Left(e)  => throw e
              case Right(x) => x

            }

//            PolyAst.llirCodegen(
//              out.map(_._2).resolve.right.get,
//              (captuerdVars.map(_.map((id, tpe) => PolyAst.Path(id.name, tpe))).resolve.right.get
//                // indexArgument.map(PolyAst.Path(_, _)).resolve.right.get
//                ): _*
//            )(0 to 10, indexArgument.map(_._1).resolve.right.get)

          case _ => ???
        }
      }) match {
        case Left(e) =>
          throw e
        case Right(x) => x
      }

    }
  }

  def offloadImpl(x: Expr[Any])(using q: Quotes): Expr[Any] = {
    import quotes.reflect.*
    val r                 = new Resolver(using q)
    val (bytes, args, fn) = r.lower(x)

    val astBytes = MsgPack.encode(fn)

    Files.write(
      Paths.get("./ast.bin").toAbsolutePath.normalize(),
      astBytes,
      StandardOpenOption.WRITE,
      StandardOpenOption.CREATE,
      StandardOpenOption.TRUNCATE_EXISTING
    )

//    println(prog.toByteArray.mkString(" "))
//    println(Program.parseFrom(prog.toByteArray).toProtoString)

    val bs = Expr(bytes)

//    val b = '{ val b = polyregion.Runtime.FFIInvocationBuilder() }

    def argExprs(b: Expr[polyregion.Runtime.FFIInvocationBuilder]) = args.map { (name, tpe) =>
      val expr = name.asExpr
      tpe match {
        case PolyAst.Type.Byte => //
          '{ $b.arg(${ expr.asExprOf[Byte] }) }
        case PolyAst.Type.Short => //
          '{ $b.arg(${ expr.asExprOf[Short] }) }
        case PolyAst.Type.Int => //
          '{ $b.arg(${ expr.asExprOf[Int] }) }
        case PolyAst.Type.Long => //
          '{ $b.arg(${ expr.asExprOf[Long] }) }
        case PolyAst.Type.Float => //
          '{ $b.arg(${ expr.asExprOf[Float] }) }
        case PolyAst.Type.Double => //
          '{ $b.arg(${ expr.asExprOf[Double] }) }

        case PolyAst.Type.Array(PolyAst.Type.Byte) =>
          '{ $b.arg(${ expr.asExprOf[Runtime.Buffer[Byte]] }.buffer) }
        case PolyAst.Type.Array(PolyAst.Type.Short) =>
          '{ $b.arg(${ expr.asExprOf[Runtime.Buffer[Short]] }.buffer) }
        case PolyAst.Type.Array(PolyAst.Type.Int) =>
          '{ $b.arg(${ expr.asExprOf[Runtime.Buffer[Int]] }.buffer) }
        case PolyAst.Type.Array(PolyAst.Type.Long) =>
          '{ $b.arg(${ expr.asExprOf[Runtime.Buffer[Long]] }.buffer) }
        case PolyAst.Type.Array(PolyAst.Type.Float) =>
          '{ $b.arg(${ expr.asExprOf[Runtime.Buffer[Float]] }.buffer) }
        case PolyAst.Type.Array(PolyAst.Type.Double) =>
          '{ $b.arg(${ expr.asExprOf[Runtime.Buffer[Double]] }.buffer) }
        case unknown =>
          println(s"???= $unknown ")
          ???
      }
    }

    val astBytesExpr = Expr(astBytes)

    '{
      val data = $bs
      println("LLVM bytes=" + data.size)
      println("PolyAst bytes=" + ${ astBytesExpr }.size)

      val b = polyregion.Runtime.FFIInvocationBuilder()
      ${ Varargs(argExprs('{ b })) }
      Runtime.ingest(data, b.invoke(_))

//      for (n <- $range)
//        $x(n)

//      Runtime.ingest(data, $argsParams: _*)

    }
  }

}
