package polyregion

import polyregion.PolyAst

import scala.quoted.{Expr, Quotes}
import scala.quoted.*
import scala.annotation.tailrec
import scala.reflect.ClassTag
import fansi.ErrorMode.Throw

import java.nio.file.{Files, Paths, StandardOpenOption}
import cats.data.EitherT
import cats.Eval
import cats.data.NonEmptyList
import polyregion.PolyAst.Program
import polyregion.PolyAst.Refs.Ref
import polyregion.internal.*

import java.lang.reflect.Modifier
import polyregion.PolyAst.PolyAstProto

object compileTime {

  extension [A](xs: Array[A]) {
    inline def foreach(inline r: Range)(inline f: Int => Unit) =
      ${ offloadImpl('f) }
  }
  inline def foreachJVM(inline range: Range)(inline x: Int => Unit): Any =
    range.foreach(x)

  inline def foreach(range: Range)(inline x: Int => Unit): Any = {
    val start = range.start
    val bound = if (range.isInclusive) range.end else range.end - 1
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

    def resolveTpe(repr: TypeRepr): Deferred[PolyAst.Types.Type] = {

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
          } yield (name, xs.toVector) match {
            case (BufferTpe, Vector(component)) => PolyAst.Types.ArrayTpe(component)
            case (n, ys)                        => PolyAst.Types.RefTpe(n, ys)
          }
        case expr =>
          resolveSym(expr).map {
            case PolyAst.Sym(VNil("scala", "Unit"))          => PolyAst.Types.Type.Empty
            case PolyAst.Sym(VNil("scala", "Boolean"))       => PolyAst.Types.BoolTpe()
            case PolyAst.Sym(VNil("scala", "Byte"))          => PolyAst.Types.ByteTpe()
            case PolyAst.Sym(VNil("scala", "Short"))         => PolyAst.Types.ShortTpe()
            case PolyAst.Sym(VNil("scala", "Int"))           => PolyAst.Types.IntTpe()
            case PolyAst.Sym(VNil("scala", "Long"))          => PolyAst.Types.LongTpe()
            case PolyAst.Sym(VNil("scala", "Float"))         => PolyAst.Types.FloatTpe()
            case PolyAst.Sym(VNil("scala", "Double"))        => PolyAst.Types.DoubleTpe()
            case PolyAst.Sym(VNil("scala", "Char"))          => PolyAst.Types.CharTpe()
            case PolyAst.Sym(VNil("java", "lang", "String")) => PolyAst.Types.StringTpe()
            case sym                                         => PolyAst.Types.RefTpe(sym, VNil())
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

    def lower(x: Expr[Any]): (Array[Byte], Vector[(Ident, PolyAst.Types.Type)], PolyAst.Tree.Function) = {

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
        val captuerdVars = capturedIdents.toVector.traverse(i => resolveTpe(i.tpe).map(i -> _))

//        val indexArgument = closure.termParamss match {
//          case TermParamClause(ValDef(name, tree, None) :: Nil) :: Nil =>
//            resolveTpe(tree.tpe).subflatMap { tpe =>
//              if (tpe == PolyAst.Types.IntTpe()) (name -> tpe).success
//              else s"${tpe} != ${PolyAst.Types.IntTpe()}".fail
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
          case Nil => (depth, VNil(), VNil()).success.deferred
          case x :: xs =>
            resolveTerm(x, depth)
              .map((n, ref, xs) => (n, Vector(ref), xs))
              .flatMap(xs.foldLeftM(_) { case ((prev, rs, tss), term) =>
                resolveTerm(term, prev).map((curr, r, ts) => (curr, rs :+ r, ts ++ tss))
              })
        }

        def resolveTrees(depth: Int, args: List[Tree]) = args match {
          case Nil => (depth, VNil()).success.deferred
          case x :: xs =>
            resolveTree(x, depth).flatMap(xs.foldLeftM(_) { case ((prev, tss), term) =>
              resolveTree(term, prev).map((curr, ts) => (curr, ts ++ tss))
            })
        }

        def resolveTerm(term: Term, depth: Int = 0): Deferred[(Int, PolyAst.Refs.Ref, Vector[PolyAst.Tree.Stmt])] =
          term match {
            case i @ Ident(name) =>
              resolveTpe(i.tpe).map { tpe =>
                val named = PolyAst.Named(i.symbol.name, tpe)

                val tree = if (i.symbol.name != name) {
                  println(s"Name aliased = Seen name `${name}` = ${i.symbol.name}")
                  // return new name, create alias back to original:

                  // Vector(
                  //   PolyAst.Tree
                  //     .Var(named, PolyAst.Tree.Alias(PolyAst.Refs.Select(PolyAst.Named(i.symbol.name, tpe), VNil())))
                  // )
                  VNil()
                } else VNil()

              (depth, PolyAst.Refs.Select(named, VNil()), tree)
              }
            case c @ Literal(BooleanConstant(v)) => ((depth, PolyAst.Refs.BoolConst(v), VNil())).success.deferred
            case c @ Literal(IntConstant(v))     => ((depth, PolyAst.Refs.IntConst(v), VNil())).success.deferred
            case c @ Literal(FloatConstant(v))   => ((depth, PolyAst.Refs.FloatConst(v), VNil())).success.deferred
            case c @ Literal(DoubleConstant(v))  => ((depth, PolyAst.Refs.DoubleConst(v), VNil())).success.deferred
            case c @ Literal(LongConstant(v))    => ((depth, PolyAst.Refs.LongConst(v), VNil())).success.deferred
            case c @ Literal(CharConstant(v))    => ((depth, PolyAst.Refs.CharConst(v.toInt), VNil())).success.deferred
            case c @ Literal(UnitConstant())     => ((depth, PolyAst.Refs.Ref.Empty, VNil())).success.deferred
            case ap @ Apply(Select(qualifier, name), args) =>
              for {
                tpe <- resolveTpe(ap.tpe)
                (lhsDepth, lhsRef, lhsTrees) <- resolveTerm(qualifier, depth + 1) // go down here
                (maxDepth, argRefs, argTrees) <- resolveTerms(lhsDepth, args)

                path = PolyAst.Named(s"v${depth}", tpe)
                tree = // don't save a ref for unit methods
                  if (tpe == PolyAst.Types.Type.Empty) PolyAst.Tree.Effect(lhsRef, name, argRefs)
                  else PolyAst.Tree.Var(path, PolyAst.Tree.Invoke(lhsRef, name, argRefs, tpe))
              } yield (maxDepth, PolyAst.Refs.Select(path, VNil()), ((argTrees ++ lhsTrees) :+ tree))
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
                r <- lhsRef match {
                  case PolyAst.Refs.Select(name, VNil()) =>
                    (
                      maxDepth,
                      PolyAst.Refs.Ref.Empty,
                      lhsTrees ++ rhsTrees :+ PolyAst.Tree.Mut(name, PolyAst.Tree.Alias(rhsRef))
                    ).success.deferred
                  case bad => s"Illegal assign LHS: ${bad}".fail.deferred
                }
              } yield r
            case wd @ While(cond, body) =>
              for {
                (condDepth, condRef, condTrees) <- resolveTerm(cond, depth + 1)
                (maxDepth, _, bodyTrees)        <- resolveTerm(body, condDepth)
              } yield {
                val block = condTrees match {
                  case VNil() => ??? // this is illegal, while needs a bool predicate
                  case xs     => PolyAst.Tree.Block(xs, PolyAst.Tree.Alias(condRef))
                }
                (maxDepth, PolyAst.Refs.Ref.Empty, Vector(PolyAst.Tree.While(block, bodyTrees)))
              }
            case _ => s"[$depth] Unhandled: $term".fail.deferred
          }

        def resolveTree(c: Tree, depth: Int = 0): Deferred[(Int, Vector[PolyAst.Tree.Stmt])] = c match {
          case ValDef(name, tpe, Some(rhs)) =>
            for {
              t                  <- resolveTpe(tpe.tpe)
              (depth, ref, tree) <- resolveTerm(rhs, depth)
            } yield (depth, tree :+ PolyAst.Tree.Var(PolyAst.Named(name, t), PolyAst.Tree.Alias(ref)))
          case ValDef(name, tpe, None) => s"Unexpected variable $name:$tpe".fail.deferred
          case t: Term                 => resolveTerm(t, depth).map((depth, ref, tree) => (depth, tree)) // discard ref
        }

        block match {
          case Block(stat, expr) =>
            pprint.pprintln(stat :+ expr)

            val out = (stat :+ expr).foldLeftM((0, VNil.empty[PolyAst.Tree.Stmt])) { case ((prev, ts), stmt) =>
              resolveTree(stmt, prev).map((curr, tss) =>
                (curr, (ts :+ PolyAst.Tree.Comment(stmt.show.replaceAll("\n", ""))) ++ tss)
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
              LLVMBackend.codegen(stmts, args*),
              identArgs,
              PolyAst.Tree.Function(closureName, args, PolyAst.Types.Type.Empty, stmts)
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
    val r                   = new Resolver(using q)
    val (bytes, args, prog) = r.lower(x)

    Files.write(
      Paths.get("./ast.bin").toAbsolutePath.normalize(),
      prog.toByteArray,
      StandardOpenOption.WRITE,
      StandardOpenOption.CREATE,
      StandardOpenOption.TRUNCATE_EXISTING
    )

//    println(prog.toByteArray.mkString(" "))
//    println(Program.parseFrom(prog.toByteArray).toProtoString)

    val bs = Expr(bytes)

    val argExprs = args.map { (name, tpe) =>
      val expr = name.asExpr
      tpe match {
        case PolyAst.Types.ByteTpe() => //
          '{ Runtime.Buffer[Byte](${ expr.asExprOf[Byte] }).pointer -> Runtime.LibFfi.Type.SInt8 }
        case PolyAst.Types.ShortTpe() => //
          '{ Runtime.Buffer[Short](${ expr.asExprOf[Short] }).pointer -> Runtime.LibFfi.Type.SInt16 }
        case PolyAst.Types.IntTpe() => //
          '{ Runtime.Buffer[Int](${ expr.asExprOf[Int] }).pointer -> Runtime.LibFfi.Type.SInt32 }
        case PolyAst.Types.LongTpe() => //
          '{ Runtime.Buffer[Long](${ expr.asExprOf[Long] }).pointer -> Runtime.LibFfi.Type.SInt64 }
        case PolyAst.Types.FloatTpe() => //
          '{ Runtime.Buffer[Float](${ expr.asExprOf[Float] }).pointer -> Runtime.LibFfi.Type.Float }
        case PolyAst.Types.DoubleTpe() => //
          '{ Runtime.Buffer[Double](${ expr.asExprOf[Double] }).pointer -> Runtime.LibFfi.Type.Double }

        case PolyAst.Types.ArrayTpe(PolyAst.Types.ByteTpe()) =>
          '{ ${ expr.asExprOf[Runtime.Buffer[Byte]] }.pointer -> Runtime.LibFfi.Type.Ptr }
        case PolyAst.Types.ArrayTpe(PolyAst.Types.ShortTpe()) =>
          '{ ${ expr.asExprOf[Runtime.Buffer[Short]] }.pointer -> Runtime.LibFfi.Type.Ptr }
        case PolyAst.Types.ArrayTpe(PolyAst.Types.IntTpe()) =>
          '{ ${ expr.asExprOf[Runtime.Buffer[Int]] }.pointer -> Runtime.LibFfi.Type.Ptr }
        case PolyAst.Types.ArrayTpe(PolyAst.Types.LongTpe()) =>
          '{ ${ expr.asExprOf[Runtime.Buffer[Long]] }.pointer -> Runtime.LibFfi.Type.Ptr }
        case PolyAst.Types.ArrayTpe(PolyAst.Types.FloatTpe()) =>
          '{ ${ expr.asExprOf[Runtime.Buffer[Float]] }.pointer -> Runtime.LibFfi.Type.Ptr }
        case PolyAst.Types.ArrayTpe(PolyAst.Types.DoubleTpe()) =>
          '{ ${ expr.asExprOf[Runtime.Buffer[Double]] }.pointer -> Runtime.LibFfi.Type.Ptr }
        case unknown =>
          println(s"???= $unknown ")
          ???
      }

    }

    val argsParams = Varargs(argExprs)
    val astBytes   = Expr(prog.toByteArray)

    '{
      val data = $bs
      println("LLVM bytes=" + data.size)
      println("PolyAst bytes=" + ${ astBytes }.size)

//      for (n <- $range)
//        $x(n)

      Runtime.ingest(data, $argsParams: _*)

    }
  }

}
