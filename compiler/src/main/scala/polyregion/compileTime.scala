package polyregion

import polyregion.PolyAst
import scala.quoted.{Expr, Quotes}
import scala.quoted.*
import scala.annotation.tailrec
import scala.reflect.ClassTag
import fansi.ErrorMode.Throw
import java.nio.file.Paths
import cats.data.EitherT
import cats.Eval
import cats.data.NonEmptyList

import polyregion.internal._

object compileTime {


  extension [A](xs : Array[A]){
    inline def foreach(inline r : Range)(inline f : Int => Unit) = {

      ${ foreachImpl('r)('f) }
    }
  }
  inline def foreachJVM(inline range: Range)(inline x: Int => Unit): Any = {
    range.foreach(x)
  }

  inline def foreach(inline range: Range)(inline x: Int => Unit): Any =
    ${ foreachImpl('range)('x) }

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

  class Resolver(using val q: Quotes) {

    import quotes.reflect.*
    import cats.syntax.all.*

    def extractClosureBody(term: Term): Result[DefDef] = term match {
      case Block(List(defStmt: DefDef), Closure(Ident(name), _)) =>
        if (defStmt.name == name) defStmt.success
        else s"Closure name mismatch: def block was `${defStmt.name}` but closure is named `${name}`".fail
      case bad => s"Illegal closure tree: ${bad}".fail
    }

    def extractInlineBlock(term: Term): Result[Term] = term match {
      case Inlined(_, _, Block(_, term)) => term.success
      case x                             => s"Illegal top-level inline tree: ${x}".fail
    }

    def resolveTpe(repr: TypeRepr): Deferred[PolyAst.Type] = {

      @tailrec def resolveSym(ref: TypeRepr): Result[PolyAst.Sym] = ref match {
        case ThisType(tpe) => resolveSym(tpe)
        case tpe: NamedType =>
          tpe.classSymbol match {
            case None => s"Named type is not a class: ${tpe}".fail
            case Some(sym) if sym.name == "<root>" => // discard root package
              resolveSym(tpe.qualifier)
            case Some(sym) =>
              PolyAst.Sym(sym.fullName).success
          }
        // case NoPrefix()    => None.success
        case invalid => s"Invalid type ${invalid}".fail
      }

      repr.dealias.widenTermRefByName.simplified match {
        case tpe @ AppliedType(ctor, args) =>
          for {
            name <- resolveSym(ctor).deferred
            xs   <- args.traverse(resolveTpe(_))
          } yield PolyAst.Type(name, xs.toVector)
        case expr => resolveSym(expr).map(PolyAst.Type(_, VNil)).deferred
      }
    }

    def collectTree[A](in: Tree)(f: PartialFunction[Tree, List[A]]) = {
      val acc = new TreeAccumulator[List[A]] {
        def foldTree(xs: List[A], tree: Tree)(owner: Symbol): List[A] =
          f.andThen(_ ::: xs).applyOrElse(tree, foldOverTree(xs, _)(owner))
      }
      acc.foldOverTree(Nil, in)(Symbol.noSymbol)
    }

    // class ClosureCaptureAccumulator(name: String) extends TreeAccumulator[List[Ident]] {
    //   def foldTree(xs: List[Ident], tree: Tree)(owner: Symbol): List[Ident] = tree match {
    //     case ident: Ident =>
    //       val owner = ident.symbol.owner
    //       if (owner.flags.is(Flags.Method) && owner.name == name) xs
    //       else ident :: xs
    //     case _ => foldOverTree(xs, tree)(owner)
    //   }
    // }

    def lower(range: Expr[Range])(x: Expr[Int => Unit]): (Array[Byte], List[(Ident, PolyAst.Type)]) = {

      val terms = x.asTerm
      println(s"foreach@${terms.pos}")

      (for {
        block   <- extractInlineBlock(terms)
        closure <- extractClosureBody(block)
      } yield {

        // println(s"${PolyAst.Primitives.All}")

        val closureName = closure.name

        val captures = collectTree[Ident](closure) { case ident: Ident =>
          val owner = ident.symbol.owner
          if (owner.flags.is(Flags.Method) && owner.name == closureName) Nil else ident :: Nil
        }

        val capturedIdents = captures
          .distinctBy(_.symbol.pos)
        val captuerdVars = capturedIdents.traverse(i => resolveTpe(i.tpe).map(i -> _))

        val indexArgument = closure.termParamss match {
          case TermParamClause(ValDef(name, tree, None) :: Nil) :: Nil =>
            resolveTpe(tree.tpe).subflatMap { tpe =>
              if (tpe == PolyAst.Primitives.Int) (name -> tpe).success
              else s"${tpe} != ${PolyAst.Primitives.Int}".fail
            }
          case bad => s"Illegal index parameter pattern ${bad}".fail.deferred
        }
        println(s"========${closureName}=========")
        println(Paths.get(".").toAbsolutePath)
        println(s"-> JVM args:" + java.lang.management.ManagementFactory.getRuntimeMXBean().getInputArguments)
        println(s"-> JVM name:" + java.lang.management.ManagementFactory.getRuntimeMXBean().getName)
        println(s" -> name:    ${closureName}")
        println(s" -> arg:     ${indexArgument.map((l, r) => l -> r.repr).resolve}")
        println(s" -> captures (names):   ${capturedIdents.map(_.name)}")
        println(
          s"    ${captuerdVars.map(xs => xs.map((n, t) => s"${n.name}: ${t.repr}").mkString("\n    ")).resolve}"
        )
        println(Printer.TreeShortCode.getClass)
        println(s" -> body(short):\n${closure.show(using Printer.TreeShortCode).indent(4)}")
        println(s" -> body(long):\n${closure.show(using Printer.TreeAnsiCode).indent(4)}")

        val idents = collectTree[Ident](closure) { case i @ Ident(name) => i :: Nil }
        println(idents)
        // pprint.pprintln(closure)

        def resolveTerm(term: Term, depth: Int = 0): Deferred[(Int, PolyAst.Ref, Vector[PolyAst.Stmt])] = term match {
          case i @ Ident(name) =>
            resolveTpe(i.tpe).map(tpe => (depth, PolyAst.Select(PolyAst.Path(name, tpe), VNil), VNil))
          case c @ Literal(BooleanConstant(v)) => ((depth, PolyAst.BoolConst(v), VNil)).success.deferred
          case c @ Literal(IntConstant(v))     => ((depth, PolyAst.IntConst(v), VNil)).success.deferred
          case c @ Literal(FloatConstant(v))   => ((depth, PolyAst.FloatConst(v), VNil)).success.deferred
          case c @ Literal(DoubleConstant(v))  => ((depth, PolyAst.DoubleConst(v), VNil)).success.deferred
          case c @ Literal(LongConstant(v))    => ((depth, PolyAst.LongConst(v), VNil)).success.deferred
          case c @ Literal(CharConstant(v))    => ((depth, PolyAst.CharConst(v.toInt), VNil)).success.deferred
          case c @ Literal(UnitConstant())     => ((depth, PolyAst.UnitConst(), VNil)).success.deferred
          case ap @ Apply(Select(qualifier, name), args) =>
            for {
              tpe <- resolveTpe(ap.tpe)
              (lhsDepth, lhsRef, lhsTrees) <- resolveTerm(qualifier, depth + 1) // go down here
              (maxDepth, argRefs, argTrees) <- args match {
                case Nil => (0, VNil, VNil).success.deferred
                case x :: xs =>
                  resolveTerm(x, lhsDepth)
                    .map((n, ref, xs) => (n, Vector(ref), xs))
                    .flatMap(xs.foldLeftM(_) { case ((prev, rs, tss), term) =>
                      resolveTerm(term, prev).map((curr, r, ts) => (curr, rs :+ r, ts ++ tss))
                    })
              }

              path = PolyAst.Path(s"v${depth}", tpe)
              tree = // don't save a ref for unit methods
                if (tpe == PolyAst.Primitives.Unit) PolyAst.Stmt.Effect(lhsRef, name, argRefs)
                else PolyAst.Stmt.Var(path.name, tpe, PolyAst.Expr.Invoke(lhsRef, name, argRefs, tpe))
            } yield (maxDepth, PolyAst.Ref.Select(path), ((argTrees ++ lhsTrees) :+ tree))
          case _ => s"[$depth] Unhandled: $term".fail.deferred
        }

        def resolveTree(c: Tree, depth: Int = 0): Deferred[(Int, Vector[PolyAst.Stmt])] = c match {
          case ValDef(name, tpe, Some(rhs)) =>
            for {
              t                  <- resolveTpe(tpe.tpe)
              (depth, ref, tree) <- resolveTerm(rhs, depth)
            } yield (depth, tree :+ PolyAst.Stmt.Var(name, t, PolyAst.Expr.Alias(ref)))
          case ValDef(name, tpe, None) => s"Unexpected variable $name:$tpe".fail.deferred
          case t: Term                 => resolveTerm(t, depth).map((depth, ref, tree) => (depth, tree)) // discard ref
        }

        closure.rhs match {
          case Some(Block(stat, expr)) =>
            pprint.pprintln(stat :+ expr)

            val out = (stat :+ expr).foldLeftM((0, VNil[PolyAst.Stmt])) { case ((prev, ts), stmt) =>
              resolveTree(stmt, prev).map((curr, tss) => (curr, (ts :+ PolyAst.Stmt.Comment(stmt.show)) ++ tss))
            }

            println(out.map(x => x._2.mkString("\n")).resolve.fold(_.getMessage, identity))

            val x = for {
              stmts     <- out.map(_._2)
              args      <- captuerdVars.map(_.map((id, tpe) => PolyAst.Path(id.name, tpe)))
              identArgs <- captuerdVars.map(_.map((id, tpe) => (id, tpe)))
              idx       <- indexArgument.map(_._1)
            } yield (PolyAst.llirCodegen(stmts, args*)(0 to 10, idx), identArgs)
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

          case None => ???
        }
      }).right.get

    }
  }

  def foreachImpl(range: Expr[Range])(x: Expr[Int => Unit])(using q: Quotes): Expr[Any] = {
    import quotes.reflect.*
    val r             = new Resolver(using q)
    val (bytes, args) = r.lower(range)(x)

    val bs = Expr(bytes)

    val argExprs = args.map { (name, tpe) =>
      val expr = name.asExpr
      tpe match {
        case PolyAst.Primitives.Byte => //
          '{ Runtime.Buffer[Byte](${ expr.asExprOf[Byte] }).pointer -> Runtime.LibFfi.Type.SInt8 }
        case PolyAst.Primitives.Short => //
          '{ Runtime.Buffer[Short](${ expr.asExprOf[Short] }).pointer -> Runtime.LibFfi.Type.SInt16 }
        case PolyAst.Primitives.Int => //
          '{ Runtime.Buffer[Int](${ expr.asExprOf[Int] }).pointer -> Runtime.LibFfi.Type.SInt32 }
        case PolyAst.Primitives.Long => //
          '{ Runtime.Buffer[Long](${ expr.asExprOf[Long] }).pointer -> Runtime.LibFfi.Type.SInt64 }
        case PolyAst.Primitives.Float => //
          '{ Runtime.Buffer[Float](${ expr.asExprOf[Float] }).pointer -> Runtime.LibFfi.Type.Float }
        case PolyAst.Primitives.Double => //
          '{ Runtime.Buffer[Double](${ expr.asExprOf[Double] }).pointer -> Runtime.LibFfi.Type.Double }

        case PolyAst.Intrinsics.ByteBuffer =>
          '{ ${ expr.asExprOf[Runtime.Buffer[Byte]] }.pointer -> Runtime.LibFfi.Type.Ptr }
        case PolyAst.Intrinsics.ShortBuffer =>
          '{ ${ expr.asExprOf[Runtime.Buffer[Short]] }.pointer -> Runtime.LibFfi.Type.Ptr }
        case PolyAst.Intrinsics.IntBuffer =>
          '{ ${ expr.asExprOf[Runtime.Buffer[Int]] }.pointer -> Runtime.LibFfi.Type.Ptr }
        case PolyAst.Intrinsics.LongBuffer =>
          '{ ${ expr.asExprOf[Runtime.Buffer[Long]] }.pointer -> Runtime.LibFfi.Type.Ptr }
        case PolyAst.Intrinsics.FloatBuffer =>
          '{ ${ expr.asExprOf[Runtime.Buffer[Float]] }.pointer -> Runtime.LibFfi.Type.Ptr }
        case PolyAst.Intrinsics.DoubleBuffer =>
          '{ ${ expr.asExprOf[Runtime.Buffer[Double]] }.pointer -> Runtime.LibFfi.Type.Ptr }
        case unknown =>
          println(s"???= $unknown ")
          ???
      }

    }

    val argsParams = Varargs(argExprs)

    '{
      val data = $bs
      println("LLVM bytes=" + data.size)
//      for (n <- $range)
//        $x(n)

      Runtime.ingest(data, $argsParams: _*)

    }
  }

  def showExprImpl(x: Expr[Any])(using q: Quotes): Expr[Any] = {
    import quotes.reflect.*
    given Printer[Tree] = Printer.TreeStructure

    println("CP:" + System.getProperty("java.class.path"))

    println("[RAW TREE]: " + x.show) // tree => source
    pprint.pprintln(x.asTerm)        // term => AST

    // val resolver = new Resolver(using q)

    // resolver.extractInlineBlock(x.asTerm)

    // println(">>>!"+ x.asTerm ) // tree => source

    val FloatTpe = TypeRepr.of[Float]
    val IntTpe   = TypeRepr.of[Int]

    val FloatArrayTpe = TypeRepr.of[Array[Float]]

    // var sym = Symbol.spliceOwner

    // pprint.pprintln("~~ "+sym.owner.fullName)
    // pprint.pprintln("~~ "+sym.owner.owner.tree)

    // MyTraverser().traverseTree(sym.tree)(Symbol.noSymbol)

    // pprint.pprintln("s="+x.asTerm.pos.startLine + ":" +x.asTerm.pos.startColumn + " e="+ x.asTerm.pos.endLine + ":" +x.asTerm.pos.endColumn   )

    // println(x.asTerm.pos.sourceFile)

// AppliedType(TypeRef(TermRef(ThisType(TypeRef(NoPrefix,module class <root>)),object scala),Array),List(TypeRef(TermRef(ThisType(TypeRef(NoPrefix,module class <root>)),object scala),Float))) #MACRO DEf

// AppliedType(TypeRef(ThisType(TypeRef(NoPrefix,module class scala)),class Array),List(TypeRef(ThisType(TypeRef(NoPrefix,module class scala)),class Float))) # inferred
// AppliedType(TypeRef(ThisType(TypeRef(NoPrefix,module class scala)),class Array),List(TypeRef(TermRef(ThisType(TypeRef(NoPrefix,module class <root>)),object scala),class Float))) #annotated
// AppliedType(TypeRef(ThisType(TypeRef(NoPrefix,module class scala)),class Array),List(TypeRef(TermRef(ThisType(TypeRef(NoPrefix,module class <root>)),object scala),class Float)))

    import cats.syntax.all.*
    import cats.Eval

    def resolveTpe(repr: TypeRepr): Eval[Result[PolyAst.Type]] = {

      @tailrec def resolveSym(ref: TypeRepr): Result[PolyAst.Sym] = ref match {
        case ThisType(tpe) => resolveSym(tpe)
        case tpe: NamedType =>
          tpe.classSymbol match {
            case None => s"Named type is not a class: ${tpe}".fail
            case Some(sym) if sym.name == "<root>" => // discard root package
              resolveSym(tpe.qualifier)
            case Some(sym) =>
              PolyAst.Sym(sym.fullName).success
          }
        // case NoPrefix()    => None.success
        case invalid => s"Invalid type ${invalid}".fail
      }

      repr.dealias.widenTermRefByName.simplified match {
        case tpe @ AppliedType(ctor, args) =>
          args.traverse(resolveTpe(_)).map { xs =>
            for {
              name   <- resolveSym(ctor)
              params <- xs.sequence
            } yield PolyAst.Type(name, params.toVector)
          }
        case expr => Eval.now(resolveSym(expr).map(PolyAst.Type(_, VNil)))
      }
    }

    // this.scala.X
    // this.<root>.scala.X

    println(">>>>" + FloatArrayTpe.widenTermRefByName.simplified.dealias)
    println(">>>>" + IntTpe.widenTermRefByName.simplified.dealias)

    class MyTraverser extends TreeTraverser {
      override def traverseTree(tree: Tree)(owner: Symbol): Unit = {
        println(s"Tree> ${tree.show} ${tree.symbol}")

        tree match {
          case i @ Ident(name) =>
            println(s"\t-->${name} : ${i.symbol.owner}")
            println(s"\t-->${i.tpe.widenTermRefByName.simplified.dealias}")
            println(s"\t-->${resolveTpe(i.tpe).value.map(_.repr)}")

          case _ =>
        }

        traverseTreeChildren(tree)(owner)
      }
    }

    // xs.update(n, xs.apply(n).+(ys.apply(n).*(scalar))))
    MyTraverser().traverseTree(x.asTerm)(Symbol.noSymbol)
    // println(x.asTerm)

    // pprint.pprintln(Symbol.classSymbol("hps.Stage$").tree)

    // x.asTerm match{
    //   case Inlined(_, _, Inlined(_, _, Block(_, Block(List(DefDef(_, _, _, Some(preRhs))), _)))) =>

    //     preRhs match {
    //       case Block(List(ValDef(_, _, Some(x))), _) => pprint.pprintln(""+x.symbol.tree)
    //     }

    //     pprint.pprintln(preRhs)
    // }

    x
    //    given Printer[Tree] = Printer.TreeStructure
    //    x.show
  }

}
