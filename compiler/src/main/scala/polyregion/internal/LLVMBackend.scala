package polyregion.internal

import polyregion.LLVM_

import java.nio.ByteBuffer
import java.nio.file.{Files, Paths, StandardOpenOption}

object LLVMBackend {
  // import polyregion.Runtime.LibFfi.Type

  import polyregion.ast.PolyAst.*

  def codegen(tree: Vector[Stmt], input: Named*) = {

    val mod = new LLVM_.Module("a")
    import org.bytedeco.llvm.LLVM.{LLVMValueRef, LLVMTypeRef}
    import org.bytedeco.llvm.global.LLVM.*

    def tpe2llir(tpe: Type): LLVMTypeRef =
      tpe match {
        case Type.Bool               => mod.i1
        case Type.Byte               => mod.i8
        case Type.Short              => mod.i16
        case Type.Int                => mod.i32
        case Type.Long               => mod.i64
        case Type.Float              => mod.float
        case Type.Double             => mod.double
        case Type.Array(Type.Byte)   => mod.ptr(mod.i8)
        case Type.Array(Type.Short)  => mod.ptr(mod.i16)
        case Type.Array(Type.Int)    => mod.ptr(mod.i32)
        case Type.Array(Type.Long)   => mod.ptr(mod.i64)
        case Type.Array(Type.Float)  => mod.ptr(mod.float)
        case Type.Array(Type.Double) => mod.ptr(mod.double)
        case unknown =>
          println(s"???= $unknown")
          ???
      }

    val args = input.map { case Named(name, tpe) => name -> tpe2llir(tpe) }

    mod.function("lambda", mod.void, args: _*) { case (params, fn, builder) =>
      def loadOne(key: String, context: Map[String, LLVMValueRef]) =
        LLVMBuildLoad(builder, context(key), s"${key}_value")

      def resolveRef(r: Term, context: Map[String, LLVMValueRef]): LLVMValueRef = {
        println(">resolveRef:" + r.repr)
        r match {
          case r @ Term.Select(Nil, Named(name, tpe)) =>
            context.get(name) match {
              case Some(x) =>
                tpe match {
                  case Type.Array(comp) =>
                    println(s"Load array:$name:$tpe = ${x}")
                    x
                  case _ =>
                    println(s"Load ref from stack:$name:$tpe = ${x}")
                    loadOne(name, context)
                }
              case None =>
                println(s"var not found: ${name}")
                ???
              // local var in fn

            }
          // if arg => use
          // else

          case Term.BoolConst(v)   => ???
          case Term.ByteConst(v)   => mod.constInt(mod.i8, v)
          case Term.ShortConst(v)  => mod.constInt(mod.i16, v)
          case Term.IntConst(v)    => mod.constInt(mod.i32, v)
          case Term.LongConst(v)   => mod.constInt(mod.i64, v)
          case Term.FloatConst(v)  => mod.constReal(mod.float, v)
          case Term.DoubleConst(v) => mod.constReal(mod.double, v)
          case Term.CharConst(v)   => mod.constInt(mod.i8, v)
          case Term.StringConst(v) => ???
        }
      }

      def resolveExpr(e: Expr, key: String, context: Map[String, LLVMValueRef]): LLVMValueRef = {
        println(s">  resolveExpr { ($key) :" + e.repr)
        val r = e match {
          case Expr.Invoke(lhs, "+", rhs :: Nil, tpe @ (Type.Float | Type.Double)) =>
            if (lhs.tpe != tpe) {
              println(s"Cannot unify result ref ($tpe) with invoke($tpe)")
              ???
            }
            LLVMBuildFAdd(builder, resolveRef(lhs, context), resolveRef(rhs, context), s"${key}_+")
          case Expr.Invoke(lhs, "+", rhs :: Nil, tpe @ (Type.Int)) =>
            if (lhs.tpe != tpe) {
              println(s"Cannot unify result ref ($tpe) with invoke($tpe)")
              ???
            }
            LLVMBuildAdd(builder, resolveRef(lhs, context), resolveRef(rhs, context), s"${key}_+")

          case Expr.Invoke(lhs, "*", rhs :: Nil, tpe @ (Type.Float | Type.Double)) =>
            if (lhs.tpe != tpe) {
              println(s"Cannot unify result ref ($tpe) with invoke($tpe)")
              ???
            }
            LLVMBuildFMul(builder, resolveRef(lhs, context), resolveRef(rhs, context), s"${key}_*")
          case Expr.Invoke(lhs, "<", rhs :: Nil, tpe @ (Type.Bool)) =>
            if (lhs.tpe != rhs.tpe || rhs.tpe != Type.Int) {
              println(s"Cannot unify result lhs (${lhs.tpe}) with rhs (${rhs.tpe}) for binary `<`")
              ???
            }
            LLVMBuildICmp(builder, LLVMIntSLT, resolveRef(lhs, context), resolveRef(rhs, context), s"${key}_<")
          case Expr.Index(lhs, idx, tpe) =>
            // getelementptr; load
            val ptr = mod.gepInbound(builder, s"${key}_ptr")(resolveRef(lhs, context), resolveRef(idx, context))
            LLVMBuildLoad(builder, ptr, s"${key}_value")

          // load
          //
          case Expr.Alias(ref) =>
            // load
            resolveRef(ref, context)
          // case Tree.Block(stmts, expr) =>
          //   resolveExpr(
          //     expr,
          //     "block",
          //     stmts.foldLeft(context) { case (c, s: Stmt) =>
          //       resolveStmt(s, c)
          //     }
          //   )

          //              LLVMBuildLoad(builder, resolveRef(ref, context), s"${key}_alias")
        }
        println(">  resolveExpr } :" + e.repr)
        r
      }

      def resolveStmt(t: Stmt, context: Map[String, LLVMValueRef]): (Map[String, LLVMValueRef]) = {
        println(">resolveStmt { :" + t.repr)
        val r = t match {
          case Stmt.Comment(_) => context // discard

          case Stmt.Update(Term.Select(Nil, Named(name, Type.Array(component))), idx, value) =>
            // getelementptr; store

            // context(name) is ptr to array here, don't load
            val ptr = mod.gepInbound(builder, s"${name}_ptr")(context(name), resolveRef(idx, context))
            LLVMBuildStore(builder, resolveRef(value, context), ptr)
            context

          case Stmt.Effect(
                Term.Select(Nil, Named(name, Type.Array(component))),
                "update",
                offset :: value :: Nil
              ) =>
            // getelementptr; store

            // context(name) is ptr to array here, don't load
            val ptr = mod.gepInbound(builder, s"${name}_ptr")(context(name), resolveRef(offset, context))
            LLVMBuildStore(builder, resolveRef(value, context), ptr)
            context
          case Stmt.Var(Named(key, tpe), rhs) =>
            // store <rhs>

            tpe match {
              case Type.Array(comp) =>
//                LLVMBuildStore(builder, resolveExpr(rhs, s"${key}_var_rhs", context), context(key))
                context + (key -> resolveExpr(rhs, s"${key}_var_rhs", context))

              case _ =>
                val allocate = LLVMBuildAlloca(builder, tpe2llir(tpe), s"${key}_stack_ptr")
                LLVMBuildStore(builder, resolveExpr(rhs, s"${key}_var_rhs", context), allocate)
                context + (key -> allocate)
            }

          case Stmt.Mut(Term.Select(Nil, Named(key, tpe)), ref) =>
            LLVMBuildStore(builder, resolveExpr(ref, s"${key}_mut", context), context(key))
            context
          case Stmt.While(cond, body) =>
            val loopTest = LLVMAppendBasicBlock(fn, s"loop_test")
            val loopBody = LLVMAppendBasicBlock(fn, s"loop_body")
            val loopExit = LLVMAppendBasicBlock(fn, s"loop_exit")

            LLVMBuildBr(builder, loopTest) // goto loop_test:
            LLVMPositionBuilderAtEnd(builder, loopTest)
            val continue = resolveExpr(cond, "loop", context)
            LLVMBuildCondBr(builder, continue, loopBody, loopExit) // goto loop_test:

            LLVMPositionBuilderAtEnd(builder, loopBody)
            val ctx = body.foldLeft(context) { case (c, s: Stmt) =>
              resolveStmt(s, c)
            }
            LLVMBuildBr(builder, loopTest)

            LLVMPositionBuilderAtEnd(builder, loopExit)

            ctx

        }
        println(">resolveStmt } :" + t.repr)

        r
      }

//      mod.i32loop(builder, fn)(
//        mod.constInt(mod.i32, range.start),
//        mod.constInt(mod.i32, range.end),
//        range.step,
//        induction
//      ) { n =>
//        tree.foldLeft(params + (induction -> n)) { case (context, s: Stmt) =>
//          resolveStmt(s, context)
//        }
//
//      }

      //        tree.foreach {
      //          case e: polyregion.Runtime.PolyAst.Expr => resolveExpr(e, "???")
      //          case s: polyregion.Runtime.PolyAst.Stmt => resolveStmt(s)
      //        }

      println(s"Input:$params")

      val stackParams = input.foldLeft(params) { case (m, Named(name, tpe)) =>
        tpe match {
          case Type.Array(comp) => m
          case _ =>
            val stackVar = LLVMBuildAlloca(builder, tpe2llir(tpe), s"${name}_stack_ptr")
            LLVMBuildStore(builder, m(name), stackVar)
            m + (name -> stackVar)
        }
      }

      tree.foldLeft(stackParams) { case (context, s: Stmt) =>
        resolveStmt(s, context)
      }
      LLVMBuildRetVoid(builder)

    }
    mod.validate()
    mod.dump()
    mod.optimise()
    mod.dump()

    val buffer = LLVMWriteBitcodeToMemoryBuffer(mod.module)
    val start  = LLVMGetBufferStart(buffer)
    val end    = LLVMGetBufferSize(buffer)

    val arr = Array.ofDim[Byte](end.toInt)
    start.limit(end).get(arr)

    val chan =
      Files.newByteChannel(
        Paths.get("./obj_raw.bc").toAbsolutePath.normalize(),
        StandardOpenOption.WRITE,
        StandardOpenOption.CREATE,
        StandardOpenOption.TRUNCATE_EXISTING
      )
    chan.write(ByteBuffer.wrap(arr))
    chan.close()

    LLVMDisposeMemoryBuffer(buffer)

    arr
  }

}
