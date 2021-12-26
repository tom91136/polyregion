package polyregion.internal

import polyregion.LLVM_
import polyregion.Runtime.LibFfi.Type

import java.nio.ByteBuffer
import java.nio.file.{Files, Paths, StandardOpenOption}

object LLVMBackend {

  import polyregion.PolyAst.*

  def codegen(tree: Vector[Tree.Stmt], input: Named*) = {

    val mod = new LLVM_.Module("a")
    import org.bytedeco.llvm.LLVM.{LLVMValueRef, LLVMTypeRef}
    import org.bytedeco.llvm.global.LLVM.*

    def tpe2llir(tpe: Types.Type): LLVMTypeRef =
      tpe match {
        case Types.BoolTpe()                   => mod.i1
        case Types.ByteTpe()                   => mod.i8
        case Types.ShortTpe()                  => mod.i16
        case Types.IntTpe()                    => mod.i32
        case Types.LongTpe()                   => mod.i64
        case Types.FloatTpe()                  => mod.float
        case Types.DoubleTpe()                 => mod.double
        case Types.ArrayTpe(Types.ByteTpe())   => mod.ptr(mod.i8)
        case Types.ArrayTpe(Types.ShortTpe())  => mod.ptr(mod.i16)
        case Types.ArrayTpe(Types.IntTpe())    => mod.ptr(mod.i32)
        case Types.ArrayTpe(Types.LongTpe())   => mod.ptr(mod.i64)
        case Types.ArrayTpe(Types.FloatTpe())  => mod.ptr(mod.float)
        case Types.ArrayTpe(Types.DoubleTpe()) => mod.ptr(mod.double)
        case unknown =>
          println(s"???= $unknown")
          ???
      }

    val args = input.map { case Named(name, tpe) => name -> tpe2llir(tpe) }

    mod.function("lambda", mod.void, args: _*) { case (params, fn, builder) =>
      def loadOne(key: String, context: Map[String, LLVMValueRef]) =
        LLVMBuildLoad(builder, context(key), s"${key}_value")

      def resolveRef(r: Refs.Ref, context: Map[String, LLVMValueRef]): LLVMValueRef = {
        println(">resolveRef:" + r.repr)
        r match {
          case r @ Refs.Select(Named(name, tpe), VNil()) =>
            context.get(name) match {
              case Some(x) =>
                tpe match {
                  case Types.ArrayTpe(comp) =>
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

          case Refs.BoolConst(v)   => ???
          case Refs.ByteConst(v)   => mod.constInt(mod.i8, v)
          case Refs.ShortConst(v)  => mod.constInt(mod.i16, v)
          case Refs.IntConst(v)    => mod.constInt(mod.i32, v)
          case Refs.LongConst(v)   => mod.constInt(mod.i64, v)
          case Refs.FloatConst(v)  => mod.constReal(mod.float, v)
          case Refs.DoubleConst(v) => mod.constReal(mod.double, v)
          case Refs.CharConst(v)   => mod.constInt(mod.i8, v)
          case Refs.StringConst(v) => ???
          case Refs.NullConst()    => ???
          case Refs.Ref.Empty      => ???
        }
      }

      def resolveExpr(e: Tree.Expr, key: String, context: Map[String, LLVMValueRef]): LLVMValueRef = {
        println(s">  resolveExpr { ($key) :" + e.repr)
        val r = e match {
          case Tree.Invoke(lhs, "+", Vector(rhs), tpe @ (Types.FloatTpe() | Types.DoubleTpe())) =>
            if (lhs.tpe != tpe) {
              println(s"Cannot unify result ref ($tpe) with invoke($tpe)")
              ???
            }
            LLVMBuildFAdd(builder, resolveRef(lhs, context), resolveRef(rhs, context), s"${key}_+")
          case Tree.Invoke(lhs, "+", Vector(rhs), tpe @ (Types.IntTpe())) =>
            if (lhs.tpe != tpe) {
              println(s"Cannot unify result ref ($tpe) with invoke($tpe)")
              ???
            }
            LLVMBuildAdd(builder, resolveRef(lhs, context), resolveRef(rhs, context), s"${key}_+")

          case Tree.Invoke(lhs, "*", Vector(rhs), tpe @ (Types.FloatTpe() | Types.DoubleTpe())) =>
            if (lhs.tpe != tpe) {
              println(s"Cannot unify result ref ($tpe) with invoke($tpe)")
              ???
            }
            LLVMBuildFMul(builder, resolveRef(lhs, context), resolveRef(rhs, context), s"${key}_*")
          case Tree.Invoke(lhs, "<", Vector(rhs), tpe @ (Types.BoolTpe())) =>
            if (lhs.tpe != rhs.tpe || rhs.tpe != Types.IntTpe()) {
              println(s"Cannot unify result lhs (${lhs.tpe}) with rhs (${rhs.tpe}) for binary `<`")
              ???
            }
            LLVMBuildICmp(builder, LLVMIntSLT, resolveRef(lhs, context), resolveRef(rhs, context), s"${key}_<")
          case Tree.Index(lhs, idx, tpe) =>
            // getelementptr; load
            val ptr = mod.gepInbound(builder, s"${key}_ptr")(resolveRef(lhs, context), resolveRef(idx, context))
            LLVMBuildLoad(builder, ptr, s"${key}_value")

          // load
          //
          case Tree.Alias(ref) =>
            // load
            resolveRef(ref, context)
          // case Tree.Block(stmts, expr) =>
          //   resolveExpr(
          //     expr,
          //     "block",
          //     stmts.foldLeft(context) { case (c, s: Tree.Stmt) =>
          //       resolveStmt(s, c)
          //     }
          //   )

          //              LLVMBuildLoad(builder, resolveRef(ref, context), s"${key}_alias")
        }
        println(">  resolveExpr } :" + e.repr)
        r
      }

      def resolveStmt(t: Tree.Stmt, context: Map[String, LLVMValueRef]): (Map[String, LLVMValueRef]) = {
        println(">resolveStmt { :" + t.repr)
        val r = t match {
          case Tree.Comment(_) => context // discard

          case Tree.Update(Refs.Select(Named(name, Types.ArrayTpe(component)), VNil()), idx, value) =>
            // getelementptr; store

            // context(name) is ptr to array here, don't load
            val ptr = mod.gepInbound(builder, s"${name}_ptr")(context(name), resolveRef(idx, context))
            LLVMBuildStore(builder, resolveRef(value, context), ptr)
            context

          case Tree.Effect(
                Refs.Select(Named(name, Types.ArrayTpe(component)), VNil()),
                "update",
                Vector(offset, value)
              ) =>
            // getelementptr; store

            // context(name) is ptr to array here, don't load
            val ptr = mod.gepInbound(builder, s"${name}_ptr")(context(name), resolveRef(offset, context))
            LLVMBuildStore(builder, resolveRef(value, context), ptr)
            context
          case Tree.Var(Named(key, tpe), rhs) =>
            // store <rhs>

            tpe match {
              case Types.ArrayTpe(comp) =>
//                LLVMBuildStore(builder, resolveExpr(rhs, s"${key}_var_rhs", context), context(key))
                context + (key -> resolveExpr(rhs, s"${key}_var_rhs", context))

              case _ =>
                val allocate = LLVMBuildAlloca(builder, tpe2llir(tpe), s"${key}_stack_ptr")
                LLVMBuildStore(builder, resolveExpr(rhs, s"${key}_var_rhs", context), allocate)
                context + (key -> allocate)
            }

          case Tree.Mut(Refs.Select(Named(key, tpe), VNil()), ref) =>
            LLVMBuildStore(builder, resolveExpr(ref, s"${key}_mut", context), context(key))
            context
          case Tree.While(cond, body) =>
            val loopTest = LLVMAppendBasicBlock(fn, s"loop_test")
            val loopBody = LLVMAppendBasicBlock(fn, s"loop_body")
            val loopExit = LLVMAppendBasicBlock(fn, s"loop_exit")

            LLVMBuildBr(builder, loopTest) // goto loop_test:
            LLVMPositionBuilderAtEnd(builder, loopTest)
            val continue = resolveExpr(cond, "loop", context)
            LLVMBuildCondBr(builder, continue, loopBody, loopExit) // goto loop_test:

            LLVMPositionBuilderAtEnd(builder, loopBody)
            val ctx = body.foldLeft(context) { case (c, s: Tree.Stmt) =>
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
//        tree.foldLeft(params + (induction -> n)) { case (context, s: Tree.Stmt) =>
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
          case Types.ArrayTpe(comp) => m
          case _ =>
            val stackVar = LLVMBuildAlloca(builder, tpe2llir(tpe), s"${name}_stack_ptr")
            LLVMBuildStore(builder, m(name), stackVar)
            m + (name -> stackVar)
        }
      }

      tree.foldLeft(stackParams) { case (context, s: Tree.Stmt) =>
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
