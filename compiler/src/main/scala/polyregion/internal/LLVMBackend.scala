package polyregion.internal

import polyregion.LLVM_
import polyregion.Runtime.LibFfi.Type

import java.nio.ByteBuffer
import java.nio.file.{Files, Paths, StandardOpenOption}

object LLVMBackend {

  import polyregion.PolyAst.*

  def codegen(tree: Vector[Tree.Stmt], input: Path*)(range: Range, induction: String) = {

    val mod = new LLVM_.Module("a")
    import org.bytedeco.llvm.LLVM.{LLVMValueRef, LLVMTypeRef}
    import org.bytedeco.llvm.global.LLVM.*

    def tpe2llir(tpe: Types.Type): LLVMTypeRef =
      tpe match {
        case Types.ByteTpe()         => mod.i8
        case Types.ShortTpe()        => mod.i16
        case Types.IntTpe()          => mod.i32
        case Types.LongTpe()         => mod.i64
        case Types.FloatTpe()        => mod.float
        case Types.DoubleTpe()       => mod.double
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

    val args = input.map { case Path(name, tpe) => name -> tpe2llir(tpe) }

    mod.function("lambda", mod.void, args: _*) { case (params, fn, builder) =>
      def resolveRef(r: Refs.Ref, context: Map[String, LLVMValueRef]): LLVMValueRef = {
        println(">>>" + r.repr)
        r match {
          case r @ Refs.Select(Path(name, tpe), VNil() ) =>
            context.get(name) match {
              case Some(x) => x
              case None =>
                println(s"var not found: ${name}")
                ???
              // local var in fn

            }
          // if arg => use
          // else

          case Refs.Select(head, xs) => ???
          case Refs.BoolConst(v)     => ???
          case Refs.ByteConst(v)     => mod.constInt(mod.i8, v)
          case Refs.ShortConst(v)    => mod.constInt(mod.i16, v)
          case Refs.IntConst(v)      => mod.constInt(mod.i32, v)
          case Refs.LongConst(v)     => mod.constInt(mod.i64, v)
          case Refs.FloatConst(v)    => mod.constReal(mod.float, v)
          case Refs.DoubleConst(v)   => mod.constReal(mod.double, v)
          case Refs.CharConst(v)     => mod.constInt(mod.i8, v)
          case Refs.StringConst(v)   => ???
          case Refs.NullConst()   => ???
          case Refs.Ref.Empty      => ???
        }
      }

      def resolveExpr(e: Tree.Expr, key: String, context: Map[String, LLVMValueRef]): LLVMValueRef = {
        println(">" + e.repr)
        e match {
          case Tree.Invoke(lhs, "+", Vector(rhs), tpe @ (Types.FloatTpe() | Types.DoubleTpe())) =>
            if (lhs.tpe != tpe) {
              println(s"Cannot unify result ref ($tpe) with invoke($tpe)")
              ???
            }
            LLVMBuildFAdd(builder, resolveRef(lhs, context), resolveRef(rhs, context), s"${key}_+")

          case Tree.Invoke(lhs, "*", Vector(rhs), tpe @ (Types.FloatTpe() | Types.DoubleTpe())) =>
            if (lhs.tpe != tpe) {
              println(s"Cannot unify result ref ($tpe) with invoke($tpe)")
              ???
            }
            LLVMBuildFMul(builder, resolveRef(lhs, context), resolveRef(rhs, context), s"${key}_*")

          case Tree.Invoke(lhs, "apply", Vector(offset), tpe) =>
            // getelementptr; load
            val ptr = mod.gepInbound(builder, s"${key}_ptr")(resolveRef(lhs, context), resolveRef(offset, context))
            LLVMBuildLoad(builder, ptr, s"${key}_value")
          case Tree.Alias(ref) =>
            // load
            resolveRef(ref, context)
          //              LLVMBuildLoad(builder, resolveRef(ref, context), s"${key}_alias")
        }
      }

      def resolveOne(t: Tree.Stmt, context: Map[String, LLVMValueRef]): Map[String, LLVMValueRef] = {
        println(">" + t.repr)
        t match {
          case Tree.Comment(_) => context // discard
          case Tree.Effect(Refs.Select(Path(name, Types.ArrayTpe(component)), VNil()), "update", Vector(offset, value)) =>

            // getelementptr; store

            val ptr = mod.gepInbound(builder, s"${name}_ptr")(context(name), resolveRef(offset, context))
            LLVMBuildStore(builder, resolveRef(value, context), ptr)
            context
          case Tree.Var(key, tpe, rhs) =>
            // store <rhs>
            Map(key -> resolveExpr(rhs, key, context)) ++ context

          case Tree.Mut(lhs, ref)     => ???
          case Tree.While(cond, body) => ???

        }
      }

      mod.i32loop(builder, fn)(
        mod.constInt(mod.i32, range.start),
        mod.constInt(mod.i32, range.end),
        range.step,
        induction
      ) { n =>
        tree.foldLeft(params + (induction -> n)) { case (context, s: Tree.Stmt) =>
          resolveOne(s, context)
        }

      }

      //        tree.foreach {
      //          case e: polyregion.Runtime.PolyAst.Expr => resolveExpr(e, "???")
      //          case s: polyregion.Runtime.PolyAst.Stmt => resolveOne(s)
      //        }

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
