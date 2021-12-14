package polyregion

import cats.data.EitherT

import java.lang.reflect.Modifier
import scala.annotation.tailrec
import scala.reflect.ClassTag
import scala.collection.mutable

object Runtime {

  import org.bytedeco.javacpp.{Pointer, PointerPointer}

  private[polyregion] val NullPtr                  = new Pointer()
  private[polyregion] def ptrToAddress(addr: Long) = new Pointer() { address = addr }

  trait Buffer[T] extends mutable.IndexedSeq[T] {
    def deallocate(): Unit
    def pointer: Pointer
  }

  object Buffer {

    import org.bytedeco.javacpp.{
      BytePointer,
      CharPointer,
      DoublePointer,
      FloatPointer,
      IntPointer,
      LongPointer,
      ShortPointer
    }

    class DoubleBuffer(val buffer: DoublePointer) extends Buffer[Double] {
      override def update(idx: Int, elem: Double): Unit = buffer.put(idx.toLong, elem)
      override def apply(i: Int): Double                = buffer.get(i.toLong)
      override def length: Int                          = buffer.capacity().toInt
      override def deallocate(): Unit                   = buffer.deallocate()
      override val pointer: Pointer                     = buffer
    }
    class FloatBuffer(val buffer: FloatPointer) extends Buffer[Float] {
      override def update(idx: Int, elem: Float): Unit = buffer.put(idx.toLong, elem)
      override def apply(i: Int): Float                = buffer.get(i.toLong)
      override def length: Int                         = buffer.capacity().toInt
      override def deallocate(): Unit                  = buffer.deallocate()
      override val pointer: Pointer                    = buffer
    }
    class LongBuffer(val buffer: LongPointer) extends Buffer[Long] {
      override def update(idx: Int, elem: Long): Unit = buffer.put(idx.toLong, elem)
      override def apply(i: Int): Long                = buffer.get(i.toLong)
      override def length: Int                        = buffer.capacity().toInt
      override def deallocate(): Unit                 = buffer.deallocate()
      override val pointer: Pointer                   = buffer
    }
    class IntBuffer(val buffer: IntPointer) extends Buffer[Int] {
      override def update(idx: Int, elem: Int): Unit = buffer.put(idx.toLong, elem)
      override def apply(i: Int): Int                = buffer.get(i.toLong)
      override def length: Int                       = buffer.capacity().toInt
      override def deallocate(): Unit                = buffer.deallocate()
      override val pointer: Pointer                  = buffer
    }
    class ShortBuffer(val buffer: ShortPointer) extends Buffer[Short] {
      override def update(idx: Int, elem: Short): Unit = buffer.put(idx.toLong, elem)
      override def apply(i: Int): Short                = buffer.get(i.toLong)
      override def length: Int                         = buffer.capacity().toInt
      override def deallocate(): Unit                  = buffer.deallocate()
      override val pointer: Pointer                    = buffer
    }
    class ByteBuffer(val buffer: BytePointer) extends Buffer[Byte] {
      override def update(idx: Int, elem: Byte): Unit = buffer.put(idx.toLong, elem)
      override def apply(i: Int): Byte                = buffer.get(i.toLong)
      override def length: Int                        = buffer.capacity().toInt
      override def deallocate(): Unit                 = buffer.deallocate()
      override val pointer: Pointer                   = buffer
    }
    class CharBuffer(val buffer: CharPointer) extends Buffer[Char] {
      override def update(idx: Int, elem: Char): Unit = buffer.put(idx.toLong, elem)
      override def apply(i: Int): Char                = buffer.get(i.toLong)
      override def length: Int                        = buffer.capacity().toInt
      override def deallocate(): Unit                 = buffer.deallocate()
      override val pointer: Pointer                   = buffer
    }

    def apply[T <: AnyVal](xs: T*)(using tag: ClassTag[T]): Buffer[T] = tag.runtimeClass match {
      case java.lang.Double.TYPE =>
        new DoubleBuffer(new DoublePointer(xs.toArray.asInstanceOf[Array[Double]]: _*)).asInstanceOf[Buffer[T]]
      case java.lang.Float.TYPE =>
        new FloatBuffer(new FloatPointer(xs.toArray.asInstanceOf[Array[Float]]: _*)).asInstanceOf[Buffer[T]]
      case java.lang.Long.TYPE =>
        new LongBuffer(new LongPointer(xs.toArray.asInstanceOf[Array[Long]]: _*)).asInstanceOf[Buffer[T]]
      case java.lang.Integer.TYPE =>
        new IntBuffer(new IntPointer(xs.toArray.asInstanceOf[Array[Int]]: _*)).asInstanceOf[Buffer[T]]
      case java.lang.Short.TYPE =>
        new ShortBuffer(new ShortPointer(xs.toArray.asInstanceOf[Array[Short]]: _*)).asInstanceOf[Buffer[T]]
      case java.lang.Byte.TYPE =>
        new ByteBuffer(new BytePointer(xs.toArray.asInstanceOf[Array[Byte]]: _*)).asInstanceOf[Buffer[T]]
      case java.lang.Character.TYPE =>
        new CharBuffer(new CharPointer(xs.toArray.asInstanceOf[Array[Char]]: _*)).asInstanceOf[Buffer[T]]
    }

    def ofDim[T <: AnyVal](dim: Long)(using tag: ClassTag[T]): Buffer[T] = tag.runtimeClass match {
      case java.lang.Double.TYPE    => new DoubleBuffer(new DoublePointer(dim)).asInstanceOf[Buffer[T]]
      case java.lang.Float.TYPE     => new FloatBuffer(new FloatPointer(dim)).asInstanceOf[Buffer[T]]
      case java.lang.Long.TYPE      => new LongBuffer(new LongPointer(dim)).asInstanceOf[Buffer[T]]
      case java.lang.Integer.TYPE   => new IntBuffer(new IntPointer(dim)).asInstanceOf[Buffer[T]]
      case java.lang.Short.TYPE     => new ShortBuffer(new ShortPointer(dim)).asInstanceOf[Buffer[T]]
      case java.lang.Byte.TYPE      => new ByteBuffer(new BytePointer(dim)).asInstanceOf[Buffer[T]]
      case java.lang.Character.TYPE => new CharBuffer(new CharPointer(dim)).asInstanceOf[Buffer[T]]
    }

    def ofDim[T <: AnyVal: ClassTag](dim: Int): Buffer[T] = ofDim[T](dim.toLong)

  }

  object PolyAst {

    case class Sym(fqcn: List[String]) {
      def repr: String = fqcn.mkString(".")
    }

    object Sym {
      def apply(raw: String): Sym = {
        require(!raw.isBlank)
        // normalise dollar
        Sym(raw.split('.').toList)
      }
    }
    case class Type(sym: Sym, args: List[Type]) {
      def repr: String = args match {
        case Nil => sym.repr
        case xs  => s"${sym.repr}[${xs.map(_.repr).mkString(",")}]"
      }
      def args(xs: Type*): Type = copy(args = xs.toList)
      def ctor: Type            = copy(args = Nil)
    }
    object Type {

      def apply[T <: AnyRef](using tag: ClassTag[T]): Type = {
        // normalise naming differences
        // Java        => package.Companion$Member
        // Scala Macro => package.Companion$.Member
        @tailrec def go(cls: Class[_], xs: List[String] = Nil, companion: Boolean = false): List[String] = {
          val name = cls.getSimpleName + (if (companion) "$" else "")
          cls.getEnclosingClass match {
            case null => cls.getPackageName :: name :: xs
            case c    => go(c, name :: xs, Modifier.isStatic(cls.getModifiers))
          }
        }
        Type(Sym(go(tag.runtimeClass)), Nil)
      }

      // XXX we can't do [T: ClassTag] becase it resolves to the unboxed class
      def apply(name: String): Type = try {
        Class.forName(name) // resolve it first to make sure it's actually there
        Type(Sym(name), Nil)
      } catch { t => throw new AssertionError(s"Cannot resolve ${name} for Type constant: ${t.getMessage}") }

    }

    case class StructDef(
        members: List[(String, Type)]
        //TODO methods
    )

    object Primitives {
      val Unit    = Type("scala.Unit")
      val Boolean = Type("scala.Boolean")
      val Byte    = Type("scala.Byte")
      val Short   = Type("scala.Short")
      val Int     = Type("scala.Int")
      val Long    = Type("scala.Long")
      val Float   = Type("scala.Float")
      val Double  = Type("scala.Double")
      val Char    = Type("scala.Char")
      val String  = Type("java.lang.String")
      val All     = List(Unit, Boolean, Byte, Short, Int, Long, Float, Double, Char, String)
    }

    object Intrinsics {
      val DoubleBuffer = Type[Buffer[_]].args(Primitives.Double)
      val FloatBuffer  = Type[Buffer[_]].args(Primitives.Float)
      val LongBuffer   = Type[Buffer[_]].args(Primitives.Long)
      val IntBuffer    = Type[Buffer[_]].args(Primitives.Int)
      val ShortBuffer  = Type[Buffer[_]].args(Primitives.Short)
      val ByteBuffer   = Type[Buffer[_]].args(Primitives.Byte)
      val CharBuffer   = Type[Buffer[_]].args(Primitives.Char)
      def Buffer       = Type[Buffer[_]]

    }

    case class Path(name: String, tpe: Type) {
      def repr: String = s"($name:${tpe.repr})"
    }

    enum Ref(show: => String, val tpe: Type) {
      case Select(head: Path, tail: List[Path] = Nil)
          extends Ref((head :: tail).map(_.repr).mkString("."), tail.lastOption.getOrElse(head).tpe)
      case BoolConst(value: Boolean) extends Ref(s"Boolean(`$value)`", Primitives.Boolean)
      case ByteConst(value: Byte) extends Ref(s"Byte(`$value)`", Primitives.Byte)
      case ShortConst(value: Short) extends Ref(s"Short(`$value)`", Primitives.Short)
      case IntConst(value: Int) extends Ref(s"Int(`$value)`", Primitives.Int)
      case LongConst(value: Long) extends Ref(s"Long(`$value)`", Primitives.Long)
      case FloatConst(value: Float) extends Ref(s"Float(`$value)`", Primitives.Float)
      case DoubleConst(value: Double) extends Ref(s"Double(`$value)`", Primitives.Double)
      case CharConst(value: Char) extends Ref(s"Char(`$value`)", Primitives.Char)
      case StringConst(value: String) extends Ref(s"String(`$value`)", Primitives.String)
      case UnitConst() extends Ref("()", Primitives.Unit)
      case NullConst(resolved: Type)
          extends Ref(s"(null: ${resolved.repr})", resolved) // null is Nothing which will be concrete after Typer?
      def repr: String              = show
      override def toString: String = repr

    }

    sealed trait Tree {
      def repr: String
      override def toString: String = repr
    }

    enum Expr(show: => String, tpe: Type) extends Tree {
      case Alias(ref: Ref) extends Expr(s"(~>${ref.repr})", ref.tpe)
      case Invoke(lhs: Ref, name: String, args: Vector[Ref], tpe: Type)
          extends Expr(s"${lhs.repr}<$name>(${args.map(_.repr).mkString(",")})", tpe)
      def repr: String = show
    }

    enum Stmt(show: => String) extends Tree {
      case Comment(value: String) extends Stmt(s" // $value") // discard at backend

      case Var(key: String, tpe: Type, rhs: Expr) extends Stmt(s"var $key : ${tpe.repr} = ${rhs.repr}")
      case Effect(lhs: Ref, name: String, args: Vector[Ref])
          extends Stmt(s"${lhs.repr}<$name>(${args.map(_.repr).mkString(",")}) : Unit")
      case Mut(lhs: Ref, ref: Expr) extends Stmt(s"${lhs.repr} := ${ref.repr}")
      case While(cond: Expr, body: Vector[Tree])
          extends Stmt(s"while(${cond.repr}{\n${body.map(_.repr).mkString("\n")}\n}")
//      case Block(exprs: List[Tree]) extends Stmt(exprs.map(_.repr).mkString("{\n", "\n", "\n}"))
      def repr: String = show
    }

    def llirCodegen(tree: Vector[Stmt], input: Path*)(range : Range, induction: String) = {

      val mod = new LLVM_.Module("a")
      import org.bytedeco.llvm.LLVM.{LLVMValueRef, LLVMTypeRef}
      import org.bytedeco.llvm.global.LLVM.*

      def tpe2llir(tpe: Type): LLVMTypeRef =
        tpe match {
          case Primitives.Byte         => mod.i8
          case Primitives.Short        => mod.i16
          case Primitives.Int          => mod.i32
          case Primitives.Long         => mod.i64
          case Primitives.Float        => mod.float
          case Primitives.Double       => mod.double
          case Intrinsics.ByteBuffer   => mod.ptr(mod.i8)
          case Intrinsics.ShortBuffer  => mod.ptr(mod.i16)
          case Intrinsics.IntBuffer    => mod.ptr(mod.i32)
          case Intrinsics.LongBuffer   => mod.ptr(mod.i64)
          case Intrinsics.FloatBuffer  => mod.ptr(mod.float)
          case Intrinsics.DoubleBuffer => mod.ptr(mod.double)
          case unknown =>
            println(s"???= $unknown :=: ${Intrinsics.FloatBuffer}")
            ???
        }

      val args = input.map { case Path(name, tpe) => name -> tpe2llir(tpe) }

      mod.function("lambda", mod.void, args: _*) { case (params, fn, builder) =>
        def resolveRef(r: Ref, context: Map[String, LLVMValueRef]): LLVMValueRef = {
          println(">>>" + r.repr)
          r match {
            case r @ Ref.Select(Path(name, tpe), Nil) =>
              context.get(name) match {
                case Some(x) => x
                case None =>
                  println(s"var not found: ${name}")
                  ???
                // local var in fn

              }
            // if arg => use
            // else

            case Ref.Select(head, xs) => ???
            case Ref.BoolConst(v)     => ???
            case Ref.ByteConst(v)     => mod.constInt(mod.i8, v)
            case Ref.ShortConst(v)    => mod.constInt(mod.i16, v)
            case Ref.IntConst(v)      => mod.constInt(mod.i32, v)
            case Ref.LongConst(v)     => mod.constInt(mod.i64, v)
            case Ref.FloatConst(v)    => mod.constReal(mod.float, v)
            case Ref.DoubleConst(v)   => mod.constReal(mod.double, v)
            case Ref.CharConst(v)     => mod.constInt(mod.i8, v)
            case Ref.StringConst(v)   => ???
            case Ref.UnitConst()      => ???
            case Ref.NullConst(tpe)   => ???
          }
        }

        def resolveExpr(e: Expr, key: String, context: Map[String, LLVMValueRef]): LLVMValueRef = {
          println(">" + e.repr)
          e match {
            case Expr.Invoke(lhs, "+", Vector(rhs), tpe @ (Primitives.Float | Primitives.Double)) =>
              if (lhs.tpe != tpe) {
                println(s"Cannot unify result ref ($tpe) with invoke($tpe)")
                ???
              }
              LLVMBuildFAdd(builder, resolveRef(lhs, context), resolveRef(rhs, context), s"${key}_+")

            case Expr.Invoke(lhs, "*", Vector(rhs), tpe @ (Primitives.Float | Primitives.Double)) =>
              if (lhs.tpe != tpe) {
                println(s"Cannot unify result ref ($tpe) with invoke($tpe)")
                ???
              }
              LLVMBuildFMul(builder, resolveRef(lhs, context), resolveRef(rhs, context), s"${key}_*")

            case Expr.Invoke(lhs, "apply", Vector(offset), tpe) =>
              // getelementptr; load
              val ptr = mod.gepInbound(builder, s"${key}_ptr")(resolveRef(lhs, context), resolveRef(offset, context))
              LLVMBuildLoad(builder, ptr, s"${key}_value")
            case Expr.Alias(ref) =>
              // load
              resolveRef(ref, context)
//              LLVMBuildLoad(builder, resolveRef(ref, context), s"${key}_alias")
          }
        }

        def resolveOne(t: Stmt, context: Map[String, LLVMValueRef]): Map[String, LLVMValueRef] = {
          println(">" + t.repr)
          t match {
            case Stmt.Comment(_) => context // discard
            case Stmt.Effect(Ref.Select(Path(name, tpe), Nil), "update", Vector(offset, value))
                if tpe.ctor == Intrinsics.Buffer =>
              // getelementptr; store

              val ptr = mod.gepInbound(builder, s"${name}_ptr")(context(name), resolveRef(offset, context))
              LLVMBuildStore(builder, resolveRef(value, context), ptr)
              context
            case Stmt.Var(key, tpe, rhs) =>
              // store <rhs>
              Map(key -> resolveExpr(rhs, key, context)) ++ context

            case Stmt.Mut(lhs, ref)     => ???
            case Stmt.While(cond, body) => ???

          }
        }



        mod.i32loop(builder, fn)(mod.constInt(mod.i32, range.start),mod.constInt(mod.i32, range.end), range.step, induction){
          n =>
            tree.foldLeft(params) { case (context, s: polyregion.Runtime.PolyAst.Stmt) =>
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

    }

  }

  def ingest(ast: PolyAst.Tree, captures: Map[String, Buffer[?]]) =
    ???

  private[polyregion] object LibFfi {
    import org.bytedeco.libffi.ffi_cif
    import org.bytedeco.libffi.ffi_type
    import org.bytedeco.libffi.global.ffi.*
    import org.bytedeco.libffi.presets.ffi

    enum Type(private[LibFfi] val actual: ffi_type) {
      case UInt8 extends Type(ffi_type_uint8())
      case SInt8 extends Type(ffi_type_sint8())
      case UInt16 extends Type(ffi_type_uint16())
      case SInt16 extends Type(ffi_type_sint16())
      case UInt32 extends Type(ffi_type_sint32())
      case SInt32 extends Type(ffi_type_sint32())
      case UInt64 extends Type(ffi_type_uint64())
      case SInt64 extends Type(ffi_type_sint64())
      case Float extends Type(ffi_type_float())
      case Double extends Type(ffi_type_double())
      case Ptr extends Type(ffi_type_pointer())
      case Void extends Type(ffi_type_void())
    }

    inline def invoke(addr: Long, rtn: (Pointer, Type), in: (Pointer, Type)*): Either[Exception, Unit] = {
      import org.bytedeco.javacpp.LongPointer
      val argTypes = new PointerPointer[Pointer](in.size)
      in.zipWithIndex.foreach { case ((_, tpe), i) => argTypes.put(i, tpe.actual) }

      val argValues = new PointerPointer[Pointer](in.size)
      in.zipWithIndex.foreach {
        case ((p, Type.Ptr), i) => argValues.put(i, new LongPointer(Array(p.address()): _*))
        case ((p, _), i)        => argValues.put(i, p)
      }

      val (rtnPtr, rtnTpe) = rtn
      def raise(s: String) = Left(new Exception(s"ffi_prep_cif failed with: ${s}"))

      val cif = new ffi_cif()
      ffi_prep_cif(cif, ffi.FFI_DEFAULT_ABI(), in.size, rtnTpe.actual, argTypes) match {
        case FFI_OK          => Right(ffi_call(cif, ptrToAddress(addr), rtnPtr, argValues))
        case FFI_BAD_TYPEDEF => raise("FFI_BAD_TYPEDEF")
        case FFI_BAD_ARGTYPE => raise("FFI_BAD_ARGTYPE")
        case FFI_BAD_ABI     => raise("FFI_BAD_ABI")
        case unknown         => raise(s"$unknown (unknown ffi_status)")
      }
    }

  }

}
