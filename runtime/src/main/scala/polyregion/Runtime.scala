package polyregion

import polyregion.Runtime.LibFfi.Type
//import polyregion.Runtime.PolyAst.{Expr, Intrinsics, Path, Primitives, Ref, Stmt, Type}

import java.lang.reflect.Modifier
import java.nio.ByteBuffer
import java.nio.file.{Files, Paths, StandardOpenOption}
import scala.annotation.tailrec
import scala.reflect.ClassTag
import scala.collection.mutable
import scala.quoted.ToExpr

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

  object PolyAstUnused {

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
      val Buffer       = Type[Buffer[_]]
      val DoubleBuffer = Buffer.args(Primitives.Double)
      val FloatBuffer  = Buffer.args(Primitives.Float)
      val LongBuffer   = Buffer.args(Primitives.Long)
      val IntBuffer    = Buffer.args(Primitives.Int)
      val ShortBuffer  = Buffer.args(Primitives.Short)
      val ByteBuffer   = Buffer.args(Primitives.Byte)
      val CharBuffer   = Buffer.args(Primitives.Char)

    }

    case class Path(name: String, tpe: Type) {
      def repr: String = s"($name:${tpe.repr})"
    }

    enum Ref(show: => String, val tpe: Type) {
      case Select(head: Path, tail: List[Path] = Nil)
          extends Ref((head :: tail).map(_.repr).mkString("."), tail.lastOption.getOrElse(head).tpe)

      case ByteConst(value: Byte ) extends Ref(s"Byte(`$value)`", Primitives.Byte)
      case CharConst(value: Char ) extends Ref(s"Char(`$value`)", Primitives.Char)
      case ShortConst(value: Short ) extends Ref(s"Short(`$value)`", Primitives.Short)
      case IntConst(value: Int ) extends Ref(s"Int(`$value)`", Primitives.Int)
      case LongConst(value: Long ) extends Ref(s"Long(`$value)`", Primitives.Long)

      case FloatConst(value: Float) extends Ref(s"Float(`$value)`", Primitives.Float)
      case DoubleConst(value: Double) extends Ref(s"Double(`$value)`", Primitives.Double)

      case BoolConst(value: Boolean) extends Ref(s"Boolean(`$value)`", Primitives.Boolean)

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


  }



  def ingest(bitcode : Array[Byte], in: (Pointer, Type)*) = {
    println(s"Ingesting ${bitcode.length} bytes...")
    val mod = new LLVM_.Module("a")
    mod.load(bitcode)
    mod.optimise()
    mod.dump()
    println(s"ORC params: ${in.map { case (l,r) => l.toString + " = " + r }.toList} ")
    new OrcJIT_(mod).invokeORC("lambda", NullPtr -> LibFfi.Type.Void, in:_* )

  }

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
