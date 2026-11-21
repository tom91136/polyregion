package polyregion.javalang

import cats.syntax.all.*
import net.bytebuddy.jar.asm.{ClassReader, ClassVisitor, FieldVisitor, MethodVisitor, Opcodes, Type}
import org.jetbrains.java.decompiler.struct.gen.{CodeType, VarType}
import polyregion.ast.{PolyAST as p, *}
import polyregion.javalang.Interop.absent

import scala.collection.mutable

// instance state as a class file declares it; JavaTyper.lowerStruct is the same job for a class javac has a symbol for
object ClassLayouts {

  type ClassBytes = String => Option[Array[Byte]]

  private val Global = p.Type.Space.Global

  private def nameOf(t: VarType): String =
    Option(t.value).filter(_ => t.`type` == CodeType.OBJECT).fold(String.valueOf(t))(_.replace('/', '.'))

  def readable(t: VarType): String = {
    val base = Option(t.value).filter(_ => t.`type` == CodeType.OBJECT).fold(VarType.getChar(t.`type`))(v => s"L$v;")
    Type.getType("[" * t.arrayDim + base).getClassName
  }

  def typeOf(t: VarType): Result[p.Type] = {
    val base = t.`type` match {
      case CodeType.BOOLEAN                    => p.Type.Bool1.success
      case CodeType.BYTE | CodeType.BYTECHAR   => p.Type.IntS8.success
      case CodeType.SHORT | CodeType.SHORTCHAR => p.Type.IntS16.success
      case CodeType.CHAR                       => p.Type.IntU16.success
      case CodeType.INT                        => p.Type.IntS32.success
      case CodeType.LONG                       => p.Type.IntS64.success
      case CodeType.FLOAT                      => p.Type.Float32.success
      case CodeType.DOUBLE                     => p.Type.Float64.success
      case CodeType.VOID                       => p.Type.Unit0.success
      case CodeType.NULL                       => "Unsupported `null`; PolyRegion has no device heap".fail
      case _ =>
        s"Cannot lower the JVM type `${nameOf(t)}`; only primitives and their arrays travel to a device".fail
    }
    base.map(b => (1 to t.arrayDim).foldLeft(b)((c, _) => p.Type.Ptr(c, Global)))
  }

  private val Platform = List("java/", "javax/", "jdk/", "sun/", "com/sun/")

  private case class Shape(
      access: Int,
      superName: Option[String],
      member: Option[(String, String)],
      fields: List[(String, String)],
      methods: List[(String, String, Int)]
  )

  private class ShapeCollector extends ClassVisitor(Opcodes.ASM9) {
    val header  = mutable.ListBuffer.empty[(String, Option[String], Int)]
    val methods = mutable.ListBuffer.empty[(String, String, Int)]
    val inners  = mutable.ListBuffer.empty[(String, String, String)]
    val fields  = mutable.ListBuffer.empty[(String, String)]

    override def visit(v: Int, a: Int, n: String, s: String, sup: String, itf: Array[String]): Unit =
      header += ((n, Option(sup), a))

    override def visitMethod(a: Int, n: String, d: String, s: String, e: Array[String]): MethodVisitor = {
      methods += ((n, d, a))
      absent
    }

    override def visitInnerClass(n: String, outer: String, inner: String, a: Int): Unit =
      for {
        o <- Option(outer)
        i <- Option(inner)
      } inners += ((n, o, i))

    override def visitField(a: Int, n: String, d: String, s: String, v: Object): FieldVisitor = {
      if ((a & (Opcodes.ACC_STATIC | Opcodes.ACC_SYNTHETIC)) == 0) fields += n -> d
      absent
    }
  }

  private def shapeOf(bytes: Array[Byte]): Shape = {
    val collector = ShapeCollector()
    ClassReader(bytes).accept(collector, ClassReader.SKIP_CODE | ClassReader.SKIP_DEBUG | ClassReader.SKIP_FRAMES)
    val (self, parent, access) = collector.header.headOption.getOrElse(("", None, 0))
    val member                 = collector.inners.collectFirst { case (n, outer, inner) if n == self => outer -> inner }
    Shape(access, parent, member, collector.fields.toList, collector.methods.toList)
  }

  private def shapeFor(internal: String, bytesOf: ClassBytes): Result[Shape] =
    bytesOf(internal.replace('/', '.'))
      .failIfEmpty(s"No class file for `${internal.replace('/', '.')}`")
      .flatMap(b => Either.catchNonFatal(shapeOf(b)))

  // a member class is named as javac names its symbol (`Outer.Inner`), a local or anonymous one by its binary name
  def structName(internal: String, bytesOf: ClassBytes): Result[String] =
    shapeFor(internal, bytesOf).flatMap { shape =>
      shape.member.fold(internal.replace('/', '.').success)((outer, inner) =>
        structName(outer, bytesOf).map(o => s"$o.$inner")
      )
    }

  // one class gets one layout: javac's wins, and the class file's members must be a part of it
  def merge(sourced: List[p.StructDef], bytecode: List[p.StructDef]): Result[List[p.StructDef]] = {
    val known = sourced.map(d => d.name -> d).toMap
    bytecode
      .traverse { d =>
        known.get(d.name).fold(List(d).success) { source =>
          val have = source.members.map(m => m.symbol -> m.tpe).toMap
          d.members.find(m => !have.get(m.symbol).contains(m.tpe)) match {
            case None => List.empty[p.StructDef].success
            case Some(m) =>
              (s"`${d.name.repr}` has one layout from its source and another from its class file: member " +
                s"`${m.symbol}: ${m.tpe.repr}` of the class file is not in the source layout").fail
          }
        }
      }
      .map(sourced ++ _.flatten)
  }

  def isPlatform(internal: String): Boolean = Platform.exists(internal.startsWith(_))

  // fields the layout cannot carry are left out, an access to one is refused where it happens
  def structOf(internal: String, bytesOf: ClassBytes): Result[p.StructDef] =
    for {
      name  <- structName(internal, bytesOf)
      shape <- shapeFor(internal, bytesOf)
      members = shape.fields.flatMap((field, desc) => typeOf(VarType(desc)).toOption.map(t => p.Named(field, t)))
    } yield p.StructDef(p.Sym(name), Nil, members, Nil)

  // the access flags of the class and of the method it declares itself, none for an inherited one
  def accessOf(method: MethodRef, bytesOf: ClassBytes): Result[(Int, Option[Int])] =
    shapeFor(method.owner, bytesOf).map { s =>
      s.access -> s.methods.collectFirst { case (method.name, method.desc, a) => a }
    }

  def superOf(internal: String, bytesOf: ClassBytes): Result[Option[String]] =
    shapeFor(internal, bytesOf).map(_.superName)
}
