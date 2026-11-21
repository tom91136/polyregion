package polyregion.javalang

import net.bytebuddy.description.method.MethodDescription
import net.bytebuddy.description.modifier.{TypeManifestation, Visibility}
import net.bytebuddy.implementation.Implementation
import net.bytebuddy.implementation.bytecode.ByteCodeAppender
import net.bytebuddy.jar.asm.{MethodVisitor, Opcodes}
import net.bytebuddy.matcher.ElementMatchers
import net.bytebuddy.{ByteBuddy, ClassFileVersion}
import net.bytebuddy.dynamic.scaffold.subclass.ConstructorStrategy
import polyregion.jvm.twin.{GeneratedTwin, Layout, Marshal, Twins}
import polyregion.jvm.twin.Layout.Kind

import java.nio.ByteBuffer

object TwinEmitter {

  case class HostField(owner: String, name: String, descriptor: String, reachable: Boolean, readOnly: Boolean)

  case class Emitted(fqcn: String, bytes: Array[Byte])

  private class Body(stack: Int)(emit: MethodVisitor => Unit) extends ByteCodeAppender {
    override def apply(
        mv: MethodVisitor,
        ctx: Implementation.Context,
        method: MethodDescription
    ): ByteCodeAppender.Size = {
      emit(mv)
      ByteCodeAppender.Size(stack, FirstFreeLocal + 1)
    }
  }

  // the host object goes in the slot after `this` and the arguments of write and read, which share a signature
  private val FirstFreeLocal = 1 + classOf[GeneratedTwin].getMethods.find(_.getName == "write").get.getParameterCount

  private def internal(name: String): String = name.replace('.', '/')

  private def desc(cls: Class[?]): String = cls.descriptorString

  private val Twin       = internal(classOf[GeneratedTwin].getName)
  private val Buf        = internal(classOf[ByteBuffer].getName)
  private val BufDesc    = desc(classOf[ByteBuffer])
  private val MarshalD   = desc(classOf[Marshal])
  private val ObjectD    = desc(classOf[Object])
  private val LayoutD    = desc(classOf[Layout])
  private val TwinsD     = desc(classOf[Twins])
  private val PutArray   = s"($MarshalD$ObjectD${BufDesc}II)V"
  private val PutStruct  = s"($MarshalD$ObjectD$ObjectD${BufDesc}II)V"
  private val ReadStruct = s"($MarshalD$ObjectD${BufDesc}II)V"

  private case class Access(put: String, get: String, getDesc: String)

  private val Prims = Map(
    'Z' -> Access("put", "get", "B"),
    'B' -> Access("put", "get", "B"),
    'C' -> Access("putChar", "getChar", "C"),
    'S' -> Access("putShort", "getShort", "S"),
    'I' -> Access("putInt", "getInt", "I"),
    'J' -> Access("putLong", "getLong", "J"),
    'F' -> Access("putFloat", "getFloat", "F"),
    'D' -> Access("putDouble", "getDouble", "D")
  )

  private def generatable(layout: Layout, host: Map[String, HostField]): Boolean =
    layout.fields.forall { f =>
      host.get(f.name).exists { h =>
        h.reachable && h.descriptor == f.descriptor && (f.kind match {
          case Kind.PRIM   => f.descriptor.length == 1 && Prims.contains(f.descriptor.head)
          case Kind.ARRAY  => f.descriptor.startsWith("[")
          case Kind.STRUCT => f.descriptor.startsWith("L")
        })
      }
    }

  // the twin sits in the host class's package so package-private fields stay reachable; anything else goes through a reflective twin
  def emit(fqcn: String, layout: Layout, host: Map[String, HostField]): Option[Emitted] =
    Option.when(layout.className.nonEmpty && generatable(layout, host)) {
      val hostInternal = internal(layout.className)
      val self         = internal(fqcn)
      def offset(mv: MethodVisitor, at: Int, i: Int): Unit = {
        mv.visitVarInsn(Opcodes.ILOAD, at)
        mv.visitVarInsn(Opcodes.ALOAD, 0)
        mv.visitFieldInsn(Opcodes.GETFIELD, self, "offsets", "[I")
        mv.visitLdcInsn(i)
        mv.visitInsn(Opcodes.IALOAD)
        mv.visitInsn(Opcodes.IADD)
      }
      def helper(
          mv: MethodVisitor,
          name: String,
          desc: String,
          i: Int,
          root: Boolean,
          value: Option[HostField]
      ): Unit = {
        mv.visitVarInsn(Opcodes.ALOAD, 0)
        mv.visitVarInsn(Opcodes.ALOAD, 1)
        if (root) mv.visitVarInsn(Opcodes.ALOAD, FirstFreeLocal)
        value.foreach { h =>
          mv.visitVarInsn(Opcodes.ALOAD, FirstFreeLocal)
          mv.visitFieldInsn(Opcodes.GETFIELD, internal(h.owner), h.name, h.descriptor)
        }
        mv.visitVarInsn(Opcodes.ALOAD, 3)
        mv.visitVarInsn(Opcodes.ILOAD, 4)
        mv.visitLdcInsn(i)
        mv.visitMethodInsn(Opcodes.INVOKEVIRTUAL, self, name, desc, false)
      }
      val constructor = Body(3) { mv =>
        mv.visitVarInsn(Opcodes.ALOAD, 0)
        mv.visitVarInsn(Opcodes.ALOAD, 1)
        mv.visitLdcInsn(layout.name)
        mv.visitMethodInsn(
          Opcodes.INVOKEVIRTUAL,
          internal(classOf[Twins].getName),
          "layout",
          s"(Ljava/lang/String;)$LayoutD",
          false
        )
        mv.visitVarInsn(Opcodes.ALOAD, 1)
        mv.visitMethodInsn(Opcodes.INVOKESPECIAL, Twin, "<init>", s"($LayoutD$TwinsD)V", false)
        mv.visitInsn(Opcodes.RETURN)
      }
      val write = Body(8) { mv =>
        mv.visitVarInsn(Opcodes.ALOAD, 2)
        mv.visitTypeInsn(Opcodes.CHECKCAST, hostInternal)
        mv.visitVarInsn(Opcodes.ASTORE, FirstFreeLocal)
        layout.fields.zipWithIndex.foreach { (f, i) =>
          val h = host(f.name)
          f.kind match {
            case Kind.PRIM =>
              mv.visitVarInsn(Opcodes.ALOAD, 3)
              offset(mv, 4, i)
              mv.visitVarInsn(Opcodes.ALOAD, FirstFreeLocal)
              mv.visitFieldInsn(Opcodes.GETFIELD, internal(h.owner), h.name, h.descriptor)
              val a = Prims(f.descriptor.head)
              val putDesc = f.descriptor.head match {
                case 'Z' => "B"
                case c   => c.toString
              }
              mv.visitMethodInsn(Opcodes.INVOKEVIRTUAL, Buf, a.put, s"(I$putDesc)$BufDesc", false)
              mv.visitInsn(Opcodes.POP)
            case Kind.ARRAY  => helper(mv, "putArray", PutArray, i, root = false, value = Some(h))
            case Kind.STRUCT => helper(mv, "putStruct", PutStruct, i, root = true, value = Some(h))
          }
        }
        mv.visitVarInsn(Opcodes.ALOAD, 0)
        mv.visitVarInsn(Opcodes.ALOAD, 3)
        mv.visitVarInsn(Opcodes.ILOAD, 4)
        mv.visitMethodInsn(Opcodes.INVOKEVIRTUAL, self, "putTag", s"(${BufDesc}I)V", false)
        mv.visitInsn(Opcodes.RETURN)
      }
      val read = Body(8) { mv =>
        mv.visitVarInsn(Opcodes.ALOAD, 2)
        mv.visitTypeInsn(Opcodes.CHECKCAST, hostInternal)
        mv.visitVarInsn(Opcodes.ASTORE, FirstFreeLocal)
        layout.fields.zipWithIndex.foreach { (f, i) =>
          val h = host(f.name)
          f.kind match {
            case Kind.PRIM if !f.readOnly =>
              mv.visitVarInsn(Opcodes.ALOAD, FirstFreeLocal)
              if (f.descriptor == "Z") {
                mv.visitVarInsn(Opcodes.ALOAD, 3)
                offset(mv, 4, i)
                mv.visitMethodInsn(Opcodes.INVOKESTATIC, Twin, "getBool", s"(${BufDesc}I)Z", false)
              } else {
                val a = Prims(f.descriptor.head)
                mv.visitVarInsn(Opcodes.ALOAD, 3)
                offset(mv, 4, i)
                mv.visitMethodInsn(Opcodes.INVOKEVIRTUAL, Buf, a.get, s"(I)${a.getDesc}", false)
              }
              mv.visitFieldInsn(Opcodes.PUTFIELD, internal(h.owner), h.name, h.descriptor)
            case Kind.STRUCT => helper(mv, "readStruct", ReadStruct, i, root = true, value = None)
            case _           =>
          }
        }
        mv.visitInsn(Opcodes.RETURN)
      }
      val bytes = ByteBuddy(ClassFileVersion.JAVA_V8)
        .subclass(classOf[GeneratedTwin], ConstructorStrategy.Default.NO_CONSTRUCTORS)
        .name(fqcn)
        .modifiers(Visibility.PUBLIC, TypeManifestation.FINAL)
        .defineConstructor(Visibility.PUBLIC)
        .withParameters(classOf[Twins])
        .intercept(Implementation.Simple(constructor))
        .method(ElementMatchers.named("write"))
        .intercept(Implementation.Simple(write))
        .method(ElementMatchers.named("read"))
        .intercept(Implementation.Simple(read))
        .make()
        .getBytes
      Emitted(fqcn, bytes)
    }
}
