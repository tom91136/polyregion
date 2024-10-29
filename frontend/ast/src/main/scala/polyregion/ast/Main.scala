package polyregion.ast

import polyregion.ast.CppNlohmannJsonCodecGen
import polyregion.ast.CppStructGen.*
import polyregion.ast.{MsgPack, PolyAST}

import java.lang.annotation.Target
import java.math.BigInteger
import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path, Paths, StandardOpenOption}
import java.nio.{ByteBuffer, ByteOrder}
import java.security.MessageDigest
import scala.collection.mutable.ArrayBuffer
import scala.runtime.RichInt

private[polyregion] object Main {

  import PolyAST.*

  private def md5(s: String): String = {
    val md5 = MessageDigest.getInstance("MD5");
    md5.update(StandardCharsets.UTF_8.encode(s));
    String.format("%032x", new BigInteger(1, md5.digest()));
  }

  private def overwrite(path: Path)(content: String) = Files.write(
    path,
    content.getBytes(StandardCharsets.UTF_8),
    StandardOpenOption.TRUNCATE_EXISTING,
    StandardOpenOption.CREATE,
    StandardOpenOption.WRITE
  )

  def generateJniBindings(): Unit = {
    import java.lang.reflect.{Constructor, Field, Method, Modifier}

    val pending: List[(Class[?], Field => Boolean, Constructor[?] => Boolean, Method => Boolean)] =
      List(
        (
          classOf[java.nio.ByteBuffer],
          _ => false,
          _ => false,
          m => Set("allocate", "allocateDirect").contains(m.getName)
        ),
        (classOf[polyregion.jvm.runtime.Property], _ => false, _ => true, _ => false),
        (classOf[polyregion.jvm.runtime.Dim3], _ => true, _ => true, _ => false),
        (classOf[polyregion.jvm.runtime.Policy], _ => true, _ => true, _ => false),
        (classOf[polyregion.jvm.runtime.Device.Queue], _ => false, _ => true, _ => false),
        (classOf[polyregion.jvm.runtime.Device], _ => false, _ => true, _ => false),
        (classOf[polyregion.jvm.runtime.Platform], _ => false, _ => true, _ => false),
        (classOf[polyregion.jvm.compiler.Event], _ => false, _ => true, _ => false),
        (classOf[polyregion.jvm.compiler.Layout], _ => false, _ => true, _ => false),
        (classOf[polyregion.jvm.compiler.Layout.Member], _ => false, _ => true, _ => false),
        (classOf[polyregion.jvm.compiler.Options], _ => true, _ => true, _ => false),
        (classOf[polyregion.jvm.compiler.Compilation], _ => false, _ => true, _ => false),
        (classOf[java.lang.String], _ => false, _ => false, _ => false),
        (classOf[java.lang.Runnable], _ => true, _ => true, _ => true),
        (classOf[java.io.File], _ => false, _ => false, m => m.getName == "delete")
      )

    println("Generating C++ mirror for JNI...")

    val knownClasses: Set[String] = pending.map(_._1.getName).toSet
    val (headers, impls)          = pending.map(CppJniBindGen.reflectJniSource(knownClasses, _, _, _, _)).unzip
    val header =
      s"""#include <jni.h>
         |#include <optional>
         |#include <memory>
         |namespace polyregion::generated {
         |${headers.mkString("\n")}
         |}// polyregion::generated""".stripMargin
    val impl =
      s"""#include "mirror.h"
         |using namespace polyregion::generated;
         |${impls.mkString("\n")}
         |""".stripMargin
    println(s"Generated ADT=${(header + impl).count(_ == '\n')} lines")

    val target = Paths.get("../native/bindings/jvm/generated/").toAbsolutePath.normalize

    println(s"Writing to $target")

    Files.createDirectories(target)
    overwrite(target.resolve("mirror.h"))(header)
    overwrite(target.resolve("mirror.cpp"))(impl)

    List(
      classOf[polyregion.jvm.compiler.Compiler],
      classOf[polyregion.jvm.runtime.Platforms],
      classOf[polyregion.jvm.runtime.Platform],
      classOf[polyregion.jvm.Natives]
    ).foreach { r =>
      val (name, header) = CppJniBindGen.generateRegisterNative(r)
      overwrite(target.resolve(s"${name.toLowerCase}.h"))(header)
    }

    println("Done")
  }

  def generateAstBindings(): Unit = {

    println("Generating C++ mirror...")

    val structs =
      deriveStruct[SourcePosition]()
        :: deriveStruct[Named]()
        :: deriveStruct[Type.Kind]()
        :: deriveStruct[Type.Space]()
        :: deriveStruct[Type]()
        :: deriveStruct[Expr]()
        :: deriveStruct[Overload]()
        :: deriveStruct[Spec]()
        :: deriveStruct[Intr]()
        :: deriveStruct[Math]()
        :: deriveStruct[Stmt]()
        :: deriveStruct[Signature]()
        :: deriveStruct[Function.Attr]()
        :: deriveStruct[Arg]()
        :: deriveStruct[Function]()
        :: deriveStruct[Program]()
        :: deriveStruct[StructDef]()
        :: deriveStruct[StructLayoutMember]()
        :: deriveStruct[StructLayout]()
        :: deriveStruct[CompileEvent]()
        :: deriveStruct[CompileResult]()
        :: Nil //

    val namespace         = "polyregion::polyast"
    val adtFileName       = "polyast"
    val jsonCodecFileName = "polyast_codec"

    val adtSources       = structs.flatMap(_.emit())
    val jsonCodecSources = structs.flatMap(CppNlohmannJsonCodecGen.emit(_))

    val adtHeader = StructSource.emitHeader(namespace, adtSources)
    val adtImpl   = StructSource.emitImpl(namespace, adtFileName, adtSources)

    val AdtHash = md5(adtHeader + adtImpl)

    println(s"Generated ADT=${(adtImpl + adtHeader).count(_ == '\n')} lines")

    val jsonCodecHeader = CppNlohmannJsonCodecGen.emitHeader(namespace, jsonCodecSources)
    val jsonCodecImpl   = CppNlohmannJsonCodecGen.emitImpl(namespace, jsonCodecFileName, AdtHash, jsonCodecSources)

    val target = Paths.get("../native/polyast/generated/").toAbsolutePath.normalize
    println(s"Generated Codec=${(jsonCodecHeader + jsonCodecImpl).count(_ == '\n')} lines")

    println(s"MD5=${AdtHash}")

    println(s"Writing to $target")

    Files.createDirectories(target)

    overwrite(target.resolve("polyast.h"))(adtHeader)
    overwrite(target.resolve("polyast.cpp"))(adtImpl)
    overwrite(target.resolve("polyast_codec.h"))(jsonCodecHeader)
    overwrite(target.resolve("polyast_codec.cpp"))(jsonCodecImpl)

    println("Done")
  }

  def main(args: Array[String]): Unit = {
    generateAstBindings()
    generateJniBindings()
  }

}
