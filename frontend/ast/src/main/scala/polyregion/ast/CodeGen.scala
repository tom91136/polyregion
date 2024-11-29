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
import cats.syntax.all.*

private[polyregion] object CodeGen {

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

  private def generateJniBindings(): Unit = {
    import java.lang.reflect.{Constructor, Field, Method, Modifier}

    val classes: List[(Class[?], Field => Boolean, Constructor[?] => Boolean, Method => Boolean)] =
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

    val registerNatives = List(
      classOf[polyregion.jvm.compiler.Compiler],
      classOf[polyregion.jvm.runtime.Platforms],
      classOf[polyregion.jvm.runtime.Platform],
      classOf[polyregion.jvm.Natives]
    ).map { c =>
      val (name, header) = CppJniBindGen.generateRegisterNative(c)
      s"${name.toLowerCase}.h" -> header
    }

    println("Generating C++ mirror for JNI...")

    val knownClasses: Set[String] = classes.map(_._1.getName).toSet
    val (headers, impls)          = classes.map(CppJniBindGen.reflectJniSource(knownClasses, _, _, _, _)).unzip
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

    val registerNativeHeaders = registerNatives.map(_._2).mkString("\n")

    val AdtHash = md5(header + impl + registerNativeHeaders)
    println(s"MD5=${AdtHash}")
    println(s"Generated ADT=${(header + impl).count(_ == '\n')} lines")

    val target = Paths.get("../native/bindings/jvm/generated/").toAbsolutePath.normalize
    println(s"Writing to $target")
    Files.createDirectories(target)
    overwrite(target.resolve("mirror.h"))(header)
    overwrite(target.resolve("mirror.cpp"))(impl)
    registerNatives.foreach((name, header) => overwrite(target.resolve(name))(header))
    println("Done")
  }

  private def generateAstBindings() = {

    println("Generating C++ mirror for PolyAST...")

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
        :: deriveStruct[StructDef]()
        :: deriveStruct[Program]()
        :: deriveStruct[StructLayoutMember]()
        :: deriveStruct[StructLayout]()
        :: deriveStruct[CompileEvent]()
        :: deriveStruct[CompileResult]()
        :: Nil //

    val (reprProtos, reprImpls) = compiletime.generateReprSource[PolyAST.type]


    val namespace         = "polyregion::polyast"
    val adtFileName       = "polyast"
    val jsonCodecFileName = "polyast_codec"

    val adtSources       = structs.flatMap(_.emit())
    val jsonCodecSources = structs.flatMap(CppNlohmannJsonCodecGen.emit(_))

    val adtHeader = StructSource.emitHeader(namespace, adtSources)
    val adtImpl   = StructSource.emitImpl(namespace, adtFileName, adtSources)

    val reprHeader = s"""|#pragma once
                         |
                         |#include <optional>
                         |#include "ast.h"
                         |
                         |namespace $namespace {
                         |$reprProtos
                         |}
                         |""".stripMargin

    val reprImpl = s"""|#include "polyast_repr.h"
                       |#include "aspartame/all.hpp"
                       |#include "fmt/core.h"
                       |
                       |using namespace aspartame;
                       |using namespace std::string_literals;
                       |
                       |namespace $namespace {
                       |$reprImpls
                       |}
                       |""".stripMargin

    val adtHash = md5(adtHeader + adtImpl)

    val jsonCodecHeader = CppNlohmannJsonCodecGen.emitHeader(namespace, jsonCodecSources)
    val jsonCodecImpl   = CppNlohmannJsonCodecGen.emitImpl(namespace, jsonCodecFileName, adtHash, jsonCodecSources)

    println(
      s"Generated ${(adtHeader + adtImpl + jsonCodecHeader + jsonCodecImpl + reprHeader + reprImpl).count(_ == '\n')} lines"
    )
    println(s"MD5=${adtHash}")

    adtHash -> (() => {
      val target = Paths.get("../native/polyast/generated/").toAbsolutePath.normalize
      println(s"Writing to $target")
      Files.createDirectories(target)
      overwrite(target.resolve("polyast.h"))(adtHeader)
      overwrite(target.resolve("polyast.cpp"))(adtImpl)
      overwrite(target.resolve("polyast_codec.h"))(jsonCodecHeader)
      overwrite(target.resolve("polyast_codec.cpp"))(jsonCodecImpl)
      overwrite(target.resolve("polyast_repr.h"))(reprHeader)
      overwrite(target.resolve("polyast_repr.cpp"))(reprImpl)
      println("Done")
    })
  }

  private val (polyASTHash, writePolyASTSources) = generateAstBindings()

  def polyASTVersioned[A](x: A) = MsgPack.Versioned(polyASTHash, x)

  def main(args: Array[String]): Unit = {
    writePolyASTSources()
    generateJniBindings()
  }

}
