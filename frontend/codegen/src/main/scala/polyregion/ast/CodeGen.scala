package polyregion.ast

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
    String.format("%032x", BigInteger(1, md5.digest()));
  }

  private def adtFingerprint(nodes: List[StructNode]): String = {
    def tpeRepr(t: CppType): String = {
      val args = t.ctors match {
        case Nil => ""
        case xs  => xs.map(tpeRepr).mkString("<", ",", ">")
      }
      s"${t.namespace.mkString("::")}::${t.name}/${t.kind}$args"
    }
    def go(n: StructNode): String = {
      val name    = (n.tpe.namespace :+ n.tpe.name).mkString("::")
      val parent  = n.parentTpe.fold("")((p, _) => tpeRepr(p))
      val members = n.members.map((mn, mt) => s"$mn:${tpeRepr(mt)}").mkString(",")
      val vars    = n.variants.map(go).mkString(";")
      s"$name/${n.tpe.kind}<$parent>[$members]{$vars}"
    }
    nodes.map(go).mkString("\n")
  }

  private[ast] def generatedNameCollisions(
      nodes: List[StructNode],
      roots: List[CppMsgPackCodecGen.Root]
  ): List[String] = {
    def flatten(node: StructNode): List[StructNode] = node :: node.variants.flatMap(flatten)
    val flattened                                   = nodes.flatMap(flatten)
    val typeCollisions = flattened
      .groupBy(node => (node.tpe.namespace, node.tpe.name))
      .collect {
        case ((namespace, name), duplicates) if duplicates.size > 1 =>
          s"type ${(namespace :+ name).mkString("::")}"
      }
      .toList
    val codecCollisions = flattened
      .groupBy(node => (node.tpe.namespace, node.tpe.name.toLowerCase))
      .collect {
        case ((namespace, stem), duplicates) if duplicates.map(_.tpe.name).distinct.size > 1 =>
          s"codec ${(namespace :+ stem).mkString("::")}"
      }
      .toList
    val rootCollisions = roots
      .groupBy(_.stem)
      .collect { case (stem, duplicates) if duplicates.size > 1 => s"MsgPack root $stem" }
      .toList
    (typeCollisions ::: codecCollisions ::: rootCollisions).sorted
  }

  private def validateGeneratedNames(nodes: List[StructNode], roots: List[CppMsgPackCodecGen.Root]): Unit = {
    val collisions = generatedNameCollisions(nodes, roots)
    require(collisions.isEmpty, s"Generated C++ symbol collisions: ${collisions.mkString(", ")}")
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

  private def astCoreStructs: List[StructNode] =
    deriveStruct[Sym]()
      :: deriveStruct[SourcePosition]()
      :: deriveStruct[Origin]()
      :: deriveStruct[Named]()
      :: deriveStruct[Type.Kind]()
      :: deriveStruct[Type.Space]()
      :: deriveStruct[Type]()
      :: deriveStruct[PathStep]()
      :: deriveStruct[Region]()
      :: deriveStruct[Term]()
      :: deriveStruct[Expr]()
      :: deriveStruct[Overload]()
      :: deriveStruct[AtomicOp]()
      :: deriveStruct[MemScope]()
      :: deriveStruct[MemOrder]()
      :: deriveStruct[Direction]()
      :: deriveStruct[Spec]()
      :: deriveStruct[Intr]()
      :: deriveStruct[Math]()
      :: deriveStruct[ExceptionKind]()
      :: deriveStruct[Handler]()
      :: deriveStruct[Stmt]()
      :: deriveStruct[Signature]()
      :: deriveStruct[InvokeSignature]()
      :: Nil

  private def astProgramStructs: List[StructNode] =
    deriveStruct[Function.Visibility]()
      :: deriveStruct[Function.FpMode]()
      :: deriveStruct[Function.Affinity]()
      :: deriveStruct[Arg.Access]()
      :: deriveStruct[Arg.SizeExpr]()
      :: deriveStruct[Arg.Extent]()
      :: deriveStruct[Arg.Boundary]()
      :: deriveStruct[Arg]()
      :: deriveStruct[FunctionDecl]()
      :: deriveStruct[CallConvention]()
      :: deriveStruct[Function]()
      :: deriveStruct[StructDef]()
      :: deriveStruct[Mirror]()
      :: deriveStruct[Pass.Phase]()
      :: deriveStruct[MetaEntry]()
      :: deriveStruct[Program]()
      :: deriveStruct[StructLayout.Member]()
      :: deriveStruct[StructLayout]()
      :: deriveStruct[Compile.Event]()
      :: deriveStruct[Pass.Arg]()
      :: deriveStruct[Pass.Spec]()
      :: deriveStruct[Pass.Pipeline]()
      :: deriveStruct[Pass.RunResult]()
      :: deriveStruct[Compile.Result]()
      :: Nil

  private def astPackageStructs: List[StructNode] =
    deriveStruct[Interface]()
      :: deriveStruct[Package]()
      :: Nil

  // These are transport envelopes for the one-off package service, not part
  // of the persistent Package schema. Keep their fingerprint independent so
  // adding a service operation cannot invalidate stored Program/Package data.
  private def packageWireStructs: List[StructNode] =
    deriveStruct[Package.TypeSize]()
      :: deriveStruct[Package.LinkRequest]()
      :: deriveStruct[Package.SymRequest]()
      :: deriveStruct[Package.ReturnConvention]()
      :: deriveStruct[Package.EntryArgBinding]()
      :: deriveStruct[Package.SymResolvedProgram]()
      :: Nil

  private def generateAstBindings() = {

    println("Generating C++ mirror for PolyAST...")

    val programStructs = astCoreStructs ::: astProgramStructs
    val packageStructs = programStructs ::: astPackageStructs
    val wireStructs    = packageWireStructs
    val structs        = packageStructs ::: wireStructs

    val (reprProtos, reprImpls): (String, String) = compiletime.generateReprSource[PolyAST.type]

    val namespace         = "polyregion::polyast"
    val adtFileName       = "polyast"
    val jsonCodecFileName = "polyast_codec"

    val adtSources          = structs.flatMap(_.emit())
    val jsonCodecSources    = structs.flatMap(CppNlohmannJsonCodecGen.emit)
    val msgpackCodecSources = structs.flatMap(CppMsgPackCodecGen.emit)
    val msgpackRoots = {
      import CppMsgPackCodecGen.{Root, Schema}
      List(
        Root.raw[Program]("program"),
        Root.versioned[Program]("hashed_program", Schema.Program, "Program", "Program"),
        Root.versioned[Interface]("interface", Schema.Package, "Interface", "Interface"),
        Root.versioned[Package]("package", Schema.Package, "Package", "Package"),
        Root.versioned[Package](
          "package_service_result",
          Schema.PackageWire,
          "package-service Package",
          "package-service wire"
        ),
        Root.versioned[Package.LinkRequest](
          "packagelinkrequest",
          Schema.PackageWire,
          "PackageLinkRequest",
          "package-service wire"
        ),
        Root.versioned[Package.SymRequest](
          "packagesymrequest",
          Schema.PackageWire,
          "PackageSymRequest",
          "package-service wire"
        ),
        Root.versioned[Package.SymResolvedProgram](
          "resolvedsymprogram",
          Schema.PackageWire,
          "PackageSymResolvedProgram",
          "package-service wire"
        ),
        Root.raw[List[StructDef]]("structdefs"),
        Root.versioned[List[StructDef]](
          "hashed_structdefs",
          Schema.Program,
          "StructDef list",
          "Program"
        ),
        Root.raw[Compile.Result]("compileresult"),
        Root.raw[Pass.RunResult]("passrunresult")
      )
    }
    validateGeneratedNames(structs, msgpackRoots)

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

    // Keep Type.canonicalName and its Term constructor in sync.
    val jitConsts: List[(PolyAST.Type, String, String)] = List(
      (PolyAST.Type.Bool1, "std::int8_t", "rd(std::int8_t{}) != 0"),
      (PolyAST.Type.IntU8, "std::int8_t", "rd(std::int8_t{})"),
      (PolyAST.Type.IntU16, "std::uint16_t", "rd(std::uint16_t{})"),
      (PolyAST.Type.IntU32, "std::int32_t", "rd(std::int32_t{})"),
      (PolyAST.Type.IntU64, "std::int64_t", "rd(std::int64_t{})"),
      (PolyAST.Type.IntS8, "std::int8_t", "rd(std::int8_t{})"),
      (PolyAST.Type.IntS16, "std::int16_t", "rd(std::int16_t{})"),
      (PolyAST.Type.IntS32, "std::int32_t", "rd(std::int32_t{})"),
      (PolyAST.Type.IntS64, "std::int64_t", "rd(std::int64_t{})"),
      (PolyAST.Type.Float32, "float", "rd(float{})"),
      (PolyAST.Type.Float64, "double", "rd(double{})")
    )
    val jitCases = jitConsts
      .map((t, cTpe, readExpr) =>
        s"""  if (typeName == "${t.canonicalName}") return n == sizeof($cTpe) ? std::optional<Term::Any>(Term::${t}Const($readExpr)) : std::nullopt;"""
      )
      .mkString("\n")
    val jitHeader = s"""|#pragma once
                        |// GENERATED by polyregion.ast.CodeGen - do not edit
                        |
                        |#include <cstdint>
                        |#include <cstring>
                        |#include <optional>
                        |#include <string_view>
                        |
                        |#include "ast.h"
                        |
                        |namespace $namespace {
                        |[[nodiscard]] inline std::optional<Term::Any> jitConstFromTypeName(std::string_view typeName, const void *bytes, size_t n) {
                        |  const auto rd = [&](auto z) { decltype(z) x{}; if (n >= sizeof(x)) std::memcpy(&x, bytes, sizeof(x)); return x; };
                        |$jitCases
                        |  return std::nullopt;
                        |}
                        |}
                        |""".stripMargin

    val adtHash         = md5(adtFingerprint(structs))
    val programHash     = md5(adtFingerprint(programStructs))
    val packageHash     = md5(adtFingerprint(packageStructs))
    val packageWireHash = md5(packageHash + "\n" + adtFingerprint(wireStructs))

    val jsonCodecHeader = CppCodecGen.emitHeader(namespace, jsonCodecSources, msgpackRoots)
    val jsonCodecImpl = CppCodecGen.emitImpl(
      namespace,
      jsonCodecFileName,
      adtHash,
      programHash,
      packageHash,
      packageWireHash,
      jsonCodecSources,
      msgpackCodecSources,
      msgpackRoots
    )

    println(
      s"Generated ${(adtHeader + adtImpl + jsonCodecHeader + jsonCodecImpl + reprHeader + reprImpl).count(_ == '\n')} lines"
    )
    println(s"MD5=${adtHash}")

    (programHash, packageHash) -> (() => {
      val target = Paths.get("../native/polyast/generated/").toAbsolutePath.normalize
      val scalaTarget = Paths
        .get("compiler/src/main/scala/polyregion/scalalang/generated/PolyASTWireSchema.scala")
        .toAbsolutePath
        .normalize
      val packageWireScalaTarget = Paths
        .get("ast/src/main/scala/polyregion/ast/generated/PolyPackageWireSchema.scala")
        .toAbsolutePath
        .normalize
      println(s"Writing to $target")
      Files.createDirectories(target)
      overwrite(target.resolve("polyast.h"))(adtHeader)
      overwrite(target.resolve("polyast.cpp"))(adtImpl)
      overwrite(target.resolve("polyast_codec.h"))(jsonCodecHeader)
      overwrite(target.resolve("polyast_codec.cpp"))(jsonCodecImpl)
      overwrite(target.resolve("polyast_repr.h"))(reprHeader)
      overwrite(target.resolve("polyast_repr.cpp"))(reprImpl)
      overwrite(target.resolve("polyast_jit.h"))(jitHeader)
      overwrite(scalaTarget)(
        s"""package polyregion.scalalang.generated
           |
           |import javax.annotation.processing.Generated
           |
           |@Generated(Array("polyregion.ast.CodeGen"))
           |private[scalalang] object PolyASTWireSchema {
           |  inline val ProgramHash = "$programHash"
           |  inline val PackageHash = "$packageHash"
           |}
           |""".stripMargin
      )
      Files.createDirectories(packageWireScalaTarget.getParent)
      overwrite(packageWireScalaTarget)(
        s"""package polyregion.ast.generated
           |
           |import javax.annotation.processing.Generated
           |
           |@Generated(Array("polyregion.ast.CodeGen"))
           |private[polyregion] object PolyPackageWireSchema {
           |  inline val Hash = "$packageWireHash"
           |}
           |""".stripMargin
      )
      println("Done")
    })
  }

  private val ((programHash, packageHash), writePolyASTSources) = generateAstBindings()

  def programVersioned(x: Program)            = MsgPack.Versioned(programHash, x)
  def structDefsVersioned(x: List[StructDef]) = MsgPack.Versioned(programHash, x)
  def interfaceVersioned(x: Interface)        = MsgPack.Versioned(packageHash, x)
  def packageVersioned(x: Package)            = MsgPack.Versioned(packageHash, x)

  private def writeConventions(): Unit = {
    val target = Paths.get("../native/common/generated/polyregion/conventions.h").toAbsolutePath.normalize
    println(s"Writing conventions to $target")
    Files.createDirectories(target.getParent)
    overwrite(target)(ConventionsCodeGen.conventionsHeader)
    println("Done")
  }

  private def writeEnums(): Unit = {
    val header   = Paths.get("../native/common/generated/polyregion/enums.h").toAbsolutePath.normalize
    val javaBase = Paths.get("binding-jvm/src/main/java/polyregion/jvm").toAbsolutePath.normalize
    println(s"Writing enums to $header and Java mirrors under $javaBase")
    Files.createDirectories(header.getParent)
    overwrite(header)(EnumCodeGen.cppHeader)
    EnumCodeGen.javaMirrors.foreach { (rel, src) =>
      val target = javaBase.resolve(rel)
      Files.createDirectories(target.getParent)
      overwrite(target)(src)
    }
    println("Done")
  }

  private def writeAbiSources(
      name: String,
      header: String,
      symbols: String,
      exports: String, //
      headerSrc: String,
      symbolsSrc: String,
      exportsSrc: String
  ): Unit = {
    val headerT  = Paths.get(header).toAbsolutePath.normalize
    val symbolsT = Paths.get(symbols).toAbsolutePath.normalize
    val exportsT = Paths.get(exports).toAbsolutePath.normalize
    println(s"Writing $name ABI to $headerT")
    for (p <- List(headerT, symbolsT, exportsT)) Files.createDirectories(p.getParent)
    overwrite(headerT)(headerSrc)
    overwrite(symbolsT)(symbolsSrc)
    overwrite(exportsT)(exportsSrc)
    println("Done")
  }

  def main(args: Array[String]): Unit = {
    writePolyASTSources()
    writeAbiSources(
      "PolyPass",
      "../native/polyc/include/polyregion/polypass.h",
      "../native/polyc/generated/polypass_symbols.h",
      "../native/polyc/generated/polypass-exports.txt",
      CAbiCodeGen.polyPassHeader,
      CAbiCodeGen.polyPassSymbolsHeader,
      CAbiCodeGen.polyPassExportsList
    )
    writeAbiSources(
      "PolyPackage",
      "../native/common/generated/polyregion/polypackage.h",
      "../native/common/generated/polyregion/polypackage_symbols.h",
      "../native/polyc/generated/polypackage-exports.txt",
      CAbiCodeGen.polyPackageHeader,
      CAbiCodeGen.polyPackageSymbolsHeader,
      CAbiCodeGen.polyPackageExportsList
    )
    writeAbiSources(
      "polyc_jit",
      "../native/common/generated/polyregion/polyc_jit.h",
      "../native/common/generated/polyregion/polyc_jit_symbols.h",
      "../native/polyc/generated/polyc_jit-exports.txt",
      CAbiCodeGen.polyJitHeader,
      CAbiCodeGen.polyJitSymbolsHeader,
      CAbiCodeGen.polyJitExportsList
    )
    writeConventions()
    writeEnums()
    generateJniBindings()
  }

}
