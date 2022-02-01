package polyregion.ast

import polyregion.data.NlohmannJsonCodec
import polyregion.data.Cpp.*
import polyregion.data.MsgPack

import java.nio.file.Paths
import java.nio.file.Files
import java.nio.file.StandardOpenOption
import java.lang.annotation.Target
import scala.collection.mutable.ArrayBuffer
import cats.conversions.variance

import java.nio.file.Path
import polyregion.PolyregionCompiler

import java.nio.{ByteBuffer, ByteOrder}
import java.security.MessageDigest
import java.nio.charset.StandardCharsets
import java.math.BigInteger

object CppCodeGen {

  import PolyAst._

  private def md5(s: String): String = {
    val md5 = MessageDigest.getInstance("MD5");
    md5.update(StandardCharsets.UTF_8.encode(s));
    String.format("%032x", new BigInteger(1, md5.digest()));
  }

  private val structs = deriveStruct[Sym]() //
    :: deriveStruct[TypeKind]()
    :: deriveStruct[Type]()
    :: deriveStruct[Named]()
    :: deriveStruct[Position]()
    :: deriveStruct[Term]()
    :: deriveStruct[Expr]()
    :: deriveStruct[Stmt]()
    :: deriveStruct[StructDef]()
    :: deriveStruct[Function]()
    :: deriveStruct[Program]()
    :: Nil

  private val namespace         = "polyregion::polyast"
  private val adtFileName       = "polyast"
  private val jsonCodecFileName = "polyast_codec"

  private val adtSources       = structs.flatMap(_.emit)
  private val jsonCodecSources = structs.flatMap(NlohmannJsonCodec.emit(_))

  private val adtHeader = StructSource.emitHeader(namespace, adtSources)
  private val adtImpl   = StructSource.emitImpl(namespace, adtFileName, adtSources)

  final val AdtHash = md5(adtHeader + adtImpl)

  @main def main(): Unit = {

    println("Generating C++ ADT...")
    println(s"Generated ADT=${(adtImpl + adtHeader).count(_ == '\n')} lines")

    val jsonCodecHeader = NlohmannJsonCodec.emitHeader(namespace, jsonCodecSources)
    val jsonCodecImpl   = NlohmannJsonCodec.emitImpl(namespace, jsonCodecFileName, AdtHash, jsonCodecSources)
    val target          = Paths.get(".").resolve("native/compiler/generated/").normalize.toAbsolutePath
    println(s"Generated Codec=${(jsonCodecHeader + jsonCodecImpl).count(_ == '\n')} lines")

    println(s"MD5=${AdtHash}")

    println(s"Writing to $target")

    Files.createDirectories(target)

    def overwrite(path: Path)(content: String) = Files.writeString(
      path,
      content,
      StandardOpenOption.TRUNCATE_EXISTING,
      StandardOpenOption.CREATE,
      StandardOpenOption.WRITE
    )

    overwrite(target.resolve("polyast.h"))(adtHeader)
    overwrite(target.resolve("polyast.cpp"))(adtImpl)

    overwrite(target.resolve("polyast_codec.h"))(jsonCodecHeader)
    overwrite(target.resolve("polyast_codec.cpp"))(jsonCodecImpl)

    println("Done")

    ()
  }

}
