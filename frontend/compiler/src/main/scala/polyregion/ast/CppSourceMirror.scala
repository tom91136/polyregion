package polyregion.ast

import polyregion.ast.mirror.CppNlohmannJsonCodecGen
import polyregion.ast.mirror.CppStructGen.*
import polyregion.ast.{MsgPack, PolyAst}

import java.lang.annotation.Target
import java.math.BigInteger
import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path, Paths, StandardOpenOption}
import java.nio.{ByteBuffer, ByteOrder}
import java.security.MessageDigest
import scala.collection.mutable.ArrayBuffer
import scala.runtime.RichInt

private[polyregion] object CppSourceMirror {

  import PolyAst.*

  private inline def md5(s: String): String = {
    val md5 = MessageDigest.getInstance("MD5");
    md5.update(StandardCharsets.UTF_8.encode(s));
    String.format("%032x", new BigInteger(1, md5.digest()));
  }

  private final val structs =
    deriveStruct[Sym]()
      :: deriveStruct[Named]()
      :: deriveStruct[TypeKind]()
      :: deriveStruct[Type]()
      :: deriveStruct[SourcePosition]()
      :: deriveStruct[Term]()
      // :: deriveStruct[NullaryIntrinsicKind]()
      // :: deriveStruct[UnaryIntrinsicKind]()
      // :: deriveStruct[BinaryIntrinsicKind]()
      :: deriveStruct[Type.Space]()
      :: deriveStruct[Overload]()
      :: deriveStruct[Spec]()
      :: deriveStruct[Intr]()
      :: deriveStruct[Math]()
      :: deriveStruct[Expr]()
      :: deriveStruct[Stmt]()
      :: deriveStruct[StructMember]()
      :: deriveStruct[StructDef]()
      :: deriveStruct[Signature]()
      :: deriveStruct[InvokeSignature]()
      :: deriveStruct[Function.Kind]()
      :: deriveStruct[Function.Attr]()
      :: deriveStruct[Arg]()
      :: deriveStruct[Function]()
      :: deriveStruct[Program]()
      :: Nil //

  private final val namespace         = "polyregion::polyast"
  private final val adtFileName       = "polyast"
  private final val jsonCodecFileName = "polyast_codec"

  // val initNS = symbolNames[PolyAst.type].head

  private final val adtSources       = structs.flatMap(_.emit)
  private final val jsonCodecSources = structs.flatMap(CppNlohmannJsonCodecGen.emit(_))

  private final val adtHeader = StructSource.emitHeader(namespace, adtSources)
  private final val adtImpl   = StructSource.emitImpl(namespace, adtFileName, adtSources)

  final val AdtHash = md5(adtHeader + adtImpl)

  inline def encode[A: MsgPack.Codec](x: A): Array[Byte] =
    MsgPack.encode(MsgPack.Versioned(AdtHash, x))

  @main def main(): Unit = {

    println("Generating C++ mirror...")
    println(s"Generated ADT=${(adtImpl + adtHeader).count(_ == '\n')} lines")

    val jsonCodecHeader = CppNlohmannJsonCodecGen.emitHeader(namespace, jsonCodecSources)
    val jsonCodecImpl   = CppNlohmannJsonCodecGen.emitImpl(namespace, jsonCodecFileName, AdtHash, jsonCodecSources)

    val target = Paths.get("../native/compiler/generated/").toAbsolutePath.normalize
    println(s"Generated Codec=${(jsonCodecHeader + jsonCodecImpl).count(_ == '\n')} lines")

    println(s"MD5=${AdtHash}")

    println(s"Writing to $target")

    Files.createDirectories(target)

    def overwrite(path: Path)(content: String) = Files.write(
      path,
      content.getBytes(StandardCharsets.UTF_8),
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
