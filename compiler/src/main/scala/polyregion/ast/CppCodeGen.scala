package polyregion.ast

import polyregion.data.Cpp._
import polyregion.data.MsgPack
import java.nio.file.Paths
import java.nio.file.Files
import java.nio.file.StandardOpenOption
import java.lang.annotation.Target

import ujson.Arr
import scala.collection.mutable.ArrayBuffer

object CppCodeGen {

  case class AA(s: String)

  @main def main(): Unit = {

    println("\n=========\n")

    import PolyAstUnused._
    val s = Sym("a.b")

    import MsgPack.Codec.Compact.derived

    given MsgPack.Codec[Sym]      = derived
    given MsgPack.Codec[TypeKind] = derived
    given MsgPack.Codec[Type]     = derived
    given MsgPack.Codec[Term]     = derived
    given MsgPack.Codec[Named]    = derived
    given MsgPack.Codec[Position] = derived
    given MsgPack.Codec[Intr]     = derived
    given MsgPack.Codec[Stmt]     = derived
    given MsgPack.Codec[Expr]     = derived
    given MsgPack.Codec[Tree]     = derived

    val ast: Tree = Stmt.Cond(
      Expr.Alias(Term.BoolConst(true)),
      Stmt.Return(Expr.Alias(Term.Select(Nil, Named("a", Type.Float)))) :: Nil,
      Stmt.Return(Expr.Alias(Term.IntConst(1))) :: Nil
    )

    println(MsgPack.encodeMsg(ast))
    println(MsgPack.encode(ast).length)

    // println(MsgPack.encodeMsg(ast))
    println(MsgPack.decode[Tree](MsgPack.encode(ast)))

    println(MsgPack.decode[Tree](MsgPack.encode(ast)).right.get == ast)
    // given ReadWriter[TypeKind] = macroRW
    // given ReadWriter[Sym]      = macroRW
    // given ReadWriter[Type]     = macroRW

    // println(write(s, indent = 2, escapeUnicode = true))

    val alts = deriveStruct[PolyAstUnused.Sym]().emit //
      ::: deriveStruct[PolyAstUnused.TypeKind]().emit
      ::: deriveStruct[PolyAstUnused.Type]().emit
      ::: deriveStruct[PolyAstUnused.Named]().emit
      ::: deriveStruct[PolyAstUnused.Position]().emit
      ::: deriveStruct[PolyAstUnused.Term]().emit
      ::: deriveStruct[PolyAstUnused.Tree]().emit
      ::: deriveStruct[PolyAstUnused.Function]().emit
      ::: deriveStruct[PolyAstUnused.StructDef]().emit

    val header = StructSource.emitHeader("polyregion::polyast", alts)
    // println(header)
    println("\n=========\n")
    val impl = StructSource.emitImpl("polyregion::polyast", "polyast", alts)
    // println(impl)

    val target = Paths.get(".").resolve("native/src/generated/").normalize.toAbsolutePath

    Files.createDirectories(target)
    println(s"Dest=${target}")
    println("\n=========\n")

    Files.writeString(
      target.resolve("polyast.cpp"),
      impl,
      StandardOpenOption.TRUNCATE_EXISTING,
      StandardOpenOption.CREATE,
      StandardOpenOption.WRITE
    )
    Files.writeString(
      target.resolve("polyast.h"),
      header,
      StandardOpenOption.TRUNCATE_EXISTING,
      StandardOpenOption.CREATE,
      StandardOpenOption.WRITE
    )

    println(summon[ToCppType[PolyAstUnused.TypeKind.Fractional.type]]().qualified)
//    import Cpp.*
//    println(T1Mid.T1ALeaf(Nil, List("a", "b"), 23, T1Mid.T1BLeaf))

    // println(Cpp.deriveStruct[Alt]().map(_.emitSource).mkString("\n"))
    // println(Cpp.deriveStruct[FirstTop]().map(_.emitSource).mkString("\n"))
    // println(Cpp.deriveStruct[First]().map(_.emitSource).mkString("\n"))
//    println(Cpp.deriveStruct[Foo]().map(_.emitSource).mkString("\n"))
    ()
  }

}
