package polyregion.ast

import polyregion.ast.CppMsgPackCodecGen.Root
import polyregion.ast.CppStructGen.{CppType, StructNode}

class CodeGenSuite extends munit.FunSuite {

  private def node(name: String) = StructNode(
    CppType(List("polyregion", "polyast"), name, CppType.Kind.Data),
    Nil,
    None
  )

  test("generated names reject case-folded codec and duplicate root symbols") {
    val root = Root.raw[PolyAST.Program]("program")
    assertEquals(
      CodeGen.generatedNameCollisions(List(node("Value"), node("VALUE")), List(root, root)),
      List("MsgPack root program", "codec polyregion::polyast::value")
    )
  }

  test("semantic names do not depend on repr") {
    val symbol = PolyAST.Sym(List("vendor", "algorithm"))
    val tpe = PolyAST.Type.Ptr(
      PolyAST.Type.Struct(symbol, List(PolyAST.Type.IntS32)),
      PolyAST.Type.Space.Local
    )

    assertEquals(symbol.fqcn, "vendor.algorithm")
    assertEquals(tpe.canonicalName, "vendor.algorithm<I32>*^Local")
    assertEquals(
      PolyAST
        .Signature(
          symbol,
          List(PolyAST.Type.Var("T", Some(4))),
          Some(tpe),
          List(PolyAST.Type.Var("T", Some(4))),
          List(PolyAST.Type.IntU32),
          List(PolyAST.Type.IntU64),
          PolyAST.Type.Unit0
        )
        .signatureKey,
      "vendor.algorithm<I32>*^Local.vendor.algorithm<#T:size=4>(#T:size=4)[U32;U64]:Unit0"
    )
  }
}
