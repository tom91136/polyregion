package polyregion.ast

import java.nio.file.Files
import scala.io.Source
import scala.util.Using

import polyregion.ast.PolyAST as p

class LibraryCodeGenSuite extends munit.FunSuite {

  private def golden(name: String): String =
    Using.resource(Source.fromResource(s"library-codegen/$name"))(_.mkString)

  private val t = p.Type.Var("T")
  private val transform = p.FunctionDecl(
    p.Sym("example.transform"),
    List("T"),
    None,
    List(
      p.Arg(
        p.Named("in", p.Type.Ptr(t, p.Type.Space.Global)),
        boundary = Some(p.Arg.Boundary(p.Arg.Access.Read, p.Arg.Extent.Elements(p.Arg.SizeExpr.Param(2))))
      ),
      p.Arg(
        p.Named("out", p.Type.Ptr(t, p.Type.Space.Global)),
        boundary = Some(p.Arg.Boundary(p.Arg.Access.Write, p.Arg.Extent.Elements(p.Arg.SizeExpr.Param(2))))
      ),
      p.Arg(p.Named("n", p.Type.IntS32)),
      p.Arg(p.Named("op", p.Type.Exec(Nil, List(t), t)))
    ),
    Nil,
    Nil,
    p.Type.Unit0,
    p.Function.Affinity.Host
  )
  private val count = p.FunctionDecl(
    p.Sym("example.count"),
    List("T"),
    None,
    List(
      p.Arg(
        p.Named("in", p.Type.Ptr(t, p.Type.Space.Global)),
        boundary = Some(p.Arg.Boundary(p.Arg.Access.Read, p.Arg.Extent.Elements(p.Arg.SizeExpr.Param(1))))
      ),
      p.Arg(p.Named("n", p.Type.IntS32))
    ),
    Nil,
    Nil,
    p.Type.IntS32,
    p.Function.Affinity.Host
  )
  private val countValue = count.copy(args = count.args :+ p.Arg(p.Named("value", t)))
  private val library    = p.LibraryDef(p.Sym("example"), List(transform, count))
  private val fortran = LibraryCodeGen.FortranConfig(
    "example_ffi",
    List(LibraryCodeGen.FortranVariant("r4", Map("T" -> "real(c_float)"), List("c_float")))
  )
  private val scala = LibraryCodeGen.ScalaConfig("example.bindings", "ExampleImport")

  test("library declarations project exact identities into C++, Fortran and Scala") {
    val cpp = LibraryCodeGen.cppHeader(library)
    val f90 = LibraryCodeGen.fortranModule(library, fortran)
    val sc  = LibraryCodeGen.scalaTrait(library, scala)

    assertEquals(cpp, golden("example.hpp"))
    assertEquals(f90, golden("example.f90"))
    assertEquals(sc, golden("ExampleImport.scala"))

    assert(cpp.contains("template <class T, class Op>"))
    assert(cpp.contains("clang::annotate(\"polyregion_import:example:example.transform\")"))
    assert(cpp.contains("inline void transform(const T *in, T *out, std::int32_t n, Op op)"))

    assert(f90.contains("module example_ffi"))
    assert(f90.contains("polyregion_import:example:example.transform:r4"))
    assert(f90.contains("real(c_float), intent(in) :: in(*)"))
    assert(f90.contains("procedure(polyregion_transform_op_r4) :: op"))

    assert(sc.contains("package example.bindings"))
    assert(sc.contains("@ExampleImport.PolyregionImport(\"example\", \"example.transform\")"))
    assert(sc.contains("def transform[T](in: Array[T], out: Array[T], n: Int, op: T => T): Unit"))
    assert(!(cpp + f90 + sc).toLowerCase.contains("spectra"))
  }

  test("projection order and stale-output checks are deterministic") {
    val reversed = library.copy(decls = library.decls.reverse)
    val cpp      = LibraryCodeGen.cppHeader(library)

    assertEquals(LibraryCodeGen.cppHeader(reversed), cpp)
    assertEquals(LibraryCodeGen.fortranModule(reversed, fortran), LibraryCodeGen.fortranModule(library, fortran))
    assertEquals(LibraryCodeGen.scalaTrait(reversed, scala), LibraryCodeGen.scalaTrait(library, scala))

    val path = Files.createTempFile("polyregion-library-codegen", ".hpp")
    Files.writeString(path, "stale")
    assert(LibraryCodeGen.checkCurrent(path, cpp).isLeft)
    Files.writeString(path, cpp)
    assertEquals(LibraryCodeGen.checkCurrent(path, cpp), Right(()))
    assert(LibraryCodeGen.checkCurrent(path.resolveSibling("missing-generated-output"), cpp).isLeft)
  }

  test("overloads retain distinct Fortran procedures") {
    val overloaded = library.copy(decls = countValue :: library.decls)
    val f90        = LibraryCodeGen.fortranModule(overloaded, fortran)

    assert(f90.contains("module procedure polyregion_count_o0_r4"))
    assert(f90.contains("module procedure polyregion_count_o1_r4"))
    assertEquals(LibraryCodeGen.fortranModule(overloaded.copy(decls = overloaded.decls.reverse), fortran), f90)
  }

  test("pointer declarations require boundary metadata") {
    val invalid =
      library.copy(decls = List(count.copy(args = count.args.updated(0, count.args.head.copy(boundary = None)))))
    val error = intercept[IllegalArgumentException](LibraryCodeGen.cppHeader(invalid))

    assert(error.getMessage.contains("has no boundary"))
  }

  test("C++ arrays and ordinary functions produce valid header definitions") {
    val array = p.FunctionDecl(
      p.Sym("example.array"),
      Nil,
      None,
      List(p.Arg(p.Named("values", p.Type.Arr(p.Type.IntS32, 4, p.Type.Space.Global)))),
      Nil,
      Nil,
      p.Type.Unit0,
      p.Function.Affinity.Host
    )
    val cpp = LibraryCodeGen.cppHeader(library.copy(decls = List(array)))

    assertEquals(cpp, golden("example-array.hpp"))
    assert(cpp.contains("inline void array(std::int32_t (&values)[4])"))
  }

  test("Scala preserves nested callable structure and qualifies struct types from the root") {
    val nested = p.FunctionDecl(
      p.Sym("example.nested"),
      Nil,
      None,
      List(
        p.Arg(
          p.Named(
            "op",
            p.Type.Exec(Nil, List(p.Type.Exec(Nil, List(p.Type.IntS32), p.Type.IntS32)), p.Type.Bool1)
          )
        ),
        p.Arg(p.Named("value", p.Type.Struct(p.Sym("model.Value"), Nil)))
      ),
      Nil,
      Nil,
      p.Type.Unit0,
      p.Function.Affinity.Host
    )
    val source = LibraryCodeGen.scalaTrait(library.copy(decls = List(nested)), scala)

    assert(source.contains("op: (Int => Int) => Boolean"))
    assert(source.contains("value: _root_.model.Value"))
  }

  test("lossy overloads and Fortran case collisions are rejected") {
    val sameArity     = library.copy(decls = List(count, count.copy(rtn = p.Type.IntS64)))
    val overloadError = intercept[IllegalArgumentException](LibraryCodeGen.scalaTrait(sameArity, scala))
    assert(overloadError.getMessage.contains("cannot portably distinguish overloads"))

    val lower = count.copy(name = p.Sym("example.lookup"))
    val upper = countValue.copy(name = p.Sym("example.Lookup"))
    val caseError = intercept[IllegalArgumentException](
      LibraryCodeGen.fortranModule(library.copy(decls = List(lower, upper)), fortran)
    )
    assert(caseError.getMessage.contains("differ only by case"))
  }

  test("target identifiers are validated before source emission") {
    val cppReserved = library.copy(name = p.Sym("class"), decls = List(count.copy(name = p.Sym("class.count"))))
    assert(intercept[IllegalArgumentException](LibraryCodeGen.cppHeader(cppReserved)).getMessage.contains("reserved"))

    val scalaReserved = library.copy(decls = List(count.copy(name = p.Sym("example.match"))))
    assert(
      intercept[IllegalArgumentException](LibraryCodeGen.scalaTrait(scalaReserved, scala)).getMessage
        .contains("reserved")
    )

    val fortranInvalid =
      library.copy(decls = List(count.copy(args = count.args.updated(1, p.Arg(p.Named("_n", p.Type.IntS32))))))
    assert(
      intercept[IllegalArgumentException](LibraryCodeGen.fortranModule(fortranInvalid, fortran)).getMessage.contains(
        "invalid Fortran"
      )
    )
  }
}
