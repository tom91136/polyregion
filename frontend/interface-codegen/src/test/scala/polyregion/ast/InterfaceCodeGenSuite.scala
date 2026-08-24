package polyregion.ast

import java.nio.file.Files
import scala.io.Source
import scala.util.Using

import polyregion.ast.PolyAST as p

class InterfaceCodeGenSuite extends munit.FunSuite {

  private def golden(name: String): String =
    Using.resource(Source.fromResource(s"interface-codegen/$name"))(_.mkString)

  private val t = p.Type.Var("T")
  private val u = p.Type.Var("U")
  private val transform = p.FunctionDecl(
    p.Sym("example.transform"),
    List(p.Type.Var("T"), p.Type.Var("U")),
    None,
    List(
      p.Arg(
        p.Named("in", p.Type.Ptr(t, p.Type.Space.Global)),
        boundary = Some(p.Arg.Boundary(p.Arg.Access.Read, p.Arg.Extent.Elements(p.Arg.SizeExpr.Param(2))))
      ),
      p.Arg(
        p.Named("out", p.Type.Ptr(u, p.Type.Space.Global)),
        boundary = Some(p.Arg.Boundary(p.Arg.Access.Write, p.Arg.Extent.Elements(p.Arg.SizeExpr.Param(2))))
      ),
      p.Arg(p.Named("n", p.Type.IntS32)),
      p.Arg(p.Named("op", p.Type.Exec(Nil, List(t), u)))
    ),
    Nil,
    Nil,
    p.Type.Unit0,
    p.Function.Affinity.Host
  )
  private val count = p.FunctionDecl(
    p.Sym("example.count"),
    List(p.Type.Var("T")),
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
  private val countValue   = count.copy(args = count.args :+ p.Arg(p.Named("value", t)))
  private val interfaceDef = p.Interface(p.Sym("example"), List(transform, count))
  private val fortran      = InterfaceCodeGen.FortranConfig("example_ffi")
  private val scala        = InterfaceCodeGen.ScalaConfig("example.bindings", "ExampleInterface")

  test("interface declarations project exact identities into C++, Fortran and Scala") {
    val cpp = InterfaceCodeGen.cppHeader(interfaceDef)
    val f90 = InterfaceCodeGen.fortranModule(interfaceDef, fortran)
    val sc  = InterfaceCodeGen.scalaObject(interfaceDef, scala)

    assertEquals(cpp, golden("example.hpp"))
    assertEquals(f90, golden("example.f90"))
    assertEquals(sc, golden("ExampleInterface.scala"))

    assert(cpp.contains("template <class T, class U, class Op>"))
    assert(cpp.contains("clang::annotate(\"polyregion_interface:example:example.transform\")"))
    assert(cpp.contains("inline void transform(const T *in, U *out, std::int32_t n, Op op)"))

    assert(f90.contains("module example_ffi"))
    assert(f90.contains("polyregion_interface:example:example.transform"))
    assert(f90.contains("type(*), dimension(*), intent(in) :: in"))
    assert(f90.contains("type(*), dimension(*), intent(inout) :: out"))
    assert(f90.contains("procedure() :: op"))
    assert(!f90.contains("polyregion_transform_r4"))

    assert(sc.contains("package example.bindings"))
    assert(sc.contains("object ExampleInterface"))
    assert(!sc.contains("trait ExampleInterface"))
    assert(sc.contains("@compileTimeOnly(\"polyregion_interface:example:example.transform\")"))
    assert(sc.contains("def transform[T, U](in: Array[T], out: Array[U], n: Int, op: T => U): Unit"))
    assert(!(cpp + f90 + sc).toLowerCase.contains("spectra"))
  }

  test("Fortran adapts erased generic results to output arguments") {
    val reduce = p.FunctionDecl(
      p.Sym("example.reduce"),
      List(p.Type.Var("T")),
      None,
      List(
        transform.args.head.copy(
          boundary = Some(p.Arg.Boundary(p.Arg.Access.Read, p.Arg.Extent.Elements(p.Arg.SizeExpr.Param(1))))
        ),
        p.Arg(p.Named("n", p.Type.IntS32)),
        p.Arg(p.Named("init", t)),
        p.Arg(p.Named("op", p.Type.Exec(Nil, List(t, t), t)))
      ),
      Nil,
      Nil,
      t,
      p.Function.Affinity.Host
    )
    val f90 = InterfaceCodeGen.fortranModule(interfaceDef.copy(declarations = List(reduce)), fortran)

    assert(f90.contains("subroutine polyregion_reduce(in, n, init, op, polyregion_result)"))
    assert(f90.contains("type(*), intent(in) :: init"))
    assert(f90.contains("procedure() :: op"))
    assert(f90.contains("type(*), intent(inout) :: polyregion_result"))
    assert(!f90.contains("function polyregion_reduce"))

    val collision = reduce.copy(args = reduce.args :+ p.Arg(p.Named("polyregion_result", p.Type.IntS32)))
    val renamed   = InterfaceCodeGen.fortranModule(interfaceDef.copy(declarations = List(collision)), fortran)
    assert(renamed.contains("polyregion_result, polyregion_result_)"))
    assert(renamed.contains("type(*), intent(inout) :: polyregion_result_"))
  }

  test("Fortran rejects overloads that collide after generic result adaptation") {
    val returned = p.FunctionDecl(
      p.Sym("example.select"),
      List(p.Type.Var("T")),
      None,
      List(p.Arg(p.Named("value", t))),
      Nil,
      Nil,
      t,
      p.Function.Affinity.Host
    )
    val unit = returned.copy(args = List(p.Arg(p.Named("lhs", t)), p.Arg(p.Named("rhs", t))), rtn = p.Type.Unit0)
    val error = intercept[IllegalArgumentException](
      InterfaceCodeGen.fortranModule(interfaceDef.copy(declarations = List(returned, unit)), fortran)
    )

    assert(error.getMessage.contains("after result adaptation with 2 arguments"))
  }

  test("Fortran rejects overloads that mix projected functions and subroutines") {
    val function = p.FunctionDecl(
      p.Sym("example.select"),
      Nil,
      None,
      List(p.Arg(p.Named("value", p.Type.IntS32))),
      Nil,
      Nil,
      p.Type.IntS32,
      p.Function.Affinity.Host
    )
    val subroutine = function.copy(
      args = List(p.Arg(p.Named("lhs", p.Type.IntS32)), p.Arg(p.Named("rhs", p.Type.IntS32))),
      rtn = p.Type.Unit0
    )
    val error = intercept[IllegalArgumentException](
      InterfaceCodeGen.fortranModule(interfaceDef.copy(declarations = List(function, subroutine)), fortran)
    )

    assert(error.getMessage.contains("cannot combine function and subroutine overloads"))
  }

  test("Fortran erases explicitly named struct values") {
    val point = p.Type.Struct(p.Sym("model.Point"), Nil)
    val inspect = p.FunctionDecl(
      p.Sym("example.inspect_point"),
      Nil,
      None,
      List(
        p.Arg(
          p.Named("points", p.Type.Ptr(point, p.Type.Space.Global)),
          boundary = Some(p.Arg.Boundary(p.Arg.Access.Read, p.Arg.Extent.Elements(p.Arg.SizeExpr.Param(1))))
        ),
        p.Arg(p.Named("n", p.Type.IntS32)),
        p.Arg(p.Named("point", point))
      ),
      Nil,
      Nil,
      p.Type.Unit0,
      p.Function.Affinity.Host
    )
    val f90 = InterfaceCodeGen.fortranModule(interfaceDef.copy(declarations = List(inspect)), fortran)

    assert(f90.contains("type(*), dimension(*), intent(in) :: points"))
    assert(f90.contains("type(*), intent(in) :: point"))
  }

  test("projection order and stale-output checks are deterministic") {
    val reversed = interfaceDef.copy(declarations = interfaceDef.declarations.reverse)
    val cpp      = InterfaceCodeGen.cppHeader(interfaceDef)

    assertEquals(InterfaceCodeGen.cppHeader(reversed), cpp)
    assertEquals(
      InterfaceCodeGen.fortranModule(reversed, fortran),
      InterfaceCodeGen.fortranModule(interfaceDef, fortran)
    )
    assertEquals(InterfaceCodeGen.scalaObject(reversed, scala), InterfaceCodeGen.scalaObject(interfaceDef, scala))

    val path = Files.createTempFile("polyregion-interface-codegen", ".hpp")
    Files.writeString(path, "stale")
    assert(InterfaceCodeGen.checkCurrent(path, cpp).isLeft)
    Files.writeString(path, cpp)
    assertEquals(InterfaceCodeGen.checkCurrent(path, cpp), Right(()))
    assert(InterfaceCodeGen.checkCurrent(path.resolveSibling("missing-generated-output"), cpp).isLeft)
  }

  test("overloads retain distinct Fortran procedures") {
    val overloaded = interfaceDef.copy(declarations = countValue :: interfaceDef.declarations)
    val f90        = InterfaceCodeGen.fortranModule(overloaded, fortran)

    assert(f90.contains("module procedure polyregion_count_o0"))
    assert(f90.contains("module procedure polyregion_count_o1"))
    assertEquals(
      InterfaceCodeGen.fortranModule(overloaded.copy(declarations = overloaded.declarations.reverse), fortran),
      f90
    )
  }

  test("pointer declarations require boundary metadata") {
    val invalid =
      interfaceDef.copy(declarations =
        List(count.copy(args = count.args.updated(0, count.args.head.copy(boundary = None))))
      )
    val error = intercept[IllegalArgumentException](InterfaceCodeGen.cppHeader(invalid))

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
    val cpp = InterfaceCodeGen.cppHeader(interfaceDef.copy(declarations = List(array)))

    assertEquals(cpp, golden("example-array.hpp"))
    assert(cpp.contains("inline void array(std::int32_t (&values)[4])"))
  }

  test("C++ callables are checked against const lvalue arguments") {
    val inspect = p.FunctionDecl(
      p.Sym("example.inspect"),
      List(p.Type.Var("T")),
      None,
      List(p.Arg(p.Named("op", p.Type.Exec(Nil, List(t), t)))),
      Nil,
      Nil,
      p.Type.Unit0,
      p.Function.Affinity.Host
    )
    val cpp = InterfaceCodeGen.cppHeader(interfaceDef.copy(declarations = List(inspect)))

    assert(cpp.contains("std::invoke_result_t<Op &, const T &>"))
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
    val source = InterfaceCodeGen.scalaObject(interfaceDef.copy(declarations = List(nested)), scala)

    assert(source.contains("op: (Int => Int) => Boolean"))
    assert(source.contains("value: _root_.model.Value"))
  }

  test("lossy overloads and Fortran case collisions are rejected") {
    val sameArity     = interfaceDef.copy(declarations = List(count, count.copy(rtn = p.Type.IntS64)))
    val overloadError = intercept[IllegalArgumentException](InterfaceCodeGen.scalaObject(sameArity, scala))
    assert(overloadError.getMessage.contains("cannot portably distinguish overloads"))

    val lower = count.copy(name = p.Sym("example.lookup"))
    val upper = countValue.copy(name = p.Sym("example.Lookup"))
    val caseError = intercept[IllegalArgumentException](
      InterfaceCodeGen.fortranModule(interfaceDef.copy(declarations = List(lower, upper)), fortran)
    )
    assert(caseError.getMessage.contains("differ only by case"))
  }

  test("target identifiers are validated before source emission") {
    val cppReserved =
      interfaceDef.copy(name = p.Sym("class"), declarations = List(count.copy(name = p.Sym("class.count"))))
    assert(intercept[IllegalArgumentException](InterfaceCodeGen.cppHeader(cppReserved)).getMessage.contains("reserved"))

    val scalaReserved = interfaceDef.copy(declarations = List(count.copy(name = p.Sym("example.match"))))
    assert(
      intercept[IllegalArgumentException](InterfaceCodeGen.scalaObject(scalaReserved, scala)).getMessage
        .contains("reserved")
    )

    val fortranInvalid =
      interfaceDef.copy(
        declarations = List(count.copy(args = count.args.updated(1, p.Arg(p.Named("_n", p.Type.IntS32)))))
      )
    assert(
      intercept[IllegalArgumentException](InterfaceCodeGen.fortranModule(fortranInvalid, fortran)).getMessage.contains(
        "invalid Fortran"
      )
    )
  }
}
