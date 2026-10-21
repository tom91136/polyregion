package polyregion.spectra

import java.nio.file.{Files, Paths}

import polyregion.ast.PolyAST as p
import polyregion.ast.{validate, validateInterfaceDeclaration}

import scala.jdk.CollectionConverters.*
import scala.util.Using

class SpectraSuite extends munit.FunSuite {

  private val root = Paths.get(sys.props("user.dir")).toAbsolutePath.normalize

  test("declarations are structurally valid") {
    val interfaceDef = Spectra.interfaceDef

    assertEquals(interfaceDef.name, p.Sym("spectra"))
    assertEquals(interfaceDef.declarations.flatMap(_.validate), Nil)
    assertEquals(interfaceDef.declarations.flatMap(_.validateInterfaceDeclaration), Nil)
  }

  test("generated APIs are current") {
    assertEquals(SpectraCodeGen.checkGenerated(root), Nil)
  }

  test("generated C++ test matrix covers every declaration") {
    val dir = root.resolve("test/generated")
    val actual = Using.resource(Files.list(dir)) { paths =>
      paths.iterator.asScala
        .filter(path => Files.isRegularFile(path) && path.getFileName.toString.endsWith(".cpp"))
        .map(_.getFileName.toString)
        .toSet
    }
    val expected     = Spectra.interfaceDef.declarations.map(decl => s"${decl.name.last}.cpp").toSet
    val declarations = Spectra.interfaceDef.declarations.map(decl => s"${decl.name.last}.cpp" -> decl).toMap

    assertEquals(actual, expected)
    assertEquals(actual.size, 71)
    actual.foreach { name =>
      val text = Files.readString(dir.resolve(name))
      assert(text.contains(s"#pragma region case: ${name.stripSuffix(".cpp")}"), name)
      assert(text.contains("#pragma region do: polycpp {spectra_stdpar} -o {output} {input}"), name)
      assert(text.contains("#pragma region requires: pass"), name)
      assert(text.contains("#include <polyregion/spectra.hpp>"), name)
      assert(text.contains("#include \"support.hpp\""), name)
      assert(text.contains(s"// ${SpectraTestCodeGen.Banner}"), name)
      assert(text.contains("spectra_test_value4"), name)
      assert(text.contains("spectra_test_value8"), name)
      assert(text.contains("spectra_test_value12"), name)
      declarations(name).tpeVars.size match {
        case 1 => ()
        case 2 => assert(text.contains("w4-w8/"), name)
        case 3 =>
          assert(text.contains("w4-w8-w8/"), name)
          assert(text.contains("w12-w12-w8/"), name)
        case arity => fail(s"$name has unsupported type arity $arity")
      }
    }
  }

  test("generated C++ coverage combines rather than multiplies type and size rows") {
    val reduce = Files.readString(root.resolve("test/generated/reduce.cpp"))
    assert(reduce.contains("w4/n31"))
    assert(reduce.contains("w4/n777"))
    assert(!reduce.contains("w8/n31"))
    assert(!reduce.contains("w12/n777"))

    val transformBinary = Files.readString(root.resolve("test/generated/transform_binary.cpp"))
    assertEquals("ok &= run<".r.findAllIn(transformBinary).size, 15)
  }

  test("generated Scala and Fortran consumers instantiate every declaration") {
    val scalaSurface   = Files.readString(root.resolve("test/scala/polyregion/check_surface.scala"))
    val fortranSurface = Files.readString(root.resolve("test/fortran/check_surface.f90"))

    Spectra.interfaceDef.declarations.foreach { decl =>
      val name = decl.name.last
      assertEquals(raw"spectra\.$name\(".r.findAllIn(scalaSurface).size, 1, s"Scala: $name")
      assertEquals(raw"polyregion_$name\(".r.findAllIn(fortranSurface).size, 1, s"Fortran: $name")
    }
    List("w4", "w8", "w12").foreach { width =>
      assert(scalaSurface.contains(width), s"Scala surface has no $width instantiation")
      assert(fortranSurface.contains(width), s"Fortran surface has no $width instantiation")
    }
  }

  test("generated C++ API exposes one native implementation hook") {
    val header = SpectraCodeGen.cppHeader

    assert(header.contains("#ifdef POLYREGION_SPECTRA_IMPLEMENT"))
    assert(header.contains("POLYREGION_SPECTRA_IMPLEMENT(reduce, in, n, init, op);"))
    assertEquals(
      "(?<!#define )POLYREGION_SPECTRA_IMPLEMENT\\(".r.findAllIn(header).size,
      Spectra.interfaceDef.declarations.size
    )
  }

  test("public type variables are unconstrained") {
    val variables = Spectra.interfaceDef.declarations.flatMap(_.tpeVars)
    assert(variables.nonEmpty)
    assert(variables.forall(_.exactSizeInBytes.isEmpty))
  }

  test("erased value algorithms carry their semantics as callables") {
    val callableArguments = Spectra.interfaceDef.declarations.map { decl =>
      decl.name.last -> decl.args.count(_.named.tpe match {
        case _: p.Type.Exec => true
        case _              => false
      })
    }.toMap

    List("count", "find", "remove", "remove_copy", "replace", "replace_copy", "sequence").foreach { name =>
      assertEquals(callableArguments(name), 1, name)
    }
  }
}
