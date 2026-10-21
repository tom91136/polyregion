package polyregion.spectra

import java.nio.file.Path

import polyregion.ast.PolyAST as p

private[spectra] object SpectraTestCodeGen {

  val Banner = "AUTO-GENERATED from polyregion.spectra.Spectra.interfaceDef via SpectraCodeGen. DO NOT EDIT."
  private val declarations = Spectra.interfaceDef.declarations

  private final class CppBlock private (val lines: List[String])

  private object CppBlock {
    def apply(source: String): CppBlock = new CppBlock(source.linesIterator.toList)
  }

  private case class SemanticCase(
      n: Int,
      setup: CppBlock, // C++ that declares + fills inputs/outputs, scalars, sizes, and the host ref
      functors: Map[String, String] = Map(), // functor param name -> body (elem/pred: raw expr; mutate: statement)
      verify: CppBlock                       // C++ that sets `bool ok` from device output / __r vs the ref
  )

  private enum SizeExpr {
    case Fixed(value: Int)
    case Around(boundary: Int)

    def values: List[Int] = this match {
      case Fixed(value)     => List(value)
      case Around(boundary) => List(boundary - 1, boundary, boundary + 1)
    }
  }

  private val boundarySizeProfile = List(SizeExpr.Around(32), SizeExpr.Around(256), SizeExpr.Fixed(777))
  private val sizeSweptAlgorithms = Set(
    "reduce",
    "transform",
    "inclusive_scan",
    "exclusive_scan",
    "transform_reduce",
    "transform_inclusive_scan",
    "transform_exclusive_scan",
    "transform_binary",
    "inner_product",
    "copy",
    "reverse",
    "fill",
    "for_each",
    "adjacent_difference"
  )
  private val emptySizeAlgorithms = Set("min_element", "max_element", "minmax_element")
  private val multiGroupAlgorithms =
    Set("find", "inclusive_scan", "exclusive_scan", "transform_inclusive_scan", "transform_exclusive_scan")

  private case class WidthCase(types: List[String], label: String)

  private val w4  = "spectra_test_value4"
  private val w8  = "spectra_test_value8"
  private val w12 = "spectra_test_value12"

  private def widthCase(types: String*): WidthCase =
    WidthCase(types.toList, types.iterator.map(_.stripPrefix("spectra_test_value")).mkString("w", "-w", ""))

  private def widthCases(arity: Int): List[WidthCase] = arity match {
    case 1 => List(widthCase(w4), widthCase(w8), widthCase(w12))
    case 2 =>
      List(
        widthCase(w4, w4),
        widthCase(w8, w8),
        widthCase(w12, w12),
        widthCase(w4, w8),
        widthCase(w8, w12),
        widthCase(w12, w4)
      )
    case 3 =>
      List(
        widthCase(w4, w4, w4),
        widthCase(w4, w8, w8),
        widthCase(w4, w12, w12),
        widthCase(w8, w4, w8),
        widthCase(w8, w8, w12),
        widthCase(w8, w12, w4),
        widthCase(w12, w4, w12),
        widthCase(w12, w8, w4),
        widthCase(w12, w12, w8)
      )
    case other => sys.error(s"unsupported Spectra test type arity: $other")
  }

  private def testSizes(name: String, baseline: Int): List[Int] = {
    val swept = if (sizeSweptAlgorithms(name)) boundarySizeProfile.flatMap(_.values) else Nil
    val empty = if (emptySizeAlgorithms(name)) List(0) else Nil
    val multi = if (multiGroupAlgorithms(name)) List(1 << 18) else Nil
    (swept ++ empty ++ multi).distinct.filterNot(_ == baseline)
  }

  private val testHelpers =
    """using spectra_test_value4 = std::int32_t;
      #using spectra_test_value8 = std::int64_t;
      #
      #template <class T> struct spectra_test_allocator {
      #  using value_type = T;
      #
      #  spectra_test_allocator() noexcept = default;
      #  template <class U> spectra_test_allocator(const spectra_test_allocator<U> &) noexcept {}
      #
      #  [[nodiscard]] T *allocate(std::size_t n) {
      ##if defined(SPECTRA_NATIVE_THRUST) && defined(__HIPCC__)
      #    void *ptr = nullptr;
      #    if (hipMallocManaged(&ptr, n * sizeof(T)) != hipSuccess) throw std::bad_alloc();
      #    return static_cast<T *>(ptr);
      ##elif defined(SPECTRA_NATIVE_THRUST) && defined(__CUDACC__)
      #    void *ptr = nullptr;
      #    if (cudaMallocManaged(&ptr, n * sizeof(T)) != cudaSuccess) throw std::bad_alloc();
      #    return static_cast<T *>(ptr);
      ##elif defined(SPECTRA_NATIVE_ONEDPL)
      #    T *ptr = sycl::malloc_shared<T>(n, oneapi::dpl::execution::dpcpp_default.queue());
      #    if (!ptr) throw std::bad_alloc();
      #    return ptr;
      ##else
      #    return std::allocator<T>{}.allocate(n);
      ##endif
      #  }
      #
      #  void deallocate(T *ptr, std::size_t n) noexcept {
      ##if defined(SPECTRA_NATIVE_THRUST) && defined(__HIPCC__)
      #    (void)n;
      #    (void)hipFree(ptr);
      ##elif defined(SPECTRA_NATIVE_THRUST) && defined(__CUDACC__)
      #    (void)n;
      #    (void)cudaFree(ptr);
      ##elif defined(SPECTRA_NATIVE_ONEDPL)
      #    (void)n;
      #    sycl::free(ptr, oneapi::dpl::execution::dpcpp_default.queue());
      ##else
      #    std::allocator<T>{}.deallocate(ptr, n);
      ##endif
      #  }
      #
      #  template <class U> bool operator==(const spectra_test_allocator<U> &) const noexcept { return true; }
      #  template <class U> bool operator!=(const spectra_test_allocator<U> &) const noexcept { return false; }
      #};
      #
      #template <class T> using spectra_test_vector = std::vector<T, spectra_test_allocator<T>>;
      #
      #struct spectra_test_value12 {
      #  std::int32_t x, y, z;
      #  SPECTRA_TEST_FN constexpr spectra_test_value12(std::int32_t value = 0) : x(value), y(value), z(value) {}
      #  SPECTRA_TEST_FN constexpr spectra_test_value12(std::int32_t x_, std::int32_t y_, std::int32_t z_) : x(x_), y(y_), z(z_) {}
      #  template <class I, typename std::enable_if<std::is_integral<I>::value, int>::type = 0>
      #  SPECTRA_TEST_FN constexpr explicit operator I() const {
      #    return static_cast<I>(x);
      #  }
      #};
      #static_assert(sizeof(spectra_test_value4) == 4);
      #static_assert(sizeof(spectra_test_value8) == 8);
      #static_assert(sizeof(spectra_test_value12) == 12);
      #static_assert(std::is_trivially_copyable_v<spectra_test_value12>);
      #
      #SPECTRA_TEST_FN constexpr spectra_test_value12 operator+(spectra_test_value12 a, spectra_test_value12 b) {
      #  return {a.x + b.x, a.y + b.y, a.z + b.z};
      #}
      #SPECTRA_TEST_FN constexpr spectra_test_value12 operator-(spectra_test_value12 a, spectra_test_value12 b) {
      #  return {a.x - b.x, a.y - b.y, a.z - b.z};
      #}
      #SPECTRA_TEST_FN constexpr spectra_test_value12 operator*(spectra_test_value12 a, spectra_test_value12 b) {
      #  return {a.x * b.x, a.y * b.y, a.z * b.z};
      #}
      #SPECTRA_TEST_FN constexpr bool operator==(spectra_test_value12 a, spectra_test_value12 b) { return a.x == b.x && a.y == b.y && a.z == b.z; }
      #SPECTRA_TEST_FN constexpr bool operator!=(spectra_test_value12 a, spectra_test_value12 b) { return !(a == b); }
      #SPECTRA_TEST_FN constexpr bool operator<(spectra_test_value12 a, spectra_test_value12 b) {
      #  return a.x != b.x ? a.x < b.x : a.y != b.y ? a.y < b.y : a.z < b.z;
      #}
      #SPECTRA_TEST_FN constexpr bool operator>(spectra_test_value12 a, spectra_test_value12 b) { return b < a; }
      #SPECTRA_TEST_FN constexpr bool operator<=(spectra_test_value12 a, spectra_test_value12 b) { return !(b < a); }
      #SPECTRA_TEST_FN constexpr bool operator>=(spectra_test_value12 a, spectra_test_value12 b) { return !(a < b); }
      #
      #template <class T> static bool near_(T a, T b) {
      #  if constexpr (std::is_floating_point<T>::value) {
      #    double d = a > b ? double(a) - double(b) : double(b) - double(a);
      #    double m = double(a > 0 ? a : -a), mb = double(b > 0 ? b : -b), s = m > mb ? m : mb;
      #    return d <= 1e-6 * (s + 1.0);
      #  } else return a == b;
      #}
      #template <class A, class B> static bool vecNear(const A &a, const B &b, int n) {
      #  for (int i = 0; i < n; ++i)
      #    if (!near_(a[i], b[i])) return false;
      #  return true;
      #}
      #template <class A, class B> static bool sameSeq(const A &a, const B &b, int n) {
      #  for (int i = 0; i < n; ++i)
      #    if (a[i] != b[i]) return false;
      #  return true;
      #}""".stripMargin('#')

  private def cppTestHeader: String =
    s"""#pragma once
       |
       |// $Banner
       |// Host-oracle comparison helpers shared by the generated Spectra API tests.
       |#include <cstddef>
       |#include <cstdint>
       |#include <cstdio>
       |#include <memory>
       |#include <new>
       |#include <type_traits>
       |#include <vector>
       |
       |#if defined(SPECTRA_NATIVE_THRUST) && defined(__HIPCC__)
       |  #include <hip/hip_runtime_api.h>
       |  #define SPECTRA_TEST_FN __host__ __device__
       |#elif defined(SPECTRA_NATIVE_THRUST) && defined(__CUDACC__)
       |  #include <cuda_runtime_api.h>
       |  #define SPECTRA_TEST_FN __host__ __device__
       |#elif defined(SPECTRA_NATIVE_ONEDPL)
       |  #include <oneapi/dpl/execution>
       |  #include <sycl/sycl.hpp>
       |  #define SPECTRA_TEST_FN
       |#else
       |  #define SPECTRA_TEST_FN
       |#endif
       |
       |$testHelpers
       |""".stripMargin

  private def indentBlock(block: CppBlock, pad: String): String =
    block.lines.map(line => if (line.isEmpty) line else pad + line).mkString("\n")

  private def cppType(tpe: p.Type): String = tpe match {
    case p.Type.Var(name, _) => name
    case p.Type.IntS32       => "int"
    case p.Type.Bool1        => "bool"
    case p.Type.Unit0        => "void"
    case other               => sys.error(s"unsupported Spectra test type: $other")
  }

  private def cppTestFunctor(
      name: String,
      signature: p.Type.Exec,
      body: String,
      useOnedplKeyEqual: Boolean,
      captured: Boolean
  ): String = {
    val names = signature.args match {
      case Nil                       => Nil
      case List(tpe)                 => List(if (tpe == p.Type.IntS32) "i" else "x")
      case List(_, _)                => List("a", "b")
      case List(_, _, p.Type.IntS32) => List("a", "b", "i")
      case other                     => sys.error(s"unsupported Spectra callable arity: ${other.size}")
    }
    val parameters = signature.args.zip(names).map { case (tpe, arg) => s"${cppType(tpe)} $arg" }.mkString(", ")
    val result = signature.rtn match {
      case p.Type.Unit0 => s"$body;"
      case rtn          => s"return ${cppType(rtn)}($body);"
    }
    val capturedResult = signature.rtn match {
      case p.Type.Unit0 => s"live ? void($body) : void($body);"
      case rtn          => s"return live ? ${cppType(rtn)}($body) : ${cppType(rtn)}($body);"
    }
    val lambda =
      if (captured) s"  const auto $name = [live] SPECTRA_TEST_FN($parameters) { $capturedResult };"
      else s"  const auto $name = [] SPECTRA_TEST_FN($parameters) { $result };"
    if (useOnedplKeyEqual)
      s"""#if defined(SPECTRA_NATIVE_ONEDPL)
         |  const spectra_test_key_equal<${cppType(signature.args.head)}> $name{};
         |#else
         |$lambda
         |#endif""".stripMargin
    else lambda
  }

  private def needsOnedplKeyEqual(name: String): Boolean =
    name == "inclusive_scan_by_key" || name == "exclusive_scan_by_key"

  private def cppTestCallArg(arg: p.Arg): String = arg.named.tpe match {
    case p.Type.Ptr(_, _) => s"${arg.named.symbol}.data()"
    case _                => arg.named.symbol
  }

  private def cppTestRetCap(tpe: p.Type): String = tpe match {
    case p.Type.Unit0 => ""
    case other        => s"const ${cppType(other)} __r = "
  }

  private def callableArgs(decl: p.FunctionDecl): List[p.Arg] =
    decl.args.collect { case arg @ p.Arg(p.Named(_, _: p.Type.Exec, _), _, _) => arg }

  private def cppTestRun(decl: p.FunctionDecl, captured: Boolean): String = {
    val name = decl.name.last
    val sem  = oracles.getOrElse(name, sys.error(s"no test oracle for algo '$name'"))
    val fns  = callableArgs(decl)
    val fdecls = fns
      .map(arg =>
        cppTestFunctor(
          arg.named.symbol,
          arg.named.tpe.asInstanceOf[p.Type.Exec],
          sem.functors.getOrElse(
            arg.named.symbol,
            sys.error(s"algo '$name': no functor body for '${arg.named.symbol}'")
          ),
          arg.named.symbol == "eq" && needsOnedplKeyEqual(name),
          captured
        )
      )
      .mkString("\n")
    val (runName, live) = if (captured) ("run_captured", "\n  const bool live = n >= 0;") else ("run", "")
    val templateParams  = decl.tpeVars.map(variable => s"class ${variable.name}").mkString(", ")
    val templateArgs    = decl.tpeVars.map(_.name).mkString(", ")
    val call =
      s"  ${cppTestRetCap(decl.rtn)}spectra::$name<$templateArgs>(${decl.args.map(cppTestCallArg).mkString(", ")});"
    s"""template <$templateParams> static bool $runName(const char *scenario, int n) {
       |${indentBlock(sem.setup, "  ")}$live
       |$fdecls
       |$call
       |${indentBlock(sem.verify, "  ")}
       |  std::fprintf(stderr, "  %-24s<%-18s> %s\\n", "$name", scenario, ok ? "OK" : "MISMATCH");
       |  return ok;
       |}""".stripMargin
  }

  private def cppTestFile(decl: p.FunctionDecl): String = {
    val name = decl.name.last
    val sem  = oracles(name)
    val namespaceFunctors =
      if (needsOnedplKeyEqual(name)) {
        val body = sem.functors.getOrElse("eq", sys.error(s"algo '$name': no functor body for 'eq'"))
        s"""#if defined(SPECTRA_NATIVE_ONEDPL)
           |template <class K> struct spectra_test_key_equal {
           |  SPECTRA_TEST_FN bool operator()(K a, K b) const { return bool($body); }
           |};
           |#endif
           |""".stripMargin
      } else ""
    val widths = widthCases(decl.tpeVars.size)
    val typeCalls = widths
      .map(width => s"  ok &= run<${width.types.mkString(", ")}>(\"${width.label}/n${sem.n}\", ${sem.n});")
    val baseline = widths.head
    val sizeCalls = testSizes(name, sem.n)
      .map(n => s"  ok &= run<${baseline.types.mkString(", ")}>(\"${baseline.label}/n$n\", $n);")
    val capturedCalls =
      if (callableArgs(decl).isEmpty) Nil
      else
        (sem.n :: testSizes(name, sem.n).maxOption.toList).map(n =>
          s"  ok &= run_captured<${baseline.types.mkString(", ")}>(\"${baseline.label}/n$n/captured\", $n);"
        )
    val runs =
      if (callableArgs(decl).isEmpty) cppTestRun(decl, captured = false)
      else s"${cppTestRun(decl, captured = false)}\n\n${cppTestRun(decl, captured = true)}"
    val calls = (typeCalls ++ sizeCalls ++ capturedCalls).mkString("\n")
    s"""#pragma region case: $name
       |#pragma region offload-only
       |#pragma region do: polycpp {spectra_stdpar} -o {output} {input}
       |#pragma region do: {output}
       |#pragma region requires: pass
       |
       |// $Banner
       |// Self-checking Spectra API test for `$name`: deterministic input -> device call -> host-oracle comparison.
       |#include <algorithm>
       |#include <cstdio>
       |#include <iterator>
       |
       |#include <polyregion/spectra.hpp>
       |
       |#include "support.hpp"
       |
       |$namespaceFunctors$runs
       |
       |int main() {
       |  bool ok = true;
       |$calls
       |  std::printf("%s", ok ? "pass" : "fail");
       |  return ok ? 0 : 1;
       |}
       |""".stripMargin
  }

  private case class SurfaceValue(
      suffix: String,
      scalaType: String,
      scalaZero: String,
      fortranType: String,
      fortranZero: String
  )

  private val surfaceValues = List(
    SurfaceValue("4", "Int", "0", "integer(c_int32_t)", "0_c_int32_t"),
    SurfaceValue("8", "Long", "0L", "integer(c_int64_t)", "0_c_int64_t"),
    SurfaceValue(
      "12",
      "SpectraSurfaceValue12",
      "SpectraSurfaceValue12(0, 0, 0)",
      "type(spectra_surface_value12)",
      "spectra_surface_value12(0, 0, 0)"
    )
  )
  private val surfaceBool = SurfaceValue("bool", "Boolean", "false", "logical(c_bool)", ".false.")
  private val surfaceUnit = SurfaceValue("unit", "Unit", "()", "", "")

  private def surfaceMapping(decl: p.FunctionDecl): Map[String, SurfaceValue] = {
    val base = Math.floorMod(decl.name.last.hashCode, surfaceValues.size)
    decl.tpeVars.zipWithIndex.map { case (variable, offset) =>
      variable.name -> surfaceValues((base + offset) % surfaceValues.size)
    }.toMap
  }

  private def surfaceValue(tpe: p.Type, mapping: Map[String, SurfaceValue]): SurfaceValue = tpe match {
    case p.Type.Var(name, _) => mapping.getOrElse(name, sys.error(s"unbound Spectra surface type variable: $name"))
    case p.Type.IntS32       => surfaceValues.head
    case p.Type.Bool1        => surfaceBool
    case p.Type.Unit0        => surfaceUnit
    case other               => sys.error(s"unsupported Spectra surface type: $other")
  }

  private def surfaceProfile(decl: p.FunctionDecl, mapping: Map[String, SurfaceValue]): String =
    decl.tpeVars.map(variable => s"${variable.name}=w${mapping(variable.name).suffix}").mkString(", ")

  private def scalaSurfaceCallable(signature: p.Type.Exec, mapping: Map[String, SurfaceValue]): String = {
    val parameters = signature.args.zipWithIndex
      .map { case (tpe, index) => s"arg$index: ${surfaceValue(tpe, mapping).scalaType}" }
      .mkString(", ")
    s"($parameters) => ${surfaceValue(signature.rtn, mapping).scalaZero}"
  }

  private def scalaSurfaceArg(arg: p.Arg, mapping: Map[String, SurfaceValue]): String = arg.named.tpe match {
    case p.Type.Ptr(comp, _)    => s"values${surfaceValue(comp, mapping).suffix}"
    case variable: p.Type.Var   => s"value${surfaceValue(variable, mapping).suffix}"
    case p.Type.IntS32          => "n"
    case signature: p.Type.Exec => scalaSurfaceCallable(signature, mapping)
    case other                  => surfaceValue(other, mapping).scalaZero
  }

  private def scalaSurface: String = {
    val calls = declarations.map { decl =>
      val mapping = surfaceMapping(decl)
      val profile = surfaceProfile(decl, mapping)
      val comment = Option.when(profile.nonEmpty)(s"    // ${decl.name.last}: $profile\n").getOrElse("")
      val args    = decl.args.map(scalaSurfaceArg(_, mapping))
      val inline  = s"    spectra.${decl.name.last}(${args.mkString(", ")})"
      val call =
        if (inline.length <= 120) inline
        else s"    spectra.${decl.name.last}(\n${args.map("      " + _).mkString(",\n")}\n    )"
      comment + call
    }
    s"""package polyregion
       |
       |// $Banner
       |// Compile-only instantiations of every generated Scala declaration; runtime package coverage is separate.
       |final case class SpectraSurfaceValue12(x: Int, y: Int, z: Int)
       |
       |object check_surface {
       |  def compile(): Unit = InterfaceSurfaceCheck.erase {
       |    val n        = 4
       |    val values4  = new Array[Int](n)
       |    val values8  = new Array[Long](n)
       |    val values12 = new Array[SpectraSurfaceValue12](n)
       |    val value4   = 0
       |    val value8   = 0L
       |    val value12  = SpectraSurfaceValue12(0, 0, 0)
       |
       |${calls.mkString("\n\n")}
       |  }
       |}
       |""".stripMargin
  }

  private def fortranSurfaceCallable(
      decl: p.FunctionDecl,
      arg: p.Arg,
      signature: p.Type.Exec,
      mapping: Map[String, SurfaceValue]
  ): String = {
    val name              = s"spectra_surface_${decl.name.last}_${arg.named.symbol}"
    val logicalParameters = signature.args.indices.map(index => s"arg$index").toList
    val parameters        = (logicalParameters ++ Option.when(signature.rtn != p.Type.Unit0)("r")).mkString(", ")
    val declarations = signature.args.zipWithIndex
      .map { case (tpe, index) => s"    ${surfaceValue(tpe, mapping).fortranType}, intent(in) :: arg$index" }
      .mkString("\n")
    if (signature.rtn == p.Type.Unit0)
      s"""  subroutine $name($parameters)
         |$declarations
         |  end subroutine $name""".stripMargin
    else {
      val result = surfaceValue(signature.rtn, mapping)
      s"""  subroutine $name($parameters)
         |$declarations
         |    ${result.fortranType}, intent(out) :: r
         |    r = ${result.fortranZero}
         |  end subroutine $name""".stripMargin
    }
  }

  private def fortranSurfaceArg(
      decl: p.FunctionDecl,
      arg: p.Arg,
      mapping: Map[String, SurfaceValue]
  ): String = arg.named.tpe match {
    case p.Type.Ptr(comp, _)  => s"values${surfaceValue(comp, mapping).suffix}"
    case variable: p.Type.Var => s"value${surfaceValue(variable, mapping).suffix}"
    case p.Type.IntS32        => "n"
    case _: p.Type.Exec       => s"spectra_surface_${decl.name.last}_${arg.named.symbol}"
    case other                => surfaceValue(other, mapping).fortranZero
  }

  private def fortranSurfaceCall(decl: p.FunctionDecl): String = {
    val mapping     = surfaceMapping(decl)
    val logicalArgs = decl.args.map(fortranSurfaceArg(decl, _, mapping))
    val (prefix, args) = decl.rtn match {
      case p.Type.Unit0 => s"call polyregion_${decl.name.last}" -> logicalArgs
      case variable: p.Type.Var =>
        s"call polyregion_${decl.name.last}" -> (logicalArgs :+ s"value${surfaceValue(variable, mapping).suffix}")
      case p.Type.Bool1  => s"logical_result = polyregion_${decl.name.last}" -> logicalArgs
      case p.Type.IntS32 => s"integer_result = polyregion_${decl.name.last}" -> logicalArgs
      case other         => sys.error(s"unsupported Spectra Fortran surface result: $other")
    }
    val formattedArgs = args.zipWithIndex.map { case (value, argIndex) =>
      if (argIndex == args.size - 1) s"    $value)" else s"    $value, &"
    }
    val profile = surfaceProfile(decl, mapping)
    val comment = Option.when(profile.nonEmpty)(s"  ! ${decl.name.last}: $profile\n").getOrElse("")
    s"$comment  $prefix( &\n${formattedArgs.mkString("\n")}"
  }

  private def fortranSurface: String = {
    val mapped = declarations.map(decl => decl -> surfaceMapping(decl))
    val callbacks = mapped.flatMap { case (decl, mapping) =>
      decl.args.collect { case arg @ p.Arg(p.Named(_, signature: p.Type.Exec, _), _, _) =>
        fortranSurfaceCallable(decl, arg, signature, mapping)
      }
    }
    val calls = declarations.map(fortranSurfaceCall)
    s"""! $Banner
       |! Compile-only instantiations of every generated Fortran declaration; runtime package coverage is separate.
       |module spectra_surface_values
       |  use iso_c_binding
       |  implicit none
       |  type, bind(c) :: spectra_surface_value12
       |    integer(c_int32_t) :: x, y, z
       |  end type spectra_surface_value12
       |contains
       |${callbacks.mkString("\n\n")}
       |end module spectra_surface_values
       |
       |program check
       |  use iso_c_binding
       |  use spectra_api
       |  use spectra_surface_values
       |  implicit none
       |  integer(c_int32_t), parameter :: n = 4
       |  integer(c_int32_t) :: values4(n), value4, integer_result
       |  integer(c_int64_t) :: values8(n), value8
       |  type(spectra_surface_value12) :: values12(n), value12
       |  logical(c_bool) :: logical_result
       |
       |${calls.mkString("\n\n")}
       |end program check
       |""".stripMargin
  }

  // oracles keyed by algo name; input data + functor op + host reference for each
  private val oracles: Map[String, SemanticCase] = Map(
    // --- reduce / scan / transform combine family (n=256, in[i]=i%7) ---
    "reduce" -> SemanticCase(
      256,
      CppBlock(
        """spectra_test_vector<T> in(n);
          |for (int i = 0; i < n; ++i)
          |  in[i] = T(i % 7);
          |const T init = T(3);
          |T ref = init;
          |for (int i = 0; i < n; ++i)
          |  ref = T(ref + in[i]);""".stripMargin
      ),
      Map("op" -> "a + b"),
      CppBlock("const bool ok = near_(__r, ref);")
    ),
    "transform" -> SemanticCase(
      256,
      CppBlock(
        """spectra_test_vector<T> in(n);
          |spectra_test_vector<U> out(n, U(0)), ref(n);
          |for (int i = 0; i < n; ++i) {
          |  in[i] = T(i % 7);
          |  ref[i] = U(in[i] * T(2) + T(1));
          |}""".stripMargin
      ),
      Map("op" -> "x * 2 + 1"),
      CppBlock("const bool ok = sameSeq(out, ref, n);")
    ),
    "inclusive_scan" -> SemanticCase(
      256,
      CppBlock(
        """spectra_test_vector<T> in(n), out(n, T(0)), ref(n);
          |for (int i = 0; i < n; ++i)
          |  in[i] = T(i % 7);
          |{
          |  T acc = in[0];
          |  ref[0] = acc;
          |  for (int i = 1; i < n; ++i) {
          |    acc = T(acc + in[i]);
          |    ref[i] = acc;
          |  }
          |}""".stripMargin
      ),
      Map("op" -> "a + b"),
      CppBlock("const bool ok = vecNear(out, ref, n);")
    ),
    "exclusive_scan" -> SemanticCase(
      256,
      CppBlock(
        """spectra_test_vector<T> in(n), out(n, T(0)), ref(n);
          |const T init = T(5);
          |for (int i = 0; i < n; ++i)
          |  in[i] = T(i % 7);
          |{
          |  T acc = init;
          |  for (int i = 0; i < n; ++i) {
          |    ref[i] = acc;
          |    acc = T(acc + in[i]);
          |  }
          |}""".stripMargin
      ),
      Map("op" -> "a + b"),
      CppBlock("const bool ok = vecNear(out, ref, n);")
    ),
    "transform_reduce" -> SemanticCase(
      256,
      CppBlock(
        """spectra_test_vector<T> in(n);
          |for (int i = 0; i < n; ++i)
          |  in[i] = T(i % 7);
          |const U init = U(1);
          |U ref = init;
          |for (int i = 0; i < n; ++i)
          |  ref = U(ref + U(in[i] * in[i]));""".stripMargin
      ),
      Map("map" -> "x * x", "op" -> "a + b"),
      CppBlock("const bool ok = near_(__r, ref);")
    ),
    "transform_inclusive_scan" -> SemanticCase(
      256,
      CppBlock(
        """spectra_test_vector<T> in(n);
          |spectra_test_vector<U> out(n, U(0)), ref(n);
          |for (int i = 0; i < n; ++i)
          |  in[i] = T(i % 7);
          |{
          |  U acc = U(in[0] + T(1));
          |  ref[0] = acc;
          |  for (int i = 1; i < n; ++i) {
          |    acc = U(acc + U(in[i] + T(1)));
          |    ref[i] = acc;
          |  }
          |}""".stripMargin
      ),
      Map("map" -> "x + 1", "op" -> "a + b"),
      CppBlock("const bool ok = vecNear(out, ref, n);")
    ),
    "transform_exclusive_scan" -> SemanticCase(
      256,
      CppBlock(
        """spectra_test_vector<T> in(n);
          |spectra_test_vector<U> out(n, U(0)), ref(n);
          |const U init = U(2);
          |for (int i = 0; i < n; ++i)
          |  in[i] = T(i % 7);
          |{
          |  U acc = init;
          |  for (int i = 0; i < n; ++i) {
          |    ref[i] = acc;
          |    acc = U(acc + U(in[i] + T(1)));
          |  }
          |}""".stripMargin
      ),
      Map("map" -> "x + 1", "op" -> "a + b"),
      CppBlock("const bool ok = vecNear(out, ref, n);")
    ),
    "transform_binary" -> SemanticCase(
      256,
      CppBlock(
        """spectra_test_vector<T> a(n);
          |spectra_test_vector<U> b(n);
          |spectra_test_vector<V> out(n, V(0)), ref(n);
          |for (int i = 0; i < n; ++i) {
          |  a[i] = T(i % 7);
          |  b[i] = U((i * 3) % 5);
          |  ref[i] = V(a[i]) + V(b[i]);
          |}""".stripMargin
      ),
      Map("op" -> "a + b"),
      CppBlock("const bool ok = sameSeq(out, ref, n);")
    ),
    "inner_product" -> SemanticCase(
      256,
      CppBlock(
        """spectra_test_vector<T> a(n);
          |spectra_test_vector<U> b(n);
          |for (int i = 0; i < n; ++i) {
          |  a[i] = T(i % 7);
          |  b[i] = U((i * 3) % 5);
          |}
          |const V init = V(0);
          |V ref = init;
          |for (int i = 0; i < n; ++i)
          |  ref = V(ref + V(a[i]) * V(b[i]));""".stripMargin
      ),
      Map("op_reduce" -> "a + b", "op_product" -> "a * b"),
      CppBlock("const bool ok = near_(__r, ref);")
    ),
    // --- segmented by-key (n=256, 8 segments of 32) ---
    "reduce_by_key" -> SemanticCase(
      256,
      CppBlock(
        """const int nseg = n / 32;
          |spectra_test_vector<K> keys(n), kout(n, K(-1)), kref(nseg);
          |spectra_test_vector<V> vals(n), vout(n, V(-1)), vref(nseg);
          |for (int i = 0; i < n; ++i) {
          |  keys[i] = K(i / 32);
          |  vals[i] = V(i % 5 + 1);
          |}
          |for (int s = 0; s < nseg; ++s) {
          |  kref[s] = K(s);
          |  V acc = V(0);
          |  for (int i = 0; i < 32; ++i)
          |    acc = V(acc + vals[s * 32 + i]);
          |  vref[s] = acc;
          |}""".stripMargin
      ),
      Map("eq" -> "a == b", "op" -> "a + b"),
      CppBlock("const bool ok = vecNear(kout, kref, nseg) && vecNear(vout, vref, nseg);")
    ),
    "inclusive_scan_by_key" -> SemanticCase(
      256,
      CppBlock(
        """spectra_test_vector<K> keys(n);
          |spectra_test_vector<V> vals(n), out(n, V(0)), ref(n);
          |for (int i = 0; i < n; ++i) {
          |  keys[i] = K(i / 32);
          |  vals[i] = V(i % 5 + 1);
          |}
          |{
          |  V acc = vals[0];
          |  ref[0] = acc;
          |  for (int i = 1; i < n; ++i) {
          |    acc = (keys[i] == keys[i - 1]) ? V(acc + vals[i]) : vals[i];
          |    ref[i] = acc;
          |  }
          |}""".stripMargin
      ),
      Map("eq" -> "a == b", "op" -> "a + b"),
      CppBlock("const bool ok = vecNear(out, ref, n);")
    ),
    "exclusive_scan_by_key" -> SemanticCase(
      256,
      CppBlock(
        """spectra_test_vector<K> keys(n);
          |spectra_test_vector<V> vals(n), out(n, V(0)), ref(n);
          |const V init = V(0);
          |for (int i = 0; i < n; ++i) {
          |  keys[i] = K(i / 32);
          |  vals[i] = V(i % 5 + 1);
          |}
          |{
          |  V acc = init;
          |  for (int i = 0; i < n; ++i) {
          |    if (i == 0 || keys[i] != keys[i - 1]) acc = init;
          |    ref[i] = acc;
          |    acc = V(acc + vals[i]);
          |  }
          |}""".stripMargin
      ),
      Map("eq" -> "a == b", "op" -> "a + b"),
      CppBlock("const bool ok = vecNear(out, ref, n);")
    ),
    // --- sort / partition / compaction / set-op / generate mutation family (n=200, perm) ---
    "sort" -> SemanticCase(
      200,
      CppBlock(
        """spectra_test_vector<T> data(n), ref;
          |for (int i = 0; i < n; ++i)
          |  data[i] = T((i * 37 + 11) % n);
          |ref = data;
          |std::sort(ref.begin(), ref.end());""".stripMargin
      ),
      Map("op" -> "a < b"),
      CppBlock("const bool ok = sameSeq(data, ref, n);")
    ),
    "stable_sort" -> SemanticCase(
      200,
      CppBlock(
        """spectra_test_vector<T> data(n), ref;
          |for (int i = 0; i < n; ++i)
          |  data[i] = T((i * 37 + 11) % n);
          |ref = data;
          |std::stable_sort(ref.begin(), ref.end());""".stripMargin
      ),
      Map("op" -> "a < b"),
      CppBlock("const bool ok = sameSeq(data, ref, n);")
    ),
    "sort_by_key" -> SemanticCase(
      200,
      CppBlock(
        """spectra_test_vector<K> keys(n), kref(n);
          |spectra_test_vector<V> values(n), vref(n);
          |for (int i = 0; i < n; ++i) {
          |  const int x = (i * 37 + 11) % n;
          |  keys[i] = K(x);
          |  values[i] = V(x * 2);
          |  kref[i] = K(i);
          |  vref[i] = V(i * 2);
          |}""".stripMargin
      ),
      Map("op" -> "a < b"),
      CppBlock("const bool ok = sameSeq(keys, kref, n) && sameSeq(values, vref, n);")
    ),
    "partition" -> SemanticCase(
      200,
      CppBlock(
        """spectra_test_vector<T> data(n), orig;
          |for (int i = 0; i < n; ++i)
          |  data[i] = T((i * 37 + 11) % n);
          |orig = data;
          |std::sort(orig.begin(), orig.end());""".stripMargin
      ),
      Map("op" -> "x < T(100)"),
      CppBlock(
        """int split = 0;
          |while (split < n && op(data[split]))
          |  ++split;
          |bool ok = true;
          |for (int i = split; i < n; ++i)
          |  if (op(data[i])) ok = false;
          |spectra_test_vector<T> a = data;
          |std::sort(a.begin(), a.end());
          |if (!sameSeq(a, orig, n)) ok = false;""".stripMargin
      )
    ),
    "stable_partition" -> SemanticCase(
      200,
      CppBlock(
        """spectra_test_vector<T> data(n), ref;
          |for (int i = 0; i < n; ++i)
          |  data[i] = T((i * 37 + 11) % n);
          |ref = data;
          |std::stable_partition(ref.begin(), ref.end(), [](T x) { return x < T(100); });""".stripMargin
      ),
      Map("op" -> "x < T(100)"),
      CppBlock("const bool ok = sameSeq(data, ref, n);")
    ),
    "for_each" -> SemanticCase(
      200,
      CppBlock(
        """spectra_test_vector<T> data(n), ref(n);
          |for (int i = 0; i < n; ++i) {
          |  data[i] = T((i * 37 + 11) % n);
          |  ref[i] = T(data[i] * 2);
          |}""".stripMargin
      ),
      Map("op" -> "x = T(x * 2)"),
      CppBlock("const bool ok = sameSeq(data, ref, n);")
    ),
    "for_each_n" -> SemanticCase(
      200,
      CppBlock(
        """spectra_test_vector<T> data(n), ref(n);
          |for (int i = 0; i < n; ++i) {
          |  data[i] = T((i * 37 + 11) % n);
          |  ref[i] = T(data[i] + 1);
          |}""".stripMargin
      ),
      Map("op" -> "x = T(x + 1)"),
      CppBlock("const bool ok = sameSeq(data, ref, n);")
    ),
    "generate" -> SemanticCase(
      200,
      CppBlock("spectra_test_vector<T> out(n, T(-1));"),
      Map("op" -> "7"),
      CppBlock(
        """bool ok = true;
          |for (int i = 0; i < n; ++i)
          |  if (out[i] != T(7)) ok = false;""".stripMargin
      )
    ),
    "generate_n" -> SemanticCase(
      200,
      CppBlock("spectra_test_vector<T> out(n, T(-1));"),
      Map("op" -> "4"),
      CppBlock(
        """bool ok = true;
          |for (int i = 0; i < n; ++i)
          |  if (out[i] != T(4)) ok = false;""".stripMargin
      )
    ),
    "tabulate" -> SemanticCase(
      200,
      CppBlock("spectra_test_vector<T> out(n, T(-1));"),
      Map("op" -> "i * 3"),
      CppBlock(
        """bool ok = true;
          |for (int i = 0; i < n; ++i)
          |  if (out[i] != T(i * 3)) ok = false;""".stripMargin
      )
    ),
    "copy_if" -> SemanticCase(
      200,
      CppBlock(
        """spectra_test_vector<T> in(n), out(n, T(-1)), ref;
          |for (int i = 0; i < n; ++i)
          |  in[i] = T((i * 37 + 11) % n);
          |for (int i = 0; i < n; ++i)
          |  if ((long long)(in[i]) % 2 == 0) ref.push_back(in[i]);""".stripMargin
      ),
      Map("op" -> "(long long)(x) % 2 == 0"),
      CppBlock(
        """const bool ok = __r == (int)ref.size() && __r >= 0 && __r <= n && sameSeq(out, ref, __r);
          |if (!ok) {
          |  std::printf(" result=%d expected=%d out:", __r, (int)ref.size());
          |  for (int i = 0; i < 8 && i < n; ++i)
          |    std::printf(" %lld", (long long)out[i]);
          |  std::printf(" ref:");
          |  for (int i = 0; i < 8 && i < (int)ref.size(); ++i)
          |    std::printf(" %lld", (long long)ref[i]);
          |  std::printf("\n");
          |}""".stripMargin
      )
    ),
    "remove_copy_if" -> SemanticCase(
      200,
      CppBlock(
        """spectra_test_vector<T> in(n), out(n, T(-1)), ref;
          |for (int i = 0; i < n; ++i)
          |  in[i] = T((i * 37 + 11) % n);
          |for (int i = 0; i < n; ++i)
          |  if (!((long long)(in[i]) % 2 == 0)) ref.push_back(in[i]);""".stripMargin
      ),
      Map("op" -> "(long long)(x) % 2 == 0"),
      CppBlock("const bool ok = __r == (int)ref.size() && sameSeq(out, ref, __r);")
    ),
    "remove_if" -> SemanticCase(
      200,
      CppBlock(
        """spectra_test_vector<T> data(n), ref;
          |for (int i = 0; i < n; ++i)
          |  data[i] = T((i * 37 + 11) % n);
          |for (int i = 0; i < n; ++i)
          |  if (!((long long)(data[i]) % 2 == 0)) ref.push_back(data[i]);""".stripMargin
      ),
      Map("op" -> "(long long)(x) % 2 == 0"),
      CppBlock("const bool ok = __r == (int)ref.size() && sameSeq(data, ref, __r);")
    ),
    "unique" -> SemanticCase(
      200,
      CppBlock(
        """spectra_test_vector<T> data(n), ref;
          |for (int i = 0; i < n; ++i)
          |  data[i] = T(i / 4);
          |for (int i = 0; i < n; ++i)
          |  if (ref.empty() || ref.back() != data[i]) ref.push_back(data[i]);""".stripMargin
      ),
      Map("eq" -> "a == b"),
      CppBlock("const bool ok = __r == (int)ref.size() && sameSeq(data, ref, __r);")
    ),
    "unique_copy" -> SemanticCase(
      200,
      CppBlock(
        """spectra_test_vector<T> in(n), out(n, T(-1)), ref;
          |for (int i = 0; i < n; ++i)
          |  in[i] = T(i / 4);
          |for (int i = 0; i < n; ++i)
          |  if (ref.empty() || ref.back() != in[i]) ref.push_back(in[i]);""".stripMargin
      ),
      Map("eq" -> "a == b"),
      CppBlock("const bool ok = __r == (int)ref.size() && sameSeq(out, ref, __r);")
    ),
    "merge" -> SemanticCase(
      200,
      CppBlock(
        """const int na = n / 2, nb = n / 2;
          |spectra_test_vector<T> a(na), b(nb), out(na + nb, T(-1)), ref(na + nb);
          |for (int i = 0; i < na; ++i)
          |  a[i] = T(2 * i);
          |for (int i = 0; i < nb; ++i)
          |  b[i] = T(i);
          |std::merge(a.begin(), a.end(), b.begin(), b.end(), ref.begin());""".stripMargin
      ),
      Map("op" -> "a < b"),
      CppBlock("const bool ok = sameSeq(out, ref, na + nb);")
    ),
    "set_difference" -> SemanticCase(
      200,
      CppBlock(
        """const int na = n / 2, nb = n / 2;
          |spectra_test_vector<T> a(na), b(nb), out(na + nb, T(-1)), ref;
          |for (int i = 0; i < na; ++i)
          |  a[i] = T(2 * i);
          |for (int i = 0; i < nb; ++i)
          |  b[i] = T(i);
          |std::set_difference(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(ref));""".stripMargin
      ),
      Map("op" -> "a < b"),
      CppBlock("const bool ok = __r == (int)ref.size() && sameSeq(out, ref, __r);")
    ),
    "set_intersection" -> SemanticCase(
      200,
      CppBlock(
        """const int na = n / 2, nb = n / 2;
          |spectra_test_vector<T> a(na), b(nb), out(na + nb, T(-1)), ref;
          |for (int i = 0; i < na; ++i)
          |  a[i] = T(2 * i);
          |for (int i = 0; i < nb; ++i)
          |  b[i] = T(i);
          |std::set_intersection(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(ref));""".stripMargin
      ),
      Map("op" -> "a < b"),
      CppBlock("const bool ok = __r == (int)ref.size() && sameSeq(out, ref, __r);")
    ),
    "set_union" -> SemanticCase(
      200,
      CppBlock(
        """const int na = n / 2, nb = n / 2;
          |spectra_test_vector<T> a(na), b(nb), out(na + nb, T(-1)), ref;
          |for (int i = 0; i < na; ++i)
          |  a[i] = T(2 * i);
          |for (int i = 0; i < nb; ++i)
          |  b[i] = T(i);
          |std::set_union(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(ref));""".stripMargin
      ),
      Map("op" -> "a < b"),
      CppBlock("const bool ok = __r == (int)ref.size() && sameSeq(out, ref, __r);")
    ),
    // --- query: predicates / extrema / order (n=300) ---
    "count_if" -> SemanticCase(
      300,
      CppBlock(
        """spectra_test_vector<T> in(n);
          |for (int i = 0; i < n; ++i)
          |  in[i] = T(i % 11);
          |int ref = 0;
          |for (int i = 0; i < n; ++i)
          |  if ((long long)(in[i]) % 2 == 0) ++ref;""".stripMargin
      ),
      Map("op" -> "(long long)(x) % 2 == 0"),
      CppBlock("const bool ok = __r == ref;")
    ),
    "all_of" -> SemanticCase(
      300,
      CppBlock(
        """spectra_test_vector<T> in(n);
          |for (int i = 0; i < n; ++i)
          |  in[i] = T(i % 11);""".stripMargin
      ),
      Map("op" -> "x >= T(0)"),
      CppBlock("const bool ok = (__r == true);")
    ),
    "any_of" -> SemanticCase(
      300,
      CppBlock(
        """spectra_test_vector<T> in(n);
          |for (int i = 0; i < n; ++i)
          |  in[i] = T(i % 11);""".stripMargin
      ),
      Map("op" -> "x == T(5)"),
      CppBlock("const bool ok = (__r == true);")
    ),
    "none_of" -> SemanticCase(
      300,
      CppBlock(
        """spectra_test_vector<T> in(n);
          |for (int i = 0; i < n; ++i)
          |  in[i] = T(i % 11);""".stripMargin
      ),
      Map("op" -> "x > T(100)"),
      CppBlock("const bool ok = (__r == true);")
    ),
    "find_if" -> SemanticCase(
      300,
      CppBlock(
        """spectra_test_vector<T> in(n);
          |for (int i = 0; i < n; ++i)
          |  in[i] = T(i % 11);
          |int ref = n;
          |for (int i = 0; i < n; ++i)
          |  if (in[i] == T(7)) {
          |    ref = i;
          |    break;
          |  }""".stripMargin
      ),
      Map("op" -> "x == T(7)"),
      CppBlock("const bool ok = __r == ref;")
    ),
    "find_if_not" -> SemanticCase(
      300,
      CppBlock(
        """spectra_test_vector<T> in(n);
          |for (int i = 0; i < n; ++i)
          |  in[i] = T(i % 11);
          |int ref = n;
          |for (int i = 0; i < n; ++i)
          |  if (!(in[i] < T(7))) {
          |    ref = i;
          |    break;
          |  }""".stripMargin
      ),
      Map("op" -> "x < T(7)"),
      CppBlock("const bool ok = __r == ref;")
    ),
    "partition_point" -> SemanticCase(
      300,
      CppBlock(
        """spectra_test_vector<T> in(n);
          |for (int i = 0; i < n; ++i)
          |  in[i] = i < 120 ? T(1) : T(0);""".stripMargin
      ),
      Map("op" -> "x != T(0)"),
      CppBlock("const bool ok = __r == 120;")
    ),
    "is_partitioned" -> SemanticCase(
      300,
      CppBlock(
        """spectra_test_vector<T> data(n);
          |for (int i = 0; i < n; ++i)
          |  data[i] = i < 120 ? T(1) : T(0);""".stripMargin
      ),
      Map("op" -> "x != T(0)"),
      CppBlock("const bool ok = (__r == true);")
    ),
    "min_element" -> SemanticCase(
      300,
      CppBlock(
        """spectra_test_vector<T> in(n);
          |for (int i = 0; i < n; ++i)
          |  in[i] = T(i % 11);""".stripMargin
      ),
      Map("op" -> "a < b"),
      CppBlock("const bool ok = near_(__r, n == 0 ? T{} : T(0));")
    ),
    "max_element" -> SemanticCase(
      300,
      CppBlock(
        """spectra_test_vector<T> in(n);
          |for (int i = 0; i < n; ++i)
          |  in[i] = T(i % 11);""".stripMargin
      ),
      Map("op" -> "a < b"),
      CppBlock("const bool ok = near_(__r, n == 0 ? T{} : T(10));")
    ),
    "minmax_element" -> SemanticCase(
      300,
      CppBlock(
        """spectra_test_vector<T> in(n), min_out(1, T(-1)), max_out(1, T(-1));
          |for (int i = 0; i < n; ++i)
          |  in[i] = T(i % 11);""".stripMargin
      ),
      Map("op" -> "a < b"),
      CppBlock("const bool ok = near_(min_out[0], n == 0 ? T{} : T(0)) && near_(max_out[0], n == 0 ? T{} : T(10));")
    ),
    "is_sorted" -> SemanticCase(
      300,
      CppBlock(
        """spectra_test_vector<T> data(n);
          |for (int i = 0; i < n; ++i)
          |  data[i] = T(i);""".stripMargin
      ),
      Map("op" -> "a < b"),
      CppBlock("const bool ok = (__r == true);")
    ),
    "is_sorted_until" -> SemanticCase(
      300,
      CppBlock(
        """spectra_test_vector<T> data(n);
          |for (int i = 0; i < n; ++i)
          |  data[i] = T(i);
          |int ref = n;
          |for (int i = 1; i < n; ++i)
          |  if (data[i] < data[i - 1]) {
          |    ref = i;
          |    break;
          |  }""".stripMargin
      ),
      Map("op" -> "a < b"),
      CppBlock("const bool ok = __r == ref;")
    ),
    // --- query: search / bounds (n=300) ---
    "equal" -> SemanticCase(
      300,
      CppBlock(
        """spectra_test_vector<T> a(n), b(n);
          |for (int i = 0; i < n; ++i) {
          |  a[i] = T(i % 11);
          |  b[i] = T(i % 11);
          |}""".stripMargin
      ),
      Map("op" -> "a == b"),
      CppBlock("const bool ok = (__r == true);")
    ),
    "mismatch" -> SemanticCase(
      300,
      CppBlock(
        """spectra_test_vector<T> a(n), b(n);
          |for (int i = 0; i < n; ++i) {
          |  a[i] = T(i % 11);
          |  b[i] = T(i % 11);
          |}
          |b[50] = T(99);
          |int ref = n;
          |for (int i = 0; i < n; ++i)
          |  if (a[i] != b[i]) {
          |    ref = i;
          |    break;
          |  }""".stripMargin
      ),
      Map("op" -> "a == b"),
      CppBlock(
        """const bool ok = __r == ref;
          |if (!ok)
          |  std::printf(" result=%d expected=%d a0=%lld b0=%lld a50=%lld b50=%lld\n", __r, ref, (long long)a[0], (long long)b[0], (long long)a[50],
          |              (long long)b[50]);""".stripMargin
      )
    ),
    "search" -> SemanticCase(
      300,
      CppBlock(
        """spectra_test_vector<T> in(n);
          |for (int i = 0; i < n; ++i)
          |  in[i] = T(i % 11);
          |const int m = 5;
          |spectra_test_vector<T> sub(in.begin() + 10, in.begin() + 10 + m);
          |int ref = n - m + 1;
          |for (int i = 0; i <= n - m; ++i) {
          |  bool f = true;
          |  for (int k = 0; k < m; ++k)
          |    if (!(in[i + k] == sub[k])) {
          |      f = false;
          |      break;
          |    }
          |  if (f) {
          |    ref = i;
          |    break;
          |  }
          |}""".stripMargin
      ),
      Map("op" -> "a == b"),
      CppBlock("const bool ok = __r == ref;")
    ),
    "search_n" -> SemanticCase(
      300,
      CppBlock(
        """spectra_test_vector<T> in(n);
          |for (int i = 0; i < n; ++i)
          |  in[i] = T(i % 5);
          |in[40] = in[41] = in[42] = T(9);
          |const int count = 3;
          |const T value = T(9);
          |int ref = n - count + 1;
          |for (int i = 0; i <= n - count; ++i) {
          |  bool f = true;
          |  for (int k = 0; k < count; ++k)
          |    if (!(in[i + k] == value)) {
          |      f = false;
          |      break;
          |    }
          |  if (f) {
          |    ref = i;
          |    break;
          |  }
          |}""".stripMargin
      ),
      Map("op" -> "a == b"),
      CppBlock("const bool ok = __r == ref;")
    ),
    "includes" -> SemanticCase(
      300,
      CppBlock(
        """const int na = n, nb = n / 2;
          |spectra_test_vector<T> a(na), b(nb);
          |for (int i = 0; i < na; ++i)
          |  a[i] = T(i);
          |for (int j = 0; j < nb; ++j)
          |  b[j] = T(2 * j);""".stripMargin
      ),
      Map("op" -> "a < b"),
      CppBlock("const bool ok = (__r == true);")
    ),
    "lower_bound" -> SemanticCase(
      300,
      CppBlock(
        """spectra_test_vector<T> in(n);
          |for (int i = 0; i < n; ++i)
          |  in[i] = T(i);
          |const T value = T(137);""".stripMargin
      ),
      Map("op" -> "a < b"),
      CppBlock("const bool ok = __r == 137;")
    ),
    "upper_bound" -> SemanticCase(
      300,
      CppBlock(
        """spectra_test_vector<T> in(n);
          |for (int i = 0; i < n; ++i)
          |  in[i] = T(i);
          |const T value = T(137);""".stripMargin
      ),
      Map("op" -> "a < b"),
      CppBlock("const bool ok = __r == 138;")
    ),
    "binary_search" -> SemanticCase(
      300,
      CppBlock(
        """spectra_test_vector<T> in(n);
          |for (int i = 0; i < n; ++i)
          |  in[i] = T(i);
          |const T value = T(137);""".stripMargin
      ),
      Map("op" -> "a < b"),
      CppBlock("const bool ok = (__r == true);")
    ),
    // --- movement + concrete value ops (n=256, in[i]=(i*17+3)%97) ---
    "copy" -> SemanticCase(
      256,
      CppBlock(
        """spectra_test_vector<T> in(n), out(n, T(-1));
          |for (int i = 0; i < n; ++i)
          |  in[i] = T((i * 17 + 3) % 97);""".stripMargin
      ),
      Map(),
      CppBlock("const bool ok = sameSeq(out, in, n);")
    ),
    "copy_n" -> SemanticCase(
      256,
      CppBlock(
        """spectra_test_vector<T> in(n), out(n, T(-1));
          |for (int i = 0; i < n; ++i)
          |  in[i] = T((i * 17 + 3) % 97);""".stripMargin
      ),
      Map(),
      CppBlock("const bool ok = sameSeq(out, in, n);")
    ),
    "reverse" -> SemanticCase(
      256,
      CppBlock(
        """spectra_test_vector<T> data(n), ref(n);
          |for (int i = 0; i < n; ++i)
          |  data[i] = T((i * 17 + 3) % 97);
          |for (int i = 0; i < n; ++i)
          |  ref[i] = data[n - 1 - i];""".stripMargin
      ),
      Map(),
      CppBlock("const bool ok = sameSeq(data, ref, n);")
    ),
    "reverse_copy" -> SemanticCase(
      256,
      CppBlock(
        """spectra_test_vector<T> in(n), out(n, T(-1)), ref(n);
          |for (int i = 0; i < n; ++i)
          |  in[i] = T((i * 17 + 3) % 97);
          |for (int i = 0; i < n; ++i)
          |  ref[i] = in[n - 1 - i];""".stripMargin
      ),
      Map(),
      CppBlock("const bool ok = sameSeq(out, ref, n);")
    ),
    "fill" -> SemanticCase(
      256,
      CppBlock(
        """spectra_test_vector<T> out(n, T(-1));
          |const T v = T(9);""".stripMargin
      ),
      Map(),
      CppBlock(
        """bool ok = true;
          |for (int i = 0; i < n; ++i)
          |  if (out[i] != T(9)) ok = false;""".stripMargin
      )
    ),
    "fill_n" -> SemanticCase(
      256,
      CppBlock(
        """spectra_test_vector<T> out(n, T(-1));
          |const T v = T(5);""".stripMargin
      ),
      Map(),
      CppBlock(
        """bool ok = true;
          |for (int i = 0; i < n; ++i)
          |  if (out[i] != T(5)) ok = false;""".stripMargin
      )
    ),
    "swap_ranges" -> SemanticCase(
      256,
      CppBlock(
        """spectra_test_vector<T> a(n), b(n), ra(n), rb(n);
          |for (int i = 0; i < n; ++i) {
          |  a[i] = T((i * 17 + 3) % 97);
          |  b[i] = T(i);
          |}
          |ra = b;
          |rb = a;""".stripMargin
      ),
      Map(),
      CppBlock("const bool ok = sameSeq(a, ra, n) && sameSeq(b, rb, n);")
    ),
    "gather" -> SemanticCase(
      256,
      CppBlock(
        """const int in_n = n;
          |spectra_test_vector<int> map(n);
          |spectra_test_vector<T> in(in_n), out(n, T(-1)), ref(n);
          |for (int i = 0; i < n; ++i) {
          |  in[i] = T((i * 17 + 3) % 97);
          |  map[i] = n - 1 - i;
          |}
          |for (int i = 0; i < n; ++i)
          |  ref[i] = in[map[i]];""".stripMargin
      ),
      Map(),
      CppBlock("const bool ok = sameSeq(out, ref, n);")
    ),
    "scatter" -> SemanticCase(
      256,
      CppBlock(
        """const int out_n = n;
          |spectra_test_vector<int> map(n);
          |spectra_test_vector<T> in(n), out(out_n, T(-1)), ref(out_n, T(-1));
          |for (int i = 0; i < n; ++i) {
          |  in[i] = T((i * 17 + 3) % 97);
          |  map[i] = n - 1 - i;
          |}
          |for (int i = 0; i < n; ++i)
          |  ref[map[i]] = in[i];""".stripMargin
      ),
      Map(),
      CppBlock("const bool ok = sameSeq(out, ref, n);")
    ),
    "count" -> SemanticCase(
      256,
      CppBlock(
        """spectra_test_vector<T> in(n);
          |for (int i = 0; i < n; ++i)
          |  in[i] = T((i * 17 + 3) % 97);
          |const T value = T(3);
          |int ref = 0;
          |for (int i = 0; i < n; ++i)
          |  if (in[i] == value) ++ref;""".stripMargin
      ),
      Map("eq" -> "a == b"),
      CppBlock("const bool ok = __r == ref;")
    ),
    "find" -> SemanticCase(
      256,
      CppBlock(
        """spectra_test_vector<T> in(n);
          |for (int i = 0; i < n; ++i)
          |  in[i] = T(i + 1 < n ? 1 : 7);
          |const T value = T(7);
          |int ref = n;
          |for (int i = 0; i < n; ++i)
          |  if (in[i] == value) {
          |    ref = i;
          |    break;
          |  }""".stripMargin
      ),
      Map("eq" -> "a == b"),
      CppBlock("const bool ok = __r == ref;")
    ),
    "remove" -> SemanticCase(
      256,
      CppBlock(
        """spectra_test_vector<T> data(n), ref;
          |for (int i = 0; i < n; ++i)
          |  data[i] = T((i * 17 + 3) % 97);
          |const T value = T(2);
          |for (int i = 0; i < n; ++i)
          |  if (!(data[i] == value)) ref.push_back(data[i]);""".stripMargin
      ),
      Map("eq" -> "a == b"),
      CppBlock("const bool ok = __r == (int)ref.size() && sameSeq(data, ref, __r);")
    ),
    "remove_copy" -> SemanticCase(
      256,
      CppBlock(
        """spectra_test_vector<T> in(n), out(n, T(-1)), ref;
          |for (int i = 0; i < n; ++i)
          |  in[i] = T((i * 17 + 3) % 97);
          |const T value = T(4);
          |for (int i = 0; i < n; ++i)
          |  if (!(in[i] == value)) ref.push_back(in[i]);""".stripMargin
      ),
      Map("eq" -> "a == b"),
      CppBlock("const bool ok = __r == (int)ref.size() && sameSeq(out, ref, __r);")
    ),
    "replace" -> SemanticCase(
      256,
      CppBlock(
        """spectra_test_vector<T> io(n), ref(n);
          |for (int i = 0; i < n; ++i)
          |  io[i] = T((i * 17 + 3) % 97);
          |const T oldv = T(5), newv = T(88);
          |for (int i = 0; i < n; ++i)
          |  ref[i] = (io[i] == oldv) ? newv : io[i];""".stripMargin
      ),
      Map("eq" -> "a == b"),
      CppBlock("const bool ok = sameSeq(io, ref, n);")
    ),
    "replace_copy" -> SemanticCase(
      256,
      CppBlock(
        """spectra_test_vector<T> in(n), out(n, T(0)), ref(n);
          |for (int i = 0; i < n; ++i)
          |  in[i] = T((i * 17 + 3) % 97);
          |const T oldv = T(6), newv = T(42);
          |for (int i = 0; i < n; ++i)
          |  ref[i] = (in[i] == oldv) ? newv : in[i];""".stripMargin
      ),
      Map("eq" -> "a == b"),
      CppBlock("const bool ok = sameSeq(out, ref, n);")
    ),
    "replace_if" -> SemanticCase(
      200,
      CppBlock(
        """spectra_test_vector<T> data(n), ref(n);
          |for (int i = 0; i < n; ++i)
          |  data[i] = T((i * 37 + 11) % n);
          |const T new_value = T(-1);
          |for (int i = 0; i < n; ++i)
          |  ref[i] = (data[i] >= T(100)) ? new_value : data[i];""".stripMargin
      ),
      Map("op" -> "x >= T(100)"),
      CppBlock("const bool ok = sameSeq(data, ref, n);")
    ),
    "replace_copy_if" -> SemanticCase(
      200,
      CppBlock(
        """spectra_test_vector<T> in(n), out(n, T(0)), ref(n);
          |for (int i = 0; i < n; ++i)
          |  in[i] = T((i * 37 + 11) % n);
          |const T new_value = T(-1);
          |for (int i = 0; i < n; ++i)
          |  ref[i] = (in[i] >= T(100)) ? new_value : in[i];""".stripMargin
      ),
      Map("op" -> "x >= T(100)"),
      CppBlock("const bool ok = sameSeq(out, ref, n);")
    ),
    "sequence" -> SemanticCase(
      256,
      CppBlock(
        """spectra_test_vector<T> out(n, T(-1)), ref(n);
          |const T init = T(0), step = T(1);
          |for (int i = 0; i < n; ++i)
          |  ref[i] = T(init + T(i) * step);""".stripMargin
      ),
      Map("value_at" -> "a + T(i) * b"),
      CppBlock("const bool ok = sameSeq(out, ref, n);")
    ),
    "adjacent_difference" -> SemanticCase(
      256,
      CppBlock(
        """spectra_test_vector<T> in(n), out(n, T(0)), ref(n);
          |for (int i = 0; i < n; ++i)
          |  in[i] = T((i * 17 + 3) % 97);
          |ref[0] = in[0];
          |for (int i = 1; i < n; ++i)
          |  ref[i] = T(in[i] - in[i - 1]);""".stripMargin
      ),
      Map("op" -> "a - b"),
      CppBlock("const bool ok = sameSeq(out, ref, n);")
    )
  )

  def textOutputs(root: Path): List[(Path, String)] = {
    val expected = declarations.map(_.name.last).toSet
    val missing  = expected -- oracles.keySet
    val extra    = oracles.keySet -- expected
    require(missing.isEmpty, s"Spectra semantic test cases missing: ${missing.toList.sorted.mkString(", ")}")
    require(extra.isEmpty, s"Spectra semantic test cases have no declaration: ${extra.toList.sorted.mkString(", ")}")
    require(
      sizeSweptAlgorithms.subsetOf(expected),
      s"Spectra test size profiles have no declaration: ${(sizeSweptAlgorithms -- expected).toList.sorted.mkString(", ")}"
    )
    declarations.foreach { decl =>
      val name = decl.name.last
      val expectedFunctors = decl.args.collect { case p.Arg(p.Named(symbol, _: p.Type.Exec, _), _, _) =>
        symbol
      }.toSet
      val actualFunctors = oracles(name).functors.keySet
      require(
        actualFunctors == expectedFunctors,
        s"Spectra callable test cases for $name: expected ${expectedFunctors.toList.sorted.mkString(", ")}, " +
          s"found ${actualFunctors.toList.sorted.mkString(", ")}"
      )
    }

    val testRoot = root.resolve("test/generated")
    List(
      root.resolve("test/scala/polyregion/check_surface.scala") -> scalaSurface,
      root.resolve("test/fortran/check_surface.f90")            -> fortranSurface,
      testRoot.resolve("support.hpp")                           -> cppTestHeader
    ) ++ declarations.map(decl => testRoot.resolve(s"${decl.name.last}.cpp") -> cppTestFile(decl))
  }
}
