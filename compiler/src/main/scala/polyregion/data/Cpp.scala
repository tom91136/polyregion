package polyregion.data

import polyregion.data.Cpp.ToCppType

import scala.deriving.*
import scala.compiletime.{constValue, erasedValue, error, summonInline}
import scala.collection.immutable.LazyList.cons
import polyregion.data.Cpp.StructSource
import java.nio.file.Paths
import java.nio.file.Files
import java.nio.file.StandardOpenOption
import scala.annotation.tailrec
import fansi.Str
import cats.syntax.all._

object Cpp {

  extension [A](xs: Seq[A]) {
    def csv: String = xs.mkString(", ")
    def ssv: String = xs.mkString(" ")
    def sym(prefix: String = "", delim: String = "", suffix: String): String =
      if (xs.isEmpty) "" else xs.mkString(prefix, delim, suffix)
  }

  case class CppType(
      name: String,
      movable: Boolean = false,
      constexpr: Boolean = true,
      sum: Boolean = false,
      streamOp: (String, String) => List[String] = (s, v) => List(s"$s << $v;"),
      include: List[String] = Nil
  ) {
    def declaredName = if (sum) s"${name}Base" else name
  }

  enum CppAttrs(val repr: String) {
    case Constexpr extends CppAttrs("constexpr")
    case Explicit extends CppAttrs("explicit")
    case Const extends CppAttrs("const")
    def cond(p: Boolean) = if (p) Some(this) else None
  }

  case class StructSource(
      name: String,
      parent: Option[String],
      stmts: List[String],
      implStmts: List[String],
      includes: List[String],
      forwardDeclStmts: List[String]
  )
  object StructSource {
    def emitHeader(namespace: String, xs: List[StructSource]) = {
      val includes =
        ("memory" :: "variant" :: "iterator" :: xs.flatMap(_.includes)).distinct.sorted
          .map(incl => s"#include <$incl>")
          .mkString("\n")

      val clsDefs = xs.flatMap { c =>
        "" ::
          s"struct ${c.parent.fold(c.name)(s"${c.name} : " + _)} {" //
          :: c.stmts.map("  " + _) :::                              //
          "};" ::                                                   //
          c.forwardDeclStmts                                        //
      }

      s"""|#pragma once
          |
          |$includes
          |
          |namespace $namespace {
          |${clsDefs.sym("\n", "\n", "\n")}
          |} // namespace $namespace
          | 
          |""".stripMargin
    }
    def emitImpl(namespace: String, headerName: String, xs: List[StructSource]) =
      s"""|#include "$headerName.h"
          |
          |namespace $namespace {
          |
          |${xs.map(_.implStmts.mkString("\n")).mkString("\n\n")}
          |
          |} // namespace $namespace
          |""".stripMargin
  }

  case class StructNode(
      tpe: CppType,
      members: List[(String, CppType)],
      parent: Option[(StructNode, List[String])],
      variants: List[StructNode] = Nil
  ) {
    def name = tpe.declaredName
    def emit: List[StructSource] = {
      val ctorInit = members.map { (n, tpe) =>
        if (tpe.sum) s"$n(std::make_shared<${tpe.name}>($n))"
        else if (tpe.movable) s"$n(std::move($n))"
        else s"$n($n)"
      }
      val (classDecl, ctorArgs, ctorChain) = parent match {
        case None => (tpe.name, members, ctorInit)
        case Some((base, Nil)) =>
          (
            s"${name} : ${base.name}",
            members ::: base.members,
            s"${base.name}(${base.members.map((n, _) => n).csv})" :: ctorInit
          )
        case Some((base, xs)) =>
          (
            s"${name} : ${base.name}",
            members,
            s"${base.name}(${xs.csv})" :: ctorInit
          )
      }

      import CppAttrs.*
      //TODO re-enable constexpr later
      val ctorAttrs = List(Constexpr.cond(tpe.constexpr && false), Explicit.cond(ctorArgs.sizeIs == 1)).flatten

      val ctorChainExpr = ctorChain match {
        case Nil => " = default;"
        case xs  => s" noexcept : ${xs.csv} {}"
      }
      val ctorArgExpr  = ctorArgs.map((n, t) => if (t.sum) s"const ${t.name} &$n" else s"${t.name} $n").csv
      val ctorAttrExpr = ctorAttrs.map(_.repr).sym("", " ", " ")

      val memberStmts = members.map { (n, t) =>
        if (t.sum) s"std::shared_ptr<${t.name}> $n;" else s"${t.name} $n;"
      }
      val mainCtorStmts = s"$ctorAttrExpr$name($ctorArgExpr)$ctorChainExpr" :: Nil
      val (singletonClsStmts, singletonClsImplStmts) = members match {
        case Nil =>
          (Nil, Nil)

        // (
        //   List(
        //     s"$name(const $name &) = delete;",
        //     s"$name &operator=(const $name &) = delete;",
        //     s"static constexpr $name &val() { return _; }",
        //     s"private:",
        //     s"static $name _;"
        //   ),
        //   List(s"$name $name::_ = $name();")
        // )
        case _ => (Nil, Nil)
      }

      val streamMethodProto = s"std::ostream &operator<<(std::ostream &os, const ${name} &x)"
      val streamStmts =
        // parent.map((c, _) => ("os << \"base=\";" :: c.tpe.streamOp("os", "x"))).toList :::
        members.map((n, tpe) =>
          if (tpe.sum) s"std::visit([&os](auto &&arg) { os << arg; }, *x.$n);" :: Nil
          else tpe.streamOp("os", s"x.$n")
        )
      val streamMethodStmts = s"$streamMethodProto {" ::
        s"  os << \"${name}(\";" ::
        streamStmts.intercalate("os << ',';" :: Nil).map("  " + _) :::
        "  os << ')';" ::
        "  return os;" ::
        "}" :: Nil

      def foldVariants(xs: List[StructNode]): List[String] =
        xs.flatMap(s => s.name :: foldVariants(s.variants))

      val variantStmt =
        if (tpe.sum) {
          val allVariants = foldVariants(variants)
          allVariants
            .map(v => s"struct $v;") :+ s"using ${tpe.name} = std::variant<${allVariants.csv}>;"
        } else Nil

      val stmts     = memberStmts ::: mainCtorStmts ::: singletonClsStmts ::: s"friend $streamMethodProto;" :: Nil
      val implStmts = singletonClsImplStmts ::: streamMethodStmts
      val includes  = members.flatMap(_._2.include)

      StructSource(name, parent.map(_._1.name), stmts, implStmts, includes, variantStmt) :: variants.flatMap(_.emit)
    }
  }

  trait ToCppTerm[A] extends (A => String)
  object ToCppTerm {
    given ToCppTerm[String]       = x => s"\"$x\""
    given ToCppTerm[List[String]] = x => x.mkString(".")
    given ToCppTerm[Int]          = x => s"$x"
    inline given derived[T](using m: Mirror.Of[T]): ToCppTerm[T] = (x: T) =>
      inline m match
        case s: Mirror.SumOf[T] =>
          s"${constValue[s.MirroredLabel]}() /*sum*/"
        case p: Mirror.ProductOf[T] =>
          // Leaf
          s"${constValue[p.MirroredLabel]}() /*prod*/"
        case x => error(s"Unhandled derive: $x")
  }

  trait ToCppType[A] extends (() => CppType)
  object ToCppType {
    given ToCppType[Int]     = () => CppType("int32_t", constexpr = true, include = List("cstdint"))
    given ToCppType[Long]    = () => CppType("int64_t", constexpr = true, include = List("cstdint"))
    given ToCppType[Float]   = () => CppType("float", constexpr = true, include = Nil)
    given ToCppType[Double]  = () => CppType("double", constexpr = true, include = Nil)
    given ToCppType[Short]   = () => CppType("int16_t", constexpr = true, include = List("cstdint"))
    given ToCppType[Char]    = () => CppType("uint16_t", constexpr = true, include = List("cstdint"))
    given ToCppType[Byte]    = () => CppType("int8_t", constexpr = true, include = List("cstdint"))
    given ToCppType[Boolean] = () => CppType("bool", constexpr = true, include = Nil)
    given ToCppType[String] = () =>
      CppType(
        "std::string",
        movable = true,
        constexpr = false,
        sum = false,
        streamOp = (s, v) => s"""$s << '"' << $v << '"';""" :: Nil,
        include = List("string")
      )
    given [A: ToCppType]: ToCppType[List[A]] = { () =>
      val tpe = summon[ToCppType[A]]()
      CppType(
        s"std::vector<${tpe.name}>",
        movable = true,
        constexpr = false,
        sum = false,
        streamOp = { (s, v) =>
          List(
            s"$s << '{';",
            s"if (!$v.empty()) {",
            s"  std::for_each($v.begin(), std::prev($v.end()), [&$s](auto &&x) { ${tpe.streamOp(s, "x").mkString(";")} $s << ','; });",
            s"  ${tpe.streamOp(s, s"$v.back()").mkString(";")}",
            s"}",
            s"$s << '}';"
          )
        },
        include = List("vector")
      )
    }

    // inline def forAll[T <: Tuple](p: CppType => Boolean): Boolean =
    //   inline erasedValue[T] match
    //     case _: EmptyTuple => true
    //     case _: (t *: ts)  => p(summonInline[ToCppType[t]]()) && forAll[ts](p)

    inline given derived[T](using m: Mirror.Of[T]): ToCppType[T] = () =>
      inline m match
        case s: Mirror.SumOf[T] =>
          CppType(
            constValue[s.MirroredLabel],
            movable = true,
            constexpr = false, // forAll[m.MirroredElemTypes](_.constexpr),
            streamOp = (o, v) => List(s"$o << static_cast<const ${constValue[s.MirroredLabel]}Base &>($v);"),
            sum = true
          )
        case p: Mirror.ProductOf[T] =>
          CppType(
            constValue[p.MirroredLabel],
            movable = true,
            constexpr = false, // forAll[p.MirroredElemTypes](_.constexpr),
            streamOp = (o, v) => List(s"$o << static_cast<const ${constValue[p.MirroredLabel]} &>($v);"),
            sum = false
          )
        case x => error(s"Unhandled derive: $x")
  }

  inline def deriveSum[N <: Tuple, T <: Tuple](parent: Option[(StructNode, List[String])] = None): List[StructNode] =
    inline (erasedValue[N], erasedValue[T]) match
      case (_: EmptyTuple, _: EmptyTuple) => Nil
      case (_: (n *: ns), _: (t *: ts)) =>
        deriveStruct[t](parent)(using
          summonInline[ToCppType[t]],
          summonInline[ToCppTerm[t]],
          summonInline[Mirror.Of[t]]
        ) :: deriveSum[ns, ts](parent)

  inline def deriveProduct[L <: Tuple, T <: Tuple]: List[(String, CppType)] =
    inline (erasedValue[L], erasedValue[T]) match
      case (_: EmptyTuple, _: EmptyTuple) => Nil
      case (_: (l *: ls), _: (t *: ts)) => (s"${constValue[l]}", summonInline[ToCppType[t]]()) :: deriveProduct[ls, ts]

  inline def deriveStruct[T: ToCppType: ToCppTerm](parent: Option[(StructNode, List[String])] = None)(using
      m: Mirror.Of[T]
  ): StructNode =
    inline m match
      case s: Mirror.SumOf[T] =>
        val ctorTerms = compileTime.primaryCtorApplyTerms[s.MirroredType, String, ToCppTerm]
        val members   = compileTime.sumTypeCtorParams[s.MirroredType, CppType, ToCppType]
        val sum = StructNode(
          summon[ToCppType[s.MirroredType]](),
          members,
          parent.map((s, _) => (s, ctorTerms))
        )
        sum.copy(variants = deriveSum[s.MirroredElemLabels, s.MirroredElemTypes](Some((sum, ctorTerms))))
      case p: Mirror.ProductOf[T] =>
        val members = deriveProduct[p.MirroredElemLabels, p.MirroredElemTypes]
        val terms   = compileTime.primaryCtorApplyTerms[p.MirroredType, String, ToCppTerm]
        StructNode(
          summon[ToCppType[p.MirroredType]](),
          members,
          parent.map((s, _) => (s, terms))
        )
      case x => error(s"Unhandled derive: $x")

}

object Foo {

  enum Alt(x: Int) {
    case A extends Alt(1)
    case B extends Alt(2)
  }
  case class Foo(s: String, xs: List[List[String]])

  sealed abstract class FirstTop(val u: Int, val xx: Alt) derives Cpp.ToCppType

  enum T1Mid(val foo: Int) extends FirstTop(foo, Alt.B) {
    case T1ALeaf(xs: List[Int], z: List[String], y: Int, first: FirstTop) extends T1Mid(42)
    case T1BLeaf extends T1Mid(11)
  }
  enum T2Mid(val x: String, xx: Alt, z: Int) extends FirstTop(z, xx) {
    case T2ALeaf extends T2Mid("z", Alt.B, 1)
    case T2BLeaf(that: String, i: Int) extends T2Mid("myconst", Alt.B, 2)
  }

  // enum Opt[+T] {
  //   case Sm(t: T, u: T)
  //   case Nn
  // }

  // enum Base(val u: Int) {
  //   case This(x: Int, s: String) extends Base(x)
  //   case That(a: String) extends Base(42)
  // }

  @main def main2(): Unit =
//    println(Opt.Sm(2, 7) == Opt.Sm(1 + 1, 7))
//    import Form.*
//    val x = summon[Form[Opt[Int]]]
    // compileTime.show[ToCppType[Int]]
    // compileTime.nonCaseFieldParams[First]
    // compileTime.nonCaseFieldParams[T1] // enum T1 extends First("") {
    // compileTime.nonCaseFieldParams[T2, ToCppType]
    // compileTime.nonCaseFieldParams[First, ToCppType]
    // val xs = compileTime.nonCaseFieldParams[T2, ToCppType]
    // xs.foreach { z =>
    // println(">>" + z._1 + " : " + z._2())
    // }

    //  enum T2(val x: String) extends First(x)  {
    //    case T2B extends T2("z")

    // compileTime.primaryCtorApplyTerms[T2.T2B, String, Cpp.ToCppTerm]
    // compileTime.primaryCtorApplyTerms[T1, String, Cpp.ToCppTerm]

    println("\n=========\n")

  implicit val n: Cpp.ToCppType[polyregion.PolyAstUnused.Type]    = Cpp.ToCppType.derived
  implicit val nn: Cpp.ToCppType[polyregion.PolyAstUnused.Kind]   = Cpp.ToCppType.derived
  implicit val nnn: Cpp.ToCppType[polyregion.PolyAstUnused.Named] = Cpp.ToCppType.derived
  implicit val nnnn: Cpp.ToCppType[polyregion.PolyAstUnused.Ref]  = Cpp.ToCppType.derived

  implicit val _n: Cpp.ToCppTerm[polyregion.PolyAstUnused.Type]    = Cpp.ToCppTerm.derived
  implicit val _nn: Cpp.ToCppTerm[polyregion.PolyAstUnused.Kind]   = Cpp.ToCppTerm.derived
  implicit val _nnn: Cpp.ToCppTerm[polyregion.PolyAstUnused.Named] = Cpp.ToCppTerm.derived
  implicit val _nnnn: Cpp.ToCppTerm[polyregion.PolyAstUnused.Ref]  = Cpp.ToCppTerm.derived
//

  val alts = Cpp.deriveStruct[polyregion.PolyAstUnused.Sym]().emit
    ::: Cpp.deriveStruct[polyregion.PolyAstUnused.Kind]().emit
    ::: Cpp.deriveStruct[polyregion.PolyAstUnused.Type]().emit
    ::: Cpp.deriveStruct[polyregion.PolyAstUnused.Named]().emit
    ::: Cpp.deriveStruct[polyregion.PolyAstUnused.Position]().emit
    ::: Cpp.deriveStruct[polyregion.PolyAstUnused.Ref]().emit
    ::: Cpp.deriveStruct[polyregion.PolyAstUnused.U]().emit
  // ::: Cpp.deriveStruct[polyregion.PolyAstUnused.Intr]().emit
  // ::: Cpp.deriveStruct[polyregion.PolyAstUnused.Tree]().emit
  // ::: Cpp.deriveStruct[polyregion.PolyAstUnused.Ref]().emit
// //                 ::: Cpp.deriveStruct[polyregion.PolyAstUnused.Tree]().emit
//   val header = StructSource.emitHeader("foo", alts)
//   println(header)
//   println("\n=========\n")
//   val impl = StructSource.emitImpl("foo", "foo", alts)
//   println(impl)
//   println("\n=========\n")

//   Files.writeString(
//     Paths.get("/home/tom/polyregion/native/src/foo.cpp"),
//     impl,
//     StandardOpenOption.TRUNCATE_EXISTING,
//     StandardOpenOption.CREATE,
//     StandardOpenOption.WRITE
//   )
//   Files.writeString(
//     Paths.get("/home/tom/polyregion/native/src/foo.h"),
//     header,
//     StandardOpenOption.TRUNCATE_EXISTING,
//     StandardOpenOption.CREATE,
//     StandardOpenOption.WRITE
//   )
//    import Cpp.*
//    println(T1Mid.T1ALeaf(Nil, List("a", "b"), 23, T1Mid.T1BLeaf))

  // println(Cpp.deriveStruct[Alt]().map(_.emitSource).mkString("\n"))
  // println(Cpp.deriveStruct[FirstTop]().map(_.emitSource).mkString("\n"))
  // println(Cpp.deriveStruct[First]().map(_.emitSource).mkString("\n"))
//    println(Cpp.deriveStruct[Foo]().map(_.emitSource).mkString("\n"))
  ()
}
