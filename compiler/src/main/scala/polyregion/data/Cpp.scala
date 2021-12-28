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

  import collection.immutable.ListSet
  import collection.mutable.{Builder, LinkedHashMap => MMap}

  extension [A](t: Traversable[A]) {

    def groupByOrderedUnique[K](f: A => K): Map[K, ListSet[A]] =
      groupByGen(ListSet.newBuilder[A])(f)

    def groupByOrdered[K](f: A => K): Map[K, List[A]] =
      groupByGen(List.newBuilder[A])(f)

    def groupByGen[K, C[_]](makeBuilder: => Builder[A, C[A]])(f: A => K): Map[K, C[A]] = {
      val map = MMap[K, Builder[A, C[A]]]()
      for (i <- t) {
        val key = f(i)
        val builder = map.get(key) match {
          case Some(existing) => existing
          case None =>
            val newBuilder = makeBuilder
            map(key) = newBuilder
            newBuilder
        }
        builder += i
      }
      map.view.mapValues(_.result).toMap
    }
  }

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
    def declaredName = if (sum) s"Base" else name
  }

  enum CppAttrs(val repr: String) {
    case Constexpr extends CppAttrs("constexpr")
    case Explicit extends CppAttrs("explicit")
    case Const extends CppAttrs("const")
    def cond(p: Boolean) = if (p) Some(this) else None
  }

  case class StructSource(
      namespaces: List[String],
      name: String,
      parent: Option[String],
      stmts: List[String],
      implStmts: List[String],
      includes: List[String],
      forwardDeclStmts: List[String]
  )
  object StructSource {

    val RequiredIncludes = List("memory", "variant", "iterator")
    def emitHeader(namespace: String, xs: List[StructSource]) = {

      // def collect[A](xs: List[StructSource])(f: StructSource => A): List[A] =
      //   xs.flatMap(s => f(s) :: collect(s.nested)(f))

      // val u = collect(xs)(_.includes).flatten

      val includes = (RequiredIncludes ::: xs.flatMap(_.includes)) //
        .distinct.sorted
        .map(incl => s"#include <$incl>")
        .mkString("\n")

      // def emitCls(xs: List[StructSource]): List[String] =
      //   xs.flatMap { c =>
      //     "" ::
      //       s"struct ${c.parent.fold(c.name)(s"${c.name} : " + _)} {"        //
      //       :: c.stmts.map("  " + _) ::: emitCls(c.nested).map("  " + _) ::: //
      //       "};" ::                                                          //
      //       c.forwardDeclStmts                                               //
      //   }                                                                    //

      val nscs = xs.map(x => x.namespaces -> List(x)).toList.flatMap { (ns, clss) =>
        val clsDefs = clss.flatMap { c =>
          "" ::
            s"struct ${c.parent.fold(c.name)(s"${c.name} : " + _)} {" //
            :: c.stmts.map("  " + _) :::                              //
            "};" ::                                                   //
            c.forwardDeclStmts                                        //
        }
      s"namespace ${ns.mkString("::")} { " ::
        clsDefs.map("  " + _) :::
        "}" :: Nil
      }
      // val clsDefs = xs.flatMap { c =>
      //   "" ::
      //     s"struct ${c.parent.fold(c.name)(s"${c.name} : " + _)} {" //
      //     :: c.stmts.map("  " + _) :::                              //
      //     "};" ::                                                   //
      //     c.forwardDeclStmts                                        //
      // }
      // val clsDefs = emitCls(xs)

      s"""|#pragma once
          |
          |$includes
          |
          |namespace $namespace {
          |${nscs.sym("\n", "\n", "\n")}
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
    def traceUp(p: Option[(StructNode, List[String])]): List[StructNode] =
      p match {
        case Some((x, _)) => x :: traceUp(x.parent)
        case None         => Nil
      }
    lazy val namespace = traceUp(parent).map(_.tpe.name).reverse
    def emit: List[StructSource] = {

      val ctorInit = members.map { (n, tpe) =>
        if (tpe.sum) s"$n(std::make_shared<${tpe.name}::Any>($n))"
        else if (tpe.movable) s"$n(std::move($n))"
        else s"$n($n)"
      }
      val (ctorArgs, ctorChain) = parent match {
        case None => (members, ctorInit)
        case Some((base, Nil)) =>
          (
            members ::: base.members,
            s"${base.name}(${base.members.map((n, _) => n).csv})" :: ctorInit
          )
        case Some((base, xs)) =>
          (
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
      val ctorArgExpr  = ctorArgs.map((n, t) => if (t.sum) s"const ${t.name}::Any &$n" else s"${t.name} $n").csv
      val ctorAttrExpr = ctorAttrs.map(_.repr).sym("", " ", " ")

      val memberStmts = members.map { (n, t) =>
        if (t.sum) s"std::shared_ptr<${t.name}::Any> $n;" else s"${t.name} $n;"
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
      val nss = if (tpe.sum) namespace :+ tpe.name else namespace

      def ns(name : String) = {
        nss.sym("", "::", "::") + name
      }

      val streamMethodProto = s"std::ostream &operator<<(std::ostream &os, const ${ns(name)} &x)"
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
          allVariants.map(v => s"struct $v;") :+ s"using Any = std::variant<${allVariants.csv}>;"
        } else Nil

      val stmts     = memberStmts ::: mainCtorStmts ::: singletonClsStmts ::: s"friend $streamMethodProto;" :: Nil
      val implStmts = singletonClsImplStmts ::: streamMethodStmts
      val includes  = members.flatMap(_._2.include)


      val struct = StructSource(
        nss,
        name,
        parent.map(x => nss.mkString("::") + "::" + x._1.name),
        stmts,
        implStmts,
        includes,
        variantStmt
      )

      // if (tpe.sum) {
      //   StructSource(tpe.name, None, Nil, Nil, Nil, Nil, struct :: variants.map(_.emit))
      // } else struct

      struct :: variants.flatMap(_.emit)

    }
  }

  trait ToCppTerm[A] extends (Option[A] => String)
  object ToCppTerm {
    given ToCppTerm[String]       = x => s"\"${x.getOrElse("")}\""
    given ToCppTerm[List[String]] = x => x.toList.flatten.mkString(".")
    given ToCppTerm[Int]          = x => s"${x.getOrElse(0)}"
    inline given derived[T](using m: Mirror.Of[T]): ToCppTerm[T] = (_: Option[T]) =>
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
        s"std::vector<${if (tpe.sum) s"${tpe.name}::Any" else tpe.name}>",
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
            s"${constValue[s.MirroredLabel]}",
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

  implicit val n: Cpp.ToCppType[polyregion.PolyAstUnused.Type]      = Cpp.ToCppType.derived
  implicit val nn: Cpp.ToCppType[polyregion.PolyAstUnused.TypeKind] = Cpp.ToCppType.derived
  implicit val nnn: Cpp.ToCppType[polyregion.PolyAstUnused.Named]   = Cpp.ToCppType.derived
  implicit val nnnn: Cpp.ToCppType[polyregion.PolyAstUnused.Term]   = Cpp.ToCppType.derived

  implicit val _n: Cpp.ToCppTerm[polyregion.PolyAstUnused.Type]      = Cpp.ToCppTerm.derived
  implicit val _nn: Cpp.ToCppTerm[polyregion.PolyAstUnused.TypeKind] = Cpp.ToCppTerm.derived
  implicit val _nnn: Cpp.ToCppTerm[polyregion.PolyAstUnused.Named]   = Cpp.ToCppTerm.derived
  implicit val _nnnn: Cpp.ToCppTerm[polyregion.PolyAstUnused.Term]   = Cpp.ToCppTerm.derived
//

  val alts = Cpp.deriveStruct[polyregion.PolyAstUnused.Sym]().emit
    ::: Cpp.deriveStruct[polyregion.PolyAstUnused.TypeKind]().emit
    ::: Cpp.deriveStruct[polyregion.PolyAstUnused.Type]().emit
    ::: Cpp.deriveStruct[polyregion.PolyAstUnused.Named]().emit
    ::: Cpp.deriveStruct[polyregion.PolyAstUnused.Position]().emit
    ::: Cpp.deriveStruct[polyregion.PolyAstUnused.Term]().emit
    ::: Cpp.deriveStruct[polyregion.PolyAstUnused.Tree]().emit

  val header = StructSource.emitHeader("foo", alts)
  // println(header)
  println("\n=========\n")
  val impl = StructSource.emitImpl("foo", "foo", alts)
  // println(impl)
  println("\n=========\n")

  Files.writeString(
    Paths.get("/home/tom/polyregion/native/src/foo.cpp"),
    impl,
    StandardOpenOption.TRUNCATE_EXISTING,
    StandardOpenOption.CREATE,
    StandardOpenOption.WRITE
  )
  Files.writeString(
    Paths.get("/home/tom/polyregion/native/src/foo.h"),
    header,
    StandardOpenOption.TRUNCATE_EXISTING,
    StandardOpenOption.CREATE,
    StandardOpenOption.WRITE
  )
//    import Cpp.*
//    println(T1Mid.T1ALeaf(Nil, List("a", "b"), 23, T1Mid.T1BLeaf))

  // println(Cpp.deriveStruct[Alt]().map(_.emitSource).mkString("\n"))
  // println(Cpp.deriveStruct[FirstTop]().map(_.emitSource).mkString("\n"))
  // println(Cpp.deriveStruct[First]().map(_.emitSource).mkString("\n"))
//    println(Cpp.deriveStruct[Foo]().map(_.emitSource).mkString("\n"))
  ()
}
