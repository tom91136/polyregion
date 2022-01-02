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
import polyregion.data.compileTime.CtorTermSelect
import cats.conversions.variance

object Cpp {

  extension [A](xs: Seq[A]) {
    def csv: String = xs.mkString(", ")
    def ssv: String = xs.mkString(" ")
    def sym(prefix: String = "", delim: String = "", suffix: String): String =
      if (xs.isEmpty) "" else xs.mkString(prefix, delim, suffix)
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
      forwardDeclStmts: List[String],
      nsDeclStmts: List[String]
  )

  object StructSource {

    val RequiredIncludes = List("memory", "variant", "iterator", "sstream")
    def emitHeader(namespace: String, xs: List[StructSource]) = {

      def nsStart(n: String) = if (n.isEmpty) "" else s"namespace $n { "
      def nsEnd(n: String)   = if (n.isEmpty) "" else s"} // namespace $n"

      val (_, stmts) = xs.foldLeft[(Option[String], List[String])]((None, Nil)) { case ((ns, stmts), cls) =>
        val moreStmts = "" ::
          cls.forwardDeclStmts :::                                        //
          s"struct ${cls.parent.fold(cls.name)(s"${cls.name} : " + _)} {" //
          :: cls.stmts.map("  " + _) :::                                  //
          "};" ::                                                         //
          cls.nsDeclStmts                                                 //
        val nsName = cls.namespaces.mkString("::")
        ns match {
          case None                   => (Some(nsName), stmts ::: nsStart(nsName) :: moreStmts)
          case Some(x) if x != nsName => (Some(nsName), stmts ::: nsEnd(x) :: nsStart(nsName) :: moreStmts)
          case Some(x)                => (Some(x), stmts ::: moreStmts)
        }
      }

      val includes = (RequiredIncludes ::: xs.flatMap(_.includes)) //
        .distinct.sorted
        .map(incl => s"#include <$incl>")
        .mkString("\n")

      val shared = //
        s"""|
            |template <typename... T> //
            |using Alternative = std::variant<std::shared_ptr<T>...>;
            |
            |template <typename... T> //
            |constexpr std::variant<T...> unwrap(const Alternative<T...> &a) {
            |  return std::visit([](auto &&arg) { return std::variant<T...>(*arg); }, a);
            |}
            |
            |template <typename... T> //
            |constexpr std::variant<T...> operator*(const Alternative<T...> &a) {
            |  return unwrap(a);
            |}
            |
            |template <auto member, class... T> //
            |constexpr auto select(const Alternative<T...> &a) {
            |  return std::visit([](auto &&arg) { return *(arg).*member; }, a);
            |}
            |
            |template <typename T> //
            |std::string to_string(const T& x) {
            |  std::ostringstream ss;
            |  ss << x;
            |  return ss.str();
            |}
            |
            |template <typename T, typename... Ts> //
            |constexpr std::optional<T> get_opt(const Alternative<Ts...> &a) {
            |  if (const std::shared_ptr<T> *v = std::get_if<std::shared_ptr<T>>(&a)) return {**v};
            |  else
            |    return {};
            |}""".stripMargin

      s"""|#pragma once
          |
          |$includes
          |
          |namespace $namespace {
          |$shared
          |#pragma clang diagnostic push
          |#pragma ide diagnostic ignored "google-explicit-constructor"
          |${stmts.sym("\n", "\n", "\n")}
          |} // namespace $namespace
          |#pragma clang diagnostic pop
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

    def clsName(qualified: Boolean) = {
      val name = if (tpe.kind == CppType.Kind.Base) "Base" else tpe.name
      if (qualified) tpe.namespace.sym("", "::", "::") + name else name
    }

    def emit: List[StructSource] = {

      val ctorInit = members.map { (n, tpe) =>
        if (tpe.kind == CppType.Kind.Base) s"$n(std::move($n))"
        else if (tpe.movable) s"$n(std::move($n))"
        else s"$n($n)"
      }
      val (ctorArgs, ctorChain) = parent match {
        case None => (members, ctorInit)
        case Some((base, Nil)) =>
          (members ::: base.members) ->
            (s"${base.clsName(qualified = true)}(${base.members.map((n, _) => n).csv})" :: ctorInit)

        case Some((base, xs)) =>
          members ->
            (s"${base.clsName(qualified = true)}(${xs.csv})" :: ctorInit)
      }

      import CppAttrs.*
      //TODO re-enable constexpr later
      val ctorAttrs = List(Constexpr.cond(tpe.constexpr && false), Explicit.cond(ctorArgs.sizeIs == 1)).flatten

      val ctorChainExpr = ctorChain match {
        case Nil => " = default;"
        case xs  => s" noexcept : ${xs.csv} {}"
      }

      val ctorArgExpr  = ctorArgs.map((n, t) => s"${t.ref(qualified = true)} $n").csv
      val ctorAttrExpr = ctorAttrs.map(_.repr).sym("", " ", " ")
      val ctorStmt     = s"$ctorAttrExpr${clsName(qualified = false)}($ctorArgExpr)$ctorChainExpr"
      val memberStmts  = members.map((n, t) => s"${t.ref(qualified = true)} $n;")

      def ns(name: String) =
        tpe.namespace.sym("", "::", "::") + name

      val hasMoreSumTypes = variants.exists(_.tpe.kind == CppType.Kind.Base)

      val (streamSig, streamImpl) =
        if (tpe.kind == CppType.Kind.Base && hasMoreSumTypes) {
          (Nil, Nil)
        } else {
          // val streamStmts =
          //   if (tpe.kind == CppType.Kind.Base) List(s"std::visit([&os](auto &&arg) { os << *arg; }, x);") :: Nil
          //   else
          //     members.map((n, tpe) =>
          //       if (tpe.kind == CppType.Kind.Base) s"std::visit([&os](auto &&arg) { os << *arg; }, x.$n);" :: Nil
          //       else tpe.streamOp("os", s"x.$n")
          //     )

          val streamStmts =
            if (tpe.kind == CppType.Kind.Base) "std::visit([&os](auto &&arg) { os << *arg; }, x);" :: Nil
            else {
              val fields = members.map((n, tpe) =>
                if (tpe.kind == CppType.Kind.Base && false)
                  s"std::visit([&os](auto &&arg) { os << *arg; }, x.$n);" :: Nil
                else tpe.streamOp("os", s"x.$n")
              )

              s"os << \"${tpe.name}(\";" ::
                fields.intercalate("os << ',';" :: Nil) :::
                "os << ')';" :: Nil
            }

          val streamMethodStmts =
            s"std::ostream &${ns("operator<<")}(std::ostream &os, const ${tpe.ref(qualified = true)} &x) {" ::
              streamStmts.map("  " + _) :::
              "  return os;" ::
              "}" :: Nil

          val streamProto =
            s"friend std::ostream &operator<<(std::ostream &os, const ${tpe.ref(qualified = true)} &);" :: Nil
          (streamProto, streamMethodStmts)
        }

      def foldVariants(xs: List[StructNode]): List[String] = xs.flatMap(s =>
        if (s.tpe.kind == CppType.Kind.Variant) s.tpe.name :: foldVariants(s.variants) else foldVariants(s.variants)
      )

      val variantStmt =
        if (tpe.kind == CppType.Kind.Base) {
          val allVariants   = foldVariants(variants)
          val memberGetters = members.map((n, t) => s"std::shared_ptr<${t.ref(qualified = true)}> $n(const Any &x);")
          allVariants.map(v => s"struct $v;") ::: //
            s"using Any = Alternative<${allVariants.csv}>;" :: Nil
        } else Nil

      val conversion =
        if (tpe.kind == CppType.Kind.Variant)
          s"operator Any() const { return std::make_shared<${tpe.name}>(*this); };" :: Nil
        else Nil

      val visibility = if (tpe.kind == CppType.Kind.Base) "protected:" :: Nil else Nil

      val (nsDecl, nsImpl) = if (tpe.kind == CppType.Kind.Base && !hasMoreSumTypes) {
        members.map { (n, t) =>
          val arg = tpe.ref(qualified = true)
          val rtn = t.ref(qualified = true)
          (
            s"$rtn $n(const $arg&);",
            s"$rtn ${ns(n)}(const $arg& x){ return select<&${clsName(qualified = true)}::$n>(x); }"
          )
        }.unzip
      } else (Nil, Nil)

      StructSource(
        namespaces = tpe.namespace,
        name = clsName(qualified = false),
        parent = parent.map(x => x._1.clsName(qualified = true)),
        stmts = memberStmts ::: visibility ::: ctorStmt :: conversion ::: streamSig,
        implStmts = streamImpl ::: nsImpl,
        includes = members.flatMap(_._2.include),
        variantStmt,
        nsDecl
      ) :: variants.flatMap(_.emit)
    }
  }

  object CppType {
    enum Kind { case StdLib, Data, Base, Variant }
    object Kind {
      def apply(c: compileTime.MirrorKind) = c match {
        case compileTime.MirrorKind.CaseClass   => Data
        case compileTime.MirrorKind.CaseProduct => Variant
        case compileTime.MirrorKind.CaseSum     => Base
      }
    }
  }
  case class CppType(
      namespace: List[String] = Nil,
      name: String,
      kind: CppType.Kind = CppType.Kind.StdLib,
      movable: Boolean = false,
      constexpr: Boolean = true,
      streamOp: (String, String) => List[String] = (s, v) => List(s"$s << $v;"),
      include: List[String] = Nil,
      ctors: List[CppType] = Nil
  ) {
    def ns(name: String) = s"${namespace.sym("", "::", "::")}${name}"
    // def qualified        = ns(name)

    def applied(qualified: Boolean): String = ctors match {
      case Nil => name
      case xs  => s"${name}<${ctors.map(_.ref(qualified)).csv}>"
    }

    def ref(qualified: Boolean = true): String = kind match {
      case CppType.Kind.Variant | CppType.Kind.Data | CppType.Kind.StdLib =>
        if (qualified) ns(applied(qualified)) else applied(qualified)
      case CppType.Kind.Base =>
        if (qualified) ns("Any") else "Any"
    }
  }

  trait ToCppTerm[A] extends (Option[A] => ToCppTerm.Value)
  object ToCppTerm {
    type Value = String | compileTime.CtorTermSelect[CppType]
    given ToCppTerm[String]                              = x => s"\"${x.getOrElse("")}\""
    given ToCppTerm[compileTime.CtorTermSelect[CppType]] = { x => x.getOrElse("") }
    inline given derived[T](using m: Mirror.Of[T]): ToCppTerm[T] = { (_: Option[T]) =>
      inline m match {
        case s: Mirror.SumOf[T]     => summonInline[ToCppType[s.MirroredMonoType]]().ref(qualified = true) + "()"
        case p: Mirror.ProductOf[T] => summonInline[ToCppType[p.MirroredMonoType]]().ref(qualified = true) + "()"
        case x                      => error(s"Unhandled derive: $x")
      }
    }
  }

  trait ToCppType[A] extends (() => CppType)
  object ToCppType {
    given ToCppType[Int]     = () => CppType(Nil, "int32_t", constexpr = true, include = List("cstdint"))
    given ToCppType[Long]    = () => CppType(Nil, "int64_t", constexpr = true, include = List("cstdint"))
    given ToCppType[Short]   = () => CppType(Nil, "int16_t", constexpr = true, include = List("cstdint"))
    given ToCppType[Char]    = () => CppType(Nil, "uint16_t", constexpr = true, include = List("cstdint"))
    given ToCppType[Byte]    = () => CppType(Nil, "int8_t", constexpr = true, include = List("cstdint"))
    given ToCppType[Boolean] = () => CppType(Nil, "bool", constexpr = true, include = Nil)
    given ToCppType[Float]   = () => CppType(Nil, "float", constexpr = true, include = Nil)
    given ToCppType[Double]  = () => CppType(Nil, "double", constexpr = true, include = Nil)
    given ToCppType[String] = () =>
      CppType(
        "std" :: Nil,
        "string",
        movable = true,
        constexpr = false,
        streamOp = (s, v) => s"""$s << '"' << $v << '"';""" :: Nil,
        include = List("string")
      )
    given [A: ToCppType]: ToCppType[List[A]] = { () =>
      val tpe = summon[ToCppType[A]]()
      CppType(
        "std" :: Nil,
        s"vector",
        movable = true,
        constexpr = false,
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
        include = List("vector"),
        ctors = tpe :: Nil
      )
    }

    // inline def forAll[T <: Tuple](p: CppType => Boolean): Boolean =
    //   inline erasedValue[T] match
    //     case _: EmptyTuple => true
    //     case _: (t *: ts)  => p(summonInline[ToCppType[t]]()) && forAll[ts](p)

    inline given derived[T](using m: Mirror.Of[T]): ToCppType[T] = { () =>
      val (ns, kind) = compileTime.mirrorMeta[m.MirroredMonoType]
      val name       = constValue[m.MirroredLabel]
      inline m match {
        case s: Mirror.SumOf[T] =>
          CppType(
            ns,
            name,
            movable = true,
            constexpr = false, // forAll[m.MirroredElemTypes](_.constexpr),
            kind = CppType.Kind(kind)
          )
        case p: Mirror.ProductOf[T] =>
          CppType(
            ns,
            name,
            movable = true,
            constexpr = false, // forAll[p.MirroredElemTypes](_.constexpr),
            kind = CppType.Kind(kind)
          )
        case x => error(s"Unhandled derive: $x")
      }
    }
  }

  inline def deriveSum[N <: Tuple, T <: Tuple](parent: Option[(StructNode, List[String])] = None): List[StructNode] =
    inline (erasedValue[N], erasedValue[T]) match {
      case (_: EmptyTuple, _: EmptyTuple) => Nil
      case (_: (n *: ns), _: (t *: ts)) =>
        deriveStruct[t](parent)(using
          summonInline[ToCppType[t]],
          summonInline[ToCppTerm[t]],
          summonInline[Mirror.Of[t]]
        ) :: deriveSum[ns, ts](parent)
    }

  inline def deriveProduct[L <: Tuple, T <: Tuple]: List[(String, CppType)] =
    inline (erasedValue[L], erasedValue[T]) match {
      case (_: EmptyTuple, _: EmptyTuple) => Nil
      case (_: (l *: ls), _: (t *: ts)) => (s"${constValue[l]}", summonInline[ToCppType[t]]()) :: deriveProduct[ls, ts]
    }
  inline def deriveStruct[T: ToCppType: ToCppTerm](parent: Option[(StructNode, List[String])] = None)(using
      m: Mirror.Of[T]
  ): StructNode = {
    val ctorTerms =
      compileTime.primaryCtorApplyTerms[m.MirroredType, ToCppTerm.Value, ToCppTerm, CppType, ToCppType].map {
        case x: String                   => x
        case CtorTermSelect((x, _), Nil) => x
        case CtorTermSelect((x, xt), (y, yt) :: Nil) =>
          if (xt.kind == CppType.Kind.Base) s"${xt.ns(y)}($x)"
          else s"$x.$y"
        case CtorTermSelect(x, xs) => throw new RuntimeException(s"multiple path ${x :: xs} is not supported")
      }
    val tpe     = summon[ToCppType[m.MirroredType]]()
    val applied = parent.map((s, _) => (s, ctorTerms))
    inline m match {
      case s: Mirror.SumOf[T] =>
        val members = compileTime.sumTypeCtorParams[s.MirroredType, CppType, ToCppType]
        val sum     = StructNode(tpe, members, applied)
        sum.copy(variants = deriveSum[s.MirroredElemLabels, s.MirroredElemTypes](Some((sum, ctorTerms))))
      case p: Mirror.ProductOf[T] =>
        val members = deriveProduct[p.MirroredElemLabels, p.MirroredElemTypes]
        StructNode(tpe, members, applied)
      case x => error(s"Unhandled derive: $x")
    }
  }
}
