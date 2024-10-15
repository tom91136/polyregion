package polyregion.ast.mirror

import cats.conversions.variance
import cats.syntax.all.*
import fansi.Str
import polyregion.ast.mirror.CppStructGen.{StructSource, ToCppType}
import polyregion.ast.mirror.compiletime
// import polyregion.ast.mirror.compiletime.CtorTermSelect

import polyregion.ast.mirror.CppStructGen.ToCppTerm.Value

import java.nio.file.{Files, Paths, StandardOpenOption}
import scala.annotation.tailrec
import scala.collection.immutable.LazyList.cons
import scala.compiletime.{constValue, erasedValue, error, summonInline}
import scala.deriving.*
import polyregion.ast.mirror.CppStructGen.CppType.Kind

private[polyregion] object CppStructGen {

  extension [A](xs: Seq[A]) {
    def csv: String = xs.mkString(", ")
    def ssv: String = xs.mkString(" ")
    def sym(prefix: String = "", delim: String = "", suffix: String): String =
      if (xs.isEmpty) "" else xs.mkString(prefix, delim, suffix)
  }

  enum CppAttrs(val repr: String) {
    case Constexpr extends CppAttrs("constexpr")
    case Explicit  extends CppAttrs("explicit")
    case Const     extends CppAttrs("const")
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
      nsDeclStmts: List[String],
      stdSpecialisationsDeclStmts: List[String => List[String]],
      stdSpecialisationsStmts: List[String => List[String]],
      headerImplStmts: String => List[String]
  )

  object StructSource {

    val RequiredIncludes = List("memory", "variant", "iterator", "sstream", "optional", "algorithm")
    def emitHeader(namespace: String, xs: List[StructSource]) = {

      def nsStart(n: String) = if (n.isEmpty) "" else s"namespace $n { "
      def nsEnd(n: String)   = if (n.isEmpty) "" else s"} // namespace $n"

      val (_, stmts, forwardDeclStmts) = xs.foldLeft[(Option[String], List[String], List[String])]((None, Nil, Nil)) {
        case ((ns, implStmts, forwardDeclStmts), cls) =>
          val moreImplStmts = "" ::
            s"struct POLYREGION_EXPORT ${cls.parent.fold(cls.name)(s"${cls.name} : " + _)} {" //
            :: cls.stmts.map("  " + _) :::                                                    //
            "};" ::                                                                           //
            cls.nsDeclStmts                                                                   //
          val nsName = cls.namespaces.mkString("::")
          ns match {
            case None =>
              (
                Some(nsName),
                implStmts ::: nsStart(nsName) :: moreImplStmts,
                forwardDeclStmts ::: nsStart(nsName) :: cls.forwardDeclStmts
              )
            case Some(x) if x != nsName =>
              (
                Some(nsName),
                implStmts ::: nsEnd(x) :: nsStart(nsName) :: moreImplStmts,
                forwardDeclStmts ::: nsEnd(x) :: nsStart(nsName) :: cls.forwardDeclStmts
              )
            case Some(x) => (Some(x), implStmts ::: moreImplStmts, forwardDeclStmts ::: cls.forwardDeclStmts)
          }
      }

      val includes = (RequiredIncludes ::: xs.flatMap(_.includes)) //
        .distinct.sorted
        .map(incl => s"#include <$incl>")
        .mkString("\n")

      val stdSpecialisationDecl = //
        s"""|namespace std {
            |
            |template <typename T> struct std::hash<std::vector<T>> {
            |  std::size_t operator()(std::vector<T> const &xs) const noexcept {
            |    std::size_t seed = xs.size();
            |    for (auto &x : xs) {
            |      seed ^= std::hash<T>()(x) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            |    }
            |    return seed;
            |  }
            |};
            |
            |${xs.flatMap(_.stdSpecialisationsDeclStmts.flatMap(_(namespace))).sym("\n", "\n", "\n")}
            |}
            |""".stripMargin

      val shared = //
        s"""|
            |
            |template <typename... Ts> class alternatives {
            |  template <class T> struct id {
            |    using type = T;
            |  };
            |
            |public:
            |  template <typename T, typename... Us> static constexpr bool all_unique_impl() {
            |    if constexpr (sizeof...(Us) == 0) return true;
            |    else
            |      return (!std::is_same_v<T, Us> && ...) && all_unique_impl<Us...>();
            |  }
            |
            |  template <size_t N, typename T, typename... Us> static constexpr auto at_impl() {
            |    if constexpr (N == 0) return id<T>();
            |    else
            |      return at_impl<N - 1, Us...>();
            |  }
            |
            |  static constexpr size_t size = sizeof...(Ts);
            |  template <typename T> static constexpr bool contains = (std::is_same_v<T, Ts> || ...);
            |  template <typename T> static constexpr bool all = (std::is_same_v<T, Ts> && ...);
            |  static constexpr bool all_unique = all_unique_impl<Ts...>();
            |  template <size_t N> using at = typename decltype(at_impl<N, Ts...>())::type;
            |};
            |
            |template <typename F, typename Ret, typename A, typename... Rest> //
            |A arg1_(Ret (F::*)(A, Rest...));
            |template <typename F, typename Ret, typename A, typename... Rest> //
            |A arg1_(Ret (F::*)(A, Rest...) const);
            |template <typename F> struct arg1 { using type = decltype(arg1_(&F::operator())); };
            |template <typename T> using arg1_t = typename arg1<T>::type;
            |
            |template <typename T> //
            |std::string to_string(const T& x) {
            |  std::ostringstream ss;
            |  ss << x;
            |  return ss.str();
            |}
            |
            |""".stripMargin

      s"""|#ifndef _MSC_VER
          |  #pragma clang diagnostic push
          |  #pragma clang diagnostic ignored "-Wunknown-pragmas"
          |#endif
          |
          |#pragma once
          |
          |$includes
          |#include "polyregion/export.h"
          |
          |namespace $namespace {
          |$shared
          |#ifndef _MSC_VER
          |  #pragma clang diagnostic push
          |  #pragma ide diagnostic ignored "google-explicit-constructor"
          |#endif
          |${forwardDeclStmts.sym("\n", "\n", "\n")}
          |${stmts.sym("\n", "\n", "\n")}
          |} // namespace $namespace
          |#ifndef _MSC_VER
          |  #pragma clang diagnostic pop // ide google-explicit-constructor
          |#endif
          |${xs.flatMap(_.headerImplStmts(namespace)).mkString("\n")}
          |$stdSpecialisationDecl
          |#ifndef _MSC_VER
          |  #pragma clang diagnostic pop // -Wunknown-pragmas
          |#endif
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
          |
          |${xs.flatMap(_.stdSpecialisationsStmts.flatMap(_(namespace))).sym("\n", "\n", "\n")}
          |
          |""".stripMargin

  }

  case class StructNode(
      tpe: CppType,
      members: List[(String, CppType)],
      parentTpe: Option[(CppType, List[String])],
      variants: List[StructNode] = Nil
  ) {

    def clsName(qualified: Boolean) = {
      val name = if (tpe.kind == CppType.Kind.Base) "Base" else tpe.name
      if (qualified) tpe.namespace.sym("", "::", "::") + name else name
    }

    def emit(parent: Option[StructNode] = None): List[StructSource] = {

      val ctorInit = members.map { (n, tpe) =>
        if (tpe.kind == CppType.Kind.Base) s"$n(std::move($n))"
        else if (tpe.movable) s"$n(std::move($n))"
        else s"$n($n)"
      }
      val (ctorArgs, ctorChain) = (parent, parentTpe) match {

        case (Some(base), Some(_, Nil)) =>
          (members ::: base.members) ->
            (s"${base.clsName(qualified = true)}(${base.members.map((n, _) => n).csv})" :: ctorInit)
        case (Some(base), Some(_, xs)) =>
          members ->
            (s"${base.clsName(qualified = true)}(${xs.csv})" :: ctorInit)
        case (_, _) => (members, ctorInit)
      }

      import CppAttrs.*
      // TODO re-enable constexpr later
      val ctorAttrs = List(Constexpr.cond(tpe.constexpr && false), Explicit.cond(ctorArgs.sizeIs == 1)).flatten

      val ctorChainExpr = ctorChain match {
        case Nil => " = default;"
        case xs  => s" noexcept : ${xs.csv} {}"
      }

      val ctorArgExpr  = ctorArgs.map((n, t) => s"${t.ref(qualified = true)} $n").csv
      val ctorAttrExpr = ctorAttrs.map(_.repr).sym("", " ", " ")
      val ctorStmt = s"$ctorAttrExpr${clsName(qualified = false)}($ctorArgExpr)${ctorChain match {
          case Nil => ";"
          case _   => " noexcept;";
        }}"
      val memberStmts = members.map((n, t) => s"${t.ref(qualified = true)} $n;")

      val ctorStmtImpl = s"${clsName(qualified = true)}::${clsName(qualified = false)}($ctorArgExpr)${ctorChain match {
          case Nil => " = default;"
          case xs  => s" noexcept : ${xs.csv} {}"
        }}"

      def ns(name: String) =
        tpe.namespace.sym("", "::", "::") + name

      val hasMoreSumTypes = variants.exists(_.tpe.kind == CppType.Kind.Base)

      val (streamSig, streamImpl) = tpe.kind match {
        case CppType.Kind.Base => (Nil, Nil)
        case _ =>
          (
            s"POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const ${clsName(qualified = true)} &);" :: Nil,
            tpe.namespace match {
              case Nil =>
                s"std::ostream &operator<<(std::ostream &os, const ${clsName(qualified = true)} &x) { return x.dump(os); }" :: Nil
              case xs =>
                s"namespace ${xs.mkString("::")} { std::ostream &operator<<(std::ostream &os, const ${clsName(qualified = true)} &x) { return x.dump(os); } }" :: Nil
            }
          )
      }

      val (dumpSig, dumpImpl) = tpe.kind match {
        case CppType.Kind.Base if hasMoreSumTypes => (Nil, Nil)
        case CppType.Kind.Base =>
          (
            s"[[nodiscard]] POLYREGION_EXPORT virtual std::ostream &dump(std::ostream &os) const = 0;" :: Nil,
            Nil
          )
        case _ =>
          val stmts =
            s"std::ostream &${clsName(qualified = true)}::dump(std::ostream &os) const {" ::
              s"  os << \"${tpe.name}(\";" ::
              members.map((n, tpe) => tpe.streamOp("os", s"$n").map("  " + _)).intercalate("  os << ',';" :: Nil) :::
              "  os << ')';" ::
              "  return os;" ::
              "}" :: Nil
          val sig =
            s"[[nodiscard]] POLYREGION_EXPORT std::ostream &dump(std::ostream &os) const${
                if (tpe.kind == CppType.Kind.Variant) " override" else ""
              };" :: Nil
          (sig, stmts)
      }

      val visibility = if (tpe.kind == CppType.Kind.Base) "protected:" :: Nil else Nil

      val (stdSpecialisationsDeclStmts, stdSpecialisationsStmts) = {

        val name = if (tpe.kind == CppType.Kind.Base) ns("Any") else clsName(qualified = true)

        val stdSpecialisationsDeclStmts = (ns: String) =>
          s"template <> struct std::hash<$ns::$name> {" ::
            s"  std::size_t operator()(const $ns::$name &) const noexcept;" ::
            s"};" :: Nil

        val stdSpecialisationsStmts = (ns: String) =>
          s"std::size_t std::hash<$ns::$name>::operator()(const $ns::$name &x) const noexcept { return x.hash_code(); }"
            :: Nil

        (stdSpecialisationsDeclStmts, stdSpecialisationsStmts)
      }

      val (equalitySig, equalityImpl) = {
        val name = clsName(qualified = true)

        def mkEqStmt(lhs: String, rhs: String, n: String, tpe: CppType) =
          (tpe.kind, tpe.namespace ::: tpe.name :: Nil, tpe.ctors) match {
            case (CppType.Kind.StdLib, "std" :: "optional" :: Nil, x :: Nil) if x.kind == CppType.Kind.Base =>
              s"( (!$lhs$n && !$rhs$n) || ($lhs$n && $rhs$n && *$lhs$n == *$rhs$n) )"
            case (CppType.Kind.StdLib, "std" :: "vector" :: Nil, x :: Nil) if x.kind == CppType.Kind.Base =>
              s"std::equal($lhs$n.begin(), $lhs$n.end(), $rhs$n.begin(), [](auto &&l, auto &&r) { return l == r; })"
            case _ => s"($lhs$n == $rhs$n)"
          }

        tpe.kind match {
          case CppType.Kind.Base =>
            (
              s"[[nodiscard]] POLYREGION_EXPORT virtual bool operator==(const $name &) const = 0;" :: Nil,
              Nil
            )
          case CppType.Kind.Variant =>
            (
              s"[[nodiscard]] POLYREGION_EXPORT bool operator==(const Base &) const override;" ::
                s"[[nodiscard]] POLYREGION_EXPORT bool operator==(const $name &) const;" :: Nil,
              s"[[nodiscard]] POLYREGION_EXPORT bool $name::operator==(const $name& rhs) const {" ::
                (members match {
                  case Nil => "  return true;" :: Nil
                  case xs =>
                    xs.map((n, tpe) => mkEqStmt("this->", "rhs.", n, tpe)).mkString("  return ", " && ", ";") :: Nil
                }) ::: "}"
                ::
                s"[[nodiscard]] POLYREGION_EXPORT bool $name::operator==(const Base& rhs_) const {" ::
                s"  if(rhs_.id() != variant_id) return false;" ::
                (members match {
                  case Nil => "  return true;" :: Nil
                  case xs =>
                    s"  return this->operator==(static_cast<const $name&>(rhs_)); // NOLINT(*-pro-type-static-cast-downcast)" :: Nil
                }) ::: "}" :: Nil
            )
          case _ =>
            (
              s"[[nodiscard]] POLYREGION_EXPORT bool operator==(const $name &) const;" :: Nil,
              s"[[nodiscard]] POLYREGION_EXPORT bool $name::operator==(const $name& rhs) const {" ::
                (members match {
                  case Nil => "  return true;" :: Nil
                  case xs  => xs.map((n, tpe) => mkEqStmt("", "rhs.", n, tpe)).mkString("  return ", " && ", ";") :: Nil
                }) ::: "}" :: Nil
            )
        }
      }

      val (idSig, idImpl) = (tpe.kind, parent) match {
        case (CppType.Kind.Variant, Some(p)) =>
          val idxValue = p.variants.indexWhere(_.tpe == tpe) match {
            case -1 => s"#error \"assert failed: invalid parent for variant ${tpe}\""
            case n  => s"$n"
          }
          (
            s"constexpr static uint32_t variant_id = $idxValue;" ::
              s"[[nodiscard]] POLYREGION_EXPORT uint32_t id() const override;" :: Nil,
            s"uint32_t ${clsName(qualified = true)}::id() const { return variant_id; };" :: Nil
          )
        case (CppType.Kind.Base, _) =>
          (
            s"[[nodiscard]] POLYREGION_EXPORT virtual uint32_t id() const = 0;" :: Nil,
            Nil
          )
        case (_, _) => (Nil, Nil)
      }

      val (hashCodeSig, hashCodeImpl) = (tpe.kind, parent) match {

        case (CppType.Kind.Variant, Some(p)) =>
          (
            s"[[nodiscard]] POLYREGION_EXPORT size_t hash_code() const override;" :: Nil,
            s"size_t ${clsName(qualified = true)}::hash_code() const { " ::
              "  size_t seed = variant_id;" ::
              members.map((n, t) =>
                s"  seed ^= std::hash<decltype($n)>()($n) + 0x9e3779b9 + (seed << 6) + (seed >> 2);"
              ) :::
              "  return seed;" ::
              "}" :: Nil
          )
        case (CppType.Kind.Base, _) =>
          (
            s"[[nodiscard]] POLYREGION_EXPORT virtual size_t hash_code() const = 0;" :: Nil,
            Nil
          )
        case (CppType.Kind.Data, _) =>
          (
            s"[[nodiscard]] POLYREGION_EXPORT size_t hash_code() const;" :: Nil,
            s"size_t ${clsName(qualified = true)}::hash_code() const { " ::
              "  size_t seed = 0;" ::
              members.map((n, t) =>
                s"  seed ^= std::hash<decltype($n)>()($n) + 0x9e3779b9 + (seed << 6) + (seed >> 2);"
              ) :::
              "  return seed;" ::
              "}" :: Nil
          )
        case (_, _) => (Nil, Nil)
      }

      def foldVariants(xs: List[StructNode]): List[String] = xs.flatMap(s =>
        if (s.tpe.kind == CppType.Kind.Variant) s.tpe.name :: foldVariants(s.variants) else foldVariants(s.variants)
      )

      val (conversionSig, conversionImpl) = if (tpe.kind == CppType.Kind.Variant) {
        (
          s"POLYREGION_EXPORT operator Any() const;" :: Nil,
          s"${clsName(qualified = true)}::operator ${ns("Any()")} const { return std::static_pointer_cast<Base>(std::make_shared<${tpe.name}>(*this)); }" :: Nil
        )
      } else (Nil, Nil)

      val (widenSig, widenImpl) = if (tpe.kind == CppType.Kind.Variant) {
        (
          s"[[nodiscard]] POLYREGION_EXPORT Any widen() const;" :: Nil,
          s"${ns("Any")} ${clsName(qualified = true)}::widen() const { return Any(*this); };" :: Nil
        )
      } else (Nil, Nil)

      val variantStmt =
        if (tpe.kind == CppType.Kind.Base) {
          val allVariants = foldVariants(variants)
          // val memberGetters = members.map((n, t) => s"std::shared_ptr<${t.ref(qualified = true)}> $n(const Any &x);")
          // allVariants.map(v => s"struct $v;") ::: //
          //   s"using Any = Alternative<${allVariants.csv}>;" :: Nil

          s"""
             |struct POLYREGION_EXPORT Base;
             |class Any {
             |  std::shared_ptr<Base> _v;
             |public:
             |  Any(std::shared_ptr<Base> _v) : _v(std::move(_v)) {}
             |  Any(const Any& other) : _v(other._v) {}
             |  Any(Any&& other) noexcept : _v(std::move(other._v)) {}
             |  Any& operator=(const Any& other) { return *this = Any(other); }
             |  Any& operator=(Any&& other) noexcept {
             |    std::swap(_v, other._v);
             |    return *this;
             |  }
             |  POLYREGION_EXPORT virtual std::ostream &dump(std::ostream &os) const; 
             |  POLYREGION_EXPORT friend std::ostream &operator<<(std::ostream &os, const Any &x);
             |  [[nodiscard]] POLYREGION_EXPORT bool operator==(const Any &rhs) const;
             |  [[nodiscard]] POLYREGION_EXPORT bool operator!=(const Any &rhs) const;
             |  [[nodiscard]] POLYREGION_EXPORT uint32_t id() const;
             |  [[nodiscard]] POLYREGION_EXPORT size_t hash_code() const;
             |${members.map((name, tpe) => s"  ${tpe.ref(true)} ${name}() const;").mkString("\n")}
             |  template<typename T> [[nodiscard]] constexpr POLYREGION_EXPORT bool is() const;
             |  template<typename T> [[nodiscard]] constexpr POLYREGION_EXPORT std::optional<T> get() const;
             |  template<typename... F> constexpr POLYREGION_EXPORT auto match_total(F &&...fs) const;
             |};
            """.stripMargin :: Nil
        } else if (tpe.kind == CppType.Kind.Data) {

          s"struct ${tpe.name};" :: Nil
        } else Nil

      val variantImpl = if (tpe.kind == CppType.Kind.Base) {

        s"uint32_t ${tpe.ref(true)}::id() const { return _v->id(); }" ::
          s"size_t ${tpe.ref(true)}::hash_code() const { return _v->hash_code(); }" ::
          members.map((name, t) => s"${t.ref(true)} ${tpe.ref(true)}::${name}() const { return _v->${name}; }") :::
          s"std::ostream &${tpe.ref(true)}::dump(std::ostream &os) const { return _v->dump(os); }" ::
          (tpe.namespace match {
            case Nil =>
              s"std::ostream &operator<<(std::ostream &os, const ${tpe.ref(false)} &x) { return x.dump(os); }"
            case xs =>
              s"namespace ${tpe.namespace.mkString("::")} { std::ostream &operator<<(std::ostream &os, const ${tpe.ref(false)} &x) { return x.dump(os); } }"
          }) ::
          s"bool ${tpe.ref(true)}::operator==(const ${tpe.ref(false)} &rhs) const { return _v->operator==(*rhs._v) ; }" ::
          s"bool ${tpe.ref(true)}::operator!=(const ${tpe.ref(false)} &rhs) const { return !_v->operator==(*rhs._v) ; }" ::
          Nil
      } else Nil

      val (nsDecl, nsImpl) = if (tpe.kind == CppType.Kind.Base && !hasMoreSumTypes && false) {
        members.map { (n, t) =>
          val arg = tpe.ref(qualified = true)
          val rtn = t.ref(qualified = true)
          (
            s"POLYREGION_EXPORT $rtn $n(const $arg&);",
            s"$rtn ${ns(n)}(const $arg& x){ return select<&${clsName(qualified = true)}::$n>(x); }"
          )
        }.unzip
      } else (Nil, Nil)

      val headerImpl = { (ns: String) =>

        def selectChain(xs: List[(String, String)], catchAll: String): List[String] = {
          def fmtOne(cond: String, stmt: String) = s"if($cond) { $stmt }"
          xs match {
            case (cond, stmt) :: Nil => fmtOne(cond, stmt) :: Nil
            case (cond, stmt) :: xs =>
              (fmtOne(cond, stmt) :: xs.map("else " + fmtOne(_, _))) ::: s"else { $catchAll }" :: Nil
            case Nil => catchAll :: Nil
          }
        }

        val sel = selectChain(variants.map(s => s.tpe.applied(true) -> "/*?*/"), "/*???*/")

        if (tpe.kind == CppType.Kind.Base) {
          val qualifiedNs = (ns :: tpe.namespace).mkString("::")
          s"namespace $qualifiedNs{ using All = alternatives<${variants.map(_.tpe.applied(true)).mkString(", ")}>; }" ::
            s"""|template<typename T> constexpr POLYREGION_EXPORT bool $ns::${tpe.ref(true)}::is() const { 
                |  static_assert(($qualifiedNs::All::contains<T>), "type not part of the variant");
                |  return T::variant_id == _v->id();
              |}""".stripMargin ::
            s"""|template<typename T> constexpr POLYREGION_EXPORT std::optional<T> $ns::${tpe.ref(true)}::get() const { 
                |  static_assert(($qualifiedNs::All::contains<T>), "type not part of the variant");
                |  if (T::variant_id == _v->id()) return {*std::static_pointer_cast<T>(_v)};
                |  else return {};
                |}""".stripMargin ::
            s"""|template<typename ...Fs> constexpr POLYREGION_EXPORT auto $ns::${tpe.ref(
                 true
               )}::match_total(Fs &&...fs) const { 
                |  using Ts = alternatives<std::decay_t<arg1_t<Fs>>...>;
                |  using Rs = alternatives<std::invoke_result_t<Fs, std::decay_t<arg1_t<Fs>>>...>;
                |  using R0 = typename Rs::template at<0>;
                |  static_assert($qualifiedNs::All::size == sizeof...(Fs), "match is not total as case count is not equal to variant's size");
                |  static_assert(($qualifiedNs::All::contains<std::decay_t<arg1_t<Fs>>> && ...), "one or more cases not part of the variant");
                |  static_assert((Rs::template all<R0>), "all cases must return the same type");
                |  static_assert(Ts::all_unique, "one or more cases overlap");
                |  uint32_t id = _v->id();
                |  if constexpr (std::is_void_v<R0>) {
                |    ([&]() -> bool {
                |      using T = std::decay_t<arg1_t<Fs>>;
                |      if (T::variant_id == id) {
                |        fs(*std::static_pointer_cast<T>(_v));
                |        return true;
                |      }
                |      return false;
                |    }() || ...);
                |    return;
                |  } else {
                |    std::optional<R0> r;
                |    ([&]() -> bool {
                |      using T = std::decay_t<arg1_t<Fs>>;
                |      if (T::variant_id == id) {
                |        r = fs(*std::static_pointer_cast<T>(_v));
                |        return true;
                |      }
                |      return false;
                |    }() || ...);
                |    return *r;
                |  }
                |
                |}""".stripMargin ::
            "" ::
            Nil
        } else Nil
      }

      StructSource(
        namespaces = tpe.namespace,
        name = clsName(qualified = false),
        parent = parent.map(_.clsName(qualified = true)),
        stmts = memberStmts //
          ::: idSig         //
          ::: hashCodeSig   //
          ::: dumpSig       //
          ::: equalitySig
          ::: visibility
          ::: ctorStmt :: conversionSig ::: widenSig ::: streamSig,
        implStmts =
          ctorStmtImpl :: idImpl ::: hashCodeImpl ::: streamImpl ::: dumpImpl ::: equalityImpl ::: conversionImpl ::: widenImpl ::: nsImpl ::: variantImpl,
        includes = members.flatMap(_._2.include),
        variantStmt,
        nsDecl,
        stdSpecialisationsDeclStmts :: Nil,
        stdSpecialisationsStmts :: Nil,
        headerImpl
      ) :: variants.flatMap(_.emit(Some(this)))
    }
  }

  object CppType {
    enum Kind { case StdLib, Data, Base, Variant }
    object Kind {
      def apply(c: compiletime.MirrorKind) = c match {
        case compiletime.MirrorKind.CaseClass   => Data
        case compiletime.MirrorKind.CaseProduct => Variant
        case compiletime.MirrorKind.CaseSum     => Base
      }
    }
  }
  case class CppType(
      namespace: List[String] = Nil,
      name: String,
      kind: CppType.Kind = CppType.Kind.StdLib,
      movable: Boolean = false,
      constexpr: Boolean = true,
      initialiser: Boolean = false,
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

    type Value = compiletime.Value[CppType]
    // given ToCppTerm[String] = x => compiletime.Value.Const(s"\"${x.getOrElse("")}\"")
    given ToCppTerm[Value] = { x => x.getOrElse(compiletime.Value.Const("")) }
    inline given derived[T](using m: Mirror.Of[T]): ToCppTerm[T] = { (x: Option[T]) =>
      inline m match {
        case s: Mirror.SumOf[T]     => compiletime.Value.CtorAp(summonInline[ToCppType[s.MirroredMonoType]](), Nil)
        case p: Mirror.ProductOf[T] => compiletime.Value.CtorAp(summonInline[ToCppType[p.MirroredMonoType]](), Nil)
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
    given [A: ToCppType, C[_] <: scala.collection.Seq[_]]: ToCppType[C[A]] = { () =>
      val tpe = summon[ToCppType[A]]()
      CppType(
        "std" :: Nil,
        s"vector",
        movable = true,
        constexpr = false,
        initialiser = true,
        streamOp = { (s, v) =>
          List(
            s"$s << '{';",
            s"if (!$v.empty()) {",
            // s"  for (auto &&x_ : $v) { ${tpe.streamOp(s, "x_").mkString(";")} $s << ','; }",
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
    given [A: ToCppType]: ToCppType[Option[A]] = { () =>
      val tpe = summon[ToCppType[A]]()
      CppType(
        "std" :: Nil,
        s"optional",
        movable = true,
        constexpr = false,
        streamOp = { (s, v) =>
          List(
            s"$s << '{';",
            s"if ($v) {",
            s"${tpe.streamOp(s, s"(*$v)").map("  " + _).mkString("\n")}",
            s"}",
            s"$s << '}';"
          )
        },
        include = List("optional"),
        ctors = tpe :: Nil
      )
    }
    given ToCppType[None.type] = { () =>
      CppType(
        "std" :: Nil,
        s"optional",
        movable = true,
        constexpr = false,
        initialiser = true,
        streamOp = { (s, v) => List(s"$s << '{}';") },
        include = List("optional")
      )
    }

    // inline def forAll[T <: Tuple](p: CppType => Boolean): Boolean =
    //   inline erasedValue[T] match
    //     case _: EmptyTuple => true
    //     case _: (t *: ts)  => p(summonInline[ToCppType[t]]()) && forAll[ts](p)

    inline given derived[T](using m: Mirror.Of[T]): ToCppType[T] = { () =>
      val (ns, kind) = compiletime.mirrorMeta[m.MirroredMonoType]
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

  inline def deriveSum[N <: Tuple, T <: Tuple](parent: Option[(CppType, List[String])] = None): List[StructNode] =
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
  inline def deriveStruct[T: ToCppType: ToCppTerm](parent: Option[(CppType, List[String])] = None)(using
      m: Mirror.Of[T]
  ): StructNode = {

    def write(x: ToCppTerm.Value): String =
      x match {
        case compiletime.Value.Const(value)            => value
        case compiletime.Value.TermSelect((x, _), Nil) => x
        case compiletime.Value.TermSelect((x, xt), (y, yt) :: Nil) =>
          if (xt.kind == CppType.Kind.Base) s"$x.${y}()"
          else s"$x.$y"
        case compiletime.Value.TermSelect(x, xs) =>
          throw new RuntimeException(s"multiple path ${x :: xs} is not supported")
        case compiletime.Value.CtorAp(tpe, args) =>
          if (!tpe.initialiser) tpe.ref(qualified = true) + s"(${args.map(write(_)).mkString(",")})"
          else s"{${args.map(write(_)).mkString(",")}}"
        case _ => ???
      }

    val ctorTerms =
      compiletime.primaryCtorApplyTerms[m.MirroredType, ToCppTerm.Value, ToCppTerm, CppType, ToCppType].map(write(_))
    val tpe     = summon[ToCppType[m.MirroredType]]()
    val applied = parent.map((s, _) => (s, ctorTerms))
    inline m match {
      case s: Mirror.SumOf[T] =>
        val members = compiletime.sumTypeCtorParams[s.MirroredType, CppType, ToCppType]
        StructNode(tpe, members, applied, deriveSum[s.MirroredElemLabels, s.MirroredElemTypes](Some(tpe -> ctorTerms)))
      case p: Mirror.ProductOf[T] =>
        val members = deriveProduct[p.MirroredElemLabels, p.MirroredElemTypes]
        StructNode(tpe, members, applied)
      case x => error(s"Unhandled derive: $x")
    }
  }
}
