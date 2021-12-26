package polyregion.data

import scala.deriving.*
import scala.compiletime.{constValue, erasedValue, error, summonInline}

object Cpp {

  case class CppType(name: String, movable: Boolean = false, include: List[String] = Nil)
  case class Struct(
      name: String,
      sum: Boolean,
      members: List[(CppType, String)],
      parent: Option[Struct],
      variants: List[Struct] = Nil
  ) {
    def declaredName: String = if (sum) s"${name}Base" else name
    def emitSource: String = {
      val ctorInit = members.map((tpe, n) => if (tpe.movable) s"$n(std::move($n))" else s"$n($n)")
      val (baseCls, ctorArgs, ctorChain) = parent match {
        case Some(s) =>
          (
            Some(s.declaredName),
            members ::: s.members,
            s"${s.declaredName}(${s.members.map((_, n) => n).mkString(", ")})" :: ctorInit
          )
        case None => (None, members, ctorInit)
      }
      val explicit = ctorArgs match {
        case _ :: Nil => "explicit "
        case _        => ""
      }
      val ctorChainExpr = ctorChain match {
        case Nil => ""
        case xs  => xs.mkString(" : ", ", ", "")
      }
      val ctorArgExpr = ctorArgs.map((t, n) => s"${t.name} $n").mkString(", ")
      val variantExpr =
        if (sum) s"using $name = std::variant<${variants.map(s => s.declaredName).mkString(", ")}>;" else ""
      val str = s"""struct $declaredName${baseCls.fold("")(" : " + _)} {
         |  ${members.map((t, n) => s"${t.name} $n;").mkString("\n  ")}
         |  $explicit$declaredName($ctorArgExpr)$ctorChainExpr {}
         |};""".stripMargin

      str + "\n" + variants.map(_.emitSource).mkString("\n") + s"\n$variantExpr"
    }
  }

  trait ToCppType[A] { def apply(): CppType }
  object ToCppType {
    given ToCppType[Int] with    { def apply(): CppType = CppType("int32_t", include = "cstdint" :: Nil)          }
    given ToCppType[Short] with  { def apply(): CppType = CppType("int16_t", include = "cstdint" :: Nil)          }
    given ToCppType[String] with { def apply(): CppType = CppType("std::string", movable = true, "string" :: Nil) }
    given [A](using ev: ToCppType[A]): ToCppType[List[A]] with {
      def apply(): CppType = CppType(s"std::vector<${ev().name}>", movable = true, "vector" :: Nil)
    }

    inline given derived[T](using m: Mirror.Of[T]): ToCppType[T] = () =>
      inline m match
        case s: Mirror.SumOf[T]     => CppType(s"${constValue[s.MirroredLabel]}Base", movable = true, Nil)
        case p: Mirror.ProductOf[T] => CppType(s"${constValue[p.MirroredLabel]}", movable = true, Nil)
        case x                      => error(s"Unhandled derive: $x")
  }

  inline def deriveSum[N <: Tuple, T <: Tuple](parent: Option[Struct] = None): List[Struct] =
    inline (erasedValue[N], erasedValue[T]) match
      case (_: EmptyTuple, _: EmptyTuple) => Nil
      case (_: (n *: ns), _: (t *: ts)) =>
        deriveStruct[t](parent)(using summonInline[Mirror.Of[t]]) ::: deriveSum[ns, ts](parent)

  inline def deriveProduct[L <: Tuple, T <: Tuple]: List[(CppType, String)] =
    inline (erasedValue[L], erasedValue[T]) match
      case (_: EmptyTuple, _: EmptyTuple) => Nil
      case (_: (l *: ls), _: (t *: ts)) => (summonInline[ToCppType[t]](), s"${constValue[l]}") :: deriveProduct[ls, ts]

  inline def deriveStruct[T](parent: Option[Struct] = None)(using m: Mirror.Of[T]): List[Struct] = inline m match
    case s: Mirror.SumOf[T] =>
      val sum = Struct(s"${constValue[s.MirroredLabel]}", true, Nil, parent)

      sum.copy(variants = deriveSum[s.MirroredElemLabels, s.MirroredElemTypes](Some(sum))) :: Nil

    case p: Mirror.ProductOf[T] =>
      val members = deriveProduct[p.MirroredElemLabels, p.MirroredElemTypes]
      Struct(constValue[p.MirroredLabel], false, members, parent) :: Nil
    case x => error(s"Unhandled derive: $x")

}

object Foo {


  enum A{
    case B
  }
  case class Foo(s: String, xs: List[List[String]])

  sealed abstract class First(val u: String, val xx : A) derives Cpp.ToCppType

  enum T1 extends First("", A.B) {
    case T1A(xs: List[Int], z: List[String], foo: Int, first: First)
    case T1B(b: String)
  }
  enum T2(val x: String, xx : A) extends First(x, xx)  {
    case T2A extends T2("z", A.B)
    case T2B(that: String, i: Int) extends T2("myconst", A.B)
  }

  enum Opt[+T] {
    case Sm(t: T, u: T)
    case Nn
  }

  enum Base(val u: Int) {
    case This(x: Int, s: String) extends Base(x)
    case That(a: String) extends Base(42)
  }

  @main def main2(): Unit = {

//    println(Opt.Sm(2, 7) == Opt.Sm(1 + 1, 7))
//    import Form.*
//    val x = summon[Form[Opt[Int]]]

    // compileTime.isMemberOfSealedHierarchy[First]
    // compileTime.isMemberOfSealedHierarchy[T1] // enum T1 extends First("") {
    compileTime.isMemberOfSealedHierarchy[T2]
    // compileTime.isMemberOfSealedHierarchy[T2.T2A.type]
    compileTime.isMemberOfSealedHierarchy[T2.T2B]

    //  enum T2(val x: String) extends First(x)  {
    //    case T2B extends T2("z")


  //  println(Cpp.deriveStruct[T2.T2B]().map(_.emitSource).mkString("\n"))
//    println(Cpp.deriveStruct[Foo]().map(_.emitSource).mkString("\n"))
    ()
  }
}
