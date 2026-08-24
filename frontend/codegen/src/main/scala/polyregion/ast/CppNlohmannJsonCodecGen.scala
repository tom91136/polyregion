package polyregion.ast

import polyregion.ast.CppStructGen.*
import cats.syntax.all.*

private[polyregion] case class CppNlohmannJsonCodecGen(
    namespace: List[String],
    decls: List[String],
    impls: List[String]
)
private[polyregion] object CppNlohmannJsonCodecGen {

  def fromJsonFn(t: CppType) = t.ref(qualified = false).toLowerCase + "_from_json"
  def toJsonFn(t: CppType)   = t.ref(qualified = false).toLowerCase + "_to_json"
  def jsonAt(idx: Int)       = s"j_.at($idx)"

  def fromJsonBody(s: StructNode) = if (s.tpe.kind == CppType.Kind.Base) {
    s"size_t ord_ = ${jsonAt(0)}.get<size_t>();" ::
      s"const auto &t_ = ${jsonAt(1)};" ::
      "switch (ord_) {" ::
      s.variants.zipWithIndex.map((c, i) => s"case ${i}: return ${c.tpe.ns(fromJsonFn(c.tpe))}(t_);") :::
      s"default: throw std::out_of_range(\"Bad ordinal \" + std::to_string(ord_));" ::
      "}" :: Nil
  } else {

    val ctorInvocation = s.members match {
      case (name, _) :: Nil => s"${s.tpe.ref(qualified = true)}($name)"
      case _                => s.members.map(_._1).mkString("{", ", ", "}")
    }

    s.members.zipWithIndex.flatMap { case ((name, tpe), idx) =>
      tpe.kind match {
        case CppType.Kind.StdLib =>
          (tpe.namespace ::: tpe.name :: Nil, tpe.ctors) match {
            case ("std" :: "optional" :: Nil, x :: Nil) =>
              val nested =
                if (x.kind != CppType.Kind.StdLib)
                  s"${x.ns(fromJsonFn(x))}(${jsonAt(idx)})"
                else
                  s"${jsonAt(idx)}.get<${x.ref(qualified = true)}>()"

              s"auto $name = ${jsonAt(idx)}.is_null() ? std::nullopt : std::make_optional($nested);" :: Nil
            case ("std" :: (c @ ("vector" | "set")) :: Nil, x :: Nil) if x.kind != CppType.Kind.StdLib =>
              s"${tpe.ref(qualified = true)} $name;" ::
                s"for(const auto &v_ : ${jsonAt(idx)}) { ${name}.${c match {
                    case "vector" => "emplace_back"
                    case "set"    => "emplace"
                  }}(${x.ns(fromJsonFn(x))}(v_)); }"
                :: Nil
            case _ => s"auto $name = ${jsonAt(idx)}.get<${tpe.ref(qualified = true)}>();" :: Nil
          }
        case _ => s"auto $name = ${tpe.ns(fromJsonFn(tpe))}(${jsonAt(idx)});" :: Nil
      }
    } :::
      s"return ${ctorInvocation};" ::
      Nil
  }

  def toJsonBody(s: StructNode) = if (s.tpe.kind == CppType.Kind.Base) {
    "return x_.match_total(" ::
      s.variants.zipWithIndex
        .map((c, i) =>
          s"[](const ${c.tpe.ref(qualified = true)} &y_) -> json { return {$i, ${c.tpe.ns(toJsonFn(c.tpe))}(y_)}; }" :: Nil
        )
        .intercalate("," :: Nil) :::
      ");" :: Nil
  } else {
    s.members.flatMap { case (name, tpe) =>
      tpe.kind match {
        case CppType.Kind.StdLib =>
          (tpe.namespace ::: tpe.name :: Nil, tpe.ctors) match {
            case ("std" :: "optional" :: Nil, x :: Nil) =>
              val deref =
                if (x.kind != CppType.Kind.StdLib) s"${x.ns(toJsonFn(x))}(*x_.${name})"
                else s"json(*x_.${name})"
              s"auto $name = x_.${name} ? $deref : json();" :: Nil

            case ("std" :: ("vector" | "set") :: Nil, x :: Nil) if x.kind != CppType.Kind.StdLib =>
              s"std::vector<json> $name;" ::
                s"for(const auto &v_ : x_.${name}) { ${name}.emplace_back(${x.ns(toJsonFn(x))}(v_)); }"
                :: Nil
            case _ => s"auto $name = x_.${name};" :: Nil
          }
        case _ => s"auto $name = ${tpe.ns(toJsonFn(tpe))}(x_.${name});" :: Nil
      }
    } :::
      s"return json::array(${s.members.map(_._1).mkString("{", ", ", "}")});" ::
      Nil
  }

  def emit(s: StructNode): List[CppNlohmannJsonCodecGen] = {
    val fromJsonImpl = "" ::
      s"${s.tpe.ref(qualified = true)} ${s.tpe.ns(fromJsonFn(s.tpe))}(const json& j_) { " :: //
      fromJsonBody(s).map("  " + _) :::                                                      //
      "}" :: Nil                                                                             //

    val toJsonImpl = "" ::
      s"json ${s.tpe.ns(toJsonFn(s.tpe))}(const ${s.tpe.ref(qualified = true)}& x_) { " :: //
      toJsonBody(s).map("  " + _) :::
      "}" :: Nil //

    val decls =
      s"[[nodiscard]] POLYREGION_EXPORT ${s.tpe.ref(qualified = true)} ${fromJsonFn(s.tpe)}(const json &);" ::
        s"[[nodiscard]] POLYREGION_EXPORT json ${toJsonFn(s.tpe)}(const ${s.tpe.ref(qualified = true)} &);" ::
        Nil

    s.variants.flatMap(s => emit(s)) :+
      CppNlohmannJsonCodecGen(
        s.tpe.namespace,
        decls,
        fromJsonImpl ::: toJsonImpl
      )
  }

}
