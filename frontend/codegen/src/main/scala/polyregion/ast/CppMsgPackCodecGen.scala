package polyregion.ast

import cats.syntax.all.*
import polyregion.ast.CppStructGen.*

private[polyregion] case class CppMsgPackCodecGen(
    namespace: List[String],
    forwardDecls: List[String],
    impls: List[String]
)

private[polyregion] object CppMsgPackCodecGen {

  enum Schema {
    case Program, Package, PackageWire
  }

  enum Envelope {
    case Raw
    case Versioned(schema: Schema, payloadLabel: String, hashLabel: String)
  }

  case class Root(stem: String, tpe: CppType, envelope: Envelope)
  object Root {
    inline def raw[T: ToCppType: MsgPack.Codec](stem: String): Root =
      Root(stem, summon[ToCppType[T]](), Envelope.Raw)

    inline def versioned[T: ToCppType: MsgPack.Codec](
        stem: String,
        schema: Schema,
        payloadLabel: String,
        hashLabel: String
    ): Root =
      Root(stem, summon[ToCppType[T]](), Envelope.Versioned(schema, payloadLabel, hashLabel))
  }

  private def fromMsgpackFn(t: CppType)       = t.ref(qualified = false).toLowerCase + "_from_msgpack"
  private def toMsgpackFn(t: CppType)         = t.ref(qualified = false).toLowerCase + "_to_msgpack"
  private def fromMsgpackFieldsFn(t: CppType) = t.ref(qualified = false).toLowerCase + "_fields_from_msgpack"
  private def toMsgpackFieldsFn(t: CppType)   = t.ref(qualified = false).toLowerCase + "_fields_to_msgpack"

  private def isInt32Like(tpe: CppType) =
    tpe.kind == CppType.Kind.StdLib &&
      tpe.namespace.isEmpty &&
      Set("int8_t", "uint16_t", "int16_t", "int32_t").contains(tpe.name)

  private def readValue(tpe: CppType, name: String): List[String] =
    tpe.kind match {
      case CppType.Kind.StdLib =>
        (tpe.namespace ::: tpe.name :: Nil, tpe.ctors) match {
          case ("std" :: "optional" :: Nil, x :: Nil) =>
            s"${tpe.ref(qualified = true)} $name;" ::
              s"if(!r_.tryReadNil()) {" ::
              readValue(x, s"${name}_value").map("  " + _) :::
              s"  $name = std::move(${name}_value);" ::
              "}" :: Nil
          case ("std" :: "vector" :: Nil, x :: Nil) =>
            s"${tpe.ref(qualified = true)} $name;" ::
              "{" ::
              s"  auto ${name}_size = r_.readArrayHeader();" ::
              s"  $name.reserve(${name}_size);" ::
              s"  for(size_t ${name}_idx = 0; ${name}_idx < ${name}_size; ++${name}_idx) {" ::
              readValue(x, s"${name}_elem").map("    " + _) :::
              s"    ${name}.emplace_back(std::move(${name}_elem));" ::
              "  }" ::
              "}" :: Nil
          case ("std" :: "set" :: Nil, x :: Nil) =>
            s"${tpe.ref(qualified = true)} $name;" ::
              "{" ::
              s"  auto ${name}_size = r_.readArrayHeader();" ::
              s"  for(size_t ${name}_idx = 0; ${name}_idx < ${name}_size; ++${name}_idx) {" ::
              readValue(x, s"${name}_elem").map("    " + _) :::
              s"    ${name}.emplace(std::move(${name}_elem));" ::
              "  }" ::
              "}" :: Nil
          case ("std" :: "string" :: Nil, Nil) => s"auto $name = r_.readString();" :: Nil
          case ("bool" :: Nil, Nil)            => s"auto $name = r_.readBoolean();" :: Nil
          case ("float" :: Nil, Nil)           => s"auto $name = r_.readFloat32();" :: Nil
          case ("double" :: Nil, Nil)          => s"auto $name = r_.readFloat64();" :: Nil
          case ("int64_t" :: Nil, Nil)         => s"auto $name = r_.readInt64();" :: Nil
          case _ if isInt32Like(tpe) =>
            s"auto $name = static_cast<${tpe.ref(qualified = true)}>(r_.readInt32());" :: Nil
          case _ => s"auto $name = r_.readInt32();" :: Nil
        }
      case _ => s"auto $name = ${tpe.ns(fromMsgpackFn(tpe))}(r_);" :: Nil
    }

  private def writeValue(tpe: CppType, value: String, depth: Int = 0): List[String] =
    tpe.kind match {
      case CppType.Kind.StdLib =>
        (tpe.namespace ::: tpe.name :: Nil, tpe.ctors) match {
          case ("std" :: "optional" :: Nil, x :: Nil) =>
            s"if($value) {" ::
              writeValue(x, s"(*$value)", depth + 1).map("  " + _) :::
              "} else {" ::
              "  w_.writeNil();" ::
              "}" :: Nil
          case ("std" :: ("vector" | "set") :: Nil, x :: Nil) =>
            val elem = s"v${depth}_"
            s"w_.writeArrayHeader($value.size());" ::
              s"for(const auto &$elem : $value) {" ::
              writeValue(x, elem, depth + 1).map("  " + _) :::
              "}" :: Nil
          case ("std" :: "string" :: Nil, Nil) => s"w_.writeString($value);" :: Nil
          case ("bool" :: Nil, Nil)            => s"w_.writeBoolean($value);" :: Nil
          case ("float" :: Nil, Nil)           => s"w_.writeFloat32($value);" :: Nil
          case ("double" :: Nil, Nil)          => s"w_.writeFloat64($value);" :: Nil
          case ("int64_t" :: Nil, Nil)         => s"w_.writeInt64($value);" :: Nil
          case _                               => s"w_.writeInt32(static_cast<int32_t>($value));" :: Nil
        }
      case _ => s"${tpe.ns(toMsgpackFn(tpe))}(w_, $value);" :: Nil
    }

  private def readFieldsBody(s: StructNode, countExpr: String): List[String] = {
    val ctorInvocation = s.members match {
      case (name, _) :: Nil => s"${s.tpe.ref(qualified = true)}($name)"
      case _                => s.members.map(_._1).mkString("{", ", ", "}")
    }
    s"if($countExpr != ${s.members.size}) throw std::runtime_error(\"Expected ${s.tpe.ref(qualified = true)} with ${s.members.size} field(s)\");" ::
      s.members.flatMap { case (name, tpe) => readValue(tpe, name) } :::
      s"return $ctorInvocation;" :: Nil
  }

  private def writeFieldsBody(s: StructNode): List[String] =
    s.members.flatMap { case (name, tpe) => writeValue(tpe, s"x_.$name") }

  def emit(s: StructNode): List[CppMsgPackCodecGen] = {
    val forwardDecls =
      if (s.tpe.kind == CppType.Kind.Base)
        s"${s.tpe.ref(qualified = true)} ${fromMsgpackFn(s.tpe)}(MsgpackReader &);" ::
          s"void ${toMsgpackFn(s.tpe)}(MsgpackWriter &, const ${s.tpe.ref(qualified = true)} &);" :: Nil
      else
        s"${s.tpe.ref(qualified = true)} ${fromMsgpackFieldsFn(s.tpe)}(MsgpackReader &, size_t);" ::
          s"void ${toMsgpackFieldsFn(s.tpe)}(MsgpackWriter &, const ${s.tpe.ref(qualified = true)} &);" ::
          s"${s.tpe.ref(qualified = true)} ${fromMsgpackFn(s.tpe)}(MsgpackReader &);" ::
          s"void ${toMsgpackFn(s.tpe)}(MsgpackWriter &, const ${s.tpe.ref(qualified = true)} &);" :: Nil

    val impls =
      if (s.tpe.kind == CppType.Kind.Base) {
        val fromCases = s.variants.zipWithIndex.map { case (c, i) =>
          s"case $i: return ${c.tpe.ns(fromMsgpackFieldsFn(c.tpe))}(r_, n_ - 1);"
        }
        val fromNullaryCases = s.variants.zipWithIndex.map { case (c, i) =>
          if (c.members.isEmpty) s"case $i: return ${c.tpe.ns(fromMsgpackFieldsFn(c.tpe))}(r_, 0);"
          else s"case $i: throw std::runtime_error(\"Expected array payload for non-nullary sum ordinal\");"
        }
        val toCases = s.variants.zipWithIndex
          .map { case (c, i) =>
            val body =
              if (c.members.isEmpty) s"w_.writeInt32($i);" :: Nil
              else
                s"w_.writeArrayHeader(${c.members.size + 1});" ::
                  s"w_.writeInt32($i);" ::
                  s"${c.tpe.ns(toMsgpackFieldsFn(c.tpe))}(w_, y_);" :: Nil
            s"[&](const ${c.tpe.ref(qualified = true)} &y_) -> void {" :: body.map("  " + _) ::: "}" :: Nil
          }
          .intercalate("," :: Nil)
        "" ::
          s"${s.tpe.ref(qualified = true)} ${s.tpe.ns(fromMsgpackFn(s.tpe))}(MsgpackReader& r_) {" ::
          "  if(r_.nextIsArray()) {" ::
          "    auto n_ = r_.readArrayHeader();" ::
          "    if(n_ == 0) throw std::runtime_error(\"Expected non-empty sum payload\");" ::
          "    auto ord_ = r_.readInt32();" ::
          "    switch(ord_) {" ::
          fromCases.map("      " + _) :::
          "      default: throw std::out_of_range(\"Bad ordinal \" + std::to_string(ord_));" ::
          "    }" ::
          "  } else {" ::
          "    auto ord_ = r_.readInt32();" ::
          "    switch(ord_) {" ::
          fromNullaryCases.map("      " + _) :::
          "      default: throw std::out_of_range(\"Bad ordinal \" + std::to_string(ord_));" ::
          "    }" ::
          "  }" ::
          "}" ::
          "" ::
          s"void ${s.tpe.ns(toMsgpackFn(s.tpe))}(MsgpackWriter& w_, const ${s.tpe.ref(qualified = true)}& x_) {" ::
          "  x_.match_total(" ::
          toCases.map("    " + _) :::
          "  );" ::
          "}" :: Nil
      } else
        "" ::
          s"${s.tpe.ref(qualified = true)} ${s.tpe.ns(fromMsgpackFieldsFn(s.tpe))}(MsgpackReader& r_, size_t n_) {" ::
          readFieldsBody(s, "n_").map("  " + _) :::
          "}" ::
          "" ::
          s"void ${s.tpe.ns(toMsgpackFieldsFn(s.tpe))}(MsgpackWriter& w_, const ${s.tpe.ref(qualified = true)}& x_) {" ::
          writeFieldsBody(s).map("  " + _) :::
          "}" ::
          "" ::
          s"${s.tpe.ref(qualified = true)} ${s.tpe.ns(fromMsgpackFn(s.tpe))}(MsgpackReader& r_) {" ::
          "  auto n_ = r_.readArrayHeader();" ::
          s"  return ${s.tpe.ns(fromMsgpackFieldsFn(s.tpe))}(r_, n_);" ::
          "}" ::
          "" ::
          s"void ${s.tpe.ns(toMsgpackFn(s.tpe))}(MsgpackWriter& w_, const ${s.tpe.ref(qualified = true)}& x_) {" ::
          s"  w_.writeArrayHeader(${s.members.size});" ::
          s"  ${s.tpe.ns(toMsgpackFieldsFn(s.tpe))}(w_, x_);" ::
          "}" :: Nil

    s.variants.flatMap(emit) :+ CppMsgPackCodecGen(s.tpe.namespace, forwardDecls, impls)
  }

  def rootDecls(root: Root): List[String] = {
    val tpe = root.tpe.ref(qualified = true)
    s"[[nodiscard]] POLYREGION_EXPORT std::vector<uint8_t> ${root.stem}_to_msgpack(const $tpe&);" ::
      s"[[nodiscard]] POLYREGION_EXPORT $tpe ${root.stem}_from_msgpack(const uint8_t*, const uint8_t*);" ::
      s"[[nodiscard]] POLYREGION_EXPORT $tpe ${root.stem}_from_msgpack(const std::vector<uint8_t>&);" :: Nil
  }

  private def schemaHash(schema: Schema): String = schema match {
    case Schema.Program     => "ProgramHash"
    case Schema.Package     => "PackageHash"
    case Schema.PackageWire => "PackageWireHash"
  }

  def rootImpl(root: Root): List[String] = {
    val tpe = root.tpe.ref(qualified = true)
    val envelopeWrite = root.envelope match {
      case Envelope.Raw => Nil
      case Envelope.Versioned(schema, _, _) =>
        "w_.writeArrayHeader(2);" :: s"w_.writeString(std::string(${schemaHash(schema)}));" :: Nil
    }
    val envelopeRead = root.envelope match {
      case Envelope.Raw => Nil
      case Envelope.Versioned(schema, payloadLabel, hashLabel) =>
        val hash = schemaHash(schema)
        s"auto n_ = r_.readArrayHeader();" ::
          s"if(n_ != 2) throw std::runtime_error(\"Expected versioned $payloadLabel array of size 2\");" ::
          "auto hash_ = r_.readString();" ::
          s"if(hash_ != $hash) throw std::runtime_error(\"Expecting $hashLabel hash to be \" + std::string($hash) + \", but was \" + hash_);" :: Nil
    }
    "" ::
      s"std::vector<uint8_t> ${root.stem}_to_msgpack(const $tpe& x_) {" ::
      "  return encodeInterned([&](MsgpackWriter& w_) {" ::
      (envelopeWrite ::: writeValue(root.tpe, "x_")).map("    " + _) :::
      "  });" ::
      "}" ::
      "" ::
      s"$tpe ${root.stem}_from_msgpack(const uint8_t* begin_, const uint8_t* end_) {" ::
      "  return decodeMaybeInterned(begin_, end_, [](MsgpackReader& r_) {" ::
      (envelopeRead ::: readValue(root.tpe, "value_") ::: "return value_;" :: Nil).map("    " + _) :::
      "  });" ::
      "}" ::
      "" ::
      s"$tpe ${root.stem}_from_msgpack(const std::vector<uint8_t>& xs_) {" ::
      s"  return ${root.stem}_from_msgpack(xs_.data(), xs_.data() + xs_.size());" ::
      "}" :: Nil
  }
}
