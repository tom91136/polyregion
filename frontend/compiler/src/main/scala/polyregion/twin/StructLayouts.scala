package polyregion.twin

import polyregion.ast.PolyAST as p
import polyregion.jvm.compiler as ct
import polyregion.jvm.twin.Layout
import polyregion.jvm.twin.Layout.{Field, Kind}

object StructLayouts {

  val TagMember: String = "#tag"

  private def descriptorOf(tpe: p.Type): Option[String] = tpe match {
    case p.Type.Ptr(comp, _) => descriptorOf(comp).map("[" + _)
    case p.Type.Bool1        => Some("Z")
    case p.Type.IntS8        => Some("B")
    case p.Type.IntU16       => Some("C")
    case p.Type.IntS16       => Some("S")
    case p.Type.IntS32       => Some("I")
    case p.Type.IntS64       => Some("J")
    case p.Type.Float32      => Some("F")
    case p.Type.Float64      => Some("D")
    case _                   => None
  }

  /** The device layout of `sdef` as the compiler computed it. Members the compiler laid out but the struct does not
    * declare are dropped, except the dispatch tag. `nameOf` maps struct names to the keys twins are looked up by,
    * `descriptors` supplies the JVM descriptor of a host field the struct definition cannot express (a reference),
    * `readOnly` names the fields the kernel never changes, which may be `final` on the host.
    */
  def of(
      sdef: p.StructDef,
      layout: ct.Layout,
      className: String = "",
      tag: Option[Int] = None,
      nameOf: p.Sym => String = _.fqcn,
      descriptors: String => String = _ => "",
      readOnly: String => Boolean = _ => false
  ): Layout = {
    val laid = layout.members.map(m => m.name -> m).toMap
    val fields = sdef.members.flatMap { named =>
      laid.get(named.symbol).map { m =>
        val kind = named.tpe match {
          case p.Type.Struct(_, _) => Kind.STRUCT
          case p.Type.Ptr(_, _)    => Kind.ARRAY
          case _                   => Kind.PRIM
        }
        val element = (named.tpe match {
          case p.Type.Struct(name, _)                => Some(name)
          case p.Type.Ptr(p.Type.Struct(name, _), _) => Some(name)
          case _                                     => None
        }).fold("")(nameOf)
        val descriptor =
          Some(descriptors(named.symbol)).filter(_.nonEmpty).orElse(descriptorOf(named.tpe)).getOrElse("")
        Field(
          named.symbol,
          descriptor,
          kind,
          m.offsetInBytes.toInt,
          m.sizeInBytes.toInt,
          element,
          readOnly(named.symbol)
        )
      }
    }.toArray
    val tagOffset = laid.get(TagMember).fold(-1)(_.offsetInBytes.toInt)
    Layout(
      nameOf(sdef.name),
      className,
      layout.sizeInBytes.toInt,
      layout.alignment.toInt,
      tagOffset,
      if (tagOffset >= 0) tag.getOrElse(-1) else -1,
      fields
    )
  }
}
