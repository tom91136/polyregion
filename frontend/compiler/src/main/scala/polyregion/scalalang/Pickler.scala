package polyregion.scalalang

import cats.syntax.all.*
import polyregion.ast.pass.Devirtualise
import polyregion.ast.{PolyAST as p, *}
import polyregion.jvm.runtime as rt
import polyregion.jvm.twin.{FieldAccess, Layout, Marshal, TwinException}
import polyregion.prism.{Prism, StdLib}

import java.nio.{ByteBuffer, ByteOrder}
import scala.annotation.tailrec
import scala.quoted.*
import polyregion.prism.StdLib.MutableSeq

object Pickler {

  // Macro-time dump gate; matches `compiletime.DEBUG` semantics (POLYREGION_DEBUG=1).
  private val DEBUG                                   = sys.env.contains(Env.PolyregionDebug)
  private inline def debug(inline s: => String): Unit = if (DEBUG) println(s)

  def tpeAsRuntimeTpe(t: p.Type): rt.Type = t match {
    case p.Type.Bool1        => rt.Type.BOOL
    case p.Type.IntS8        => rt.Type.BYTE
    case p.Type.IntU16       => rt.Type.CHAR
    case p.Type.IntS16       => rt.Type.SHORT
    case p.Type.IntS32       => rt.Type.INT
    case p.Type.IntS64       => rt.Type.LONG
    case p.Type.Float32      => rt.Type.FLOAT
    case p.Type.Float64      => rt.Type.DOUBLE
    case p.Type.Ptr(_, _)    => rt.Type.PTR
    case p.Type.Struct(_, _) => rt.Type.PTR
    case p.Type.Unit0        => rt.Type.VOID
    case illegal             => throw RuntimeException(s"tpeAsRuntimeTpe: Illegal $illegal")
  }

  def readPrim(using q: Quotes) //
  (source: Expr[java.nio.ByteBuffer], byteOffset: Expr[Int], tpe: p.Type): Expr[Any] = tpe match {
    case p.Type.Float32 => '{ $source.getFloat($byteOffset) }
    case p.Type.Float64 => '{ $source.getDouble($byteOffset) }
    case p.Type.Bool1   => '{ if ($source.get($byteOffset) == 0) false else true }
    case p.Type.IntS8   => '{ $source.get($byteOffset) }
    case p.Type.IntU16  => '{ $source.getChar($byteOffset) }
    case p.Type.IntS16  => '{ $source.getShort($byteOffset) }
    case p.Type.IntS32  => '{ $source.getInt($byteOffset) }
    case p.Type.IntS64  => '{ $source.getLong($byteOffset) }
    case p.Type.Unit0   => '{ () }
    case x =>
      throw RuntimeException(s"Cannot get ${x.repr} from buffer, it is not a primitive type")

  }

  def writePrim(using q: Quotes) //
  (target: Expr[java.nio.ByteBuffer], byteOffset: Expr[Int], tpe: p.Type, value: Expr[Any]): Expr[Unit] = tpe match {
    case p.Type.Float32 => '{ $target.putFloat($byteOffset, ${ value.asExprOf[Float] }) }
    case p.Type.Float64 => '{ $target.putDouble($byteOffset, ${ value.asExprOf[Double] }) }
    case p.Type.Bool1   => '{ $target.put($byteOffset, if (!${ value.asExprOf[Boolean] }) 0.toByte else 1.toByte) }
    case p.Type.IntS8   => '{ $target.put($byteOffset, ${ value.asExprOf[Byte] }) }
    case p.Type.IntU16  => '{ $target.putChar($byteOffset, ${ value.asExprOf[Char] }) }
    case p.Type.IntS16  => '{ $target.putShort($byteOffset, ${ value.asExprOf[Short] }) }
    case p.Type.IntS32  => '{ $target.putInt($byteOffset, ${ value.asExprOf[Int] }) }
    case p.Type.IntS64  => '{ $target.putLong($byteOffset, ${ value.asExprOf[Long] }) }
    case p.Type.Unit0   => '{ () }
    case x =>
      throw RuntimeException(
        s"Cannot put ${x.repr} into buffer, it is not a primitive type (source is `${value.show}`)"
      )
  }

  private case class StructMapping[A](
      source: p.StructDef,
      sizeInBytes: Long,
      write: Option[(p.Mirror, A => A)],
      read: Option[(p.Mirror, A => A)],
      members: List[StructMapping.Member[A]],
      tagOffset: Int,
      tag: Int
  )
  private object StructMapping {
    case class Member[A](name: String, tpe: p.Type, sizeInBytes: Long, offsetInBytes: Long, descriptor: String)
  }

  @SuppressWarnings(Array("bonsai:complexity"))
  private def mkStructMapping(using
      q: Quoted
  )( //
      sdef: p.StructDef,
      layouts: Map[p.StructDef, (Layout, Option[Prism])]
  ): StructMapping[q.Term] = {
    val (layout, maybePrism) =
      layouts.getOrElse(sdef, q.report.errorAndAbort(s"Unseen sdef ${sdef.repr}, known reprs: ${layouts}"))
    val layoutTable = layout.fields.map(f => f.name -> f).toMap
    val members = Devirtualise.hostMembers(sdef).map { named =>
      layoutTable
        .get(named.symbol)
        .map(f => StructMapping.Member[q.Term](named.symbol, named.tpe, f.size, f.offset, f.descriptor))
        .getOrElse(q.report.errorAndAbort(s"Layout $layout is missing ${named.repr} from ${sdef.repr}"))
    }

    val (write, read) = maybePrism.map { case (mirror, (write, read, update)) =>
      (
        (mirror, (root: q.Term) => write(q.underlying, root.asExpr).asTerm),
        (mirror, (restored: q.Term) => read(q.underlying, restored.asExpr).asTerm)
      )
    }.unzip

    StructMapping(sdef, layout.size, write, read, members, layout.tagOffset, layout.tag)
  }

  def deriveAllRepr(using
      q: Quoted
  )( //
      lut: Map[p.Sym, (p.StructDef, Option[Prism])],
      sdef: p.StructDef,
      repr: q.TypeRepr,
      tags: Map[p.Sym, Int] = Map.empty
  ): Map[p.StructDef, q.TypeRepr] = {
    def go(using q: Quoted)(sdef: p.StructDef, repr: q.TypeRepr, added: Set[p.Sym]): List[(p.StructDef, q.TypeRepr)] = {
      val added0 = added + sdef.name
      def descend(name: p.Sym, member: String): List[(p.StructDef, q.TypeRepr)] =
        if (added0.contains(name)) Nil
        else
          lut.get(name) match {
            case None              => Nil
            case Some((target, _)) => go(target, q.TermRef(repr, member), added0)
          }
      // For array-element struct types (`Type.Ptr(Struct, ...)`) we have no parent TermRef path
      // to descend through. The parent struct here is always a single-type-param container
      // (ListBuffer[A], Buffer[A], Array[A], MutableSeq[A]) where the lone type arg IS the array
      // element type — extract it from the parent's AppliedType so the element's write/read
      // helpers get registered.
      def descendArrayElement(name: p.Sym): List[(p.StructDef, q.TypeRepr)] =
        if (added0.contains(name)) Nil
        else {
          val elemReprOpt = repr.widenTermRefByName match {
            case q.AppliedType(_, head :: _) => Some(head)
            case _                           => None
          }
          (lut.get(name), elemReprOpt) match {
            case (Some((sdef, None)), Some(elemRepr)) => go(sdef, elemRepr, added0)
            case _                                    => Nil
          }
        }
      def descendCandidates: List[(p.StructDef, q.TypeRepr)] =
        if (Dispatch.isRoot(sdef, tags))
          Dispatch.candidateReprs(repr, tags).flatMap { (name, candidate) =>
            lut.get(name).filterNot(x => added0.contains(x._1.name)).toList.flatMap((d, _) => go(d, candidate, added0))
          }
        else Nil
      (sdef, repr.widenTermRefByName) :: descendCandidates ::: sdef.members.flatMap {
        case p.Named(member, p.Type.Struct(name, _), _)           => descend(name, member)
        case p.Named(_, p.Type.Ptr(p.Type.Struct(name, _), _), _) => descendArrayElement(name)
        case _                                                    => Nil
      }
    }
    go(sdef, repr, Set()).toMap
  }

  private def mkMethodSym(using q: Quoted)(name: String, rtn: q.TypeRepr, args: (String, q.TypeRepr)*) =
    q.Symbol.newMethod(
      q.Symbol.spliceOwner,
      name,
      q.MethodType(args.map(_._1).toList)(paramInfosExp = _ => args.map(_._2).toList, resultTypeExp = _ => rtn)
    )

  private def mkMethodDef(using q: Quoted)(sym: q.Symbol)(impl: PartialFunction[List[q.Tree], Expr[Any]]) = q.DefDef(
    sym,
    {
      case (argList0 :: Nil) =>
        impl
          .lift(argList0)
          .fold(q.report.errorAndAbort(s"Definition is not defined for input ${argList0}"))(expr =>
            Some(expr.asTerm.changeOwner(sym))
          )
      case bad => q.report.errorAndAbort(s"Unexpected argument in method body: expected ${sym.signature}, got ${bad}")
    }
  )

  def generateAll(using
      q: Quoted
  )(
      lut: Map[p.Sym, p.StructDef],
      layouts: Map[p.StructDef, (Layout, Option[Prism])],
      reprs: Map[p.StructDef, q.TypeRepr],
      residentPointer: Expr[Any => Long],
      tags: Map[p.Sym, Int] = Map.empty
  ) = {
    given Quotes = q.underlying

    // For struct types that appear with multiple distinct generic instantiations in the same
    // capture set (e.g. `Monoid[Int]`, `Monoid[Float]`, `Monoid[Float2]` all sharing the polymorphic
    // `Monoid` StructDef), reprs only tracks ONE instantiation — whichever ended up last in the
    // merged map. Pinning the read/write method signatures to that one would reject the other
    // call sites at typecheck time. Widening to `Any` for empty-member structs (which are opaque
    // to the kernel anyway) lets all instantiations call the same method safely.
    def methodParamRepr(sdef: p.StructDef, repr: q.TypeRepr): q.TypeRepr =
      if (sdef.members.isEmpty) q.TypeRepr.of[Any] else repr
    def methodSyms(prefix: String, rtn: (p.StructDef, q.TypeRepr) => q.TypeRepr)(
        extra: => Seq[(String, q.TypeRepr)]
    ): Map[p.Sym, q.Symbol] =
      reprs.toList
        .map((sdef, repr) =>
          sdef.name -> mkMethodSym(
            s"${prefix}_${sdef.name.fqcn}",
            rtn(sdef, repr),
            (("root" -> methodParamRepr(sdef, repr)) +: extra)*
          )
        )
        .toMap

    val marshalArg = "m" -> q.TypeRepr.of[Marshal]

    // Index by struct name (p.Sym) rather than StructDef instance: callers' `lut(name)` can return
    // a StructDef whose `parents` (or other derived fields) differ from the one stored in `reprs`,
    // even when they refer to the same logical type — using the name as the key avoids that
    // identity mismatch.
    val writeSymbols: Map[p.Sym, q.Symbol] = methodSyms("write", (_, _) => q.TypeRepr.of[Long])(Seq(marshalArg))
    val fillSymbols: Map[p.Sym, q.Symbol] = methodSyms("fill", (_, _) => q.TypeRepr.of[Unit])(
      Seq(marshalArg, "buffer" -> q.TypeRepr.of[ByteBuffer], "base" -> q.TypeRepr.of[Int])
    )

    def ptrArgs: Seq[(String, q.TypeRepr)] = Seq("ptr" -> q.TypeRepr.of[Long], marshalArg)

    val readSymbols: Map[p.Sym, q.Symbol]   = methodSyms("read", methodParamRepr)(ptrArgs)
    val updateSymbols: Map[p.Sym, q.Symbol] = methodSyms("update", (_, _) => q.TypeRepr.of[Unit])(ptrArgs)

    def allocateBuffer(size: Expr[Int]) = '{ Marshal.allocate($size) }

    def callWrite(name: p.Sym, root: Expr[Any], m: Expr[Marshal]) =
      q.Apply(q.Ref(writeSymbols(name)), List(root.asTerm, m.asTerm)).asExprOf[Long]

    def callFill(name: p.Sym, root: Expr[Any], m: Expr[Marshal], buffer: Expr[ByteBuffer], base: Expr[Int]) =
      q.Apply(q.Ref(fillSymbols(name)), List(root.asTerm, m.asTerm, buffer.asTerm, base.asTerm)).asExprOf[Unit]

    def callRead(name: p.Sym, root: Expr[Any], ptr: Expr[Long], m: Expr[Marshal]) =
      q.Apply(q.Ref(readSymbols(name)), List(root.asTerm, ptr.asTerm, m.asTerm)).asExpr

    def callUpdate(name: p.Sym, root: Expr[Any], ptr: Expr[Long], m: Expr[Marshal]) =
      q.Apply(q.Ref(updateSymbols(name)), List(root.asTerm, ptr.asTerm, m.asTerm)).asExprOf[Unit]

    val mappings: Map[p.Sym, StructMapping[q.Term]] =
      reprs.keys.map(mkStructMapping(_, layouts)).map(m => m.source.name -> m).toMap

    // fields the generated code cannot name, e.g. `private` members, go through the layout-checked VarHandle fallback
    def fieldSymbol(recv: q.Term, name: String): q.Symbol = recv.tpe.widenTermRefByName.typeSymbol.fieldMember(name)

    def reachable(recv: q.Term, name: String): Option[q.Term] = {
      val sym = fieldSymbol(recv, name)
      Option.when(sym.isNoSymbol || !sym.flags.is(q.Flags.Private))(q.Select.unique(recv, name))
    }

    def fieldAccess(recv: q.Term, m: StructMapping.Member[q.Term]): Expr[FieldAccess] = {
      val owner = recv.asExprOf[AnyRef]
      '{ FieldAccess.of($owner.getClass, ${ Expr(m.name) }, ${ Expr(m.descriptor) }) }
    }

    def isPlainArray(recv: q.Term, m: StructMapping.Member[q.Term]): Boolean = {
      val sym = fieldSymbol(recv, m.name)
      !sym.isNoSymbol &&
      recv.tpe.widenTermRefByName.memberType(sym).widenTermRefByName.dealias.simplified.typeSymbol == q.defn.ArrayClass
    }

    @tailrec def stageable(owner: p.StructDef, member: String, comp: p.Type): Boolean = comp match {
      case p.Type.Ptr(inner, _) => stageable(owner, member, inner)
      case p.Type.Bool1 | p.Type.IntS8 | p.Type.IntU16 | p.Type.IntS16 | p.Type.IntS32 | p.Type.IntS64 |
          p.Type.Float32 | p.Type.Float64 =>
        true
      case p.Type.Struct(name, _) if lut.get(name).exists(_.members.nonEmpty) =>
        q.report.errorAndAbort(
          s"Array member `$member` of `${owner.name.fqcn}` holds `${name.fqcn}` elements, " +
            "only arrays of primitives and arrays of such arrays can be captured"
        )
      case _ => false
    }

    def planArray(recv: q.Term, mapping: StructMapping[q.Term], m: StructMapping.Member[q.Term]): Option[p.Type] =
      m.tpe match {
        case p.Type.Ptr(comp, _) if isPlainArray(recv, m) && stageable(mapping.source, m.name, comp) => Some(comp)
        case _                                                                                       => None
      }

    def hasMutableField(tpe: q.TypeRepr): Boolean =
      tpe.baseClasses.exists(_.declaredFields.exists(_.flags.is(q.Flags.Mutable)))

    def readField(recv: q.Term, m: StructMapping.Member[q.Term]): q.Term =
      reachable(recv, m.name).getOrElse {
        recv.tpe.widenTermRefByName.memberType(fieldSymbol(recv, m.name)).widenTermRefByName.asType match {
          case '[t] => '{ ${ fieldAccess(recv, m) }.get(${ recv.asExprOf[AnyRef] }).asInstanceOf[t] }.asTerm
        }
      }

    def writeField(recv: q.Term, m: StructMapping.Member[q.Term], value: Expr[Any]): Option[Expr[Unit]] =
      Option.when(fieldSymbol(recv, m.name).flags.is(q.Flags.Mutable)) {
        reachable(recv, m.name) match {
          case Some(accessible) => q.Assign(accessible, value.asTerm).asExprOf[Unit]
          case None =>
            '{ ${ fieldAccess(recv, m) }.set(${ recv.asExprOf[AnyRef] }, $value) }
        }
      }

    def dumpTarget(marker: String, buffer: Expr[ByteBuffer]) = '{
      if (sys.env.contains(Env.PolyregionDebug)) {
        val array = new Array[Byte]($buffer.remaining())
        $buffer.duplicate().get(array)
        println(s"${${ Expr(marker) }} target => 0x${array.map(byte => f"$byte%02x").mkString(" ")}")
      }
    }

    def elementSize(comp: p.Type): Int = comp match {
      case p.Type.Struct(name, _) if lut.contains(name) => layouts(lut(name))._1.size
      case _                                            => tpeAsRuntimeTpe(comp).sizeInBytes()
    }

    def writeArray[t: Type](expr: Expr[StdLib.MutableSeq[t]], comp: p.Type, m: Expr[Marshal]): Expr[Long] = {
      // Kernel ABI for arrays of structs: elements are stored INLINE (struct bytes packed into the
      // data buffer at `i * structSize`, no pointer indirection). The slot size must therefore
      // match the struct's layout size, not the pointer size.
      val elementSizeInBytes = elementSize(comp)
      if (elementSizeInBytes == 0) '{
        // Unit-element arrays carry no data — allocate a 1-byte placeholder so the kernel still
        // gets a valid pointer and we don't try to allocate or write into a 0-byte buffer.
        $m.stage(${ allocateBuffer('{ 1 }) })
      }
      else
        '{
          val arrBuffer = ${ allocateBuffer('{ ${ Expr(elementSizeInBytes) } * $expr.length_ }) }
          if (sys.env.contains(Env.PolyregionDebug))
            println(s"[marshal]: write array  [${$expr.length_} * ${${ Expr(comp.repr) }}] ${$expr.data} => $arrBuffer")
          var i = 0
          while (i < $expr.length_) {
            ${
              val elementOffset = '{ ${ Expr(elementSizeInBytes) } * i }
              comp match {
                case p.Type.Struct(name, _) if lut.contains(name) =>
                  // null elements (result holder pre-fill) stay zeroed
                  '{
                    val elem = $expr(i)
                    if (elem != null) ${ callFill(name, 'elem, m, 'arrBuffer, elementOffset) }
                    ()
                  }
                case t =>
                  writePrim('arrBuffer, elementOffset, t, '{ $expr(i) })
              }
            }
            i += 1
          }
          val arrPtr = $m.stage(arrBuffer)
          ${
            comp match {
              case p.Type.Struct(name, _) if lut.contains(name) =>
                // a later reference to an element resolves to its slot, but one captured earlier keeps its own copy
                '{
                  var j = 0
                  while (j < $expr.length_) {
                    val elem = $expr(j)
                    if (elem != null) $m.bind(elem, arrPtr + j.toLong * ${ Expr(elementSizeInBytes) })
                    j += 1
                  }
                }
              case _ => '{ () }
            }
          }
          arrPtr
        }
    }

    def isRoot(mapping: StructMapping[q.Term]): Boolean = mapping.tagOffset >= 0 && mapping.tag < 0

    case class Candidate(mapping: Option[StructMapping[q.Term]], repr: q.TypeRepr, tag: Int)

    def candidatesOf(mapping: StructMapping[q.Term]): List[Candidate] =
      reprs
        .get(mapping.source)
        .toList
        .flatMap(Dispatch.candidateReprs(_, tags))
        .map { (name, repr) =>
          val mapping = mappings.get(name)
          if (mapping.isEmpty && layouts.keys.exists(_.name == name))
            q.report.errorAndAbort(
              s"Candidate ${name.repr} has a layout but no mapping, its fields would not be written"
            )
          Candidate(mapping, repr, tags(name))
        }
        .sortBy(_.tag)

    def writeTag(buffer: Expr[ByteBuffer], base: Expr[Int], mapping: StructMapping[q.Term]): Expr[Unit] =
      if (mapping.tagOffset < 0) '{ () }
      else '{ $buffer.putInt($base + ${ Expr(mapping.tagOffset) }, ${ Expr(mapping.tag) }) }

    def fillMapping[t: Type](
        root: Expr[t],
        rootAfterPrism: Expr[t],
        m: Expr[Marshal],
        buffer: Expr[ByteBuffer],
        base: Expr[Int],
        mapping: StructMapping[q.Term]
    ): Expr[Unit] =
      if (!isRoot(mapping)) fillConcrete(root, rootAfterPrism, m, buffer, base, mapping)
      else
        candidatesOf(mapping).foldRight(writeTag(buffer, base, mapping)) { (candidate, otherwise) =>
          candidate.repr.asType match {
            case '[c] =>
              '{
                ($root: @unchecked) match {
                  case typed: c =>
                    ${
                      candidate.mapping.fold(writeTag(buffer, base, mapping.copy(tag = candidate.tag)))(
                        fillMapping[c]('typed, 'typed, m, buffer, base, _)
                      )
                    }
                  case _ => $otherwise
                }
              }
          }
        }

    def fillConcrete[t: Type](
        root: Expr[t],
        rootAfterPrism: Expr[t],
        m: Expr[Marshal],
        buffer: Expr[ByteBuffer],
        base: Expr[Int],
        mapping: StructMapping[q.Term]
    ): Expr[Unit] = Expr.block(
      mapping.members.map { member =>
        val memberOffset = '{ $base + ${ Expr(member.offsetInBytes.toInt) } }
        ((rootAfterPrism, member.tpe): @scala.unchecked) match {
          case ('{ $seq: StdLib.MutableSeq[t] }, p.Type.Ptr(comp, _)) =>
            val ptr = '{
              val resident = $residentPointer($root)
              if (resident != 0L) resident
              else {
                val known = $m.dataPointerOf($root, -1L)
                if (known != -1L) known
                else {
                  val staged = ${ writeArray[t](seq, comp, m) }
                  $m.bindData($root, staged)
                  staged
                }
              }
            }
            writePrim(buffer, memberOffset, p.Type.IntS64, ptr)
          case (_, p.Type.Ptr(_, _)) if planArray(rootAfterPrism.asTerm, mapping, member).isDefined =>
            val array = '{ ${ readField(rootAfterPrism.asTerm, member).asExpr }.asInstanceOf[AnyRef] }
            val where = Expr(s"member `${member.name}` of ${mapping.source.name.fqcn}")
            writePrim(
              buffer,
              memberOffset,
              p.Type.IntS64,
              '{ if ($array == null) 0L else $m.array($array, $where, "") }
            )
          case (_, p.Type.Struct(name, _)) if lut.get(name).exists(_.members.isEmpty) =>
            // zero-width slot, nothing to write
            '{ () }
          case (_, p.Type.Struct(name, _)) if lut.contains(name) =>
            // laid out inline in the parent, as the kernel reads it
            val child = readField(rootAfterPrism.asTerm, member)
            val fill  = callFill(name, child.asExpr, m, buffer, memberOffset)
            if (hasMutableField(child.tpe.widenTermRefByName)) {
              val where = Expr(s"member `${member.name}` of ${mapping.source.name.fqcn}")
              '{
                val nested = ${ child.asExprOf[AnyRef] }
                if (nested != null) $m.claimInline(nested, $where)
                $fill
              }
            } else fill
          case (_, p.Type.Struct(_, _)) =>
            // opaque to the kernel, no slot
            '{ () }
          case (_, p.Type.Ptr(_, _)) =>
            // not a MutableSeq prism target and unused after intrinsification
            '{ () }
          case (_, _) =>
            writePrim(buffer, memberOffset, member.tpe, readField(rootAfterPrism.asTerm, member).asExpr)
        }
      } :+ writeTag(buffer, base, mapping),
      '{ () }
    )

    def readArray[t: Type](
        seq: Expr[StdLib.MutableSeq[t]],
        comp: p.Type,
        m: Expr[Marshal],
        memberOffset: Expr[Int],
        buffer: Expr[ByteBuffer],
        mapping: StructMapping[q.Term]
    ): Expr[Unit] = {
      val elementSizeInBytes = elementSize(comp)
      if (elementSizeInBytes == 0) '{ () } // Unit-element array — nothing to read back.
      else
        '{

          val arrayLen = ${
            mapping.members.headOption match {
              case Some(lengthMember) if lengthMember.tpe == p.Type.IntS32 =>
                Pickler
                  .readPrim(buffer, Expr(lengthMember.offsetInBytes.toInt), p.Type.IntS32)
                  .asExprOf[Int]
              case _ =>
                q.report.errorAndAbort(s"Illegal structure while encoding read for member ${mapping} ")
            }
          }

          val arrPtr    = ${ readPrim(buffer, memberOffset, p.Type.IntS64).asExprOf[Long] }
          val arrBuffer = $m.host(arrPtr, ${ Expr(elementSizeInBytes) } * arrayLen)
          var i         = 0
          while (i < arrayLen) {
            $seq(i) = ${
              val elementOffset = '{ ${ Expr(elementSizeInBytes) } * i }
              comp match {
                case p.Type.Struct(name, _) if mappings.get(name).exists(isRoot(_)) =>
                  val structMapping = mappings(name)
                  '{
                    val existing = $seq(i)
                    Option(existing).foreach(e =>
                      ${
                        updateMapping[t]('e, '{ arrPtr + ${ elementOffset }.toLong }, m, structMapping)
                      }
                    )
                    existing
                  }
                case p.Type.Struct(name, _) if lut.contains(name) =>
                  // Element bytes live INLINE at arrBuffer[i*structSize]. Reconstruct the struct
                  // directly via readMapping using the inline element address — bypassing callRead
                  // since the result-holder slot was never written by the host (so the identity map
                  // would have no entry for it and readMethod would throw).
                  val structMapping = mappings(name)
                  '{
                    val elemPtr = arrPtr + ${ elementOffset }.toLong
                    ${ readMapping[t]('elemPtr, m, structMapping) }
                  }
                case t =>
                  readPrim('arrBuffer, elementOffset, t).asExprOf[t]
              }
            }
            i += 1
          }
          if (sys.env.contains(Env.PolyregionDebug))
            println(
              s"[marshal]: read array  [${arrayLen} * ${${ Expr(comp.repr) }}]  ${$seq} => $arrBuffer(0x${arrPtr.toHexString})"
            )
        }
    }

    def readBackArray(array: Expr[AnyRef], comp: p.Type, m: Expr[Marshal]): Expr[Unit] = comp match {
      case p.Type.Ptr(inner, _) =>
        '{ $array.asInstanceOf[Array[AnyRef]].foreach(row => ${ readBackArray('row, inner, m) }) }
      case _ => '{ $m.readBack($array) }
    }

    def updateMapping[t: Type](
        root: Expr[t],
        ptr: Expr[Long],
        m: Expr[Marshal],
        mapping: StructMapping[q.Term]
    ): Expr[Unit] =
      if (!isRoot(mapping)) updateConcrete(root, ptr, m, mapping)
      else
        candidatesOf(mapping).foldRight('{ () }) { (candidate, otherwise) =>
          candidate.repr.asType match {
            case '[c] =>
              candidate.mapping.fold(otherwise)(cm =>
                '{
                  ($root: @unchecked) match {
                    case typed: c => ${ updateMapping[c]('typed, ptr, m, cm) }
                    case _        => $otherwise
                  }
                }
              )
          }
        }

    def updateConcrete[t: Type](
        root: Expr[t],
        ptr: Expr[Long],
        m: Expr[Marshal],
        mapping: StructMapping[q.Term]
    ): Expr[Unit] = '{
      val buffer = $m.host($ptr, ${ Expr(mapping.sizeInBytes.toInt) })
      if (sys.env.contains(Env.PolyregionDebug))
        println(s"[marshal]: update object  ${$root} <- $buffer(0x${$ptr.toHexString})")

      ${ dumpTarget(">>>", 'buffer) }

      ${
        Expr.block(
          mapping.members.map { member =>
            val memberOffset = Expr(member.offsetInBytes.toInt)
            ((root, member.tpe): @scala.unchecked) match {
              case (_, p.Type.Ptr(_, _))
                  if mapping.write.isEmpty && planArray(root.asTerm, mapping, member).isDefined =>
                val array = '{ ${ readField(root.asTerm, member).asExpr }.asInstanceOf[AnyRef] }
                '{ if ($array != null) ${ readBackArray(array, planArray(root.asTerm, mapping, member).get, m) } }
              case (_, p.Type.Ptr(comp, _)) =>
                // Restore arrays only when root is a MutableSeq prism (parameterised type with
                // a write witness); other Ptr fields (e.g. opaque singleton members) skip.
                (
                  mapping.write,
                  root.asTerm.tpe.widenTermRefByName match {
                    case q.AppliedType(_, x :: Nil) => Some(x.asType)
                    case _                          => None
                  }
                ) match {
                  case (Some((_, write)), Some('[t])) =>
                    val seq = write(root.asTerm).asExprOf[MutableSeq[t]]
                    '{
                      if ($residentPointer($root) == 0L)
                        ${ readArray[t](seq, comp, m, memberOffset, 'buffer, mapping) }
                    }
                  case _ => '{ () }
                }

              case (_, p.Type.Struct(name, _)) if lut.get(name).exists(_.members.nonEmpty) =>
                // Inline-laid-out nested struct field: update the same instance in place from the parent's
                // bytes. A nested value is never reassigned, so only its mutable fields and arrays change.
                readField(root.asTerm, member).asExpr match {
                  case '{ $nested: u } =>
                    updateMapping[u](nested, '{ $ptr + ${ memberOffset }.toLong }, m, mappings(name))
                }
              case (_, p.Type.Struct(name, _)) if lut.contains(name) =>
                '{ () }
              case (_, p.Type.Struct(_, _)) =>
                // Opaque struct member — kernel doesn't write this back, skip update.
                '{ () }
              case (_, _) =>
                // Primitive field write-back: only assign when the field is actually mutable
                // (a `var`). Immutable `val` fields can't be reassigned and the kernel can't
                // mutate them either. Also skip prismed mappings — for those the member
                // names belong to the mirror class, not the host source type, so calling
                // `m.select(root)` on the source would fail to typecheck.
                if (mapping.write.isDefined) '{ () }
                else writeField(root.asTerm, member, readPrim('buffer, memberOffset, member.tpe)).getOrElse('{ () })
            }
          },
          '{ () }
        )
      }
    }

    def readMapping[t: Type](
        ptr: Expr[Long],
        m: Expr[Marshal],
        mapping: StructMapping[q.Term]
    ): Expr[t] = '{
      val buffer = $m.host($ptr, ${ Expr(mapping.sizeInBytes.toInt) })
      if (sys.env.contains(Env.PolyregionDebug))
        println(
          s"[marshal]: mk object ${${ Expr(q.TypeRepr.of[t].widenTermRefByName.show) }} <- $buffer(0x${$ptr.toHexString})"
        )
      ${

        // See if we have a ctor:
        debug(">> t= " + q.TypeRepr.of[t].widenTermRefByName.show)
        // For empty-member structs we use `Any` as the param type (see methodParamRepr); their
        // typeSymbol is Any, which has no usable primary ctor. Treat that as the no-ctor case
        // and just return null (the kernel doesn't read these back into a Scala value).
        val tSym       = q.TypeRepr.of[t].widenTermRefByName.typeSymbol
        val tIsAnyLike = tSym == q.defn.AnyClass || tSym == q.defn.ObjectClass || tSym.isNoSymbol

        val ownerSym = tSym.maybeOwner
        val outerLost =
          ownerSym.isClassDef && !ownerSym.flags.is(q.Flags.Module) &&
            !Retyper.ownersList(q.Symbol.spliceOwner).contains(ownerSym)

        if (tIsAnyLike) '{ null.asInstanceOf[t] }
        else if (outerLost)
          '{
            throw TwinException(
              ${
                Expr(s"Cannot rebuild an instance of the inner class ${tSym.fullName}, its outer instance is not known")
              }
            )
          }
        else
          q.TypeRepr.of[t].widenTermRefByName.typeSymbol.primaryConstructor.tree match {
            case q.DefDef("<init>", ps, _, None) if ps.forall {
                  case q.TypeParamClause(_) | q.TermParamClause(Nil) => true
                  case _                                             => false
                } => // default no-arg ctor
              // For traits we can't `new T()`. Trait-typed mirror structs (e.g. a `given Monoid[A]`
              // captured into the kernel as an opaque struct with no members) aren't modified by the
              // kernel — just return the captured root unchanged. For concrete classes with no
              // params, instantiate normally.
              val sym     = q.TypeRepr.of[t].widenTermRefByName.typeSymbol
              val isTrait = sym.flags.is(q.Flags.Trait)
              if (isTrait || mapping.members.isEmpty) {
                '{ null.asInstanceOf[t] }
              } else
                '{
                  val root = ${
                    q.Select
                      .unique(q.New(q.TypeIdent(q.TypeRepr.of[t].typeSymbol)), "<init>")
                      .appliedToArgs(Nil)
                      .asExpr
                  }.asInstanceOf[t]
                  ${ updateMapping('root, ptr, m, mapping) }
                  root
                }
            case q.DefDef("<init>", ps, _, None) =>
              // See if we can match up the type args and val args
              val typeDefs = ps.collect { case q.TypeParamClause(xs) => xs }.flatten
              val valDefs  = ps.collect { case q.TermParamClause(xs) => xs }.flatten

              debug(q.TypeRepr.of[t].widenTermRefByName.toString)

              val args = q.TypeRepr.of[t].widenTermRefByName match {
                case q.AppliedType(_, xs) => xs
                case _                    => Nil
              }

              def unconstructible(why: String): Expr[t] =
                '{
                  throw TwinException(
                    ${ Expr(s"Cannot rebuild an instance of ${tSym.fullName}, $why") }
                  )
                }

              if (typeDefs.size != args.size) unconstructible("its type parameters do not match the captured type")
              else if (valDefs.size != mapping.members.size)
                unconstructible("its fields are not all constructor parameters")
              else {

                val terms = mapping.members.map { member =>
                  val memberOffset = Expr(member.offsetInBytes.toInt)
                  member.tpe match {
                    case p.Type.Ptr(comp, _) =>
                      '{ ??? }.asTerm
                    case p.Type.Struct(name, _) if lut.contains(name) =>
                      // Nested struct field laid out inline — pass `parentPtr + memberOffset` as
                      // the struct pointer so callRead wraps the inline slice and reconstructs.
                      val nestedPtr = '{ $ptr + ${ memberOffset }.toLong }
                      callRead(name, '{ null }, nestedPtr, m).asTerm
                    case p.Type.Struct(_, _) =>
                      // Opaque struct field — synthesise null since we can't reconstruct it from kernel side.
                      '{ null }.asTerm
                    case _ => readPrim('buffer, memberOffset, member.tpe).asTerm
                  }
                }
                // Use asInstanceOf[t] (not asExprOf[t]) so we don't fail when `t` is a singleton type
                // (e.g. captured `Foo.FooConst.type`); the constructed `new Foo(...)` is structurally compatible.
                val ctorCall = q.Select
                  .unique(q.New(q.TypeIdent(q.TypeRepr.of[t].typeSymbol)), "<init>")
                  .appliedToTypes(args)
                  .appliedToArgs(terms)
                '{ ${ ctorCall.asExpr }.asInstanceOf[t] }
              }
            case _ => '{ throw TwinException(${ Expr(s"Cannot rebuild an instance of ${tSym.fullName}") }) }
          }
      }
    }

    def writeMethod(symbol: q.Symbol, mapping: StructMapping[q.Term]) = mkMethodDef(symbol) {
      case List(root: q.Term, m: q.Term) =>
        ((root.asExpr, m.asExpr): @unchecked) match {
          case ('{ $root: t }, '{ $m: Marshal }) =>
            '{
              val known = $m.pointerOf($root, -1L)
              if (known != -1L) known
              else if ($root == null) { $m.bind($root, 0L); 0L }
              else {
                $m.requireNotInlined($root)
                val buffer = ${ allocateBuffer(Expr(mapping.sizeInBytes.toInt)) }
                ${ callFill(mapping.source.name, root, m, 'buffer, '{ 0 }) }
                ${ dumpTarget("???", 'buffer) }
                val ptr = $m.stage(buffer)
                if (sys.env.contains(Env.PolyregionDebug))
                  println(s"[marshal]: object  ${$root} => $buffer(0x${ptr.toHexString})")
                $m.bind($root, ptr)
                ptr
              }
            }
        }
    }

    def fillMethod(symbol: q.Symbol, mapping: StructMapping[q.Term]) = mkMethodDef(symbol) {
      case List(root: q.Term, m: q.Term, buffer: q.Term, base: q.Term) =>
        (
          (
            root.asExpr,
            mapping.write.fold(root)((_, f) => f(root)).asExpr,
            m.asExpr,
            buffer.asExpr,
            base.asExpr
          ): @unchecked
        ) match {
          case (
                '{ $root: t },
                '{ $rootAfterPrismExpr: u },
                '{ $m: Marshal },
                '{ $buffer: ByteBuffer },
                '{ $base: Int }
              ) =>
            '{
              val rootAfterPrism = $rootAfterPrismExpr
              ${ fillMapping(root, 'rootAfterPrism, m, buffer, base, mapping) }
            }
        }
    }

    def readMethod(symbol: q.Symbol, mapping: StructMapping[q.Term]) = mkMethodDef(symbol) {
      case List(root: q.Term, ptr: q.Term, m: q.Term) =>
        ((root.asExpr, ptr.asExpr, m.asExpr): @unchecked) match {
          case ('{ $rootExpr: t }, '{ $ptrExpr: Long }, '{ $m: Marshal }) =>
            def readThroughPrism(readPtr: Expr[Long], write: q.Term => q.Term, read: q.Term => q.Term): Expr[Any] =
              // XXX To derive the type of the prism's mirror, we need to  reconstruct it from the mirror's symbol.
              // However, synthesising the TypeRepr for non-trivial types (e.g Seq[A], or (A,B)) is hard to get right.
              // In theory, something like  `q.TypeIdent(q.Symbol.requiredClass(mirror.struct.name.fqcn))` should work
              // but most often then not we get a symbol that exists but does not have a tree.

              // As a workaround to all this, we simply instantiate the write prism with a dummy term of the correct type
              // and then extract the return type that way. The instantiated tree will never spliced anywhere so it safe to do so.

              write(rootExpr.asTerm).tpe.widenTermRefByName.asType match {
                case '[t] =>
                  '{
                    val restored = ${ readMapping[t](readPtr, m, mapping) }
                    ${ read('restored.asTerm).asExpr }
                  }
              }
            '{
              val root: t  = $rootExpr
              val writePtr = $m.pointerOf(root, -1L)
              val readPtr  = $ptrExpr
              if (readPtr == 0L) null.asInstanceOf[t] // object reassignment for var to null
              else if (root == null) ${
                // a nested inline struct or a var that was null has no earlier object to update
                (mapping.write, mapping.read) match {
                  case (Some((_, write)), Some((_, read))) => readThroughPrism('readPtr, write, read)
                  case (_, None)                           => readMapping[t]('readPtr, m, mapping)
                  case _                                   => ???
                }
              }
              else if (writePtr == -1L) // object not previously written, fail
                throw RuntimeException(
                  s"Val root object $root was not previously written, cannot read from to 0x${readPtr.toHexString}"
                )
              else if (writePtr == readPtr) // same ptr, do the update
                ${
                  // If we have a prism, then a full read is required as we need a concrete instance.
                  (mapping.write, mapping.read) match {
                    case (Some((_, write)), Some((_, read))) => readThroughPrism('readPtr, write, read)
                    case (_, None) => '{ ${ updateMapping('root, 'readPtr, m, mapping) }; root }
                    case _         => ???
                  }
                }
              else { // object reassignment for var
                // Make sure we update the old writePtr (possibly orphaned, unless reassigned somewhere else) first.
                // This is to make sure modified object without a root (e.g through reassignment) is corrected updated.
                ${ callUpdate(mapping.source.name, 'root, 'writePtr, m) }
                // Now, readPtr is either a new allocation or a an existing one, possibly shared.
                // We check that it hasn't already been read/updated yet (the object may be recursive) and proceed to
                // create the object.
                $m.objectAt(readPtr) match {
                  case null => // not seen yet
                    ${
                      (mapping.write, mapping.read) match {
                        case (Some((_, write)), Some((_, read))) =>
                          debug(s"sss = ${rootExpr.asTerm.tpe.widenTermRefByName.show}")
                          debug(s"sss = ${write(rootExpr.asTerm).tpe.widenTermRefByName.show}")
                          readThroughPrism('readPtr, write, read)
                        case (_, None) => readMapping[t]('readPtr, m, mapping)
                        case _         => ???
                      }
                    }
                  case existing => existing.asInstanceOf[t] // Existing allocation found, use it.
                }
              }
            }
        }
    }

    def updateMethod(symbol: q.Symbol, mapping: StructMapping[q.Term]) = mkMethodDef(symbol) {
      case List(root: q.Term, ptr: q.Term, m: q.Term) =>
        ((root.asExpr, ptr.asExpr, m.asExpr): @unchecked) match {
          case ('{ $rootExpr: t }, '{ $ptrExpr: Long }, '{ $m: Marshal }) =>
            '{
              val root     = $rootExpr
              val writePtr = $m.pointerOf(root, -1L)
              val readPtr  = $ptrExpr
              if (writePtr == 0L && readPtr == 0L) () // was null, still null, no-op
              else if (writePtr == -1L)               // object not previously written, fail
                throw RuntimeException(
                  s"Val root object $root was not previously written, cannot update from to 0x${readPtr.toHexString}"
                )
              else if (writePtr == readPtr) { // same ptr, do the update
                ${ updateMapping('root, 'readPtr, m, mapping) }

                $m.bindObject(readPtr, root)
              } else // object reassignment for val, fail
                throw RuntimeException(
                  s"Cannot update immutable val, setting $root (0x${writePtr.toHexString}) to 0x${readPtr.toHexString}"
                )
              ()
            }
        }
    }

    val writeDefs  = writeSymbols.map((name, symbol) => writeMethod(symbol, mappings(name))).toList
    val fillDefs   = fillSymbols.map((name, symbol) => fillMethod(symbol, mappings(name))).toList
    val readDefs   = readSymbols.map((name, symbol) => readMethod(symbol, mappings(name))).toList
    val updateDefs = updateSymbols.map((name, symbol) => updateMethod(symbol, mappings(name))).toList

    val allDefs = writeDefs ::: fillDefs ::: readDefs ::: updateDefs

    (allDefs, callWrite, callRead, callUpdate)
  }

}
