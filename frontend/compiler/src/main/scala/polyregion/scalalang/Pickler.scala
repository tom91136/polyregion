package polyregion.scalalang

import cats.syntax.all.*
import polyregion.ast.{PolyAST as p, *}
import polyregion.jvm.{compiler as ct, runtime as rt}
import polyregion.prism.StdLib

import java.nio.{ByteBuffer, ByteOrder}
import scala.quoted.*
import polyregion.prism.StdLib.MutableSeq

object Pickler {

  def liftTpe(using q: Quotes)(t: p.Type) = t match {
    case p.Type.Bool1   => Type.of[Boolean]
    case p.Type.IntU16  => Type.of[Char]
    case p.Type.IntS8   => Type.of[Byte]
    case p.Type.IntS16  => Type.of[Short]
    case p.Type.IntS32  => Type.of[Int]
    case p.Type.IntS64  => Type.of[Long]
    case p.Type.Float32 => Type.of[Float]
    case p.Type.Float64 => Type.of[Double]
    case p.Type.Unit0   => Type.of[Unit]
    case illegal        => throw new RuntimeException(s"liftTpe: Cannot lift $illegal")
  }

  def tpeAsRuntimeTpe(t: p.Type): rt.Type = t match {
    case p.Type.Bool1              => rt.Type.BOOL
    case p.Type.IntS8              => rt.Type.BYTE
    case p.Type.IntU16             => rt.Type.CHAR
    case p.Type.IntS16             => rt.Type.SHORT
    case p.Type.IntS32             => rt.Type.INT
    case p.Type.IntS64             => rt.Type.LONG
    case p.Type.Float32            => rt.Type.FLOAT
    case p.Type.Float64            => rt.Type.DOUBLE
    case p.Type.Ptr(_, _, _)       => rt.Type.PTR
    case p.Type.Struct(_, _, _, _) => rt.Type.PTR
    case p.Type.Unit0              => rt.Type.VOID
    case illegal                   => throw new RuntimeException(s"tpeAsRuntimeTpe: Illegal $illegal")
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
    case p.Type.Unit0   => '{ $source.get($byteOffset); () }
    case x =>
      throw new RuntimeException(s"Cannot get ${x.repr} from buffer, it is not a primitive type")

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
    case p.Type.Unit0   => '{ $target.put($byteOffset, 0.toByte) }
    case x =>
      throw new RuntimeException(
        s"Cannot put ${x.repr} into buffer, it is not a primitive type (source is `${value.show}`)"
      )
  }

  private case class StructMapping[A](
      source: p.StructDef,
      sizeInBytes: Long,
      write: Option[(p.Mirror, A => A)],
      read: Option[(p.Mirror, A => A)],
      // update: Option[(A, A) => A],
      members: List[StructMapping.Member[A]]
  )
  private object StructMapping {
    case class Member[A](tpe: p.Type, sizeInBytes: Long, offsetInBytes: Long, select: A => A)
  }

  private def mkStructMapping(using q: Quoted)( //
      sdef: p.StructDef,
      layouts: Map[p.StructDef, (ct.Layout, Option[polyregion.prism.Prism])]
  ): StructMapping[q.Term] = {
    val (layout, maybePrism) = layouts
      .get(sdef)
      .getOrElse(q.report.errorAndAbort(s"Unseen sdef ${sdef.repr}, known reprs: ${layouts}"))
    val layoutTable = layout.members.map(m => m.name -> m).toMap
    val members = sdef.members.map { named =>
      val fieldName = named.symbol
      layoutTable
        .get(fieldName)
        .map(m =>
          StructMapping.Member(
            named.tpe,
            m.sizeInBytes,
            m.offsetInBytes,
            (recv: q.Term) => q.Select.unique(recv, fieldName): q.Term
          )
        )
        .getOrElse(q.report.errorAndAbort(s"Layout $layout is missing ${named.repr} from ${sdef.repr}"))
    }

    val (write, read) = maybePrism.map { case (mirror, (write, read, update)) =>
      (
        (mirror, (root: q.Term) => write(q.underlying, root.asExpr).asTerm),
        (mirror, (restored: q.Term) => read(q.underlying, restored.asExpr).asTerm)
      )
    }.unzip

    // val (write, read, update) = maybePrism match {
    //   case None =>
    //     (
    //       (root: q.Term) => root,
    //       (root: q.Term) => root,
    //       // (root: q.Term, restored: q.Term) => root
    //     )
    //   case Some((write, read, update)) =>
    //     (
    //       (root: q.Term) => write(q.underlying, root.asExpr).asTerm,
    //       (root: q.Term) => read(q.underlying, root.asExpr).asTerm,
    //       // (root: q.Term, restored: q.Term) => update(q.underlying, root.asExpr, restored.asExpr).asTerm
    //     )
    // }
    StructMapping(sdef, layout.sizeInBytes, write, read, members)
  }

  def deriveAllRepr(using q: Quoted)( //
      lut: Map[p.Sym, (p.StructDef, Option[polyregion.prism.Prism])],
      sdef: p.StructDef,
      repr: q.TypeRepr
  ): Map[p.StructDef, q.TypeRepr] = {
    def go(using q: Quoted)(sdef: p.StructDef, repr: q.TypeRepr, added: Set[p.Sym]): List[(p.StructDef, q.TypeRepr)] = {
      val added0 = added + sdef.name
      def descend(name: p.Sym, member: String): List[(p.StructDef, q.TypeRepr)] =
        if (added0.contains(name)) Nil
        else
          lut.get(name) match {
            case None               => Nil
            case Some((sdef, None)) => go(sdef, q.TermRef(repr, member), added0)
            case Some((sdef, Some((_, (from, to, x))))) =>
              given Quotes = q.underlying
              repr.widenTermRefByName.asType match {
                case '[t] =>
                  val prismRepr =
                    from(q.underlying, '{ scala.compiletime.uninitialized: t }).asTerm.tpe.widenTermRefByName
                  go(sdef, q.TermRef(prismRepr, member), added0)
              }
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
      (sdef, repr.widenTermRefByName) :: sdef.members.flatMap {
        case p.Named(member, p.Type.Struct(name, _, _, _))              => descend(name, member)
        case p.Named(_, p.Type.Ptr(p.Type.Struct(name, _, _, _), _, _)) => descendArrayElement(name)
        case _                                                          => Nil
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

  def generateAll(using q: Quoted)(
      lut: Map[p.Sym, p.StructDef],
      layouts: Map[p.StructDef, (ct.Layout, Option[polyregion.prism.Prism])],
      reprs: Map[p.StructDef, q.TypeRepr],
      pointerOfBuffer: Expr[ByteBuffer => Long],
      bufferOfPointer: Expr[(Long, Long) => ByteBuffer]
  ) = {
    given Quotes = q.underlying

    import polyregion.jvm.runtime.Platforms

    type PtrMapTpe = scala.collection.mutable.Map[Any, Long]
    type ObjMapTpe = scala.collection.mutable.Map[Long, Any]

    // For struct types that appear with multiple distinct generic instantiations in the same
    // capture set (e.g. `Monoid[Int]`, `Monoid[Float]`, `Monoid[Float2]` all sharing the polymorphic
    // `Monoid` StructDef), reprs only tracks ONE instantiation — whichever ended up last in the
    // merged map. Pinning the read/write method signatures to that one would reject the other
    // call sites at typecheck time. Widening to `Any` for empty-member structs (which are opaque
    // to the kernel anyway) lets all instantiations call the same method safely.
    def methodParamRepr(sdef: p.StructDef, repr: q.TypeRepr): q.TypeRepr =
      if (sdef.members.isEmpty) q.TypeRepr.of[Any] else repr
    // Index by struct name (p.Sym) rather than StructDef instance: callers' `lut(name)` can return
    // a StructDef whose `parents` (or other derived fields) differ from the one stored in `reprs`,
    // even when they refer to the same logical type — using the name as the key avoids that
    // identity mismatch.
    val writeSymbols: Map[p.Sym, q.Symbol] = reprs.toList
      .map((sdef, repr) =>
        sdef.name -> mkMethodSym(
          s"write_${sdef.name.repr}",
          q.TypeRepr.of[Long],
          "root"   -> methodParamRepr(sdef, repr),
          "ptrMap" -> q.TypeRepr.of[PtrMapTpe]
        )
      )
      .toMap

    val readSymbols: Map[p.Sym, q.Symbol] = reprs.toList
      .map((sdef, repr) =>
        sdef.name -> mkMethodSym(
          s"read_${sdef.name.repr}",
          methodParamRepr(sdef, repr),
          "root"   -> methodParamRepr(sdef, repr),
          "ptr"    -> q.TypeRepr.of[Long],
          "ptrMap" -> q.TypeRepr.of[PtrMapTpe],
          "objMap" -> q.TypeRepr.of[ObjMapTpe]
        )
      )
      .toMap

    val updateSymbols: Map[p.Sym, q.Symbol] = reprs.toList
      .map((sdef, repr) =>
        sdef.name -> mkMethodSym(
          s"update_${sdef.name.repr}",
          q.TypeRepr.of[Unit],
          "root"   -> methodParamRepr(sdef, repr),
          "ptr"    -> q.TypeRepr.of[Long],
          "ptrMap" -> q.TypeRepr.of[PtrMapTpe],
          "objMap" -> q.TypeRepr.of[ObjMapTpe]
        )
      )
      .toMap

    def allocateBuffer(size: Expr[Int]) = '{ ByteBuffer.allocateDirect($size).order(ByteOrder.nativeOrder()) }

    def callWrite(name: p.Sym, root: Expr[Any], ptrMap: Expr[PtrMapTpe]) =
      q.Apply(q.Ref(writeSymbols(name)), List(root.asTerm, ptrMap.asTerm)).asExprOf[Long]

    def callRead(name: p.Sym, root: Expr[Any], ptr: Expr[Long], ptrMap: Expr[PtrMapTpe], objMap: Expr[ObjMapTpe]) =
      q.Apply(q.Ref(readSymbols(name)), List(root.asTerm, ptr.asTerm, ptrMap.asTerm, objMap.asTerm)).asExpr

    def callUpdate(name: p.Sym, root: Expr[Any], ptr: Expr[Long], ptrMap: Expr[PtrMapTpe], objMap: Expr[ObjMapTpe]) =
      q.Apply(q.Ref(updateSymbols(name)), List(root.asTerm, ptr.asTerm, ptrMap.asTerm, objMap.asTerm))
        .asExprOf[Unit]

    // Moved up so writeArray/readArray can access struct mappings (for inline-struct array ABI).
    val mappings: Map[p.Sym, StructMapping[q.Term]] =
      reprs.keys.map(mkStructMapping(_, layouts)).map(m => m.source.name -> m).toMap

    def writeArray[t: Type](expr: Expr[StdLib.MutableSeq[t]], comp: p.Type, ptrMap: Expr[PtrMapTpe]): Expr[Long] = {
      // Kernel ABI for arrays of structs: elements are stored INLINE (struct bytes packed into the
      // data buffer at `i * structSize`, no pointer indirection). The slot size must therefore
      // match the struct's layout size, not the pointer size.
      val elementSizeInBytes = comp match {
        case p.Type.Struct(name, _, _, _) if lut.contains(name) =>
          layouts(lut(name))._1.sizeInBytes.toInt
        case _ => tpeAsRuntimeTpe(comp).sizeInBytes()
      }
      if (elementSizeInBytes == 0) '{
        // Unit-element arrays carry no data — allocate a 1-byte placeholder so the kernel still
        // gets a valid pointer and we don't try to allocate or write into a 0-byte buffer.
        val arrBuffer = ${ allocateBuffer('{ 1 }) }
        $pointerOfBuffer(arrBuffer)
      }
      else
        '{
          val arrBuffer = ${ allocateBuffer('{ ${ Expr(elementSizeInBytes) } * $expr.length_ }) }
          val arrPtr    = $pointerOfBuffer(arrBuffer)
          println(
            s"[bind]: write array  [${$expr.length_} * ${${ Expr(comp.repr) }}] ${$expr.data} => $arrBuffer(0x${arrPtr.toHexString})"
          )
          var i = 0
          while (i < $expr.length_) {
            ${
              val elementOffset = '{ ${ Expr(elementSizeInBytes) } * i }
              comp match {
                case p.Type.Struct(name, _, _, _) if lut.contains(name) =>
                  // Inline struct slot — the kernel reads/writes struct bytes directly at
                  // arrBuffer[i*structSize..(i+1)*structSize] (no pointer indirection). For non-null
                  // inputs we serialise via callWrite (which yields a struct-sized buffer pointer)
                  // and memcpy those bytes inline. For null inputs (result-holder pre-fill, see
                  // BaseSuite.offload1) we leave the slot zeroed since the kernel will overwrite it.
                  '{
                    val elem = $expr(i)
                    if (elem != null) {
                      val structPtr = ${ callWrite(name, 'elem, ptrMap) }
                      val srcBuf    = $bufferOfPointer(structPtr, ${ Expr(elementSizeInBytes) }.toLong)
                      val tmp       = new Array[Byte](${ Expr(elementSizeInBytes) })
                      srcBuf.duplicate().get(tmp)
                      val dst = arrBuffer.duplicate()
                      dst.position($elementOffset)
                      dst.put(tmp)
                    }
                    ()
                  }
                case t =>
                  writePrim('arrBuffer, elementOffset, t, '{ $expr(i) })
              }
            }
            i += 1
          }
          arrPtr
        }
    }

    def writeMapping[t: Type](
        root: Expr[t],
        rootAfterPrism: Expr[t],
        ptrMap: Expr[PtrMapTpe],
        mapping: StructMapping[q.Term]
    ) = '{
      val buffer = ${ allocateBuffer(Expr(mapping.sizeInBytes.toInt)) }
      val ptr    = $pointerOfBuffer(buffer)
      println(s"[bind]: object  ${$root}(prism=${$rootAfterPrism}) => $buffer(0x${ptr.toHexString})")
      ${
        Varargs(mapping.members.map { m =>
          val memberOffset = Expr(m.offsetInBytes.toInt)
          (rootAfterPrism, m.tpe) match {
            case ('{ $seq: StdLib.MutableSeq[t] }, p.Type.Ptr(comp, _, _)) =>
              val ptr = writeArray[t](seq, comp, ptrMap)
              writePrim('buffer, memberOffset, p.Type.IntS64, ptr)
            case (_, p.Type.Struct(name, _, _, _)) if lut.get(name).exists(_.members.isEmpty) =>
              // Empty-member struct field (e.g. trait-based mirrors like `Monoid` whose
              // dispatch is resolved without reading the field). The backend lays these out
              // inline as 0-byte slots, so there's nothing to write — and trying to put an
              // 8-byte pointer at the slot's offset would overflow the parent buffer when the
              // parent is composed of nothing but such slots.
              '{ () }
            case (_, p.Type.Struct(name, _, _, _)) if lut.contains(name) =>
              // Nested struct field — laid out INLINE in the parent buffer at memberOffset
              // (matches the kernel's `functionBoundary=false` member layout). callWrite
              // produces a struct-sized buffer; memcpy its bytes into the parent slot.
              val nestedSize = layouts(lut(name))._1.sizeInBytes.toInt
              '{
                val nestedPtr = ${ callWrite(name, m.select(rootAfterPrism.asTerm).asExpr, ptrMap) }
                val srcBuf    = $bufferOfPointer(nestedPtr, ${ Expr(nestedSize) }.toLong)
                val tmp       = new Array[Byte](${ Expr(nestedSize) })
                srcBuf.duplicate().get(tmp)
                val dst = buffer.duplicate()
                dst.position($memberOffset)
                dst.put(tmp)
                ()
              }
            case (_, p.Type.Struct(_, _, _, _)) =>
              // Opaque struct member (e.g. scala.Option) — kernel doesn't access this field;
              // skip (the inline layout has no slot for it anyway).
              '{ () }
            case (_, _) =>
              writePrim('buffer, memberOffset, m.tpe, m.select(rootAfterPrism.asTerm).asExpr)
          }
        })
      }

      def toByteArray(buffer: ByteBuffer): Array[Byte] = {
        val array = new Array[Byte](buffer.remaining())
        buffer.duplicate().get(array)
        array
      }

      println(s"??? target => 0x${toByteArray(buffer).map(byte => f"$byte%02x").mkString(" ")}")
      $ptrMap += ($root -> ptr)
      ptr
    }

    def readArray[t: Type](
        seq: Expr[StdLib.MutableSeq[t]],
        comp: p.Type,
        ptrMap: Expr[PtrMapTpe],
        objMap: Expr[ObjMapTpe],
        memberOffset: Expr[Int],
        buffer: Expr[ByteBuffer],
        mapping: StructMapping[q.Term]
    ): Expr[Unit] = {
      // Mirror the inline-struct slot sizing used in writeArray.
      val elementSizeInBytes = comp match {
        case p.Type.Struct(name, _, _, _) if lut.contains(name) =>
          layouts(lut(name))._1.sizeInBytes.toInt
        case _ => tpeAsRuntimeTpe(comp).sizeInBytes()
      }
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
          val arrBuffer = $bufferOfPointer(arrPtr, ${ Expr(elementSizeInBytes) } * arrayLen)
          var i         = 0
          while (i < arrayLen) {
            $seq(i) = ${
              val elementOffset = '{ ${ Expr(elementSizeInBytes) } * i }
              comp match {
                case p.Type.Struct(name, _, _, _) if lut.contains(name) =>
                  // Element bytes live INLINE at arrBuffer[i*structSize]. Reconstruct the struct
                  // directly via readMapping using the inline element address — bypassing callRead
                  // since the result-holder slot was never written by the host (so ptrMap.get(root)
                  // would be None and readMethod would throw).
                  val structMapping = mappings(name)
                  '{
                    val elemPtr = arrPtr + ${ elementOffset }.toLong
                    ${ readMapping[t]('elemPtr, ptrMap, objMap, structMapping) }
                  }
                case t =>
                  readPrim('arrBuffer, elementOffset, t).asExprOf[t]
              }
            }
            i += 1
          }
          println(
            s"[bind]: read array  [${arrayLen} * ${${ Expr(comp.repr) }}]  ${$seq} => $arrBuffer(0x${arrPtr.toHexString})"
          )
        }
    }

    def updateMapping[t: Type](
        root: Expr[t],
        ptr: Expr[Long],
        ptrMap: Expr[PtrMapTpe],
        objMap: Expr[ObjMapTpe],
        mapping: StructMapping[q.Term]
    ) = '{
      val buffer = $bufferOfPointer($ptr, ${ Expr(mapping.sizeInBytes.toInt) })
      println(s"[bind]: update object  ${$root} <- $buffer(0x${$ptr.toHexString})")

      def toByteArray(buffer: ByteBuffer): Array[Byte] = {
        val array = new Array[Byte](buffer.remaining())
        buffer.duplicate().get(array)
        array
      }

      println(s">>> target => 0x${toByteArray(buffer).map(byte => f"$byte%02x").mkString(" ")}")

      ${
        Varargs(
          mapping.members.map { m =>
            val memberOffset = Expr(m.offsetInBytes.toInt)
            (root, m.tpe) match {
              case (_, p.Type.Ptr(comp, _, _)) =>
                (
                  mapping.write,
                  root.asTerm.tpe.widenTermRefByName match {
                    case q.AppliedType(_, x :: Nil) => x.asType
                    case t =>
                      q.report.errorAndAbort(
                        s"Unexpected type while matching on arrays ${t.show}, the correct shape is F[t]"
                      )
                  }
                ) match {
                  case (Some((_, write)), '[t]) =>
                    val seq = write(root.asTerm).asExprOf[MutableSeq[t]]
                    readArray[t](seq, comp, ptrMap, objMap, memberOffset, 'buffer, mapping)
                  case (_, _) =>
                    q.report.errorAndAbort("Missing write prism for array type, something isn't right here")
                }

              case (_, p.Type.Struct(name, _, _, _)) if lut.contains(name) =>
                // Inline-laid-out nested struct field. Both empty- and non-empty-member cases
                // resolve to a no-op here: case class fields are vals (immutable) so the kernel
                // can't mutate them, and any same-instance update would be a fresh allocation
                // we'd have no way to splice back into the host's owning struct anyway. Read-only
                // captures stay consistent with the host because we never overwrite their fields.
                '{ () }
              case (_, p.Type.Struct(_, _, _, _)) =>
                // Opaque struct member — kernel doesn't write this back, skip update.
                '{ () }
              case (_, _) =>
                // Primitive field write-back: only assign when the field is actually mutable
                // (a `var`). Immutable `val` fields can't be reassigned and the kernel can't
                // mutate them either. Also skip prismed mappings — for those the member
                // names belong to the mirror class, not the host source type, so calling
                // `m.select(root)` on the source would fail to typecheck.
                if (mapping.write.isDefined) '{ () }
                else {
                  val fieldSel = m.select(root.asTerm)
                  if (fieldSel.symbol.flags.is(q.Flags.Mutable))
                    q.Assign(fieldSel, readPrim('buffer, memberOffset, m.tpe).asTerm).asExprOf[Unit]
                  else '{ () }
                }
            }
          }
        )
      }
    }

    def readMapping[t: Type](
        ptr: Expr[Long],
        ptrMap: Expr[PtrMapTpe],
        objMap: Expr[ObjMapTpe],
        mapping: StructMapping[q.Term]
    ): Expr[t] = '{
      val buffer = $bufferOfPointer($ptr, ${ Expr(mapping.sizeInBytes.toInt) })
      println(
        s"[bind]: mk object ${${ Expr(q.TypeRepr.of[t].widenTermRefByName.show) }} <- $buffer(0x${$ptr.toHexString})"
      )
      ${

        // See if we have a ctor:
        println(">> t= " + q.TypeRepr.of[t].widenTermRefByName.show)
        // For empty-member structs we use `Any` as the param type (see methodParamRepr); their
        // typeSymbol is Any, which has no usable primary ctor. Treat that as the no-ctor case
        // and just return null (the kernel doesn't read these back into a Scala value).
        val tSym       = q.TypeRepr.of[t].widenTermRefByName.typeSymbol
        val tIsAnyLike = tSym == q.defn.AnyClass || tSym == q.defn.ObjectClass || tSym.isNoSymbol

        if (tIsAnyLike) '{ null.asInstanceOf[t] }
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
                  ${ updateMapping('root, ptr, ptrMap, objMap, mapping) }
                  root
                }
            case q.DefDef("<init>", ps, _, None) =>
              // See if we can match up the type args and val args
              val typeDefs = ps.collect { case q.TypeParamClause(xs) => xs }.flatten
              val valDefs  = ps.collect { case q.TermParamClause(xs) => xs }.flatten

              println(q.TypeRepr.of[t].widenTermRefByName)

              val args = q.TypeRepr.of[t].widenTermRefByName match {
                case q.AppliedType(_, xs) => xs
                case _                    => Nil
              }

              if (typeDefs.size != args.size) {
                ???
              }
              if (valDefs.size != mapping.members.size) {
                ???
              }

              val terms = mapping.members.map { m =>
                val memberOffset = Expr(m.offsetInBytes.toInt)
                m.tpe match {
                  case p.Type.Ptr(comp, _, _) =>
                    // readArray[t](seq, comp, ptrMap, objMap, memberOffset, 'buffer, mapping)
                    '{ ??? }.asTerm
                  case p.Type.Struct(name, _, _, _) if lut.contains(name) =>
                    // Nested struct field laid out inline — pass `parentPtr + memberOffset` as
                    // the struct pointer so callRead wraps the inline slice and reconstructs.
                    val nestedPtr = '{ $ptr + ${ memberOffset }.toLong }
                    callRead(name, '{ null }, nestedPtr, ptrMap, objMap).asTerm
                  case p.Type.Struct(_, _, _, _) =>
                    // Opaque struct field — synthesise null since we can't reconstruct it from kernel side.
                    '{ null }.asTerm
                  case _ => readPrim('buffer, memberOffset, m.tpe).asTerm
                }
              }
              // Use asInstanceOf[t] (not asExprOf[t]) so we don't fail when `t` is a singleton type
              // (e.g. captured `Foo.FooConst.type`); the constructed `new Foo(...)` is structurally compatible.
              val ctorCall = q.Select
                .unique(q.New(q.TypeIdent(q.TypeRepr.of[t].typeSymbol)), "<init>")
                .appliedToTypes(args)
                .appliedToArgs(terms)
              '{ ${ ctorCall.asExpr }.asInstanceOf[t] }
            case _ => ???
          }

        // println("ctor = " + q.TypeRepr.of[t].typeSymbol.primaryConstructor.tree)

      }
    }

    def writeMethod(symbol: q.Symbol, mapping: StructMapping[q.Term]) = mkMethodDef(symbol) {
      case List(root: q.Term, ptrMap: q.Term) =>
        ((root.asExpr, mapping.write.fold(root)((_, f) => f(root)).asExpr, ptrMap.asExpr): @unchecked) match {
          case ('{ $root: t }, '{ $rootAfterPrismExpr: u }, '{ $ptrMap: PtrMapTpe }) =>
            '{
              $ptrMap.get($root) match {
                case Some(existing)        => existing
                case None if $root == null => $ptrMap += ($root -> 0); 0
                case None =>
                  val rootAfterPrism = $rootAfterPrismExpr
                  ${ writeMapping(root, 'rootAfterPrism, ptrMap, mapping) }
              }
            }
        }
    }

    def readMethod(symbol: q.Symbol, mapping: StructMapping[q.Term]) = mkMethodDef(symbol) {
      case List(root: q.Term, ptr: q.Term, ptrMap: q.Term, objMap: q.Term) =>
        ((root.asExpr, ptr.asExpr, ptrMap.asExpr, objMap.asExpr): @unchecked) match {
          case ('{ $rootExpr: t }, '{ $ptrExpr: Long }, '{ $ptrMap: PtrMapTpe }, '{ $objMap: ObjMapTpe }) =>
            '{
              val root: t = $rootExpr
              ($ptrMap.get(root), $ptrExpr) match {
                case (_, 0) => null.asInstanceOf[t] // object reassignment for var to null
                case (Some(writePtr), readPtr) if writePtr == readPtr => // same ptr, do the update
                  ${
                    // If we have a prism, then a full read is required as we need a concrete instance.
                    (mapping.write, mapping.read) match {
                      case (Some((_, write)), Some((mirror, read))) =>
                        // XXX To derive the type of the prism's mirror, we need to  reconstruct it from the mirror's symbol.
                        // However, synthesising the TypeRepr for non-trivial types (e.g Seq[A], or (A,B)) is hard to get right.
                        // In theory, something like  `q.TypeIdent(q.Symbol.requiredClass(mirror.struct.name.repr))` should work
                        // but most often then not we get a symbol that exists but does not have a tree.

                        // As a workaround to all this, we simply instantiate the write prism with a dummy term of the correct type
                        // and then extract the return type that way. The instantiated tree will never spliced anywhere so it safe to do so.

                        write(rootExpr.asTerm).tpe.widenTermRefByName.asType match {
                          case '[t] =>
                            '{
                              val restored = ${ readMapping[t]('readPtr, ptrMap, objMap, mapping) }
                              ${ read('restored.asTerm).asExpr }
                            }
                        }
                      case (_, None) => '{ ${ updateMapping('root, 'readPtr, ptrMap, objMap, mapping) }; root }
                      case _         => ???
                    }
                  }
                case (Some(writePtr), readPtr) => // object reassignment for var
                  // Make sure we update the old writePtr (possibly orphaned, unless reassigned somewhere else) first.
                  // This is to make sure modified object without a root (e.g through reassignment) is corrected updated.
                  ${ callUpdate(mapping.source.name, 'root, 'writePtr, ptrMap, objMap) }
                  // Now, readPtr is either a new allocation or a an existing one, possibly shared.
                  // We check that it hasn't already been read/updated yet (the object may be recursive) and proceed to
                  // create the object.
                  $objMap.get(readPtr) match {
                    case Some(existing) => existing.asInstanceOf[t] // Existing allocation found, use it.
                    case None =>
                      ${
                        (mapping.write, mapping.read) match {
                          case (Some((_, write)), Some((mirror, read))) =>
                            // XXX To derive the type of the prism's mirror, we need to reconstruct it from the mirror's symbol.
                            // However, synthesising the TypeRepr for non-trivial types (e.g Seq[A], or (A,B)) is hard to get right.
                            // In theory, something like  `q.TypeIdent(q.Symbol.requiredClass(mirror.struct.name.repr))` should work
                            // but most often then not we get a symbol that exists but does not have a tree.

                            // As a workaround to all this, we simply instantiate the write prism with a dummy term of the correct type
                            // and then extract the return type that way. The instantiated tree will never spliced anywhere so it safe to do so.

                            println(s"sss = ${rootExpr.asTerm.tpe.widenTermRefByName.show}")
                            println(s"sss = ${write(rootExpr.asTerm).tpe.widenTermRefByName.show}")
                            write(rootExpr.asTerm).tpe.widenTermRefByName.asType match {
                              case '[t] =>
                                '{
                                  val restored = ${ readMapping[t]('readPtr, ptrMap, objMap, mapping) }
                                  ${ read('restored.asTerm).asExpr }
                                }
                            }
                          case (_, None) => readMapping[t]('readPtr, ptrMap, objMap, mapping)
                          case _         => ???
                        }
                      }

                    // ${ readMapping[t]('readPtr, ptrMap, objMap, mapping) }
                  }
                case (None, readPtr) => // object not previously written, fail
                  throw new RuntimeException(
                    s"Val root object ${root} was not previously written, cannot read from to 0x${readPtr.toHexString}"
                  )
              }
            }
        }
    }
    def updateMethod(symbol: q.Symbol, mapping: StructMapping[q.Term]) = mkMethodDef(symbol) {
      case List(root: q.Term, ptr: q.Term, ptrMap: q.Term, objMap: q.Term) =>
        ((root.asExpr, ptr.asExpr, ptrMap.asExpr, objMap.asExpr): @unchecked) match {
          case ('{ $rootExpr: t }, '{ $ptrExpr: Long }, '{ $ptrMap: PtrMapTpe }, '{ $objMap: ObjMapTpe }) =>
            '{
              val root = $rootExpr
              ($ptrMap.get(root), $ptrExpr) match {
                case (Some(0), 0) => () // was null, still null, no-op
                case (Some(writePtr), readPtr) if writePtr == readPtr => // same ptr, do the update
                  ${ updateMapping('root, 'readPtr, ptrMap, objMap, mapping) }

                  $objMap += (readPtr -> root)
                case (Some(writePtr), readPtr) => // object reassignment for val, fail
                  throw new RuntimeException(
                    s"Cannot update immutable val, setting ${root} (0x${writePtr.toHexString}) to 0x${readPtr.toHexString}"
                  )
                case (None, readPtr) => // object not previously written, fail
                  throw new RuntimeException(
                    s"Val root object ${root} was not previously written, cannot update from to 0x${readPtr.toHexString}"
                  )
              }
              ()
            }
        }
    }

    val writeDefs  = writeSymbols.map((name, symbol) => writeMethod(symbol, mappings(name))).toList
    val readDefs   = readSymbols.map((name, symbol) => readMethod(symbol, mappings(name))).toList
    val updateDefs = updateSymbols.map((name, symbol) => updateMethod(symbol, mappings(name))).toList

    val allDefs = writeDefs ::: readDefs ::: updateDefs

    (allDefs, callWrite, callRead, callUpdate)
  }

}
