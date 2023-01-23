package polyregion.scalalang

import cats.syntax.all.*
import polyregion.ast.{PolyAst as p, *}
import polyregion.jvm.{compiler as ct, runtime as rt}
import polyregion.prism.StdLib

import java.nio.{ByteBuffer, ByteOrder}
import scala.quoted.*
import polyregion.prism.StdLib.MutableSeq

object Pickler {

  def liftTpe(using q: Quotes)(t: p.Type) = t match {
    case p.Type.Bool   => Type.of[Boolean]
    case p.Type.Char   => Type.of[Char]
    case p.Type.Byte   => Type.of[Byte]
    case p.Type.Short  => Type.of[Short]
    case p.Type.Int    => Type.of[Int]
    case p.Type.Long   => Type.of[Long]
    case p.Type.Float  => Type.of[Float]
    case p.Type.Double => Type.of[Double]
    case p.Type.Unit   => Type.of[Unit]
    case illegal       => throw new RuntimeException(s"liftTpe: Cannot lift $illegal")
  }

  def tpeAsRuntimeTpe(t: p.Type): rt.Type = t match {
    case p.Type.Bool               => rt.Type.BOOL
    case p.Type.Byte               => rt.Type.BYTE
    case p.Type.Char               => rt.Type.CHAR
    case p.Type.Short              => rt.Type.SHORT
    case p.Type.Int                => rt.Type.INT
    case p.Type.Long               => rt.Type.LONG
    case p.Type.Float              => rt.Type.FLOAT
    case p.Type.Double             => rt.Type.DOUBLE
    case p.Type.Array(_)           => rt.Type.PTR
    case p.Type.Struct(_, _, _, _) => rt.Type.PTR
    case p.Type.Unit               => rt.Type.VOID
    case illegal                   => throw new RuntimeException(s"tpeAsRuntimeTpe: Illegal $illegal")
  }

  def readPrim(using q: Quotes) //
  (source: Expr[java.nio.ByteBuffer], byteOffset: Expr[Int], tpe: p.Type): Expr[Any] = tpe match {
    case p.Type.Float  => '{ $source.getFloat($byteOffset) }
    case p.Type.Double => '{ $source.getDouble($byteOffset) }
    case p.Type.Bool   => '{ if ($source.get($byteOffset) == 0) false else true }
    case p.Type.Byte   => '{ $source.get($byteOffset) }
    case p.Type.Char   => '{ $source.getChar($byteOffset) }
    case p.Type.Short  => '{ $source.getShort($byteOffset) }
    case p.Type.Int    => '{ $source.getInt($byteOffset) }
    case p.Type.Long   => '{ $source.getLong($byteOffset) }
    case p.Type.Unit   => '{ $source.get($byteOffset); () }
    case x =>
      throw new RuntimeException(s"Cannot get ${x.repr} from buffer, it is not a primitive type")

  }

  def writePrim(using q: Quotes) //
  (target: Expr[java.nio.ByteBuffer], byteOffset: Expr[Int], tpe: p.Type, value: Expr[Any]): Expr[Unit] = tpe match {
    case p.Type.Float  => '{ $target.putFloat($byteOffset, ${ value.asExprOf[Float] }) }
    case p.Type.Double => '{ $target.putDouble($byteOffset, ${ value.asExprOf[Double] }) }
    case p.Type.Bool   => '{ $target.put($byteOffset, if (!${ value.asExprOf[Boolean] }) 0.toByte else 1.toByte) }
    case p.Type.Byte   => '{ $target.put($byteOffset, ${ value.asExprOf[Byte] }) }
    case p.Type.Char   => '{ $target.putChar($byteOffset, ${ value.asExprOf[Char] }) }
    case p.Type.Short  => '{ $target.putShort($byteOffset, ${ value.asExprOf[Short] }) }
    case p.Type.Int    => '{ $target.putInt($byteOffset, ${ value.asExprOf[Int] }) }
    case p.Type.Long   => '{ $target.putLong($byteOffset, ${ value.asExprOf[Long] }) }
    case p.Type.Unit   => '{ $target.put($byteOffset, 0.toByte) }
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
    case class Member[A](tpe: p.Type, mut: Boolean, sizeInBytes: Long, offsetInBytes: Long, select: A => A)
  }

  private def mkStructMapping(using q: Quoted)( //
      sdef: p.StructDef,
      layouts: Map[p.StructDef, (ct.Layout, Option[polyregion.prism.Prism])]
  ): StructMapping[q.Term] = {
    val (layout, maybePrism) = layouts
      .get(sdef)
      .getOrElse(q.report.errorAndAbort(s"Unseen sdef ${sdef.repr}, known reprs: ${layouts}"))
    val layoutTable = layout.members.map(m => m.name -> m).toMap
    val members = sdef.members.map { case p.StructMember(named, mut) =>
      layoutTable
        .get(named.symbol)
        .map(m =>
          StructMapping.Member(named.tpe, mut, m.sizeInBytes, m.offsetInBytes, q.Select.unique(_, named.symbol))
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
      (sdef, repr.widenTermRefByName) :: sdef.members.flatMap(
        _.named match {
          case (p.Named(_, p.Type.Struct(name, _, _,_))) if added0.contains(name) => Nil
          case (p.Named(member, p.Type.Struct(name, _, _,_))) =>
            lut(name) match {
              case (sdef, None)                     => go(sdef, q.TermRef(repr, member), added0)
              case (sdef, Some((_, (from, to, x)))) =>
                // If we have a prism of struct def that has a nested structs, apply the prism then find out the type repr.
                // This is required because the mirrored struct will almost certainly have a different layout compare the source;
                // the field names and type will not match.
                given Quotes = q.underlying
                repr.widenTermRefByName.asType match {
                  case '[t] =>
                    val prismRepr =
                      from(q.underlying, '{ scala.compiletime.uninitialized: t }).asTerm.tpe.widenTermRefByName
                    go(sdef, q.TermRef(prismRepr, member), added0)
                }
            }
          case _ => Nil
        }
      )
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

    val writeSymbols = reprs.map((sdef, repr) =>
      sdef -> mkMethodSym(
        s"write_${sdef.name.repr}",
        q.TypeRepr.of[Long],
        "root"   -> repr,
        "ptrMap" -> q.TypeRepr.of[PtrMapTpe]
      )
    )

    val readSymbols = reprs.map((sdef, repr) =>
      sdef -> mkMethodSym(
        s"read_${sdef.name.repr}",
        repr,
        "root"   -> repr,
        "ptr"    -> q.TypeRepr.of[Long],
        "ptrMap" -> q.TypeRepr.of[PtrMapTpe],
        "objMap" -> q.TypeRepr.of[ObjMapTpe]
      )
    )

    val updateSymbols = reprs.map((sdef, repr) =>
      sdef -> mkMethodSym(
        s"update_${sdef.name.repr}",
        q.TypeRepr.of[Unit],
        "root"   -> repr,
        "ptr"    -> q.TypeRepr.of[Long],
        "ptrMap" -> q.TypeRepr.of[PtrMapTpe],
        "objMap" -> q.TypeRepr.of[ObjMapTpe]
      )
    )

    def allocateBuffer(size: Expr[Int]) = '{ ByteBuffer.allocateDirect($size).order(ByteOrder.nativeOrder()) }

    def callWrite(name: p.Sym, root: Expr[Any], ptrMap: Expr[PtrMapTpe]) =
      q.Apply(q.Ref(writeSymbols(lut(name))), List(root.asTerm, ptrMap.asTerm)).asExprOf[Long]

    def callRead(name: p.Sym, root: Expr[Any], ptr: Expr[Long], ptrMap: Expr[PtrMapTpe], objMap: Expr[ObjMapTpe]) =
      q.Apply(q.Ref(readSymbols(lut(name))), List(root.asTerm, ptr.asTerm, ptrMap.asTerm, objMap.asTerm)).asExpr

    def callUpdate(name: p.Sym, root: Expr[Any], ptr: Expr[Long], ptrMap: Expr[PtrMapTpe], objMap: Expr[ObjMapTpe]) =
      q.Apply(q.Ref(updateSymbols(lut(name))), List(root.asTerm, ptr.asTerm, ptrMap.asTerm, objMap.asTerm))
        .asExprOf[Unit]

    def writeArray[t: Type](expr: Expr[StdLib.MutableSeq[t]], comp: p.Type, ptrMap: Expr[PtrMapTpe]): Expr[Long] = {
      val elementSizeInBytes = tpeAsRuntimeTpe(comp).sizeInBytes()
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
              case p.Type.Struct(name, _, _,_) =>
                val ptr = callWrite(name, '{ $expr(i) }, ptrMap)
                writePrim('arrBuffer, elementOffset, p.Type.Long, ptr)
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
            case ('{ $seq: StdLib.MutableSeq[t] }, p.Type.Array(comp)) =>
              val ptr = writeArray[t](seq, comp, ptrMap)
              writePrim('buffer, memberOffset, p.Type.Long, ptr)
            case (_, p.Type.Struct(name, _, _,_)) =>
              val ptr = callWrite(name, m.select(rootAfterPrism.asTerm).asExpr, ptrMap)
              writePrim('buffer, memberOffset, p.Type.Long, ptr)
            case (_, _) =>
              writePrim('buffer, memberOffset, m.tpe, m.select(rootAfterPrism.asTerm).asExpr)
          }
        })
      }
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
      val elementSizeInBytes = tpeAsRuntimeTpe(comp).sizeInBytes()
      '{

        val arrayLen = ${
          mapping.members.headOption match {
            case Some(lengthMember) if lengthMember.tpe == p.Type.Int =>
              Pickler
                .readPrim(buffer, Expr(lengthMember.offsetInBytes.toInt), p.Type.Int)
                .asExprOf[Int]
            case _ =>
              q.report.errorAndAbort(s"Illegal structure while encoding read for member ${mapping} ")
          }
        }

        val arrPtr    = ${ readPrim(buffer, memberOffset, p.Type.Long).asExprOf[Long] }
        val arrBuffer = $bufferOfPointer(arrPtr, ${ Expr(elementSizeInBytes) } * arrayLen)
        var i         = 0
        while (i < arrayLen) {
          $seq(i) = ${
            val elementOffset = '{ ${ Expr(elementSizeInBytes) } * i }
            comp match {
              case p.Type.Struct(name, _, _,_) =>
                val arrElemPtr = readPrim('arrBuffer, elementOffset, p.Type.Long).asExprOf[Long]
                callRead(name, '{ $seq(i) }, arrElemPtr, ptrMap, objMap).asExprOf[t]
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
      ${
        Varargs(
          mapping.members.map { m =>
            val memberOffset = Expr(m.offsetInBytes.toInt)
            (root, m.tpe) match {
              case (_, p.Type.Array(comp)) =>
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

              case (_, p.Type.Struct(name, _, _,_)) =>
                val structPtr = readPrim('buffer, memberOffset, p.Type.Long).asExprOf[Long]
                if (m.mut) {
                  q.Assign(
                    m.select(root.asTerm),
                    callRead(name, m.select(root.asTerm).asExpr, structPtr, ptrMap, objMap).asTerm
                  ).asExprOf[Unit]
                } else {
                  // Don't update if there's a prism
                  if (!mapping.write.isDefined) {
                    callUpdate(name, m.select(root.asTerm).asExpr, structPtr, ptrMap, objMap)
                  } else '{ () }
                }
              case (_, _) =>
                if (m.mut) {
                  q.Assign(m.select(root.asTerm), readPrim('buffer, memberOffset, m.tpe).asTerm).asExprOf[Unit]
                } else '{ () } // otherwise no-op
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
        println(">> " + q.TypeRepr.of[t].widenTermRefByName.typeSymbol.primaryConstructor.tree.show)

        q.TypeRepr.of[t].widenTermRefByName.typeSymbol.primaryConstructor.tree match {
          case q.DefDef("<init>", ps, _, None) if ps.forall {
                case q.TypeParamClause(_) | q.TermParamClause(Nil) => true
                case _                                             => false
              } => // default no-arg ctor
            // TODO make sure we copy in the type args
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
                case p.Type.Array(comp) =>
                  // readArray[t](seq, comp, ptrMap, objMap, memberOffset, 'buffer, mapping)
                  '{ ??? }.asTerm
                case p.Type.Struct(name, _, _,_) =>
                  val structPtr = readPrim('buffer, memberOffset, p.Type.Long).asExprOf[Long]
                  callRead(name, '{ null }, structPtr, ptrMap, objMap).asTerm
                case _ => readPrim('buffer, memberOffset, m.tpe).asTerm
              }
            }
            q.Select
              .unique(q.New(q.TypeIdent(q.TypeRepr.of[t].typeSymbol)), "<init>")
              .appliedToTypes(args)
              .appliedToArgs(terms)
              .asExprOf[t]
          case _ => ???
        }

        // println("ctor = " + q.TypeRepr.of[t].typeSymbol.primaryConstructor.tree)

      }
    }

    def writeMethod(symbol: q.Symbol, mapping: StructMapping[q.Term]) = mkMethodDef(symbol) {
      case List(root: q.Term, ptrMap: q.Term) =>
        (root.asExpr, mapping.write.fold(root)((_, f) => f(root)).asExpr, ptrMap.asExpr) match {
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
        (root.asExpr, ptr.asExpr, ptrMap.asExpr, objMap.asExpr) match {
          case ('{ $rootExpr: t }, '{ $ptrExpr: Long }, '{ $ptrMap: PtrMapTpe }, '{ $objMap: ObjMapTpe }) =>
            '{
              val root: t = $rootExpr
              ($ptrMap.get(root), $ptrExpr) match {
                case (_, 0) => null // object reassignment for var to null
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
        (root.asExpr, ptr.asExpr, ptrMap.asExpr, objMap.asExpr) match {
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

    val mappings   = reprs.keys.map(mkStructMapping(_, layouts)).map(m => m.source -> m).toMap
    val writeDefs  = writeSymbols.map((sdef, symbol) => writeMethod(symbol, mappings(sdef))).toList
    val readDefs   = readSymbols.map((sdef, symbol) => readMethod(symbol, mappings(sdef))).toList
    val updateDefs = updateSymbols.map((sdef, symbol) => updateMethod(symbol, mappings(sdef))).toList

    val allDefs = writeDefs ::: readDefs ::: updateDefs

    (allDefs, callWrite, callRead, callUpdate)
  }

}
