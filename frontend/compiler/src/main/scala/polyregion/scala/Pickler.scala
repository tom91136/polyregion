package polyregion.scala

import polyregion.{Member, PolyregionCompiler}
import polyregion.ast.{PolyAst as p, *}
import polyregion.scala.{NativeStruct, *}

import scala.quoted.*

object Pickler {

  // inline def nativeStructOf[A]: NativeStruct[A] = ${ nativeStructOfImpl[A] }

  transparent inline def liftTpe(using q: Quotes)(t: p.Type) = t match {
    case p.Type.Bool   => Type.of[Boolean]
    case p.Type.Char   => Type.of[Char]
    case p.Type.Byte   => Type.of[Byte]
    case p.Type.Short  => Type.of[Short]
    case p.Type.Int    => Type.of[Int]
    case p.Type.Long   => Type.of[Long]
    case p.Type.Float  => Type.of[Float]
    case p.Type.Double => Type.of[Double]
    case p.Type.Unit   => Type.of[Unit]
  }

  inline def tpeAsRuntimeTpe(t: p.Type): Byte = t match {
    case p.Type.Bool            => polyregion.PolyregionRuntime.TYPE_BOOL
    case p.Type.Byte            => polyregion.PolyregionRuntime.TYPE_BYTE
    case p.Type.Char            => polyregion.PolyregionRuntime.TYPE_CHAR
    case p.Type.Short           => polyregion.PolyregionRuntime.TYPE_SHORT
    case p.Type.Int             => polyregion.PolyregionRuntime.TYPE_INT
    case p.Type.Long            => polyregion.PolyregionRuntime.TYPE_LONG
    case p.Type.Float           => polyregion.PolyregionRuntime.TYPE_FLOAT
    case p.Type.Double          => polyregion.PolyregionRuntime.TYPE_DOUBLE
    case p.Type.Array(_)        => polyregion.PolyregionRuntime.TYPE_PTR
    case p.Type.Struct(_, _, _) => polyregion.PolyregionRuntime.TYPE_PTR
    case p.Type.Unit            => polyregion.PolyregionRuntime.TYPE_VOID
    case unknown =>
      println(s"tpeAsRuntimeTpe ??? = $unknown ")
      ???
  }

  inline def sizeOf(using q: Quoted)(tpe: p.Type, repr: q.TypeRepr): Int = tpe match {
    case p.Type.Float  => java.lang.Float.BYTES
    case p.Type.Double => java.lang.Double.BYTES
    case p.Type.Bool   => java.lang.Byte.BYTES
    case p.Type.Byte   => java.lang.Byte.BYTES
    case p.Type.Char   => java.lang.Character.BYTES
    case p.Type.Short  => java.lang.Short.BYTES
    case p.Type.Int    => java.lang.Integer.BYTES
    case p.Type.Long   => java.lang.Long.BYTES
    case p.Type.Unit   => java.lang.Byte.BYTES
    case p.Type.Struct(_, _, _) =>
      val sdef   = Retyper.structDef0(repr.typeSymbol).getOrElse(???)
      val layout = PolyregionCompiler.layoutOf(MsgPack.encode(MsgPack.Versioned(CppSourceMirror.AdtHash, sdef)))
      layout.sizeInBytes.toInt
  }

  inline def readPrimitiveAtOffset //
  (using q: Quotes)                //
  (buffer: Expr[java.nio.ByteBuffer], byteOffset: Expr[Int], tpe: p.Type): Expr[Any] =
    tpe match {
      case p.Type.Float  => '{ $buffer.getFloat($byteOffset) }
      case p.Type.Double => '{ $buffer.getDouble($byteOffset) }
      case p.Type.Bool   => '{ if ($buffer.get($byteOffset) == 0) false else true }
      case p.Type.Byte   => '{ $buffer.get($byteOffset) }
      case p.Type.Char   => '{ $buffer.getChar($byteOffset) }
      case p.Type.Short  => '{ $buffer.getShort($byteOffset) }
      case p.Type.Int    => '{ $buffer.getInt($byteOffset) }
      case p.Type.Long   => '{ $buffer.getLong($byteOffset) }
      case p.Type.Unit   => '{ $buffer.get($byteOffset); () }
      case _             => ???
    }

  inline def writePrimitiveAtOffset //
  (using q: Quotes)                 //
  (buffer: Expr[java.nio.ByteBuffer], byteOffset: Expr[Int], tpe: p.Type, value: Expr[Any]): Expr[Unit] =
    tpe match {
      case p.Type.Float  => '{ $buffer.putFloat($byteOffset, ${ value.asExprOf[Float] }) }
      case p.Type.Double => '{ $buffer.putDouble($byteOffset, ${ value.asExprOf[Double] }) }
      case p.Type.Bool   => '{ $buffer.put($byteOffset, if (!${ value.asExprOf[Boolean] }) 0.toByte else 1.toByte) }
      case p.Type.Byte   => '{ $buffer.put($byteOffset, ${ value.asExprOf[Byte] }) }
      case p.Type.Char   => '{ $buffer.putChar($byteOffset, ${ value.asExprOf[Char] }) }
      case p.Type.Short  => '{ $buffer.putShort($byteOffset, ${ value.asExprOf[Short] }) }
      case p.Type.Int    => '{ $buffer.putInt($byteOffset, ${ value.asExprOf[Int] }) }
      case p.Type.Long   => '{ $buffer.putLong($byteOffset, ${ value.asExprOf[Long] }) }
      case p.Type.Unit   => '{ $buffer.put($byteOffset, 0.toByte) }
      case _             => ???
    }

  def readUniform   //
  (using q: Quoted) //
  (buffer: Expr[java.nio.ByteBuffer], index: Expr[Int], tpe: p.Type, repr: q.TypeRepr): Expr[Any] = {
    import q.given
    tpe match {
      case p.Type.Struct(name, tpeVars, args) =>
        // find out the total size of this struct first, it could be nested arbitrarily but the top level's size must
        // reflect the total size; this is consistent with C's `sizeof(struct T)`
        val sdef       = Retyper.structDef0(repr.typeSymbol).getOrElse(???)
        val layout     = PolyregionCompiler.layoutOf(MsgPack.encode(MsgPack.Versioned(CppSourceMirror.AdtHash, sdef)))
        val byteOffset = '{ ${ Expr(layout.sizeInBytes.toInt) } * $index }
        val fields     = sdef.members.zip(layout.members)
        val terms = fields.map { (named, m) =>
          readPrimitiveAtOffset(buffer, '{ $byteOffset + ${ Expr(m.offsetInBytes.toInt) } }, named.tpe).asTerm
        }
        q.Select
          .unique(q.New(q.TypeIdent(repr.typeSymbol)), "<init>")
          .appliedToArgs(terms)
          .asExpr
      case p.Type.Array(component) =>
        // TODO handle special case for where value == wrapped buffers; just unwrap it here
        repr.asType match {
          case '[scala.collection.immutable.Seq[t]] => ??? // make a new one
          case '[scala.collection.mutable.Seq[t]]   => ??? // write to existing if exists or make a new one
          case illegal                              => ???

        }
      // '{
      //       val xs = ${ value.asExprOf[scala.collection.immutable.Seq[_]] }
      //       var i  = 0
      //       while (i < xs.size) {  xs(i) =  ${ readUniform(buffer, '{ i }, tpe, compRepr,  ) }; i += 1 }
      //       ()
      //     }

      case p.Type.String => ???
      case t             => readPrimitiveAtOffset(buffer, '{ $index * ${ Expr(sizeOf(t, repr)) } }, t)
    }
  }

  def writeUniform  //
  (using q: Quoted) //
  (buffer: Expr[java.nio.ByteBuffer], index: Expr[Int], tpe: p.Type, repr: q.TypeRepr, value: Expr[Any]): Expr[Unit] = {
    import q.given
    tpe match {
      case p.Type.Struct(name, tpeVars, args) =>
        // find out the total size of this struct first, it could be nested arbitrarily but the top level's size must
        // reflect the total size; this is consistent with C's `sizeof(struct T)`
        val sdef       = Retyper.structDef0(repr.typeSymbol).getOrElse(???)
        val layout     = PolyregionCompiler.layoutOf(MsgPack.encode(MsgPack.Versioned(CppSourceMirror.AdtHash, sdef)))
        val byteOffset = '{ ${ Expr(layout.sizeInBytes.toInt) } * $index }
        val fields     = sdef.members.zip(layout.members)
        val terms = fields.map { (named, m) =>
          writePrimitiveAtOffset(
            buffer,
            '{ $byteOffset + ${ Expr(m.offsetInBytes.toInt) } },
            named.tpe,
            q.Select.unique(value.asTerm, named.symbol).asExpr
          )
        }
        Expr.block(terms, '{ () })
      case p.Type.Array(comp) =>
        // TODO handle special case for where value == wrapped buffers; just unwrap it here
        repr.asType match {
          case x @ '[scala.Array[t]] =>
            // TODO specialise for <Type>Buffer variants, there's a put <Type> array for all variants
            '{
              val xs = ${ value.asExprOf[x.Underlying] }
              var i  = 0
              while (i < xs.length) { ${ writeUniform(buffer, 'i, comp, q.TypeRepr.of[t], '{ xs(i) }) }; i += 1 }
              ()
            }
          case x @ '[scala.collection.Seq[t]] =>
            '{
              val xs = ${ value.asExprOf[x.Underlying] }
              var i  = 0
              while (i < xs.length) { ${ writeUniform(buffer, 'i, comp, q.TypeRepr.of[t], '{ xs(i) }) }; i += 1 }
              ()
            }
          case illegal => q.report.errorAndAbort(s"Unsupported type for writing ${repr.show}")
        }

      case p.Type.String => ???
      case t             => writePrimitiveAtOffset(buffer, '{ $index * ${ Expr(sizeOf(t, repr)) } }, t, value)
    }
  }

}