package polyregion.scala

import polyregion.jvm.compiler.Compiler
import polyregion.ast.{PolyAst as p, *}
import polyregion.jvm.compiler.Layout.Member

import scala.quoted.*

object Pickler {

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

  inline def tpeAsRuntimeTpe(t: p.Type): polyregion.jvm.runtime.Type = t match {
    case p.Type.Bool            => polyregion.jvm.runtime.Type.BOOL
    case p.Type.Byte            => polyregion.jvm.runtime.Type.BYTE
    case p.Type.Char            => polyregion.jvm.runtime.Type.CHAR
    case p.Type.Short           => polyregion.jvm.runtime.Type.SHORT
    case p.Type.Int             => polyregion.jvm.runtime.Type.INT
    case p.Type.Long            => polyregion.jvm.runtime.Type.LONG
    case p.Type.Float           => polyregion.jvm.runtime.Type.FLOAT
    case p.Type.Double          => polyregion.jvm.runtime.Type.DOUBLE
    case p.Type.Array(_)        => polyregion.jvm.runtime.Type.PTR
    case p.Type.Struct(_, _, _) => polyregion.jvm.runtime.Type.PTR
    case p.Type.Unit            => polyregion.jvm.runtime.Type.VOID
    case unknown =>
      println(s"tpeAsRuntimeTpe ??? = $unknown ")
      ???
  }

  inline def getPrimitive(using q: Quotes) //
  (buffer: Expr[java.nio.ByteBuffer], byteOffset: Expr[Int], tpe: p.Type): Expr[Any] = tpe match {
    case p.Type.Float  => '{ $buffer.getFloat($byteOffset) }
    case p.Type.Double => '{ $buffer.getDouble($byteOffset) }
    case p.Type.Bool   => '{ if ($buffer.get($byteOffset) == 0) false else true }
    case p.Type.Byte   => '{ $buffer.get($byteOffset) }
    case p.Type.Char   => '{ $buffer.getChar($byteOffset) }
    case p.Type.Short  => '{ $buffer.getShort($byteOffset) }
    case p.Type.Int    => '{ $buffer.getInt($byteOffset) }
    case p.Type.Long   => '{ $buffer.getLong($byteOffset) }
    case p.Type.Unit   => '{ $buffer.get($byteOffset); () }
    case x             => q.reflect.report.errorAndAbort(s"Type $x is not a primitive type")
  }

  inline def putPrimitive(using q: Quotes) //
  (buffer: Expr[java.nio.ByteBuffer], byteOffset: Expr[Int], tpe: p.Type, value: Expr[Any]): Expr[Unit] = tpe match {
    case p.Type.Float  => '{ $buffer.putFloat($byteOffset, ${ value.asExprOf[Float] }) }
    case p.Type.Double => '{ $buffer.putDouble($byteOffset, ${ value.asExprOf[Double] }) }
    case p.Type.Bool   => '{ $buffer.put($byteOffset, if (!${ value.asExprOf[Boolean] }) 0.toByte else 1.toByte) }
    case p.Type.Byte   => '{ $buffer.put($byteOffset, ${ value.asExprOf[Byte] }) }
    case p.Type.Char   => '{ $buffer.putChar($byteOffset, ${ value.asExprOf[Char] }) }
    case p.Type.Short  => '{ $buffer.putShort($byteOffset, ${ value.asExprOf[Short] }) }
    case p.Type.Int    => '{ $buffer.putInt($byteOffset, ${ value.asExprOf[Int] }) }
    case p.Type.Long   => '{ $buffer.putLong($byteOffset, ${ value.asExprOf[Long] }) }
    case p.Type.Unit   => '{ $buffer.put($byteOffset, 0.toByte) }
    case x             => q.reflect.report.errorAndAbort(s"Type $x is not a primitive type", value)
  }

  inline def layoutOf(using q: Quoted)(compiler: Compiler, repr: q.TypeRepr): polyregion.jvm.compiler.Layout = {
    val sdef = Retyper.structDef0(repr.typeSymbol).getOrElse(???)
    println(s"A=${sdef} ${repr.widenTermRefByName}")
    compiler.layoutOf(CppSourceMirror.encode(sdef), ???)
  }

  inline def sizeOf(using q: Quoted)(compiler: Compiler,tpe: p.Type, repr: q.TypeRepr): Int = tpe match {
    case p.Type.Float           => polyregion.jvm.runtime.Type.FLOAT.sizeInBytes
    case p.Type.Double          => polyregion.jvm.runtime.Type.DOUBLE.sizeInBytes
    case p.Type.Bool            => polyregion.jvm.runtime.Type.BYTE.sizeInBytes
    case p.Type.Byte            => polyregion.jvm.runtime.Type.BYTE.sizeInBytes
    case p.Type.Char            => polyregion.jvm.runtime.Type.CHAR.sizeInBytes
    case p.Type.Short           => polyregion.jvm.runtime.Type.SHORT.sizeInBytes
    case p.Type.Int             => polyregion.jvm.runtime.Type.INT.sizeInBytes
    case p.Type.Long            => polyregion.jvm.runtime.Type.LONG.sizeInBytes
    case p.Type.Unit            => polyregion.jvm.runtime.Type.VOID.sizeInBytes
    case p.Type.Array(_)        => polyregion.jvm.runtime.Type.PTR.sizeInBytes
    case p.Type.Struct(_, _, _) => layoutOf(compiler, repr).sizeInBytes.toInt
    case x                      => q.quotes.reflect.report.errorAndAbort(s"Cannot get size of type $x")
  }

  def readStruct(using q: Quoted) //
  (compiler: Compiler, buffer: Expr[java.nio.ByteBuffer], index: Expr[Int], repr: q.TypeRepr) = {
    import q.given
    // Find out the total size of this struct first, it could be nested arbitrarily but the top level's size must
    // reflect the total size; this is consistent with C's `sizeof(struct T)`.
    val sdef       = Retyper.structDef0(repr.typeSymbol).getOrElse(???)
    val layout     = compiler.layoutOf(CppSourceMirror.encode(sdef), ???)
    val byteOffset = '{ ${ Expr(layout.sizeInBytes.toInt) } * $index }
    val fields     = sdef.members.zip(layout.members)
    val terms = fields.map { (named, m) =>
      getPrimitive(buffer, '{ $byteOffset + ${ Expr(m.offsetInBytes.toInt) } }, named.tpe).asTerm
    }
    q.Select
      .unique(q.New(q.TypeIdent(repr.typeSymbol)), "<init>")
      .appliedToArgs(terms)
      .asExpr
  }

  inline def writeStruct(using q: Quoted) //
  (compiler: Compiler,buffer: Expr[java.nio.ByteBuffer], index: Expr[Int], repr: q.TypeRepr, value: Expr[Any]) = {
    import q.given
    // Find out the total size of this struct first, it could be nested arbitrarily but the top level's size must
    // reflect the total size; this is consistent with C's `sizeof(struct T)`.
    val sdef       = Retyper.structDef0(repr.typeSymbol).getOrElse(???)
    val layout     = compiler.layoutOf(CppSourceMirror.encode(sdef), ???)
    val byteOffset = '{ ${ Expr(layout.sizeInBytes.toInt) } * $index }
    val fields     = sdef.members.zip(layout.members)
    val terms = fields.map { (named, m) =>
      putPrimitive(
        buffer,
        '{ $byteOffset + ${ Expr(m.offsetInBytes.toInt) } },
        named.tpe,
        q.Select.unique(value.asTerm, named.symbol).asExpr
      )
    }
    Expr.block(terms, '{ () })
  }


  def putAll(using q: Quoted) //
  (compiler: Compiler, b: Expr[java.nio.ByteBuffer], tpe: p.Type, repr: q.TypeRepr, v: Expr[Any]): Expr[Unit] = {
    import q.given
    import p.{Type => PT}

    inline def put[t: Type](comp: PT, i: Expr[Int], v: Expr[Any]) = comp match {
      case p.Type.Struct(_, _, _) => writeStruct(compiler, b, i, q.TypeRepr.of[t], v)
      case c                      => putPrimitive(b, '{ $i * ${ Expr(sizeOf(compiler, comp, q.TypeRepr.of[t])) } }, c, v)
    }

    (tpe, repr.asType) match {
      case (PT.String, _)                              => ???
      case (PT.Array(PT.Array(_)), _)                  => ??? // TODO handle nested arrays
      case (PT.Array(PT.Byte), x @ '[Array[Byte]])     => '{ $b.put(${ v.asExprOf[x.Underlying] }) }
      case (PT.Array(PT.Char), x @ '[Array[Char]])     => '{ $b.asCharBuffer.put(${ v.asExprOf[x.Underlying] }) }
      case (PT.Array(PT.Int), x @ '[Array[Int]])       => '{ $b.asIntBuffer.put(${ v.asExprOf[x.Underlying] }) }
      case (PT.Array(PT.Short), x @ '[Array[Short]])   => '{ $b.asShortBuffer.put(${ v.asExprOf[x.Underlying] }) }
      case (PT.Array(PT.Long), x @ '[Array[Long]])     => '{ $b.asLongBuffer.put(${ v.asExprOf[x.Underlying] }) }
      case (PT.Array(PT.Float), x @ '[Array[Float]])   => '{ $b.asFloatBuffer.put(${ v.asExprOf[x.Underlying] }) }
      case (PT.Array(PT.Double), x @ '[Array[Double]]) => '{ $b.asDoubleBuffer.put(${ v.asExprOf[x.Underlying] }) }
      case (PT.Array(comp), x @ '[scala.Array[t]]) =>
        '{
          val xs = ${ v.asExprOf[x.Underlying] }
          var i  = 0; while (i < xs.length) { ${ put(comp, 'i, '{ xs(i) }) }; i += 1 }
        }
      case (p.Type.Array(comp), x @ '[scala.collection.Seq[t]]) =>
        // Readonly, so whether collection is mutable or not doesn't matter.
        '{
          val xs = ${ v.asExprOf[x.Underlying] }
          var i  = 0; while (i < xs.length) { ${ put(comp, 'i, '{ xs(i) }) }; i += 1 }
        }
      case (p.Type.Array(comp), x @ '[java.util.List[t]]) =>
        '{
          val xs = ${ v.asExprOf[x.Underlying] }
          var i  = 0; while (i < xs.size) { ${ put(comp, 'i, '{ xs.get(i) }) }; i += 1 }
        }
      case (t @ p.Type.Array(_), illegal) =>
        q.report.errorAndAbort(s"Unsupported type ${t.repr} (${v.show}:${repr.show}) for writing to ByteBuffer.", v)
      case (t, '[x]) => put[x](t, '{ 0 }, v)
      case (t, _)    => q.report.errorAndAbort(s"Type information unavailable for ${t.repr}")
    }
  }

  def getAllMutable(using q: Quoted) //
  (compiler: Compiler,b: Expr[java.nio.ByteBuffer], tpe: p.Type, repr: q.TypeRepr, v: Expr[Any]): Expr[Unit] = {
    import q.given
    import p.{Type => PT}

    inline def get[t: Type](comp: PT, i: Expr[Int]) = (comp match {
      case p.Type.Struct(_, _, _) => readStruct(compiler, b, i, q.TypeRepr.of[t])
      case c                      => getPrimitive(b, '{ $i * ${ Expr(sizeOf(compiler, comp, q.TypeRepr.of[t])) } }, c)
    }).asExprOf[t]

    (tpe, repr.asType) match {
      case (PT.String, _)                              => ???
      case (PT.Array(PT.Array(_)), _)                  => ??? // TODO handle nested arrays
      case (PT.Array(PT.Byte), x @ '[Array[Byte]])     => '{ $b.get(${ v.asExprOf[x.Underlying] }) }
      case (PT.Array(PT.Char), x @ '[Array[Char]])     => '{ $b.asCharBuffer.get(${ v.asExprOf[x.Underlying] }) }
      case (PT.Array(PT.Int), x @ '[Array[Int]])       => '{ $b.asIntBuffer.get(${ v.asExprOf[x.Underlying] }) }
      case (PT.Array(PT.Short), x @ '[Array[Short]])   => '{ $b.asShortBuffer.get(${ v.asExprOf[x.Underlying] }) }
      case (PT.Array(PT.Long), x @ '[Array[Long]])     => '{ $b.asLongBuffer.get(${ v.asExprOf[x.Underlying] }) }
      case (PT.Array(PT.Float), x @ '[Array[Float]])   => '{ $b.asFloatBuffer.get(${ v.asExprOf[x.Underlying] }) }
      case (PT.Array(PT.Double), x @ '[Array[Double]]) => '{ $b.asDoubleBuffer.get(${ v.asExprOf[x.Underlying] }) }
      case (PT.Array(comp), x @ '[scala.Array[t]]) =>
        '{
          val xs = ${ v.asExprOf[x.Underlying] }
          var i  = 0; while (i < xs.length) { xs(i) = ${ get[t](comp, 'i) }; i += 1 }
        }
      case (p.Type.Array(comp), x @ '[scala.collection.mutable.Seq[t]]) =>
        // Unlike putAll, this is mutable only otherwise we can't write.
        '{
          val xs = ${ v.asExprOf[x.Underlying] }
          var i  = 0; while (i < xs.length) { xs(i) = ${ get[t](comp, 'i) }; i += 1 }
        }
      case (p.Type.Array(comp), x @ '[java.util.List[t]]) =>
        '{
          val xs = ${ v.asExprOf[x.Underlying] }
          var i  = 0; while (i < xs.size) { xs.set(i, ${ get[t](comp, 'i) }); i += 1 }
        }
      case (t @ p.Type.Array(_), illegal) =>
        q.report.errorAndAbort(s"Unsupported non-mutable type ${t.repr} for reading from ByteBuffer: ${repr.show}")
      // case (t, '[x]) => get[x](t, '{ 0 }).asExprOf[Unit]
      case (t, _)    => q.report.errorAndAbort(s"Type information unavailable for ${t.repr}")
    }
  }

  // TODO 
  // Array[Solid]        = ( S[N]... )
  // Solid               = ( S )
  // RefA{RefB}          = ( RefA{*T, N:Int, RefB{*U, N':Int}}, T[N]..., U[N']... )
  // Ref                 = ( Ref{*T, N:Int, *U, N':Int ...}, T[N]..., U[N']... )
  // Array[Ref] xs       = ( Ref{*T, N:Int, *U, N':Int ...}[xs.size]..., T[N * xs.size]..., U[N' * xs.size]...  )
  // Array[Array[Solid]] = (Ref{*T, N1}[N0]..., T[N0*N1]...)


  // def writeUniform  //
  // (using q: Quoted) //
  // (buffer: Expr[java.nio.ByteBuffer], index: Expr[Int], tpe: p.Type, repr: q.TypeRepr, value: Expr[Any]): Expr[Unit] = {
  //   import q.given
  //   println(s"Write s=${repr} ${tpe}")

  //   tpe match {
  //     case p.Type.Struct(name, tpeVars, args) =>
  //       writeStruct(buffer, index, repr, value)
  //     case p.Type.Array(comp) =>
  //       // TODO handle special case for where value == wrapped buffers; just unwrap it here
  //       repr.asType match {
  //         case x @ '[scala.Array[t]] =>
  //           // TODO specialise for <Type>Buffer variants, there's a put <Type> array for all variants
  //           '{
  //             val xs = ${ value.asExprOf[x.Underlying] }
  //             var i  = 0
  //             while (i < xs.length) { ${ writeUniform(buffer, 'i, comp, q.TypeRepr.of[t], '{ xs(i) }) }; i += 1 }
  //             ()
  //           }
  //         case x @ '[scala.collection.Seq[t]] =>
  //           '{
  //             val xs = ${ value.asExprOf[x.Underlying] }
  //             var i  = 0
  //             while (i < xs.length) { ${ writeUniform(buffer, 'i, comp, q.TypeRepr.of[t], '{ xs(i) }) }; i += 1 }
  //             ()
  //           }

  //         case illegal => q.report.errorAndAbort(s"Unsupported type for writing ${repr.show}")
  //       }

  //     case t =>
  //       putPrimitive(buffer, '{ $index * ${ Expr(sizeOf(t, repr)) } }, t, value)
  //   }
  // }


  // def readUniform(using q: Quoted) //
  // (buffer: Expr[java.nio.ByteBuffer], index: Expr[Int], tpe: p.Type, repr: q.TypeRepr): Expr[Any] = {
  //   import q.given
  //   tpe match {
  //     case p.Type.Struct(name, tpeVars, args) => readStruct(buffer, index, repr)
  //     case p.Type.Array(component) =>
  //       repr.asType match {
  //         case '[scala.collection.immutable.Seq[t]] => ??? // make a new one
  //         case '[scala.collection.mutable.Seq[t]]   => ??? // write to existing if exists or make a new one
  //         case illegal                              => ???

  //       }
  //     // '{
  //     //       val xs = ${ value.asExprOf[scala.collection.immutable.Seq[_]] }
  //     //       var i  = 0
  //     //       while (i < xs.size) {  xs(i) =  ${ readUniform(buffer, '{ i }, tpe, compRepr,  ) }; i += 1 }
  //     //       ()
  //     //     }
  //     case p.Type.String => ???
  //     case t             => getPrimitive(buffer, '{ $index * ${ Expr(sizeOf(t, repr)) } }, t)
  //   }
  // }

}
