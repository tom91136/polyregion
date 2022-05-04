package polyregion.scala

import cats.Eval
import cats.data.EitherT
import cats.syntax.all.*
import polyregion.{Member, PolyregionCompiler}
import polyregion.ast.{CppSourceMirror, MsgPack, PolyAst as p, *}
import polyregion.scala.{NativeStruct, *}

import java.lang.reflect.Modifier
import java.nio.file.{Files, Paths, StandardOpenOption}
import java.time.Duration
import java.util.concurrent.CountDownLatch
import java.util.concurrent.atomic.AtomicReference
import scala.annotation.{compileTimeOnly, tailrec}
import scala.collection.Factory
import scala.collection.immutable.ArraySeq
import scala.concurrent.ExecutionContext
import scala.quoted.*
import scala.reflect.ClassTag

// @compileTimeOnly("This class only exists at compile-time to expose offload methods")
object compiletime {

  inline def showExpr(inline x: Any): Any = ${ showExprImpl('x) }

  def showExprImpl(x: Expr[Any])(using q: Quotes): Expr[Any] = {
    import quotes.reflect.*
    given Printer[Tree] = Printer.TreeAnsiCode
    println(x.show)
    pprint.pprintln(x.asTerm) // term => AST

    implicit val Q = Quoted(q)
    val is = Q.collectTree(x.asTerm) {
      case s: Q.Select => s :: Nil
      case s: Q.Ident  => s :: Nil
      case _           => Nil
    }
    println("===")
    println(s"IS=${is
      .filter(x => x.symbol.isDefDef || true)
      .reverse
      .map(x => x -> x.tpe.dealias.widenTermRefByName.simplified)
      .map((x, tpe) => s"-> $x : ${x.show} `${x.symbol.fullName}` : ${tpe.show}\n\t${tpe}")
      .mkString("\n")}")
    println("===")
    x
  }

  inline def showTpe[B]: Unit = ${ showTpeImpl[B] }

  def showTpeImpl[B: Type](using Quotes): Expr[Unit] = {
    import quotes.reflect.*
    println(TypeRepr.of[B].typeSymbol.tree.show)
    import pprint.*
    pprint.pprintln(TypeRepr.of[B].typeSymbol)
    '{}
  }

  // inline def nativeStructOf[A]: NativeStruct[A] = ${ nativeStructOfImpl[A] }

  private transparent inline def liftTpe(using q: Quotes)(t: p.Type) = t match {
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

  private inline def tpeAsRuntimeTpe(t: p.Type): Byte = t match {
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

  private inline def sizeOf(using q: Quoted)(tpe: p.Type, repr: q.TypeRepr): Int = tpe match {
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

  private inline def readPrimitiveAtOffset //
  (using q: Quotes)                        //
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

  private inline def writePrimitiveAtOffset //
  (using q: Quotes)                         //
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

  private def readUniform //
  (using q: Quoted)       //
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

  private def writeUniform //
  (using q: Quoted)        //
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
          case x @ '[scala.collection.Seq[t]] =>
            '{
              val xs = ${ value.asExprOf[x.Underlying] }
              var i  = 0
              while (i < xs.size) { ${ writeUniform(buffer, '{ i }, tpe, q.TypeRepr.of[t], '{ xs(i) }) }; i += 1 }
              ()
            }
          case illegal => ???
        }

      case p.Type.String => ???
      case t             => writePrimitiveAtOffset(buffer, '{ $index * ${ Expr(sizeOf(t, repr)) } }, t, value)
    }
  }

  // def nativeStructOfImpl(using q: Quoted)(clsSymbol: q.Symbol): Expr[NativeStruct[A]] = {
  //   import scala.quoted.*
  //   Retyper.structDef0(clsSymbol) match {
  //     case Left(e)     => throw e
  //     case Right(sdef) =>
  //       // StructDef =>

  //       println(s"Do sdef=${sdef.repr}")
  //       val layout = PolyregionCompiler.layoutOf(MsgPack.encode(MsgPack.Versioned(CppSourceMirror.AdtHash, sdef)))
  //       println(s"layout=${layout}")

  //       val tpeSym = q.TypeTree.of[A].symbol

  //       def encodeField(
  //           buffer: Expr[java.nio.ByteBuffer],
  //           offset: Expr[Int],
  //           a: Expr[A],
  //           named: p.Named,
  //           m: Member
  //       ) = {
  //         val byteOffset = '{ $offset + ${ Expr(m.offsetInBytes.toInt) } }
  //         val value      = q.Select.unique(a.asTerm, named.symbol)
  //         named.tpe match {
  //           case p.Type.Float  => '{ ${ buffer }.putFloat(${ byteOffset }, ${ value.asExprOf[Float] }); () }
  //           case p.Type.Double => '{ ${ buffer }.putDouble(${ byteOffset }, ${ value.asExprOf[Double] }); () }

  //           case p.Type.Bool =>
  //             '{ ${ buffer }.put(${ byteOffset }, (if (!${ value.asExprOf[Boolean] }) 0 else 1).toByte); () }
  //           case p.Type.Byte  => '{ ${ buffer }.put(${ byteOffset }, ${ value.asExprOf[Byte] }); () }
  //           case p.Type.Char  => '{ ${ buffer }.putChar(${ byteOffset }, ${ value.asExprOf[Char] }); () }
  //           case p.Type.Short => '{ ${ buffer }.putShort(${ byteOffset }, ${ value.asExprOf[Short] }); () }
  //           case p.Type.Int   => '{ ${ buffer }.putInt(${ byteOffset }, ${ value.asExprOf[Int] }); () }
  //           case p.Type.Long  => '{ ${ buffer }.putLong(${ byteOffset }, ${ value.asExprOf[Long] }); () }

  //           case p.Type.String                      => ???
  //           case p.Type.Unit                        => ???
  //           case p.Type.Struct(name, tpeVars, args) => ???
  //           case p.Type.Array(component)            => ???
  //         }

  //       }

  //       def decodeField(buffer: Expr[java.nio.ByteBuffer], offset: Expr[Int], named: p.Named, m: Member) = {
  //         val byteOffset = '{ $offset + ${ Expr(m.offsetInBytes.toInt) } }
  //         named.tpe match {
  //           case p.Type.Float  => '{ ${ buffer }.getFloat(${ byteOffset }) }
  //           case p.Type.Double => '{ ${ buffer }.getDouble(${ byteOffset }) }

  //           case p.Type.Bool  => '{ if (${ buffer }.get(${ byteOffset }) == 0) false else true }
  //           case p.Type.Byte  => '{ ${ buffer }.get(${ byteOffset }) }
  //           case p.Type.Char  => '{ ${ buffer }.getChar(${ byteOffset }) }
  //           case p.Type.Short => '{ ${ buffer }.getShort(${ byteOffset }) }
  //           case p.Type.Int   => '{ ${ buffer }.getInt(${ byteOffset }) }
  //           case p.Type.Long  => '{ ${ buffer }.getLong(${ byteOffset }) }

  //           case p.Type.String                      => ???
  //           case p.Type.Unit                        => ???
  //           case p.Type.Struct(name, tpeVars, args) => ???

  //           case p.Type.Array(component) => ???

  //         }

  //       }

  //       val fields = sdef.members.zip(layout.members)
  //       '{
  //         new NativeStruct[A] {
  //           override val name        = ${ Expr(tpeSym.fullName) }
  //           override val sizeInBytes = ${ Expr(layout.sizeInBytes.toInt) }
  //           // override def member                                          = Vector()
  //           override def encode(buffer: java.nio.ByteBuffer, index: Int, a: A): Unit = {
  //             val offset = sizeInBytes * index
  //             ${ Expr.ofList(fields.map((named, member) => encodeField('buffer, 'offset, 'a, named, member))) }
  //           }

  //           override def decode(buffer: java.nio.ByteBuffer, index: Int): A = {
  //             val offset = sizeInBytes * index

  //             ${
  //               val tpeTree  = Q.TypeTree.of[A]
  //               val isObject = tpeTree.tpe.typeSymbol.flags.is(Q.Flags.Module)
  //               if (isObject) {
  //                 // Remapper.selectObject(tpeTree.tpe.typeSymbol)
  //                 '{ ??? }.asExprOf[A]
  //               } else {
  //                 Q.Select
  //                   .unique(Q.New(tpeTree), "<init>")
  //                   .appliedToArgs(fields.map((named, member) => decodeField('buffer, 'offset, named, member).asTerm))
  //                   .asExprOf[A]
  //               }
  //             }
  //           }
  //         }
  //       }
  //   }

  // }

  // def nativeStructOfImpl[A: Type](using q: Quotes): Expr[NativeStruct[A]] = {
  //   import quotes.reflect.*
  //   println(TypeRepr.of[A].typeSymbol.tree.show)

  //   implicit val Q = Quoted(q)

  // }

  inline def foreachJVMPar(inline range: Range, inline n: Int = java.lang.Runtime.getRuntime.availableProcessors())(
      inline x: Int => Unit
  )(using ec: ExecutionContext): Unit = {
    val latch = new CountDownLatch(n)
    Runtime.splitStatic(range)(n).foreach { r =>
      ec.execute { () =>
        try foreachJVM(r)(x)
        finally latch.countDown()
      }
    }
    latch.await()
  }

  inline def foreachJVM(inline range: Range)(inline x: Int => Unit): Unit = {
    val start = range.start
    val bound = if (range.isInclusive) range.end - 1 else range.end
    val step  = range.step
    var i     = start
    while (i < bound) {
      x(i)
      i += step
    }
  }

  inline def reduceJVM[A](inline range: Range) //
  (inline empty: A, inline f: Int => A)(inline g: (A, A) => A): A = {
    val start = range.start
    val bound = if (range.isInclusive) range.end - 1 else range.end
    val step  = range.step
    var i     = start
    var acc   = empty
    while (i < bound) {
      acc = g(f(i), acc)
      i += step
    }
    acc
  }

  inline def reduceJVMPar[A](
      inline range: Range,
      inline n: Int = java.lang.Runtime.getRuntime.availableProcessors()
  ) //
  (inline empty: A, inline f: Int => A)(inline g: (A, A) => A)(using ec: ExecutionContext): A = {
    val latch = new CountDownLatch(n)
    val acc   = new AtomicReference[A](empty)
    Runtime.splitStatic(range)(n).foreach { r =>
      ec.execute { () =>
        val x = reduceJVM(r)(empty, f)(g)
        try acc.getAndUpdate(g(_, x))
        finally latch.countDown()
      }
    }
    latch.await()
    acc.get()
  }

  inline def foreach(inline range: Range)(inline x: Int => Unit): Unit = {
    val start = range.start
    val bound = if (range.isInclusive) range.end - 1 else range.end
    val step  = range.step
    offload {
      var i = start
      while (i < bound) {
        x(i)
        i += step
      }
    }
  }

  inline def foreachPar(inline range: Range, inline n: Int = java.lang.Runtime.getRuntime.availableProcessors())(
      inline x: Int => Unit
  )(using ec: ExecutionContext): Unit = {
    val latch = new CountDownLatch(n)
    Runtime.splitStatic(range)(n).foreach { r =>
      ec.execute { () =>
        try foreach(r)(x)
        finally latch.countDown()
      }
    }
    latch.await()
  }

  inline def reduce[A](inline range: Range) //
  (inline empty: A, inline f: Int => A)(inline g: (A, A) => A): A = {
    val start = range.start
    val bound = if (range.isInclusive) range.end - 1 else range.end
    val step  = range.step
    offload {
      var i   = start
      var acc = empty
      while (i < bound) {
        acc = g(f(i), acc)
        i += step
      }
      acc
    }
  }

  inline def reducePar[A](inline range: Range, inline n: Int = java.lang.Runtime.getRuntime.availableProcessors()) //
  (inline empty: A, inline f: Int => A)(inline g: (A, A) => A)(using ec: ExecutionContext): A = {
    val latch = new CountDownLatch(n)
    val acc   = new AtomicReference[A](empty)
    Runtime.splitStatic(range)(n).foreach { r =>
      ec.execute { () =>
        val x = reduce(r)(empty, f)(g)
        try acc.getAndUpdate(g(_, x))
        finally latch.countDown()
      }
    }
    latch.await()
    acc.get()
  }

  inline def offload[A](inline x: => A): A = ${ offloadImpl[A]('x) }

  // def noComplexReturn(kind: String): Nothing = throw new AssertionError(
  //   s"Compiler bug: returning $kind is not possible, " +
  //     "it should have been transformed to an out param in one of the passes."
  // )

  private def offloadImpl[A: Type](x: Expr[Any])(using q: Quotes): Expr[A] = {
    implicit val Q = Quoted(q)
    val result = for {
      (captures, prog, log) <- Compiler.compileExpr(x)
      _ = println(log.render)
      serialisedAst <- Either.catchNonFatal(MsgPack.encode(MsgPack.Versioned(CppSourceMirror.AdtHash, prog)))
      // _ <- Either.catchNonFatal(throw new RuntimeException("STOP"))
      // layout <- Either.catchNonFatal(PolyregionCompiler.layoutOf(MsgPack.encode(MsgPack.Versioned(CppCodeGen.AdtHash, prog.defs))))
      //   _= println(s"layout=${layout}")

      // c <- Right(new polyregion.Compilation())
      c <- Either.catchNonFatal(PolyregionCompiler.compile(serialisedAst, true, PolyregionCompiler.BACKEND_LLVM))
    } yield {

      println(s"Messages=\n  ${c.messages}")
      println(s"Program=${c.program.length}")
      println(s"Elapsed=\n${c.events.sortBy(_.epochMillis).mkString("\n")}")

      val programBytesExpr = Expr(c.program)
      val astBytesExpr     = Expr(serialisedAst)
      val fnName           = Expr("lambda")

      val captureExprs = captures.map { (name, ref) =>
        val buffer = name.tpe match {
          case p.Type.Array(comp) =>
            ref.tpe.asType match {
              case x @ '[scala.collection.Seq[t]] =>
                '{
                  val xs = ${ ref.asExprOf[x.Underlying] }
                  java.nio.ByteBuffer.allocateDirect(${ Expr(sizeOf(name.tpe, ref.tpe)) } * xs.size)
                }
              case illegal => ???
            }
          case _ =>
            '{
              java.nio.ByteBuffer.allocateDirect(${ Expr(sizeOf(name.tpe, ref.tpe)) })
            }
        }
        '{
          println(s"mkBuffer: ${${ buffer }}")
          ${ writeUniform(buffer, '{ 0 }, name.tpe, ref.tpe, ref.asExpr) }
          ${ buffer }
        }
      }
      val captureTps = captures.map((name, _) => tpeAsRuntimeTpe(name.tpe))

      // def wrap(buffer: Expr[java.nio.ByteBuffer], comp: p.Type) =
      //   comp match {
      //     case p.Type.Unit   => '{ Buffer.view[Unit](${ buffer }) }
      //     case p.Type.Float  => '{ Buffer.view[Float](${ buffer }) }
      //     case p.Type.Double => '{ Buffer.view[Double](${ buffer }) }
      //     case p.Type.Bool   => '{ Buffer.view[Boolean](${ buffer }) }
      //     case p.Type.Byte   => '{ Buffer.view[Byte](${ buffer }) }
      //     case p.Type.Char   => '{ Buffer.view[Char](${ buffer }) }
      //     case p.Type.Short  => '{ Buffer.view[Short](${ buffer }) }
      //     case p.Type.Int    => '{ Buffer.view[Int](${ buffer }) }
      //     case p.Type.Long   => '{ Buffer.view[Long](${ buffer }) }
      //     case _             => ???
      //   }

      val code = '{

        val bytes    = $programBytesExpr
        val argTypes = new Array[Byte](${ Expr(captureTps.size) })
        ${ Varargs(captureTps.zipWithIndex.map((e, i) => '{ argTypes(${ Expr(i) }) = ${ Expr(e) } })) }

        val argBuffers = new Array[java.nio.ByteBuffer](${ Expr(captureExprs.size) })
        ${ Varargs(captureExprs.zipWithIndex.map((e, i) => '{ argBuffers(${ Expr(i) }) = $e })) }

        ${
          (prog.entry.rtn match {
            case p.Type.Unit   => '{ polyregion.PolyregionRuntime.invoke(bytes, $fnName, argTypes, argBuffers) }
            case p.Type.Float  => '{ polyregion.PolyregionRuntime.invokeFloat(bytes, $fnName, argTypes, argBuffers) }
            case p.Type.Double => '{ polyregion.PolyregionRuntime.invokeDouble(bytes, $fnName, argTypes, argBuffers) }
            case p.Type.Bool   => '{ polyregion.PolyregionRuntime.invokeBool(bytes, $fnName, argTypes, argBuffers) }
            case p.Type.Byte   => '{ polyregion.PolyregionRuntime.invokeByte(bytes, $fnName, argTypes, argBuffers) }
            case p.Type.Char   => '{ polyregion.PolyregionRuntime.invokeChar(bytes, $fnName, argTypes, argBuffers) }
            case p.Type.Short  => '{ polyregion.PolyregionRuntime.invokeShort(bytes, $fnName, argTypes, argBuffers) }
            case p.Type.Int    => '{ polyregion.PolyregionRuntime.invokeInt(bytes, $fnName, argTypes, argBuffers) }
            case p.Type.Long   => '{ polyregion.PolyregionRuntime.invokeLong(bytes, $fnName, argTypes, argBuffers) }
            case x @ p.Type.Struct(_, _, _) =>
              // val imp = Expr.summon[NativeStruct[A]] match {
              //   case None    => Q.report.errorAndAbort(s"Cannot find NativeStruct for type ${Type.of[A]}")
              //   case Some(x) => x
              // }
              '{
                var buffer = polyregion.PolyregionRuntime.invokeObject(bytes, $fnName, argTypes, argBuffers, -1)
                ${ readUniform('buffer, '{ 0 }, x, Q.TypeRepr.of[A]) }
                // Buffer.structViewAny[A](buffer)(using $imp)(0)
              }
            case x @ p.Type.Array(comp) =>
              //  ${
              //   Type.of[A] match {
              //     case '[Buffer[a]] => wrap('{ buffer }, comp) // passthrough
              //     case '[Array[a]]  => '{ ${ wrap('{ buffer }, comp) }.copyToArray }
              //     case m            => ???
              //   }
              // }
              '{
                var buffer = polyregion.PolyregionRuntime.invokeObject(bytes, $fnName, argTypes, argBuffers, -1)
                ${ readUniform('buffer, '{ 0 }, x, Q.TypeRepr.of[A]) }
              }
          }).asExprOf[A]
        }
      }
      println("Code=" + code.show)
      code
    }

//    println(prog.toByteArray.mkString(" "))
//    println(Program.parseFrom(prog.toByteArray).toProtoString)

//    val b = '{ val b = polyregion.Runtime.FFIInvocationBuilder() }

    result match {
      case Left(e)  => throw e
      case Right(x) =>
        // println("Code=" + x.show)
        x
    }

  }

}
