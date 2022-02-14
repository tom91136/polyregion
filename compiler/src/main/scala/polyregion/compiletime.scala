package polyregion

import cats.Eval
import cats.data.EitherT
import cats.syntax.all.*
import polyregion.ast.{CppCodeGen, PolyAst}
import polyregion.ast.PolyAst as p

import polyregion.data.MsgPack
import polyregion.*

import java.lang.reflect.Modifier
import java.nio.file.{Files, Paths, StandardOpenOption}
import java.time.Duration
import java.util.concurrent.CountDownLatch
import java.util.concurrent.atomic.AtomicReference
import scala.annotation.tailrec
import scala.collection.immutable.ArraySeq
import scala.concurrent.ExecutionContext
import scala.quoted.*
import scala.reflect.ClassTag
import scala.annotation.compileTimeOnly
import scala.collection.Factory

@compileTimeOnly("This class only exists at compile-time to expose offload methods")
object compiletime {

  inline def showExpr(inline x: Any): Any = ${ showExprImpl('x) }

  def showExprImpl(x: Expr[Any])(using q: Quotes): Expr[Any] = {
    import quotes.reflect.*
    given Printer[Tree] = Printer.TreeStructure
    pprint.pprintln(x.asTerm) // term => AST
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

  inline def nativeStructOf[A]: NativeStruct[A] = ${ nativeStructOfImpl[A] }

  def nativeStructOfImpl[A: Type](using q: Quotes): Expr[NativeStruct[A]] = {
    import quotes.reflect.*
    println(TypeRepr.of[A].typeSymbol.tree.show)

    implicit val Q = compiler.Quoted(q)

    compiler.Retyper.lowerProductType[A].resolve match {
      case Left(e) => throw e
      case Right(sdef) =>
        val layout = PolyregionCompiler.layoutOf(MsgPack.encode(MsgPack.Versioned(CppCodeGen.AdtHash, sdef)))
        println(s"layout=${layout}")

        val tpeSym = TypeTree.of[A].symbol

        def encodeField(
            buffer: Expr[java.nio.ByteBuffer],
            offset: Expr[Int],
            a: Expr[A],
            named: p.Named,
            m: Member
        ) = {
          val byteOffset = '{ $offset + ${ Expr(m.offsetInBytes.toInt) } }
          val value      = Select.unique(a.asTerm, named.symbol)
          named.tpe match {
            case p.Type.Float  => '{ ${ buffer }.putFloat(${ byteOffset }, ${ value.asExprOf[Float] }); () }
            case p.Type.Double => '{ ${ buffer }.putDouble(${ byteOffset }, ${ value.asExprOf[Double] }); () }

            case p.Type.Bool =>
              '{ ${ buffer }.put(${ byteOffset }, (if (!${ value.asExprOf[Boolean] }) 0 else 1).toByte); () }
            case p.Type.Byte  => '{ ${ buffer }.put(${ byteOffset }, ${ value.asExprOf[Byte] }); () }
            case p.Type.Char  => '{ ${ buffer }.putChar(${ byteOffset }, ${ value.asExprOf[Char] }); () }
            case p.Type.Short => '{ ${ buffer }.putShort(${ byteOffset }, ${ value.asExprOf[Short] }); () }
            case p.Type.Int   => '{ ${ buffer }.putInt(${ byteOffset }, ${ value.asExprOf[Int] }); () }
            case p.Type.Long  => '{ ${ buffer }.putLong(${ byteOffset }, ${ value.asExprOf[Long] }); () }

            case p.Type.String           => ???
            case p.Type.Unit             => ???
            case p.Type.Struct(name)     => ???
            case p.Type.Array(component) => ???
          }

        }

        def decodeField(buffer: Expr[java.nio.ByteBuffer], offset: Expr[Int], named: p.Named, m: Member) = {
          val byteOffset = '{ $offset + ${ Expr(m.offsetInBytes.toInt) } }
          named.tpe match {
            case p.Type.Float  => '{ ${ buffer }.getFloat(${ byteOffset }) }
            case p.Type.Double => '{ ${ buffer }.getDouble(${ byteOffset }) }

            case p.Type.Bool  => '{ if (${ buffer }.get(${ byteOffset }) == 0) false else true }
            case p.Type.Byte  => '{ ${ buffer }.get(${ byteOffset }) }
            case p.Type.Char  => '{ ${ buffer }.getChar(${ byteOffset }) }
            case p.Type.Short => '{ ${ buffer }.getShort(${ byteOffset }) }
            case p.Type.Int   => '{ ${ buffer }.getInt(${ byteOffset }) }
            case p.Type.Long  => '{ ${ buffer }.getLong(${ byteOffset }) }

            case p.Type.String           => ???
            case p.Type.Unit             => ???
            case p.Type.Struct(name)     => ???
            case p.Type.Array(component) => ???
          }

        }

        val fields = sdef.members.zip(layout.members)
        '{
          new NativeStruct[A] {
            override val name        = ${ Expr(tpeSym.fullName) }
            override val sizeInBytes = ${ Expr(layout.sizeInBytes.toInt) }
            // override def member                                          = Vector()
            override def encode(buffer: java.nio.ByteBuffer, index: Int, a: A): Unit = {
              val offset = sizeInBytes * index
              ${ Expr.ofList(fields.map((named, member) => encodeField('buffer, 'offset, 'a, named, member))) }
            }

            override def decode(buffer: java.nio.ByteBuffer, index: Int): A = {
              val offset = sizeInBytes * index
              ${
                Select
                  .unique(New(TypeTree.of[A]), "<init>")
                  .appliedToArgs(fields.map((named, member) => decodeField('buffer, 'offset, named, member).asTerm))
                  .asExprOf[A]
              }
            }
          }
        }
    }

  }

  // def deriveCaseClassNativeStructEncode[A](expr: Expr[A], l: polyregion.Layout)(using quotes: Quotes): Expr[Unit] = {
  //   import quotes.reflect.*
  //   val fields: List[Symbol] = TypeTree.of[A].symbol.caseFields

  //   val receiver = expr.asTerm

  //   def encodeField(t: Term, field: Symbol) =
  //     fields.map { field =>
  //       Select(receiver, field)
  //     }

  //   /** Create a quoted String representation of a given field of the case class */
  //   def showField(caseClassTerm: Term, field: Symbol): Expr[String] =
  //     val fieldValDef: ValDef = field.tree.asInstanceOf[ValDef] // TODO remove cast
  //     val fieldTpe: TypeRepr  = fieldValDef.tpt.tpe
  //     val fieldName: String   = fieldValDef.name

  //     val tcl: Term             = lookupShowFor(fieldTpe)      // Show[$fieldTpe]
  //     val fieldValue: Term      = Select(caseClassTerm, field) // v.field
  //     val strRepr: Expr[String] = applyShow(tcl, fieldValue).asExprOf[String]
  //     '{ s"${${ Expr(fieldName) }}: ${${ strRepr }}" } // summon[Show[$fieldTpe]].show(v.field)

  //   def showBody(v: Expr[A], buffer: java.nio.ByteBuffer): Expr[String] =
  //     val vTerm: Term                     = v.asTerm
  //     val valuesExprs: List[Expr[String]] = fields.map(showField(vTerm, _))
  //     val exprOfList: Expr[List[String]]  = Expr.ofList(valuesExprs)
  //     '{ $exprOfList.mkString(", ") }

  //   '{
  //     new Show[T] {
  //       override def show(t: T): String =
  //         s"{ ${${ showBody('{ t }) }} }"
  //     }
  //   }
  // }

  inline def foreachJVMPar(inline range: Range)(inline x: Int => Unit)(using ec: ExecutionContext): Unit = {
    val n     = java.lang.Runtime.getRuntime.availableProcessors()
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

  inline def reduceJVMPar[A](inline range: Range) //
  (inline empty: A, inline f: Int => A)(inline g: (A, A) => A)(using ec: ExecutionContext): A = {
    val n     = java.lang.Runtime.getRuntime.availableProcessors()
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

  inline def foreachPar(inline range: Range)(inline x: Int => Unit)(using ec: ExecutionContext): Unit = {
    val n     = java.lang.Runtime.getRuntime.availableProcessors()
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

  inline def reducePar[A](inline range: Range) //
  (inline empty: A, inline f: Int => A)(inline g: (A, A) => A)(using ec: ExecutionContext): A = {
    val n     = java.lang.Runtime.getRuntime.availableProcessors()
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

  private def tpeAsRuntimeTpe(t: p.Type)(using Quotes) = t match {
    case p.Type.Bool      => '{ PolyregionRuntime.TYPE_BOOL }
    case p.Type.Byte      => '{ PolyregionRuntime.TYPE_BYTE }
    case p.Type.Char      => '{ PolyregionRuntime.TYPE_CHAR }
    case p.Type.Short     => '{ PolyregionRuntime.TYPE_SHORT }
    case p.Type.Int       => '{ PolyregionRuntime.TYPE_INT }
    case p.Type.Long      => '{ PolyregionRuntime.TYPE_LONG }
    case p.Type.Float     => '{ PolyregionRuntime.TYPE_FLOAT }
    case p.Type.Double    => '{ PolyregionRuntime.TYPE_DOUBLE }
    case p.Type.Array(_)  => '{ PolyregionRuntime.TYPE_PTR }
    case p.Type.Struct(_) => '{ PolyregionRuntime.TYPE_PTR }
    case p.Type.Unit      => '{ PolyregionRuntime.TYPE_VOID }
    case unknown =>
      Integer.val
      println(s"tpeAsRuntimeTpe ??? = $unknown ")
      ???
  }

  def noComplexReturn(kind: String): Nothing = throw new AssertionError(
    s"Compiler bug: returning $kind is not possible, " +
      "it should have been transformed to an out param in one of the passes."
  )

  private def offloadImpl[A: Type](x: Expr[Any])(using q: Quotes): Expr[A] = {
    implicit val Q = compiler.Quoted(q)
    val result = for {
      (outReturnParams, captures, prog) <- compiler.Compiler.compileExpr(x)
      serialisedAst <- Either.catchNonFatal(MsgPack.encode(MsgPack.Versioned(CppCodeGen.AdtHash, prog)))
      // _ <- Either.catchNonFatal(throw new RuntimeException("STOP"))
      // layout <- Either.catchNonFatal(PolyregionCompiler.layoutOf(MsgPack.encode(MsgPack.Versioned(CppCodeGen.AdtHash, prog.defs))))
      //   _= println(s"layout=${layout}")
      c <- Either.catchNonFatal(PolyregionCompiler.compile(serialisedAst, true, PolyregionCompiler.BACKEND_LLVM))
    } yield {

      println(s"Program=${c.program.length}")
      println(s"Elapsed=\n${c.events.sortBy(_.epochMillis).mkString("\n")}")
      println(s"Messages=\n  ${c.messages}")

      val programBytesExpr = Expr(c.program)
      val astBytesExpr     = Expr(serialisedAst)
      val fnName           = Expr("lambda")

      val (rtnBufferTpe, rtnBufferExpr) = (
        tpeAsRuntimeTpe(prog.entry.rtn),
        prog.entry.rtn match {
          case p.Type.Unit          => '{ Buffer.empty[Int] }
          case p.Type.Float         => '{ Buffer.ref[Float] }
          case p.Type.Double        => '{ Buffer.ref[Double] }
          case p.Type.Bool          => '{ Buffer.ref[Boolean] }
          case p.Type.Byte          => '{ Buffer.ref[Byte] }
          case p.Type.Char          => '{ Buffer.ref[Char] }
          case p.Type.Short         => '{ Buffer.ref[Short] }
          case p.Type.Int           => '{ Buffer.ref[Int] }
          case p.Type.Long          => '{ Buffer.ref[Long] }
          case x @ p.Type.Struct(_) => noComplexReturn(s"struct (${x.repr})")
          case x @ p.Type.Array(_)  => noComplexReturn(s"array (${x.repr})")
        }
      )

      println(s"out=${outReturnParams}")

      val outReturnParamExpr = outReturnParams match {
        case Nil => None
        case (t @ p.Type.Struct(sym)) :: Nil =>
          val imp = Expr.summon[NativeStruct[A]] match {
            case Some(x) => x
            case None =>
              Q.report.errorAndAbort(
                s"No implicit found for return struct value ${Q.TypeRepr.of[NativeStruct[A]].show}"
              )
          }
          // val x = TypeRepr.of[A].asType
          Some(
            (
              t,
              '{
                println(s"imp=${${ imp }}")

                val u = Buffer.ofZeroedAny(1)(using $imp.asInstanceOf[NativeStruct[Any]])
                println(s"u=$u")
                u.asInstanceOf[Buffer[A]]
              }
            )
          )
        case (t @ p.Type.Array(comp)) :: Nil =>
          // nested arrays will be pointers

          val imp = Expr.summon[Factory[_, A]] match {
            case Some(x) => x
            case None =>
              Q.report.errorAndAbort(
                s"No implicit found for return collection value ${Q.TypeRepr.of[Factory[_, A]].show}"
              )
          }

          println(">>>" + imp.show)

          Some(
            (
              t,
              '{
                val u = Buffer.ofZeroedAny(1)(using $imp.asInstanceOf[NativeStruct[Any]])
                println(s"u=$u")
                // '{ $imp.fromSpecific() }
                u.asInstanceOf[Buffer[A]]
              }
            )
          )

        case bad :: Nil =>
          throw new AssertionError(s"Compiler bug: expecting struct or array out return param but was $bad")
        case xs => throw new AssertionError(s"Compiler bug: more than one out return param ($xs)!?")
      }

      val captureExprs = captures.map { (ident, tpe) =>
        val expr = ident.asExpr
        val wrapped = tpe match {
          case p.Type.Unit   => '{ Buffer.empty[Int] }
          case p.Type.Bool   => '{ Buffer[Boolean](${ expr.asExprOf[Boolean] }) }
          case p.Type.Byte   => '{ Buffer[Byte](${ expr.asExprOf[Byte] }) }
          case p.Type.Char   => '{ Buffer[Char](${ expr.asExprOf[Char] }) }
          case p.Type.Short  => '{ Buffer[Short](${ expr.asExprOf[Short] }) }
          case p.Type.Int    => '{ Buffer[Int](${ expr.asExprOf[Int] }) }
          case p.Type.Long   => '{ Buffer[Long](${ expr.asExprOf[Long] }) }
          case p.Type.Float  => '{ Buffer[Float](${ expr.asExprOf[Float] }) }
          case p.Type.Double => '{ Buffer[Double](${ expr.asExprOf[Double] }) }
          case p.Type.Struct(_) =>
            val tc = Q.TypeRepr.of[NativeStruct].appliedTo(ident.tpe.widenTermRefByName)
            val imp = Q.Implicits.search(tc) match {
              case ok: Q.ImplicitSearchSuccess   => ok.tree.asExpr
              case fail: Q.ImplicitSearchFailure => Q.report.errorAndAbort(fail.explanation, ident.asExpr)
            }
            '{ Buffer.refAny(${ expr.asExprOf[Any] })(using $imp.asInstanceOf[NativeStruct[Any]]) }
          case p.Type.Array(p.Type.Bool)      => expr.asExprOf[Buffer[Boolean]]
          case p.Type.Array(p.Type.Char)      => expr.asExprOf[Buffer[Char]]
          case p.Type.Array(p.Type.Byte)      => expr.asExprOf[Buffer[Byte]]
          case p.Type.Array(p.Type.Short)     => expr.asExprOf[Buffer[Short]]
          case p.Type.Array(p.Type.Int)       => expr.asExprOf[Buffer[Int]]
          case p.Type.Array(p.Type.Long)      => expr.asExprOf[Buffer[Long]]
          case p.Type.Array(p.Type.Float)     => expr.asExprOf[Buffer[Float]]
          case p.Type.Array(p.Type.Double)    => expr.asExprOf[Buffer[Double]]
          case p.Type.Array(p.Type.Struct(s)) => expr.asExprOf[Buffer[_]]
          case unknown =>
            println(s"???= $unknown ")
            ???
        }
        '{
          // println(s"mkBuffer: ${${ wrapped }}")
          ${ wrapped }.buffer
        }
      }
      val captureTps = captures.map((_, t) => tpeAsRuntimeTpe(t))

      val code = outReturnParamExpr match {
        case Some((t, expr)) =>
          '{
            val programBytes = $programBytesExpr

            val rtnBuffer = ${ rtnBufferExpr }
            val rtnType   = ${ rtnBufferTpe }

            val outReturn = ${ expr }

            val argTypes   = Array(${ Varargs(tpeAsRuntimeTpe(t) :: captureTps) }*)
            val argBuffers = Array(${ Varargs('{ outReturn.buffer } :: captureExprs) }*)

            PolyregionRuntime.invoke(programBytes, ${ fnName }, rtnType, rtnBuffer.buffer, argTypes, argBuffers)

            ${
              t match {
                case s @ p.Type.Struct(_) => '{ outReturn(0) }
                case s @ p.Type.Array(_)  => '{ ().asInstanceOf[A] }
                case _                    => ???
              }
            }
          }
        case None =>
          '{
            val programBytes = $programBytesExpr
            val rtnBuffer    = ${ rtnBufferExpr }
            val rtnType      = ${ rtnBufferTpe }

            val argTypes   = Array(${ Varargs(captureTps) }*)
            val argBuffers = Array(${ Varargs(captureExprs) }*)

            PolyregionRuntime.invoke(programBytes, ${ fnName }, rtnType, rtnBuffer.buffer, argTypes, argBuffers)
            ${
              prog.entry.rtn match {
                case s @ p.Type.Struct(_) => noComplexReturn(s"struct ${s}")
                case p.Type.Unit          => '{ ().asInstanceOf[A] }
                case _                    => '{ rtnBuffer(0).asInstanceOf[A] }
              }
            }
          }

      }

      // println("Code=" + code.show)
      code
    }

//    println(prog.toByteArray.mkString(" "))
//    println(Program.parseFrom(prog.toByteArray).toProtoString)

//    val b = '{ val b = polyregion.Runtime.FFIInvocationBuilder() }

    result match {
      case Left(e)  => throw e
      case Right(x) => x
    }

  }

}
