package polyregion

import scala.quoted.{Expr, Quotes}
import scala.quoted.*
import scala.annotation.tailrec
import scala.reflect.ClassTag
import fansi.ErrorMode.Throw

import java.nio.file.{Files, Paths, StandardOpenOption}
import cats.data.EitherT
import cats.Eval
import cats.syntax.all.*
import cats.data.NonEmptyList
import polyregion.ast.PolyAst
import polyregion.internal.*

import java.lang.reflect.Modifier
import polyregion.data.MsgPack

import java.time.Duration
import java.util.concurrent.CountDownLatch
import scala.collection.immutable.ArraySeq
import scala.concurrent.ExecutionContext

object compiletime {

  // extension [A](xs: Array[A]) {
  //   inline def foreach(inline r: Range)(inline f: Int => Unit) =
  //     ${ offloadImpl('f) }
  // }

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

  def nativeStructOfImpl[A: Type](using Quotes): Expr[NativeStruct[A]] = {
    import quotes.reflect.*
    println(TypeRepr.of[A].typeSymbol.tree.show)

    val xform = new AstTransformer()
    xform.lowerProductType[A].resolve match {
      case Left(e) => throw e
      case Right(sdef) =>
        val layout = PolyregionCompiler.layoutOf(MsgPack.encode(sdef), false)
        println(s"layout=${layout}")

        val tpeSym = TypeTree.of[A].symbol

        def encodeField(
            buffer: Expr[java.nio.ByteBuffer],
            offset: Expr[Int],
            a: Expr[A],
            named: PolyAst.Named,
            m: Member
        ) = {
          val byteOffset = '{ $offset + ${ Expr(m.offsetInBytes.toInt) } }
          val value      = Select.unique(a.asTerm, named.symbol)
          named.tpe match {
            case PolyAst.Type.Float  => '{ ${ buffer }.putFloat(${ byteOffset }, ${ value.asExprOf[Float] }); () }
            case PolyAst.Type.Double => '{ ${ buffer }.putDouble(${ byteOffset }, ${ value.asExprOf[Double] }); () }

            case PolyAst.Type.Bool =>
              '{ ${ buffer }.put(${ byteOffset }, (if (!${ value.asExprOf[Boolean] }) 0 else 1).toByte); () }
            case PolyAst.Type.Byte  => '{ ${ buffer }.put(${ byteOffset }, ${ value.asExprOf[Byte] }); () }
            case PolyAst.Type.Char  => '{ ${ buffer }.putChar(${ byteOffset }, ${ value.asExprOf[Char] }); () }
            case PolyAst.Type.Short => '{ ${ buffer }.putShort(${ byteOffset }, ${ value.asExprOf[Short] }); () }
            case PolyAst.Type.Int   => '{ ${ buffer }.putInt(${ byteOffset }, ${ value.asExprOf[Int] }); () }
            case PolyAst.Type.Long  => '{ ${ buffer }.putLong(${ byteOffset }, ${ value.asExprOf[Long] }); () }

            case PolyAst.Type.String                   => ???
            case PolyAst.Type.Unit                     => ???
            case PolyAst.Type.Struct(name)             => ???
            case PolyAst.Type.Array(component, length) => ???
          }

        }

        def decodeField(buffer: Expr[java.nio.ByteBuffer], offset: Expr[Int], named: PolyAst.Named, m: Member) = {
          val byteOffset = '{ $offset + ${ Expr(m.offsetInBytes.toInt) } }
          named.tpe match {
            case PolyAst.Type.Float  => '{ ${ buffer }.getFloat(${ byteOffset }) }
            case PolyAst.Type.Double => '{ ${ buffer }.getDouble(${ byteOffset }) }

            case PolyAst.Type.Bool  => '{ if (${ buffer }.get(${ byteOffset }) == 0) false else true }
            case PolyAst.Type.Byte  => '{ ${ buffer }.get(${ byteOffset }) }
            case PolyAst.Type.Char  => '{ ${ buffer }.getChar(${ byteOffset }) }
            case PolyAst.Type.Short => '{ ${ buffer }.getShort(${ byteOffset }) }
            case PolyAst.Type.Int   => '{ ${ buffer }.getInt(${ byteOffset }) }
            case PolyAst.Type.Long  => '{ ${ buffer }.getLong(${ byteOffset }) }

            case PolyAst.Type.String                   => ???
            case PolyAst.Type.Unit                     => ???
            case PolyAst.Type.Struct(name)             => ???
            case PolyAst.Type.Array(component, length) => ???
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

  inline def foreachPar(range: Range)(inline x: Int => Unit)(using ec: ExecutionContext): Unit = {
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

  inline def offload[A](inline x: => A): A = ${ offloadImpl[A]('x) }

  private def offloadImpl[A: Type](x: Expr[Any])(using q: Quotes): Expr[A] = {
    import quotes.reflect.*
    val xform = new AstTransformer(using q)

    val result = for {
      (captures, fn) <- xform.lower(x)
      serialisedAst  <- Either.catchNonFatal(MsgPack.encode(fn))
      _ <- Either.catchNonFatal(
        Files.write(
          Paths.get("./ast.bin").toAbsolutePath.normalize(),
          serialisedAst,
          StandardOpenOption.WRITE,
          StandardOpenOption.CREATE,
          StandardOpenOption.TRUNCATE_EXISTING
        )
      )

      c <- Either.catchNonFatal(PolyregionCompiler.compile(serialisedAst, true, PolyregionCompiler.BACKEND_LLVM))

    } yield {

      println(s"Program=${c.program.length}")
      println(s"Elapsed=\n${c.events.sortBy(_.epochMillis).mkString("\n")}")
      println(s"Messages=\n  ${c.messages}")
      println(s"Asm=\n${c.disassembly}")

      val programBytesExpr = Expr(c.program)
      val astBytesExpr     = Expr(serialisedAst)
      val fnName           = Expr("lambda")

      val tpeAsRuntimeTpe = (t: PolyAst.Type) =>
        t match {
          case PolyAst.Type.Bool        => '{ PolyregionRuntime.TYPE_BOOL }
          case PolyAst.Type.Byte        => '{ PolyregionRuntime.TYPE_BYTE }
          case PolyAst.Type.Char        => '{ PolyregionRuntime.TYPE_CHAR }
          case PolyAst.Type.Short       => '{ PolyregionRuntime.TYPE_SHORT }
          case PolyAst.Type.Int         => '{ PolyregionRuntime.TYPE_INT }
          case PolyAst.Type.Long        => '{ PolyregionRuntime.TYPE_LONG }
          case PolyAst.Type.Float       => '{ PolyregionRuntime.TYPE_FLOAT }
          case PolyAst.Type.Double      => '{ PolyregionRuntime.TYPE_DOUBLE }
          case PolyAst.Type.Array(_, _) => '{ PolyregionRuntime.TYPE_PTR }
          case PolyAst.Type.Unit        => '{ PolyregionRuntime.TYPE_VOID }
          case unknown =>
            println(s"tpeAsRuntimeTpe ??? = $unknown ")
            ???
        }

      val rtnBufferExpr = fn.rtn match {
        case PolyAst.Type.Unit   => '{ Buffer.empty[Int] }
        case PolyAst.Type.Float  => '{ Buffer.ref[Float] }
        case PolyAst.Type.Double => '{ Buffer.ref[Double] }
        case PolyAst.Type.Bool   => '{ Buffer.ref[Boolean] }
        case PolyAst.Type.Byte   => '{ Buffer.ref[Byte] }
        case PolyAst.Type.Char   => '{ Buffer.ref[Char] }
        case PolyAst.Type.Short  => '{ Buffer.ref[Short] }
        case PolyAst.Type.Int    => '{ Buffer.ref[Int] }
        case PolyAst.Type.Long   => '{ Buffer.ref[Long] }
        case unknown =>
          println(s"rtnBufferExpr ??? = $unknown ")
          ???
      }

      val rtnExpr = (buffer: Expr[Buffer[?]]) =>
        fn.rtn match {
          case PolyAst.Type.Unit => '{ ().asInstanceOf[A] }
          case _                 => '{ $buffer(0).asInstanceOf[A] }
        }

      val captureExprs = captures.map { (ident, tpe) =>
        val expr = ident.asExpr
        val wrapped = tpe match {
          case PolyAst.Type.Unit   => '{ Buffer.empty[Int] }
          case PolyAst.Type.Bool   => '{ Buffer[Boolean](${ expr.asExprOf[Boolean] }) }
          case PolyAst.Type.Byte   => '{ Buffer[Byte](${ expr.asExprOf[Byte] }) }
          case PolyAst.Type.Char   => '{ Buffer[Char](${ expr.asExprOf[Char] }) }
          case PolyAst.Type.Short  => '{ Buffer[Short](${ expr.asExprOf[Short] }) }
          case PolyAst.Type.Int    => '{ Buffer[Int](${ expr.asExprOf[Int] }) }
          case PolyAst.Type.Long   => '{ Buffer[Long](${ expr.asExprOf[Long] }) }
          case PolyAst.Type.Float  => '{ Buffer[Float](${ expr.asExprOf[Float] }) }
          case PolyAst.Type.Double => '{ Buffer[Double](${ expr.asExprOf[Double] }) }

          case PolyAst.Type.Array(PolyAst.Type.Bool, None)      => expr.asExprOf[Buffer[Boolean]]
          case PolyAst.Type.Array(PolyAst.Type.Char, None)      => expr.asExprOf[Buffer[Char]]
          case PolyAst.Type.Array(PolyAst.Type.Byte, None)      => expr.asExprOf[Buffer[Byte]]
          case PolyAst.Type.Array(PolyAst.Type.Short, None)     => expr.asExprOf[Buffer[Short]]
          case PolyAst.Type.Array(PolyAst.Type.Int, None)       => expr.asExprOf[Buffer[Int]]
          case PolyAst.Type.Array(PolyAst.Type.Long, None)      => expr.asExprOf[Buffer[Long]]
          case PolyAst.Type.Array(PolyAst.Type.Float, None)     => expr.asExprOf[Buffer[Float]]
          case PolyAst.Type.Array(PolyAst.Type.Double, None)    => expr.asExprOf[Buffer[Double]]
          case PolyAst.Type.Array(PolyAst.Type.Struct(s), None) => expr.asExprOf[Buffer[_]]
          case unknown =>
            println(s"???= $unknown ")
            ???
        }
        '{
          // println(s"mkBuffer: ${${ wrapped }}")
          ${ wrapped }.buffer
        }
      }

      '{
        val programBytes = $programBytesExpr
        // val astBytes     = $astBytesExpr

        // println("Program bytes=" + programBytes.size)
        // println("PolyAst bytes=" + astBytes.size)

        val rtnBuffer = ${ rtnBufferExpr }
        val rtnType   = ${ tpeAsRuntimeTpe(fn.rtn) }

        val argTypes   = Array(${ Varargs(fn.args.map(n => tpeAsRuntimeTpe(n.tpe))) }*)
        val argBuffers = Array(${ Varargs(captureExprs) }*)

        // println(s"Invoking with ${argTypes.zip(argBuffers).toList}")

        PolyregionRuntime.invoke(programBytes, ${ fnName }, rtnType, rtnBuffer.buffer, argTypes, argBuffers)

        ${ rtnExpr('rtnBuffer) }
      }
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
