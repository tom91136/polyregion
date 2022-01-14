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

object compileTime {

  extension [A](xs: Array[A]) {
    inline def foreach(inline r: Range)(inline f: Int => Unit) =
      ${ offloadImpl('f) }
  }

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

  inline def foreachJVM(inline range: Range)(inline x: Int => Unit): Any =
    range.foreach(x)

  inline def foreach(range: Range)(inline x: Int => Unit): Any = {
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

  inline def offload(inline x: Any): Any = ${ offloadImpl('x) }

  def offloadImpl(x: Expr[Any])(using q: Quotes): Expr[Any] = {
    import quotes.reflect.*
    val xform = new AstTransformer(using q)


    PolyregionCompiler.load()

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
      println(s"Elapsed=\n${c.events.map(e => s"[${e.epochMillis}] ${e.name}: ${e.elapsedNanos}").mkString("\n")}")
      println(s"Messages=\n  ${c.messages}")
      println(s"Asm=\n${c.disassembly}")

      val programBytesExpr = Expr(c.program)
      val astBytesExpr     = Expr(serialisedAst)
      val fnName           = Expr("lambda")

      val tpeAsRuntimeTpe = (t: PolyAst.Type) =>
        t match {
          case PolyAst.Type.Bool     => '{ PolyregionRuntime.TYPE_BOOL }
          case PolyAst.Type.Byte     => '{ PolyregionRuntime.TYPE_BYTE }
          case PolyAst.Type.Char     => '{ PolyregionRuntime.TYPE_CHAR }
          case PolyAst.Type.Short    => '{ PolyregionRuntime.TYPE_SHORT }
          case PolyAst.Type.Int      => '{ PolyregionRuntime.TYPE_INT }
          case PolyAst.Type.Long     => '{ PolyregionRuntime.TYPE_LONG }
          case PolyAst.Type.Float    => '{ PolyregionRuntime.TYPE_FLOAT }
          case PolyAst.Type.Double   => '{ PolyregionRuntime.TYPE_DOUBLE }
          case PolyAst.Type.Array(_) => '{ PolyregionRuntime.TYPE_PTR }
          case PolyAst.Type.Unit     => '{ PolyregionRuntime.TYPE_VOID }
          case unknown =>
            println(s"tpeAsRuntimeTpe ??? = $unknown ")
            ???
        }

      val rtnBufferExpr = fn.rtn match {
        case PolyAst.Type.Unit   => '{ Buffer.nil[Int] } // 0 length return
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

      val captureExprs = captures.map { (ident, tpe) =>
        val expr = ident.asExpr
        val wrapped = tpe match {
          case PolyAst.Type.Unit   => '{ Buffer.nil[Int] }
          case PolyAst.Type.Bool   => '{ Buffer[Boolean](${ expr.asExprOf[Boolean] }) }
          case PolyAst.Type.Byte   => '{ Buffer[Byte](${ expr.asExprOf[Byte] }) }
          case PolyAst.Type.Char   => '{ Buffer[Char](${ expr.asExprOf[Char] }) }
          case PolyAst.Type.Short  => '{ Buffer[Short](${ expr.asExprOf[Short] }) }
          case PolyAst.Type.Int    => '{ Buffer[Int](${ expr.asExprOf[Int] }) }
          case PolyAst.Type.Long   => '{ Buffer[Long](${ expr.asExprOf[Long] }) }
          case PolyAst.Type.Float  => '{ Buffer[Float](${ expr.asExprOf[Float] }) }
          case PolyAst.Type.Double => '{ Buffer[Double](${ expr.asExprOf[Double] }) }

          case PolyAst.Type.Array(PolyAst.Type.Byte)   => expr.asExprOf[Buffer[Byte]]
          case PolyAst.Type.Array(PolyAst.Type.Short)  => expr.asExprOf[Buffer[Short]]
          case PolyAst.Type.Array(PolyAst.Type.Int)    => expr.asExprOf[Buffer[Int]]
          case PolyAst.Type.Array(PolyAst.Type.Long)   => expr.asExprOf[Buffer[Long]]
          case PolyAst.Type.Array(PolyAst.Type.Float)  => expr.asExprOf[Buffer[Float]]
          case PolyAst.Type.Array(PolyAst.Type.Double) => expr.asExprOf[Buffer[Double]]
          case unknown =>
            println(s"???= $unknown ")
            ???
        }
        '{ ${ wrapped }.buffer }
      }

      '{
        val programBytes = $programBytesExpr
        val astBytes     = $astBytesExpr

        println("Program bytes=" + programBytes.size)
        println("PolyAst bytes=" + astBytes.size)

        val rtnBuffer = ${ rtnBufferExpr }.buffer
        val rtnType   = ${ tpeAsRuntimeTpe(fn.rtn) }

        val argTypes   = Array(${ Varargs(fn.args.map(n => tpeAsRuntimeTpe(n.tpe))) }*)
        val argBuffers = Array(${ Varargs(captureExprs) }*)

        println(s"Invoking with ${argTypes.zip(argBuffers).toList}")

        PolyregionRuntime.load()
        PolyregionRuntime.invoke(programBytes, ${ fnName }, rtnType, rtnBuffer, argTypes, argBuffers)
        // Runtime.ingest(data, b.invoke(_))
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
