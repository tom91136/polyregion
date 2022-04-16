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

    implicit val Q = Quoted(q)

    Retyper.lowerClassType[A] match {
      case Left(e) => throw e
      case Right(sdef) =>
        val layout = PolyregionCompiler.layoutOf(MsgPack.encode(MsgPack.Versioned(CppSourceMirror.AdtHash, sdef)))
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
    case p.Type.Bool      => '{ polyregion.PolyregionRuntime.TYPE_BOOL }
    case p.Type.Byte      => '{ polyregion.PolyregionRuntime.TYPE_BYTE }
    case p.Type.Char      => '{ polyregion.PolyregionRuntime.TYPE_CHAR }
    case p.Type.Short     => '{ polyregion.PolyregionRuntime.TYPE_SHORT }
    case p.Type.Int       => '{ polyregion.PolyregionRuntime.TYPE_INT }
    case p.Type.Long      => '{ polyregion.PolyregionRuntime.TYPE_LONG }
    case p.Type.Float     => '{ polyregion.PolyregionRuntime.TYPE_FLOAT }
    case p.Type.Double    => '{ polyregion.PolyregionRuntime.TYPE_DOUBLE }
    case p.Type.Array(_)  => '{ polyregion.PolyregionRuntime.TYPE_PTR }
    case p.Type.Struct(_) => '{ polyregion.PolyregionRuntime.TYPE_PTR }
    case p.Type.Unit      => '{ polyregion.PolyregionRuntime.TYPE_VOID }
    case unknown =>
      println(s"tpeAsRuntimeTpe ??? = $unknown ")
      ???
  }

  def noComplexReturn(kind: String): Nothing = throw new AssertionError(
    s"Compiler bug: returning $kind is not possible, " +
      "it should have been transformed to an out param in one of the passes."
  )

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

      // println(s"Program=${c.program.length}")
      // println(s"Elapsed=\n${c.events.sortBy(_.epochMillis).mkString("\n")}")
      // println(s"Messages=\n  ${c.messages}")

      val programBytesExpr = Expr(c.program)
      val astBytesExpr     = Expr(serialisedAst)
      val fnName           = Expr("lambda")

      transparent inline def liftTpe(t: p.Type) = t match {
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

      val captureExprs = captures.map { (name, ident) =>
        val expr = ident.asExpr
        val wrapped = name.tpe match {
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
          case p.Type.Array(p.Type.Unit)      => expr.asExprOf[Buffer[Int]] // TODO test me
          case p.Type.Array(p.Type.Struct(s)) => expr.asExprOf[Buffer[_]]
          case unknown =>
            println(s"???= $unknown ")
            ???
        }
        '{
          // println(s"mkBuffer: ${${ wrapped }}")
          ${ wrapped }.backingBuffer
        }
      }
      val captureTps = captures.map((name, _) => tpeAsRuntimeTpe(name.tpe))

      def wrap(buffer: Expr[java.nio.ByteBuffer], comp: p.Type) =
        comp match {
          case p.Type.Unit   => '{ Buffer.view[Unit](${ buffer }) }
          case p.Type.Float  => '{ Buffer.view[Float](${ buffer }) }
          case p.Type.Double => '{ Buffer.view[Double](${ buffer }) }
          case p.Type.Bool   => '{ Buffer.view[Boolean](${ buffer }) }
          case p.Type.Byte   => '{ Buffer.view[Byte](${ buffer }) }
          case p.Type.Char   => '{ Buffer.view[Char](${ buffer }) }
          case p.Type.Short  => '{ Buffer.view[Short](${ buffer }) }
          case p.Type.Int    => '{ Buffer.view[Int](${ buffer }) }
          case p.Type.Long   => '{ Buffer.view[Long](${ buffer }) }
          case _             => ???
        }

      val code = '{

        val bytes      = $programBytesExpr
        val argTypes   = Array(${ Varargs(captureTps) }*)
        val argBuffers = Array(${ Varargs(captureExprs) }*)

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
            case x @ p.Type.Struct(_) =>
              val imp = Expr.summon[NativeStruct[A]] match {
                case None    => Q.report.errorAndAbort(s"Cannot find NativeStruct for type ${Type.of[A]}")
                case Some(x) => x
              }
              '{
                var buffer = polyregion.PolyregionRuntime.invokeObject(bytes, $fnName, argTypes, argBuffers, -1)
                Buffer.structViewAny[A](buffer)(using $imp)(0)
              }
            case p.Type.Array(comp) =>
              '{
                var buffer = polyregion.PolyregionRuntime.invokeObject(bytes, $fnName, argTypes, argBuffers, -1)
                ${
                  Type.of[A] match {
                    case '[Buffer[a]] => wrap('{ buffer }, comp) // passthrough
                    case '[Array[a]]  => '{ ${ wrap('{ buffer }, comp) }.copyToArray }
                    case m            => ???
                  }
                }
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
      case Left(e) => throw e
      case Right(x) =>
        // println("Code=" + x.show)
        x
    }

  }

}
