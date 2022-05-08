package polyregion.scala

import polyregion.PolyregionCompiler
import polyregion.ast.{CppSourceMirror, MsgPack, PolyAst as p, *}
import polyregion.scala.{NativeStruct, *}

import java.util.concurrent.atomic.AtomicReference
import scala.annotation.{compileTimeOnly, tailrec}
import scala.quoted.*
import scala.util.Try

@compileTimeOnly("This class only exists at compile-time to expose offload methods")
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

  private[polyregion] inline def symbolNameOf[B]: String       = ${ symbolNameOfImpl[B] }
  private def symbolNameOfImpl[B: Type](using q: Quotes): Expr[String] = Expr(q.reflect.TypeRepr.of[B].typeSymbol.fullName)

  inline def foreachJVMPar(inline range: Range, inline n: Int = java.lang.Runtime.getRuntime.availableProcessors())(
      inline x: Int => Unit
  )(using ec: scala.concurrent.ExecutionContext): Unit = {
    val latch = new java.util.concurrent.CountDownLatch(n)
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
  (inline empty: A, inline f: Int => A)(inline g: (A, A) => A)(using ec: scala.concurrent.ExecutionContext): A = {
    val latch = new java.util.concurrent.CountDownLatch(n)
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
  )(using ec: scala.concurrent.ExecutionContext): Unit = {
    val latch = new java.util.concurrent.CountDownLatch(n)
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
  (inline empty: A, inline f: Int => A)(inline g: (A, A) => A)(using ec: scala.concurrent.ExecutionContext): A = {
    val latch = new java.util.concurrent.CountDownLatch(n)
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

  inline def deriveNativeStruct[A]: NativeStruct[A] = ${ deriveNativeStructImpl[A] }

  def deriveNativeStructImpl[A: Type](using q: Quotes): Expr[NativeStruct[A]] = {
    given Q: Quoted = Quoted(q)
    mkNativeStruct(Q.TypeRepr.of[A]).asExprOf[NativeStruct[A]]
  }

  def mkNativeStruct(using q: Quoted)(repr: q.TypeRepr): Expr[NativeStruct[Any]] = {
    import q.given
    repr.asType match {
      case '[a] =>
        val layout = Pickler.layoutOf(repr)

        '{
          new NativeStruct[a] {
            override val name        = ${ Expr(repr.typeSymbol.fullName) }
            override val sizeInBytes = ${ Expr(layout.sizeInBytes.toInt) }
            override def encode(buffer: java.nio.ByteBuffer, index: Int, a: a): Unit =
              ${ Pickler.writeStruct('buffer, 'index, repr, 'a) }

            override def decode(buffer: java.nio.ByteBuffer, index: Int): a =
              ${ Pickler.readStruct('buffer, 'index, repr).asExprOf[a] }
          }
        }.asExprOf[NativeStruct[Any]]
    }
  }

  inline def offload[A](inline x: => A): A = ${ offloadImpl[A]('x) }
  private def offloadImpl[A: Type](x: Expr[Any])(using q: Quotes): Expr[A] = {
    implicit val Q = Quoted(q)
    val result = for {
      (captures, prog, log) <- Compiler.compileExpr(x)
      _ = println(log.render)
      serialisedAst <- Try(MsgPack.encode(MsgPack.Versioned(CppSourceMirror.AdtHash, prog))).toEither
      // _ <- Either.catchNonFatal(throw new RuntimeException("STOP"))
      // layout <- Either.catchNonFatal(PolyregionCompiler.layoutOf(MsgPack.encode(MsgPack.Versioned(CppCodeGen.AdtHash, prog.defs))))
      //   _= println(s"layout=${layout}")

      // c <- Right(new polyregion.Compilation())
      c <- Try((PolyregionCompiler.compile(serialisedAst, true, PolyregionCompiler.BACKEND_LLVM))).toEither
    } yield {

      println(s"Messages=\n  ${c.messages}")
      println(s"Program=${c.program.length}")
      println(s"Elapsed=\n${c.events.sortBy(_.epochMillis).mkString("\n")}")

      val programBytesExpr = Expr(c.program)
      val astBytesExpr     = Expr(serialisedAst)
      val fnName           = Expr("lambda")

      val captureExprs = captures.map { (name, ref) =>
        println(s"Capture ${name.repr} : ${ref}")
        (name.tpe, ref.tpe.asType) match {
          case (p.Type.Array(comp), x @ '[polyregion.scala.Buffer[a]]) =>
            '{ ${ ref.asExprOf[x.Underlying] }.backingBuffer }
          case (p.Type.Array(comp), x @ '[scala.Array[t]]) =>
            '{
              val xs = ${ ref.asExprOf[x.Underlying] }
              val buffer = java.nio.ByteBuffer
                .allocateDirect(${ Expr(Pickler.sizeOf(comp, Q.TypeRepr.of[t])) } * xs.size)
                .order(java.nio.ByteOrder.nativeOrder)
              ${ Pickler.writeUniform('buffer, '{ 0 }, name.tpe, ref.tpe, ref.asExpr) }
              buffer
            }
          case (p.Type.Array(comp), x @ '[scala.collection.Seq[t]]) =>
            '{
              val xs = ${ ref.asExprOf[x.Underlying] }
              val buffer = java.nio.ByteBuffer
                .allocateDirect(${ Expr(Pickler.sizeOf(comp, Q.TypeRepr.of[t])) } * xs.size)
                .order(java.nio.ByteOrder.nativeOrder)
              ${ Pickler.writeUniform('buffer, '{ 0 }, name.tpe, ref.tpe, ref.asExpr) }
              buffer
            }
          case (_, _) =>
            '{
              val buffer = java.nio.ByteBuffer
                .allocateDirect(${ Expr(Pickler.sizeOf(name.tpe, ref.tpe)) })
                .order(java.nio.ByteOrder.nativeOrder)
              ${ Pickler.writeUniform('buffer, '{ 0 }, name.tpe, ref.tpe, ref.asExpr) }
              buffer
            }
        }

      }
      val captureTps = captures.map((name, _) => Pickler.tpeAsRuntimeTpe(name.tpe))

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
        ${ Expr.block(captureTps.zipWithIndex.map((e, i) => '{ argTypes(${ Expr(i) }) = ${ Expr(e) } }), '{ () }) }

        val argBuffers = new Array[java.nio.ByteBuffer](${ Expr(captureExprs.size) })
        ${ Expr.block(captureExprs.zipWithIndex.map((e, i) => '{ argBuffers(${ Expr(i) }) = $e }), '{ () }) }

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
                ${ Pickler.readUniform('buffer, '{ 0 }, x, Q.TypeRepr.of[A]) }
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
                ${ Pickler.readUniform('buffer, '{ 0 }, x, Q.TypeRepr.of[A]) }
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
