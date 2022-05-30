package polyregion.scala

import polyregion.ast.{CppSourceMirror, MsgPack, PolyAst as p, *}
import polyregion.scala as srt
import polyregion.jvm.{runtime => rt}
import polyregion.jvm.{compiler => ct}

import java.nio.ByteBuffer
import java.nio.file.Paths
import java.util.concurrent.atomic.AtomicReference
import scala.annotation.{compileTimeOnly, tailrec}
import scala.collection.immutable.VectorMap
import scala.quoted.*
import scala.util.Try

@compileTimeOnly("This class only exists at compile-time to expose offload methods")
object compiletime {

//  inline def showExpr(inline x: Any): Any = ${ showExprImpl('x) }
//  def showExprImpl(x: Expr[Any])(using q: Quotes): Expr[Any] = {
//    import q.reflect.*
//    given Printer[Tree] = Printer.TreeAnsiCode
//    println(x.show)
//    pprint.pprintln(x.asTerm) // term => AST
//
//    implicit val Q = Quoted(q)
//    val is = Q.collectTree(x.asTerm) {
//      case s: Q.Select => s :: Nil
//      case s: Q.Ident  => s :: Nil
//      case _           => Nil
//    }
//    println("===")
//    println(s"IS=${is
//      .filter(x => x.symbol.isDefDef || true)
//      .reverse
//      .map(x => x -> x.tpe.dealias.widenTermRefByName.simplified)
//      .map((x, tpe) => s"-> $x : ${x.show} `${x.symbol.fullName}` : ${tpe.show}\n\t${tpe}")
//      .mkString("\n")}")
//    println("===")
//    x
//  }
//
//  inline def showTpe[B]: Unit = ${ showTpeImpl[B] }
//  def showTpeImpl[B: Type](using q: Quotes): Expr[Unit] = {
//    import q.reflect.*
//    println(TypeRepr.of[B].typeSymbol.tree.show)
//    import pprint.*
//    pprint.pprintln(TypeRepr.of[B].typeSymbol)
//    '{}
//  }
//
//  private[polyregion] inline def symbolNameOf[B]: String = ${ symbolNameOfImpl[B] }
//  private def symbolNameOfImpl[B: Type](using q: Quotes): Expr[String] = Expr(
//    q.reflect.TypeRepr.of[B].typeSymbol.fullName
//  )

//  inline def foreachJVMPar(inline range: Range, inline n: Int = java.lang.Runtime.getRuntime.availableProcessors())(
//      inline x: Int => Unit
//  )(using ec: scala.concurrent.ExecutionContext): Unit = {
//    val latch = new java.util.concurrent.CountDownLatch(n)
//    Runtime.splitStatic(range)(n).foreach { r =>
//      ec.execute { () =>
//        try foreachJVM(r)(x)
//        finally latch.countDown()
//      }
//    }
//    latch.await()
//  }
//
//  inline def foreachJVM(inline range: Range)(inline x: Int => Unit): Unit = {
//    val start = range.start
//    val bound = if (range.isInclusive) range.end - 1 else range.end
//    val step  = range.step
//    var i     = start
//    while (i < bound) {
//      x(i)
//      i += step
//    }
//  }
//
//  inline def reduceJVM[A](inline range: Range) //
//  (inline empty: A, inline f: Int => A)(inline g: (A, A) => A): A = {
//    val start = range.start
//    val bound = if (range.isInclusive) range.end - 1 else range.end
//    val step  = range.step
//    var i     = start
//    var acc   = empty
//    while (i < bound) {
//      acc = g(f(i), acc)
//      i += step
//    }
//    acc
//  }
//
//  inline def reduceJVMPar[A](
//      inline range: Range,
//      inline n: Int = java.lang.Runtime.getRuntime.availableProcessors()
//  ) //
//  (inline empty: A, inline f: Int => A)(inline g: (A, A) => A)(using ec: scala.concurrent.ExecutionContext): A = {
//    val latch = new java.util.concurrent.CountDownLatch(n)
//    val acc   = new AtomicReference[A](empty)
//    Runtime.splitStatic(range)(n).foreach { r =>
//      ec.execute { () =>
//        val x = reduceJVM(r)(empty, f)(g)
//        try acc.getAndUpdate(g(_, x))
//        finally latch.countDown()
//      }
//    }
//    latch.await()
//    acc.get()
//  }
//
//  inline def foreach(inline range: Range)(inline x: Int => Unit): Unit = {
//    val start = range.start
//    val bound = if (range.isInclusive) range.end - 1 else range.end
//    val step  = range.step
//    offload {
//      var i = start
//      while (i < bound) {
//        x(i)
//        i += step
//      }
//    }
//  }
//
//  inline def foreachPar(inline range: Range, inline n: Int = java.lang.Runtime.getRuntime.availableProcessors())(
//      inline x: Int => Unit
//  )(using ec: scala.concurrent.ExecutionContext): Unit = {
//    val latch = new java.util.concurrent.CountDownLatch(n)
//    Runtime.splitStatic(range)(n).foreach { r =>
//      ec.execute { () =>
//        try foreach(r)(x)
//        finally latch.countDown()
//      }
//    }
//    latch.await()
//  }
//
//  inline def reduce[A](inline range: Range) //
//  (inline empty: A, inline f: Int => A)(inline g: (A, A) => A): A = {
//    val start = range.start
//    val bound = if (range.isInclusive) range.end - 1 else range.end
//    val step  = range.step
//    offload {
//      var i   = start
//      var acc = empty
//      while (i < bound) {
//        acc = g(f(i), acc)
//        i += step
//      }
//      acc
//    }
//  }
//
//  inline def reducePar[A](inline range: Range, inline n: Int = java.lang.Runtime.getRuntime.availableProcessors()) //
//  (inline empty: A, inline f: Int => A)(inline g: (A, A) => A)(using ec: scala.concurrent.ExecutionContext): A = {
//    val latch = new java.util.concurrent.CountDownLatch(n)
//    val acc   = new AtomicReference[A](empty)
//    Runtime.splitStatic(range)(n).foreach { r =>
//      ec.execute { () =>
//        val x = reduce(r)(empty, f)(g)
//        try acc.getAndUpdate(g(_, x))
//        finally latch.countDown()
//      }
//    }
//    latch.await()
//    acc.get()
//  }

//  inline def deriveNativeStruct[A]: NativeStruct[A] = ${ deriveNativeStructImpl[A] }
//
//  def deriveNativeStructImpl[A: Type](using q: Quotes): Expr[NativeStruct[A]] = {
//    given Q: Quoted = Quoted(q)
//    mkNativeStruct(Q.TypeRepr.of[A]).asExprOf[NativeStruct[A]]
//  }
//
//  def mkNativeStruct(using q: Quoted)(repr: q.TypeRepr): Expr[NativeStruct[Any]] = {
//    import q.given
//    repr.asType match {
//      case '[a] =>
//        val layout = Pickler.layoutOf(repr)
//
//        '{
//          new NativeStruct[a] {
//            override val name        = ${ Expr(repr.typeSymbol.fullName) }
//            override val sizeInBytes = ${ Expr(layout.sizeInBytes.toInt) }
//            override def encode(buffer: java.nio.ByteBuffer, index: Int, a: a): Unit =
//              ${ Pickler.writeStruct('buffer, 'index, repr, 'a) }
//
//            override def decode(buffer: java.nio.ByteBuffer, index: Int): a =
//              ${ Pickler.readStruct('buffer, 'index, repr).asExprOf[a] }
//          }
//        }.asExprOf[NativeStruct[Any]]
//    }
//  }

  type Device = rt.Device
  type Queue  = rt.Device.Queue

//  inline def offload[A](inline x: => A): A = ${ offloadImpl[A]('x) }

  inline def offload[A](inline queue: Queue, inline cb: Runnable)(inline f: => A): A = ${
    offloadImpl[A]('queue, 'f, 'cb)
  }

//  inline def offload(inline d: Device, inline x: Range, inline y: Range = Range(0, 0), inline z: Range = Range(0, 0))
//  /*             */ (inline f: => (Int, Int, Int) => Unit): Unit = ${ offloadImpl[Unit](d, 'x, 'y, 'z, 'f) }

  private def offloadImpl[A: Type](queue: Expr[Queue], f: Expr[Any], cb: Expr[Runnable])(using
      q: Quotes
  ): Expr[A] = {
    implicit val Q = Quoted(q)
    val result = for {
      (captures, prog, log) <- Compiler.compileExpr(f)
      _ = println(log.render)
      serialisedAst <- Try(MsgPack.encode(MsgPack.Versioned(CppSourceMirror.AdtHash, prog))).toEither
      options = ct.Options(ct.Compiler.TargetObjectLLVM_x86_64, "")
      c <- Try((ct.Compiler.compile(serialisedAst, true, options))).toEither
    } yield {

      println(s"Messages=\n  ${c.messages}")
      println(s"Program=${c.program.length}")
      println(s"Elapsed=\n${c.events.sortBy(_.epochMillis).mkString("\n")}")

      val fnName     = Expr("lambda")
      val moduleName = Expr(prog.entry.name.repr)

      val (captureTpeOrdinal, captureTpeSizes) = captures.map { (name, _) =>
        val tpe = Pickler.tpeAsRuntimeTpe(name.tpe)
        tpe.value -> tpe.sizeInBytes
      }.unzip

      def allocBuffer(size: Expr[Int]) = '{
        java.nio.ByteBuffer.allocateDirect($size).order(java.nio.ByteOrder.nativeOrder)
      }

      val bufferExprs = captures.zipWithIndex.foldLeft(VectorMap.empty[Int, (Int, Expr[ByteBuffer])]) {
        case (acc, ((name, ref), idx)) =>
          (name.tpe, ref.tpe.asType) match {
            case (p.Type.Array(comp), x @ '[srt.Buffer[a]]) =>
              acc + (idx -> (acc.size, '{ ${ ref.asExprOf[x.Underlying] }.backingBuffer }))
            case (p.Type.Array(comp), x @ '[scala.Array[t]]) =>
              acc + (
                idx ->
                  (acc.size, '{
                    val xs     = ${ ref.asExprOf[x.Underlying] }
                    val buffer = ${ allocBuffer('{ ${ Expr(Pickler.sizeOf(comp, Q.TypeRepr.of[t])) } * xs.size }) }
                    ${ Pickler.writeUniform('buffer, '{ 0 }, name.tpe, ref.tpe, ref.asExpr) }
                    buffer
                  })
              )
            case (p.Type.Array(comp), x @ '[scala.collection.Seq[t]]) =>
              acc + (
                idx ->
                  (acc.size, '{
                    val xs     = ${ ref.asExprOf[x.Underlying] }
                    val buffer = ${ allocBuffer('{ ${ Expr(Pickler.sizeOf(comp, Q.TypeRepr.of[t])) } * xs.size }) }
                    ${ Pickler.writeUniform('buffer, '{ 0 }, name.tpe, ref.tpe, ref.asExpr) }
                    buffer
                  })
              )
            case _ => acc
          }
      }

      val code = '{

        if (! $queue.device.moduleLoaded($moduleName)) {
          $queue.device.loadModule($moduleName, ${ Expr(c.program) })
        }

        val captureTpeOrdinals = ${ Expr(captureTpeOrdinal.toArray) } :+ 1.toByte /*1 = void*/
        val captureValues = ByteBuffer.allocate(${ Expr(captureTpeSizes.sum) }).order(java.nio.ByteOrder.nativeOrder)

        if ($queue.device.sharedAddressSpace) {

          val capturePointers = rt.Runtime.directBufferPointers(Array(${
            Varargs(bufferExprs.values.map(_._2).toSeq)
          }: _*))

          println(s"Ptrs:${capturePointers.toList}")

          ${
            val (_, stmts) = captures.zipWithIndex.foldLeft((0, List.empty[Expr[Any]])) {
              case ((byteOffset, exprs), ((name, ref), idx)) =>
                val expr = bufferExprs.get(idx) match {
                  case Some((capturePointerIdx, _)) =>
                    '{ captureValues.putLong(${ Expr(byteOffset) }, capturePointers(${ Expr(capturePointerIdx) })) }
                  case None => Pickler.writePrimitiveAtOffset('captureValues, Expr(byteOffset), name.tpe, ref.asExpr)
                }
                (byteOffset + Pickler.tpeAsRuntimeTpe(name.tpe).sizeInBytes, exprs :+ expr)
            }
            Expr.block(stmts, '{ () })
          }

          $queue.enqueueInvokeAsync(
            $moduleName,
            $fnName,
            captureTpeOrdinals,
            captureValues.array,
            rt.Policy(rt.Dim3(1, 1, 1)),
            $cb
          )

        } else {

          val p = $queue.device.malloc(0, rt.Access.RW)

          $queue.enqueueHostToDeviceAsync(???, p, 0, null)

          val pointers = rt.Runtime.directBufferPointers(Array(${
            Varargs(bufferExprs.values.map(_._2).toSeq)
          }: _*))

          $queue.enqueueInvokeAsync(
            $moduleName,
            $fnName,
            captureTpeOrdinals,
            captureValues.array,
            rt.Policy(rt.Dim3(1, 1, 1)),
            $cb
          )

          $queue.enqueueDeviceToHostAsync(p, ???, 0, null)

          $queue.device.free(p)

          ???
        }

        null.asInstanceOf[A]

      }

      // val captureTps = captures.map((name, _) => Pickler.tpeAsRuntimeTpe(name.tpe).value)

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

      java.nio.file.Files.write(
        Paths.get(".").toAbsolutePath.resolve("program.o"),
        c.program,
        java.nio.file.StandardOpenOption.CREATE,
        java.nio.file.StandardOpenOption.TRUNCATE_EXISTING
      )
      // println()

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
