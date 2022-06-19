package polyregion.scala

import polyregion.ast.{CppSourceMirror, MsgPack, PolyAst as p, *}
import polyregion.scala as srt
import polyregion.jvm.{runtime => rt}
import polyregion.jvm.{compiler => cp}

import java.nio.ByteBuffer
import java.nio.file.Paths
import java.util.concurrent.atomic.AtomicReference
import scala.annotation.{compileTimeOnly, tailrec}
import scala.collection.immutable.VectorMap
import scala.quoted.*
import scala.util.Try
import java.nio.ByteOrder

@compileTimeOnly("This class only exists at compile-time to expose offload methods")
object compiletime {

  inline def showExpr(inline x: Any): Any = ${ showExprImpl('x) }
  def showExprImpl(x: Expr[Any])(using q: Quotes): Expr[Any] = {
    import q.reflect.*
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
//
  inline def showTpe[B]: Unit = ${ showTpeImpl[B] }
  def showTpeImpl[B: Type](using q: Quotes): Expr[Unit] = {
    import q.reflect.*
    println(">>" + TypeRepr.of[B].typeSymbol.tree.show)
    println(">>" + TypeRepr.of[B].widenTermRefByName.dealias.typeSymbol.tree.show)
    import pprint.*
    pprint.pprintln(TypeRepr.of[B].typeSymbol)
    '{}
  }
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

//  inline def offload[A](inline x: => A): A = ${ offloadImpl[A]('x) }

  case class ReifiedConfig(target: cp.Target, arch: String, opt: cp.Opt)

  def reifyConfig(x: srt.Config[_, _]): List[ReifiedConfig] =
    x.targets.map((t, o) => ReifiedConfig(t.arch, t.uarch, o.value))

  def reifyConfigFromTpe[C: Type](c: List[ReifiedConfig] = Nil)(using q: Quotes): Result[List[ReifiedConfig]] = {
    import q.reflect.*
    Type.of[C] match {
      case '[srt.Config[target, opt]] =>
        TypeRepr.of[target].widenTermRefByName.dealias.simplified match {
          case Refinement(target, xs, TypeBounds(l @ ConstantType(StringConstant(arch)), h)) if l == h =>
            val vendorTpe   = target.select(TypeRepr.of[polyregion.scala.Target#Arch].typeSymbol).dealias
            val cpTargetTpe = TypeRepr.of[cp.Target]
            if (vendorTpe <:< cpTargetTpe) {
              (ReifiedConfig(
                target = cp.Target.valueOf(vendorTpe.termSymbol.name),
                arch = arch,
                opt = Opt.valueOf(TypeRepr.of[opt].termSymbol.name).value
              ) :: Nil).success
            } else s"Target type `${vendorTpe}` != ${cpTargetTpe} ".fail
          case bad => s"Target type `${bad}` does not contain a String constant refinement or has illegal bounds".fail
        }
      case '[x *: xs] =>
        for {
          x  <- reifyConfigFromTpe[x](c)
          xs <- reifyConfigFromTpe[xs](c)
        } yield x ::: xs
      case '[EmptyTuple] => Nil.success
      case '[illegal] =>
        s"The type `${TypeRepr.of[illegal].show}` cannot be used as a configuration, it must be either a ${TypeRepr.of[srt.Config].show} or a tuple of such type.".fail
    }
  }

  inline def offload[C](inline queue: rt.Device.Queue, inline cb: Callback[Unit])(inline f: => Unit): Unit = ${
    offloadImpl[C]('queue, 'f, 'cb)
  }

//  inline def offload(inline d: Device, inline x: Range, inline y: Range = Range(0, 0), inline z: Range = Range(0, 0))
//  /*             */ (inline f: => (Int, Int, Int) => Unit): Unit = ${ offloadImpl[Unit](d, 'x, 'y, 'z, 'f) }

  private def offloadImpl[C: Type](
      queue: Expr[rt.Device.Queue],
      f: Expr[Any],
      cb: Expr[Callback[Unit]]
  )(using q: Quotes): Expr[Unit] = {
    implicit val Q = Quoted(q)

    val result = for {
      configs                <- reifyConfigFromTpe[C]()
      (captures, prog0, log) <- Compiler.compileExpr(f)
      prog = prog0.copy(entry = prog0.entry.copy(name = p.Sym("lambda")))
      _    = println(log.render)
      _    = println("Configs = " + configs)
      serialisedAst <- Try(MsgPack.encode(MsgPack.Versioned(CppSourceMirror.AdtHash, prog))).toEither
      compiler = cp.Compiler.create()
      c <- Try(
        (compiler.compile(serialisedAst, true, cp.Options.of(configs(0).target, configs(0).arch), configs(0).opt.value))
      ).toEither
    } yield {

      println(s"Messages=\n  ${c.messages}")
      println(s"Features=\n  ${c.features.toList}")
      println(s"Program=${c.program.length}")
      println(s"Elapsed=\n${c.events.sortBy(_.epochMillis).mkString("\n")}")

      val fnName     = Expr(prog.entry.name.repr)
      val moduleName = Expr(prog0.entry.name.repr)

      val (captureTpeOrdinals, captureTpeSizes) = captures.map { (name, _) =>
        val tpe = Pickler.tpeAsRuntimeTpe(name.tpe)
        tpe.value -> tpe.sizeInBytes
      }.unzip
      val returnTpeOrdinal = Pickler.tpeAsRuntimeTpe(prog.entry.rtn).value
      val returnTpeSize    = Pickler.tpeAsRuntimeTpe(prog.entry.rtn).sizeInBytes

      def bindFnValues(target: Expr[ByteBuffer]) = {
        val (_, stmts) = captures.zipWithIndex.foldLeft((0, List.empty[Expr[Any]])) {
          case ((byteOffset, exprs), ((name, ref), idx)) =>
            inline def register[ts: Type, t: Type](arr: p.Type.Array, mutable: Boolean, size: Expr[ts] => Expr[Int]) =
              '{
                $target.putLong(
                  ${ Expr(byteOffset) },
                  $queue.registerAndInvalidateIfAbsent[ts](
                    ${ ref.asExprOf[ts] },
                    xs => ${ Expr(Pickler.sizeOf(compiler, arr.component, Q.TypeRepr.of[t])) } * ${ size('xs) },
                    (xs, bb) => ${ Pickler.putAll(compiler, 'bb, arr, Q.TypeRepr.of[ts], 'xs) },
                    ${
                      if (!mutable) null
                      else '{ (bb, xs) => ${ Pickler.getAllMutable(compiler, 'bb, arr, Q.TypeRepr.of[ts], 'xs) } }
                    },
                    null
                  )
                )
              }

            val expr = (name.tpe, ref.tpe.asType) match {
              case (p.Type.Array(comp), x @ '[srt.Buffer[_]]) =>
                '{
                  $target.putLong(
                    ${ Expr(byteOffset) },
                    $queue.registerAndInvalidateIfAbsent(
                      ${ ref.asExprOf[x.Underlying] },
                      ${ ref.asExprOf[x.Underlying] }.backingBuffer,
                      null
                    )
                  )
                }
              case (arr @ p.Type.Array(_), ts @ '[Array[t]]) =>
                register[ts.Underlying, t](arr, mutable = true, x => '{ $x.length })
              case (arr @ p.Type.Array(_), ts @ '[scala.collection.mutable.Seq[t]]) =>
                register[ts.Underlying, t](arr, mutable = true, x => '{ $x.length })
              case (arr @ p.Type.Array(_), ts @ '[java.util.List[t]]) =>
                register[ts.Underlying, t](arr, mutable = true, x => '{ $x.size })
              case (arr @ p.Type.Array(_), ts @ '[scala.collection.immutable.Seq[t]]) =>
                register[ts.Underlying, t](arr, mutable = false, x => '{ $x.length })
              case (t, _) =>
                Pickler.putPrimitive(target, Expr(byteOffset), t, ref.asExpr)
            }
            (byteOffset + Pickler.tpeAsRuntimeTpe(name.tpe).sizeInBytes, exprs :+ expr)
        }
        Expr.block(stmts, '{ () })
      }

      val code = '{

        val available = $queue.device.features()
        val missing   = scala.collection.mutable.Set[String]()
        ${ Expr.block(c.features.map(f => '{ if (!available.contains(${ Expr(f) })) missing += ${ Expr(f) } }).toList, '{()}) }

        if (missing.nonEmpty) {
          ${ cb }(
            Left(
              new java.lang.RuntimeException(
                s"Device (${$queue.device.name}) does not have the required feature(s): ${missing.mkString(",")}, this device has/have ${available}"
              )
            )
          )
        } else {
          // We got everything
          if (! $queue.device.moduleLoaded($moduleName)) {
            $queue.device.loadModule($moduleName, ${ Expr(c.program) })
          }

          val fnTpeOrdinals = ${ Expr((captureTpeOrdinals :+ returnTpeOrdinal).toArray) }
          val fnValues =
            ByteBuffer.allocate(${ Expr(captureTpeSizes.sum + returnTpeSize) }).order(ByteOrder.nativeOrder)

          ${ bindFnValues('fnValues) }

          // $queue.enqueueHostToDeviceAsync(???, p, 0, null)

          $queue.enqueueInvokeAsync(
            $moduleName,
            $fnName,
            fnTpeOrdinals,
            fnValues.array,
            rt.Policy(rt.Dim3(1, 1, 1)),
            if ($queue.device.sharedAddressSpace) (() => ${ cb }(Right(()))): Runnable else null
          )

          $queue.syncAll(if (! $queue.device.sharedAddressSpace) (() => ${ cb }(Right(()))): Runnable else null)
        }

        // $queue.enqueueDeviceToHostAsync(p, ???, 0, null)

        // $queue.device.free(p)

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
      compiler.close()
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
