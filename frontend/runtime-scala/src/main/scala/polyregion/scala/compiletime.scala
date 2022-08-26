package polyregion.scala

import cats.syntax.all.*
import polyregion.ast.{CppSourceMirror, MsgPack, PolyAst as p, *}
import polyregion.jvm.{compiler as ct, runtime as rt}
import polyregion.scala as srt

import java.nio.file.Paths
import java.nio.{ByteBuffer, ByteOrder}
import java.util.concurrent.atomic.{AtomicLong, AtomicReference}
import scala.annotation.{compileTimeOnly, tailrec}
import scala.collection.immutable.VectorMap
import scala.collection.mutable.ArrayBuffer
import scala.quoted.*
import scala.util.Try
import cats.Eval
import polyregion.ast.PolyAst.Sym

@compileTimeOnly("This class only exists at compile-time to expose offload methods")
object compiletime {

  def time[R](block: => R): R = {
    val t0     = System.nanoTime()
    val result = block // call-by-name
    val t1     = System.nanoTime()
    println("Elapsed time: " + ((t1 - t0) / 1.0e6) + "ms")
    result
  }

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

    val ignore = Set(
      "module-info",
      "module-info$"
    )

    inline def declarationsFast(x: Symbol)(inline effect: Symbol => Unit) = {
      var current: List[Symbol] = x :: Nil
      while (current != Nil) {
        var next: List[Symbol] = Nil
        current.foreach { c =>
          if (
            (
              c.isPackageDef && !ignore.contains(c.fullName)
            ) || (c.isClassDef && !ignore.contains(c.fullName) && c.exists && !c.isNoSymbol)
          ) {

            def doIt =
              try {
                effect(c)
                next =
                  try c.declarations ::: next
                  catch { case u => next }
              } catch {
                case x => () // x.printStackTrace()
              }

            c.name.lastIndexOf("$") match {

              case -1 =>
                doIt

              case n =>
                val id = c.name.substring(n + 1)
                if (id.nonEmpty && id.forall(_.isDigit)) println(s"## ${id} => ${c.fullName}")
                else doIt // println("miss = "+c.fullName)

            }

          }
        }
        current = next
      }
    }
    object C {
      var acc = ArrayBuffer[Symbol]()
    }

    // def declarations(x: Symbol) = LazyList
    //   .unfold(LazyList(x)) { xs =>
    //     // println(s"Read ${xs.toList}")
    //     val ys = xs
    //       .filter(x => x.isClassDef || x.isPackageDef)
    //       .filterNot(x => x == defn.ScalaPackage)
    //       .filterNot(x => x == defn.JavaLangPackage)
    //     // .filterNot(x => x.isPackageDef && x.name == "sun")
    //     // .filterNot(x => x.isPackageDef && x.name == "java")
    //     // .filterNot(x => x.isPackageDef && x.name == "javax")
    //     // .filterNot(x => x.isPackageDef && x.name == "jdk")
    //     // .filterNot(x => x.isPackageDef && x.name == "netscape")
    //     // .filterNot(x => x.isPackageDef && x.name == "scala")
    //     // .filterNot(x => x.isPackageDef && x.name == "org")
    //     // .filterNot(x => x.isPackageDef && x.name == "com")

    //     val zs = ys.flatMap(c =>
    //       try c.declarations
    //       catch { case u => LazyList.empty }
    //     )
    //     if (ys.isEmpty) None else Some((ys, zs))
    //   }
    //   .flatten

    println(s"Prev=${C.acc.size}")

    // println(s"C=${Class.forName("com.google.common.io.BaseEncoding")}")

    // println(this.getClass.getResourceAsStream("com/google/common/io/BaseEncoding.class"))

    // println("~~~ "+Symbol.requiredClass("com.google.common.io.BaseEncoding") )

    // declarationsFast(Symbol.requiredPackage("polyregion.foo"))(x => println(">>"+x.fullName))
    // println("~~~ "+Symbol.requiredClass("polyregion.foo.WeakIdentityHashMap$").declarations)
    // println("~~~ "+declarationsFast(Symbol.requiredClass("sun.awt.WeakIdentityHashMap$1"))(c => ()))

    // ???
    //   var N   = 0
    val acc = ArrayBuffer[Symbol]()
    //   try
    //     // println(defn.RootPackage.declarations.flatMap(x => x.declarations))
    // time(declarationsFast(defn.RootPackage)(c => acc += c))

    //   catch {
    //     case _ => ()
    //   }

    println(s"N=${acc.size}")

    // declarationsFast(Symbol.requiredClass("java.util.concurrent.LinkedTransferQueue"))(x => println("~~~"+x.fullName))
    // println(Symbol.requiredClass("java.util.concurrent").declarations.map(x => x -> x.istyp))
    println(Symbol.requiredClass("polyregion.foo.C").declarations)

    declarationsFast(Symbol.requiredPackage("polyregion.foo"))(x => println(x))

    //   C.acc = acc

    //   scala.sys.runtime.gc()

    // println(time(acc.find(_.name == "a?")))
    report.info("hey!")

    println(">> " + x.asTerm.symbol.children)
    // println("~ " + time(declarations(defn.RootPackage).filter(_.isClassDef).size))
    // println("~ " + time(declarations(defn.RootPackage).filter(_.isClassDef).size))
    // println("~ " + time(declarations(defn.RootPackage).filter(_.isClassDef).size))
    println(System.getProperty("java.class.path"))

    println("===")
    println(s"IS=${is
      .filter(x => x.symbol.isDefDef || true)
      .reverse
      .map(x => x -> x.tpe.dealias.widenTermRefByName.simplified)
      .map((x, tpe) => s"-> $x : ${x.show} `${x.symbol.fullName}` : ${tpe.show}\n\t${tpe}\n\t${x.symbol.tree.show}")
      .mkString("\n")}")
    println("===")
    x
  }

  inline def showTpe[B]: Unit = ${ showTpeImpl[B] }
  def showTpeImpl[B: Type](using q: Quotes): Expr[Unit] = {
    import q.reflect.*
    println(">>" + TypeRepr.of[B].typeSymbol.tree.show)
    println(">>" + TypeRepr.of[B].widenTermRefByName.dealias.typeSymbol.tree.show)
    import pprint.*
    pprint.pprintln(TypeRepr.of[B].typeSymbol)
    '{}
  }

  inline def deriveNativeStruct[A]: NativeStruct[A] = ${ deriveNativeStructImpl[A] }
  def deriveNativeStructImpl[A: Type](using q: Quotes): Expr[NativeStruct[A]] = {
    import q.reflect.*
    val Q = Quoted(q)

    val compiler = ct.Compiler.create()

    val repr   = Q.TypeRepr.of[A]
    val opt    = ct.Options.of(ct.Target.LLVM_HOST, "native")
    val layout = Pickler.layoutOf(using Q)(compiler, opt, repr)

    val code = '{
      new NativeStruct[A] {
        override def sizeInBytes: Int = ${ Expr(layout.sizeInBytes.toInt) }
        override def decode(buffer: ByteBuffer, index: Int): A = ${
          Pickler.getStruct(using Q)(compiler, opt, 'buffer, '{ 0 }, 'index, repr).asExprOf[A]
        }
        override def encode(buffer: ByteBuffer, index: Int, a: A): Unit = ${
          Pickler.putStruct(using Q)(compiler, opt, 'buffer, '{ 0 }, 'index, repr, 'a)
        }
      }
    }

    compiler.close()
    given Q.Printer[Q.Tree] = Q.Printer.TreeAnsiCode
    println("Code=" + code.asTerm.show)
    code
  }

  private case class ReifiedConfig(target: ct.Target, arch: String, opt: ct.Opt)

  private def reifyConfig(x: srt.Config[_, _]): List[ReifiedConfig] =
    x.targets.map((t, o) => ReifiedConfig(t.arch, t.uarch, o.value))

  private def reifyConfigFromTpe[C: Type](using
      q: Quotes
  )(c: List[ReifiedConfig] = Nil): Result[List[ReifiedConfig]] = {
    import q.reflect.*
    Type.of[C] match {
      case '[srt.Config[target, opt]] =>
        TypeRepr.of[target].widenTermRefByName.dealias.simplified match {
          case Refinement(target, xs, TypeBounds(l @ ConstantType(StringConstant(arch)), h)) if l == h =>
            val vendorTpe   = target.select(TypeRepr.of[polyregion.scala.Target#Arch].typeSymbol).dealias
            val cpTargetTpe = TypeRepr.of[ct.Target]
            if (vendorTpe <:< cpTargetTpe) {
              (ReifiedConfig(
                target = ct.Target.valueOf(vendorTpe.termSymbol.name),
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

  private inline def checked[A](e: Result[A]): A = e match {
    case Left(e)  => throw e
    case Right(x) => x
  }

  inline def offload0[C](inline queue: rt.Device.Queue, inline cb: Callback[Unit])(inline f: Any): Unit =
    ${ generate0[C]('queue, 'f, 'cb) }
  private def generate0[C: Type](using
      q: Quotes
  )(queue: Expr[rt.Device.Queue], f: Expr[Any], cb: Expr[Callback[Unit]]) = checked(for {
    cs   <- reifyConfigFromTpe[C]()
    expr <- generate(using Quoted(q))(cs, queue, f, '{ rt.Dim3(1, 1, 1) }, cb)
  } yield expr)

  inline def offload1[C](inline queue: rt.Device.Queue, inline rangeX: Range, inline cb: Callback[Unit])(
      inline f: Any
  ): Unit = ${ generate1[C]('queue, 'f, 'rangeX, 'cb) }
  private def generate1[C: Type](using
      q: Quotes
  )(queue: Expr[rt.Device.Queue], f: Expr[Any], rangeX: Expr[Range], cb: Expr[Callback[Unit]]) = checked(for {
    cs   <- reifyConfigFromTpe[C]()
    expr <- generate(using Quoted(q))(cs, queue, f, '{ rt.Dim3($rangeX.size, 1, 1) }, cb)
  } yield expr)

  inline def offload2[C](
      inline queue: rt.Device.Queue,
      inline rangeX: Range,
      inline rangeY: Range,
      inline cb: Callback[Unit]
  )(inline f: Any): Unit = ${ generate2[C]('queue, 'f, 'rangeX, 'rangeY, 'cb) }
  private def generate2[C: Type](using
      q: Quotes
  )(queue: Expr[rt.Device.Queue], f: Expr[Any], rangeX: Expr[Range], rangeY: Expr[Range], cb: Expr[Callback[Unit]]) =
    checked(for {
      cs   <- reifyConfigFromTpe[C]()
      expr <- generate(using Quoted(q))(cs, queue, f, '{ rt.Dim3($rangeX.size, $rangeY.size, 1) }, cb)
    } yield expr)

  inline def offload3[C](
      inline queue: rt.Device.Queue,
      inline rangeX: Range,
      inline rangeY: Range,
      inline rangeZ: Range,
      inline cb: Callback[Unit]
  )(inline f: Any): Unit = ${ generate3[C]('queue, 'f, 'rangeX, 'rangeY, 'rangeZ, 'cb) }
  private def generate3[C: Type](using
      q: Quotes
  )(
      queue: Expr[rt.Device.Queue],
      f: Expr[Any],
      rangeX: Expr[Range],
      rangeY: Expr[Range],
      rangeZ: Expr[Range],
      cb: Expr[Callback[Unit]]
  ) = checked(for {
    cs   <- reifyConfigFromTpe[C]()
    expr <- generate(using Quoted(q))(cs, queue, f, '{ rt.Dim3($rangeX.size, $rangeY.size, $rangeZ.size) }, cb)
  } yield expr)

  private val ProgramCounter = AtomicLong(0)
  private def generate(using q: Quoted)(
      configs: List[ReifiedConfig],
      queue: Expr[rt.Device.Queue],
      f: Expr[Any],
      dim: Expr[rt.Dim3],
      cb: Expr[Callback[Unit]]
  ) = for {
    // configs               <- reifyConfigFromTpe[C](using q.underlying)()
    (captures, prog0, log) <- Compiler.compileExpr(f)
    prog = prog0.copy(entry = prog0.entry.copy(name = p.Sym(s"lambda${ProgramCounter.getAndIncrement()}")))
    _    = println(log.render)

    serialisedAst <- Either.catchNonFatal(MsgPack.encode(MsgPack.Versioned(CppSourceMirror.AdtHash, prog)))
    compiler = ct.Compiler.create()

    compilations <- configs.traverse(c =>
      Either
        .catchNonFatal(compiler.compile(serialisedAst, true, ct.Options.of(c.target, c.arch), c.opt.value))
        .map(c -> _)
    )

  } yield {

    compilations.foreach { (config, c) =>
      println(s"Config=${config}")
      println(s"Program=${c.program.length}")
      println(s"Messages=\n  ${c.messages}")
      println(s"Features=\n  ${c.features.toList}")
      println(s"Elapsed=\n${c.events.sortBy(_.epochMillis).mkString("\n")}")
      println(s"Captures=\n${captures}")
    }

    given Quotes = q.underlying

    val fnName     = Expr(prog.entry.name.repr)
    val moduleName = Expr(s"${prog0.entry.name.repr}@${ProgramCounter.getAndIncrement()}")
    val (captureTpeOrdinals, captureTpeSizes) = captures.map { (name, _) =>
      val tpe = Pickler.tpeAsRuntimeTpe(name.tpe)
      tpe.value -> tpe.sizeInBytes
    }.unzip
    val returnTpeOrdinal = Pickler.tpeAsRuntimeTpe(prog.entry.rtn).value
    val returnTpeSize    = Pickler.tpeAsRuntimeTpe(prog.entry.rtn).sizeInBytes

    val code = '{

      val cb_ : Callback[Unit] = $cb

      // Validate device features support this code object.
      val miss      = ArrayBuffer[(String, Set[String])]()
      val available = Set($queue.device.features(): _*)

      lazy val modules = Array[(String, Set[String], Array[Byte])](${
        Varargs(compilations.map { (config, compilation) =>
          Expr((s"${config.arch}@${config.opt}(${config.target})", Set(compilation.features: _*), compilation.program))

        })
      }: _*)

      var found = -1
      var i     = 0
      while (i < modules.size && found == -1) {
        val (name, required, _) = modules(i)
        val missing             = required -- available
        if (missing.isEmpty) found = i
        else miss += ((name, missing))
        i += 1
      }
      if (found == -1) {
        cb_(
          Left(
            new java.lang.RuntimeException(
              s"Device (${$queue.device.name}) with features `${available.mkString(",")}` does not meet the requirement for any of the following binaries: ${miss
                .map((config, missing) => s"$config (missing ${missing.mkString(",")})")
                .mkString(",")} "
            )
          )
        )
      } else {

        // We got everything, load the code object
        if (! $queue.device.moduleLoaded($moduleName)) {
          $queue.device.loadModule($moduleName, modules(found)._3)
        }

        // Allocate parameter and return type ordinals and value buffers
        val fnTpeOrdinals = ${ Expr((captureTpeOrdinals :+ returnTpeOrdinal).toArray) }
        val fnValues =
          ByteBuffer.allocate(${ Expr(captureTpeSizes.sum + returnTpeSize) }).order(ByteOrder.nativeOrder)

        ${
          q.Match(
            'found.asTerm,
            compilations.zipWithIndex.map { case ((c, _), i) =>
              q.CaseDef(
                q.Literal(q.IntConstant(i)),
                None,
                bindCapturesToBuffer(compiler, ct.Options.of(c.target, c.arch), 'fnValues, queue, captures).asTerm
              )
            }
          ).asExprOf[Unit]
        }

        // ${ bindCapturesToBuffer(compiler, ???, 'fnValues, queue, captures) }

        println("Dispatch tid=" + Thread.currentThread.getId)
        // Dispatch.
        $queue.enqueueInvokeAsync(
          $moduleName,
          $fnName,
          fnTpeOrdinals,
          fnValues.array,
          rt.Policy($dim),
          if ($queue.device.sharedAddressSpace) { () =>
            println("Done s tid=" + Thread.currentThread.getId + " cb" + cb_)
            cb_(Right(()))
          }: Runnable
          else null
        )

        // Sync.
        $queue.syncAll(if (! $queue.device.sharedAddressSpace) (() => cb_(Right(()))): Runnable else null)
      }
    }
    given q.Printer[q.Tree] = q.Printer.TreeAnsiCode
    println("Code=" + code.asTerm.show)
    compiler.close()
    code
  }

  def bindCapturesToBuffer(using q: Quoted)(
      compiler: ct.Compiler,
      opt: ct.Options,
      target: Expr[ByteBuffer],
      queue: Expr[rt.Device.Queue],
      captures: List[(p.Named, q.Term)]
  ) = {
    given Quotes = q.underlying

    val (_, stmts) = captures.zipWithIndex.foldLeft((0, List.empty[Expr[Any]])) {
      case ((byteOffset, exprs), ((name, ref), idx)) =>
        inline def bindArray[ts: Type, t: Type](arr: p.Type.Array, mutable: Boolean, size: Expr[ts] => Expr[Int]) = '{
          val pointer = $queue.registerAndInvalidateIfAbsent[ts](
            /*object*/ ${ ref.asExprOf[ts] },
            /*sizeOf*/ xs => ${ Expr(Pickler.sizeOf(compiler, opt, arr.component, q.TypeRepr.of[t])) } * ${ size('xs) },
            /*write */ (xs, bb) => ${ Pickler.putAll(compiler, opt, 'bb, arr, q.TypeRepr.of[ts], 'xs) },
            /*read  */ ${
              // TODO what about var?
              if (!mutable) '{ (bb, xs) =>
                throw new AssertionError(s"No writeback for immutable type " + ${ Expr(arr.repr) })
              }
              else '{ (bb, xs) => ${ Pickler.getAllMutable(compiler, opt, 'bb, arr, q.TypeRepr.of[ts], 'xs) } }
            },
            null
          )

          $target.putLong(${ Expr(byteOffset) }, pointer)
          ()
        }

        inline def bindStruct[t: Type](struct: p.Type.Struct) = '{

          val pointer = $queue.registerAndInvalidateIfAbsent[t](
            /*object*/ ${ ref.asExprOf[t] },
            /*sizeOf*/ x => ${ Expr(Pickler.sizeOf(compiler, opt, struct, q.TypeRepr.of[t])) },
            /*write */ (x, bb) => ${ Pickler.putAll(compiler, opt, 'bb, struct, q.TypeRepr.of[t], 'x) },
            /*read  */ null, // (bb, x) => throw new AssertionError(s"No write-back for struct type " + ${ Expr(struct.repr) }),
            null
          )

          $target.putLong(${ Expr(byteOffset) }, pointer)
          ()
        }

        val expr = (name.tpe, ref.tpe.asType) match {
          case (p.Type.Array(comp), x @ '[srt.Buffer[_]]) =>
            '{
              $target.putLong(
                ${ Expr(byteOffset) },
                $queue.registerAndInvalidateIfAbsent(
                  ${ ref.asExprOf[x.Underlying] },
                  ${ ref.asExprOf[x.Underlying] }.backing,
                  null
                )
              )
            }
          case (arr @ p.Type.Array(_), ts @ '[Array[t]]) =>
            bindArray[ts.Underlying, t](arr, mutable = true, x => '{ $x.length })
          case (arr @ p.Type.Array(_), ts @ '[scala.collection.mutable.Seq[t]]) =>
            bindArray[ts.Underlying, t](arr, mutable = true, x => '{ $x.length })
          case (arr @ p.Type.Array(_), ts @ '[java.util.List[t]]) =>
            bindArray[ts.Underlying, t](arr, mutable = true, x => '{ $x.size })
          case (arr @ p.Type.Array(_), ts @ '[scala.collection.immutable.Seq[t]]) =>
            bindArray[ts.Underlying, t](arr, mutable = false, x => '{ $x.length })
          case (s @ p.Type.Struct(_, _, _), '[t]) =>
            // bindArray[ts.Underlying, t](arr, mutable = false, x => '{ $x.length })
            // Pickler.writeStruct(compiler, opt, target, Expr(byteOffset), Expr(0), ref.tpe, ref.asExpr)
            bindStruct[t](s)
          case (t, _) =>
            // Pickler.putAll(compiler, opt, target, t, ref.tpe, ref.asExpr)
            Pickler.putPrimitive(target, Expr(byteOffset), t, ref.asExpr)
        }
        (byteOffset + Pickler.tpeAsRuntimeTpe(name.tpe).sizeInBytes, exprs :+ expr)
    }
    Expr.block(stmts, '{ () })
  }

}
