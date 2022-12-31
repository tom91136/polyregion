package polyregion.scala

import cats.syntax.all.*
import polyregion.ast.{CppSourceMirror, MsgPack, PolyAst as p, *}
import polyregion.jvm.{compiler as ct, runtime as rt}
import polyregion.prism.StdLib
import polyregion.scala as srt

import java.nio.file.Paths
import java.nio.{ByteBuffer, ByteOrder}
import java.util.concurrent.atomic.{AtomicLong, AtomicReference}
import scala.annotation.{compileTimeOnly, tailrec}
import scala.collection.immutable.VectorMap
import scala.collection.mutable.ArrayBuffer
import scala.quoted.*
import scala.util.Try

@compileTimeOnly("This class only exists at compile-time to expose offload methods")
object compiletime {

  inline def showExpr(inline x: Any): Any = ${ showExprImpl('x) }
  def showExprImpl(x: Expr[Any])(using q: Quotes): Expr[Any] = {
    import q.reflect.*
    given Printer[Tree] = Printer.TreeAnsiCode
    println(x.show)
//    pprint.pprintln(x.asTerm) // term => AST

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
  def showTpeImpl[B: Type](using q: Quotes): Expr[Unit] = {
    import q.reflect.*
    println(">>" + TypeRepr.of[B].typeSymbol.tree.show)
    println(">>" + TypeRepr.of[B].widenTermRefByName.dealias.typeSymbol.tree.show)
//    import pprint.*
//    pprint.pprintln(TypeRepr.of[B].typeSymbol)
    '{}
  }

//  inline def deriveNativeStruct[A]: NativeStruct[A] = ${ deriveNativeStructImpl[A] }
//  def deriveNativeStructImpl[A: Type](using q: Quotes): Expr[NativeStruct[A]] = {
//    import q.reflect.*
//    val Q = Quoted(q)
//
//    val compiler = ct.Compiler.create()
//
//    val repr   = Q.TypeRepr.of[A]
//    val opt    = ct.Options.of(ct.Target.LLVM_HOST, "native")
//    val layout = Pickler.layoutOf(using Q)(compiler, opt, repr)
//
//    val code = '{
//      new NativeStruct[A] {
//        override def sizeInBytes: Int = ${ Expr(layout.sizeInBytes.toInt) }
//        override def decode(buffer: ByteBuffer, index: Int): A = ${
//          ???
//          // Pickler.getStruct(using Q)(compiler, opt, 'buffer, '{ 0 }, 'index, repr).asExprOf[A]
//        }
//        override def encode(buffer: ByteBuffer, index: Int, a: A): Unit = ${
//          // Pickler.putStruct(using Q)(compiler, opt, 'buffer, '{ 0 }, 'index, ???, repr, 'a)
//          ???
//        }
//      }
//    }
//
//    compiler.close()
//    given Q.Printer[Q.Tree] = Q.Printer.TreeAnsiCode
//    println("Code=" + code.asTerm.show)
//    code
//  }

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

    // Name actual => type actual
    // configs               <- reifyConfigFromTpe[C](using q.underlying)()
    (captures, prismRefs, prog0, log) <- Compiler.compileExpr(f)

    resolveStructDef = (s: p.Type.Struct) => prog0.defs.find(_.name == s.name)

    // Match up the struct types from the capture args so that we know how to copy the structs during argument pickling.
    capturesWithStructDefs <- captures.traverse {
      case (n @ p.Named(_, p.Type.Struct(name, _, _)), term) =>
        prog0.defs
          .find(_.name == name)
          .failIfEmpty(s"Missing structure def from capture args ${n.repr}")
          .map(d => (n, Some(d), term))
      case (n @ p.Named(_, tpe), term) => (n, None, term).success
    }

    prog: p.Program = prog0.copy(entry = prog0.entry.copy(name = p.Sym(s"lambda${ProgramCounter.getAndIncrement()}")))
    _               = println(log.render)

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
      println(s"Captures=\n${capturesWithStructDefs}")
    }

    given Quotes = q.underlying

    val fnName     = Expr(prog.entry.name.repr)
    val moduleName = Expr(s"${prog0.entry.name.repr}@${ProgramCounter.getAndIncrement()}")
    val (captureTpeOrdinals, captureTpeSizes) = capturesWithStructDefs.map { (name, _, _) =>
      val tpe = Pickler.tpeAsRuntimeTpe(name.tpe)
      tpe.value -> tpe.sizeInBytes
    }.unzip
    val returnTpeOrdinal = Pickler.tpeAsRuntimeTpe(prog.entry.rtn).value
    val returnTpeSize    = Pickler.tpeAsRuntimeTpe(prog.entry.rtn).sizeInBytes

    println(s"Prog defs: ${prog.defs}")

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
            compilations.zipWithIndex.map { case ((config, compilation), i) =>
              val layouts = compilation.layouts.map(l => p.Sym(l.name.toList) -> l).toMap

              q.CaseDef(
                q.Literal(q.IntConstant(i)),
                None,
                bindCapturesToBuffer(
                  prog.defs.map(sd => sd -> layouts(sd.name)).toMap,
                  prog.defs.map(s => s.name -> s).toMap,
                  'fnValues,
                  queue,
                  capturesWithStructDefs
                ).asTerm
              )
            }
          ).asExprOf[Unit]
        }

        println("Dispatch tid=" + Thread.currentThread.getId)
        println(s"fnTpeOrdinals=${fnTpeOrdinals.toList}")
        println(s"fnValues.array=${fnValues.array.toList}")
        // Dispatch.
        $queue.enqueueInvokeAsync(
          $moduleName,
          $fnName,
          fnTpeOrdinals,
          fnValues.array,
          rt.Policy($dim),
          { () =>
            println("Kernel completed, tid=" + Thread.currentThread.getId + " cb=" + cb_)

            // $queue.syncAll(() => cb_(Right(())))
            try $queue.syncAll(() => cb_(Right(())))
            catch {
              case e: Throwable =>
                e.printStackTrace()
              // cb_(Left(new Exception("Cannot sync",e)))
            }
          }
        )

      }
    }
    given q.Printer[q.Tree] = q.Printer.TreeAnsiCode
    println("Code=" + code.asTerm.show)
    compiler.close()
    code
  }

  def bindStruct[t: Type](using
      q: Quoted
  )(
      layouts: Map[p.StructDef, ct.Layout],
      lut: Map[p.Sym, p.StructDef],
      queue: Expr[rt.Device.Queue],
      ref: Expr[t],
      struct: p.StructDef
  ): Expr[Long] = {
    given Quotes = q.underlying
    '{
      println(s"bind struct ${${ Expr(struct.name.repr) }}, object ${$ref}")

      if ($ref == null) 0L
      else
        $queue.registerAndInvalidateIfAbsent[t](
          /* object */ ${ ref.asExprOf[t] },
          /* sizeOf */ x => ${ Expr(layouts(struct).sizeInBytes.toInt) },
          /* write  */ (x, bb) => {

            println(
              s"bind struct ${${ Expr(struct.name.repr) }}, write  $x => $bb(0x${Platforms.pointerOfDirectBuffer(bb).toHexString})"
            )
            if (x != null) {
              ${
                Pickler.putStruct(
                  layouts,
                  'bb,
                  '{ 0 },
                  struct,
                  q.TypeRepr.of[t],
                  'x,
                  (s, expr) =>
                    expr match {
                      case '{ $es: t } =>
                        println(s"Do ${s}")
                        bindStruct(layouts, lut, queue, es, lut(s.name))
                    },
                  (s, expr) =>
                    expr match {
                      case '{ $xs: StdLib.MutableSeq[v] } =>
                        '{
                          println(
                            s"bind struct ${${ Expr(struct.name.repr) }}, array type ${${ Expr(s.repr) }} => ${$expr}"
                          )
                          ${ bindArray[v](layouts, lut, queue, xs, s) }
                        }
                    }
                )
              }
            }
          },
          /* read   */ (bb, x) => {
            println(
              s"bind struct ${${ Expr(struct.name.repr) }}, read (no restore)  $bb(0x${Platforms.pointerOfDirectBuffer(bb).toHexString}) => $x"
            )
            ()
          }, // throw new AssertionError(s"No write-back for struct type " + ${ Expr(struct.repr) }),
          /* cb     */ null
        )
    }
  }

  def bindArray[t: Type](using
      q: Quoted
  )(
      layouts: Map[p.StructDef, ct.Layout],
      lut: Map[p.Sym, p.StructDef],
      queue: Expr[rt.Device.Queue],
      ref: Expr[StdLib.MutableSeq[t]],
      component: p.Type
  ): Expr[Long] = {
    given Quotes = q.underlying

    def storeElem(
        dest: Expr[ByteBuffer],
        src: Expr[StdLib.MutableSeq[t]],
        index: Expr[Int]
    ) = {
      val byteOffset = '{ $index * ${ Expr(Pickler.tpeAsRuntimeTpe(component).sizeInBytes) } }
      component match {
        case p.Type.Array(_) => ??? // Nested array should not be a thing given the current scheme!
        case s @ p.Type.Struct(_, _, _) =>
          val ptr = bindStruct(layouts, lut, queue, '{ $src($index) }, lut(s.name))
          Pickler.putPrimitive(dest, byteOffset, p.Type.Long, ptr)
        case _ =>
          Pickler.putPrimitive(dest, byteOffset, component, '{ $src($index) })
      }
    }

    def restoreElem(
        src: Expr[ByteBuffer],
        dest: Expr[StdLib.MutableSeq[t]],
        index: Expr[Int]
    ) = {
      val byteOffset = '{ $index * ${ Expr(Pickler.tpeAsRuntimeTpe(component).sizeInBytes) } }
      component match {
        case p.Type.Array(_)            => ??? // Nested array should not be a thing given the current scheme!
        case s @ p.Type.Struct(_, _, _) =>
          // val ptr = bindStruct(compiler, opt, lut, queue, '{ $expr(index) }.asTerm, lut(s.name))
          // Pickler.putPrimitive(target, i, p.Type.Long, ptr)

          // restore from long ptr stored here
          val ptr  = Pickler.getPrimitive(src, byteOffset, p.Type.Long).asExprOf[Long]
          val size = Pickler.layoutOf(layouts, q.TypeRepr.of[t]).sizeInBytes.toInt

          // O != null;  O.register... => no-op array restore
          // O == null:  (don't register, no-op) => ptr => restore [t]
          //             Pickler.mkStruct(compiler, opt, new ByteBuffer(ptr, SizeOfStruct), q.TypeRepr.of[t])

          // ...
          // free all malloc allocations after restore

          '{

            val buffer = Platforms.directBufferFromPointer($ptr, ${ Expr(size) })
            println(s"  Mk buffer = ${buffer}, ptr=0x${$ptr.toHexString}")

            val restored =
              ${

                def restoreOne(buffer: Expr[ByteBuffer], repr: q.TypeRepr): Expr[Any] = Pickler.mkStruct(
                  layouts,
                  buffer,
                  repr,
                  (tpeRepr, buffer, offset) =>
                    restoreOne(
                      '{
                        Platforms.directBufferFromPointer(
                          ${ Pickler.getPrimitive(buffer, offset, p.Type.Long).asExprOf[Long] },
                          ${ Expr(Pickler.layoutOf(layouts, tpeRepr).sizeInBytes.toInt) }
                        )
                      },
                      tpeRepr
                    )
                )

                restoreOne('buffer, q.TypeRepr.of[t]).asExprOf[t]
              }
            println(
              s"bind array: restore [${$index}] (skipping struct type ${${ Expr(s.repr) }} ptr is =${$ptr}, ${buffer} = $restored)"
            )
            $dest($index) = restored

          }
        case _ =>
          '{
            $dest($index) = ${ Pickler.getPrimitive(src, byteOffset, component).asExprOf[t] }

          }

      }
    }

    '{
      val xs = $ref
      println(s"bind array: object ${xs} ")
      $queue.registerAndInvalidateIfAbsent[StdLib.MutableSeq[t]](
        /* object */ xs,
        /* sizeOf */ x => ${ Expr(Pickler.tpeAsRuntimeTpe(component).sizeInBytes) } * x.length_,
        /* write  */ { (xs, bb) =>
          println(s"bind array: write  ${xs} => $bb(${Platforms.pointerOfDirectBuffer(bb)})")
          var i = 0
          while (i < xs.length_) {
            ${ storeElem('bb, 'xs, 'i) }
            i += 1
          }
        },
        /* read   */ (bb, xs) => {
          println(s"bind array: read  ${bb}(${Platforms.pointerOfDirectBuffer(bb)}) => $xs")
          var i = 0
          while (i < xs.length_) {
            println(s"do restore ${i}")
            ${ restoreElem('bb, 'xs, 'i) }
            i += 1
          }
        },
        /* cb     */ null
      )
    }
  }

  def bindRead(using q: Quoted)(
      layouts: Map[p.StructDef, (ct.Layout, Option[polyregion.prism.TermPrism[Any, Any]])],
      lut: Map[p.Sym, p.StructDef],
      root: q.Term,
      sdef: p.StructDef,
      input: Expr[Long],
  ): Expr[Unit] = (for {
    mapping         <- Pickler.mkStructMapping(sdef, layouts.view.mapValues(_._1))
    (_, maybePrism) <- layouts.get(sdef).failIfEmpty(s"Missing def ${sdef.repr}")
    rootThroughPrism = maybePrism match {
      case None          => ???
      case Some((_, to)) => to(q.underlying, root.asExpr, _).asTerm
    }
    given Quotes = q.underlying
  } yield '{

    val root   = ${ rootThroughPrism.asExpr }
    val buffer = ByteBuffer.allocate(mapping.sizeInBytes.toInt)
    val ptr    = Platforms.pointerOfDirectBuffer(buffer)
    //

  }).fold(e => throw new CompilerException(s"Cannot bind read for ${root.show} (sdef=${sdef.repr})", e), identity)

  def bindWrite(using q: Quoted)(
      layouts: Map[p.StructDef, (ct.Layout, Option[polyregion.prism.TermPrism[Any, Any]])],
      lut: Map[p.Sym, p.StructDef],
      root: q.Term,
      sdef: p.StructDef
  ): Expr[Long] = (for {
    mapping         <- Pickler.mkStructMapping(sdef, layouts.view.mapValues(_._1))
    (_, maybePrism) <- layouts.get(sdef).failIfEmpty(s"Missing def ${sdef.repr}")
    rootThroughPrism = maybePrism match {
      case None            => root
      case Some((from, _)) => from(q.underlying, root.asExpr).asTerm
    }
    given Quotes = q.underlying
  } yield '{
    val root   = ${ rootThroughPrism.asExpr }
    val buffer = ByteBuffer.allocate(mapping.sizeInBytes.toInt)
    val ptr    = Platforms.pointerOfDirectBuffer(buffer)
    println(s"[bind]: object  ${root} => $buffer(0x${ptr.toHexString})")
    ${
      Varargs(mapping.members.map { m =>
        val offset = Expr(m.offsetInBytes.toInt)
        ('root, m.tpe) match {
          case ('{ $expr: StdLib.MutableSeq[t] }, p.Type.Array(component)) =>
            '{
              val elementSizeInBytes = ${ Expr(Pickler.tpeAsRuntimeTpe(component).sizeInBytes()) }
              val arrayBuffer        = ByteBuffer.allocate(elementSizeInBytes * $expr.length_)
              val arrayPtr           = Platforms.pointerOfDirectBuffer(buffer)
              println(
                s"[bind]: array  [${$expr.length_} * ${${ Expr(component.repr) }} ]${$expr.data} => $arrayBuffer(0x${arrayPtr.toHexString})"
              )
              var i = 0
              while (i < $expr.length_) {
                ${
                  component match {
                    case p.Type.Struct(name, _, _) =>
                      '{
                        val ptr = ${ bindWrite(layouts, lut, '{ $expr(i) }.asTerm, lut(name)) }
                        ${ Pickler.putPrimitive('buffer, offset, p.Type.Long, 'ptr) }
                      }
                    case t =>
                      Pickler.putPrimitive('arrayBuffer, '{ elementSizeInBytes * i }, t, '{ $expr(i) })
                  }
                }
                i += 1
              }
              ${ Pickler.putPrimitive('buffer, offset, p.Type.Long, 'arrayPtr) }
            }
          case (_, p.Type.Struct(name, _, _)) =>
            '{
              val ptr = if (root == null) 0L else ${ bindWrite(layouts, lut, m.select('root.asTerm), lut(name)) }
              ${ Pickler.putPrimitive('buffer, offset, p.Type.Long, 'ptr) }
            }
          case (_, _) => Pickler.putPrimitive('buffer, offset, m.tpe, m.select('root.asTerm).asExpr)
        }
      })
    }
    ptr
  }).fold(e => throw new CompilerException(s"Cannot bind write for ${root.show} (sdef=${sdef.repr})", e), identity)

  def insertS(using q: Quoted)(
      layouts: Map[p.StructDef, (ct.Layout, Option[polyregion.prism.TermPrism[Any, Any]])],
      lut: Map[p.Sym, p.StructDef],
      queue: Option[Expr[rt.Device.Queue]],
      root: q.Term,
      sdef: p.StructDef
  ): Expr[Long] = {

    // Basic steps:
    // 1. Creating struct mapping
    // 2. Find the associated prism if we have one and apply the root for `from` and `to`
    // 2. If we see any nested struct types (including array),
    // We treat arrays differently
    // Rules:
    //  To:
    //  - (val|var) ${inst.a} == null => 0
    //  - (val|var) ${inst.a} != null => toPtr(   f(q, ${inst.a}: A ): A  )
    //  From:
    //  - (val) // no-op
    //  - (var) $ptr == nullptr => ${inst.a} := null
    //  - (var) $ptr != nullptr => ${inst.a} := f(q, ${inst.a} ?, fromPtr($ptr)) : A

    given Quotes = q.underlying

    for {
      mapping         <- Pickler.mkStructMapping(sdef, layouts.view.mapValues(_._1))
      (_, maybePrism) <- layouts.get(sdef).failIfEmpty(s"Missing def ${sdef.repr}")
      rootThroughPrism = maybePrism match {
        case None             => root
        case Some((from, to)) => from(q.underlying, root.asExpr).asTerm
      }
    } yield rootThroughPrism.asExpr match {

      case '{ $expr: StdLib.MutableSeq[t] } =>

      case '{ $expr: t } =>
        '{
          val captureRoot = $expr
          println(s"[bind]: object ${captureRoot} ")

          $queue.registerAndInvalidateIfAbsent[t](
            /* object */ captureRoot,
            /* sizeOf */ _ => ${ Expr(mapping.sizeInBytes.toInt) },
            /* write  */ { (root, bb) =>
              println(s"[bind]: write  ${root} => $bb(${Platforms.pointerOfDirectBuffer(bb)})")
              mapping.members.map { m =>
                m.tpe match {
                  case s: p.Type.Struct =>
                    val sdef = lut(s.name)

                    m.select('root.asTerm).asExpr

                    Pickler.putPrimitive('bb, Expr(m.offsetInBytes.toInt), p.Type.Long, ???)

                    ???
                  case _ =>
                    ${
                      Pickler.putPrimitive('bb, Expr(m.offsetInBytes.toInt), m.tpe, m.select('root.asTerm).asExpr)
                    }
                }
              }

            },
            /* read   */ (bb, t) =>
              //
              println(s"[bind]: read  ${bb}(${Platforms.pointerOfDirectBuffer(bb)}) => $t"),
            /* cb     */ null
          )
        }
    }

    ???
  }

  def bindCapturesToBuffer(using q: Quoted)(
      layouts: Map[p.StructDef, ct.Layout],
      lut: Map[p.Sym, p.StructDef],
      target: Expr[ByteBuffer],
      queue: Expr[rt.Device.Queue],
      capturesWithStructDefs: List[(p.Named, Option[p.StructDef], q.Term)]
  ) = {
    given Quotes = q.underlying
    val (_, stmts) = capturesWithStructDefs.zipWithIndex.foldLeft((0, List.empty[Expr[Any]])) {
      case ((byteOffset, exprs), ((name, structDef, ref), idx)) =>
        println(s"[Bind] [$idx, offset=${byteOffset}]  repr=${ref.show} name=${name.repr} (${structDef.map(_.repr)})")
        val expr = (name.tpe, structDef, ref.asExpr) match {
          case (p.Type.Array(comp), None, _) =>
            throw new RuntimeException(
              s"Top level arrays at parameter boundary is illegal: repr=${ref.show} name=${name.repr}"
            )
          case (s @ p.Type.Struct(_, _, _), None, _) =>
            throw new RuntimeException(
              s"Struct type without definition at parameter boundary is illegal: repr=${ref.show} name=${name.repr}"
            )
          case (s @ p.Type.Struct(_, _, _), Some(sdef), '{ $ref: t }) =>
            '{

              println(s">>> ${${ Expr(s.repr) }}")

              ${
                Pickler.putPrimitive(
                  target,
                  Expr(byteOffset),
                  p.Type.Long,
                  bindStruct[t](layouts, lut, queue, ref, sdef)
                )
              }
            }
          case (t, None, '{ $ref: t }) => Pickler.putPrimitive(target, Expr(byteOffset), t, ref)
          case (t, _, _) =>
            throw new RuntimeException(
              s"Unexpected type ${t.repr} at parameter boundary: repr=${ref.show} name=${name.repr}"
            )

        }
        (byteOffset + Pickler.tpeAsRuntimeTpe(name.tpe).sizeInBytes, exprs :+ expr)
    }
    Expr.block(stmts, '{ () })
  }

}
