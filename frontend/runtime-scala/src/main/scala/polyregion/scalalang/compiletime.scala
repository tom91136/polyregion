package polyregion.scalalang

import cats.syntax.all.*
import polyregion.ast.{ScalaSRR as p, *}
import polyregion.jvm.{compiler as ct, runtime as rt}
import polyregion.prism.StdLib
import polyregion.scalalang as srt

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
            val vendorTpe   = target.select(TypeRepr.of[srt.Target#Arch].typeSymbol).dealias
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
    // ${ generate0[C]('queue, 'f, 'cb) }
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
  ) = {
    val log = Log("")
    val result = for {

      // Name actual => type actual
      // configs               <- reifyConfigFromTpe[C](using q.underlying)()

      (captures, prismRefs, monoMap, prog0) <- Compiler.compileExpr(log, f)

      resolveStructDef = (s: p.Type.Struct) => prog0.defs.find(_.name == s.name)

      // Match up the struct types from the capture args so that we know how to copy the structs during argument pickling.
      capturesWithStructDefs <- captures.traverse {
        case (n @ p.Named(_, p.Type.Struct(name, _, _, _)), term) =>
          prog0.defs
            .find(_.name == name)
            .failIfEmpty(s"Missing structure def from capture args ${n.repr}")
            .map(d => (n, Some(d), term))
        case (n @ p.Named(_, tpe), term) => (n, None, term).success
      }

      prog: p.Program = prog0.copy(entry =
        prog0.entry.copy(
          name = p.Sym(s"lambda${ProgramCounter.getAndIncrement()}")
        )
      )

      _ = println(log.render(1).mkString("\n"))
      _ = println(prog.entry.repr)
      _ = println(prog.functions.map(_.repr).mkString("\n"))

      serialisedAst <- Either.catchNonFatal(MsgPack.encode(CodeGen.polyASTVersioned(prog)))
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

      val tidTpeOrdinal = Pickler.tpeAsRuntimeTpe(p.Type.Long).value
      val tidTpeSize    = Pickler.tpeAsRuntimeTpe(p.Type.Long).sizeInBytes

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
            Expr(
              (s"${config.arch}@${config.opt}(${config.target})", Set(compilation.features: _*), compilation.program)
            )

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
          val fnTpeOrdinals = ${ Expr((tidTpeOrdinal +: captureTpeOrdinals :+ returnTpeOrdinal).toArray) }
          var fnValues =
            ByteBuffer
              .allocate(${ Expr(tidTpeSize + captureTpeSizes.sum + returnTpeSize) })
              .order(ByteOrder.nativeOrder)

          val ptrMap = scala.collection.mutable.Map[Any, Long]()

          ${

            q.Match(
              'found.asTerm,
              compilations.zipWithIndex.map { case ((config, compilation), i) =>
                val layouts0 = compilation.layouts.map(l => p.Sym(l.name) -> l).toMap

                val lut     = prog.defs.map(s => s.name -> s).toMap
                val layouts = prog.defs.map(sd => sd -> (layouts0(sd.name), prismRefs.get(sd.name))).toMap
                val allReprsInCaptures = capturesWithStructDefs
                  .collect { case (_, Some(sdef), term) =>
                    Pickler.deriveAllRepr(lut.map((s, sd) => s -> (sd, prismRefs.get(s))), sdef, term.tpe)
                  }
                  .reduceLeftOption(_ ++ _)
                  .getOrElse(Map.empty)

                val (defs, write, read, update) = Pickler.generateAll(
                  lut,
                  layouts,
                  allReprsInCaptures,
                  '{ Platforms.pointerOfDirectBuffer(_) },
                  '{ Platforms.directBufferFromPointer(_, _) }
                )

                q.CaseDef(
                  q.Literal(q.IntConstant(i)),
                  None,
                  q.Block(
                    defs,
                    '{
                      ${
                        bindWrite(
                          tidTpeSize,
                          write(_, _, 'ptrMap),
                          'fnValues,
                          queue,
                          capturesWithStructDefs
                        )
                      }

                      println("Dispatch tid=" + Thread.currentThread.getId)
                      println(s"fnTpeOrdinals=${fnTpeOrdinals.toList}")
                      println(s"fnValues.array=0x${fnValues.array.map(byte => f"$byte%02x").mkString(" ")}")
                      // Dispatch.
                      $queue.enqueueInvokeAsync(
                        $moduleName,
                        $fnName,
                        fnTpeOrdinals,
                        fnValues.array,
                        rt.Policy($dim),
                        { () =>
                          println("Kernel completed, tid=" + Thread.currentThread.getId + " cb=" + cb_)

                          val objMap = scala.collection.mutable.Map[Long, Any]()

                          try {
                            ${
                              // Varargs(capturesWithStructDefs.map { case (named, maybeSdef, term) =>
                              bindRead(
                                tidTpeSize,
                                read(_, _, _, 'ptrMap, 'objMap),
                                update(_, _, _, 'ptrMap, 'objMap),
                                'fnValues,
                                queue,
                                capturesWithStructDefs
                              )
                              // })
                            }
                            println("Restore complete")
                            cb_(Right(()))
                          } catch {
                            case e: Throwable =>
                              e.printStackTrace()
                              cb_(Left(new Exception("Cannot sync", e)))
                          }

                          ()

                          // // $queue.syncAll(() => cb_(Right(())))
                          // try $queue.syncAll(() => cb_(Right(())))
                          // catch {
                          //   case e: Throwable =>
                          //     e.printStackTrace()
                          //   // cb_(Left(new Exception("Cannot sync",e)))
                          // }
                        }
                      )

                    }.asTerm
                  )
                )
              } :+ q.CaseDef(q.Wildcard(), None, '{ ??? }.asTerm)
            ).asExprOf[Unit]
          }

        }
      }
      given q.Printer[q.Tree] = q.Printer.TreeAnsiCode
      println("Code=" + code.asTerm.show)
      compiler.close()
      code
    }
    result
  }

  type PtrMapTpe = scala.collection.mutable.Map[Any, Long]
  type ObjMapTpe = scala.collection.mutable.Map[Long, Any]

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

  def bindWrite(using q: Quoted)(
      offset: Int,
      write: (p.Sym, Expr[Any]) => Expr[Long],
      target: Expr[ByteBuffer],
      queue: Expr[rt.Device.Queue],
      capturesWithStructDefs: List[(p.Named, Option[p.StructDef], q.Term)]
  ) = {
    given Quotes = q.underlying
    val (_, stmts) = capturesWithStructDefs.zipWithIndex.foldLeft((offset, List.empty[Expr[Any]])) {
      case ((byteOffset, exprs), ((name, structDef, ref), idx)) =>
        println(
          s"[Bind] [$idx, offset=${byteOffset}]  repr=${ref.show} name=${name.repr} (${structDef.map(_.repr)})"
        )

        val mutable = ref.symbol.flags.is(q.Flags.Mutable)

        val expr = (name.tpe, structDef, ref.asExpr) match {
          case (p.Type.Ptr(comp, _, _), None, _) =>
            throw new RuntimeException(
              s"Top level arrays at parameter boundary is illegal: repr=${ref.show} name=${name.repr}"
            )
          case (s @ p.Type.Struct(_, _, _, _), None, _) =>
            throw new RuntimeException(
              s"Struct type without definition at parameter boundary is illegal: repr=${ref.show} name=${name.repr}"
            )
          case (s @ p.Type.Struct(_, _, _, _), Some(sdef), '{ $ref: t }) =>
            println(s"$ref = " + ref.asTerm.symbol.flags.show)

            val rtn = write(sdef.name, ref)

            Pickler.writePrim(target, Expr(byteOffset), p.Type.Long, rtn)

          case (t, None, '{ $ref: t }) => Pickler.writePrim(target, Expr(byteOffset), t, ref)
          case (t, _, _) =>
            throw new RuntimeException(
              s"Unexpected type ${t.repr} at parameter boundary: repr=${ref.show} name=${name.repr}"
            )

        }
        (byteOffset + Pickler.tpeAsRuntimeTpe(name.tpe).sizeInBytes, exprs :+ expr)
    }
    Expr.block(stmts, '{ () })
  }

  def bindRead(using q: Quoted)(
      offset: Int,
      read: (p.Sym, Expr[Any], Expr[Long]) => Expr[Any],
      update: (p.Sym, Expr[Any], Expr[Long]) => Expr[Unit],
      target: Expr[ByteBuffer],
      queue: Expr[rt.Device.Queue],
      capturesWithStructDefs: List[(p.Named, Option[p.StructDef], q.Term)]
  ) = {
    given Quotes = q.underlying
    val (_, stmts) = capturesWithStructDefs.zipWithIndex.foldLeft((offset, List.empty[Expr[Any]])) {
      case ((byteOffset, exprs), ((name, structDef, ref), idx)) =>
        println(
          s"[Bind] <-  [$idx, offset=${byteOffset}]  repr=${ref.show} name=${name.repr} (${structDef.map(_.repr)})"
        )

        val mutable = ref.symbol.flags.is(q.Flags.Mutable)

        val expr = (name.tpe, structDef, ref.asExpr) match {
          case (p.Type.Ptr(comp, _, _), None, _) =>
            throw new RuntimeException(
              s"Top level arrays at parameter boundary is illegal: repr=${ref.show} name=${name.repr}"
            )
          case (s @ p.Type.Struct(_, _, _, _), None, _) =>
            throw new RuntimeException(
              s"Struct type without definition at parameter boundary is illegal: repr=${ref.show} name=${name.repr}"
            )
          case (s @ p.Type.Struct(_, _, _, _), Some(sdef), '{ $ref: t }) =>
            println(s"$ref = " + ref.asTerm.symbol.flags.show)
            val ptr = Pickler.readPrim(target, Expr(byteOffset), p.Type.Long).asExprOf[Long]
            '{

              println(s"Restore ${$ref}")
              ${
                if (mutable) {
                  q.Assign(ref.asTerm, read(sdef.name, ref, ptr).asTerm).asExprOf[Unit]
                } else {
                  update(sdef.name, ref, ptr)
                }
              }
            }

          case (t, None, '{ $ref: t }) =>
            '{
              println(s"[Bind] skipping primitive ${$ref}")
            }
          // Pickler.writePrim(target, Expr( byteOffset), t, ref)
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
