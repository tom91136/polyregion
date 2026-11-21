package polyregion.scalalang

import cats.syntax.all.*
import polyregion.ast.{PolyAST as p, *}
import polyregion.jvm.twin.Marshal
import polyregion.jvm.{compiler as ct, runtime as rt}
import polyregion.twin.StructLayouts
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
@SuppressWarnings(Array("bonsai:data-clump"))
object compiletime {
  import Env.debug

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

  private case class ReifiedConfig(name: String, target: ct.Target, arch: String, opt: ct.Opt)

  private def reifiedConfigs(compiler: ct.Compiler, config: LaunchConfig): Result[List[ReifiedConfig]] =
    config.targets.traverse { (name, arch) =>
      Either
        .catchNonFatal(compiler.targetByName(name))
        .map(target => ReifiedConfig(name, target, arch, config.opt))
    }

  private inline def checked(e: Result[Expr[Unit]]): Expr[Unit] = e match {
    case Left(e)  => throw e
    case Right(x) => x
  }

  inline def offload0(inline ctx: rt.Context, inline cb: Callback[Unit])(
      inline f: Any
  ): Unit = ${ generate0('ctx, 'f, 'cb) }
  private def generate0(using
      q: Quotes
  )(ctx: Expr[rt.Context], f: Expr[Any], cb: Expr[Callback[Unit]]) =
    checked(generate(using Quoted(q))(ctx, f, '{ rt.Dim3(1, 1, 1) }, cb))

  inline def offload1(
      inline ctx: rt.Context,
      inline rangeX: Range,
      inline cb: Callback[Unit]
  )(
      inline f: Any
  ): Unit = ${ generate1('ctx, 'f, 'rangeX, 'cb) }
  private def generate1(using
      q: Quotes
  )(ctx: Expr[rt.Context], f: Expr[Any], rangeX: Expr[Range], cb: Expr[Callback[Unit]]) =
    checked(generate(using Quoted(q))(ctx, f, '{ rt.Dim3($rangeX.size, 1, 1) }, cb))

  inline def offload2(
      inline ctx: rt.Context,
      inline rangeX: Range,
      inline rangeY: Range,
      inline cb: Callback[Unit]
  )(inline f: Any): Unit = ${ generate2('ctx, 'f, 'rangeX, 'rangeY, 'cb) }
  private def generate2(using
      q: Quotes
  )(
      ctx: Expr[rt.Context],
      f: Expr[Any],
      rangeX: Expr[Range],
      rangeY: Expr[Range],
      cb: Expr[Callback[Unit]]
  ) = checked(
    generate(using Quoted(q))(ctx, f, '{ rt.Dim3($rangeX.size, $rangeY.size, 1) }, cb)
  )

  inline def offload3(
      inline ctx: rt.Context,
      inline rangeX: Range,
      inline rangeY: Range,
      inline rangeZ: Range,
      inline cb: Callback[Unit]
  )(inline f: Any): Unit = ${ generate3('ctx, 'f, 'rangeX, 'rangeY, 'rangeZ, 'cb) }
  private def generate3(using
      q: Quotes
  )(
      ctx: Expr[rt.Context],
      f: Expr[Any],
      rangeX: Expr[Range],
      rangeY: Expr[Range],
      rangeZ: Expr[Range],
      cb: Expr[Callback[Unit]]
  ) = checked(
    generate(using Quoted(q))(
      ctx,
      f,
      '{ rt.Dim3($rangeX.size, $rangeY.size, $rangeZ.size) },
      cb
    )
  )

  private val ProgramCounter = AtomicLong(0)
  @SuppressWarnings(Array("bonsai:invariant-in-loop"))
  private def generate(using
      q: Quoted
  )(
      ctx: Expr[rt.Context],
      f: Expr[Any],
      dim: Expr[rt.Dim3],
      cb: Expr[Callback[Unit]]
  ) = {
    val log = RenderedLog("")
    val result = for {
      config <- LaunchConfig.fromSettings(LaunchConfig.macroSettings(using q.underlying)).leftMap(CompilerException(_))
      jit = config.jit

      // Name actual => type actual
      // configs               <- reifyConfigFromTpe[C](using q.underlying)()

      (captures, prismRefs, monoMap, programs) <- Compiler.compileExpr(log, f, StdLib.Tables, config)
      prog0 = programs.head._2

      resolveStructDef = (s: p.Type.Struct) => prog0.defs.find(_.name == s.name)

      // Match up the struct types from the capture args so that we know how to copy the structs during argument pickling.
      capturesWithStructDefs <- captures.traverse {
        case (n @ p.Named(_, p.Type.Struct(name, _), _), term) =>
          prog0.defs
            .find(_.name == name)
            .failIfEmpty(s"Missing structure def from capture args ${n.repr}")
            .map(d => (n, Some(d), term))
        case (n @ p.Named(_, tpe, _), term) => (n, None, term).success
      }

      entry0 <- prog0.entry.failIfEmpty("compiled expression has no program entry")
      entry           = entry0.copy(decl = entry0.decl.copy(name = p.Sym(s"lambda${ProgramCounter.getAndIncrement()}")))
      prog: p.Program = prog0.copy(entry = Some(entry))
      programFor      = programs.map((name, program) => name -> program.copy(entry = Some(entry))).toMap

      _ = debug(log.render(1).mkString("\n"))
      _ = debug(entry.repr)
      _ = debug(prog.functions.map(_.repr).mkString("\n"))

      serialisedAsts <- programFor.toList.traverse((name, program) =>
        Either.catchNonFatal(name -> MsgPack.encode(CodeGen.programVersioned(program)))
      )
      serialisedAstFor = serialisedAsts.toMap
      compiler         = ct.Compiler.create()
      configs <- reifiedConfigs(compiler, config)

      compilations <-
        if (jit) Nil.success
        else
          configs.traverse(c =>
            Either
              .catchNonFatal(
                compiler.compile(
                  serialisedAstFor(c.name),
                  emitAssembly = true,
                  ct.Options.of(c.target, c.arch),
                  c.opt.value
                )
              )
              .map(c -> _)
          )
      perConfigLayouts <-
        if (jit)
          configs.traverse(c =>
            Either.catchNonFatal {
              val structDefsAst = MsgPack.encode(CodeGen.structDefsVersioned(programFor(c.name).defs))
              c -> compiler.layoutOf(structDefsAst, ct.Options.of(c.target, c.arch))
            }
          )
        else compilations.map((c, comp) => c -> comp.layouts).success

    } yield {

      compilations.foreach { (config, c) =>
        debug(s"Config=${config}")
        debug(s"Program=${c.program.length}")
        debug(s"Messages=\n  ${c.messages}")
        debug(s"Features=\n  ${c.features.toList}")
        debug(s"Elapsed=\n${c.events.sortBy(_.epochMillis).mkString("\n")}")
        debug(s"Captures=\n${capturesWithStructDefs}")
      }

      given Quotes = q.underlying

      val fnName     = Expr(entry.name.fqcn)
      val moduleName = Expr(s"${entry0.name.fqcn}@${ProgramCounter.getAndIncrement()}")
      val (captureTpeOrdinals, captureTpeSizes) = capturesWithStructDefs.map { (name, _, _) =>
        val tpe = Pickler.tpeAsRuntimeTpe(name.tpe)
        tpe.value -> tpe.sizeInBytes
      }.unzip

      debug(s"Prog defs: ${prog.defs}")

      val modulesExpr: Expr[Array[rt.ModuleImage]] =
        if (jit) {
          val distinctAsts = configs.map(c => serialisedAstFor(c.name).toList).distinct
          val astsExpr     = Expr(distinctAsts.map(_.toArray))
          val jitConfigs = Expr(
            configs.map(c =>
              (c.target.name, c.arch, c.opt.value, distinctAsts.indexOf(serialisedAstFor(c.name).toList))
            )
          )
          '{
            val jitCompiler = ct.Compiler.create()
            val asts        = $astsExpr
            $jitConfigs.map { case (targetName, arch, opt, ast) =>
              val target = ct.Target.valueOf(targetName)
              val comp   = jitCompiler.compile(asts(ast), emitAssembly = false, ct.Options.of(target, arch), opt)
              rt.ModuleImage($moduleName, target, comp.features, comp.program)
            }.toArray
          }
        } else
          '{
            Array[rt.ModuleImage](${
              Varargs(compilations.map { (config, compilation) =>
                '{
                  rt.ModuleImage(
                    $moduleName,
                    ct.Target.valueOf(${ Expr(config.target.name) }),
                    ${ Expr(compilation.features.toList) }.toArray,
                    ${ Expr(compilation.program) }
                  )
                }
              })
            }*)
          }

      val code = '{

        val cb_ : Callback[Unit] = $cb

        lazy val modules = $modulesExpr

        var marshal: Marshal = null

        val outcome: Either[Throwable, Unit] =
          try {
            marshal = Marshal($ctx.queue, null, Platforms.directBufferFromPointer(_, _))
            val found = $ctx.select(modules)
            if (found == -1)
              Left(
                new java.lang.RuntimeException(
                  s"The context does not accept any of the following binaries: ${modules.map(_.target).mkString(",")}"
                )
              )
            else {

              val fnTpeOrdinals = ${ Expr(captureTpeOrdinals.toArray) }
              var fnValues =
                ByteBuffer
                  .allocate(${ Expr(captureTpeSizes.sum) })
                  .order(ByteOrder.nativeOrder)

              ${

                q.Match(
                  'found.asTerm,
                  perConfigLayouts.zipWithIndex.map { case ((config, configLayouts), i) =>
                    val configDefs  = programFor(config.name).defs
                    val lut         = configDefs.map(s => s.name -> s).toMap
                    val lutWithRefs = lut.map((s, sd) => s -> (sd, prismRefs.get(s)))
                    val layouts0    = configLayouts.map(l => p.Sym(l.name) -> l).toMap

                    val layouts = configDefs.flatMap { sd =>
                      layouts0.get(sd.name).map(layout => sd -> (StructLayouts.of(sd, layout), prismRefs.get(sd.name)))
                    }.toMap
                    val allReprsInCaptures = capturesWithStructDefs
                      .collect { case (_, Some(sdef), term) =>
                        Pickler.deriveAllRepr(lutWithRefs, sdef, term.tpe)
                      }
                      .reduceLeftOption(_ ++ _)
                      .getOrElse(Map.empty)

                    val (defs, write, read, update) = Pickler.generateAll(
                      lut,
                      layouts,
                      allReprsInCaptures,
                      '{ polyregion.scalalang.Residents.pointerOf($ctx, _) }
                    )

                    q.CaseDef(
                      q.Literal(q.IntConstant(i)),
                      None,
                      q.Block(
                        defs,
                        '{
                          ${
                            bindWrite(
                              0,
                              write(_, _, 'marshal),
                              'fnValues,
                              capturesWithStructDefs
                            )
                          }

                          if (sys.env.contains(Env.PolyregionDebug)) {
                            println("Dispatch tid=" + Thread.currentThread.getId)
                            println(s"fnTpeOrdinals=${fnTpeOrdinals.toList}")
                            println(s"fnValues.array=0x${fnValues.array.map(byte => f"$byte%02x").mkString(" ")}")
                          }
                          $ctx.launch(
                            Array(modules(found)),
                            $moduleName,
                            $fnName,
                            fnTpeOrdinals,
                            fnValues.array,
                            rt.Policy($dim)
                          )
                          marshal.sync()

                          if (sys.env.contains(Env.PolyregionDebug))
                            println("Kernel completed, tid=" + Thread.currentThread.getId + " cb=" + cb_)

                          try {
                            ${
                              bindRead(
                                0,
                                read(_, _, _, 'marshal),
                                update(_, _, _, 'marshal),
                                'fnValues,
                                capturesWithStructDefs
                              )
                            }
                            if (sys.env.contains(Env.PolyregionDebug)) println("Restore complete")
                            Right(())
                          } catch {
                            case e: Throwable => Left(Exception("Cannot sync", e))
                          }

                        }.asTerm
                      )
                    )
                  } :+ q.CaseDef(q.Wildcard(), None, '{ ??? }.asTerm)
                ).asExprOf[Either[Throwable, Unit]]
              }

            }
          } catch {
            case e: Throwable => Left(e)
          }

        val released =
          try {
            if (marshal != null) marshal.close()
            Right(())
          } catch { case e: Throwable => Left(e) }

        cb_((outcome, released) match {
          case (Left(failure), Left(closing)) => failure.addSuppressed(closing); outcome
          case (Left(_), _)                   => outcome
          case _                              => released
        })
      }
      given q.Printer[q.Tree] = q.Printer.TreeAnsiCode
      debug("Code=" + code.asTerm.show)
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

  def bindWrite(using
      q: Quoted
  )(
      offset: Int,
      write: (p.Sym, Expr[Any]) => Expr[Long],
      target: Expr[ByteBuffer],
      capturesWithStructDefs: List[(p.Named, Option[p.StructDef], q.Term)]
  ) = {
    given Quotes = q.underlying
    val (_, stmts) = capturesWithStructDefs.zipWithIndex.foldLeft((offset, List.empty[Expr[Any]])) {
      case ((byteOffset, exprs), ((name, structDef, ref), idx)) =>
        lazy val refRepr = ref.show
        debug(s"[Bind] [$idx, offset=${byteOffset}]  repr=${refRepr} name=${name.repr} (${structDef.map(_.repr)})")

        val mutable = ref.symbol.flags.is(q.Flags.Mutable)

        // Split the deep tuple match: dotc's patmat exhaustiveness blows up when the scrutinee is
        // (deep-Type-ADT, Option, Expr-with-quote-pattern) - the GADT type maximizer chases
        // every Type variant. Test the simple parts first, then the quoted ref.
        val expr = ((name.tpe, structDef): @scala.unchecked) match {
          case (p.Type.Ptr(_, _), None) =>
            throw RuntimeException(
              s"Top level arrays at parameter boundary is illegal: repr=${refRepr} name=${name.repr}"
            )
          case (p.Type.Struct(_, _), None) =>
            throw RuntimeException(
              s"Struct type without definition at parameter boundary is illegal: repr=${refRepr} name=${name.repr}"
            )
          case (p.Type.Struct(_, _), Some(sdef)) =>
            (ref.asExpr: @scala.unchecked) match {
              case '{ $r: t } =>
                debug(s"$r = " + r.asTerm.symbol.flags.show)
                val rtn = write(sdef.name, r)
                Pickler.writePrim(target, Expr(byteOffset), p.Type.IntS64, rtn)
            }
          case (t, None) =>
            (ref.asExpr: @scala.unchecked) match {
              case '{ $r: u } => Pickler.writePrim(target, Expr(byteOffset), t, r)
            }
          case (t, _) =>
            throw RuntimeException(
              s"Unexpected type ${t.repr} at parameter boundary: repr=${refRepr} name=${name.repr}"
            )
        }
        (byteOffset + Pickler.tpeAsRuntimeTpe(name.tpe).sizeInBytes, expr :: exprs)
    }
    Expr.block(stmts.reverse, '{ () })
  }

  def bindRead(using
      q: Quoted
  )(
      offset: Int,
      read: (p.Sym, Expr[Any], Expr[Long]) => Expr[Any],
      update: (p.Sym, Expr[Any], Expr[Long]) => Expr[Unit],
      target: Expr[ByteBuffer],
      capturesWithStructDefs: List[(p.Named, Option[p.StructDef], q.Term)]
  ) = {
    given Quotes = q.underlying
    val (_, stmts) = capturesWithStructDefs.zipWithIndex.foldLeft((offset, List.empty[Expr[Any]])) {
      case ((byteOffset, exprs), ((name, structDef, ref), idx)) =>
        lazy val refRepr = ref.show
        debug(s"[Bind] <-  [$idx, offset=${byteOffset}]  repr=${refRepr} name=${name.repr} (${structDef.map(_.repr)})")

        val mutable = ref.symbol.flags.is(q.Flags.Mutable)

        val expr = ((name.tpe, structDef): @scala.unchecked) match {
          case (p.Type.Ptr(_, _), None) =>
            throw RuntimeException(
              s"Top level arrays at parameter boundary is illegal: repr=${refRepr} name=${name.repr}"
            )
          case (p.Type.Struct(_, _), None) =>
            throw RuntimeException(
              s"Struct type without definition at parameter boundary is illegal: repr=${refRepr} name=${name.repr}"
            )
          case (p.Type.Struct(_, _), Some(sdef)) =>
            (ref.asExpr: @scala.unchecked) match {
              case '{ $r: t } =>
                debug(s"$r = " + r.asTerm.symbol.flags.show)
                val ptr = Pickler.readPrim(target, Expr(byteOffset), p.Type.IntS64).asExprOf[Long]
                '{
                  if (sys.env.contains(Env.PolyregionDebug)) println(s"Restore ${$r}")
                  ${
                    if (mutable) {
                      q.Assign(r.asTerm, read(sdef.name, r, ptr).asTerm).asExprOf[Unit]
                    } else {
                      update(sdef.name, r, ptr)
                    }
                  }
                }
            }
          case (_, None) =>
            (ref.asExpr: @scala.unchecked) match {
              case '{ $r: u } =>
                '{
                  if (sys.env.contains(Env.PolyregionDebug)) println(s"[Bind] skipping primitive ${$r}")
                }
            }
          case (t, _) =>
            throw RuntimeException(
              s"Unexpected type ${t.repr} at parameter boundary: repr=${refRepr} name=${name.repr}"
            )
        }
        (byteOffset + Pickler.tpeAsRuntimeTpe(name.tpe).sizeInBytes, expr :: exprs)
    }
    Expr.block(stmts.reverse, '{ () })
  }

}
