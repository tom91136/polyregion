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
    (captures, prismRefs, monoMap, prog0, log) <- Compiler.compileExpr(f)

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
                  prog.defs.map(sd => sd -> (layouts(sd.name), prismRefs.get(sd.name))).toMap,
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

  type PtrMapTpe = _root_.scala.collection.mutable.Map[Any, Long]
  type ObjMapTpe = _root_.scala.collection.mutable.Map[Long, Any]

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

  def deriveAllRepr(using q: Quoted)(
      lut: Map[p.Sym, p.StructDef],
      sdef: p.StructDef,
      repr: q.TypeRepr,
      added: Set[p.Sym] = Set()
  ): List[(p.StructDef, q.TypeRepr)] = {
    val added0 = added + sdef.name
    (sdef, repr.widenTermRefByName) :: sdef.members.flatMap(
      _.named match {
        case (p.Named(_, p.Type.Struct(name, _, _))) if added0.contains(name) => Nil
        case (p.Named(member, p.Type.Struct(name, _, _))) =>
          deriveAllRepr(lut, lut(name), q.TermRef(repr, member), added0)
        case _ => Nil
      }
    )
  }

  def mkMethodSym(using q: Quoted)(name: String, rtn: q.TypeRepr, args: (String, q.TypeRepr)*) = q.Symbol.newMethod(
    q.Symbol.spliceOwner,
    name,
    q.MethodType(args.map(_._1).toList)(paramInfosExp = _ => args.map(_._2).toList, resultTypeExp = _ => rtn)
  )

  def mkMethodDef(using q: Quoted)(sym: q.Symbol)(impl: PartialFunction[List[q.Tree], q.Term]) = q.DefDef(
    sym,
    {
      case (argList0 :: Nil) =>
        impl.lift(argList0).fold(q.report.errorAndAbort(s"Definition is not defined for input ${argList0}"))(Some(_))
      case bad => q.report.errorAndAbort(s"Unexpected argument in method body: expected ${sym.signature}, got ${bad}")
    }
  )

  def generateAll(using q: Quoted)(
      layouts: Map[p.StructDef, (ct.Layout, Option[polyregion.prism.TermPrism[Any, Any]])],
      lut: Map[p.Sym, p.StructDef],
      sdef: p.StructDef,
      repr: q.TypeRepr
  )(root: Expr[Any], ptrMap: Expr[PtrMapTpe]) = {
    given Quotes = q.underlying

    val sdefToRepr = deriveAllRepr(lut, sdef, repr).toMap

    val writeReadSymbols = sdefToRepr.map((sdef, repr) =>
      sdef -> (mkMethodSym(
        s"write_${sdef.name.repr}",
        q.TypeRepr.of[Long],
        "root"   -> repr,
        "ptrMap" -> q.TypeRepr.of[PtrMapTpe]
      ), mkMethodSym(
        s"read_${sdef.name.repr}",
        repr,
        "ptr"    -> q.TypeRepr.of[Long],
        "ptrMap" -> q.TypeRepr.of[PtrMapTpe],
        "objMap" -> q.TypeRepr.of[ObjMapTpe]
      ), mkMethodSym(
        s"update_${sdef.name.repr}",
        q.TypeRepr.of[Long],
        "root"   -> repr,
        "ptr"    -> q.TypeRepr.of[Long],
        "ptrMap" -> q.TypeRepr.of[PtrMapTpe],
        "objMap" -> q.TypeRepr.of[ObjMapTpe]
      ))
    )

    def allocateBuffer(size: Expr[Int]) = '{ ByteBuffer.allocateDirect($size).order(ByteOrder.nativeOrder()) }

    def callWrite(name: p.Sym, root: q.Term, ptrMap: Expr[PtrMapTpe]) =
      q.Apply(q.Ref(writeReadSymbols(lut(name))._1), List(root, ptrMap.asTerm)).asExprOf[Long]

    def callRead(name: p.Sym, root: q.Term, ptr: Expr[Long], ptrMap: Expr[PtrMapTpe], objMap: Expr[ObjMapTpe]) =
      q.Apply(q.Ref(writeReadSymbols(lut(name))._2), List(root, ptr.asTerm, ptrMap.asTerm, objMap.asTerm)).asExpr

    def callUpdate(name: p.Sym, root: q.Term, ptr: Expr[Long], ptrMap: Expr[PtrMapTpe], objMap: Expr[ObjMapTpe]) =
      q.Apply(q.Ref(writeReadSymbols(lut(name))._3), List(root, ptr.asTerm, ptrMap.asTerm, objMap.asTerm))
        .asExprOf[Unit]

    def writeArray[t: Type](expr: Expr[StdLib.MutableSeq[t]], comp: p.Type, ptrMap: Expr[PtrMapTpe]): Expr[Long] = {
      val elementSizeInBytes = Pickler.tpeAsRuntimeTpe(comp).sizeInBytes()
      '{
        val arrBuffer = ${ allocateBuffer('{ ${ Expr(elementSizeInBytes) } * $expr.length_ }) }
        val arrPtr    = Platforms.pointerOfDirectBuffer(arrBuffer)
        println(
          s"[bind]: array  [${$expr.length_} * ${${ Expr(comp.repr) }}]${$expr.data} => $arrBuffer(0x${arrPtr.toHexString})"
        )
        var i = 0
        while (i < $expr.length_) {
          ${
            val elementOffset = '{ ${ Expr(elementSizeInBytes) } * i }
            comp match {
              case p.Type.Struct(name, _, _) =>
                val ptr = callWrite(name, '{ $expr(i) }.asTerm, ptrMap)
                Pickler.writePrim('arrBuffer, elementOffset, p.Type.Long, ptr)
              case t =>
                Pickler.writePrim('arrBuffer, elementOffset, t, '{ $expr(i) })
            }
          }
          i += 1
        }
        arrPtr
      }
    }

    def writeMapping[t: Type](root: Expr[t], ptrMap: Expr[PtrMapTpe], mapping: Pickler.StructMapping[q.Term]) = '{
      val buffer = ${ allocateBuffer(Expr(mapping.sizeInBytes.toInt)) }
      val ptr    = Platforms.pointerOfDirectBuffer(buffer)
      println(s"[bind]: object  ${$root} => $buffer(0x${ptr.toHexString})")
      ${
        Varargs(mapping.members.map { m =>
          val memberOffset = Expr(m.offsetInBytes.toInt)
          (root, m.tpe) match {
            case ('{ $seq: StdLib.MutableSeq[t] }, p.Type.Array(comp)) =>
              val ptr = writeArray[t](seq, comp, ptrMap)
              Pickler.writePrim('buffer, memberOffset, p.Type.Long, ptr)
            case (_, p.Type.Struct(name, _, _)) =>
              val ptr = callWrite(name, m.select(root.asTerm), ptrMap)
              Pickler.writePrim('buffer, memberOffset, p.Type.Long, ptr)
            case (_, _) =>
              Pickler.writePrim('buffer, memberOffset, m.tpe, m.select(root.asTerm).asExpr)
          }
        })
      }
      $ptrMap += ($root -> ptr)
      ptr
    }

    def readArray[t: Type](
        seq: Expr[StdLib.MutableSeq[t]],
        comp: p.Type,
        ptrMap: Expr[PtrMapTpe],
        objMap: Expr[ObjMapTpe],
        memberOffset: Expr[Int],
        buffer: Expr[ByteBuffer],
        mapping: Pickler.StructMapping[q.Term]
    ): Expr[Unit] = {
      val elementSizeInBytes = Pickler.tpeAsRuntimeTpe(comp).sizeInBytes()
      '{
        val arrayLen = ${
          mapping.members.headOption match {
            case Some(lengthMember) if lengthMember.tpe == p.Type.Int =>
              Pickler
                .readPrim(buffer, Expr(lengthMember.offsetInBytes.toInt), lengthMember.tpe)
                .asExprOf[Int]
            case _ =>
              q.report.errorAndAbort(s"Illegal structure while encoding read for member ${mapping} ")
          }
        }
        val arrPtr    = ${ Pickler.readPrim(buffer, memberOffset, p.Type.Long).asExprOf[Int] }
        val arrBuffer = Platforms.directBufferFromPointer(arrPtr, ${ Expr(elementSizeInBytes) } * arrayLen)
        var i         = 0
        while (i < arrayLen) {
          $seq(i) = ${
            val elementOffset = '{ ${ Expr(elementSizeInBytes) } * i }
            comp match {
              case p.Type.Struct(name, _, _) =>
                val arrElemPtr = Pickler.readPrim('arrBuffer, elementOffset, p.Type.Long).asExprOf[Long]
                callRead(name, '{ $seq(i) }.asTerm, arrElemPtr, ptrMap, objMap).asExprOf[t]
              case t =>
                Pickler.readPrim('arrBuffer, elementOffset, t).asExprOf[t]
            }
          }
          i += 1
        }
      }
    }

    def readMapping[t: Type](
        root: Expr[t],
        ptr: Expr[Long],
        ptrMap: Expr[PtrMapTpe],
        objMap: Expr[ObjMapTpe],
        mapping: Pickler.StructMapping[q.Term]
    ) = '{
      val buffer = Platforms.directBufferFromPointer($ptr, ${ Expr(mapping.sizeInBytes.toInt) })
      println(s"[bind]: object  ${$root} <- $buffer(0x${$ptr.toHexString})")
      ${
        Varargs(
          mapping.members.map { m =>
            val memberOffset = Expr(m.offsetInBytes.toInt)
            (root, m.tpe) match {
              case ('{ $seq: StdLib.MutableSeq[t] }, p.Type.Array(comp)) =>
                readArray[t](seq, comp, ptrMap, objMap, memberOffset, 'buffer, mapping)
              case (_, p.Type.Struct(name, _, _)) =>
                val structPtr = Pickler.readPrim('buffer, memberOffset, p.Type.Long).asExprOf[Long]
                val select    = m.select(root.asTerm)
                if (m.mut) {
                  q.Assign(m.select(root.asTerm), callRead(name, select, structPtr, ptrMap, objMap).asTerm)
                    .asExprOf[Unit]
                } else {
                  callUpdate(name, select, structPtr, ptrMap, objMap)
                }
              case (_, _) =>
                if (m.mut) {
                  q.Assign(m.select(root.asTerm), Pickler.readPrim('buffer, memberOffset, m.tpe).asTerm).asExprOf[Unit]
                } else '{ () } // otherwise no-op
            }
          }
        )
      }

    }

    val (writeDefs, readDefs, updateDefs) = writeReadSymbols.toList
      .map((sdef, x) => Pickler.mkStructMapping(sdef, layouts) -> x)
      .map { case (mapping, (writeSymbol, readSymbol, updateSymbol)) =>
        val writeMethod = mkMethodDef(writeSymbol) { case List(root: q.Term, ptrMap: q.Term) =>
          (mapping.write(root).asExpr, ptrMap.asExpr) match {
            case ('{ $expr: t }, '{ $ptrMap: PtrMapTpe }) =>
              '{
                val root = $expr
                $ptrMap.get(root) match {
                  case Some(existing)       => existing
                  case None if root == null => $ptrMap += (root -> 0); 0
                  case None                 => ${ writeMapping('root, ptrMap, mapping) }
                }
              }.asTerm.changeOwner(writeSymbol)
          }
        }
        val readMethod = mkMethodDef(readSymbol) {
          case List(root: q.Term, ptr: q.Term, ptrMap: q.Term, objMap: q.Term) =>
            (mapping.write(root).asExpr, ptr.asExpr, ptrMap.asExpr, objMap.asExpr) match {
              case ('{ $rootExpr: t | Null }, '{ $ptrExpr: Long }, '{ $ptrMap: PtrMapTpe }, '{ $objMap: ObjMapTpe }) =>
                '{
                  ($ptrMap.get($rootExpr), $ptrExpr) match {
                    case (_, 0) => null // object reassignment for var to null
                    case (Some(writePtr), readPtr) if writePtr == readPtr => // same ptr, do the update
                      ${ readMapping(rootExpr, 'readPtr, ptrMap, objMap, mapping) }; $rootExpr
                    case (Some(writePtr), readPtr) if writePtr != readPtr => // object reassignment for var
                      // Make sure we update the old writePtr (possibly orphaned, unless reassigned somewhere else) first.
                      // This is to make sure modified object without a root (e.g through reassignment) is corrected updated.
                      ${ callUpdate(sdef.name, rootExpr.asTerm, 'writePtr, ptrMap, objMap) }
                      // Now, readPtr is either a new allocation or a an existing one, possibly shared.
                      // We check that it hasn't already been read/updated yet (the object may be recursive) and proceed to
                      // create the object.
                      $objMap.get(readPtr) match {
                        case Some(existing) => existing // Existing allocation found, use it.
                        case None           => ()
                      }
                    case (None, readPtr) => // object not previously written, fail
                      throw new RuntimeException(
                        s"Val root object ${$rootExpr} was not previously written, cannot restore from to 0x${readPtr.toHexString}"
                      )
                  }
                }.asTerm.changeOwner(readSymbol)
            }
        }
        val updateMethod = mkMethodDef(updateSymbol) {
          case List(root: q.Term, ptr: q.Term, ptrMap: q.Term, objMap: q.Term) =>
            (mapping.write(root).asExpr, ptr.asExpr, ptrMap.asExpr, objMap.asExpr) match {
              case ('{ $rootExpr: t }, '{ $ptrExpr: Long }, '{ $ptrMap: PtrMapTpe }, '{ $objMap: ObjMapTpe }) =>
                '{
                  ($ptrMap.get($rootExpr), $ptrExpr) match {
                    case (Some(0), 0) => () // was null, still null, no-op
                    case (Some(writePtr), readPtr) if writePtr == readPtr => // same ptr, do the update
                      ${ readMapping(rootExpr, 'readPtr, ptrMap, objMap, mapping) }
                      $objMap += (readPtr -> $rootExpr)
                    case (Some(writePtr), readPtr) if writePtr != readPtr => // object reassignment for val, fail
                      throw new RuntimeException(
                        s"Cannot update immutable val, setting ${$rootExpr} (0x${writePtr.toHexString}) to 0x${readPtr.toHexString}"
                      )
                    case (None, readPtr) => // object not previously written, fail
                      throw new RuntimeException(
                        s"Val root object ${$rootExpr} was not previously written, cannot restore from to 0x${readPtr.toHexString}"
                      )
                  }
                }.asTerm.changeOwner(updateSymbol)
            }
        }
        //
        (writeMethod, readMethod, updateMethod)
      }
      .unzip3

    q.Block(writeDefs ::: readDefs ::: updateDefs, callWrite(sdef.name, root.asTerm, ptrMap).asTerm)
  }

  def bindCapturesToBuffer(using q: Quoted)(
      layouts: Map[p.StructDef, (ct.Layout, Option[polyregion.prism.TermPrism[Any, Any]])],
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
            val rtn = generateAll(layouts, lut, sdef, ref.asTerm.tpe)(
              ref,
              '{ _root_.scala.collection.mutable.Map[Any, Long]() }
            ).asExpr

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

}
