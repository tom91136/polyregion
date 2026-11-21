package polyregion.javalang

import cats.syntax.all.*
import com.sun.source.tree.MethodTree
import net.bytebuddy.description.method.MethodDescription
import net.bytebuddy.{ByteBuddy, ClassFileVersion}
import net.bytebuddy.description.modifier.{FieldManifestation, Ownership, TypeManifestation, Visibility}
import net.bytebuddy.implementation.bytecode.ByteCodeAppender
import net.bytebuddy.implementation.{FieldAccessor, FixedValue, Implementation}
import net.bytebuddy.jar.asm.{MethodVisitor, Opcodes}
import net.bytebuddy.matcher.ElementMatchers
import polyregion.ast.pass.*
import polyregion.ast.{CodeGen, MsgPack, PolyAST as p, *}
import polyregion.jvm.twin.Layout
import polyregion.jvm.{compiler as ct, runtime as rt}
import polyregion.scalalang.Pickler
import polyregion.twin.StructLayouts

import javax.lang.model.`type`.{ArrayType, DeclaredType, TypeKind, TypeMirror}
import javax.annotation.processing.ProcessingEnvironment
import javax.lang.model.element.{Modifier, TypeElement}
import scala.jdk.CollectionConverters.*

object Offloader {

  // a CONSTANT_Utf8 entry caps at 65535 bytes
  private val ChunkSize = 32768

  private val HostTarget = ct.Target.LLVM_HOST
  private val HostArch   = "native"

  case class Executable(fqcn: String, bytes: Array[Byte], twins: List[TwinEmitter.Emitted] = Nil)

  // the backend normalises non-alphanumerics in symbols, so pre-normalise to keep the runtime's name identical
  private def normalise(s: String): String = s.map(c => if (c.isLetterOrDigit || c == '_') c else '_')

  // MonoStruct renames structs, and layouts are reported under the new names
  private def runPasses(prog: p.Program, log: Log): (Map[p.Sym, p.Sym], p.Program) = {
    val optimised = scala.Function.chain(
      List[ProgramPass](Intrinsify, Specialisation, PartialEval(), DeadArgElimination).map(pass => pass(_, log))
    )(prog)
    MonoStruct(optimised, log)
  }

  // a class named only by a field never reaches a signature and would resolve to an opaque shell
  private def withFieldStructs(
      program: p.Program,
      classTable: Map[String, TypeElement],
      tags: Map[String, Int]
  ): Result[p.Program] = {
    def go(defs: List[p.StructDef]): Result[List[p.StructDef]] = {
      val seen    = defs.map(_.name).toSet
      val missing = defs.flatMap(_.members.map(_.tpe)).flatMap(JavaTyper.structSyms).distinct.filterNot(seen)
      if (missing.isEmpty) defs.success
      else
        missing
          .flatTraverse { sym =>
            val fqn = sym.fqn.mkString(".")
            classTable.get(fqn) match {
              case Some(elem) => JavaTyper.lowerStruct(elem, tags.contains(fqn)).map(List(_))
              case None       => List.empty[p.StructDef].success
            }
          }
          .flatMap {
            // the remainder is out of this compilation's reach
            case Nil   => defs.success
            case added => go(defs ++ added)
          }
    }
    go(program.defs).map(defs => program.copy(defs = defs.distinct))
  }

  private val ResultCapture = "#result"
  private val ErrorCapture  = p.Conventions.ErrorArg

  // the backend prepends the receiver ahead of every other parameter and the runtime prepends the error block, so a
  // receiver kept apart from the captures would sit before the error block
  private def receiverAsCapture(entry: p.Function): p.Function =
    entry.receiver.fold(entry)(r =>
      entry.copy(decl = entry.decl.copy(receiver = None, termCaptures = r +: entry.termCaptures))
    )

  // the JNI layer copies arguments in but never out, so the result goes through a one-element output buffer
  private def routeReturnThroughCapture(entry: p.Function): p.Function =
    if (entry.rtn == p.Type.Unit0) entry
    else {
      val slot                  = p.Named(ResultCapture, p.Type.Ptr(entry.rtn, p.Type.Space.Global))
      val target: p.Term.Select = p.Term.Select(slot, Nil, slot.tpe)
      def rewrite(stmts: List[p.Stmt], from: Int): (List[p.Stmt], Int) =
        stmts.foldLeft((List.empty[p.Stmt], from)) { case ((done, n), stmt) =>
          val (rewritten, next) = rewriteStmt(stmt, n)
          (done ::: rewritten, next)
        }
      def rewriteStmt(stmt: p.Stmt, n: Int): (List[p.Stmt], Int) = stmt match {
        case p.Stmt.Return(e) =>
          val tmp = p.Named(s"#rtn$n", entry.rtn)
          (
            List(
              p.Stmt.Var(tmp, Some(e)),
              p.Stmt.Update(target, p.Term.IntS32Const(0), p.Term.Select(tmp, Nil, entry.rtn)),
              p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
            ),
            n + 1
          )
        case p.Stmt.Cond(c, t, f) =>
          val (t1, n1) = rewrite(t, n)
          val (f1, n2) = rewrite(f, n1)
          (List(p.Stmt.Cond(c, t1, f1)), n2)
        case p.Stmt.While(c, b) =>
          val (b1, n1) = rewrite(b, n)
          (List(p.Stmt.While(c, b1)), n1)
        case p.Stmt.ForRange(i, lb, ub, s, b) =>
          val (b1, n1) = rewrite(b, n)
          (List(p.Stmt.ForRange(i, lb, ub, s, b1)), n1)
        case p.Stmt.Annotated(inner, pos, c) =>
          val (inner1, n1) = rewriteStmt(inner, n)
          (inner1.map(p.Stmt.Annotated(_, pos, c)), n1)
        case other => (List(other), n)
      }
      entry.copy(
        decl = entry.decl.copy(termCaptures = entry.termCaptures :+ p.Arg(slot), rtn = p.Type.Unit0),
        body = rewrite(entry.body, 0)._1 :+ p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    }

  // what the launch fills a kernel parameter from; plan: a<i> captured value, l<i> its length, L<i> per-row lengths,
  // s<owner>#<field> static, t receiver, r result
  case class Slot(name: String, source: String, tpe: p.Type)

  def slotsOf(
      entry: p.Function,
      captures: List[TreeRemapper.Captured],
      hasResult: Boolean
  ): Result[List[Slot]] = {
    val sources = captures.map { c =>
      c.name -> (c.capture match {
        case TreeRemapper.Capture.Arg(i)               => s"a$i"
        case TreeRemapper.Capture.Length(i)            => s"l$i"
        case TreeRemapper.Capture.RowLengths(i)        => s"L$i"
        case TreeRemapper.Capture.StaticField(o, name) => s"s$o#$name"
        case TreeRemapper.Capture.Receiver             => "t"
      })
    }.toMap ++ Option.when(hasResult)(ResultCapture -> "r")
    (entry.moduleCaptures ::: entry.termCaptures ::: entry.args).traverse { a =>
      sources
        .get(a.named.symbol)
        .map(Slot(a.named.symbol, _, a.named.tpe))
        .failIfEmpty(s"Entry `${entry.name.repr}` takes `${a.named.symbol}`, which no launch source supplies")
    }
  }

  // the format is parsed by BinaryOffloadExecutable.invoke
  private def descriptorOf(
      entryArgs: List[String],
      slots: List[Slot],
      resultTpe: p.Type,
      layouts: String,
      twins: List[String],
      features: List[String],
      moduleName: String,
      symbol: String
  ): Result[String] = {
    val (error, launched) = entryArgs.partition(_ == ErrorCapture)
    for {
      _ <-
        if (error.size > 1 || (error.nonEmpty && !entryArgs.headOption.contains(ErrorCapture)))
          s"Entry `$symbol` has the error block in the wrong place: ${entryArgs.mkString(", ")}".fail
        else ().success
      chosen <- launched.traverse(name =>
        slots.find(_.name == name).failIfEmpty(s"Entry `$symbol` takes `$name`, which no launch source supplies")
      )
      descriptor <- Either
        .catchNonFatal {
          val ordinals =
            (rt.Type.LONG +: chosen.map(s => Pickler.tpeAsRuntimeTpe(s.tpe)) :+ rt.Type.VOID).map(_.value)
          s"$moduleName|$symbol|${ordinals.mkString(",")}|${Pickler.tpeAsRuntimeTpe(resultTpe).value}|" +
            s"${chosen.map(_.source).mkString(",")}|$layouts|${twins.mkString(",")}|${features.mkString(",")}"
        }
        .leftMap(e => RuntimeException(s"Cannot derive a launch descriptor for `$symbol`", e))
    } yield descriptor
  }

  private def descriptorOfType(env: ProcessingEnvironment, t: TypeMirror): String = t.getKind match {
    case TypeKind.BOOLEAN => "Z"
    case TypeKind.BYTE    => "B"
    case TypeKind.SHORT   => "S"
    case TypeKind.INT     => "I"
    case TypeKind.LONG    => "J"
    case TypeKind.CHAR    => "C"
    case TypeKind.FLOAT   => "F"
    case TypeKind.DOUBLE  => "D"
    case _ =>
      env.getTypeUtils.erasure(t) match {
        case a: ArrayType => "[" + descriptorOfType(env, a.getComponentType)
        case d: DeclaredType =>
          d.asElement match {
            case e: TypeElement => s"L${env.getElementUtils.getBinaryName(e).toString.replace('.', '/')};"
            case _              => ""
          }
        case _ => ""
      }
  }

  private case class Host(elem: TypeElement, binary: String, fqn: String)

  private def hostFieldsOf(env: ProcessingEnvironment, host: Host): Map[String, TwinEmitter.HostField] =
    JavaTyper
      .structFieldElements(host.elem)
      .map { f =>
        val name = f.getSimpleName.toString
        name -> TwinEmitter.HostField(
          host.binary,
          name,
          descriptorOfType(env, f.asType),
          !f.getModifiers.contains(Modifier.PRIVATE),
          f.getModifiers.contains(Modifier.FINAL)
        )
      }
      .toMap

  // offsets come from the compiler so the host packs exactly what the kernel reads, keyed by binary name
  private def layoutsOf(
      prog: p.Program,
      hosts: Map[String, Host],
      tags: Map[String, Int],
      env: ProcessingEnvironment
  ): Result[List[Layout]] =
    if (prog.defs.isEmpty) Nil.success
    else
      Either.catchNonFatal {
        import polyregion.ast.MsgPack.Codec.given
        val ast      = MsgPack.encode(CodeGen.structDefsVersioned(prog.defs))
        val compiler = ct.Compiler.create()
        val laid =
          try compiler.layoutOf(ast, ct.Options.of(HostTarget, HostArch))
          finally compiler.close()
        val nameOf = (sym: p.Sym) => hosts.get(sym.repr).fold(sym.repr)(_.binary)
        val defs   = prog.defs.map(d => d.name.repr -> d).toMap
        laid.toList.flatMap { l =>
          defs.get(l.name).map { sdef =>
            val host   = hosts.get(l.name)
            val fields = host.fold(Map.empty[String, TwinEmitter.HostField])(hostFieldsOf(env, _))
            StructLayouts.of(
              // synthesised storage has no Java field, the dispatch tag is found through the layout
              sdef.copy(members = sdef.members.filterNot(_.symbol.startsWith("#"))),
              l,
              host.fold("")(_.binary),
              host.flatMap(h => tags.get(h.fqn)),
              nameOf,
              name => fields.get(name).fold("")(_.descriptor),
              name => fields.get(name).exists(_.readOnly)
            )
          }
        }
      }

  private def compileToObject(prog: p.Program): Result[(Array[Byte], List[String], List[String])] = for {
    ast <- Either.catchNonFatal {
      MsgPack.encode(CodeGen.programVersioned(prog))
    }
    compilation <- Either.catchNonFatal {
      val compiler = ct.Compiler.create()
      try compiler.compile(ast, emitAssembly = false, ct.Options.of(HostTarget, HostArch), ct.Opt.O2.value)
      finally compiler.close()
    }
    image <- Option(compilation.program)
      .filter(_.nonEmpty)
      .failIfEmpty(s"Native compilation produced no object:\n${compilation.messages}")
  } yield (image, compilation.features.toList, compilation.entryArgs.toList)

  // an array is no constant, so chunks go in as strings and join at class load
  private class BinaryInitialiser(fqcn: String, chunks: List[String]) extends ByteCodeAppender {
    override def apply(
        mv: MethodVisitor,
        ctx: Implementation.Context,
        method: MethodDescription
    ): ByteCodeAppender.Size = {
      mv.visitLdcInsn(chunks.size)
      mv.visitTypeInsn(Opcodes.ANEWARRAY, "java/lang/String")
      chunks.zipWithIndex.foreach { (chunk, i) =>
        mv.visitInsn(Opcodes.DUP)
        mv.visitLdcInsn(i)
        mv.visitLdcInsn(chunk)
        mv.visitInsn(Opcodes.AASTORE)
      }
      val isInterface = false
      mv.visitMethodInsn(
        Opcodes.INVOKESTATIC,
        "polyregion/javalang/BinaryOffloadExecutable",
        "decodeBinary",
        "([Ljava/lang/String;)[B",
        isInterface
      )
      mv.visitFieldInsn(Opcodes.PUTSTATIC, fqcn.replace('.', '/'), "$BINARY$", "[B")
      ByteCodeAppender.Size(4, method.getStackSize)
    }
  }

  private def emitClass(fqcn: String, image: Array[Byte], descriptor: String): Result[Array[Byte]] =
    Either.catchNonFatal {
      val chunks = java.util.Base64.getEncoder.encodeToString(image).grouped(ChunkSize).toList
      // unpinned, ByteBuddy stamps the build JVM's class-file version
      ByteBuddy(ClassFileVersion.JAVA_V8)
        .subclass(classOf[BinaryOffloadExecutable])
        .name(fqcn)
        .modifiers(Visibility.PUBLIC, TypeManifestation.FINAL)
        .defineField("$BINARY$", classOf[Array[Byte]], Visibility.PUBLIC, Ownership.STATIC, FieldManifestation.FINAL)
        .initializer(BinaryInitialiser(fqcn, chunks))
        .method(ElementMatchers.named("binary"))
        .intercept(FieldAccessor.ofField("$BINARY$"))
        .method(ElementMatchers.named("descriptor"))
        .intercept(FixedValue.value(descriptor))
        .make()
        .getBytes
    }

  // a descriptor starting with this is a diagnostic thrown at invoke, so one unsupported region fails one test
  private val UnsupportedMarker = "!"
  private val ReasonLimit       = 4000

  def unsupported(fqcn: String, reason: String): Result[Executable] =
    emitClass(fqcn, Array.emptyByteArray, UnsupportedMarker + reason.take(ReasonLimit))
      .map(Executable(fqcn, _))

  def compile(sources: TreeRemapper.Sources, entry: MethodTree, fqcn: String): Result[Executable] = {
    import sources.{classTable, env}
    val log    = PluginEntry.defaultLog
    val symbol = normalise(fqcn)
    for {
      region   <- TreeRemapper.compileRegion(sources, entry)
      raw      <- withFieldStructs(region.program, classTable, region.tags)
      rawEntry <- raw.entry.failIfEmpty(s"Region `$symbol` has no entry")
      resultTpe = rawEntry.rtn
      routed = routeReturnThroughCapture(
        receiverAsCapture(rawEntry.copy(decl = rawEntry.decl.copy(name = p.Sym(symbol))))
      )
      slots <- slotsOf(routed, region.captures, resultTpe != p.Type.Unit0)
      named              = raw.copy(entry = Some(routed))
      passed             = runPasses(named, log)
      (monoToPoly, prog) = passed
      failures           = Verify(prog, log, verifyFunction = false).filter(_._2.nonEmpty)
      _ <-
        if (failures.isEmpty) ().success
        else
          s"Verification failed for `$symbol`:\n${failures
              .map((fn, es) => s"${fn.signatureKey}\n${es.map("  -> " + _).mkString("\n")}")
              .mkString("\n")}".fail
      hosts = prog.defs.flatMap { d =>
        val original = monoToPoly.getOrElse(d.name, d.name)
        val fqn      = original.fqn.mkString(".")
        classTable.get(fqn).map(e => d.name.repr -> Host(e, env.getElementUtils.getBinaryName(e).toString, fqn))
      }.toMap
      layouts  <- layoutsOf(prog, hosts, region.tags, env)
      _        <- prog.entry.failIfEmpty(s"Region `$symbol` lost its entry")
      compiled <- compileToObject(prog)
      (image, features, entryArgs) = compiled
      twins = layouts.zipWithIndex.flatMap { (layout, i) =>
        hosts.values.find(_.binary == layout.className).flatMap { host =>
          val pkg    = env.getElementUtils.getPackageOf(host.elem).getQualifiedName.toString
          val simple = fqcn.substring(fqcn.lastIndexOf('.') + 1)
          val name   = s"$simple$$Twin$$$i"
          TwinEmitter.emit(if (pkg.isEmpty) name else s"$pkg.$name", layout, hostFieldsOf(env, host))
        }
      }
      descriptor <- descriptorOf(
        entryArgs,
        slots,
        resultTpe,
        Layout.encodeAll(layouts.asJava),
        twins.map(_.fqcn),
        features,
        fqcn,
        symbol
      )
      bytes <- emitClass(fqcn, image, descriptor)
    } yield Executable(fqcn, bytes, twins)
  }
}
