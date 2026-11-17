package polyregion.javalang

import cats.syntax.all.*
import com.sun.source.tree.MethodTree
import com.sun.source.util.Trees
import com.sun.tools.javac.code.Symbol
import net.bytebuddy.description.method.MethodDescription
import net.bytebuddy.{ByteBuddy, ClassFileVersion}
import net.bytebuddy.description.modifier.{FieldManifestation, Ownership, TypeManifestation, Visibility}
import net.bytebuddy.implementation.bytecode.ByteCodeAppender
import net.bytebuddy.implementation.{FieldAccessor, FixedValue, Implementation}
import net.bytebuddy.jar.asm.{MethodVisitor, Opcodes}
import net.bytebuddy.matcher.ElementMatchers
import polyregion.ast.pass.*
import polyregion.ast.{CodeGen, MsgPack, PolyAST as p, *}
import polyregion.jvm.{compiler as ct, runtime as rt}
import polyregion.scalalang.Pickler

import java.util.Base64
import javax.annotation.processing.ProcessingEnvironment
import javax.lang.model.element.TypeElement

object Offloader {

  // a CONSTANT_Utf8 entry caps at 65535 bytes
  private val ChunkSize = 32768

  private val HostTarget = ct.Target.LLVM_HOST
  private val HostArch   = "native"

  final case class Executable(fqcn: String, bytes: Array[Byte])

  // the backend normalises non-alphanumerics in symbols, so pre-normalise to keep the runtime's name identical
  private def normalise(s: String): String = s.map(c => if (c.isLetterOrDigit || c == '_') c else '_')

  // MonoStruct renames structs, and layouts are reported under the new names
  private def runPasses(prog: p.Program, log: Log): (Map[p.Sym, p.Sym], p.Program) = {
    val optimised = scala.Function.chain(
      List[ProgramPass](Intrinsify, Specialisation, PartialEval(), DeadArgElimination).map(pass => pass(_, log))
    )(prog)
    val (monoToPoly, mono) = MonoStruct(optimised, log)
    // the native compiler runs StructuredExit too, but the descriptor is settled here so the error buffer must exist first
    (monoToPoly, hoistErrorBuffer(StructuredExit(mono, log)))
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

  // the runtime packs the launch buffer from captures alone, so the error buffer moves across; the native ABI is unchanged
  private def hoistErrorBuffer(prog: p.Program): p.Program = prog.copy(entry = prog.entry.map { entry =>
    val (error, rest) = entry.args.partition(_.named.symbol == ErrorCapture)
    if (error.isEmpty) entry
    else entry.copy(decl = entry.decl.copy(args = rest, termCaptures = entry.termCaptures ++ error))
  })

  // the JNI layer copies arguments in but never out, so the result goes through a one-element output buffer
  private def routeReturnThroughCapture(entry: p.Function): p.Function =
    if (entry.rtn == p.Type.Unit0) entry
    else {
      val slot                  = p.Named(ResultCapture, p.Type.Ptr(entry.rtn, p.Type.Space.Global))
      val target: p.Term.Select = p.Term.Select(slot, Nil, slot.tpe)
      var fresh                 = 0
      def rewrite(stmts: List[p.Stmt]): List[p.Stmt] = stmts.flatMap {
        case p.Stmt.Return(e) =>
          val tmp = p.Named(s"#rtn$fresh", entry.rtn)
          fresh += 1
          List(
            p.Stmt.Var(tmp, Some(e)),
            p.Stmt.Update(target, p.Term.IntS32Const(0), p.Term.Select(tmp, Nil, entry.rtn)),
            p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
          )
        case p.Stmt.Cond(c, t, f)             => List(p.Stmt.Cond(c, rewrite(t), rewrite(f)))
        case p.Stmt.While(c, b)               => List(p.Stmt.While(c, rewrite(b)))
        case p.Stmt.ForRange(i, lb, ub, s, b) => List(p.Stmt.ForRange(i, lb, ub, s, rewrite(b)))
        case p.Stmt.Annotated(inner, pos, c)  => rewrite(List(inner)).map(p.Stmt.Annotated(_, pos, c))
        case other                            => List(other)
      }
      entry.copy(
        decl = entry.decl.copy(termCaptures = entry.termCaptures :+ p.Arg(slot), rtn = p.Type.Unit0),
        body = rewrite(entry.body) :+ p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    }

  // `<module>|<symbol>|<launch ordinals>|<result ordinal>|<plan>|<layouts>|<tags>|<features>`, parsed by BinaryOffloadExecutable.invoke
  // plan: a<i> captured value, l<i> its length, L<i> per-row lengths, s<owner>#<field> static, t receiver, r result, e error buffer
  private def descriptorOf(
      entry: p.Function,
      resultTpe: p.Type,
      captures: List[TreeRemapper.Capture],
      layouts: String,
      tags: String,
      features: List[String],
      moduleName: String,
      symbol: String
  ): Result[String] = Either
    .catchNonFatal {
      val receiver = entry.receiver.map(a => Pickler.tpeAsRuntimeTpe(a.named.tpe)).toList
      val ordinals =
        (rt.Type.LONG +: (receiver ++ entry.termCaptures.map(a => Pickler.tpeAsRuntimeTpe(a.named.tpe))) :+ Pickler
          .tpeAsRuntimeTpe(entry.rtn)).map(_.value)
      val plan = captures.map {
        case TreeRemapper.Capture.Arg(i)               => s"a$i"
        case TreeRemapper.Capture.Length(i)            => s"l$i"
        case TreeRemapper.Capture.RowLengths(i)        => s"L$i"
        case TreeRemapper.Capture.StaticField(o, name) => s"s$o#$name"
        case TreeRemapper.Capture.Receiver             => "t"
      } ++ Option.when(resultTpe != p.Type.Unit0)("r") ++
        Option.when(entry.termCaptures.exists(_.named.symbol == ErrorCapture))("e")
      s"$moduleName|$symbol|${ordinals.mkString(",")}|${Pickler.tpeAsRuntimeTpe(resultTpe).value}|" +
        s"${plan.mkString(",")}|$layouts|$tags|${features.mkString(",")}"
    }
    .leftMap(e => RuntimeException(s"Cannot derive a launch descriptor for `$symbol`", e))

  // offsets come from the compiler so the host packs exactly what the kernel reads, keyed by binary name
  private def layoutsOf(prog: p.Program, binaryNames: Map[String, String]): Result[String] =
    if (prog.defs.isEmpty) "".success
    else
      Either.catchNonFatal {
        import polyregion.ast.MsgPack.Codec.given
        val ast      = MsgPack.encode(CodeGen.structDefsVersioned(prog.defs))
        val compiler = ct.Compiler.create()
        val layouts =
          try compiler.layoutOf(ast, ct.Options.of(HostTarget, HostArch))
          finally compiler.close()
        layouts
          .map { l =>
            // synthesised storage has no Java field, except the dispatch tag the host writes
            val members = l.members
              .filter(m => !m.name.startsWith("#") || m.name == JavaTyper.TagMember)
              .map(m => s"${m.name}@${m.offsetInBytes}:${m.sizeInBytes}")
              .mkString(",")
            s"${binaryNames.getOrElse(l.name, l.name)}:${l.sizeInBytes}:$members"
          }
          .mkString("~")
      }

  private def compileToObject(prog: p.Program): Result[(Array[Byte], List[String])] = for {
    ast <- Either.catchNonFatal {
      import polyregion.ast.MsgPack.Codec.given
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
  } yield (image, compilation.features.toList)

  // an array is no constant, so chunks go in as strings and join at class load
  private final class BinaryInitialiser(fqcn: String, chunks: List[String]) extends ByteCodeAppender {
    override def apply(
        mv: MethodVisitor,
        ctx: Implementation.Context,
        method: MethodDescription
    ): ByteCodeAppender.Size = {
      mv.visitLdcInsn(Integer.valueOf(chunks.size))
      mv.visitTypeInsn(Opcodes.ANEWARRAY, "java/lang/String")
      chunks.zipWithIndex.foreach { (chunk, i) =>
        mv.visitInsn(Opcodes.DUP)
        mv.visitLdcInsn(Integer.valueOf(i))
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
      val chunks = Base64.getEncoder.encodeToString(image).grouped(ChunkSize).toList
      // unpinned, ByteBuddy stamps the build JVM's class-file version
      new ByteBuddy(ClassFileVersion.JAVA_V8)
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

  def compile(
      env: ProcessingEnvironment,
      trees: Trees,
      classTable: Map[String, TypeElement],
      methodTable: Map[String, MethodTree],
      localTrees: Map[Symbol.MethodSymbol, MethodTree],
      entry: MethodTree,
      fqcn: String
  ): Result[Executable] = {
    val log    = PluginEntry.defaultLog
    val symbol = normalise(fqcn)
    for {
      region   <- TreeRemapper.compileRegion(env, trees, classTable, methodTable, localTrees, entry)
      raw      <- withFieldStructs(region.program, classTable, region.tags)
      rawEntry <- raw.entry.failIfEmpty(s"Region `$symbol` has no entry")
      resultTpe = rawEntry.rtn
      named = raw.copy(entry =
        Some(routeReturnThroughCapture(rawEntry.copy(decl = rawEntry.decl.copy(name = p.Sym(symbol)))))
      )
      passed             = runPasses(named, log)
      (monoToPoly, prog) = passed
      failures           = Verify(prog, log, verifyFunction = false).filter(_._2.nonEmpty)
      _ <-
        if (failures.isEmpty) ().success
        else
          s"Verification failed for `$symbol`:\n${failures
              .map((fn, es) => s"${fn.signatureKey}\n${es.map("  -> " + _).mkString("\n")}")
              .mkString("\n")}".fail
      binaryNames = prog.defs.flatMap { d =>
        val original = monoToPoly.getOrElse(d.name, d.name)
        classTable
          .get(original.fqn.mkString("."))
          .map(e => d.name.repr -> env.getElementUtils.getBinaryName(e).toString)
      }.toMap
      layouts <- layoutsOf(prog, binaryNames)
      tags = region.tags.toList
        .flatMap((fqn, tag) => classTable.get(fqn).map(e => s"${env.getElementUtils.getBinaryName(e)}=$tag"))
        .mkString("~")
      entry    <- prog.entry.failIfEmpty(s"Region `$symbol` lost its entry")
      compiled <- compileToObject(prog)
      (image, features) = compiled
      descriptor <- descriptorOf(entry, resultTpe, region.captures, layouts, tags, features, fqcn, symbol)
      bytes      <- emitClass(fqcn, image, descriptor)
    } yield Executable(fqcn, bytes)
  }
}
