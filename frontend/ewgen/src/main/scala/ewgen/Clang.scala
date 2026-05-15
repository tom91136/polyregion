package ewgen

import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.nio.file.Path

import scala.annotation.tailrec
import scala.sys.process.{Process, ProcessLogger}

import upickle.default.*

object Clang {

  case class SourceSpan(
      beginLine: Int,
      endLine: Int,
      beginOffset: Option[Long],
      endOffset: Option[Long],
      endTokLen: Option[Int]
  )

  case class FunctionDecl(
      name: String,
      qualType: String,
      span: SourceSpan,
      signature: String,
      docComment: String,
      versionTag: Option[String]
  )

  case class TypeDecl(sourceFile: String, span: SourceSpan)
  case class MacroDef(name: String, raw: String)
  case class Probe(
      functions: Vector[FunctionDecl],
      typeSlices: Vector[String],
      macros: Vector[MacroDef]
  )

  private case class Position(
      offset: Option[Long] = None,
      line: Option[Int] = None,
      tokLen: Option[Int] = None,
      file: Option[String] = None
  ) derives ReadWriter
  private case class Range(begin: Position, end: Position) derives ReadWriter
  private case class TypeInfo(qualType: String) derives ReadWriter
  private case class Node(
      kind: Option[String] = None,
      name: Option[String] = None,
      loc: Option[Position] = None,
      range: Option[Range] = None,
      `type`: Option[TypeInfo] = None,
      isImplicit: Option[Boolean] = None,
      inner: Option[Vector[Node]] = None
  ) derives ReadWriter

  private case class SourceView(text: String, lines: Vector[String])

  private val DefineRe = """^#define\s+([A-Za-z_][A-Za-z0-9_]*)(.*)$""".r
  private val OpenClSuffixRe =
    """(?s)\b([A-Za-z_][A-Za-z0-9_]*)\s*\([^;]*?\)\s+CL_(?:API|EXT)_SUFFIX__VERSION_([0-9])_([0-9])(?:_DEPRECATED)?\s*;""".r
  private val LeafDeclKinds = Set("TypedefDecl", "EnumDecl", "RecordDecl", "FunctionDecl")

  private def withTempFile[A](prefix: String, suffix: String)(use: Path => A): A = {
    val p = Files.createTempFile(prefix, suffix)
    p.toFile.deleteOnExit()
    use(p)
  }

  private def runProc(command: Vector[String]): String = {
    val out  = new StringBuilder
    val err  = new StringBuilder
    val code = Process(command) ! ProcessLogger(out.append(_).append('\n'), err.append(_).append('\n'))
    if (code != 0) sys.error(s"${command.head} failed (rc=$code): $err")
    out.toString
  }

  def probe(group: Config.Group, sourceRoot: Path): Probe = {
    val headerPaths = group.headers.map(h => sourceRoot.resolve(Path.of(h)).toString)
    val flags = Vector("-x", "c") ++
      group.includes.map(p => s"-I${sourceRoot.resolve(Path.of(p)).toString}") ++
      group.defines.map(d => s"-D$d")
    val macros   = dumpMacros(headerPaths, flags, sourceRoot)
    val versions = functionVersionTags(group, sourceRoot)

    println(s"  [clang -E -C] preprocessing ${headerPaths.mkString(", ")}")
    val rawPreproc = runPreprocessor(headerPaths, flags, Vector("-C"))
    val kept       = filteredLines(rawPreproc, sourceRoot.toString).map(_._2)
    // XXX Prepend stddef/stdint/stdbool so the filtered blob still parses on its own.
    val sourceText = "#include <stddef.h>\n#include <stdint.h>\n#include <stdbool.h>\n" + kept.mkString("\n") + "\n"

    withTempFile("ewgen-ast-blob-", ".c") { astBlob =>
      Files.writeString(astBlob, stripCommentsPreserveLines(sourceText), StandardCharsets.UTF_8)
      val (rawFns, tys) = dumpAst(astBlob, flags, group.symPrefixes)
      val view          = SourceView(sourceText, sourceText.linesIterator.toVector)
      val fns = rawFns.map(fn =>
        fn.copy(
          signature = sliceDeclaration(view, fn.span),
          docComment = leadingComment(view, fn.span.beginLine),
          versionTag = versions.get(fn.name)
        )
      )
      val slices = tys.map { td =>
        val comment = leadingComment(view, td.span.beginLine)
        val decl    = sliceDeclaration(view, td.span)
        if (comment.isEmpty) decl else s"$comment\n$decl"
      }
      Probe(fns, slices, macros)
    }
  }

  private def stripCommentsPreserveLines(s: String): String =
    Iterator
      .unfold((0, DeclScan.State())) { case (i, state) =>
        if (i >= s.length) None
        else {
          val c    = s.charAt(i)
          val next = if (i + 1 < s.length) s.charAt(i + 1) else 0.toChar
          val st   = state.step(c, next)
          Some(st.emit -> (i + st.advance, st.state))
        }
      }
      .mkString

  private object DeclScan {

    enum Mode { case Code, Block, Line, Str, Chr }

    case class State(mode: Mode = Mode.Code, escaped: Boolean = false, depth: Int = 0, lastNonWS: Char = 0.toChar) {
      def isComplete: Boolean = mode == Mode.Code && depth <= 0 && lastNonWS == ';'

      def step(c: Char, next: Char): Step = (mode, c, next) match {
        case (Mode.Block, '*', '/')                 => Step(copy(mode = Mode.Code), "  ", 2)
        case (Mode.Block, '\n', _)                  => Step(this, "\n", 1)
        case (Mode.Block, _, _)                     => Step(this, " ", 1)
        case (Mode.Line, '\n', _)                   => Step(copy(mode = Mode.Code), "\n", 1)
        case (Mode.Line, _, _)                      => Step(this, " ", 1)
        case (Mode.Str | Mode.Chr, _, _) if escaped => Step(copy(escaped = false), c.toString, 1)
        case (Mode.Str | Mode.Chr, '\\', _)         => Step(copy(escaped = true), c.toString, 1)
        case (Mode.Str, '"', _)                     => Step(copy(mode = Mode.Code), "\"", 1)
        case (Mode.Chr, '\'', _)                    => Step(copy(mode = Mode.Code), "'", 1)
        case (Mode.Str | Mode.Chr, _, _)            => Step(this, c.toString, 1)
        case (Mode.Code, '/', '*')                  => Step(copy(mode = Mode.Block), "  ", 2)
        case (Mode.Code, '/', '/')                  => Step(copy(mode = Mode.Line), "  ", 2)
        case (Mode.Code, '"', _)                    => Step(copy(mode = Mode.Str), "\"", 1)
        case (Mode.Code, '\'', _)                   => Step(copy(mode = Mode.Chr), "'", 1)
        case (Mode.Code, '{', _)                    => Step(copy(depth = depth + 1, lastNonWS = '{'), "{", 1)
        case (Mode.Code, '}', _)                    => Step(copy(depth = depth - 1, lastNonWS = '}'), "}", 1)
        case (Mode.Code, _, _) if c.isWhitespace    => Step(this, c.toString, 1)
        case (Mode.Code, _, _)                      => Step(copy(lastNonWS = c), c.toString, 1)
      }

      @tailrec final def scanString(input: String, i: Int = 0): State =
        if (i >= input.length) this
        else {
          val c = input.charAt(i)
          val n = if (i + 1 < input.length) input.charAt(i + 1) else 0.toChar
          val s = step(c, n)
          s.state.scanString(input, i + s.advance)
        }

      @tailrec final def scanUntilComplete(source: String, i: Int, limit: Int): Int =
        if (i >= limit || isComplete) i
        else {
          val c = source.charAt(i)
          val n = if (i + 1 < source.length) source.charAt(i + 1) else 0.toChar
          val s = step(c, n)
          s.state.scanUntilComplete(source, i + s.advance, limit)
        }
    }

    case class Step(state: State, emit: String, advance: Int)
  }

  private def runPreprocessor(headers: Vector[String], flags: Vector[String], extra: Vector[String]): String =
    withTempFile("ewgen-", ".c") { wrapper =>
      val body = headers.map(h => s"""#include "$h"""").mkString("\n") + "\n"
      Files.writeString(wrapper, body, StandardCharsets.UTF_8)
      runProc(Vector("clang", "-E") ++ extra ++ flags ++ Vector(wrapper.toString))
    }

  private def filteredLines(stdout: String, sourceRoot: String): Vector[(String, String)] = {
    val (_, kept) = stdout.linesIterator.foldLeft(("", Vector.empty[(String, String)])) {
      case ((_, kept), s"# ${_} \"$file\"${_}")              => (file, kept)
      case ((cur, kept), line) if cur.startsWith(sourceRoot) => (cur, kept :+ (cur, line))
      case (acc, _)                                          => acc
    }
    kept
  }

  private def functionVersionTags(group: Config.Group, sourceRoot: Path): Map[String, String] = group.versioning match {
    case None => Map.empty
    case Some(Config.VersioningStrategy.opencl) =>
      val tags = group.headers.iterator
        .map(h => sourceRoot.resolve(Path.of(h)))
        .filter(Files.exists(_))
        .flatMap(h => OpenClSuffixRe.findAllMatchIn(Files.readString(h, StandardCharsets.UTF_8)))
        .map(hit => hit.group(1) -> s"${hit.group(2)}${hit.group(3)}")
        .toMap
      println(s"  [versions] opencl: ${tags.size} tagged functions")
      tags
  }

  private def dumpMacros(headers: Vector[String], flags: Vector[String], sourceRoot: Path): Vector[MacroDef] = {
    println(s"  [clang -E -dD] ${headers.mkString(", ")}")
    val out = runPreprocessor(headers, flags, Vector("-dD"))
    filteredLines(out, sourceRoot.toString)
      .collect { case (_, DefineRe(name, tail)) =>
        MacroDef(name, s"$name$tail")
      }
      .distinctBy(_.name)
  }

  private def dumpAst(
      blob: Path,
      flags: Vector[String],
      prefixes: Vector[String]
  ): (Vector[FunctionDecl], Vector[TypeDecl]) = {
    println(s"  [clang -ast-dump=json] $blob")
    val out = runProc(Vector("clang", "-Xclang", "-ast-dump=json", "-fsyntax-only") ++ flags ++ Vector(blob.toString))
    val tu  = read[Node](out)
    val result = WalkState(prefixes = prefixes, sourceRoot = blob.toString).walk(tu)
    (result.fns.distinctBy(_.name), deduplicateTypeDecls(result.tys))
  }

  // XXX `typedef struct foo_s {...} foo_t;` shows up as RecordDecl and a TypedefDecl with nested ranges; keep only the outermost
  private def deduplicateTypeDecls(types: Vector[TypeDecl]): Vector[TypeDecl] = {
    def key(d: TypeDecl): Any = (d.span.beginOffset, d.span.endOffset, d.span.endTokLen) match {
      case (Some(b), Some(e), tok) => (d.sourceFile, b, e, tok)
      case _                       => (d.sourceFile, d.span.beginLine, d.span.endLine)
    }
    val distinct = types.distinctBy(key)
    val ranges   = distinct.flatMap(rangeOf)
    distinct.filterNot { d =>
      rangeOf(d).exists { case (b, e) =>
        ranges.exists { case (cb, ce) => cb <= b && e <= ce && (cb < b || e < ce) }
      }
    }
  }

  private def rangeOf(d: TypeDecl): Option[(Long, Long)] =
    for {
      begin <- d.span.beginOffset
      end   <- d.span.endOffset
    } yield (begin, end + d.span.endTokLen.getOrElse(1))

  private case class WalkState(
      fns: Vector[FunctionDecl] = Vector.empty,
      tys: Vector[TypeDecl] = Vector.empty,
      currentFile: String = "",
      currentLine: Int = 0,
      prefixes: Vector[String] = Vector.empty,
      sourceRoot: String = ""
  ) {
    def walk(node: Node): WalkState = {
      val locFile = node.loc.flatMap(_.file)
      val locLine = node.loc.flatMap(_.line)
      val begLine = node.range.flatMap(_.begin.line)
      val endLine = node.range.flatMap(_.end.line)
      val newFile = locFile.getOrElse(currentFile)
      val begin   = begLine.orElse(locLine).getOrElse(currentLine)
      val end     = endLine.getOrElse(begin)
      val next    = copy(currentFile = newFile, currentLine = end)

      val kind        = node.kind.getOrElse("")
      val name        = node.name.getOrElse("")
      val matched     = prefixes.exists(name.startsWith)
      val inSourceSet = newFile.nonEmpty && newFile.startsWith(sourceRoot)
      val span = SourceSpan(
        begin,
        end,
        node.range.flatMap(_.begin.offset),
        node.range.flatMap(_.end.offset),
        node.range.flatMap(_.end.tokLen)
      )
      val collected = kind match {
        case "FunctionDecl" if matched && inSourceSet && !node.isImplicit.getOrElse(false) =>
          val qt = node.`type`.map(_.qualType).getOrElse("")
          next.copy(fns = next.fns :+ FunctionDecl(name, qt, span, "", "", None))
        case "TypedefDecl" | "EnumDecl" | "RecordDecl" if inSourceSet && begin > 0 && end >= begin =>
          next.copy(tys = next.tys :+ TypeDecl(newFile, span))
        case _ => next
      }

      if (LeafDeclKinds(kind)) collected
      else node.inner.getOrElse(Vector.empty).foldLeft(collected)((a, c) => a.walk(c))
    }
  }

  private def sliceDeclaration(view: SourceView, span: SourceSpan): String =
    (span.beginOffset, span.endOffset) match {
      case (Some(b), Some(e)) => sliceDeclarationByOffset(view.text, b.toInt, e.toInt + span.endTokLen.getOrElse(1))
      case _                  => sliceDeclarationByLine(view.lines, span.beginLine, span.endLine)
    }

  private def sliceDeclarationByLine(src: Vector[String], beginLine: Int, endLine: Int): String = {
    val safeB = math.max(1, beginLine)
    val safeE = math.min(src.size, endLine)
    val base  = src.slice(safeB - 1, safeE).mkString("\n")
    val state = DeclScan.State().scanString(base)
    if (state.isComplete) base
    else {
      @tailrec def extend(i: Int, st: DeclScan.State, acc: Vector[String]): Vector[String] =
        if (i >= src.size || i >= safeE + 200 || st.isComplete) acc
        else {
          val seg = "\n" + src(i)
          extend(i + 1, st.scanString(seg), acc :+ src(i))
        }
      val extra = extend(safeE, state, Vector.empty)
      if (extra.isEmpty) base else base + "\n" + extra.mkString("\n")
    }
  }

  private def sliceDeclarationByOffset(source: String, start: Int, endExclusive: Int): String = {
    val safeStart = math.max(0, math.min(source.length, start))
    val safeEnd   = math.max(safeStart, math.min(source.length, endExclusive))
    val limit     = math.min(source.length, endExclusive + 10000)
    val initState = DeclScan.State().scanString(source.substring(safeStart, safeEnd))
    val end =
      if (initState.isComplete) safeEnd
      else initState.scanUntilComplete(source, safeEnd, limit)
    val lineEnd = source.indexOf('\n', end) match {
      case -1 => source.length
      case n  => n
    }
    val baseSlice  = source.substring(safeStart, end)
    val restOfLine = source.substring(end, lineEnd)
    if (restOfLine.contains("/*") || restOfLine.contains("//")) baseSlice + restOfLine
    else baseSlice
  }

  private def leadingComment(view: SourceView, beginLine: Int): String = {
    val src = view.lines

    def scanBack(from: Int, stop: Int => Boolean): Int = (from to 0 by -1).find(stop).getOrElse(-1)

    scanBack(math.min(src.size, beginLine - 1) - 1, j => src(j).trim.nonEmpty) match {
      case i if i < 0 => ""
      case i =>
        val trimmed = src(i).trim
        val start =
          if (trimmed.contains("*/")) scanBack(i, j => src(j).contains("/*"))
          else if (trimmed.startsWith("//")) scanBack(i - 1, j => !src(j).trim.startsWith("//")) + 1
          else -1
        if (start < 0) "" else src.slice(start, i + 1).mkString("\n")
    }
  }
}
