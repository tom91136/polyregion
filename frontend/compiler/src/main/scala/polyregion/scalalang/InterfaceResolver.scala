package polyregion.scalalang

import cats.syntax.all.*
import polyregion.ast.Traversal.*
import polyregion.ast.{MsgPack, PolyAST as p, *, given}
import polyregion.scalalang.generated.PolyASTWireSchema

import java.nio.file.{Files, Path, Paths}
import java.util.Locale
import scala.util.control.NonFatal

private[scalalang] object InterfaceResolver {

  private def configured(name: String, property: String): Option[String] =
    Option(System.getenv(name)).orElse(Option(System.getProperty(property))).map(_.trim).filter(_.nonEmpty)

  def configuredCapabilities: Set[String] =
    configured("POLYFRONT_LIBRARY_CAPABILITIES", "polyregion.library.capabilities")
      .fold(Set.empty[String])(_.split(',').iterator.map(_.trim).filter(_.nonEmpty).toSet)

  private def safePathComponent(value: String): Boolean = {
    val invalid = "\\/<>:\"|?*"
    val base    = value.takeWhile(_ != '.').toUpperCase(Locale.ROOT)
    value.nonEmpty && value != "." && value != ".." && !value.endsWith(".") && !value.endsWith(" ") &&
    value.forall(ch => ch >= ' ' && ch != 127 && !invalid.contains(ch)) &&
    !Set("CON", "PRN", "AUX", "NUL")(base) &&
    !(base.length == 4 && (base.startsWith("COM") || base.startsWith("LPT")) && base.last >= '1' && base.last <= '9')
  }

  private def decode[A: MsgPack.Codec](path: Path, expectedHash: String): Either[List[String], A] =
    try
      MsgPack
        .decode[MsgPack.Versioned[A]](Files.readAllBytes(path))
        .left
        .map(error => List(s"cannot decode package file `$path`: ${error.getMessage}"))
        .flatMap(versioned =>
          Either.cond(
            versioned.hash == expectedHash,
            versioned.t,
            List(s"package file `$path` has schema ${versioned.hash}, expected $expectedHash")
          )
        )
    catch {
      case NonFatal(error) => Left(List(s"cannot read package file `$path`: ${error.getMessage}"))
    }

  def loadPackage(packageName: String): Either[List[String], p.Package] = {
    if (!safePathComponent(packageName)) return Left(List(s"invalid package identity `$packageName`"))
    val roots = configured("POLYFRONT_LIBRARY_PATH", "polyregion.library.path").toList
      .flatMap(_.split(java.io.File.pathSeparator).toList)
      .filter(_.nonEmpty)
    val matches = roots
      .map(Paths.get(_).resolve(packageName).resolve("lib.polyast"))
      .filter(Files.isRegularFile(_))
    matches match {
      case Nil         => Left(List(s"no library package is available for interface `$packageName`"))
      case _ :: _ :: _ => Left(List(s"interface `$packageName` is ambiguous across ${matches.size} package roots"))
      case path :: Nil =>
        for {
          pack <- decode[p.Package](path, PolyASTWireSchema.PackageHash)
          _ <- Either.cond(
            pack.index.interface.name.fqn.mkString(".") == packageName,
            (),
            List(
              s"package identity differs: expected `$packageName`, got `${pack.index.interface.name.fqn.mkString(".")}`"
            )
          )
        } yield pack
    }
  }

  private def substitute(tpe: p.Type, bindings: Map[String, p.Type], bound: Set[String] = Set.empty): p.Type =
    tpe match {
      case p.Type.Var(name) if !bound(name)     => bindings.get(name).map(substitute(_, bindings, bound)).getOrElse(tpe)
      case p.Type.Struct(name, args)            => p.Type.Struct(name, args.map(substitute(_, bindings, bound)))
      case p.Type.Ptr(component, space)         => p.Type.Ptr(substitute(component, bindings, bound), space)
      case p.Type.Arr(component, length, space) => p.Type.Arr(substitute(component, bindings, bound), length, space)
      case p.Type.Exec(vars, args, rtn) =>
        val nested = bound ++ vars
        p.Type.Exec(vars, args.map(substitute(_, bindings, nested)), substitute(rtn, bindings, nested))
      case _ => tpe
    }

  private def closure(program: p.Program, root: p.Function): Either[List[String], List[p.Function]] = {
    @annotation.tailrec
    def loop(
        frontier: List[p.Function],
        reached: Set[p.Sym],
        out: List[p.Function]
    ): Either[List[String], List[p.Function]] =
      frontier match {
        case Nil                                        => Right(out.reverse)
        case function :: rest if reached(function.name) => loop(rest, reached, out)
        case function :: rest =>
          val names        = function.collectAll[p.Type].collect { case ref: p.Type.FnRef => ref.name }.distinct
          val dependencies = names.map(name => name -> program.functions.filter(_.name == name))
          val ambiguous = dependencies.collect {
            case (name, matches) if matches.size != 1 =>
              s"function `${name.repr}` has ${matches.size} package definitions"
          }
          if (ambiguous.nonEmpty) Left(ambiguous)
          else loop(dependencies.flatMap(_._2) ::: rest, reached + function.name, function :: out)
      }
    loop(List(root), Set.empty, Nil)
  }

  private def structClosure(program: p.Program, functions: List[p.Function]): Set[p.StructDef] = {
    @annotation.tailrec
    def loop(frontier: List[p.Sym], reached: Set[p.Sym], out: Set[p.StructDef]): Set[p.StructDef] = frontier match {
      case Nil                           => out
      case name :: rest if reached(name) => loop(rest, reached, out)
      case name :: rest =>
        val definitions = program.defs.filter(_.name == name)
        val next        = definitions.flatMap(_.collectAll[p.Type].collect { case p.Type.Struct(struct, _) => struct })
        loop(next ::: rest, reached + name, out ++ definitions)
    }
    val roots = functions.flatMap(_.collectAll[p.Type].collect { case p.Type.Struct(name, _) => name })
    loop(roots, Set.empty, Set.empty)
  }

  private def sizeOf(tpe: p.Type): Option[Int] = tpe match {
    case p.Type.Bool1 | p.Type.IntU8 | p.Type.IntS8     => Some(1)
    case p.Type.IntU16 | p.Type.IntS16 | p.Type.Float16 => Some(2)
    case p.Type.IntU32 | p.Type.IntS32 | p.Type.Float32 => Some(4)
    case p.Type.IntU64 | p.Type.IntS64 | p.Type.Float64 => Some(8)
    case p.Type.Unit0                                   => Some(0)
    case p.Type.Ptr(_, _) =>
      Some(System.getProperty("sun.arch.data.model", "64").toIntOption.filter(_ > 0).getOrElse(64) / 8)
    case _ => None
  }

  private def layoutsOf(tpe: p.Type): List[(p.Type, Int)] =
    sizeOf(tpe).map(tpe -> _).toList ::: (tpe match {
      case p.Type.Ptr(component, _) => layoutsOf(component)
      case _                        => Nil
    })

  def link(
      pack: p.Package,
      declaration: String,
      target: p.Expr.Invoke,
      callableDecls: List[p.FunctionDecl] = Nil,
      capabilities: Set[String] = configuredCapabilities,
      typeSizes: Map[p.Type, Int] = Map.empty
  ): Either[List[String], (List[p.Function], Set[p.StructDef])] = {
    val call = p.InvokeSignature(
      p.Sym(declaration),
      target.tpeArgs,
      None,
      target.args.map(_.tpe),
      target.rtn
    )
    val layouts = (target.args.map(_.tpe) :+ target.rtn).flatMap(layoutsOf).toMap ++ typeSizes
    for {
      resolution <- pack.index.resolve(call, callableDecls, capabilities, layouts)
      selected <- pack.program.functions.filter(_.decl == resolution.candidate.implementation) match {
        case function :: Nil => Right(function)
        case Nil => Left(List(s"selected implementation `${resolution.candidate.implementation.name.repr}` is absent"))
        case matches => Left(List(s"selected implementation is ambiguous: ${matches.size} matches"))
      }
      _ <- Either.cond(target.tpeArgs.isEmpty, (), List("generic Scala interface calls are not yet supported"))
      _ <- Either.cond(
        resolution.implementationBinding.result == InterfaceBinding.ResultBinding.Direct,
        (),
        List("Scala interface resolution does not support trailing-output implementations")
      )
      callableBindings <- resolution.implementationBinding.callables.toList
        .traverse { case (name, index) =>
          resolution.callBinding.callables
            .get(index)
            .toRight(List(s"callable binding `$name` has no call argument at index $index"))
            .map(name -> _)
        }
        .map(_.toMap)
      bindings = resolution.implementationBinding.types.view
        .mapValues(substitute(_, resolution.callBinding.types))
        .toMap ++
        callableBindings.view.mapValues(p.Type.FnRef(_))
      closed <- closure(pack.program, selected)
      functions = closed
        .map(_.modifyAll[p.Type](substitute(_, bindings)))
        .map(_.modifyAll[p.Expr] {
          case invoke: p.Expr.Invoke =>
            invoke.callee match {
              case p.Type.Var(name) =>
                callableBindings.get(name).fold(invoke)(symbol => invoke.copy(callee = p.Type.FnRef(symbol)))
              case _ => invoke
            }
          case expression => expression
        })
      renamed = functions.map(_.modifyAll[p.Type] {
        case p.Type.FnRef(name) if name == selected.name => p.Type.FnRef(target.calleeName)
        case tpe                                         => tpe
      })
      linked = renamed.map { function =>
        if (function.name != selected.name) function
        else
          function.copy(decl =
            function.decl.copy(
              name = target.calleeName,
              tpeVars = Nil,
              receiver = target.receiver.map(value => p.Arg(p.Named("this", value.tpe), None))
            )
          )
      }
    } yield linked -> structClosure(pack.program, linked)
  }
}
