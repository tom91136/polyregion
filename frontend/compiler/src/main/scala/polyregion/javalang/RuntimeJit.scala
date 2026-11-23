package polyregion.javalang

import cats.syntax.all.*
import polyregion.ast.pass.*
import polyregion.ast.{PolyAST as p, *}
import polyregion.javalang.BytecodeRemapper.symbolOf
import polyregion.javalang.LaunchCore.{EnvField, Identity, Mode, PrimOfType, Split, Twinned}
import polyregion.jvm.runtime.PolyregionRuntimeException
import polyregion.scalalang.LaunchConfig

import java.lang.invoke.{MethodHandleInfo, SerializedLambda}

object RuntimeJit {

  private val CoreIdentity = LaunchCore.CoreIdentity

  def config: Either[String, LaunchConfig] = {
    def value(key: String, env: String): Option[String] =
      sys.props.get(key).orElse(sys.env.get(env)).map(_.trim).filter(_.nonEmpty)
    def listed(key: String, env: String): List[String] =
      value(key, env).toList.flatMap(_.split(',')).map(_.trim).filter(_.nonEmpty)
    LaunchConfig.fromSettings(
      listed(LaunchConfig.TargetsKey, "POLYREGION_TARGETS").map(LaunchConfig.TargetsKey + "=" + _) ++
        value(LaunchConfig.OptKey, "POLYREGION_OPT").map(LaunchConfig.OptKey + "=" + _) ++
        value(LaunchConfig.RootsKey, "POLYFRONT_LIBRARY_PATH").map(LaunchConfig.RootsKey + "=" + _) ++
        listed(LaunchConfig.CapabilitiesKey, "POLYFRONT_LIBRARY_CAPABILITIES").map(
          LaunchConfig.CapabilitiesKey + "=" + _
        )
    )
  }

  private def classBytes(loader: ClassLoader): BytecodeResolver.ClassBytes = {
    val own = BytecodeResolver.loaderClasses(loader)
    owner => own(owner).orElse(BytecodeResolver.runtimeClasses(owner))
  }

  private def captureKind(tpe: p.Type): Option[Char] = tpe match {
    case p.Type.Ptr(component, _) if PrimOfType.contains(component) => Some('a')
    case other                                                      => PrimOfType.get(other).map(_.code)
  }

  def compile(lambda: SerializedLambda, loader: ClassLoader): Result[(String, Array[Byte])] = {
    val owner      = lambda.getImplClass.replace('/', '.')
    val name       = lambda.getImplMethodName
    val entryName  = LaunchCore.normalise(s"${owner}_$name")
    val fqcn       = s"${LaunchSite.PACKAGE}.$entryName"
    val symbol     = s"${entryName}_callable0"
    val describe   = s"`$owner.$name`"
    val bytesOf    = classBytes(loader)
    val parameters = lambda.getCapturedArgCount
    for {
      loaded <- config.leftMap(CompilerException(_))
      _      <- if (loaded.roots.nonEmpty) ().success else "polyregion.library.path is not set".fail
      hostOnly = loaded.copy(targets = (LaunchConfig.Default.targets ::: loaded.targets).distinct)
      _ <-
        if (lambda.getImplMethodKind == MethodHandleInfo.REF_invokeStatic) ().success
        else
          s"$describe is not static, so it uses `this`; a lambda lowered at run time cannot capture its receiver".fail
      root = MethodRef(owner, name, lambda.getImplMethodSignature)
      resolved <- BytecodeResolver.resolve(List(root), bytesOf)
      entryFn <- resolved.functions
        .find(_.name == symbolOf(owner, name))
        .failIfEmpty(s"$describe was not lowered")
      _ <-
        if (entryFn.args.sizeIs == parameters) ().success
        else s"$describe takes ${entryFn.args.size} parameters but captures $parameters".fail
      resultCode <- entryFn.rtn match {
        case p.Type.Unit0 => 'V'.success
        case other =>
          PrimOfType
            .get(other)
            .map(_.code)
            .failIfEmpty(
              other match {
                case p.Type.Ptr(_, _) => s"result of type `${other.repr}` escapes the region"
                case _                => s"result of type `${other.repr}` cannot be returned from the region"
              }
            )
      }
      origins = entryFn.args.zipWithIndex.map((arg, i) => arg.named.symbol -> i).toMap
      depFns  = resolved.functions.filterNot(_.decl == entryFn.decl)
      tags    = Devirtualise.tagsOf(resolved.dispatch.values.flatten.map(_.cls))
      candidates = resolved.dispatch.view.mapValues { arms =>
        arms.map(a => Devirtualise.Candidate(p.Type.Struct(a.cls, Nil), a.impl, tags(a.cls), Some(a.args)))
      }.toMap
      raw      = p.Program(Some(entryFn), depFns, resolved.structs.distinctBy(_.name))
      treeTags = tags.map((cls, tag) => cls.fqn.mkString(".") -> tag)
      captured = entryFn.copy(decl = entryFn.decl.copy(args = Nil, termCaptures = entryFn.args))
      routed   = Offloader.routeReturnThroughCapture(captured)
      optimised <- Offloader.optimise(
        treeTags,
        candidates,
        resolved.owners,
        raw,
        routed,
        symbol,
        PluginEntry.defaultLog
      )
      (_, optimisedProgram) = optimised
      unique                = LaunchCore.withUniqueNames(optimisedProgram.entry.toList ::: optimisedProgram.functions)
      program = optimisedProgram.copy(
        entry = unique.headOption.filter(_ => optimisedProgram.entry.isDefined),
        functions = unique.drop(optimisedProgram.entry.toList.size)
      )
      entry <- program.entry.failIfEmpty(s"$describe lost its entry")
      (results, plain) = entry.decl.termCaptures.partition(_.named.symbol == Offloader.ResultCapture)
      envFields <- plain.traverse { arg =>
        for {
          index <- origins.get(arg.named.symbol).failIfEmpty(s"$describe parameter `${arg.named.symbol}` has no source")
          kind <- captureKind(arg.named.tpe).failIfEmpty(
            s"$describe capture ${index + 1} of type `${arg.named.tpe.repr}` is not supported at run time; " +
              "only primitives and primitive arrays are"
          )
        } yield EnvField(arg.named, kind, index.toString)
      }
      callable <- LaunchCore.callableOf(
        hostOnly,
        entryName,
        0,
        symbol,
        Mode.Region,
        entry,
        program,
        resultCode,
        Twinned(Nil, Nil),
        Split(envFields, Nil),
        results
      )
      identity <- Identity.parse(CoreIdentity)
      assembled <- LaunchCore.assembleParts(
        hostOnly,
        fqcn,
        identity,
        p.Type.Unit0,
        'V',
        List(p.Type.IntS64, LaunchCore.callableType(callable)),
        List(p.Program.TypeSize(p.Type.IntS64, 8)),
        List("LJ1", LaunchCore.specOf(callable.environment)),
        List(callable),
        callable.result
      )
    } yield (assembled.descriptor, assembled.binary)
  }
}

// lowers a lambda that has no generated launch site, from the class file of its implementation method
class RuntimeJit extends RuntimeLowering {

  override def configuration(): String =
    RuntimeJit.config.fold(identity, c => s"${c.targetList}|${c.opt}|${c.roots}|${c.capabilityList}")

  override def lower(lambda: SerializedLambda, loader: ClassLoader): RuntimeLowering.Compiled =
    RuntimeJit
      .compile(lambda, loader)
      .fold(
        e => throw PolyregionRuntimeException(e.getMessage),
        (descriptor, binary) => RuntimeLowering.Compiled(descriptor, binary)
      )
}
