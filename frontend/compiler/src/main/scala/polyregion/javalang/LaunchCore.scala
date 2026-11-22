package polyregion.javalang

import cats.syntax.all.*
import polyregion.ast.generated.PolyPackageWireSchema
import polyregion.ast.Traversal.*
import polyregion.ast.{CodeGen, MsgPack, PolyAST as p, given, *}
import polyregion.intrinsic.IntrinsicTable
import polyregion.jvm.compiler as ct
import polyregion.jvm.twin.Layout
import polyregion.scalalang.LaunchConfig

import java.io.{ByteArrayOutputStream, DataOutputStream}
import javax.lang.model.`type`.TypeKind
import scala.jdk.CollectionConverters.*

// everything a launch needs once its callables are lowered; nothing here may touch javac internals, the run-time path loads it too
object LaunchCore {

  val CoreIdentity: String = "polyregion.core:polyregion.core.for_each"

  case class Prim(code: Char, tpe: p.Type, size: Int)

  val Prims: Map[TypeKind, Prim] = Map(
    TypeKind.INT     -> Prim('I', p.Type.IntS32, 4),
    TypeKind.LONG    -> Prim('J', p.Type.IntS64, 8),
    TypeKind.FLOAT   -> Prim('F', p.Type.Float32, 4),
    TypeKind.DOUBLE  -> Prim('D', p.Type.Float64, 8),
    TypeKind.SHORT   -> Prim('S', p.Type.IntS16, 2),
    TypeKind.BYTE    -> Prim('B', p.Type.IntS8, 1),
    TypeKind.CHAR    -> Prim('C', p.Type.IntU16, 2),
    TypeKind.BOOLEAN -> Prim('Z', p.Type.Bool1, 1)
  )

  val PrimOfType: Map[p.Type, Prim] = Prims.values.map(prim => prim.tpe -> prim).toMap

  def normalise(s: String): String = s.map(c => if (c.isLetterOrDigit || c == '_') c else '_')

  class Inexpressible(message: String) extends Exception(message)

  def inexpressible(message: String): Result[Nothing] = Left(Inexpressible(message))

  enum Mode {
    case Site
    case Region
  }

  case class Callable(
      function: p.Function,
      dependencies: List[p.Function],
      environment: Option[Environment],
      defs: List[p.StructDef],
      result: Char,
      layouts: List[Layout],
      twins: List[TwinEmitter.Emitted]
  )

  case class Field(offset: Int, kind: Char, source: String)

  object Identity {
    def parse(text: String): Result[Identity] = text.split(":", 2) match {
      case Array(packageName, declaration) => Identity(packageName, declaration).success
      case _                               => s"malformed launch identity `$text`".fail
    }
  }

  case class Identity(packageName: String, declaration: String)

  case class Twinned(layouts: List[Layout], twins: List[TwinEmitter.Emitted])

  case class EnvField(named: p.Named, kind: Char, source: String)

  case class Param(arg: p.Arg, index: Int)

  case class Split(envFields: List[EnvField], params: List[Param])

  case class Assembled(descriptor: String, binary: Array[Byte], twins: List[TwinEmitter.Emitted])

  case class Environment(definition: p.StructDef, size: Int, fields: List[Field])

  private def layoutOf(
      definition: p.StructDef,
      others: List[p.StructDef],
      config: LaunchConfig
  ): Result[ct.Layout] = {
    val compiler = ct.Compiler.create()
    try
      for {
        encoded <- Either.catchNonFatal(MsgPack.encode(CodeGen.structDefsVersioned(others :+ definition)))
        layouts <- (LaunchConfig.Default.targets ::: config.targets).distinct.traverse { target =>
          Either.catchNonFatal(
            s"${target.name}@${target.arch}" ->
              compiler
                .layoutOf(encoded, ct.Options.of(compiler.targetByName(target.name), target.arch))
                .find(_.name == definition.name.repr)
                .get
          )
        }
        shape = (layout: ct.Layout) =>
          (
            layout.sizeInBytes,
            layout.alignment,
            layout.members.map(m => (m.name, m.offsetInBytes, m.sizeInBytes)).toList
          )
        agreed <-
          if (layouts.map(entry => shape(entry._2)).distinct.size == 1) layouts.head._2.success
          else
            s"the layout of `${definition.name.repr}` differs between the targets ${layouts.map(_._1).mkString(", ")}".fail
      } yield agreed
    finally compiler.close()
  }

  private def stored(named: p.Named): p.Named = named.tpe match {
    case p.Type.Ptr(referent @ p.Type.Struct(_, _), p.Type.Space.Private) =>
      named.copy(tpe = p.Type.Ptr(referent, p.Type.Space.Global))
    case _ => named
  }

  def withUniqueNames(functions: List[p.Function]): List[p.Function] = {
    val overloaded = functions.groupBy(_.name).filter(_._2.map(_.decl).distinct.sizeIs > 1)
    if (overloaded.isEmpty) functions
    else {
      val renamed = overloaded.map { (name, group) =>
        name -> group.distinctBy(_.decl).zipWithIndex.map((fn, i) => fn.decl -> (name :+ s"$$$i"))
      }
      def choose(name: p.Sym, args: List[p.Type], rtn: p.Type): p.Sym = {
        val candidates = overloaded(name).distinctBy(_.decl)
        val arity      = args.size
        val found = candidates
          .find(fn => fn.args.map(_.named.tpe) == args && fn.rtn == rtn)
          .orElse(candidates.find(fn => fn.args.sizeIs == arity && fn.rtn == rtn))
          .getOrElse(candidates.head)
        renamed(name).find(_._1 == found.decl).get._2
      }
      functions.map { fn =>
        val rewritten = fn.modifyAll[p.Expr] {
          case invoke @ p.Expr.Invoke(p.Type.FnRef(name), _, _, args, rtn) if overloaded.contains(name) =>
            invoke.copy(callee = p.Type.FnRef(choose(name, args.map(_.tpe), rtn)))
          case other => other
        }
        if (overloaded.contains(fn.name))
          rewritten.modifyDecl(_.copy(name = renamed(fn.name).find(_._1 == fn.decl).get._2))
        else rewritten
      }
    }
  }

  def callableOf(
      config: LaunchConfig,
      entryName: String,
      index: Int,
      symbol: String,
      mode: Mode,
      entry: p.Function,
      program: p.Program,
      resultCode: Char,
      twinned: Twinned,
      split: Split,
      results: List[p.Arg]
  ): Result[Callable] =
    for {
      _ <- ().success
      envFields = split.envFields.map(e => (e.named, e.kind, e.source))
      paramArgs = split.params.map(q => (q.arg, q.index))
      receiver  = entry.decl.receiver.map(r => (r.named, 'a', "0")).toList
      fields    = receiver ++ envFields ++ results.map(arg => (arg.named, 'r', "0"))
      environment <-
        if (fields.isEmpty) None.success
        else {
          val definition = p.StructDef(p.Sym(s"${entryName}_env$index"), Nil, fields.map(f => stored(f._1)), Nil)
          layoutOf(definition, program.defs, config).map { layout =>
            val offsets = layout.members.map(m => m.name -> m.offsetInBytes.toInt).toMap
            Some(
              Environment(
                definition,
                layout.sizeInBytes.toInt,
                fields.map((named, kind, i) => Field(offsets(named.symbol), kind, i))
              )
            )
          }
        }
      self = environment.map(e =>
        p.Named("#env", p.Type.Ptr(p.Type.Struct(e.definition.name, Nil), p.Type.Space.Global))
      )
      loads = self.toList.flatMap { s =>
        fields.map { (named, _, _) =>
          val steps = List(p.PathStep.Deref, p.PathStep.Field(named.symbol))
          val field = p.Term.Select(s, steps, stored(named).tpe)
          p.Stmt.Var(
            named,
            Some(named.tpe match {
              case p.Type.Ptr(referent @ p.Type.Struct(_, _), p.Type.Space.Private) =>
                p.Expr.RefTo(
                  p.Term.Select(s, steps :+ p.PathStep.Deref, referent),
                  None,
                  referent,
                  p.Type.Space.Private,
                  p.Region.Rooted(s)
                )
              case _ => p.Expr.Alias(field)
            })
          )
        }
      }
    } yield Callable(
      entry.copy(
        decl = entry.decl.copy(
          name = p.Sym(symbol),
          receiver = self.map(p.Arg(_)),
          args = mode match {
            case Mode.Site   => paramArgs.sortBy(_._2).map(_._1)
            case Mode.Region => List(p.Arg(p.Named("#i", p.Type.IntS64)))
          },
          termCaptures = Nil,
          affinity = p.Function.Affinity.Host
        ),
        body = loads ::: entry.body,
        visibility = p.Function.Visibility.Internal,
        convention = p.CallConvention.RegularCall
      ),
      program.functions,
      environment,
      program.defs,
      resultCode,
      twinned.layouts,
      twinned.twins
    )

  def importRoot(entryName: String, declaration: p.Sym, args: List[p.Type], rtn: p.Type): p.Function =
    p.Function(
      p.FunctionDecl(
        p.Sym(entryName),
        Nil,
        None,
        args.zipWithIndex.map((tpe, index) => p.Arg(p.Named(s"arg$index", tpe))),
        Nil,
        Nil,
        rtn,
        p.Function.Affinity.Host
      ),
      Nil,
      p.Function.Visibility.Exported,
      p.Function.FpMode.Relaxed,
      p.CallConvention.RegularCall,
      Some(declaration)
    )

  def compileBundles(
      identity: Identity,
      entryName: String,
      request: Array[Byte],
      config: LaunchConfig
  ): Result[List[(String, p.Compile.Bundle)]] = {
    val compiler = ct.Compiler.create()
    try
      for {
        raw <- Either.catchNonFatal(
          compiler.compileInterface(
            identity.packageName,
            config.roots,
            identity.declaration,
            entryName,
            request,
            config.capabilityList,
            config.targetList,
            config.opt,
            0
          )
        )
        bundles <- raw.toList.traverse(bytes => MsgPack.decode[MsgPack.Versioned[p.Compile.Bundle]](bytes).map(_.t))
        usable = bundles.filter(_.unfit.isEmpty)
        _ <-
          if (usable.isEmpty)
            s"`${identity.declaration}` cannot be linked for the configured targets: ${bundles.flatMap(_.unfit).distinct.mkString("; ")}".fail
          else ().success
      } yield usable.zipWithIndex.map((bundle, group) => (s"$entryName@g$group", bundle))
    finally compiler.close()
  }

  def encodeGroups(groups: List[(String, p.Compile.Bundle)]): Array[Byte] = {
    val bytes = ByteArrayOutputStream()
    val out   = DataOutputStream(bytes)
    out.writeInt(groups.size)
    groups.foreach { (program, bundle) =>
      out.writeUTF(program)
      out.writeInt(bundle.hostObject.size)
      out.write(bundle.hostObject.toArray)
      out.writeInt(bundle.remoteModules.size)
      bundle.remoteModules.foreach { module =>
        out.writeUTF(module.moduleName)
        out.writeByte(module.target)
        out.writeInt(module.features.size)
        module.features.foreach(out.writeUTF)
        out.writeInt(module.image.size)
        out.write(module.image.toArray)
      }
    }
    out.flush()
    bytes.toByteArray
  }

  def specOf(environment: Option[Environment]): String =
    environment.fold("C0:") { e =>
      s"C${e.size}:${e.fields.map(f => s"${f.offset}${f.kind}${f.source}").mkString(",")}"
    }

  def callableType(callable: Callable): p.Type = callable.environment match {
    case None              => p.Type.FnRef(callable.function.name)
    case Some(environment) => p.Type.Struct(environment.definition.name, Nil)
  }

  def assemble(
      config: LaunchConfig,
      fqcn: String,
      identity: Identity,
      rtn: p.Type,
      rtnCode: Char,
      argumentTypes: List[p.Type],
      argumentSizes: List[p.Program.TypeSize],
      specs: List[String],
      callables: List[Callable],
      regionResult: Char,
      parent: Class[?]
  ): Result[Offloader.Executable] =
    assembleParts(config, fqcn, identity, rtn, rtnCode, argumentTypes, argumentSizes, specs, callables, regionResult)
      .flatMap(assembled =>
        Offloader
          .emitClass(fqcn, assembled.binary, assembled.descriptor, parent)
          .map(Offloader.Executable(fqcn, _, assembled.twins))
      )

  def assembleParts(
      config: LaunchConfig,
      fqcn: String,
      identity: Identity,
      rtn: p.Type,
      rtnCode: Char,
      argumentTypes: List[p.Type],
      argumentSizes: List[p.Program.TypeSize],
      specs: List[String],
      callables: List[Callable],
      regionResult: Char
  ): Result[Assembled] = {
    val entryName    = normalise(fqcn.substring(fqcn.lastIndexOf('.') + 1))
    val environments = callables.flatMap(_.environment)
    val typeSizes = environments.map(e => p.Program.TypeSize(p.Type.Struct(e.definition.name, Nil), e.size)) ++
      argumentSizes ++ PrimOfType.get(rtn).map(prim => p.Program.TypeSize(prim.tpe, prim.size))
    val functions = IntrinsicTable.dedupe(
      importRoot(entryName, p.Sym(identity.declaration), argumentTypes, rtn) :: callables.flatMap(c =>
        c.function :: c.dependencies
      )
    )
    val defs    = (environments.map(_.definition) ++ callables.flatMap(_.defs)).distinct
    val request = p.Program.LinkRequest(Nil, p.Program(None, functions, defs), Nil, typeSizes.distinct)
    for {
      encoded <- Either.catchNonFatal(MsgPack.encode(MsgPack.Versioned(PolyPackageWireSchema.Hash, request)))
      groups  <- compileBundles(identity, entryName, encoded, config)
      twins   = callables.flatMap(_.twins)
      layouts = Layout.encodeAll(callables.flatMap(_.layouts).asJava)
      descriptor =
        s"$entryName|$rtnCode|${specs.mkString(";")}|$regionResult|$layouts|${twins.map(_.fqcn).mkString(",")}"
    } yield Assembled(descriptor, encodeGroups(groups), twins)
  }
}
