package polyregion.javalang

import cats.syntax.all.*
import net.bytebuddy.jar.asm.Opcodes
import org.jetbrains.java.decompiler.struct.gen.MethodDescriptor
import polyregion.ast.{PolyAST as p, *}
import polyregion.intrinsic.IntrinsicTable
import polyregion.javalang.BytecodeRemapper.symbolOf

import scala.annotation.tailrec

// the fallback resolver: a callee with no source tree is lowered from its class file, together with the static methods it calls
object BytecodeResolver {

  type ClassBytes = ClassLayouts.ClassBytes

  case class Resolved(
      functions: List[p.Function],
      structs: List[p.StructDef],
      sites: List[OpenSite],
      dispatch: Map[p.Sym, List[Arm]],
      owners: Map[p.Sym, List[p.Sym]]
  )

  private case class Settled(site: OpenSite, owner: p.Sym, arms: List[Arm], impls: List[MethodRef])

  @tailrec def lowerable(t: p.Type): Boolean = t match {
    case p.Type.Ptr(component, _) => lowerable(component)
    case p.Type.Bool1 | p.Type.IntS8 | p.Type.IntS16 | p.Type.IntU16 | p.Type.IntS32 | p.Type.IntS64 | p.Type.Float32 |
        p.Type.Float64 | p.Type.Unit0 =>
      true
    case _ => false
  }

  def classFile(owner: String): String = s"${owner.replace('.', '/')}.class"

  // the running JDK answers for its own classes, which `--release` would otherwise hide behind stripped signatures
  def runtimeClasses: ClassBytes = owner =>
    Option(ClassLoader.getSystemResourceAsStream(classFile(owner))).map { in =>
      try in.readAllBytes()
      finally in.close()
    }

  def loaderClasses(loaders: ClassLoader*): ClassBytes = owner =>
    loaders.view.flatMap(l => Option(l.getResourceAsStream(classFile(owner)))).headOption.map { in =>
      try in.readAllBytes()
      finally in.close()
    }

  // a nested class is named with dots by the Scala frontend and with a dollar by its class file
  def locate(dotted: String, bytesOf: ClassBytes): Result[String] = {
    val parts = dotted.split('.').toList
    val candidates =
      parts.indices.map(k => (parts.dropRight(k).mkString(".") +: parts.takeRight(k)).mkString("$"))
    candidates.find(bytesOf(_).isDefined).failIfEmpty(s"No class file for `$dotted`")
  }

  // the Scala frontend names a static method after its Scala symbol, which a nested call from the bytecode does not share
  def callableAs(plain: p.Function, name: p.Sym, receiver: Option[p.Type]): p.Function = {
    val args = plain.args.map(a => p.Term.Select(a.named, Nil, a.named.tpe))
    val call = p.Expr.Invoke(p.Type.FnRef(plain.name), Nil, None, args, plain.rtn)
    val tail =
      if (plain.rtn == p.Type.Unit0)
        List(p.Stmt.Var(p.Named("#unit", p.Type.Unit0), Some(call)), p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const)))
      else {
        val result = p.Named("#result", plain.rtn)
        List(p.Stmt.Var(result, Some(call)), p.Stmt.Return(p.Expr.Alias(p.Term.Select(result, Nil, plain.rtn))))
      }
    plain.copy(
      decl = plain.decl.copy(name = name, receiver = receiver.map(t => p.Arg(p.Named("this", t)))),
      body = tail
    )
  }

  // a covered owner's method, and what it reaches, is refused under the intrinsic's name
  private def lowerRef(ref: MethodRef, origin: MethodRef, bytesOf: ClassBytes): Result[BytecodeRemapper.Lowered] =
    bytesOf(ref.owner)
      .failIfEmpty(s"No class file for `${ref.owner}` while lowering `${ref.repr}`")
      .flatMap(BytecodeRemapper.lowerWithCalls(_, ref.name, ref.desc, bytesOf, ref.instance))
      .leftMap { e =>
        val cause = s"Cannot lower `${ref.repr}` from its class file: ${e.getMessage}"
        CompilerException(
          List(ref, origin)
            .find(Natives.covered(_))
            .fold(cause)(r => s"unsupported JVM intrinsic ${Natives.keyOf(r).repr}: $cause"),
          e
        )
      }

  // roots come first and each callee follows the root that reached it
  private def walk(
      todo: List[(MethodRef, MethodRef)],
      seen: Set[MethodRef],
      done: List[BytecodeRemapper.Lowered],
      bytesOf: ClassBytes
  ): Result[(Set[MethodRef], List[BytecodeRemapper.Lowered])] = {
    @tailrec def go(
        todo: List[(MethodRef, MethodRef)],
        seen: Set[MethodRef],
        acc: List[BytecodeRemapper.Lowered]
    ): Result[(Set[MethodRef], List[BytecodeRemapper.Lowered])] = todo match {
      case Nil                           => (seen, acc).success
      case (ref, _) :: rest if seen(ref) => go(rest, seen, acc)
      case (ref, origin) :: rest =>
        lowerRef(ref, origin, bytesOf) match {
          case Left(e)        => Left(e)
          case Right(lowered) => go(rest ::: lowered.calls.map(_ -> origin), seen + ref, lowered :: acc)
        }
    }
    go(todo, seen, done)
  }

  private def stateless(site: OpenSite, bytesOf: ClassBytes): Result[Unit] =
    ClassLayouts.carriedState(site.owner, bytesOf).flatMap {
      case None => ().success
      case Some(c) =>
        (s"Cannot devirtualise `${site.repr}`: `$c` carries instance state, which the layout shared by its " +
          "implementations has no room for").fail
    }

  // the classes the region builds that a receiver of the site's class can be, with the method each declares
  private def classFileArms(
      site: OpenSite,
      allocated: List[String],
      shadowed: Set[p.Sym],
      bytesOf: ClassBytes
  ): Result[List[(Arm, MethodRef)]] =
    allocated.distinct.sorted
      .filterNot(c => ClassLayouts.isPlatform(c.replace('.', '/')))
      .filter(c => c == site.owner || ClassLayouts.ancestors(c, bytesOf).contains(site.owner))
      .flatTraverse { cls =>
        (ClassLayouts.isConcrete(cls, bytesOf), ClassLayouts.structName(cls, bytesOf)).tupled.flatMap {
          (concrete, name) =>
            val impl = MethodRef(cls, site.name, site.desc, instance = true)
            if (!concrete || shadowed(p.Sym(name))) List.empty[(Arm, MethodRef)].success
            else
              for {
                access <- ClassLayouts.accessOf(impl, bytesOf)
                _ <-
                  if (access._2.exists(m => (m & Opcodes.ACC_ABSTRACT) == 0)) ().success
                  else
                    (s"`$cls` is a candidate receiver of `${site.repr}` but inherits its implementation rather " +
                      "than declaring one, which this lowering cannot dispatch to").fail
              } yield List(Arm(p.Sym(name), symbolOf(cls, site.name), List.range(0, site.arity)) -> impl)
        }
      }

  // every open site gets the classes the compilation declares (`sourced`) and the ones the region builds; a site
  // nothing implements is refused, so a call never reaches the pass with no arm to dispatch to
  private def dispatchOf(
      sites: List[OpenSite],
      allocated: List[String],
      bytesOf: ClassBytes,
      sourced: OpenSite => Result[List[Arm]]
  ): Result[List[Settled]] =
    sites.traverse { site =>
      for {
        _     <- stateless(site, bytesOf)
        owner <- ClassLayouts.structName(site.owner, bytesOf)
        own   <- sourced(site)
        built <- classFileArms(site, allocated, own.map(_.cls).toSet, bytesOf)
        arms = (own ++ built.map(_._1)).distinctBy(_.cls).sortBy(_.cls.repr)
        _ <-
          if (arms.nonEmpty) ().success
          else
            (s"Cannot devirtualise `${site.repr}`: no class the compilation declares or the region builds " +
              "implements it, so there is nothing to compile into the kernel").fail
      } yield Settled(site, p.Sym(owner), arms, built.map(_._2))
    }

  private def resolvedOf(
      lowered: List[BytecodeRemapper.Lowered],
      settled: List[Settled] = Nil
  ): Resolved =
    Resolved(
      IntrinsicTable.dedupe(lowered.flatMap(l => l.fn :: l.helpers)),
      lowered.flatMap(_.structs).distinctBy(_.name),
      lowered.flatMap(_.sites).distinct,
      settled.map(s => s.site.callee -> s.arms).toMap,
      settled.groupMapReduce(_.owner)(_.arms.map(_.cls))(_ ++ _).view.mapValues(_.distinct).toMap
    )

  // the open calls of a closure are settled against the classes it builds, and the implementations that adds can
  // build more, so lowering runs until no new method appears
  def resolve(
      roots: List[MethodRef],
      bytesOf: ClassBytes,
      sourced: OpenSite => Result[List[Arm]] = _ => Nil.success
  ): Result[Resolved] = {
    @tailrec def go(
        todo: List[(MethodRef, MethodRef)],
        seen: Set[MethodRef],
        done: List[BytecodeRemapper.Lowered]
    ): Result[Resolved] =
      walk(todo, seen, done, bytesOf) match {
        case Left(e) => Left(e)
        case Right((seen1, lowered)) =>
          val ordered = lowered.reverse
          dispatchOf(ordered.flatMap(_.sites).distinct, ordered.flatMap(_.allocated), bytesOf, sourced) match {
            case Left(e) => Left(e)
            case Right(dispatched) =>
              val more = dispatched.flatMap { s =>
                val origin = MethodRef(s.site.owner, s.site.name, s.site.desc)
                s.impls.map(_ -> origin)
              }
              if (more.forall((ref, _) => seen1(ref))) Right(resolvedOf(ordered, dispatched))
              else go(more, seen1, lowered)
          }
      }
    go(roots.map(r => r -> r), Set.empty, Nil)
  }

  // the root's own function is returned apart from the closure it heads
  def resolveRoot(root: MethodRef, bytesOf: ClassBytes): Result[(p.Function, Resolved)] = for {
    head <- lowerRef(root, root, bytesOf)
    rest <- walk(head.calls.map(_ -> root), Set(root), List(head), bytesOf)
  } yield head.fn -> resolvedOf(rest._2.reverse)

  // the Scala frontend knows a callee by its lowered signature rather than by a JVM descriptor
  def descriptorFor(
      owner: String,
      name: String,
      params: List[p.Type],
      rtn: p.Type,
      bytesOf: ClassBytes
  ): Result[MethodRef] =
    for {
      bytes <- bytesOf(owner).failIfEmpty(s"No class file for `$owner` while resolving `$name`")
      all   <- Vineflower.staticDescriptors(bytes, name)
      matching = all.filter { d =>
        val md = MethodDescriptor.parseDescriptor(d)
        md.params.toList.traverse(ClassLayouts.typeOf(_)) == Right(params) &&
        ClassLayouts.typeOf(md.ret) == Right(rtn)
      }
      desc <- matching match {
        case d :: Nil => d.success
        case Nil =>
          (s"`$owner` declares no static `$name` taking (${params.map(_.repr).mkString(", ")}) and returning " +
            s"${rtn.repr}; its static overloads are ${all.mkString(", ")}").fail
        case many => s"`$owner.$name` is ambiguous for ${many.mkString(", ")}".fail
      }
    } yield MethodRef(owner, name, desc)
}
