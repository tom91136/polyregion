package polyregion.javalang

import cats.syntax.all.*
import org.jetbrains.java.decompiler.struct.gen.MethodDescriptor
import polyregion.ast.{PolyAST as p, *}
import polyregion.intrinsic.IntrinsicTable

import scala.annotation.tailrec

// the fallback resolver: a callee with no source tree is lowered from its class file, together with the static methods it calls
object BytecodeResolver {

  type ClassBytes = String => Option[Array[Byte]]

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
      .flatMap(BytecodeRemapper.lowerWithCalls(_, ref.name, ref.desc))
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
      bytesOf: ClassBytes
  ): Result[List[BytecodeRemapper.Lowered]] = {
    @tailrec def go(
        todo: List[(MethodRef, MethodRef)],
        seen: Set[MethodRef],
        acc: List[BytecodeRemapper.Lowered]
    ): Result[List[BytecodeRemapper.Lowered]] = todo match {
      case Nil                           => acc.reverse.success
      case (ref, _) :: rest if seen(ref) => go(rest, seen, acc)
      case (ref, origin) :: rest =>
        lowerRef(ref, origin, bytesOf) match {
          case Left(e)        => Left(e)
          case Right(lowered) => go(rest ::: lowered.calls.map(_ -> origin), seen + ref, lowered :: acc)
        }
    }
    go(todo, seen, Nil)
  }

  private def functionsOf(lowered: List[BytecodeRemapper.Lowered]): List[p.Function] =
    IntrinsicTable.dedupe(lowered.flatMap(l => l.fn :: l.helpers))

  def resolve(roots: List[MethodRef], bytesOf: ClassBytes): Result[List[p.Function]] =
    walk(roots.map(r => r -> r), Set.empty, bytesOf).map(functionsOf(_))

  // the root's own function is returned apart from the closure it heads
  def resolveRoot(root: MethodRef, bytesOf: ClassBytes): Result[(p.Function, List[p.Function])] = for {
    head <- lowerRef(root, root, bytesOf)
    rest <- walk(head.calls.map(_ -> root), Set(root), bytesOf)
  } yield head.fn -> functionsOf(head :: rest)

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
        md.params.toList.traverse(BytecodeRemapper.typeOf(_)) == Right(params) &&
        BytecodeRemapper.typeOf(md.ret) == Right(rtn)
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
