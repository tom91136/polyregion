package polyregion.javalang

import cats.syntax.all.*
import org.jetbrains.java.decompiler.struct.gen.MethodDescriptor
import polyregion.ast.{PolyAST as p, *}

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

  def lowerRef(ref: MethodRef, bytesOf: ClassBytes): Result[(p.Function, List[MethodRef])] =
    bytesOf(ref.owner)
      .failIfEmpty(s"No class file for `${ref.owner}` while lowering `${ref.repr}`")
      .flatMap(BytecodeRemapper.lowerWithCalls(_, ref.name, ref.desc))
      .leftMap(e => CompilerException(s"Cannot lower `${ref.repr}` from its class file: ${e.getMessage}", e))

  def resolve(roots: List[MethodRef], bytesOf: ClassBytes): Result[List[p.Function]] = {
    @tailrec def go(todo: List[MethodRef], seen: Set[MethodRef], acc: List[p.Function]): Result[List[p.Function]] =
      todo match {
        case Nil                      => acc.reverse.success
        case ref :: rest if seen(ref) => go(rest, seen, acc)
        case ref :: rest =>
          lowerRef(ref, bytesOf) match {
            case Left(e)              => Left(e)
            case Right((fn, callees)) => go(rest ::: callees, seen + ref, fn :: acc)
          }
      }
    go(roots, Set.empty, Nil)
  }

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
