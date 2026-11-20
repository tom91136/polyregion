package polyregion.ast

import polyregion.ast.PolyAST as p

private[ast] object JavaInterfaceCodeGen {
  import InterfaceCodeGenSupport.*

  private val Keywords = Set(
    "abstract",
    "assert",
    "boolean",
    "break",
    "byte",
    "case",
    "catch",
    "char",
    "class",
    "const",
    "continue",
    "default",
    "do",
    "double",
    "else",
    "enum",
    "extends",
    "false",
    "final",
    "finally",
    "float",
    "for",
    "goto",
    "if",
    "implements",
    "import",
    "instanceof",
    "int",
    "interface",
    "long",
    "native",
    "new",
    "null",
    "package",
    "private",
    "protected",
    "public",
    "return",
    "short",
    "static",
    "strictfp",
    "super",
    "switch",
    "synchronized",
    "this",
    "throw",
    "throws",
    "transient",
    "true",
    "try",
    "void",
    "volatile",
    "while",
    "_"
  )

  private val Annotation = "polyregion.javalang.Interface"

  private final case class Prim(name: String, boxed: String, code: Char, bytes: Int, zero: String)

  private val Prims: Map[p.Type, Prim] = Map(
    p.Type.Float32 -> Prim("float", "Float", 'F', 4, "0.0f"),
    p.Type.Float64 -> Prim("double", "Double", 'D', 8, "0.0"),
    p.Type.IntS8   -> Prim("byte", "Byte", 'B', 1, "(byte) 0"),
    p.Type.IntS16  -> Prim("short", "Short", 'S', 2, "(short) 0"),
    p.Type.IntU16  -> Prim("char", "Character", 'C', 2, "(char) 0"),
    p.Type.IntS32  -> Prim("int", "Integer", 'I', 4, "0"),
    p.Type.IntS64  -> Prim("long", "Long", 'J', 8, "0L"),
    p.Type.Bool1   -> Prim("boolean", "Boolean", 'Z', 1, "false")
  )

  private val PrimsByCode: Map[Char, Prim] = Prims.values.map(prim => prim.code -> prim).toMap

  private val Primitives: List[Prim] = List(p.Type.IntS32, p.Type.IntS64, p.Type.Float32, p.Type.Float64).map(Prims)

  private enum Choice {
    case Primitive(prim: Prim)
    case Ref
  }

  private type Assignment = Map[String, Choice]

  private def identifier(value: String, what: String): String =
    targetIdentifier(value, s"Java $what", Keywords)

  private def concrete(value: p.Type): String = value match {
    case p.Type.Unit0             => "void"
    case p.Type.Struct(name, Nil) => name.fqn.map(identifier(_, "type")).mkString(".")
    case other => Prims.get(other).map(_.name).getOrElse(fail(s"type has no exact Java interface projection: $other"))
  }

  private def concreteCode(value: p.Type): Char = value match {
    case p.Type.Unit0 => 'V'
    case other        => Prims.get(other).fold('L')(_.code)
  }

  private def componentOf(value: p.Type): Option[p.Type] = value match {
    case p.Type.Ptr(comp, _)    => Some(comp)
    case p.Type.Arr(comp, _, _) => Some(comp)
    case _                      => None
  }

  private def element(assignment: Assignment, name: String): String = assignment(name) match {
    case Choice.Primitive(prim) => prim.name
    case Choice.Ref             => name
  }

  private final case class Shape(args: List[Char], rtn: Char) {
    def name: String = s"Fn${args.mkString}_$rtn"
  }

  private final case class Callable(shape: Shape, refs: List[String])

  private def callable(assignment: Assignment, args: List[p.Type], rtn: p.Type): Callable = {
    val positions = (args :+ rtn).map {
      case p.Type.Var(name, _) =>
        assignment(name) match {
          case Choice.Primitive(prim) => (prim.code, None)
          case Choice.Ref             => ('L', Some(name))
        }
      case p.Type.Exec(_, _, _) => fail("nested callable types are not supported by the Java projection yet")
      case other                => (concreteCode(other), Option.when(concreteCode(other) == 'L')(concrete(other)))
    }
    Callable(Shape(positions.init.map(_._1), positions.last._1), positions.flatMap(_._2))
  }

  private def parameter(assignment: Assignment, scalarOnly: Set[String], arg: p.Type): (String, Option[Callable]) =
    arg match {
      case p.Type.Exec(Nil, args, rtn) =>
        val c        = callable(assignment, args, rtn)
        val typeArgs = if (c.refs.isEmpty) "" else c.refs.mkString("<", ", ", ">")
        (s"${c.shape.name}$typeArgs", Some(c))
      case p.Type.Exec(_, _, _) => fail("generic callable types are not supported by the Java projection yet")
      case p.Type.Var(name, _) =>
        assignment(name) match {
          case Choice.Primitive(prim) => (if (scalarOnly(name)) prim.boxed else prim.name, None)
          case Choice.Ref             => (name, None)
        }
      case other =>
        componentOf(other) match {
          case Some(p.Type.Var(name, _)) => (s"${element(assignment, name)}[]", None)
          case Some(comp)                => (s"${concrete(comp)}[]", None)
          case None                      => (concrete(other), None)
        }
    }

  private def result(assignment: Assignment, value: p.Type): String = value match {
    case p.Type.Var(name, _) => element(assignment, name)
    case other               => concrete(other)
  }

  private def arrayVariables(decl: p.FunctionDecl): Set[String] =
    decl.args.flatMap(arg => componentOf(arg.named.tpe)).collect { case p.Type.Var(name, _) => name }.toSet

  private def classify(decl: p.FunctionDecl): Set[String] = {
    val arrays  = arrayVariables(decl)
    val scalars = decl.args.map(_.named.tpe).collect { case p.Type.Var(name, _) => name }.toSet -- arrays
    val fqn     = decl.name.fqn.mkString(".")
    decl.tpeVars.map(_.name).filterNot(name => arrays(name) || scalars(name)).foreach { name =>
      fail(s"type variable `$name` of `$fqn` is not determined by an array or scalar argument")
    }
    decl.tpeVars.foreach { variable =>
      variable.exactSizeInBytes.filterNot(width => Primitives.exists(_.bytes == width)).foreach { width =>
        fail(s"type variable `${variable.name}` of `$fqn` has no primitive binding of $width bytes")
      }
    }
    scalars
  }

  private def assignments(decl: p.FunctionDecl): List[Assignment] = {
    val names  = decl.tpeVars.map(_.name)
    val arrays = arrayVariables(decl)
    val primitives = names.foldRight(List(List.empty[Choice])) { (_, tails) =>
      for {
        prim <- Primitives
        tail <- tails
      } yield Choice.Primitive(prim) :: tail
    }
    val reference = Option.when(names.exists(arrays))(names.map(_ -> Choice.Ref).toMap)
    primitives.map(names.zip(_).toMap) ++ reference
  }

  private final case class Stub(source: String, callables: List[Callable])

  private def stub(interfaceDef: p.Interface, decl: p.FunctionDecl, assignment: Assignment): Stub = {
    val scalarOnly = classify(decl)
    val typeParams = decl.tpeVars.map(_.name).filter(assignment(_) == Choice.Ref) match {
      case Nil   => ""
      case names => names.mkString("<", ", ", "> ")
    }
    val parameters = decl.args.map(arg => parameter(assignment, scalarOnly, arg.named.tpe) -> arg.named.symbol)
    val rendered   = parameters.map { case ((tpe, _), name) => s"$tpe $name" }
    val prefix     = s"  public static $typeParams${result(assignment, decl.rtn)} ${decl.name.last}("
    val signature = {
      val inline = s"$prefix${rendered.mkString(", ")}) {"
      if (inline.length <= 100) inline
      else s"$prefix\n${rendered.map("      " + _).mkString(",\n")}) {"
    }
    val fqn = decl.name.fqn.mkString(".")
    val source = s"""  @Interface("${interfaceIdentity(interfaceDef, decl)}")
                    |$signature
                    |    throw new UnsupportedOperationException("compiler did not replace $fqn");
                    |  }""".stripMargin
    Stub(source, parameters.flatMap(_._1._2))
  }

  private def functionalInterface(shape: Shape): String = {
    val typeNames = shape.args.zipWithIndex.collect { case ('L', i) => s"T$i" } ++ Option.when(shape.rtn == 'L')("R")
    val generics  = if (typeNames.isEmpty) "" else typeNames.mkString("<", ", ", ">")
    def named(code: Char, position: String): String = code match {
      case 'L' => position
      case 'V' => "void"
      case c   => PrimsByCode(c).name
    }
    val params = shape.args.zipWithIndex.map { case (code, i) => s"${named(code, s"T$i")} arg$i" }
    s"""  @FunctionalInterface
       |  public interface ${shape.name}$generics {
       |    ${named(shape.rtn, "R")} apply(${params.mkString(", ")});
       |  }""".stripMargin
  }

  private def validated(interfaceDef: p.Interface, config: JavaConfig): List[p.FunctionDecl] = {
    if (config.packageName.isEmpty) fail("Java package name is empty")
    config.packageName.split('.').foreach(identifier(_, "package"))
    identifier(config.className, "class")
    val decls = declarations(interfaceDef)
    validatePortableOverloads(decls, "Java")
    decls.foreach { decl =>
      identifier(decl.name.last, "method")
      decl.tpeVars.foreach(variable => identifier(variable.name, "type variable"))
      decl.args.foreach(arg => identifier(arg.named.symbol, "parameter"))
      classify(decl)
    }
    decls
  }

  def overloadCount(decl: p.FunctionDecl): Int = assignments(decl).size

  def apply(interfaceDef: p.Interface, config: JavaConfig): String = {
    val stubs  = validated(interfaceDef, config).flatMap(decl => assignments(decl).map(stub(interfaceDef, decl, _)))
    val shapes = stubs.flatMap(_.callables.map(_.shape)).distinct.sortBy(_.name).map(functionalInterface)
    val body   = (shapes ++ stubs.map(_.source)).mkString("\n\n")
    s"""package ${config.packageName};
       |
       |import $Annotation;
       |
       |public final class ${config.className} {
       |
       |  private ${config.className}() {}
       |
       |$body
       |}
       |""".stripMargin
  }

  private def zero(value: p.Type): String = Prims.get(value).fold("null")(_.zero)

  private def surfaceArgument(assignment: Assignment, arg: p.Type): String = arg match {
    case p.Type.Exec(_, args, rtn) =>
      val parameters = args.indices.map(i => s"arg$i").mkString("(", ", ", ")")
      val body = rtn match {
        case p.Type.Unit0 => "{ }"
        case p.Type.Var(name, _) =>
          assignment(name) match {
            case Choice.Primitive(prim) => prim.zero
            case Choice.Ref             => "null"
          }
        case other => zero(other)
      }
      s"$parameters -> $body"
    case p.Type.Var(name, _) =>
      assignment(name) match {
        case Choice.Primitive(prim) => prim.zero
        case Choice.Ref             => "new Value()"
      }
    case other =>
      componentOf(other) match {
        case Some(p.Type.Var(name, _)) =>
          assignment(name) match {
            case Choice.Primitive(prim) => s"${prim.name}s"
            case Choice.Ref             => "values"
          }
        case Some(comp)                     => s"new ${concrete(comp)}[n]"
        case None if other == p.Type.IntS32 => "n"
        case None                           => zero(other)
      }
  }

  def surface(interfaceDef: p.Interface, config: JavaConfig, className: String): String = {
    val decls = validated(interfaceDef, config)
    identifier(className, "class")
    val calls = decls.flatMap { decl =>
      assignments(decl).map { assignment =>
        val profile = decl.tpeVars.map { variable =>
          val shown = assignment(variable.name) match {
            case Choice.Primitive(prim) => prim.name
            case Choice.Ref             => "Value"
          }
          s"${variable.name}=$shown"
        }
        val comment = s"    // ${decl.name.last}${if (profile.isEmpty) "" else profile.mkString(": ", ", ", "")}\n"
        val args    = decl.args.map(arg => surfaceArgument(assignment, arg.named.tpe))
        val prefix  = s"    ${config.className}.${decl.name.last}("
        val inline  = s"$prefix${args.mkString(", ")});"
        comment + (if (inline.length <= 100) inline else s"$prefix\n${args.map("        " + _).mkString(",\n")});")
      }
    }
    s"""package ${config.packageName};
       |
       |public final class $className {
       |
       |  private static final class Value {}
       |
       |  public static void compile() {
       |    int n = 4;
       |    int[] ints = new int[n];
       |    long[] longs = new long[n];
       |    float[] floats = new float[n];
       |    double[] doubles = new double[n];
       |    Value[] values = new Value[n];
       |
       |${calls.mkString("\n\n")}
       |  }
       |}
       |""".stripMargin
  }
}
