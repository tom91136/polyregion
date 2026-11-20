package polyregion.javalang

import cats.syntax.all.*
import com.sun.tools.javac.code.Symbol as JSymbol
import polyregion.ast.{PolyAST as p, *}

import javax.lang.model.`type`.{ArrayType, DeclaredType, TypeKind, TypeMirror}
import javax.lang.model.element.{ElementKind, Modifier, NestingKind, TypeElement, VariableElement}
import scala.jdk.CollectionConverters.*

object JavaTyper {

  // not a Java field: the host writes it from the object's class
  final val TagMember = "#tag"

  // a local or anonymous class has no qualified name, the flat name is unique per compilation
  def structName(elem: TypeElement): String = (elem.getNestingKind, elem) match {
    case (NestingKind.LOCAL | NestingKind.ANONYMOUS, c: JSymbol.ClassSymbol) => c.flatName.toString
    case _                                                                   => elem.getQualifiedName.toString
  }

  def instanceFields(elem: TypeElement): List[VariableElement] =
    elem.getEnclosedElements.asScala.toList.collect {
      case f: VariableElement if f.getKind == ElementKind.FIELD && !f.getModifiers.contains(Modifier.STATIC) => f
    }

  // javac marks captured state synthetic and getEnclosedElements hides it
  def structFields(elem: TypeElement): List[(String, TypeMirror)] = elem match {
    case c: JSymbol.ClassSymbol
        if elem.getNestingKind == NestingKind.LOCAL || elem.getNestingKind == NestingKind.ANONYMOUS =>
      c.members().getSymbols().asScala.toList.reverse.collect {
        case v: JSymbol.VarSymbol if v.getKind == ElementKind.FIELD && !v.isStatic => v.name.toString -> v.asType
      }
    case _ => instanceFields(elem).map(f => f.getSimpleName.toString -> f.asType)
  }

  def structSyms(t: p.Type): List[p.Sym] = t match {
    case p.Type.Struct(n, args) => n :: args.flatMap(structSyms(_))
    case p.Type.Ptr(c, _)       => structSyms(c)
    case p.Type.Arr(c, _, _)    => structSyms(c)
    case p.Type.Exec(_, as, r)  => as.flatMap(structSyms(_)) ++ structSyms(r)
    case _                      => Nil
  }

  // the tag leads so every dispatch candidate agrees on its offset
  def lowerStruct(elem: TypeElement, tagged: Boolean): Result[p.StructDef] =
    structFields(elem)
      .traverse((name, mirror) => typer(mirror).map(t => p.Named(name, t)))
      .map { members =>
        val all = if (tagged) p.Named(TagMember, p.Type.IntS32) :: members else members
        p.StructDef(p.Sym(structName(elem)), Nil, all, Nil)
      }

  // a constant with state is rejected rather than flattened to an ordinal
  private def enumAsOrdinal(elem: TypeElement): Result[p.Type] = {
    val state = instanceFields(elem)
    if (state.isEmpty) p.Type.IntS32.success
    else
      (s"Cannot lower `${elem.getQualifiedName}` as an ordinal: its constants carry state " +
        s"(${state.map(_.getSimpleName).mkString(", ")})").fail
  }

  def typer(m: TypeMirror): Result[p.Type] = m.getKind match {
    case TypeKind.BOOLEAN => p.Type.Bool1.success
    case TypeKind.BYTE    => p.Type.IntS8.success
    case TypeKind.SHORT   => p.Type.IntS16.success
    case TypeKind.INT     => p.Type.IntS32.success
    case TypeKind.LONG    => p.Type.IntS64.success
    case TypeKind.CHAR    => p.Type.IntU16.success
    case TypeKind.FLOAT   => p.Type.Float32.success
    case TypeKind.DOUBLE  => p.Type.Float64.success
    case TypeKind.VOID    => p.Type.Unit0.success
    case TypeKind.NULL    => p.Type.Nothing.success
    case TypeKind.ARRAY =>
      m match {
        case array: ArrayType => typer(array.getComponentType).map(p.Type.Ptr(_, p.Type.Space.Global))
        case _                => unlowerable(m)
      }
    case TypeKind.DECLARED =>
      m match {
        case declared: DeclaredType =>
          declared.asElement match {
            case elem: TypeElement => declaredType(elem)
            case _                 => unlowerable(m)
          }
        case _ => unlowerable(m)
      }
    case _ => unlowerable(m)
  }

  private def unlowerable(m: TypeMirror): Result[p.Type] = s"Cannot lower Java type `$m` (kind=${m.getKind})".fail

  private def declaredType(elem: TypeElement): Result[p.Type] =
    elem.getQualifiedName.toString match {
      case "java.lang.Boolean"                   => p.Type.Bool1.success
      case "java.lang.Byte"                      => p.Type.IntS8.success
      case "java.lang.Short"                     => p.Type.IntS16.success
      case "java.lang.Integer"                   => p.Type.IntS32.success
      case "java.lang.Long"                      => p.Type.IntS64.success
      case "java.lang.Character"                 => p.Type.IntU16.success
      case "java.lang.Float"                     => p.Type.Float32.success
      case "java.lang.Double"                    => p.Type.Float64.success
      case "java.lang.Void"                      => p.Type.Unit0.success
      case _ if elem.getKind == ElementKind.ENUM => enumAsOrdinal(elem)
      // TODO carry type arguments through once generic structs are supported
      case _ => p.Type.Struct(p.Sym(structName(elem)), Nil).success
    }
}
