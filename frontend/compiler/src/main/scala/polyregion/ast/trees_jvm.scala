package polyregion.ast

import polyregion.ast.PolyAST as p

import java.lang.reflect.Modifier
import scala.annotation.tailrec
import scala.reflect.ClassTag

extension (e: => p.Sym.type) {

  def apply[T](using tag: ClassTag[T]): p.Sym = apply(tag.runtimeClass)

  def apply[T](cls: Class[T]): p.Sym = {
    // normalise naming differences
    // Java        => package.Companion$Member
    // Scala Macro => package.Companion$.Member
    @tailrec def go(cls: Class[?], xs: List[String] = Nil, companion: Boolean = false): List[String] = {
      val name = cls.getSimpleName + (if (companion) "$" else "")
      cls.getEnclosingClass match {
        case null => cls.getPackage.getName.split("\\.").toList ::: name :: xs
        case c    => go(c, name :: xs, Modifier.isStatic(cls.getModifiers))
      }
    }
    p.Sym(go(cls))
  }

}

object PolyAstToExpr {

  import scala.quoted.*

  given SymToExpr: ToExpr[p.Sym] with {
    def apply(x: p.Sym)(using Quotes) = '{ p.Sym(${ Expr(x.fqn) }) }
  }
  given NamedToExpr: ToExpr[p.Named] with {
    def apply(x: p.Named)(using Quotes) = '{ p.Named(${ Expr(x.symbol) }, ${ Expr(x.tpe) }) }
  }
  given StructTypeToExpr: ToExpr[p.Type.Struct] with {
    def apply(x: p.Type.Struct)(using Quotes) = '{
      p.Type.Struct(${ Expr(x.name) }, ${ Expr(x.args) })
    }
  }

  given StructDefToExpr: ToExpr[p.StructDef] with {
    def apply(x: p.StructDef)(using Quotes) = '{
      p.StructDef(
        ${ Expr(x.name) },
        ${ Expr(x.tpeVars) },
        ${ Expr(x.members) },
        ${ Expr(x.parents) }
      )
    }
  }

  given ArrayAttrToExpr: ToExpr[p.Type.Space] with {
    def apply(x: p.Type.Space)(using Quotes) = x match {
      case p.Type.Space.Local   => '{ p.Type.Space.Local }
      case p.Type.Space.Global  => '{ p.Type.Space.Global }
      case p.Type.Space.Private => '{ p.Type.Space.Private }
    }
  }

  given TypeToExpr: ToExpr[p.Type] with {
    def apply(x: p.Type)(using Quotes) = x match {
      case p.Type.Var(_)  => ???
      case p.Type.Float16 => '{ p.Type.Float16 }
      case p.Type.Float32 => '{ p.Type.Float32 }
      case p.Type.Float64 => '{ p.Type.Float64 }
      case p.Type.Bool1   => '{ p.Type.Bool1 }
      case p.Type.IntU8   => '{ p.Type.IntU8 }
      case p.Type.IntU16  => '{ p.Type.IntU16 }
      case p.Type.IntU32  => '{ p.Type.IntU32 }
      case p.Type.IntU64  => '{ p.Type.IntU64 }
      case p.Type.IntS8   => '{ p.Type.IntS8 }
      case p.Type.IntS16  => '{ p.Type.IntS16 }
      case p.Type.IntS32  => '{ p.Type.IntS32 }
      case p.Type.IntS64  => '{ p.Type.IntS64 }
      case p.Type.Unit0   => '{ p.Type.Unit0 }

      case p.Type.Struct(name, args) =>
        '{ p.Type.Struct(${ Expr(name) }, ${ Expr(args) }) }
      case p.Type.Ptr(component, space) =>
        '{ p.Type.Ptr(${ Expr(component) }, ${ Expr(space) }) }
      case p.Type.Arr(component, length, space) =>
        '{ p.Type.Arr(${ Expr(component) }, ${ Expr(length) }, ${ Expr(space) }) }
      case p.Type.Exec(tpeVars, args, rtn) => ???
      case p.Type.Nothing                  => ???
    }
  }

}
