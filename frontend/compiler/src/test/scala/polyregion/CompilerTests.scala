package polyregion

import scala.quoted.{Expr, FromExpr, Quotes, ToExpr, Varargs}
import polyregion.scalalang.{Compiler, Quoted}
import polyregion.ast.*

import java.io.ObjectOutputStream
import java.io.ByteArrayOutputStream

import java.io.ObjectInputStream
import java.io.ByteArrayInputStream

object CompilerTests {

  given ToExpr[Throwable] with {
    def apply(e: Throwable)(using q: Quotes): Expr[Throwable] = {
      val os = ByteArrayOutputStream()
      ObjectOutputStream(os).writeObject(e)
      val packs = os.toByteArray.grouped(8192).toList

      import q.reflect.*

      val (defs, refs) = packs.zipWithIndex.map { case (pack, i) =>
        val symbol = Symbol.newVal(Symbol.spliceOwner, s"pack$i", TypeRepr.of[Array[Byte]], Flags.Lazy, Symbol.noSymbol)
        (
          ValDef(symbol, Some(Expr(pack).asTerm)),
          Ref(symbol).asExprOf[Array[Byte]]
        )
      }.unzip

      Block(
        defs,
        '{
          def exceptionData: Array[Byte] = Array.concat(${ Varargs(refs) }*)
          ObjectInputStream(ByteArrayInputStream(exceptionData)).readObject().asInstanceOf[Throwable]
        }.asTerm
      ).asExprOf[Throwable]

    }
  }

  inline def compilerAssert(): Unit = ${
    compilerAssert('{
      val a = 1
      val b = 2
      a + b
      ""
//      val c = Array(1)
    })
  }
  private def compilerAssert(using q: Quotes)(expr: Expr[Any]): Expr[Unit] = {
    println("In!")

    try {

      (for {
        (captures, prismRefs, monoMap, prog0, log) <- Compiler.compileExpr(using Quoted(q))(expr)
        _ = println(log.name)
      } yield '{ () }).fold(throw _, identity)

    } catch {
      case e: Throwable =>
        '{
          val exception = ${ Expr(e) }
          exception.printStackTrace
          throw exception
        }
    }

  }

}
