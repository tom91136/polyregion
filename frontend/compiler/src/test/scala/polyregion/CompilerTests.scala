package polyregion

import polyregion.ast.*
import polyregion.scalalang.{Compiler, Quoted}

import java.io.{ByteArrayInputStream, ByteArrayOutputStream, ObjectInputStream, ObjectOutputStream}
import scala.quoted.*

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

  inline def compilerAssert(inline f: Any): Unit = ${ compilerAssert('f) }
  private def compilerAssert(using q: Quotes)(expr: Expr[Any]): Expr[Unit] = {
    println("In!")
    val l = Log("")

    try {

      val v = (for {
        (captures, prismRefs, monoMap, prog0) <- Compiler.compileExpr(using Quoted(q))(l, expr)

      } yield '{ () }).fold(x => throw x, identity)
      println(s"Log=${l.lines.size}")
      println(l.render(0).mkString("\n"))
      v

    } catch {
      case e: Throwable =>

        println(s"Log=${l.lines.size}")
        println(l.render(0).mkString("\n"))
        '{
          val exception = ${ Expr(e) }
          exception.printStackTrace
          throw exception
        }
    }

  }

}
