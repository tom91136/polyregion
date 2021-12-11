package polyregion

import scala.quoted.{Expr, Quotes}
import scala.quoted.*
object compileTime {



  enum PolyAst{
    case Sym(fqcn: String)
    case Type(sym : Sym, args : List[Sym])
    // case Call(receiver : Type, args: List[Type])
    // case Var(name : String, tpe : Type, mutable: Boolean)

    // case While()

  }
  


  inline def showExpr(inline x: Any) : Any ={
    ${showExprImpl('x)}
  }

  inline def showTpe[B] : Unit ={
    ${showTpeImpl[B]}
    ()
  }

  def showTpeImpl[B : Type](using Quotes) : Expr[Unit] = {
    import quotes.reflect.*
    println(TypeRepr.of[B].typeSymbol.tree.show)
    import pprint._
    pprint.pprintln(TypeRepr.of[B].typeSymbol)

    '{}
  }


  def showExprImpl(x : Expr[Any])(using Quotes) :  Expr[Any]= {
    import quotes.reflect.*
    given Printer[Tree] = Printer.TreeStructure

    println("CP:" + System.getProperty("java.class.path"))

    println("[RAW TREE]: "+ x.show) // tree => source
    // println(">>>!"+ x.asTerm.show) // term => AST
    // println(">>>!"+ x.asTerm ) // tree => source


    val FloatTpe = TypeRepr.of[Float]
    val IntTpe = TypeRepr.of[Int]

    val FloatArrayTpe = TypeRepr.of[Array[Float]]
    

    // var sym = Symbol.spliceOwner


    // pprint.pprintln("~~ "+sym.owner.fullName)
    // pprint.pprintln("~~ "+sym.owner.owner.tree)

    
    // MyTraverser().traverseTree(sym.tree)(Symbol.noSymbol)


    // pprint.pprintln("s="+x.asTerm.pos.startLine + ":" +x.asTerm.pos.startColumn + " e="+ x.asTerm.pos.endLine + ":" +x.asTerm.pos.endColumn   )
     
    
    // println(x.asTerm.pos.sourceFile)


// AppliedType(TypeRef(TermRef(ThisType(TypeRef(NoPrefix,module class <root>)),object scala),Array),List(TypeRef(TermRef(ThisType(TypeRef(NoPrefix,module class <root>)),object scala),Float))) #MACRO DEf

// AppliedType(TypeRef(ThisType(TypeRef(NoPrefix,module class scala)),class Array),List(TypeRef(ThisType(TypeRef(NoPrefix,module class scala)),class Float))) # inferred
// AppliedType(TypeRef(ThisType(TypeRef(NoPrefix,module class scala)),class Array),List(TypeRef(TermRef(ThisType(TypeRef(NoPrefix,module class <root>)),object scala),class Float))) #annotated
// AppliedType(TypeRef(ThisType(TypeRef(NoPrefix,module class scala)),class Array),List(TypeRef(TermRef(ThisType(TypeRef(NoPrefix,module class <root>)),object scala),class Float)))

// AppliedType =tpe[_]
// TypeRef = tpe = _
// TermRef


  def fullyQualifyCls()


  this.scala.X
  this.<root>.scala.X
  

    println(">>>>"+FloatArrayTpe.widenTermRefByName.simplified.dealias)
    println(">>>>"+IntTpe.widenTermRefByName.simplified.dealias)

    class MyTraverser extends TreeTraverser {
      override def traverseTree(tree: Tree)(owner: Symbol): Unit = {
         println(s"Tree> ${tree.show} ${tree.symbol}")

        tree match {
          case i@Ident(name) => 
            println(s"\t-->${name}")
            println(s"\t-->${i.tpe.widenTermRefByName.simplified.dealias}")

              i.tpe.widenTermRefByName.simplified.dealias match {
                case at@AppliedType(_,_) => 
                  println(s"\t--> ctor=${at.tycon.simplified}")
                  println(s"\t--> args=${at.args}")

                case _ => 
              }

          case _ => 
        }

         traverseTreeChildren(tree)(owner)
      }
    }
    MyTraverser().traverseTree(x.asTerm)(Symbol.noSymbol)
    // println(x.asTerm)

    // pprint.pprintln(Symbol.classSymbol("hps.Stage$").tree)



    // x.asTerm match{
    //   case Inlined(_, _, Inlined(_, _, Block(_, Block(List(DefDef(_, _, _, Some(preRhs))), _)))) =>

    //     preRhs match {
    //       case Block(List(ValDef(_, _, Some(x))), _) => pprint.pprintln(""+x.symbol.tree)
    //     }

    //     pprint.pprintln(preRhs)
    // }


    x
    //    given Printer[Tree] = Printer.TreeStructure
    //    x.show
  }

}