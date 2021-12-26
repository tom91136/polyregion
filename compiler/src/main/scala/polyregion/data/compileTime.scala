package polyregion.data

object compileTime {

  import scala.quoted.*

  case class ClassCtorMeta[T](
    name : String,
    ctorArgs :  List[(String, String)],
    ctorFwd : List[ClassCtorMeta.Invocation[T]]
  )

  object ClassCtorMeta{
    enum Invocation[+T]{
      case Forward(tpe: T, name : String, path: List[String] = Nil)
      case Const(tpe: T, value : Any)
      case Unsupported
    }
  }


  inline def isMemberOfSealedHierarchy[T]: Boolean =
    ${ isMemberOfSealedHierarchyImpl[T] }

  def isMemberOfSealedHierarchyImpl[T](using quotes: Quotes, tpe: Type[T]): Expr[Boolean] = {
    import quotes.reflect.*
//    val parents = TypeRepr.of[T].symbol // .baseClasses

    val symbol = TypeRepr.of[T].typeSymbol

    // case X extends B(1) => (we get a val ClsName, derive base class args and apply with given constants)
    // case X(foo:Any, x:Int) extends B(x) => we get class Def

    def matchCtor(t : Tree) = {
      t match {
        case Apply(Select(New(name), "<init>"), args) => 
          println(s"invoke $name ($args)")
      }
    }

    //DefDef(s, clauses, tree, term)
    symbol.tree match {
      case ClassDef(name, _,headCtorApply :: _,_,_) => 
        println(s"match!:$name")
        pprint.pprintln(headCtorApply)
        matchCtor(headCtorApply)
      case _ => println("No!")
    }



    import pprint.*
    //  pprint.pprintln(symbol.tree)
    // println(s"M=${symbol.tree}")
    println(s"  Ctor: ${symbol.primaryConstructor.tree.show}")
    println(symbol.isClassDef)
    

    Expr(true)
//    Expr(parents.exists(_.flags.is(Flags.Sealed)))
  }

}
