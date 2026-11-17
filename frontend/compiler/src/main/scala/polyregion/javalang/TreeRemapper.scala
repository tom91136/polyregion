package polyregion.javalang

import cats.syntax.all.*
import com.sun.source.tree.*
import com.sun.source.util.Trees
import com.sun.tools.javac.code.Symbol as JSymbol
import com.sun.tools.javac.tree.JCTree
import polyregion.ast.{PolyAST as p, *}

import javax.annotation.processing.ProcessingEnvironment
import javax.lang.model.`type`.{DeclaredType, TypeKind, TypeMirror}
import javax.lang.model.element.{ElementKind, Modifier, TypeElement}
import javax.lang.model.util.{Elements, Types}
import scala.collection.mutable
import scala.collection.mutable.ListBuffer
import scala.jdk.CollectionConverters.*

object TreeRemapper {

  import polyregion.ast.Env.debug

  final class DepCollector(
      val localTrees: Map[JSymbol.MethodSymbol, MethodTree],
      val methodTable: Map[String, MethodTree],
      val classTable: Map[String, TypeElement],
      val elements: Elements,
      val types: Types
  ) {
    val fnSyms = mutable.LinkedHashSet[JSymbol.MethodSymbol]()

    def treeOf(sym: JSymbol.MethodSymbol): Option[MethodTree] =
      localTrees.get(sym).orElse(methodTable.get(fullName(sym))).filter(t => t != null && t.getBody != null)

    // javac adds synthetic constructor params, so a call site may name a different symbol than the declaration
    def ctorOf(owner: JSymbol.ClassSymbol, arity: Int): Option[MethodTree] =
      localTrees.collectFirst {
        case (s, t)
            if s.owner == owner && s.name.toString == "<init>" && t != null && t.getBody != null &&
              t.getParameters.size == arity =>
          t
      }

    val tagOf                    = mutable.LinkedHashMap[String, Int]()
    val taggedTypes              = mutable.LinkedHashSet[String]()
    def tagFor(fqn: String): Int = tagOf.getOrElseUpdate(fqn, tagOf.size)

    val witnessed = mutable.LinkedHashSet[TypeElement]()

    val directStatics = mutable.LinkedHashMap[JSymbol.MethodSymbol, List[JSymbol.VarSymbol]]()
    val callees       = mutable.LinkedHashMap[JSymbol.MethodSymbol, List[JSymbol.MethodSymbol]]()
    var staticParams: Map[JSymbol.MethodSymbol, List[JSymbol.VarSymbol]] = Map.empty

    // Ptr drops a Java array's length, so a callee measuring an array parameter takes the count alongside it
    val measuredParams = mutable.LinkedHashMap[JSymbol.MethodSymbol, mutable.LinkedHashSet[Int]]()
    val lengthFlows    = mutable.LinkedHashMap[JSymbol.MethodSymbol, ListBuffer[(JSymbol.MethodSymbol, Int, Int)]]()
    var lengthParams: Map[JSymbol.MethodSymbol, List[Int]] = Map.empty

    var capturedClosures: Map[JSymbol, Closure] = Map.empty
  }

  final case class Closure(impl: JSymbol.MethodSymbol, captures: List[Either[p.Term, Closure]])

  enum Capture {
    case Arg(index: Int)
    case Length(index: Int)
    case RowLengths(index: Int)
    case StaticField(owner: String, name: String)
    case Receiver
  }

  enum RowLength {
    case Uniform(len: p.Term)
    case PerRow(lens: p.Term)
  }

  final case class Region(program: p.Program, captures: List[Capture], tags: Map[String, Int])

  final case class LoopCtx(
      update: List[ExpressionStatementTree],
      reeval: List[p.Stmt],
      viaLabel: Option[LabelCtx] = None
  )

  final case class LabelCtx(name: String, brk: p.Term.Select, cnt: p.Term.Select)

  final class Scope(val deps: DepCollector, val isEntry: Boolean, val planned: Boolean = true) {
    private var counter                     = 0
    val locals                              = mutable.Map[JSymbol, p.Named]()
    val paramOf                             = mutable.Map[JSymbol, Int]()
    val measured                            = mutable.LinkedHashSet[Int]()
    val argFlows                            = ListBuffer[(JSymbol.MethodSymbol, Int, Int)]()
    val arrayLengths                        = mutable.Map[JSymbol, p.Term]()
    val rowLengths                          = mutable.Map[JSymbol, RowLength]()
    var pendingArrayLength: Option[p.Term]  = None
    var pendingRowLength: Option[RowLength] = None
    val staticFields                        = mutable.LinkedHashMap[JSymbol.VarSymbol, p.Named]()
    val calls                               = mutable.LinkedHashSet[JSymbol.MethodSymbol]()
    var thisOpt: Option[p.Named]            = None
    var rtnType: p.Type                     = p.Type.Unit0
    def fresh(tpe: p.Type): p.Named         = { val n = p.Named(s"_v$counter", tpe); counter += 1; n }

    val closures = mutable.Map[JSymbol, Closure]()

    private val inlining                             = mutable.Set[JSymbol.MethodSymbol]()
    def isInlining(s: JSymbol.MethodSymbol): Boolean = inlining.contains(s)
    def whileInlining[A](s: JSymbol.MethodSymbol)(f: => A): A = {
      inlining += s
      try f
      finally inlining -= s
    }

    def withThis[A](named: Option[p.Named])(f: => A): A = {
      val saved = thisOpt
      thisOpt = named
      try f
      finally thisOpt = saved
    }

    // Java scopes nest and PolyAST bodies do not, so disjoint `for (int i ...)` loops would trip the verifier
    private val names = mutable.Set[String]()
    def uniqueName(base: String): String =
      if (names.add(base)) base
      else {
        var i         = 1
        var candidate = s"$base$$$i"
        while (!names.add(candidate)) { i += 1; candidate = s"$base$$$i" }
        candidate
      }

    private val loops                  = mutable.Stack[LoopCtx]()
    def enclosingLoop: Option[LoopCtx] = loops.headOption
    def loopDepth: Int                 = loops.size
    def inLoop[A](ctx: LoopCtx)(f: => A): A = {
      loops.push(ctx)
      try f
      finally loops.pop()
    }

    private val labels                           = mutable.Stack[LabelCtx]()
    def activeLabels: List[LabelCtx]             = labels.toList
    def labelled(name: String): Option[LabelCtx] = labels.find(_.name == name)
    def withLabel[A](ctx: LabelCtx)(f: => A): A = {
      labels.push(ctx)
      try f
      finally labels.pop()
    }

    private val yields                               = mutable.Stack[(p.Term.Select, Int)]()
    def enclosingYield: Option[(p.Term.Select, Int)] = yields.headOption
    def withYieldTarget[A](target: p.Term.Select)(f: => A): A = {
      yields.push((target, loops.size))
      try f
      finally yields.pop()
    }
  }

  private def jcType(t: Tree): TypeMirror = t.asInstanceOf[JCTree].`type`

  private def declaredName(m: TypeMirror): String =
    if (m == null || m.getKind != TypeKind.DECLARED) ""
    else m.asInstanceOf[DeclaredType].asElement.asInstanceOf[TypeElement].getQualifiedName.toString

  private def isStringTpe(m: TypeMirror): Boolean = declaredName(m) == "java.lang.String"

  private def isAssertionErrorTpe(m: TypeMirror): Boolean = declaredName(m) == "java.lang.AssertionError"

  private def assertMessage(nc: NewClassTree): Result[String] = nc.getArguments.asScala.toList match {
    case Nil                                                              => "assertion failed".success
    case (l: LiteralTree) :: Nil if l.getKind == Tree.Kind.STRING_LITERAL => l.getValue.toString.success
    case (l: LiteralTree) :: Nil                                          => String.valueOf(l.getValue).success
    case xs =>
      s"Unsupported `AssertionError` detail `${xs.mkString(", ")}`; only a literal travels to the device".fail
  }

  private def symOf(t: Tree): JSymbol = t match {
    case i: JCTree.JCIdent        => i.sym
    case f: JCTree.JCFieldAccess  => f.sym
    case v: JCTree.JCVariableDecl => v.sym
    case _                        => null
  }

  private def ownerName(sym: JSymbol): String = sym.owner match {
    case c: JSymbol.ClassSymbol => JavaTyper.structName(c)
    case o                      => Option(o).fold("")(_.toString)
  }

  private def fullName(sym: JSymbol): String = {
    val owner = ownerName(sym)
    if (owner.isEmpty) sym.name.toString else s"$owner.${sym.name}"
  }

  private val foreignPackages = List("java.", "javax.", "jdk.", "sun.", "com.sun.")
  private def isUserMethod(sym: JSymbol.MethodSymbol): Boolean =
    !foreignPackages.exists(ownerName(sym).startsWith)

  private val MetafactoryOwner = "java.lang.invoke.LambdaMetafactory"

  // LambdaToMethod runs before GENERATE, so a lambda is an invokedynamic whose bootstrap args name the impl method
  private def dynamicSym(mi: MethodInvocationTree): Option[JSymbol.DynamicMethodSymbol] =
    symOf(mi.getMethodSelect) match {
      case d: JSymbol.DynamicMethodSymbol => Some(d)
      case _                              => None
    }

  private def metafactoryImpl(d: JSymbol.DynamicMethodSymbol): Option[JSymbol.MethodSymbol] = {
    val bsm     = Option(d.bsm).map(_.baseSymbol())
    val bsmName = bsm.fold("")(_.name.toString)
    val isFactory = bsm.exists(b => ownerName(b) == MetafactoryOwner) &&
      (bsmName == "metafactory" || bsmName == "altMetafactory")
    if (!isFactory || d.staticArgs == null || d.staticArgs.length < 2) None
    else
      d.staticArgs(1) match {
        case h: JSymbol.MethodHandleSymbol =>
          h.baseSymbol() match {
            case m: JSymbol.MethodSymbol => Some(m)
            case _                       => None
          }
        case _ => None
      }
  }

  private def bind(scope: Scope, buf: ListBuffer[p.Stmt], e: p.Expr): p.Term = e match {
    case p.Expr.Alias(t) => t
    case other =>
      val n = scope.fresh(other.tpe)
      buf += p.Stmt.Var(n, Some(other))
      p.Term.Select(n, Nil, n.tpe)
  }

  private def asSelect(e: p.Expr): Result[p.Term.Select] = e match {
    case p.Expr.Alias(s: p.Term.Select) => s.success
    case other                          => s"Expected an lvalue (Term.Select), got ${other.repr}".fail
  }

  private def oneOf(tpe: p.Type): Result[p.Term] = tpe match {
    case p.Type.IntS8   => p.Term.IntS8Const(1).success
    case p.Type.IntS16  => p.Term.IntS16Const(1).success
    case p.Type.IntS32  => p.Term.IntS32Const(1).success
    case p.Type.IntS64  => p.Term.IntS64Const(1L).success
    case p.Type.Float32 => p.Term.Float32Const(1f).success
    case p.Type.Float64 => p.Term.Float64Const(1d).success
    case other          => s"Cannot synthesise a unit value for ${other.repr}".fail
  }

  private def literal(l: LiteralTree): Result[p.Expr] = l.getKind match {
    case Tree.Kind.INT_LITERAL     => p.Expr.Alias(p.Term.IntS32Const(l.getValue.asInstanceOf[Int])).success
    case Tree.Kind.LONG_LITERAL    => p.Expr.Alias(p.Term.IntS64Const(l.getValue.asInstanceOf[Long])).success
    case Tree.Kind.FLOAT_LITERAL   => p.Expr.Alias(p.Term.Float32Const(l.getValue.asInstanceOf[Float])).success
    case Tree.Kind.DOUBLE_LITERAL  => p.Expr.Alias(p.Term.Float64Const(l.getValue.asInstanceOf[Double])).success
    case Tree.Kind.BOOLEAN_LITERAL => p.Expr.Alias(p.Term.Bool1Const(l.getValue.asInstanceOf[Boolean])).success
    case Tree.Kind.CHAR_LITERAL =>
      p.Expr.Alias(p.Term.IntU16Const(l.getValue.asInstanceOf[Character].charValue)).success
    case Tree.Kind.NULL_LITERAL => JavaTyper.typer(jcType(l)).map(tpe => p.Expr.Alias(p.Term.Poison(tpe)))
    case Tree.Kind.STRING_LITERAL =>
      "Unsupported `String` literal; PolyRegion has no device-side String".fail
    case other => s"Unsupported literal kind $other".fail
  }

  // JLS 5.6.2, PolyAST has no implicit conversions
  private def promotedType(l: p.Type, r: p.Type): p.Type =
    if (l == p.Type.Float64 || r == p.Type.Float64) p.Type.Float64
    else if (l == p.Type.Float32 || r == p.Type.Float32) p.Type.Float32
    else if (l == p.Type.IntS64 || r == p.Type.IntS64) p.Type.IntS64
    else p.Type.IntS32

  private def coerce(scope: Scope, buf: ListBuffer[p.Stmt], t: p.Term, target: p.Type): p.Term =
    if (t.tpe == target) t else bind(scope, buf, p.Expr.Cast(t, target))

  private def isBoolean(t: p.Type): Boolean = t == p.Type.Bool1

  private def isArrayOfArrays(m: TypeMirror): Boolean =
    m.getKind == TypeKind.ARRAY &&
      m.asInstanceOf[javax.lang.model.`type`.ArrayType].getComponentType.getKind == TypeKind.ARRAY

  private def isScalar(t: p.Type): Boolean = t match {
    case p.Type.Bool1 | p.Type.IntS8 | p.Type.IntS16 | p.Type.IntU16 | p.Type.IntS32 | p.Type.IntS64 | p.Type.Float32 |
        p.Type.Float64 =>
      true
    case _ => false
  }

  private def staticFieldOf(t: Tree): Option[JSymbol.VarSymbol] = symOf(t) match {
    case v: JSymbol.VarSymbol if v.isStatic && v.owner.isInstanceOf[JSymbol.ClassSymbol] => Some(v)
    case _                                                                               => None
  }

  private def constOf(c: Any, tpe: p.Type): Result[p.Expr] = {
    def num(f: Number => p.Term): Result[p.Expr] = c match {
      case n: Number => p.Expr.Alias(f(n)).success
      case other     => s"Cannot lower constant `$other` as ${tpe.repr}".fail
    }
    tpe match {
      case p.Type.Bool1   => num(n => p.Term.Bool1Const(n.intValue != 0))
      case p.Type.IntS8   => num(n => p.Term.IntS8Const(n.byteValue))
      case p.Type.IntS16  => num(n => p.Term.IntS16Const(n.shortValue))
      case p.Type.IntU16  => num(n => p.Term.IntU16Const(n.intValue.toChar))
      case p.Type.IntS32  => num(n => p.Term.IntS32Const(n.intValue))
      case p.Type.IntS64  => num(n => p.Term.IntS64Const(n.longValue))
      case p.Type.Float32 => num(n => p.Term.Float32Const(n.floatValue))
      case p.Type.Float64 => num(n => p.Term.Float64Const(n.doubleValue))
      case other          => s"Unsupported constant of type ${other.repr}".fail
    }
  }

  private def isEnumType(m: TypeMirror): Boolean =
    m.getKind == TypeKind.DECLARED &&
      m.asInstanceOf[DeclaredType].asElement.getKind == ElementKind.ENUM

  private def enumOrdinal(v: JSymbol.VarSymbol): Result[Int] = v.owner match {
    case c: JSymbol.ClassSymbol =>
      c.getEnclosedElements.asScala.toList
        .filter(_.getKind == ElementKind.ENUM_CONSTANT)
        .indexWhere(_.getSimpleName.toString == v.name.toString) match {
        case -1 => s"`${fullName(v)}` is not one of `${ownerName(v)}`'s constants".fail
        case i  => i.success
      }
    case _ => s"Cannot resolve the ordinal of `${fullName(v)}`".fail
  }

  // a static has no device counterpart: it travels as a one-element buffer the entry captures and callees take as a param
  private def staticSlot(scope: Scope, v: JSymbol.VarSymbol): Result[(p.Named, p.Type)] =
    JavaTyper.typer(v.asType).flatMap { tpe =>
      if (isEnumType(v.asType))
        s"Unsupported static field `${fullName(v)}` of enum type; only an enum constant lowers".fail
      else if (!isScalar(tpe))
        s"Unsupported static field `${fullName(v)}` of type ${tpe.repr}; only primitives can travel to the device".fail
      else
        (
          scope.staticFields.getOrElseUpdate(
            v,
            p.Named(s"#static${scope.staticFields.size}", p.Type.Ptr(tpe, p.Type.Space.Global))
          ),
          tpe
        ).success
    }

  private def staticArgsFor(scope: Scope, sym: JSymbol.MethodSymbol): Result[List[p.Term]] =
    scope.deps.staticParams.getOrElse(sym, Nil).traverse { v =>
      scope.staticFields
        .get(v)
        .map(n => p.Term.Select(n, Nil, n.tpe): p.Term)
        .failIfEmpty(s"`${fullName(sym)}` reaches the static `${fullName(v)}` but its caller holds no slot for it")
    }

  private def lengthArgsFor(
      scope: Scope,
      sym: JSymbol.MethodSymbol,
      argLengths: List[Option[p.Term]]
  ): Result[List[p.Term]] =
    scope.deps.lengthParams.getOrElse(sym, Nil).traverse { i =>
      argLengths
        .lift(i)
        .flatten
        .failIfEmpty(
          s"`${fullName(sym)}` takes the `length` of `${sym.getParameters.get(i).getSimpleName}`, " +
            "but this call site cannot tell how long the array it passes is"
        )
    }

  private def staticRead(scope: Scope, v: JSymbol.VarSymbol): Result[p.Expr] =
    if (v.getKind == ElementKind.ENUM_CONSTANT)
      enumOrdinal(v).map(o => p.Expr.Alias(p.Term.IntS32Const(o)))
    else
      Option(v.getConstValue) match {
        case Some(c) => JavaTyper.typer(v.asType).flatMap(constOf(c, _))
        case None =>
          staticSlot(scope, v).map { (n, tpe) =>
            p.Expr.Index(p.Term.Select(n, Nil, n.tpe), p.Term.IntS32Const(0), tpe)
          }
      }

  private enum Lvalue {
    case Slot(sel: p.Term.Select)
    case Elem(ptr: p.Term.Select, idx: p.Term, comp: p.Type)
  }

  private def lvalueType(lv: Lvalue): p.Type = lv match {
    case Lvalue.Slot(sel)     => sel.tpe
    case Lvalue.Elem(_, _, c) => c
  }

  private def readLvalue(scope: Scope, buf: ListBuffer[p.Stmt], lv: Lvalue): p.Term = lv match {
    case Lvalue.Slot(sel)       => sel
    case Lvalue.Elem(ptr, i, c) => bind(scope, buf, p.Expr.Index(ptr, i, c))
  }

  private def writeLvalue(buf: ListBuffer[p.Stmt], lv: Lvalue, v: p.Term): Unit = lv match {
    case Lvalue.Slot(sel)       => buf += p.Stmt.Mut(sel, p.Expr.Alias(v))
    case Lvalue.Elem(ptr, i, _) => buf += p.Stmt.Update(ptr, i, v)
  }

  // no intermediate copies: binding an `xs[i]` of struct type copies it and the store would miss the array
  private def selectPath(scope: Scope, buf: ListBuffer[p.Stmt], t: ExpressionTree): Result[p.Term.Select] = t match {
    case aa: ArrayAccessTree =>
      for {
        base <- selectPath(scope, buf, aa.getExpression)
        idx  <- mapExpr(scope, buf, aa.getIndex).map(bind(scope, buf, _))
        comp <- JavaTyper.typer(jcType(aa))
      } yield p.Term.Select(base.root, base.steps :+ p.PathStep.IndexDyn(idx), comp)
    case fa: MemberSelectTree if staticFieldOf(fa).isEmpty =>
      for {
        base <- selectPath(scope, buf, fa.getExpression)
        tpe  <- JavaTyper.typer(jcType(fa))
      } yield p.Term.Select(base.root, base.steps :+ p.PathStep.Field(fa.getIdentifier.toString), tpe)
    case other => mapExpr(scope, buf, other).map(bind(scope, buf, _)).flatMap(x => asSelect(p.Expr.Alias(x)))
  }

  private def lvalue(scope: Scope, buf: ListBuffer[p.Stmt], t: ExpressionTree): Result[Lvalue] = t match {
    case aa: ArrayAccessTree =>
      for {
        arr  <- mapExpr(scope, buf, aa.getExpression).map(bind(scope, buf, _)).flatMap(t => asSelect(p.Expr.Alias(t)))
        idx  <- mapExpr(scope, buf, aa.getIndex).map(bind(scope, buf, _))
        comp <- JavaTyper.typer(jcType(aa))
      } yield Lvalue.Elem(arr, idx, comp)
    case _ =>
      staticFieldOf(t) match {
        case Some(v) if v.getConstValue == null =>
          staticSlot(scope, v).map((n, tpe) => Lvalue.Elem(p.Term.Select(n, Nil, n.tpe), p.Term.IntS32Const(0), tpe))
        case Some(v) => s"Cannot assign to the compile-time constant `${fullName(v)}`".fail
        case None    => selectPath(scope, buf, t).map(Lvalue.Slot(_))
      }
  }

  private def nullAgainst(other: p.Type, self: p.Term): Result[p.Term] =
    if (self.tpe != p.Type.Nothing) self.success
    else
      other match {
        case p.Type.Ptr(comp, space) => p.Term.NullPtrConst(comp, space, p.Region.Opaque).success
        case p.Type.Nothing          => "Cannot compare `null` against `null`".fail
        case other                   => s"Cannot compare `null` against a value of type ${other.repr}".fail
      }

  private def isReference(t: p.Type): Boolean = t.isInstanceOf[p.Type.Ptr] || t == p.Type.Nothing

  // reference == is identity, which PolyAST only expresses over integers, so both sides become addresses
  private def referenceCompare(
      scope: Scope,
      buf: ListBuffer[p.Stmt],
      l: p.Term,
      r: p.Term
  ): Result[(p.Term, p.Term)] = for {
    l0 <- nullAgainst(r.tpe, l)
    r0 <- nullAgainst(l.tpe, r)
    _ <-
      if (l0.tpe.isInstanceOf[p.Type.Ptr] && r0.tpe.isInstanceOf[p.Type.Ptr]) ().success
      else s"Cannot compare ${l0.tpe.repr} against ${r0.tpe.repr} by reference".fail
  } yield (coerce(scope, buf, l0, p.Type.IntS64), coerce(scope, buf, r0, p.Type.IntS64))

  private def promoteOperands(
      scope: Scope,
      buf: ListBuffer[p.Stmt],
      kind: Tree.Kind,
      l: p.Term,
      r: p.Term,
      rtn: p.Type
  ): Result[(p.Term, p.Term)] = kind match {
    case Tree.Kind.CONDITIONAL_AND | Tree.Kind.CONDITIONAL_OR                                 => (l, r).success
    case Tree.Kind.AND | Tree.Kind.OR | Tree.Kind.XOR if isBoolean(l.tpe) && isBoolean(r.tpe) => (l, r).success
    case Tree.Kind.EQUAL_TO | Tree.Kind.NOT_EQUAL_TO | Tree.Kind.LESS_THAN | Tree.Kind.GREATER_THAN |
        Tree.Kind.LESS_THAN_EQUAL | Tree.Kind.GREATER_THAN_EQUAL =>
      if (isReference(l.tpe) || isReference(r.tpe)) referenceCompare(scope, buf, l, r)
      else if (isBoolean(l.tpe) || isBoolean(r.tpe)) (l, r).success
      else {
        val target = promotedType(l.tpe, r.tpe)
        (coerce(scope, buf, l, target), coerce(scope, buf, r, target)).success
      }
    case _ => (coerce(scope, buf, l, rtn), coerce(scope, buf, r, rtn)).success
  }

  private def binaryOp(kind: Tree.Kind, l: p.Term, r: p.Term, rtn: p.Type): Result[p.Expr] = {
    def intr(i: p.Intr) = p.Expr.IntrOp(i).success
    val boolOperands    = l.tpe == p.Type.Bool1
    kind match {
      case Tree.Kind.PLUS                 => intr(p.Intr.Add(l, r, rtn))
      case Tree.Kind.MINUS                => intr(p.Intr.Sub(l, r, rtn))
      case Tree.Kind.MULTIPLY             => intr(p.Intr.Mul(l, r, rtn))
      case Tree.Kind.DIVIDE               => intr(p.Intr.Div(l, r, rtn))
      case Tree.Kind.REMAINDER            => intr(p.Intr.Rem(l, r, rtn))
      case Tree.Kind.LEFT_SHIFT           => intr(p.Intr.BSL(l, r, rtn))
      case Tree.Kind.RIGHT_SHIFT          => intr(p.Intr.BSR(l, r, rtn))
      case Tree.Kind.UNSIGNED_RIGHT_SHIFT => intr(p.Intr.BZSR(l, r, rtn))
      case Tree.Kind.AND if boolOperands  => intr(p.Intr.LogicAnd(l, r))
      case Tree.Kind.OR if boolOperands   => intr(p.Intr.LogicOr(l, r))
      case Tree.Kind.AND                  => intr(p.Intr.BAnd(l, r, rtn))
      case Tree.Kind.OR                   => intr(p.Intr.BOr(l, r, rtn))
      case Tree.Kind.XOR                  => intr(p.Intr.BXor(l, r, rtn))
      case Tree.Kind.CONDITIONAL_AND      => intr(p.Intr.LogicAnd(l, r))
      case Tree.Kind.CONDITIONAL_OR       => intr(p.Intr.LogicOr(l, r))
      case Tree.Kind.EQUAL_TO             => intr(p.Intr.LogicEq(l, r))
      case Tree.Kind.NOT_EQUAL_TO         => intr(p.Intr.LogicNeq(l, r))
      case Tree.Kind.LESS_THAN            => intr(p.Intr.LogicLt(l, r))
      case Tree.Kind.GREATER_THAN         => intr(p.Intr.LogicGt(l, r))
      case Tree.Kind.LESS_THAN_EQUAL      => intr(p.Intr.LogicLte(l, r))
      case Tree.Kind.GREATER_THAN_EQUAL   => intr(p.Intr.LogicGte(l, r))
      case other                          => s"Unsupported binary operator $other".fail
    }
  }

  private def arrayLength(scope: Scope, buf: ListBuffer[p.Stmt], t: ExpressionTree): Result[p.Term] =
    scope.arrayLengths.get(symOf(t)) match {
      case Some(len) => len.success
      case None =>
        val row = t match {
          case aa: ArrayAccessTree =>
            scope.rowLengths.get(symOf(aa.getExpression)).map {
              case RowLength.Uniform(len) => len.success
              case RowLength.PerRow(lens) =>
                mapExpr(scope, buf, aa.getIndex)
                  .map(i => bind(scope, buf, p.Expr.Index(lens, bind(scope, buf, i), p.Type.IntS32)))
            }
          case _ => None
        }
        row
          .orElse(
            scope.paramOf.get(symOf(t)).filterNot(_ => scope.planned).map { i =>
              scope.measured += i
              (p.Term.IntS32Const(0): p.Term).success
            }
          )
          .getOrElse(
            (s"Cannot take `length` of `$t`: only a captured or locally allocated array, or a row " +
              "indexed straight out of one, carries a length").fail
          )
    }

  private def mapExpr(scope: Scope, buf: ListBuffer[p.Stmt], t: ExpressionTree): Result[p.Expr] = t match {
    case l: LiteralTree         => literal(l)
    case par: ParenthesizedTree => mapExpr(scope, buf, par.getExpression)
    case id: IdentifierTree     => ident(scope, id)
    case fa: MemberSelectTree
        if fa.getIdentifier.toString == "length" && jcType(fa.getExpression).getKind == TypeKind.ARRAY =>
      arrayLength(scope, buf, fa.getExpression).map(p.Expr.Alias(_))
    case fa: MemberSelectTree if staticFieldOf(fa).isDefined => staticRead(scope, staticFieldOf(fa).get)
    case fa: MemberSelectTree =>
      for {
        recv <- mapExpr(scope, buf, fa.getExpression).map(bind(scope, buf, _))
        tpe  <- JavaTyper.typer(jcType(fa))
        sel  <- asSelect(p.Expr.Alias(recv))
      } yield p.Expr.Alias(p.Term.Select(sel.root, sel.steps :+ p.PathStep.Field(fa.getIdentifier.toString), tpe))
    case aa: ArrayAccessTree =>
      for {
        arr  <- mapExpr(scope, buf, aa.getExpression).map(bind(scope, buf, _))
        idx  <- mapExpr(scope, buf, aa.getIndex).map(bind(scope, buf, _))
        comp <- JavaTyper.typer(jcType(aa))
      } yield p.Expr.Index(arr, idx, comp)
    case u: UnaryTree => unary(scope, buf, u)
    case b: BinaryTree if b.getKind == Tree.Kind.CONDITIONAL_AND || b.getKind == Tree.Kind.CONDITIONAL_OR =>
      val isAnd = b.getKind == Tree.Kind.CONDITIONAL_AND
      for {
        l <- mapExpr(scope, buf, b.getLeftOperand).map(bind(scope, buf, _))
        res                   = scope.fresh(p.Type.Bool1)
        resSel: p.Term.Select = p.Term.Select(res, Nil, p.Type.Bool1)
        _                     = buf += p.Stmt.Var(res, Some(p.Expr.Alias(l)), isMutable = true)
        guard                 = if (isAnd) resSel else bind(scope, buf, p.Expr.IntrOp(p.Intr.LogicNot(resSel)))
        rhsBuf                = ListBuffer.empty[p.Stmt]
        r <- mapExpr(scope, rhsBuf, b.getRightOperand).map(bind(scope, rhsBuf, _))
        _ = rhsBuf += p.Stmt.Mut(resSel, p.Expr.Alias(r))
        _ = buf += p.Stmt.Cond(guard, rhsBuf.toList, Nil)
      } yield p.Expr.Alias(resSel)
    // string + is still a binary node after Lower, so name it rather than blame the first operand
    case b: BinaryTree if b.getKind == Tree.Kind.PLUS && isStringTpe(jcType(b)) =>
      "Unsupported string concatenation; PolyRegion has no device-side String".fail
    case b: BinaryTree =>
      for {
        l0       <- mapExpr(scope, buf, b.getLeftOperand).map(bind(scope, buf, _))
        r0       <- mapExpr(scope, buf, b.getRightOperand).map(bind(scope, buf, _))
        rtn      <- JavaTyper.typer(jcType(b))
        operands <- promoteOperands(scope, buf, b.getKind, l0, r0, rtn)
        (l, r) = operands
        e <- binaryOp(b.getKind, l, r, rtn)
      } yield e
    case c: ConditionalExpressionTree => ternary(scope, buf, c)
    case tc: TypeCastTree =>
      for {
        x  <- mapExpr(scope, buf, tc.getExpression).map(bind(scope, buf, _))
        as <- JavaTyper.typer(jcType(tc))
      } yield p.Expr.Cast(x, as)
    case na: NewArrayTree =>
      JavaTyper.typer(jcType(na)).flatMap {
        case p.Type.Ptr(comp, space) =>
          val dims = na.getDimensions.asScala.toList
          Option(na.getInitializers).map(_.asScala.toList) match {
            case Some(inits) =>
              scope.pendingArrayLength = Some(p.Term.IntS32Const(inits.size))
              val slot = bind(scope, buf, p.Expr.Alloc(comp, p.Term.IntS32Const(inits.size), space, p.Region.Opaque))
              asSelect(p.Expr.Alias(slot)).flatMap { sel =>
                inits.zipWithIndex
                  .traverse_ { (init, i) =>
                    mapExpr(scope, buf, init).map { e =>
                      buf += p.Stmt.Update(sel, p.Term.IntS32Const(i), coerce(scope, buf, bind(scope, buf, e), comp))
                      ()
                    }
                  }
                  .map(_ => p.Expr.Alias(slot))
              }
            case None =>
              dims
                .traverse(d => mapExpr(scope, buf, d).map(n => coerce(scope, buf, bind(scope, buf, n), p.Type.IntS32)))
                .flatMap {
                  case Nil => s"Array creation needs at least one dimension, got `$na`".fail
                  case len :: Nil =>
                    scope.pendingArrayLength = Some(len)
                    allocZeroed(scope, buf, comp, space, len).map(p.Expr.Alias(_))
                  case len :: rest =>
                    scope.pendingArrayLength = Some(len)
                    scope.pendingRowLength = Some(RowLength.Uniform(rest.head))
                    allocRows(scope, buf, comp, space, len, rest).map(p.Expr.Alias(_))
                }
          }
        case other => s"Cannot allocate an array of ${other.repr}".fail
      }
    case se: SwitchExpressionTree   => switchExpression(scope, buf, se)
    case a: AssignmentTree          => assign(scope, buf, a)
    case ca: CompoundAssignmentTree => compoundAssign(scope, buf, ca)
    case mi: MethodInvocationTree   => invoke(scope, buf, mi)
    case nc: NewClassTree           => construct(scope, buf, nc)
    case _: InstanceOfTree =>
      "Unsupported `instanceof`; PolyRegion has no device heap to carry a runtime type".fail
    // a pattern instanceof is a binding expression with no public kind, asking for it raises an assertion
    case le: JCTree.LetExpr =>
      if (LambdaOutliner.collectTree(le) { case _: InstanceOfTree => () :: Nil; case _ => Nil }.nonEmpty)
        "Unsupported `instanceof` pattern; PolyRegion has no device heap to carry a runtime type".fail
      else "Unsupported binding expression introduced by desugaring".fail
    case other => s"Unsupported expression: ${other.getKind} `$other`".fail
  }

  private def ident(scope: Scope, id: IdentifierTree): Result[p.Expr] = {
    val sym = symOf(id)
    scope.locals.get(sym) match {
      case Some(named) => selectExpr(Nil, named).success
      case None =>
        if (id.getName.toString == "this")
          scope.thisOpt.map(selectExpr(Nil, _)).failIfEmpty("`this` referenced outside an instance method")
        else
          (sym, scope.thisOpt) match {
            case (v: JSymbol.VarSymbol, _) if v.isStatic && v.owner.isInstanceOf[JSymbol.ClassSymbol] =>
              staticRead(scope, v)
            case (v: JSymbol.VarSymbol, Some(self)) if !v.isStatic && v.owner.isInstanceOf[JSymbol.ClassSymbol] =>
              JavaTyper
                .typer(jcType(id))
                .map(tpe => p.Expr.Alias(p.Term.Select(self, List(p.PathStep.Field(id.getName.toString)), tpe)))
            // TODO module-style references
            case _ => s"Unresolved identifier `${id.getName}` (symbol $sym)".fail
          }
    }
  }

  private def unary(scope: Scope, buf: ListBuffer[p.Stmt], u: UnaryTree): Result[p.Expr] = {
    val isInc    = u.getKind == Tree.Kind.PREFIX_INCREMENT || u.getKind == Tree.Kind.POSTFIX_INCREMENT
    val isPrefix = u.getKind == Tree.Kind.PREFIX_INCREMENT || u.getKind == Tree.Kind.PREFIX_DECREMENT
    // x++ yields the old value, so spill it before the store
    def spill(t: p.Term): p.Term = {
      val n = scope.fresh(t.tpe)
      buf += p.Stmt.Var(n, Some(p.Expr.Alias(t)))
      p.Term.Select(n, Nil, n.tpe)
    }
    def step(cur: p.Term, tpe: p.Type): Result[p.Term] =
      oneOf(tpe).map { one =>
        bind(scope, buf, p.Expr.IntrOp(if (isInc) p.Intr.Add(cur, one, tpe) else p.Intr.Sub(cur, one, tpe)))
      }
    def incDec: Result[p.Expr] =
      for {
        lv <- lvalue(scope, buf, u.getExpression)
        old = spill(readLvalue(scope, buf, lv))
        next <- step(old, lvalueType(lv))
      } yield {
        writeLvalue(buf, lv, next)
        p.Expr.Alias(if (isPrefix) next else old)
      }
    u.getKind match {
      case Tree.Kind.UNARY_MINUS =>
        for {
          x   <- mapExpr(scope, buf, u.getExpression).map(bind(scope, buf, _))
          rtn <- JavaTyper.typer(jcType(u))
        } yield p.Expr.IntrOp(p.Intr.Neg(coerce(scope, buf, x, rtn), rtn))
      case Tree.Kind.UNARY_PLUS =>
        for {
          x   <- mapExpr(scope, buf, u.getExpression).map(bind(scope, buf, _))
          rtn <- JavaTyper.typer(jcType(u))
        } yield p.Expr.IntrOp(p.Intr.Pos(coerce(scope, buf, x, rtn), rtn))
      case Tree.Kind.LOGICAL_COMPLEMENT =>
        mapExpr(scope, buf, u.getExpression).map(x => p.Expr.IntrOp(p.Intr.LogicNot(bind(scope, buf, x))))
      case Tree.Kind.BITWISE_COMPLEMENT =>
        for {
          x   <- mapExpr(scope, buf, u.getExpression).map(bind(scope, buf, _))
          rtn <- JavaTyper.typer(jcType(u))
        } yield p.Expr.IntrOp(p.Intr.BNot(coerce(scope, buf, x, rtn), rtn))
      case Tree.Kind.PREFIX_INCREMENT | Tree.Kind.POSTFIX_INCREMENT | Tree.Kind.PREFIX_DECREMENT |
          Tree.Kind.POSTFIX_DECREMENT =>
        incDec
      case other => s"Unsupported unary operator $other".fail
    }
  }

  private def ternary(scope: Scope, buf: ListBuffer[p.Stmt], c: ConditionalExpressionTree): Result[p.Expr] =
    for {
      cond <- mapExpr(scope, buf, c.getCondition).map(bind(scope, buf, _))
      tpe  <- JavaTyper.typer(jcType(c))
      result  = scope.fresh(tpe)
      _       = buf += p.Stmt.Var(result, None, isMutable = true)
      thenBuf = ListBuffer.empty[p.Stmt]
      tThen <- mapExpr(scope, thenBuf, c.getTrueExpression).map(bind(scope, thenBuf, _))
      _       = thenBuf += p.Stmt.Mut(selectTerm(Nil, result), p.Expr.Alias(coerce(scope, thenBuf, tThen, tpe)))
      elseBuf = ListBuffer.empty[p.Stmt]
      tElse <- mapExpr(scope, elseBuf, c.getFalseExpression).map(bind(scope, elseBuf, _))
      _ = elseBuf += p.Stmt.Mut(selectTerm(Nil, result), p.Expr.Alias(coerce(scope, elseBuf, tElse, tpe)))
      _ = buf += p.Stmt.Cond(cond, thenBuf.toList, elseBuf.toList)
    } yield selectExpr(Nil, result)

  private def assign(scope: Scope, buf: ListBuffer[p.Stmt], a: AssignmentTree): Result[p.Expr] =
    for {
      lv  <- lvalue(scope, buf, a.getVariable)
      rhs <- mapExpr(scope, buf, a.getExpression).map(bind(scope, buf, _))
      _ <- (lvalueType(lv), rhs.tpe) match {
        case (l: p.Type.Struct, r: p.Type.Struct) if l != r =>
          s"Cannot assign a ${r.repr} to a ${l.repr}: an object here is stored by value".fail
        case _ => ().success
      }
    } yield {
      writeLvalue(buf, lv, coerce(scope, buf, rhs, lvalueType(lv)))
      p.Expr.Alias(p.Term.Unit0Const)
    }

  private def compoundAssign(scope: Scope, buf: ListBuffer[p.Stmt], ca: CompoundAssignmentTree): Result[p.Expr] = {
    val binKind = ca.getKind match {
      case Tree.Kind.PLUS_ASSIGNMENT                 => Tree.Kind.PLUS
      case Tree.Kind.MINUS_ASSIGNMENT                => Tree.Kind.MINUS
      case Tree.Kind.MULTIPLY_ASSIGNMENT             => Tree.Kind.MULTIPLY
      case Tree.Kind.DIVIDE_ASSIGNMENT               => Tree.Kind.DIVIDE
      case Tree.Kind.REMAINDER_ASSIGNMENT            => Tree.Kind.REMAINDER
      case Tree.Kind.AND_ASSIGNMENT                  => Tree.Kind.AND
      case Tree.Kind.OR_ASSIGNMENT                   => Tree.Kind.OR
      case Tree.Kind.XOR_ASSIGNMENT                  => Tree.Kind.XOR
      case Tree.Kind.LEFT_SHIFT_ASSIGNMENT           => Tree.Kind.LEFT_SHIFT
      case Tree.Kind.RIGHT_SHIFT_ASSIGNMENT          => Tree.Kind.RIGHT_SHIFT
      case Tree.Kind.UNSIGNED_RIGHT_SHIFT_ASSIGNMENT => Tree.Kind.UNSIGNED_RIGHT_SHIFT
      case other                                     => other
    }
    // JLS 15.26.2: `E1 op= E2` is `E1 = (T)(E1 op E2)`
    def combine(current: p.Term, target: p.Type): Result[p.Term] =
      for {
        rhs0 <- mapExpr(scope, buf, ca.getExpression).map(bind(scope, buf, _))
        promoted = if (isBoolean(current.tpe)) current.tpe else promotedType(current.tpe, rhs0.tpe)
        operands <- promoteOperands(scope, buf, binKind, current, rhs0, promoted)
        (l, r) = operands
        op <- binaryOp(binKind, l, r, promoted)
      } yield coerce(scope, buf, bind(scope, buf, op), target)

    for {
      lv <- lvalue(scope, buf, ca.getVariable)
      v  <- combine(readLvalue(scope, buf, lv), lvalueType(lv))
    } yield {
      writeLvalue(buf, lv, v)
      p.Expr.Alias(p.Term.Unit0Const)
    }
  }

  private val boxTypes = Set(
    "java.lang.Integer",
    "java.lang.Long",
    "java.lang.Short",
    "java.lang.Byte",
    "java.lang.Character",
    "java.lang.Boolean",
    "java.lang.Float",
    "java.lang.Double"
  )
  private val unboxMethods = Set(
    "intValue",
    "longValue",
    "shortValue",
    "byteValue",
    "charValue",
    "booleanValue",
    "floatValue",
    "doubleValue"
  )

  private def boxIdentity(scope: Scope, buf: ListBuffer[p.Stmt], mi: MethodInvocationTree): Option[Result[p.Expr]] =
    mi.getMethodSelect match {
      case fa: MemberSelectTree =>
        val sym   = fa.asInstanceOf[JCTree.JCFieldAccess].sym
        val name  = Option(sym).fold("")(_.name.toString)
        val owner = Option(sym).map(ownerName)
        if (name == "valueOf" && owner.exists(boxTypes.contains) && mi.getArguments.size == 1)
          Some(mapExpr(scope, buf, mi.getArguments.get(0)))
        else if (unboxMethods.contains(name) && owner.exists(boxTypes.contains))
          Some(mapExpr(scope, buf, fa.getExpression))
        else None
      case _ => None
    }

  private val mathOwners = Set("java.lang.Math", "java.lang.StrictMath")
  private val EnumOwner  = "java.lang.Enum"

  private def isRecordAccessor(sym: JSymbol.MethodSymbol): Boolean = sym.owner match {
    case c: JSymbol.ClassSymbol if c.getKind == ElementKind.RECORD =>
      c.getEnclosedElements.asScala.exists(e => e.getKind == ElementKind.FIELD && e.getSimpleName == sym.name)
    case _ => false
  }

  private def zeroOf(tpe: p.Type): Result[p.Term] = tpe match {
    case p.Type.Bool1   => p.Term.Bool1Const(false).success
    case p.Type.IntS8   => p.Term.IntS8Const(0).success
    case p.Type.IntS16  => p.Term.IntS16Const(0).success
    case p.Type.IntU16  => p.Term.IntU16Const(0.toChar).success
    case p.Type.IntS32  => p.Term.IntS32Const(0).success
    case p.Type.IntS64  => p.Term.IntS64Const(0L).success
    case p.Type.Float32 => p.Term.Float32Const(0f).success
    case p.Type.Float64 => p.Term.Float64Const(0d).success
    case other          => s"No zero for ${other.repr}".fail
  }

  private def countedLoop(scope: Scope, buf: ListBuffer[p.Stmt], len: p.Term)(
      fill: (ListBuffer[p.Stmt], p.Term.Select) => Result[Unit]
  ): Result[Unit] = {
    val idx                    = scope.fresh(p.Type.IntS32)
    val idxSel: p.Term.Select  = p.Term.Select(idx, Nil, p.Type.IntS32)
    val more                   = scope.fresh(p.Type.Bool1)
    val moreSel: p.Term.Select = p.Term.Select(more, Nil, p.Type.Bool1)
    buf += p.Stmt.Var(idx, Some(p.Expr.Alias(p.Term.IntS32Const(0))), isMutable = true)
    buf += p.Stmt.Var(more, Some(p.Expr.IntrOp(p.Intr.LogicLt(idxSel, len))), isMutable = true)
    val bodyBuf = ListBuffer.empty[p.Stmt]
    fill(bodyBuf, idxSel).map { _ =>
      bodyBuf += p.Stmt.Mut(idxSel, p.Expr.IntrOp(p.Intr.Add(idxSel, p.Term.IntS32Const(1), p.Type.IntS32)))
      bodyBuf += p.Stmt.Mut(moreSel, p.Expr.IntrOp(p.Intr.LogicLt(idxSel, len)))
      buf += p.Stmt.While(moreSel, bodyBuf.toList)
      ()
    }
  }

  // new T[n] is zero-filled in Java but Alloc is not
  private def allocZeroed(
      scope: Scope,
      buf: ListBuffer[p.Stmt],
      comp: p.Type,
      space: p.Type.Space,
      len: p.Term
  ): Result[p.Term] = zeroOf(comp).flatMap { zero =>
    val slot                   = scope.fresh(p.Type.Ptr(comp, space))
    val slotSel: p.Term.Select = p.Term.Select(slot, Nil, slot.tpe)
    buf += p.Stmt.Var(slot, Some(p.Expr.Alloc(comp, len, space, p.Region.Opaque)))
    countedLoop(scope, buf, len) { (body, i) =>
      body += p.Stmt.Update(slotSel, i, zero)
      ().success
    }.map(_ => slotSel)
  }

  private def mathIntrinsic(sym: JSymbol.MethodSymbol, args: List[p.Term], rtn: p.Type): Result[p.Expr] = {
    def math(m: p.Math) = p.Expr.MathOp(m).success
    def intr(i: p.Intr) = p.Expr.IntrOp(i).success
    (sym.name.toString, args) match {
      case ("sin", x :: Nil)        => math(p.Math.Sin(x, rtn))
      case ("cos", x :: Nil)        => math(p.Math.Cos(x, rtn))
      case ("tan", x :: Nil)        => math(p.Math.Tan(x, rtn))
      case ("asin", x :: Nil)       => math(p.Math.Asin(x, rtn))
      case ("acos", x :: Nil)       => math(p.Math.Acos(x, rtn))
      case ("atan", x :: Nil)       => math(p.Math.Atan(x, rtn))
      case ("sinh", x :: Nil)       => math(p.Math.Sinh(x, rtn))
      case ("cosh", x :: Nil)       => math(p.Math.Cosh(x, rtn))
      case ("tanh", x :: Nil)       => math(p.Math.Tanh(x, rtn))
      case ("signum", x :: Nil)     => math(p.Math.Signum(x, rtn))
      case ("abs", x :: Nil)        => math(p.Math.Abs(x, rtn))
      case ("round", x :: Nil)      => math(p.Math.Round(x, rtn))
      case ("ceil", x :: Nil)       => math(p.Math.Ceil(x, rtn))
      case ("floor", x :: Nil)      => math(p.Math.Floor(x, rtn))
      case ("rint", x :: Nil)       => math(p.Math.Rint(x, rtn))
      case ("sqrt", x :: Nil)       => math(p.Math.Sqrt(x, rtn))
      case ("cbrt", x :: Nil)       => math(p.Math.Cbrt(x, rtn))
      case ("exp", x :: Nil)        => math(p.Math.Exp(x, rtn))
      case ("expm1", x :: Nil)      => math(p.Math.Expm1(x, rtn))
      case ("log", x :: Nil)        => math(p.Math.Log(x, rtn))
      case ("log1p", x :: Nil)      => math(p.Math.Log1p(x, rtn))
      case ("log10", x :: Nil)      => math(p.Math.Log10(x, rtn))
      case ("pow", x :: y :: Nil)   => math(p.Math.Pow(x, y, rtn))
      case ("atan2", x :: y :: Nil) => math(p.Math.Atan2(x, y, rtn))
      case ("hypot", x :: y :: Nil) => math(p.Math.Hypot(x, y, rtn))
      case ("min", x :: y :: Nil)   => intr(p.Intr.Min(x, y, rtn))
      case ("max", x :: y :: Nil)   => intr(p.Intr.Max(x, y, rtn))
      case (other, xs)              => s"Unsupported ${ownerName(sym)}.$other with ${xs.size} argument(s)".fail
    }
  }

  // closureOf lowers captured arguments as it goes, so it cannot serve as a predicate
  private def isClosureTree(scope: Scope, t: ExpressionTree): Boolean = t match {
    case par: ParenthesizedTree   => isClosureTree(scope, par.getExpression)
    case tc: TypeCastTree         => isClosureTree(scope, tc.getExpression)
    case id: IdentifierTree       => scope.closures.contains(symOf(id))
    case mi: MethodInvocationTree => dynamicSym(mi).flatMap(metafactoryImpl).isDefined
    case _                        => false
  }

  private def closureOf(
      scope: Scope,
      buf: ListBuffer[p.Stmt],
      t: ExpressionTree
  ): Option[Result[Closure]] = t match {
    case par: ParenthesizedTree => closureOf(scope, buf, par.getExpression)
    case tc: TypeCastTree       => closureOf(scope, buf, tc.getExpression)
    case id: IdentifierTree     => scope.closures.get(symOf(id)).map(_.success)
    case mi: MethodInvocationTree =>
      dynamicSym(mi).flatMap(metafactoryImpl).map { impl =>
        argValues(scope, buf, mi.getArguments.asScala.toList).map(Closure(impl, _))
      }
    case _ => None
  }

  // no implementation is visible for a functional value made outside the region, the JVM spins its class at run time
  private def capturedClosures(deps: DepCollector, entry: MethodTree): Map[JSymbol, Closure] = {
    val entrySym = entry.asInstanceOf[JCTree.JCMethodDecl].sym
    // $deserializeLambda$ holds a second call site per lambda, unrelated to where the value was made
    val trees = deps.localTrees.values.filter(_.getName.toString != "$deserializeLambda$").toList
    val sites = trees.flatMap { m =>
      LambdaOutliner.collectTree(m) {
        case mi: MethodInvocationTree if dynamicSym(mi).flatMap(metafactoryImpl).contains(entrySym) => mi :: Nil
        case _                                                                                      => Nil
      }
    }
    sites match {
      case site :: Nil =>
        val actuals = site.getArguments.asScala.toList.drop(if (entrySym.isStatic) 0 else 1)
        val params  = entry.getParameters.asScala.toList
        if (actuals.sizeCompare(params) != 0) Map.empty
        else
          params
            .zip(actuals)
            .flatMap {
              case (param, id: IdentifierTree) => soleClosureWrite(trees, symOf(id)).map(symOf(param) -> _)
              case _                           => None
            }
            .toMap
      case _ => Map.empty
    }
  }

  private def soleClosureWrite(trees: List[MethodTree], v: JSymbol): Option[Closure] =
    if (!v.isInstanceOf[JSymbol.VarSymbol]) None
    else {
      val writes = trees.flatMap { m =>
        LambdaOutliner.collectTree(m) {
          case d: VariableTree if symOf(d) == v               => Option(d.getInitializer).toList
          case a: AssignmentTree if symOf(a.getVariable) == v => a.getExpression :: Nil
          case _                                              => Nil
        }
      }
      writes match {
        case (mi: MethodInvocationTree) :: Nil if mi.getArguments.isEmpty =>
          dynamicSym(mi).flatMap(metafactoryImpl).map(Closure(_, Nil))
        case _ => None
      }
    }

  private def argValues(
      scope: Scope,
      buf: ListBuffer[p.Stmt],
      args: List[? <: ExpressionTree]
  ): Result[List[Either[p.Term, Closure]]] = args.traverse { a =>
    closureOf(scope, buf, a) match {
      case Some(r) => r.map(Right(_))
      case None    => mapExpr(scope, buf, a).map(e => Left(bind(scope, buf, e)))
    }
  }

  private def applyClosure(
      scope: Scope,
      buf: ListBuffer[p.Stmt],
      cl: Closure,
      samArgs: List[Either[p.Term, Closure]]
  ): Result[p.Expr] = {
    val impl   = cl.impl
    val values = cl.captures ++ samArgs
    for {
      rtn <- JavaTyper.typer(impl.getReturnType)
      recvAndValues <-
        if (impl.isStatic) (Option.empty[p.Term], values).success
        else
          values.headOption match {
            case Some(Left(r)) => (Some(r), values.tail).success
            case _ => s"Instance method `${fullName(impl)}` backs a functional value with no receiver capture".fail
          }
      (recv, rest) = recvAndValues
      e <- scope.deps.treeOf(impl) match {
        case Some(tree) => inlineImpl(scope, buf, impl, tree, recv, rest, rtn)
        case None       => callImpl(scope, buf, impl, recv, rest, rtn)
      }
    } yield {
      val named = scope.fresh(rtn)
      buf += p.Stmt.Var(named, Some(e))
      selectExpr(Nil, named)
    }
  }

  private def callImpl(
      scope: Scope,
      buf: ListBuffer[p.Stmt],
      impl: JSymbol.MethodSymbol,
      recv: Option[p.Term],
      values: List[Either[p.Term, Closure]],
      rtn: p.Type
  ): Result[p.Expr] =
    values
      .traverse {
        case Left(t) => t.success
        case Right(c) =>
          (s"`${fullName(impl)}` backs a functional value that closes over another functional " +
            s"value, but its body is not available to inline").fail
      }
      .flatMap { terms =>
        impl.asType.getParameterTypes.asScala.toList.traverse(JavaTyper.typer).flatMap { paramTpes =>
          val args = if (paramTpes.size == terms.size) terms.zip(paramTpes).map(coerce(scope, buf, _, _)) else terms
          if (mathOwners.contains(ownerName(impl))) mathIntrinsic(impl, args, rtn)
          else if (isUserMethod(impl)) {
            scope.deps.fnSyms += impl
            scope.calls += impl
            val receiverTpeArgs = recv.map(_.tpe).collect { case p.Type.Struct(_, ts) => ts }.getOrElse(Nil)
            (lengthArgsFor(scope, impl, Nil), staticArgsFor(scope, impl)).mapN((ls, ss) =>
              p.Expr.Invoke(p.Type.FnRef(p.Sym(fullName(impl))), receiverTpeArgs, recv, args ++ ls ++ ss, rtn)
            )
          } else s"Cannot lower a functional value backed by `${fullName(impl)}`, which has no body to inline".fail
        }
      }

  private def bindInlineParams(
      scope: Scope,
      buf: ListBuffer[p.Stmt],
      params: List[? <: VariableTree],
      values: List[Either[p.Term, Closure]]
  ): Result[Unit] = params.zip(values).traverse_ {
    case (v, Right(c)) => scope.closures.put(symOf(v), c); ().success
    case (v, Left(t)) =>
      JavaTyper.typer(jcType(v)).map { tpe =>
        val named = p.Named(scope.uniqueName(v.getName.toString), tpe)
        buf += p.Stmt.Var(named, Some(p.Expr.Alias(coerce(scope, buf, t, tpe))), isMutable = true)
        scope.locals.put(symOf(v), named)
        ()
      }
  }

  private def inlineBody(
      scope: Scope,
      buf: ListBuffer[p.Stmt],
      sym: JSymbol.MethodSymbol,
      tree: MethodTree,
      values: List[Either[p.Term, Closure]],
      rtn: p.Type,
      recursed: => String,
      supplier: String,
      escaping: => String
  )(self: => Result[Option[p.Named]]): Result[p.Expr] = {
    val params   = tree.getParameters.asScala.toList
    val stmts    = tree.getBody.getStatements.asScala.toList
    val returns  = LambdaOutliner.collectTree(tree.getBody) { case r: ReturnTree => r :: Nil; case _ => Nil }
    val trailing = stmts.lastOption.collect { case r: ReturnTree => r }
    if (scope.isInlining(sym)) recursed.fail
    else if (params.size != values.size)
      s"`${fullName(sym)}` takes ${params.size} parameter(s) but $supplier supplies ${values.size}".fail
    else if (returns.size > trailing.size) escaping.fail
    else
      scope.whileInlining(sym) {
        for {
          this0 <- self
          _     <- bindInlineParams(scope, buf, params, values)
          result <- scope.withThis(this0) {
            for {
              _ <- stmts.filterNot(s => trailing.contains(s)).traverse_(mapStmt(scope, buf, _))
              r <- trailing.flatMap(r => Option(r.getExpression)) match {
                case Some(e) => mapExpr(scope, buf, e).map(v => coerce(scope, buf, bind(scope, buf, v), rtn))
                case None    => (p.Term.Unit0Const: p.Term).success
              }
            } yield r
          }
        } yield p.Expr.Alias(result)
      }
  }

  private def inlineImpl(
      scope: Scope,
      buf: ListBuffer[p.Stmt],
      impl: JSymbol.MethodSymbol,
      tree: MethodTree,
      recv: Option[p.Term],
      values: List[Either[p.Term, Closure]],
      rtn: p.Type
  ): Result[p.Expr] = inlineBody(
    scope,
    buf,
    impl,
    tree,
    values,
    rtn,
    s"Functional value backed by `${fullName(impl)}` closes over itself",
    "the functional value",
    s"Cannot inline `${fullName(impl)}`: a functional value's body may only return from its last statement"
  ) {
    recv.traverse { r =>
      asSelect(p.Expr.Alias(r)).map { _ =>
        val n = p.Named(scope.uniqueName("this$"), r.tpe)
        buf += p.Stmt.Var(n, Some(p.Expr.Alias(r)))
        n
      }
    }
  }

  // structs hold their own members only, so an inherited method cannot be called against a subclass struct
  private def inlineInherited(
      scope: Scope,
      buf: ListBuffer[p.Stmt],
      sym: JSymbol.MethodSymbol,
      tree: MethodTree,
      self: p.Named,
      args: List[p.Term],
      rtn: p.Type
  ): Result[p.Expr] = inlineBody(
    scope,
    buf,
    sym,
    tree,
    args.map(Left(_)),
    rtn,
    s"`${fullName(sym)}` is inherited and recurses into itself",
    "the call site",
    s"Cannot inline the inherited `${fullName(sym)}`: only a body whose sole `return` is its last " +
      "statement can be lowered into its caller"
  )(Some(self).success)

  private def inheritedCall(
      scope: Scope,
      buf: ListBuffer[p.Stmt],
      sym: JSymbol.MethodSymbol,
      cls: JSymbol.ClassSymbol,
      recv: p.Term,
      args: List[p.Term],
      rtn: p.Type
  ): Result[p.Expr] = {
    val target  = overrideIn(scope, cls, sym).getOrElse(sym)
    val refused = s"Cannot call `${fullName(sym)}` on `${cls.flatName}` allocated in the region"
    (recv, scope.deps.treeOf(target)) match {
      case (p.Term.Select(root, Nil, _), Some(tree)) =>
        inlineInherited(scope, buf, target, tree, root, args, rtn).leftMap(e =>
          RuntimeException(s"$refused: ${e.getMessage}")
        )
      case (_, None) =>
        s"$refused: its body is not available to lower into the caller".fail
      case _ =>
        s"$refused: only a receiver held in a local can carry an inherited call".fail
    }
  }

  private def zeroFields(
      scope: Scope,
      buf: ListBuffer[p.Stmt],
      elem: TypeElement,
      root: p.Named,
      path: List[p.PathStep] = Nil
  ): Result[Unit] =
    JavaTyper.structFields(elem).traverse_ { (name, mirror) =>
      JavaTyper.typer(mirror).flatMap { tpe =>
        val steps               = path :+ p.PathStep.Field(name)
        val slot: p.Term.Select = p.Term.Select(root, steps, tpe)
        tpe match {
          case p.Type.Ptr(c, s) =>
            buf += p.Stmt.Mut(slot, p.Expr.Alias(p.Term.NullPtrConst(c, s, p.Region.Opaque)))
            ().success
          case p.Type.Struct(_, _) =>
            mirror match {
              case d: DeclaredType =>
                d.asElement match {
                  case e: TypeElement => zeroFields(scope, buf, e, root, steps)
                  case _              => s"Cannot clear the field `$name` of ${tpe.repr}".fail
                }
              case _ => s"Cannot clear the field `$name` of ${tpe.repr}".fail
            }
          case _ =>
            zeroOf(tpe).map { z =>
              buf += p.Stmt.Mut(slot, p.Expr.Alias(z)); ()
            }
        }
      }
    }

  private def construct(scope: Scope, buf: ListBuffer[p.Stmt], nc: NewClassTree): Result[p.Expr] = {
    val jc = nc.asInstanceOf[JCTree.JCNewClass]
    for {
      tpe <- JavaTyper.typer(jcType(nc))
      struct <- tpe match {
        case s: p.Type.Struct => s.success
        case other            => s"Cannot allocate ${other.repr} with `new`".fail
      }
      ctor <- Option(jc.constructor)
        .collect { case m: JSymbol.MethodSymbol => m }
        .failIfEmpty(s"`$nc` names no resolved constructor")
      elem <- (ctor.owner match {
        case c: JSymbol.ClassSymbol => Some(c)
        case _                      => None
      }).failIfEmpty(s"`${fullName(ctor)}` is not owned by a class")
      _ <- superFieldsOf(scope, elem)
      tree <- scope.deps
        .treeOf(ctor)
        .orElse(scope.deps.ctorOf(elem, jc.args.size))
        .failIfEmpty(s"No constructor body available for `${elem.flatName}`, so `new` cannot be lowered")
      args <- jc.args.asScala.toList.traverse(mapExpr(scope, buf, _).map(bind(scope, buf, _)))
      self = p.Named(scope.uniqueName("new$"), struct)
      _    = buf += p.Stmt.Var(self, None)
      _    = scope.deps.witnessed += elem
      _ <- zeroFields(scope, buf, elem, self)
      _ <- inlineCtor(scope, buf, ctor, tree, self, args)
    } yield selectExpr(Nil, self)
  }

  private def superFieldsOf(scope: Scope, elem: JSymbol.ClassSymbol): Result[Unit] =
    Option(elem.getSuperclass).map(scope.deps.types.asElement).collect { case c: TypeElement => c } match {
      case Some(sup) if JavaTyper.instanceFields(sup).nonEmpty =>
        (s"Cannot allocate `${elem.flatName}`: its supertype `${sup.getQualifiedName}` carries instance " +
          s"state (${JavaTyper.instanceFields(sup).map(_.getSimpleName).mkString(", ")}), " +
          "which this layout has no room for").fail
      case _ => ().success
    }

  private def inlineCtor(
      scope: Scope,
      buf: ListBuffer[p.Stmt],
      ctor: JSymbol.MethodSymbol,
      tree: MethodTree,
      self: p.Named,
      args: List[p.Term]
  ): Result[Unit] = {
    val params = tree.getParameters.asScala.toList
    val stmts  = tree.getBody.getStatements.asScala.toList
    if (scope.isInlining(ctor)) s"`${fullName(ctor)}` constructs itself".fail
    else if (params.size != args.size)
      s"`${fullName(ctor)}` takes ${params.size} argument(s) but ${args.size} were supplied".fail
    else
      scope.whileInlining(ctor) {
        for {
          _ <- bindInlineParams(scope, buf, params, args.map(Left(_)))
          _ <- scope.withThis(Some(self))(stmts.traverse_(mapStmt(scope, buf, _)))
        } yield ()
      }
  }

  private def samCall(scope: Scope, buf: ListBuffer[p.Stmt], mi: MethodInvocationTree): Option[Result[p.Expr]] =
    mi.getMethodSelect match {
      case fa: MemberSelectTree =>
        closureOf(scope, buf, fa.getExpression).map { clr =>
          for {
            cl   <- clr
            args <- argValues(scope, buf, mi.getArguments.asScala.toList)
            e    <- applyClosure(scope, buf, cl, args)
          } yield e
        }
      case _ => None
    }

  private def invoke(scope: Scope, buf: ListBuffer[p.Stmt], mi: MethodInvocationTree): Result[p.Expr] =
    dynamicSym(mi) match {
      case Some(d) if metafactoryImpl(d).isDefined =>
        ("A functional value has no device representation, so it only lowers where it is called " +
          "directly in the region; it cannot be stored in a field, returned, or handed to a method " +
          "whose body is not available").fail
      case Some(d) => s"Unsupported dynamic call site `${d.name}` (bootstrapped by ${Option(d.bsm).orNull})".fail
      case None    => samCall(scope, buf, mi).getOrElse(plainInvoke(scope, buf, mi))
    }

  // the candidate set is a lower bound: a class outside this compilation can implement the method, hence the host-written tag
  private def implementationsOf(
      deps: DepCollector,
      sym: JSymbol.MethodSymbol
  ): Result[List[(TypeElement, JSymbol.MethodSymbol)]] = sym.owner match {
    case owner: JSymbol.ClassSymbol =>
      val ownerTpe = deps.types.erasure(owner.asType)
      deps.classTable.values.toList
        .filter(c => c != owner && !c.getModifiers.contains(Modifier.ABSTRACT))
        .filter(c => c.getKind == ElementKind.CLASS || c.getKind == ElementKind.ENUM || c.getKind == ElementKind.RECORD)
        .filter(c => deps.types.isSubtype(deps.types.erasure(c.asType), ownerTpe))
        .filter(_.getQualifiedName.toString.nonEmpty)
        .distinctBy(_.getQualifiedName.toString)
        .sortBy(_.getQualifiedName.toString)
        .traverse { c =>
          c.getEnclosedElements.asScala.toList
            .collectFirst {
              case m: JSymbol.MethodSymbol
                  if !m.getModifiers.contains(Modifier.ABSTRACT) && deps.elements.overrides(m, sym, c) =>
                (c, m)
            }
            .failIfEmpty(
              s"`${c.getQualifiedName}` is a candidate receiver of `${fullName(sym)}` but inherits its " +
                "implementation rather than declaring one, which this lowering cannot dispatch to"
            )
        }
    case _ => s"Cannot resolve implementations of `${fullName(sym)}`: it is not owned by a class".fail
  }

  private def emitDispatch(
      scope: Scope,
      buf: ListBuffer[p.Stmt],
      sym: JSymbol.MethodSymbol,
      recv: p.Term,
      args: List[p.Term],
      argLengths: List[Option[p.Term]],
      rtn: p.Type
  ): Result[p.Expr] = for {
    self       <- asSelect(p.Expr.Alias(recv))
    candidates <- implementationsOf(scope.deps, sym)
    _ <-
      if (candidates.nonEmpty) ().success
      else
        (s"Cannot devirtualise `${fullName(sym)}`: this compilation sees no class implementing it, " +
          "so there is nothing to compile into the kernel").fail
    // every arm shares the abstract receiver, so an implementation reading its own state would use the wrong offsets
    receivers = sym.owner match {
      case c: JSymbol.ClassSymbol => c :: candidates.map(_._1)
      case _                      => candidates.map(_._1)
    }
    _ <- receivers.traverse_ { c =>
      val fields = JavaTyper.instanceFields(c)
      if (fields.isEmpty) ().success
      else
        (s"Cannot devirtualise `${fullName(sym)}`: `${c.getSimpleName}` carries instance state " +
          s"(${fields.map(_.getSimpleName).mkString(", ")}), and only stateless implementations can " +
          "share the abstract receiver's layout").fail
    }
    _                        = scope.deps.taggedTypes += ownerName(sym)
    result                   = scope.fresh(rtn)
    resultSel: p.Term.Select = p.Term.Select(result, Nil, rtn)
    _                        = if (rtn != p.Type.Unit0) buf += p.Stmt.Var(result, None, isMutable = true)
    tag = bind(
      scope,
      buf,
      p.Expr.Alias(p.Term.Select(self.root, self.steps :+ p.PathStep.Field(JavaTyper.TagMember), p.Type.IntS32))
    )
    _ <- candidates.traverse_ { (cls, impl) =>
      val fqn = cls.getQualifiedName.toString
      scope.deps.taggedTypes += fqn
      scope.deps.fnSyms += impl
      scope.calls += impl
      val hit    = bind(scope, buf, p.Expr.IntrOp(p.Intr.LogicEq(tag, p.Term.IntS32Const(scope.deps.tagFor(fqn)))))
      val armBuf = ListBuffer.empty[p.Stmt]
      (lengthArgsFor(scope, impl, argLengths), staticArgsFor(scope, impl)).mapN { (ls, ss) =>
        val armSelf = coerce(scope, armBuf, self, p.Type.Struct(p.Sym(JavaTyper.structName(cls)), Nil))
        val call    = p.Expr.Invoke(p.Type.FnRef(p.Sym(fullName(impl))), Nil, Some(armSelf), args ++ ls ++ ss, rtn)
        if (rtn == p.Type.Unit0) armBuf += p.Stmt.Var(scope.fresh(rtn), Some(call))
        else armBuf += p.Stmt.Mut(resultSel, call)
        buf += p.Stmt.Cond(hit, armBuf.toList, Nil)
        ()
      }
    }
  } yield p.Expr.Alias(if (rtn == p.Type.Unit0) p.Term.Unit0Const else resultSel)

  private def regionAllocated(scope: Scope, tpe: p.Type): Option[JSymbol.ClassSymbol] = tpe match {
    case p.Type.Struct(name, _) =>
      val fqn = name.fqn.mkString(".")
      scope.deps.witnessed.collectFirst { case c: JSymbol.ClassSymbol if JavaTyper.structName(c) == fqn => c }
    case _ => None
  }

  private def overrideIn(
      scope: Scope,
      cls: JSymbol.ClassSymbol,
      sym: JSymbol.MethodSymbol
  ): Option[JSymbol.MethodSymbol] =
    cls.getEnclosedElements.asScala.collectFirst {
      case m: JSymbol.MethodSymbol
          if !m.getModifiers.contains(Modifier.ABSTRACT) && scope.deps.elements.overrides(m, sym, cls) =>
        m
    }

  private def plainInvoke(scope: Scope, buf: ListBuffer[p.Stmt], mi: MethodInvocationTree): Result[p.Expr] =
    boxIdentity(scope, buf, mi).getOrElse {
      for {
        symAndRecv <- mi.getMethodSelect match {
          case fa: MemberSelectTree =>
            val msym = fa.asInstanceOf[JCTree.JCFieldAccess].sym.asInstanceOf[JSymbol.MethodSymbol]
            if (msym.isStatic) (msym, Option.empty[p.Term]).success
            else mapExpr(scope, buf, fa.getExpression).map(r => (msym, Some(bind(scope, buf, r))))
          case id: IdentifierTree =>
            val msym = id.asInstanceOf[JCTree.JCIdent].sym.asInstanceOf[JSymbol.MethodSymbol]
            if (msym.isStatic) (msym, Option.empty[p.Term]).success
            else
              scope.thisOpt
                .map(self => (msym, Some(selectTerm(Nil, self): p.Term)))
                .failIfEmpty(s"Cannot resolve receiver for `${id.getName}` outside an instance method")
          case other => s"Unsupported method selector ${other.getKind}".fail
        }
        (sym, recv) = symAndRecv
        rtn <- JavaTyper.typer(jcType(mi))
        e <-
          if (mi.getArguments.asScala.exists(isClosureTree(scope, _)))
            closureCall(scope, buf, sym, recv, mi.getArguments.asScala.toList, rtn)
          else directCall(scope, buf, mi, sym, recv, rtn)
      } yield {
        val named = scope.fresh(rtn)
        buf += p.Stmt.Var(named, Some(e))
        selectExpr(Nil, named)
      }
    }

  private def closureCall(
      scope: Scope,
      buf: ListBuffer[p.Stmt],
      sym: JSymbol.MethodSymbol,
      recv: Option[p.Term],
      argTrees: List[? <: ExpressionTree],
      rtn: p.Type
  ): Result[p.Expr] = scope.deps.treeOf(sym) match {
    case Some(tree) =>
      argValues(scope, buf, argTrees).flatMap(vs => inlineImpl(scope, buf, sym, tree, recv, vs, rtn))
    case None =>
      (s"`${fullName(sym)}` is handed a functional value, which has no device representation, and " +
        "its body is not available to lower into the call site").fail
  }

  private def directCall(
      scope: Scope,
      buf: ListBuffer[p.Stmt],
      mi: MethodInvocationTree,
      sym: JSymbol.MethodSymbol,
      recv: Option[p.Term],
      rtn: p.Type
  ): Result[p.Expr] =
    for {
      paramTpes <- sym.asType.getParameterTypes.asScala.toList.traverse(JavaTyper.typer)
      argTrees = mi.getArguments.asScala.toList
      lowered <- argTrees.traverse { a =>
        scope.pendingArrayLength = None
        mapExpr(scope, buf, a).map { e =>
          val len = Option(symOf(a))
            .flatMap(scope.arrayLengths.get)
            .orElse(a match { case _: NewArrayTree => scope.pendingArrayLength; case _ => None })
          (bind(scope, buf, e), len)
        }
      }
      _   = scope.pendingArrayLength = None
      raw = lowered.map(_._1)
      _ = argTrees.zipWithIndex.foreach { (a, i) =>
        Option(symOf(a)).flatMap(scope.paramOf.get).foreach(j => scope.argFlows += ((sym, i, j)))
      }
      argLengths = lowered.map(_._2)
      args       = if (paramTpes.size == raw.size) raw.zip(paramTpes).map(coerce(scope, buf, _, _)) else raw
      e <-
        if (mathOwners.contains(ownerName(sym))) mathIntrinsic(sym, args, rtn)
        else if (sym.name.toString == "<init>")
          if (args.isEmpty) (p.Expr.Alias(p.Term.Unit0Const): p.Expr).success
          else s"Unsupported constructor delegation to `${fullName(sym)}`".fail
        else if (ownerName(sym) == EnumOwner && recv.isDefined)
          if (sym.name.toString == "ordinal" && args.isEmpty) p.Expr.Alias(recv.get).success
          else s"Unsupported `${fullName(sym)}`: an enum value travels as its ordinal alone".fail
        else if (isRecordAccessor(sym) && args.isEmpty && recv.isDefined)
          asSelect(p.Expr.Alias(recv.get))
            .map(sel => p.Expr.Alias(p.Term.Select(sel.root, sel.steps :+ p.PathStep.Field(sym.name.toString), rtn)))
        else if (recv.isDefined && sym.getModifiers.contains(Modifier.ABSTRACT))
          regionAllocated(scope, recv.get.tpe) match {
            case Some(cls) =>
              overrideIn(scope, cls, sym)
                .failIfEmpty(s"`${cls.flatName}` declares no override of `${fullName(sym)}` to call")
                .flatMap { impl =>
                  scope.deps.fnSyms += impl
                  scope.calls += impl
                  (lengthArgsFor(scope, impl, argLengths), staticArgsFor(scope, impl)).mapN((ls, ss) =>
                    p.Expr.Invoke(p.Type.FnRef(p.Sym(fullName(impl))), Nil, recv, args ++ ls ++ ss, rtn)
                  )
                }
            case None => emitDispatch(scope, buf, sym, recv.get, args, argLengths, rtn)
          }
        else
          recv.flatMap(r => regionAllocated(scope, r.tpe)).filter(_ != sym.owner) match {
            case Some(cls) => inheritedCall(scope, buf, sym, cls, recv.get, args, rtn)
            case None =>
              val receiverTpeArgs = recv.map(_.tpe).collect { case p.Type.Struct(_, ts) => ts }.getOrElse(Nil)
              if (isUserMethod(sym)) { scope.deps.fnSyms += sym; scope.calls += sym }
              (lengthArgsFor(scope, sym, argLengths), staticArgsFor(scope, sym)).mapN((ls, ss) =>
                p.Expr.Invoke(p.Type.FnRef(p.Sym(fullName(sym))), receiverTpeArgs, recv, args ++ ls ++ ss, rtn)
              )
          }
    } yield e

  private def mapStmt(scope: Scope, buf: ListBuffer[p.Stmt], t: StatementTree): Result[Unit] = t match {
    case b: BlockTree => b.getStatements.asScala.toList.traverse_(mapStmt(scope, buf, _))
    case v: VariableTree if Option(v.getInitializer).exists(isClosureTree(scope, _)) =>
      closureOf(scope, buf, v.getInitializer).get.map { c =>
        scope.closures.put(symOf(v), c)
        ()
      }
    case v: VariableTree =>
      for {
        declared <- JavaTyper.typer(jcType(v))
        _ = scope.pendingArrayLength = None
        _ = scope.pendingRowLength = None
        value <- Option(v.getInitializer).traverse(mapExpr(scope, buf, _).map(bind(scope, buf, _)))
        // a local declared at a supertype but built with new keeps the concrete type, widening drops its fields
        tpe = (declared, value.map(_.tpe)) match {
          case (_: p.Type.Struct, Some(actual: p.Type.Struct)) => actual
          case _                                               => declared
        }
        named = p.Named(scope.uniqueName(v.getName.toString), tpe)
        _     = scope.locals.put(symOf(v), named)
        init  = value.map(t => p.Expr.Alias(coerce(scope, buf, t, tpe)))
        // aliasing keeps the length: enhanced-for binds the subject to a local before measuring it
        aliased = Option(v.getInitializer).flatMap(e => Option(symOf(e)))
        _ = aliased
          .flatMap(scope.arrayLengths.get)
          .orElse(scope.pendingArrayLength)
          .foreach(scope.arrayLengths.put(symOf(v), _))
        _ = aliased
          .flatMap(scope.rowLengths.get)
          .orElse(scope.pendingRowLength)
          .foreach(scope.rowLengths.put(symOf(v), _))
        _ = aliased.flatMap(scope.paramOf.get).foreach(scope.paramOf.put(symOf(v), _))
        _ = scope.pendingArrayLength = None
        _ = scope.pendingRowLength = None
      } yield { buf += p.Stmt.Var(named, init, isMutable = true); () }
    case e: ExpressionStatementTree => mapExpr(scope, buf, e.getExpression).map(_ => ())
    case i: IfTree =>
      for {
        cond <- mapExpr(scope, buf, i.getCondition).map(bind(scope, buf, _))
        thenBuf = ListBuffer.empty[p.Stmt]
        _ <- mapStmt(scope, thenBuf, i.getThenStatement)
        elseBuf = ListBuffer.empty[p.Stmt]
        _ <- Option(i.getElseStatement).traverse(mapStmt(scope, elseBuf, _))
      } yield { buf += p.Stmt.Cond(cond, thenBuf.toList, elseBuf.toList); () }
    case w: WhileLoopTree => emitWhile(scope, buf, Option(w.getCondition), List(w.getStatement), Nil)
    case f: ForLoopTree =>
      for {
        _ <- f.getInitializer.asScala.toList.traverse_(mapStmt(scope, buf, _))
        _ <- emitWhile(scope, buf, Option(f.getCondition), List(f.getStatement), f.getUpdate.asScala.toList)
      } yield ()
    case l: LabeledStatementTree => emitLabelled(scope, buf, l)
    case d: DoWhileLoopTree      => emitDoWhile(scope, buf, d)
    case s: SwitchTree           => emitSwitch(scope, buf, s)
    case b: BreakTree =>
      Option(b.getLabel).map(_.toString) match {
        case Some(name) =>
          scope
            .labelled(name)
            .failIfEmpty(s"`break $name` has no enclosing statement labelled `$name`")
            .map { l =>
              buf += p.Stmt.Mut(l.brk, p.Expr.Alias(p.Term.Bool1Const(true)))
              buf += p.Stmt.Break
              ()
            }
        case None =>
          scope.enclosingLoop.flatMap(_.viaLabel).foreach { l =>
            buf += p.Stmt.Mut(l.brk, p.Expr.Alias(p.Term.Bool1Const(true)))
          }
          buf += p.Stmt.Break
          ().success
      }
    case c: ContinueTree =>
      Option(c.getLabel).map(_.toString) match {
        case Some(name) =>
          scope
            .labelled(name)
            .failIfEmpty(s"`continue $name` has no enclosing loop labelled `$name`")
            .map { l =>
              buf += p.Stmt.Mut(l.cnt, p.Expr.Alias(p.Term.Bool1Const(true)))
              buf += p.Stmt.Break
              ()
            }
        case None =>
          scope.enclosingLoop.failIfEmpty("`continue` outside of a loop").flatMap {
            case LoopCtx(_, _, Some(l)) =>
              buf += p.Stmt.Mut(l.cnt, p.Expr.Alias(p.Term.Bool1Const(true)))
              buf += p.Stmt.Break
              ().success
            case ctx =>
              ctx.update.traverse_(mapStmt(scope, buf, _)).map { _ =>
                buf ++= ctx.reeval
                buf += p.Stmt.Cont
                ()
              }
          }
      }
    case r: ReturnTree =>
      Option(r.getExpression) match {
        case None => buf += p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const)); ().success
        case Some(e) =>
          mapExpr(scope, buf, e).map { v =>
            val coerced = coerce(scope, buf, bind(scope, buf, v), scope.rtnType)
            buf += p.Stmt.Return(p.Expr.Alias(coerced))
            ()
          }
      }
    case y: YieldTree =>
      scope.enclosingYield.failIfEmpty("`yield` outside of a switch expression").flatMap { (target, depth) =>
        if (scope.loopDepth != depth)
          "Unsupported `yield` from inside a loop in a switch expression arm".fail
        else
          mapExpr(scope, buf, y.getValue).map { v =>
            buf += p.Stmt.Mut(target, p.Expr.Alias(coerce(scope, buf, bind(scope, buf, v), target.tpe)))
            buf += p.Stmt.Break
            ()
          }
      }
    case _: EmptyStatementTree => ().success
    case t: TryTree =>
      if (t.getCatches.isEmpty) "Unsupported `finally`; PolyRegion kernels cannot unwind".fail
      else "Unsupported `try`/`catch`; PolyRegion kernels cannot throw".fail
    // the one throw a kernel answers: Lower rewrites `assert` into a throw guarded by $assertionsDisabled
    case t: ThrowTree =>
      t.getExpression match {
        case nc: NewClassTree if isAssertionErrorTpe(jcType(nc)) =>
          assertMessage(nc).map { message =>
            bind(
              scope,
              buf,
              p.Expr.SpecOp(
                p.Spec.Assert(p.Term.IntU32Const(p.Enums.AssertCode.Assert.value), p.Term.StringConst(message))
              )
            )
            ()
          }
        case _ => "Unsupported `throw`; PolyRegion kernels cannot throw".fail
      }
    case _: SynchronizedTree => "Unsupported `synchronized`; PolyRegion kernels have no monitors".fail
    case other               => s"Unsupported statement: ${other.getKind} `$other`".fail
  }

  private def allocRows(
      scope: Scope,
      buf: ListBuffer[p.Stmt],
      comp: p.Type,
      space: p.Type.Space,
      len: p.Term,
      rest: List[p.Term]
  ): Result[p.Term] = comp match {
    case p.Type.Ptr(inner, innerSpace) =>
      val outer                   = scope.fresh(p.Type.Ptr(comp, space))
      val outerSel: p.Term.Select = p.Term.Select(outer, Nil, outer.tpe)
      buf += p.Stmt.Var(outer, Some(p.Expr.Alloc(comp, len, space, p.Region.Opaque)))
      countedLoop(scope, buf, len) { (body, i) =>
        val row = rest match {
          case rowLen :: Nil  => allocZeroed(scope, body, inner, innerSpace, rowLen).map(p.Expr.Alias(_))
          case rowLen :: more => allocRows(scope, body, inner, innerSpace, rowLen, more).map(p.Expr.Alias(_))
          case Nil            => s"Missing dimension for ${comp.repr}".fail
        }
        row.map { r =>
          val rowVar = scope.fresh(comp)
          body += p.Stmt.Var(rowVar, Some(r))
          body += p.Stmt.Update(outerSel, i, p.Term.Select(rowVar, Nil, comp))
          ()
        }
      }.map(_ => outerSel)
    case other => s"Cannot allocate rows of ${other.repr}".fail
  }

  private def propagateLabels(scope: Scope, buf: ListBuffer[p.Stmt]): Unit =
    scope.activeLabels.foreach { l =>
      val escaping = bind(scope, buf, p.Expr.IntrOp(p.Intr.LogicOr(l.brk, l.cnt)))
      buf += p.Stmt.Cond(escaping, List(p.Stmt.Break), Nil)
    }

  // a run-once loop is the only shape PolyAST's unlabelled Break can leave
  private def runOnce(scope: Scope, body: List[p.Stmt]): List[p.Stmt] = {
    val once                   = scope.fresh(p.Type.Bool1)
    val onceSel: p.Term.Select = p.Term.Select(once, Nil, p.Type.Bool1)
    List(
      p.Stmt.Var(once, Some(p.Expr.Alias(p.Term.Bool1Const(true))), isMutable = true),
      p.Stmt.While(onceSel, body :+ p.Stmt.Mut(onceSel, p.Expr.Alias(p.Term.Bool1Const(false))))
    )
  }

  private def condAnf(scope: Scope, cond: Option[ExpressionTree]): Result[(p.Term, List[p.Stmt], List[p.Stmt])] = {
    val condBuf = ListBuffer.empty[p.Stmt]
    val term = cond match {
      case Some(c) => mapExpr(scope, condBuf, c).map(bind(scope, condBuf, _))
      case None    => (p.Term.Bool1Const(true): p.Term).success
    }
    term.map { t =>
      val stmts = condBuf.toList
      (
        t,
        stmts.collect { case p.Stmt.Var(n, _, _) => p.Stmt.Var(n, None, isMutable = true) },
        stmts.collect { case p.Stmt.Var(n, Some(e), _) => p.Stmt.Mut(p.Term.Select(n, Nil, n.tpe), e) }
      )
    }
  }

  private def emitWhile(
      scope: Scope,
      buf: ListBuffer[p.Stmt],
      cond: Option[ExpressionTree],
      body: List[StatementTree],
      update: List[ExpressionStatementTree]
  ): Result[Unit] = for {
    anf <- condAnf(scope, cond)
    (condTerm, hoisted, reeval) = anf
    bodyBuf                     = ListBuffer.empty[p.Stmt]
    _ <- scope.inLoop(LoopCtx(update, reeval)) {
      for {
        _ <- body.traverse_(mapStmt(scope, bodyBuf, _))
        _ <- update.traverse_(mapStmt(scope, bodyBuf, _))
      } yield ()
    }
  } yield {
    buf ++= hoisted
    buf ++= reeval
    buf += p.Stmt.While(condTerm, bodyBuf.toList ++ reeval)
    propagateLabels(scope, buf)
    ()
  }

  private def emitLabelled(scope: Scope, buf: ListBuffer[p.Stmt], l: LabeledStatementTree): Result[Unit] = {
    val brkN               = scope.fresh(p.Type.Bool1)
    val cntN               = scope.fresh(p.Type.Bool1)
    val brk: p.Term.Select = p.Term.Select(brkN, Nil, p.Type.Bool1)
    val cnt: p.Term.Select = p.Term.Select(cntN, Nil, p.Type.Bool1)
    val ctx                = LabelCtx(l.getLabel.toString, brk, cnt)
    buf += p.Stmt.Var(brkN, Some(p.Expr.Alias(p.Term.Bool1Const(false))), isMutable = true)
    buf += p.Stmt.Var(cntN, Some(p.Expr.Alias(p.Term.Bool1Const(false))), isMutable = true)
    l.getStatement match {
      case w: WhileLoopTree => emitLabelledLoop(scope, buf, ctx, Option(w.getCondition), w.getStatement, Nil)
      case f: ForLoopTree =>
        for {
          _ <- f.getInitializer.asScala.toList.traverse_(mapStmt(scope, buf, _))
          _ <- emitLabelledLoop(scope, buf, ctx, Option(f.getCondition), f.getStatement, f.getUpdate.asScala.toList)
        } yield ()
      case other =>
        val bodyBuf = ListBuffer.empty[p.Stmt]
        scope.withLabel(ctx)(mapStmt(scope, bodyBuf, other)).map { _ =>
          buf ++= runOnce(scope, bodyBuf.toList)
          propagateLabels(scope, buf)
          ()
        }
    }
  }

  private def emitLabelledLoop(
      scope: Scope,
      buf: ListBuffer[p.Stmt],
      ctx: LabelCtx,
      cond: Option[ExpressionTree],
      body: StatementTree,
      update: List[ExpressionStatementTree]
  ): Result[Unit] = for {
    anf <- condAnf(scope, cond)
    (condTerm, hoisted, reeval) = anf
    bodyBuf                     = ListBuffer.empty[p.Stmt]
    _ <- scope.withLabel(ctx) {
      scope.inLoop(LoopCtx(update, reeval, Some(ctx)))(mapStmt(scope, bodyBuf, body))
    }
    updBuf = ListBuffer.empty[p.Stmt]
    _ <- scope.inLoop(LoopCtx(update, reeval))(update.traverse_(mapStmt(scope, updBuf, _)))
  } yield {
    val escapeBuf = ListBuffer.empty[p.Stmt]
    propagateLabels(scope, escapeBuf)
    val wrapped = (p.Stmt.Mut(ctx.cnt, p.Expr.Alias(p.Term.Bool1Const(false))) ::
      runOnce(scope, bodyBuf.toList)) :+ p.Stmt.Cond(ctx.brk, List(p.Stmt.Break), Nil)
    buf ++= hoisted
    buf ++= reeval
    buf += p.Stmt.While(condTerm, wrapped ++ escapeBuf.toList ++ updBuf.toList ++ reeval)
    propagateLabels(scope, buf)
    ()
  }

  private def emitDoWhile(scope: Scope, buf: ListBuffer[p.Stmt], d: DoWhileLoopTree): Result[Unit] = {
    val flag                   = scope.fresh(p.Type.Bool1)
    val flagSel: p.Term.Select = p.Term.Select(flag, Nil, p.Type.Bool1)
    for {
      anf <- condAnf(scope, Some(d.getCondition))
      (condTerm, hoisted, reeval) = anf
      tail                        = reeval :+ p.Stmt.Mut(flagSel, p.Expr.Alias(condTerm))
      bodyBuf                     = ListBuffer.empty[p.Stmt]
      _ <- scope.inLoop(LoopCtx(Nil, tail))(mapStmt(scope, bodyBuf, d.getStatement))
    } yield {
      buf += p.Stmt.Var(flag, Some(p.Expr.Alias(p.Term.Bool1Const(true))), isMutable = true)
      buf ++= hoisted
      buf += p.Stmt.While(flagSel, bodyBuf.toList ++ tail)
      propagateLabels(scope, buf)
      ()
    }
  }

  private def caseGuards(
      scope: Scope,
      buf: ListBuffer[p.Stmt],
      selector: p.Term,
      cases: List[CaseTree]
  ): Result[(Map[CaseTree, p.Term], p.Term)] =
    cases
      .filter(!_.getExpressions.isEmpty)
      .traverse { c =>
        c.getExpressions.asScala.toList
          .traverse(e => mapExpr(scope, buf, e).map(x => coerce(scope, buf, bind(scope, buf, x), selector.tpe)))
          .map { xs =>
            val eq = xs.map(x => bind(scope, buf, p.Expr.IntrOp(p.Intr.LogicEq(selector, x))))
            c -> eq.reduce((l, r) => bind(scope, buf, p.Expr.IntrOp(p.Intr.LogicOr(l, r))))
          }
      }
      .map { matches =>
        val anyMatch = matches.map(_._2).reduceOption((l, r) => bind(scope, buf, p.Expr.IntrOp(p.Intr.LogicOr(l, r))))
        val noMatch =
          anyMatch.fold[p.Term](p.Term.Bool1Const(true))(m => bind(scope, buf, p.Expr.IntrOp(p.Intr.LogicNot(m))))
        (matches.toMap, noMatch)
      }

  // no jump table: a run-once loop walks the cases behind a started flag, which gives colon-form fall-through
  private def emitCases(
      scope: Scope,
      buf: ListBuffer[p.Stmt],
      selector: p.Term,
      cases: List[CaseTree],
      result: Option[p.Term.Select]
  ): Result[Unit] = for {
    guards <- caseGuards(scope, buf, selector, cases)
    (matchOf, noMatch)        = guards
    started                   = scope.fresh(p.Type.Bool1)
    startedSel: p.Term.Select = p.Term.Select(started, Nil, p.Type.Bool1)
    setStarted                = p.Stmt.Mut(startedSel, p.Expr.Alias(p.Term.Bool1Const(true)))
    bodyBuf                   = ListBuffer.empty[p.Stmt]
    _ = bodyBuf += p.Stmt.Var(started, Some(p.Expr.Alias(p.Term.Bool1Const(false))), isMutable = true)
    _ <- cases.traverse_ { c =>
      bodyBuf += p.Stmt.Cond(matchOf.getOrElse(c, noMatch), List(setStarted), Nil)
      val armBuf = ListBuffer.empty[p.Stmt]
      val lowered =
        if (c.getCaseKind == CaseTree.CaseKind.RULE)
          lowerArm(scope, armBuf, c.getBody, result).map { _ =>
            armBuf += p.Stmt.Break; ()
          }
        else Option(c.getStatements).fold(Nil)(_.asScala.toList).traverse_(mapStmt(scope, armBuf, _))
      lowered.map { _ =>
        bodyBuf += p.Stmt.Cond(startedSel, armBuf.toList, Nil)
        ()
      }
    }
  } yield {
    buf ++= runOnce(scope, bodyBuf.toList)
    propagateLabels(scope, buf)
    ()
  }

  private def lowerArm(
      scope: Scope,
      buf: ListBuffer[p.Stmt],
      body: Tree,
      result: Option[p.Term.Select]
  ): Result[Unit] = (body, result) match {
    case (b: BlockTree, _) => b.getStatements.asScala.toList.traverse_(mapStmt(scope, buf, _))
    case (e: ExpressionTree, Some(target)) =>
      mapExpr(scope, buf, e).map { v =>
        buf += p.Stmt.Mut(target, p.Expr.Alias(coerce(scope, buf, bind(scope, buf, v), target.tpe)))
        ()
      }
    case (s: StatementTree, None)  => mapStmt(scope, buf, s)
    case (e: ExpressionTree, None) => mapExpr(scope, buf, e).map(_ => ())
    case (other, _)                => s"Unsupported switch arm `$other`".fail
  }

  private def switchExpression(scope: Scope, buf: ListBuffer[p.Stmt], s: SwitchExpressionTree): Result[p.Expr] =
    for {
      tpe      <- JavaTyper.typer(jcType(s))
      selector <- mapExpr(scope, buf, s.getExpression).map(bind(scope, buf, _))
      result                   = scope.fresh(tpe)
      resultSel: p.Term.Select = p.Term.Select(result, Nil, tpe)
      _                        = buf += p.Stmt.Var(result, None, isMutable = true)
      _ <- scope.withYieldTarget(resultSel)(
        emitCases(scope, buf, selector, s.getCases.asScala.toList, Some(resultSel))
      )
    } yield p.Expr.Alias(resultSel)

  private def emitSwitch(scope: Scope, buf: ListBuffer[p.Stmt], s: SwitchTree): Result[Unit] =
    for {
      selector <- mapExpr(scope, buf, s.getExpression).map(bind(scope, buf, _))
      _        <- emitCases(scope, buf, selector, s.getCases.asScala.toList, None)
    } yield ()

  def compileMethod(
      env: ProcessingEnvironment,
      trees: Trees,
      m: MethodTree,
      deps: DepCollector,
      isEntry: Boolean,
      planned: Boolean
  ): Result[(p.Function, List[JSymbol.VarSymbol])] = {
    val md    = m.asInstanceOf[JCTree.JCMethodDecl]
    val scope = new Scope(deps, isEntry, planned)
    val receiver =
      if (md.sym.isStatic) None
      else
        Some(p.Arg(p.Named("this", p.Type.Struct(p.Sym(ownerName(md.sym)), Nil))))
    receiver.foreach(r => scope.thisOpt = Some(r.named))
    for {
      declared <- m.getParameters.asScala.toList.zipWithIndex.traverse { (v, i) =>
        deps.capturedClosures.get(symOf(v)).filter(_ => isEntry) match {
          case Some(c) => scope.closures.put(symOf(v), c); Option.empty[p.Arg].success
          case None =>
            JavaTyper.typer(jcType(v)).map { tpe =>
              val named = p.Named(scope.uniqueName(v.getName.toString), tpe)
              scope.locals.put(symOf(v), named)
              scope.paramOf.put(symOf(v), i)
              Some(p.Arg(named))
            }
        }
      }
      args = declared.flatten
      staticArgs <- deps.staticParams.getOrElse(md.sym, Nil).traverse { v =>
        JavaTyper.typer(v.asType).map { tpe =>
          val named = p.Named(s"#static${scope.staticFields.size}", p.Type.Ptr(tpe, p.Type.Space.Global))
          scope.staticFields.put(v, named)
          p.Arg(named)
        }
      }
      params = m.getParameters.asScala.toVector
      lengths =
        if (isEntry) params.zipWithIndex.collect { case (v, i) if jcType(v).getKind == TypeKind.ARRAY => (v, i) }
        else deps.lengthParams.getOrElse(md.sym, Nil).map(i => (params(i), i))
      lengthArgs = lengths.map { (v, i) =>
        val named = p.Named(s"#len$i", p.Type.IntS32)
        scope.arrayLengths.put(symOf(v), p.Term.Select(named, Nil, p.Type.IntS32))
        p.Arg(named)
      }
      rowLengthTpe = p.Type.Ptr(p.Type.IntS32, p.Type.Space.Global)
      rowLengths =
        if (!isEntry) Nil
        else
          params.zipWithIndex.collect {
            case (v, i) if isArrayOfArrays(jcType(v)) =>
              val named = p.Named(s"#rowlen$i", rowLengthTpe)
              scope.rowLengths.put(symOf(v), RowLength.PerRow(p.Term.Select(named, Nil, named.tpe)))
              (i, p.Arg(named))
          }
      rtn <- JavaTyper.typer(md.sym.getReturnType)
      _   = scope.rtnType = rtn
      buf = ListBuffer.empty[p.Stmt]
      _ <- Option(m.getBody).traverse_(b => mapStmt(scope, buf, b))
      body = buf.lastOption match {
        case Some(p.Stmt.Return(_))   => buf.toList
        case _ if rtn == p.Type.Unit0 => buf.toList :+ p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
        case _                        => buf.toList
      }
      _ = deps.directStatics.put(md.sym, scope.staticFields.keys.toList)
      _ = deps.callees.put(md.sym, scope.calls.toList)
      _ = deps.measuredParams.put(md.sym, scope.measured)
      _ = deps.lengthFlows.put(md.sym, scope.argFlows)
    } yield (
      p.Function(
        decl = p.FunctionDecl(
          name = p.Sym(fullName(md.sym)),
          tpeVars = Nil,
          receiver = receiver,
          // entries are laid out as [tid] [receiver] moduleCaptures ++ termCaptures ++ args, and a lambda impl's params are its captures
          args = if (isEntry) Nil else args ++ lengthArgs ++ staticArgs,
          moduleCaptures = Nil,
          // static slots follow the parameters and their lengths, in the order compileRegion plans captures
          termCaptures =
            if (isEntry) args ++ lengthArgs ++ rowLengths.map(_._2) ++ scope.staticFields.values.map(p.Arg(_))
            else Nil,
          rtn = rtn,
          affinity = p.Function.Affinity.Offload
        ),
        body = body,
        visibility = p.Function.Visibility.Exported,
        fpMode = p.Function.FpMode.Relaxed,
        convention = if (isEntry) p.CallConvention.OffloadEntry else p.CallConvention.RegularCall
      ),
      scope.staticFields.keys.toList
    )
  }

  // TODO overloads collapse on fullName
  private def compileTransitive(
      env: ProcessingEnvironment,
      trees: Trees,
      deps: DepCollector,
      entrySym: JSymbol.MethodSymbol,
      planned: Boolean
  ): Result[List[p.Function]] = {
    val compiled                 = mutable.LinkedHashMap[JSymbol.MethodSymbol, p.Function]()
    val seen                     = mutable.Set[JSymbol.MethodSymbol](entrySym)
    var error: Option[Throwable] = None
    var progress                 = true
    // GENERATE-phase trees are already lowered so getTree is null, callees resolve from methodTable
    def calleeTree(sym: JSymbol.MethodSymbol): Option[MethodTree] =
      deps.treeOf(sym).orElse(Option(trees.getTree(sym)).collect { case mt: MethodTree => mt })
    while (progress && error.isEmpty) {
      val pending = deps.fnSyms.toList.filterNot(seen.contains)
      if (pending.isEmpty) progress = false
      else
        pending.foreach { sym =>
          if (error.isEmpty && !seen.contains(sym)) {
            seen += sym
            calleeTree(sym).filter(_ != null) match {
              case Some(mt) if mt.getBody != null =>
                compileMethod(env, trees, mt, deps, isEntry = false, planned = planned) match {
                  case Right((fn, _)) => compiled.put(sym, fn)
                  case Left(e)        => error = Some(e)
                }
              case _ => debug(s"[PolyAST] skip dependency (no reachable body): ${fullName(sym)}")
            }
          }
        }
    }
    error.toLeft(compiled.values.toList)
  }

  // TODO collect struct refs nested in bodies too
  private def structSyms(fns: List[p.Function]): List[p.Sym] = {
    def sig(fn: p.Function) =
      (fn.receiver.toList ++ fn.args ++ fn.moduleCaptures ++ fn.termCaptures).map(_.named.tpe) :+ fn.rtn
    fns.flatMap(sig).flatMap(JavaTyper.structSyms).distinct
  }

  private def saturate[A, B: Ordering](
      seed: IterableOnce[(JSymbol.MethodSymbol, IterableOnce[A])]
  )(grow: mutable.LinkedHashMap[JSymbol.MethodSymbol, mutable.LinkedHashSet[A]] => Boolean)(
      key: A => B
  ): Map[JSymbol.MethodSymbol, List[A]] = {
    val acc = mutable.LinkedHashMap[JSymbol.MethodSymbol, mutable.LinkedHashSet[A]]()
    seed.iterator.foreach((f, as) => acc.getOrElseUpdate(f, mutable.LinkedHashSet()) ++= as)
    while (grow(acc)) ()
    acc.map((f, as) => f -> as.toList.sortBy(key)).toMap
  }

  // only the entry can grow a capture, so a function touching a static takes the buffer as a parameter its caller forwards
  private def planStatics(deps: DepCollector): Unit =
    deps.staticParams = saturate(
      (deps.directStatics.keys ++ deps.callees.keys).map(f => f -> deps.directStatics.getOrElse(f, Nil))
    ) { reaches =>
      var changed = false
      reaches.keys.toList.foreach { f =>
        deps.callees.getOrElse(f, Nil).foreach { g =>
          reaches.get(g).foreach { gs =>
            val fs = reaches(f)
            if (!gs.subsetOf(fs)) { fs ++= gs; changed = true }
          }
        }
      }
      changed
    }(fullName)

  private def planLengths(deps: DepCollector): Unit =
    deps.lengthParams = saturate(deps.measuredParams) { needs =>
      var changed = false
      deps.lengthFlows.foreach { (caller, flows) =>
        flows.foreach { (callee, calleeIdx, callerIdx) =>
          if (needs.get(callee).exists(_.contains(calleeIdx))) {
            val cs = needs.getOrElseUpdate(caller, mutable.LinkedHashSet())
            if (!cs.contains(callerIdx)) { cs += callerIdx; changed = true }
          }
        }
      }
      changed
    }(identity)

  def compileRegion(
      env: ProcessingEnvironment,
      trees: Trees,
      classTable: Map[String, TypeElement],
      methodTable: Map[String, MethodTree],
      localTrees: Map[JSymbol.MethodSymbol, MethodTree],
      entry: MethodTree
  ): Result[Region] = {
    val deps =
      new DepCollector(localTrees, methodTable, classTable, env.getElementUtils, env.getTypeUtils)
    val entrySym = entry.asInstanceOf[JCTree.JCMethodDecl].sym
    val params   = entry.getParameters.asScala.toList
    deps.capturedClosures = capturedClosures(deps, entry)
    for {
      // exploratory lowering, discarded: learns the call graph and statics needed to size signatures
      _ <- compileMethod(env, trees, entry, deps, isEntry = true, planned = false)
      _ <- compileTransitive(env, trees, deps, entrySym, planned = false)
      _ = planStatics(deps)
      _ = planLengths(deps)
      entryAndStatics <- compileMethod(env, trees, entry, deps, isEntry = true, planned = true)
      (entryFn, statics) = entryAndStatics
      depFns <- compileTransitive(env, trees, deps, entrySym, planned = true)
      signatureStructs <- structSyms(entryFn :: depFns).flatTraverse { sym =>
        val fqn = sym.fqn.mkString(".")
        classTable.get(fqn) match {
          case Some(elem) => JavaTyper.lowerStruct(elem, deps.taggedTypes.contains(fqn)).map(List(_))
          case None       => List.empty[p.StructDef].success
        }
      }
      allocatedStructs <- deps.witnessed.toList.traverse { elem =>
        JavaTyper.lowerStruct(elem, deps.taggedTypes.contains(elem.getQualifiedName.toString))
      }
      structs = signatureStructs ++ allocatedStructs
      shift   = if (entryFn.receiver.isDefined) 1 else 0
      captures =
        (if (shift == 1) List(Capture.Receiver) else Nil) ++
          params.zipWithIndex.collect {
            case (v, i) if !deps.capturedClosures.contains(symOf(v)) => Capture.Arg(i + shift)
          } ++
          params.zipWithIndex.collect {
            case (v, i) if jcType(v).getKind == TypeKind.ARRAY => Capture.Length(i + shift)
          } ++
          params.zipWithIndex.collect {
            case (v, i) if isArrayOfArrays(jcType(v)) => Capture.RowLengths(i + shift)
          } ++
          statics.map(v => Capture.StaticField(ownerName(v), v.name.toString))
    } yield Region(p.Program(Some(entryFn), depFns, structs.distinct), captures, deps.tagOf.toMap)
  }
}
