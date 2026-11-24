package polyregion.javalang

import cats.syntax.all.*
import com.sun.source.tree.*
import com.sun.source.util.Trees
import com.sun.tools.javac.code.Symbol as JSymbol
import com.sun.tools.javac.processing.JavacProcessingEnvironment
import com.sun.tools.javac.tree.JCTree
import polyregion.ast.{PolyAST as p, *}
import polyregion.ast.pass.Devirtualise
import polyregion.intrinsic.IntrinsicTable

import javax.annotation.processing.ProcessingEnvironment
import javax.lang.model.`type`.{ArrayType, DeclaredType, TypeKind, TypeMirror}
import javax.lang.model.element.{ElementKind, Modifier, TypeElement}
import javax.lang.model.util.{Elements, Types}
import javax.tools.{JavaFileManager, JavaFileObject, StandardLocation}
import scala.annotation.tailrec
import scala.collection.immutable.ListSet
import scala.collection.mutable
import scala.jdk.CollectionConverters.*

object TreeRemapper {

  import polyregion.ast.Env.debug

  private case class Plan(
      statics: Map[JSymbol.MethodSymbol, List[JSymbol.VarSymbol]],
      lengths: Map[JSymbol.MethodSymbol, List[Int]]
  )

  private class DepCollector(
      val localTrees: Map[JSymbol.MethodSymbol, MethodTree],
      val methodTable: Map[String, MethodTree],
      val classTable: Map[String, TypeElement],
      val elements: Elements,
      val types: Types,
      val capturedClosures: Map[JSymbol, Closure],
      val classBytes: BytecodeResolver.ClassBytes
  ) {
    val fnSyms        = mutable.LinkedHashSet.empty[JSymbol.MethodSymbol]
    val bytecodeRoots = mutable.LinkedHashSet.empty[MethodRef]
    val nativeFns     = mutable.LinkedHashSet.empty[p.Function]

    def treeOf(sym: JSymbol.MethodSymbol): Option[MethodTree] =
      localTrees.get(sym).orElse(methodTable.get(fullName(sym))).filter(bodied(_))

    // javac adds synthetic constructor params, so a call site may name a different symbol than the declaration
    def ctorOf(owner: JSymbol.ClassSymbol, arity: Int): Option[MethodTree] =
      localTrees.collectFirst {
        case (s, t) if s.owner == owner && s.name.toString == "<init>" && bodied(t) && t.getParameters.size == arity =>
          t
      }

    // the classes a dispatch names, whose structs the call sites alone would not bring in
    val dispatchClasses = mutable.LinkedHashSet.empty[TypeElement]
    val owners          = mutable.LinkedHashMap.empty[p.Sym, List[p.Sym]]

    // each virtual method with the classes implementing it, `args` picking the call's arguments an implementation takes
    val dispatch = mutable.LinkedHashMap.empty[p.Sym, List[Arm]]

    val witnessed = mutable.LinkedHashSet.empty[TypeElement]

    val outerRefs = mutable.LinkedHashSet.empty[(String, String)]

    val directStatics = mutable.LinkedHashMap.empty[JSymbol.MethodSymbol, List[JSymbol.VarSymbol]]
    val callees       = mutable.LinkedHashMap.empty[JSymbol.MethodSymbol, List[JSymbol.MethodSymbol]]

    // Ptr drops a Java array's length, so a callee measuring an array parameter takes the count alongside it
    val measuredParams = mutable.LinkedHashMap.empty[JSymbol.MethodSymbol, mutable.LinkedHashSet[Int]]
    val lengthFlows =
      mutable.LinkedHashMap.empty[JSymbol.MethodSymbol, mutable.ListBuffer[(JSymbol.MethodSymbol, Int, Int)]]
  }

  case class Closure(impl: JSymbol.MethodSymbol, captures: List[Either[p.Term, Closure]])

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

  case class Captured(name: String, capture: Capture)

  case class Region(
      program: p.Program,
      captures: List[Captured],
      tags: Map[String, Int],
      dispatch: Map[p.Sym, List[Devirtualise.Candidate]],
      owners: Map[p.Sym, List[p.Sym]],
      outerRefs: Set[(String, String)]
  )

  case class LoopCtx(
      update: List[ExpressionStatementTree],
      reeval: List[p.Stmt],
      viaLabel: Option[LabelCtx] = None
  )

  case class LabelCtx(name: String, brk: p.Term.Select, cnt: p.Term.Select)

  private class Scope(
      val deps: DepCollector,
      val isEntry: Boolean,
      val plan: Option[Plan],
      val rtnType: p.Type,
      receiver: Option[p.Named]
  ) {
    private val ids                 = Iterator.from(0)
    val locals                      = mutable.Map.empty[JSymbol, p.Named]
    val paramOf                     = mutable.Map.empty[JSymbol, Int]
    val measured                    = mutable.LinkedHashSet.empty[Int]
    val argFlows                    = mutable.ListBuffer.empty[(JSymbol.MethodSymbol, Int, Int)]
    val arrayLengths                = mutable.Map.empty[JSymbol, p.Term]
    val rowLengths                  = mutable.Map.empty[JSymbol, RowLength]
    val newArrayLengths             = mutable.Map.empty[Tree, p.Term]
    val newRowLengths               = mutable.Map.empty[Tree, RowLength]
    val staticFields                = mutable.LinkedHashMap.empty[JSymbol.VarSymbol, p.Named]
    val calls                       = mutable.LinkedHashSet.empty[JSymbol.MethodSymbol]
    def fresh(tpe: p.Type): p.Named = p.Named(s"_v${ids.next()}", tpe)
    def planned: Boolean            = plan.isDefined

    def staticParamsOf(sym: JSymbol.MethodSymbol): List[JSymbol.VarSymbol] =
      plan.fold(Nil)(_.statics.getOrElse(sym, Nil))

    def lengthParamsOf(sym: JSymbol.MethodSymbol): List[Int] =
      plan.fold(Nil)(_.lengths.getOrElse(sym, Nil))

    val closures = mutable.Map.empty[JSymbol, Closure]

    private val inlining                             = mutable.Set.empty[JSymbol.MethodSymbol]
    def isInlining(s: JSymbol.MethodSymbol): Boolean = inlining.contains(s)
    def whileInlining[A](s: JSymbol.MethodSymbol)(f: => A): A = {
      inlining += s
      try f
      finally inlining -= s
    }

    private val receivers        = mutable.Stack(receiver)
    def thisOpt: Option[p.Named] = receivers.head
    def withThis[A](named: Option[p.Named])(f: => A): A = {
      receivers.push(named)
      try f
      finally receivers.pop()
    }

    // Java scopes nest and PolyAST bodies do not, so disjoint `for (int i ...)` loops would trip the verifier
    private val names = mutable.Set.empty[String]
    def uniqueName(base: String): String = {
      @tailrec def go(candidate: String, i: Int): String =
        if (names.add(candidate)) candidate else go(s"$base$$$i", i + 1)
      go(base, 1)
    }

    private val loops                  = mutable.Stack.empty[LoopCtx]
    def enclosingLoop: Option[LoopCtx] = loops.headOption
    def loopDepth: Int                 = loops.size
    def inLoop(ctx: LoopCtx)(f: => Result[Unit]): Result[Unit] = {
      loops.push(ctx)
      try f
      finally loops.pop()
    }

    private val labels                           = mutable.Stack.empty[LabelCtx]
    def activeLabels: List[LabelCtx]             = labels.toList
    def labelled(name: String): Option[LabelCtx] = labels.find(_.name == name)
    def withLabel(ctx: LabelCtx)(f: => Result[Unit]): Result[Unit] = {
      labels.push(ctx)
      try f
      finally labels.pop()
    }

    private val yields                               = mutable.Stack.empty[(p.Term.Select, Int)]
    def enclosingYield: Option[(p.Term.Select, Int)] = yields.headOption
    def withYieldTarget[A](target: p.Term.Select)(f: => A): A = {
      yields.push((target, loops.size))
      try f
      finally yields.pop()
    }
  }

  private def jcType(t: Tree): TypeMirror = t match {
    case jc: JCTree => jc.`type`
  }

  private def declaredName(m: TypeMirror): String = Option(m)
    .collect { case d: DeclaredType if d.getKind == TypeKind.DECLARED => d.asElement }
    .collect { case e: TypeElement => e.getQualifiedName.toString }
    .getOrElse("")

  private def isStringTpe(m: TypeMirror): Boolean = declaredName(m) == "java.lang.String"

  private def isAssertionErrorTpe(m: TypeMirror): Boolean = declaredName(m) == "java.lang.AssertionError"

  private def assertMessage(nc: NewClassTree): Result[String] = nc.getArguments.asScala.toList match {
    case Nil                                                              => "assertion failed".success
    case (l: LiteralTree) :: Nil if l.getKind == Tree.Kind.STRING_LITERAL => l.getValue.toString.success
    case (l: LiteralTree) :: Nil                                          => String.valueOf(l.getValue).success
    case xs =>
      s"Unsupported `AssertionError` detail `${xs.mkString(", ")}`; only a literal travels to the device".fail
  }

  private def namedRaiseCall(name: String, message: String): p.Expr = {
    val module    = p.Sym("polyregion" :: "scalalang" :: "intrinsics$" :: Nil)
    val moduleTpe = p.Type.Struct(module, Nil)
    val receiver  = p.Term.Select(p.Named("$intrinsics", moduleTpe), Nil, moduleTpe)
    p.Expr.Invoke(
      p.Type.FnRef(p.Sym("polyregion" :: "scalalang" :: "intrinsics$" :: "raiseNamed" :: Nil)),
      Nil,
      Some(receiver),
      List(p.Term.StringConst(name), p.Term.StringConst(message)),
      p.Type.Unit0
    )
  }

  private def namedRaise(scope: Scope, nc: NewClassTree): Result[p.Expr] = {
    val throwable = scope.deps.elements.getTypeElement("java.lang.Throwable")
    val isThrowable = Option(jcType(nc)).exists { tpe =>
      Option(throwable).exists { cls =>
        scope.deps.types.isSubtype(scope.deps.types.erasure(tpe), scope.deps.types.erasure(cls.asType))
      }
    }
    val binaryName = Option(jcType(nc))
      .collect { case d: DeclaredType => d.asElement }
      .collect { case c: TypeElement => scope.deps.elements.getBinaryName(c).toString }
    (isThrowable, binaryName, nc.getArguments.asScala.toList) match {
      case (true, Some(name), (l: LiteralTree) :: Nil) if l.getKind == Tree.Kind.STRING_LITERAL =>
        namedRaiseCall(name, l.getValue.toString).success
      case _ => "Unsupported `throw`; PolyRegion kernels cannot throw".fail
    }
  }

  private def symOf(t: Tree): Option[JSymbol] = t match {
    case i: JCTree.JCIdent        => Option(i.sym)
    case f: JCTree.JCFieldAccess  => Option(f.sym)
    case v: JCTree.JCVariableDecl => Option(v.sym)
    case _                        => None
  }

  @tailrec private def unparenthesised(t: Tree): Tree = t match {
    case par: ParenthesizedTree => unparenthesised(par.getExpression)
    case other                  => other
  }

  private def methodSymOf(t: Tree): Result[JSymbol.MethodSymbol] =
    symOf(t).collect { case m: JSymbol.MethodSymbol => m }.failIfEmpty(s"`$t` does not name a method")

  private def methodDeclSym(m: MethodTree): JSymbol.MethodSymbol = m match {
    case d: JCTree.JCMethodDecl => d.sym
  }

  private def declOf(v: VariableTree): JSymbol = v match {
    case d: JCTree.JCVariableDecl => d.sym
  }

  private def bodied(t: MethodTree): Boolean = Option(t).flatMap(m => Option(m.getBody)).isDefined

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
  def dynamicSym(mi: MethodInvocationTree): Option[JSymbol.DynamicMethodSymbol] =
    symOf(mi.getMethodSelect).collect { case d: JSymbol.DynamicMethodSymbol => d }

  def metafactoryImpl(d: JSymbol.DynamicMethodSymbol): Option[JSymbol.MethodSymbol] = {
    val bsm     = Option(d.bsm).map(_.baseSymbol())
    val bsmName = bsm.fold("")(_.name.toString)
    val isFactory = bsm.exists(b => ownerName(b) == MetafactoryOwner) &&
      (bsmName == "metafactory" || bsmName == "altMetafactory")
    if (!isFactory || Option(d.staticArgs).forall(_.length < 2)) None
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

  private def bind(scope: Scope, buf: mutable.ListBuffer[p.Stmt], e: p.Expr): p.Term = e match {
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

  private def literal(l: LiteralTree): Result[p.Expr] = (l.getKind, l.getValue) match {
    case (Tree.Kind.INT_LITERAL, v: Int)         => p.Expr.Alias(p.Term.IntS32Const(v)).success
    case (Tree.Kind.LONG_LITERAL, v: Long)       => p.Expr.Alias(p.Term.IntS64Const(v)).success
    case (Tree.Kind.FLOAT_LITERAL, v: Float)     => p.Expr.Alias(p.Term.Float32Const(v)).success
    case (Tree.Kind.DOUBLE_LITERAL, v: Double)   => p.Expr.Alias(p.Term.Float64Const(v)).success
    case (Tree.Kind.BOOLEAN_LITERAL, v: Boolean) => p.Expr.Alias(p.Term.Bool1Const(v)).success
    case (Tree.Kind.CHAR_LITERAL, v: Char)       => p.Expr.Alias(p.Term.IntU16Const(v)).success
    case (Tree.Kind.NULL_LITERAL, _) => JavaTyper.typer(jcType(l)).map(tpe => p.Expr.Alias(p.Term.Poison(tpe)))
    case (Tree.Kind.STRING_LITERAL, _) =>
      "Unsupported `String` literal; PolyRegion has no device-side String".fail
    case (other, _) => s"Unsupported literal kind $other".fail
  }

  // JLS 5.6.2, PolyAST has no implicit conversions
  private def promotedType(l: p.Type, r: p.Type): p.Type =
    if (l == p.Type.Float64 || r == p.Type.Float64) p.Type.Float64
    else if (l == p.Type.Float32 || r == p.Type.Float32) p.Type.Float32
    else if (l == p.Type.IntS64 || r == p.Type.IntS64) p.Type.IntS64
    else p.Type.IntS32

  private def coerce(scope: Scope, buf: mutable.ListBuffer[p.Stmt], t: p.Term, target: p.Type): p.Term = t.tpe match {
    case same if same == target => t
    case p.Type.Ptr(pointee @ p.Type.Struct(_, _), _) if pointee == target =>
      t match {
        case sel: p.Term.Select => p.Term.Select(sel.root, sel.steps :+ p.PathStep.Deref, pointee)
        case _                  => t
      }
    case _ => bind(scope, buf, p.Expr.Cast(t, target))
  }

  // a subtype handed to a parameter of its supertype goes by reference, as a copy into the smaller supertype would truncate it
  private def coerceArg(
      scope: Scope,
      buf: mutable.ListBuffer[p.Stmt],
      t: p.Term,
      target: p.Type,
      byRef: Boolean
  ): p.Term = (t, target) match {
    case (sel @ p.Term.Select(_, _, p.Type.Struct(a, _)), p.Type.Struct(b, _)) if byRef && a != b =>
      val ref = byReference(scope, buf, sel)
      bind(scope, buf, p.Expr.Cast(ref, p.Type.Ptr(target, p.Type.Space.Private)))
    case _ => coerce(scope, buf, t, target)
  }

  private def fieldOf(base: p.Term.Select, name: String, tpe: p.Type): p.Term.Select = {
    val through = base.tpe match {
      case p.Type.Ptr(p.Type.Struct(_, _), _) => base.steps :+ p.PathStep.Deref
      case _                                  => base.steps
    }
    p.Term.Select(base.root, through :+ p.PathStep.Field(name), tpe)
  }

  private def referent(t: p.Type): p.Type = t match {
    case p.Type.Ptr(s @ p.Type.Struct(_, _), _) => s
    case other                                  => other
  }

  private def referenceTo(tpe: p.Type): p.Type = tpe match {
    case s @ p.Type.Struct(_, _) => p.Type.Ptr(s, p.Type.Space.Private)
    case other                   => other
  }

  // objects are passed by reference so that a callee writing a field is seen by its caller
  private def byReference(scope: Scope, buf: mutable.ListBuffer[p.Stmt], t: p.Term): p.Term = t match {
    case sel @ p.Term.Select(_, _, p.Type.Struct(_, _)) =>
      bind(
        scope,
        buf,
        p.Expr.RefTo(sel, None, sel.tpe, p.Type.Space.Private, p.Region.Rooted(sel.root))
      )
    case _ => t
  }

  private def isBoolean(t: p.Type): Boolean = t == p.Type.Bool1

  private def isArrayOfArrays(m: TypeMirror): Boolean = m match {
    case a: ArrayType => a.getKind == TypeKind.ARRAY && a.getComponentType.getKind == TypeKind.ARRAY
    case _            => false
  }

  private def isScalar(t: p.Type): Boolean = t match {
    case p.Type.Bool1 | p.Type.IntS8 | p.Type.IntS16 | p.Type.IntU16 | p.Type.IntS32 | p.Type.IntS64 | p.Type.Float32 |
        p.Type.Float64 =>
      true
    case _ => false
  }

  private def noteOuterRef(scope: Scope, sym: Option[JSymbol]): Unit =
    sym.flatMap(JavaTyper.outerRefOf(_)).foreach(scope.deps.outerRefs += _)

  private def classOwned(v: JSymbol): Boolean = v.owner match {
    case _: JSymbol.ClassSymbol => true
    case _                      => false
  }

  private def staticFieldOf(t: Tree): Option[JSymbol.VarSymbol] = symOf(t).collect {
    case v: JSymbol.VarSymbol if v.isStatic && classOwned(v) => v
  }

  private def constOf(c: Any, tpe: p.Type): Result[p.Expr] = Scalar.constant(tpe) match {
    case None => s"Unsupported constant of type ${tpe.repr}".fail
    case Some(lower) =>
      Scalar.of(c) match {
        case Some(n) => p.Expr.Alias(lower(n)).success
        case None    => s"Cannot lower constant `$c` as ${tpe.repr}".fail
      }
  }

  private def isEnumType(m: TypeMirror): Boolean = m match {
    case d: DeclaredType => d.getKind == TypeKind.DECLARED && d.asElement.getKind == ElementKind.ENUM
    case _               => false
  }

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
    scope.staticParamsOf(sym).traverse { v =>
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
    scope.lengthParamsOf(sym).traverse { i =>
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

  private case class ImplCall(
      impl: JSymbol.MethodSymbol,
      recv: Option[p.Term],
      values: List[Either[p.Term, Closure]],
      rtn: p.Type
  )

  private case class Dispatch(sym: JSymbol.MethodSymbol, recv: p.Term, args: List[p.Term], rtn: p.Type)

  private enum Lvalue {
    case Slot(sel: p.Term.Select)
    case Elem(ptr: p.Term.Select, idx: p.Term, comp: p.Type)

    def tpe: p.Type = this match {
      case Slot(sel)     => sel.tpe
      case Elem(_, _, c) => c
    }

    def write(buf: mutable.ListBuffer[p.Stmt], v: p.Term): Unit = this match {
      case Slot(sel)       => buf += p.Stmt.Mut(sel, p.Expr.Alias(v))
      case Elem(ptr, i, _) => buf += p.Stmt.Update(ptr, i, v)
    }
  }

  private def readLvalue(scope: Scope, buf: mutable.ListBuffer[p.Stmt], lv: Lvalue): p.Term = lv match {
    case Lvalue.Slot(sel)       => sel
    case Lvalue.Elem(ptr, i, c) => bind(scope, buf, p.Expr.Index(ptr, i, c))
  }

  // no intermediate copies: binding an `xs[i]` of struct type copies it and the store would miss the array
  private def selectPath(scope: Scope, buf: mutable.ListBuffer[p.Stmt], t: ExpressionTree): Result[p.Term.Select] =
    t match {
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
          _ = noteOuterRef(scope, symOf(fa))
        } yield fieldOf(base, fa.getIdentifier.toString, tpe)
      case other => mapExpr(scope, buf, other).map(bind(scope, buf, _)).flatMap(x => asSelect(p.Expr.Alias(x)))
    }

  private def lvalue(scope: Scope, buf: mutable.ListBuffer[p.Stmt], t: ExpressionTree): Result[Lvalue] = t match {
    case aa: ArrayAccessTree =>
      for {
        arr  <- mapExpr(scope, buf, aa.getExpression).map(bind(scope, buf, _)).flatMap(t => asSelect(p.Expr.Alias(t)))
        idx  <- mapExpr(scope, buf, aa.getIndex).map(bind(scope, buf, _))
        comp <- JavaTyper.typer(jcType(aa))
      } yield Lvalue.Elem(arr, idx, comp)
    case _ =>
      staticFieldOf(t) match {
        case Some(v) if Option(v.getConstValue).isEmpty =>
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

  private def isPtr(t: p.Type): Boolean = t match {
    case p.Type.Ptr(_, _) => true
    case _                => false
  }

  private def isReference(t: p.Type): Boolean = isPtr(t) || t == p.Type.Nothing

  // reference == is identity, which PolyAST only expresses over integers, so both sides become addresses
  private def referenceCompare(
      scope: Scope,
      buf: mutable.ListBuffer[p.Stmt],
      l: p.Term,
      r: p.Term
  ): Result[(p.Term, p.Term)] = for {
    l0 <- nullAgainst(r.tpe, l)
    r0 <- nullAgainst(l.tpe, r)
    _ <-
      if (isPtr(l0.tpe) && isPtr(r0.tpe)) ().success
      else s"Cannot compare ${l0.tpe.repr} against ${r0.tpe.repr} by reference".fail
  } yield (coerce(scope, buf, l0, p.Type.IntS64), coerce(scope, buf, r0, p.Type.IntS64))

  private def promoteOperands(
      scope: Scope,
      buf: mutable.ListBuffer[p.Stmt],
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

  private def arrayLength(scope: Scope, buf: mutable.ListBuffer[p.Stmt], t: ExpressionTree): Result[p.Term] =
    symOf(t).flatMap(scope.arrayLengths.get) match {
      case Some(len) => len.success
      case None =>
        val row = t match {
          case aa: ArrayAccessTree =>
            symOf(aa.getExpression).flatMap(scope.rowLengths.get).map {
              case RowLength.Uniform(len) => len.success
              case RowLength.PerRow(lens) =>
                mapExpr(scope, buf, aa.getIndex)
                  .map(i => bind(scope, buf, p.Expr.Index(lens, bind(scope, buf, i), p.Type.IntS32)))
            }
          case _ => None
        }
        row
          .orElse(
            symOf(t).flatMap(scope.paramOf.get).filterNot(_ => scope.planned).map { i =>
              scope.measured += i
              (p.Term.IntS32Const(0): p.Term).success
            }
          )
          .getOrElse(
            (s"Cannot take `length` of `$t`: only a captured or locally allocated array, or a row " +
              "indexed straight out of one, carries a length").fail
          )
    }

  private def guardArrayAccess(
      scope: Scope,
      buf: mutable.ListBuffer[p.Stmt],
      array: ExpressionTree,
      index: p.Term
  ): Result[Unit] =
    (arrayLength(scope, buf, array), index) match {
      case (Right(p.Term.IntS32Const(len)), p.Term.IntS32Const(idx)) if idx < 0 || idx >= len =>
        val raised = namedRaiseCall("java.lang.ArrayIndexOutOfBoundsException", "array index out of bounds")
        buf += p.Stmt.Var(scope.fresh(p.Type.Unit0), Some(raised))
        ().success
      case _ => ().success
    }

  private def mapExpr(scope: Scope, buf: mutable.ListBuffer[p.Stmt], t: ExpressionTree): Result[p.Expr] = t match {
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
        _ = noteOuterRef(scope, symOf(fa))
      } yield p.Expr.Alias(fieldOf(sel, fa.getIdentifier.toString, tpe))
    case aa: ArrayAccessTree =>
      for {
        arr  <- mapExpr(scope, buf, aa.getExpression).map(bind(scope, buf, _))
        idx  <- mapExpr(scope, buf, aa.getIndex).map(bind(scope, buf, _))
        _    <- guardArrayAccess(scope, buf, aa.getExpression, idx)
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
        rhsBuf                = mutable.ListBuffer.empty[p.Stmt]
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
              scope.newArrayLengths.put(na, p.Term.IntS32Const(inits.size))
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
                    scope.newArrayLengths.put(na, len)
                    allocZeroed(scope, buf, comp, space, len).map(p.Expr.Alias(_))
                  case len :: rest =>
                    scope.newArrayLengths.put(na, len)
                    scope.newRowLengths.put(na, RowLength.Uniform(rest.head))
                    allocRows(scope, buf, comp, space, len, rest).map(p.Expr.Alias(_))
                }
          }
        case other => s"Cannot allocate an array of ${other.repr}".fail
      }
    case se: SwitchExpressionTree   => switchExpression(scope, buf, se)
    case a: AssignmentTree          => assign(scope, buf, a)
    case ca: CompoundAssignmentTree => compoundAssign(scope, buf, ca)
    case mi: MethodInvocationTree   => invoke(scope, buf, mi)
    case nc: JCTree.JCNewClass      => construct(scope, buf, nc)
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
    sym.flatMap(scope.locals.get) match {
      case Some(named) => selectExpr(Nil, named).success
      case None =>
        if (id.getName.contentEquals("this") || id.getName.contentEquals("super"))
          scope.thisOpt.map(selectExpr(Nil, _)).failIfEmpty("`this` referenced outside an instance method")
        else
          (sym, scope.thisOpt) match {
            case (Some(v: JSymbol.VarSymbol), _) if v.isStatic && classOwned(v) =>
              staticRead(scope, v)
            case (Some(v: JSymbol.VarSymbol), Some(self)) if !v.isStatic && classOwned(v) =>
              noteOuterRef(scope, sym)
              JavaTyper
                .typer(jcType(id))
                .map(tpe => p.Expr.Alias(fieldOf(selectTerm(Nil, self), id.getName.toString, tpe)))
            // TODO module-style references
            case _ => s"Unresolved identifier `${id.getName}` (symbol ${sym.getOrElse("null")})".fail
          }
    }
  }

  private def unary(scope: Scope, buf: mutable.ListBuffer[p.Stmt], u: UnaryTree): Result[p.Expr] = {
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
        next <- step(old, lv.tpe)
      } yield {
        lv.write(buf, next)
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

  private def ternary(scope: Scope, buf: mutable.ListBuffer[p.Stmt], c: ConditionalExpressionTree): Result[p.Expr] =
    for {
      cond <- mapExpr(scope, buf, c.getCondition).map(bind(scope, buf, _))
      tpe  <- JavaTyper.typer(jcType(c))
      result  = scope.fresh(tpe)
      _       = buf += p.Stmt.Var(result, None, isMutable = true)
      thenBuf = mutable.ListBuffer.empty[p.Stmt]
      tThen <- mapExpr(scope, thenBuf, c.getTrueExpression).map(bind(scope, thenBuf, _))
      _       = thenBuf += p.Stmt.Mut(selectTerm(Nil, result), p.Expr.Alias(coerce(scope, thenBuf, tThen, tpe)))
      elseBuf = mutable.ListBuffer.empty[p.Stmt]
      tElse <- mapExpr(scope, elseBuf, c.getFalseExpression).map(bind(scope, elseBuf, _))
      _ = elseBuf += p.Stmt.Mut(selectTerm(Nil, result), p.Expr.Alias(coerce(scope, elseBuf, tElse, tpe)))
      _ = buf += p.Stmt.Cond(cond, thenBuf.toList, elseBuf.toList)
    } yield selectExpr(Nil, result)

  private def assign(scope: Scope, buf: mutable.ListBuffer[p.Stmt], a: AssignmentTree): Result[p.Expr] =
    for {
      lv  <- lvalue(scope, buf, a.getVariable)
      rhs <- mapExpr(scope, buf, a.getExpression).map(bind(scope, buf, _))
      _ <- (lv.tpe, rhs.tpe) match {
        case (l: p.Type.Struct, r: p.Type.Struct) if l != r =>
          s"Cannot assign a ${r.repr} to a ${l.repr}: an object here is stored by value".fail
        case _ => ().success
      }
    } yield {
      lv.write(buf, coerce(scope, buf, rhs, lv.tpe))
      p.Expr.Alias(p.Term.Unit0Const)
    }

  private def compoundAssign(
      scope: Scope,
      buf: mutable.ListBuffer[p.Stmt],
      ca: CompoundAssignmentTree
  ): Result[p.Expr] = {
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
      v  <- combine(readLvalue(scope, buf, lv), lv.tpe)
    } yield {
      lv.write(buf, v)
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

  private def boxIdentity(
      scope: Scope,
      buf: mutable.ListBuffer[p.Stmt],
      mi: MethodInvocationTree
  ): Option[Result[p.Expr]] =
    mi.getMethodSelect match {
      case fa: MemberSelectTree =>
        val sym   = symOf(fa)
        val name  = sym.fold("")(_.name.toString)
        val owner = sym.map(ownerName(_))
        if (name == "valueOf" && owner.exists(boxTypes.contains) && mi.getArguments.size == 1)
          Some(mapExpr(scope, buf, mi.getArguments.get(0)))
        else if (unboxMethods.contains(name) && owner.exists(boxTypes.contains))
          Some(mapExpr(scope, buf, fa.getExpression))
        else None
      case _ => None
    }

  private val EnumOwner = "java.lang.Enum"

  private def isRecordAccessor(sym: JSymbol.MethodSymbol): Boolean = sym.owner match {
    case c: JSymbol.ClassSymbol if c.getKind == ElementKind.RECORD =>
      c.getEnclosedElements.asScala.exists(e => e.getKind == ElementKind.FIELD && e.getSimpleName == sym.name)
    case _ => false
  }

  private def zeroOf(tpe: p.Type): Result[p.Term] = Scalar.zero(tpe).failIfEmpty(s"No zero for ${tpe.repr}")

  private def countedLoop(scope: Scope, buf: mutable.ListBuffer[p.Stmt], len: p.Term)(
      fill: (mutable.ListBuffer[p.Stmt], p.Term.Select) => Result[Unit]
  ): Result[Unit] = {
    val idx                    = scope.fresh(p.Type.IntS32)
    val idxSel: p.Term.Select  = p.Term.Select(idx, Nil, p.Type.IntS32)
    val more                   = scope.fresh(p.Type.Bool1)
    val moreSel: p.Term.Select = p.Term.Select(more, Nil, p.Type.Bool1)
    buf += p.Stmt.Var(idx, Some(p.Expr.Alias(p.Term.IntS32Const(0))), isMutable = true)
    buf += p.Stmt.Var(more, Some(p.Expr.IntrOp(p.Intr.LogicLt(idxSel, len))), isMutable = true)
    val bodyBuf = mutable.ListBuffer.empty[p.Stmt]
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
      buf: mutable.ListBuffer[p.Stmt],
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

  private def descriptorOf(deps: DepCollector, sym: JSymbol.MethodSymbol): String = {
    def one(t: TypeMirror): String = deps.types.erasure(t) match {
      case a: ArrayType => s"[${one(a.getComponentType)}"
      case d: DeclaredType =>
        d.asElement match {
          case c: TypeElement => s"L${deps.elements.getBinaryName(c).toString.replace('.', '/')};"
          case _              => "V"
        }
      case other =>
        other.getKind match {
          case TypeKind.BOOLEAN => "Z"
          case TypeKind.BYTE    => "B"
          case TypeKind.SHORT   => "S"
          case TypeKind.INT     => "I"
          case TypeKind.LONG    => "J"
          case TypeKind.CHAR    => "C"
          case TypeKind.FLOAT   => "F"
          case TypeKind.DOUBLE  => "D"
          case _                => "V"
        }
    }
    s"(${sym.getParameters.asScala.map(v => one(v.asType)).mkString})${one(sym.getReturnType)}"
  }

  private def binaryOwnerName(deps: DepCollector, sym: JSymbol.MethodSymbol): String = sym.owner match {
    case c: JSymbol.ClassSymbol => deps.elements.getBinaryName(c).toString
    case other                  => other.toString
  }

  private def intrinsicOf(deps: DepCollector, sym: JSymbol.MethodSymbol, args: List[p.Term]): Option[p.Expr] =
    Option
      .when(sym.isStatic)(Natives.resolve(calleeOf(deps, sym), args))
      .flatten
      .map {
        case IntrinsicTable.Resolved.Inline(expr) => expr
        case IntrinsicTable.Resolved.Call(fn, invoke) =>
          deps.nativeFns += fn
          invoke
      }

  private def calleeOf(deps: DepCollector, sym: JSymbol.MethodSymbol): MethodRef =
    MethodRef(binaryOwnerName(deps, sym), sym.name.toString, descriptorOf(deps, sym))

  private def isBytecodeCallee(deps: DepCollector, sym: JSymbol.MethodSymbol): Boolean =
    sym.isStatic && !sym.getModifiers.contains(Modifier.ABSTRACT) && deps.treeOf(sym).isEmpty

  private def bytecodeCall(
      deps: DepCollector,
      sym: JSymbol.MethodSymbol,
      args: List[p.Term],
      rtn: p.Type
  ): p.Expr = {
    val ref = calleeOf(deps, sym)
    deps.bytecodeRoots += ref
    p.Expr.Invoke(p.Type.FnRef(BytecodeRemapper.symbolOf(ref.owner, ref.name)), Nil, None, args, rtn)
  }

  // closureOf lowers captured arguments as it goes, so it cannot serve as a predicate
  @tailrec
  private def isClosureTree(scope: Scope, t: ExpressionTree): Boolean = t match {
    case par: ParenthesizedTree   => isClosureTree(scope, par.getExpression)
    case tc: TypeCastTree         => isClosureTree(scope, tc.getExpression)
    case id: IdentifierTree       => symOf(id).exists(scope.closures.contains)
    case mi: MethodInvocationTree => dynamicSym(mi).flatMap(metafactoryImpl(_)).isDefined
    case _                        => false
  }

  @tailrec
  private def closureOf(
      scope: Scope,
      buf: mutable.ListBuffer[p.Stmt],
      t: ExpressionTree
  ): Option[Result[Closure]] = t match {
    case par: ParenthesizedTree => closureOf(scope, buf, par.getExpression)
    case tc: TypeCastTree       => closureOf(scope, buf, tc.getExpression)
    case id: IdentifierTree     => symOf(id).flatMap(scope.closures.get).map(_.success)
    case mi: MethodInvocationTree =>
      dynamicSym(mi).flatMap(metafactoryImpl(_)).map { impl =>
        argValues(scope, buf, mi.getArguments.asScala.toList).map(Closure(impl, _))
      }
    case _ => None
  }

  // no implementation is visible for a functional value made outside the region, the JVM spins its class at run time
  private def capturedClosures(
      localTrees: Map[JSymbol.MethodSymbol, MethodTree],
      entry: MethodTree
  ): Map[JSymbol, Closure] = {
    val entrySym = methodDeclSym(entry)
    // $deserializeLambda$ holds a second call site per lambda, unrelated to where the value was made
    val trees = localTrees.values.filter(_.getName.toString != "$deserializeLambda$").toList
    val sites = trees.flatMap { m =>
      LambdaOutliner.collectTree(m) {
        case mi: MethodInvocationTree if dynamicSym(mi).flatMap(metafactoryImpl(_)).contains(entrySym) => mi :: Nil
        case _                                                                                         => Nil
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
              case (param, id: IdentifierTree) => symOf(id).flatMap(soleClosureWrite(trees, _)).map(declOf(param) -> _)
              case _                           => None
            }
            .toMap
      case _ => Map.empty
    }
  }

  private def soleClosureWrite(trees: List[MethodTree], v: JSymbol): Option[Closure] = v match {
    case _: JSymbol.VarSymbol =>
      val writes = trees.flatMap { m =>
        LambdaOutliner.collectTree(m) {
          case d: VariableTree if declOf(d) == v                     => Option(d.getInitializer).toList
          case a: AssignmentTree if symOf(a.getVariable).contains(v) => a.getExpression :: Nil
          case _                                                     => Nil
        }
      }
      writes match {
        case (mi: MethodInvocationTree) :: Nil if mi.getArguments.isEmpty =>
          dynamicSym(mi).flatMap(metafactoryImpl(_)).map(Closure(_, Nil))
        case _ => None
      }
    case _ => None
  }

  private def argValues(
      scope: Scope,
      buf: mutable.ListBuffer[p.Stmt],
      args: List[? <: ExpressionTree]
  ): Result[List[Either[p.Term, Closure]]] = args.traverse { a =>
    closureOf(scope, buf, a) match {
      case Some(r) => r.map(Right(_))
      case None    => mapExpr(scope, buf, a).map(e => Left(bind(scope, buf, e)))
    }
  }

  private def applyClosure(
      scope: Scope,
      buf: mutable.ListBuffer[p.Stmt],
      cl: Closure,
      samArgs: List[Either[p.Term, Closure]]
  ): Result[p.Expr] = {
    val impl   = cl.impl
    val values = cl.captures ++ samArgs
    for {
      _   <- refuseStub(impl)
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
        case Some(tree) => inlineImpl(scope, buf, ImplCall(impl, recv, rest, rtn), tree)
        case None       => callImpl(scope, buf, ImplCall(impl, recv, rest, rtn))
      }
    } yield {
      val named = scope.fresh(rtn)
      buf += p.Stmt.Var(named, Some(e))
      selectExpr(Nil, named)
    }
  }

  private def callImpl(
      scope: Scope,
      buf: mutable.ListBuffer[p.Stmt],
      call: ImplCall
  ): Result[p.Expr] = {
    import call.*
    values
      .traverse {
        case Left(t) => t.success
        case Right(_) =>
          (s"`${fullName(impl)}` backs a functional value that closes over another functional " +
            "value, but its body is not available to inline").fail
      }
      .flatMap { terms =>
        impl.asType.getParameterTypes.asScala.toList.traverse(JavaTyper.typer).flatMap { paramTpes =>
          val args = if (paramTpes.size == terms.size) terms.zip(paramTpes).map(coerce(scope, buf, _, _)) else terms
          intrinsicOf(scope.deps, impl, args) match {
            case Some(e) => e.success
            case None if isUserMethod(impl) =>
              scope.deps.fnSyms += impl
              scope.calls += impl
              val receiverTpeArgs =
                recv.map(r => referent(r.tpe)).collect { case p.Type.Struct(_, ts) => ts }.getOrElse(Nil)
              (lengthArgsFor(scope, impl, Nil), staticArgsFor(scope, impl)).mapN((ls, ss) =>
                p.Expr.Invoke(
                  p.Type.FnRef(p.Sym(fullName(impl))),
                  receiverTpeArgs,
                  recv.map(byReference(scope, buf, _)),
                  args.map(byReference(scope, buf, _)) ++ ls ++ ss,
                  rtn
                )
              )
            case None =>
              s"Cannot lower a functional value backed by `${fullName(impl)}`, which has no body to inline".fail
          }
        }
      }
  }

  private def bindInlineParams(
      scope: Scope,
      buf: mutable.ListBuffer[p.Stmt],
      params: List[? <: VariableTree],
      values: List[Either[p.Term, Closure]]
  ): Result[Unit] = params.zip(values).traverse_ {
    case (v, Right(c)) => scope.closures.put(declOf(v), c); ().success
    case (v, Left(t)) =>
      JavaTyper.typer(jcType(v)).map { declared =>
        val tpe   = if (referent(t.tpe) == declared) t.tpe else declared
        val named = p.Named(scope.uniqueName(v.getName.toString), tpe)
        buf += p.Stmt.Var(named, Some(p.Expr.Alias(coerce(scope, buf, t, tpe))), isMutable = true)
        scope.locals.put(declOf(v), named)
        ()
      }
  }

  private def inlineBody(
      scope: Scope,
      buf: mutable.ListBuffer[p.Stmt],
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
      buf: mutable.ListBuffer[p.Stmt],
      call: ImplCall,
      tree: MethodTree
  ): Result[p.Expr] = {
    import call.*
    inlineBody(
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
  }

  // structs hold their own members only, so an inherited method cannot be called against a subclass struct
  private def inlineInherited(
      scope: Scope,
      buf: mutable.ListBuffer[p.Stmt],
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
      buf: mutable.ListBuffer[p.Stmt],
      d: Dispatch,
      cls: JSymbol.ClassSymbol
  ): Result[p.Expr] =
    inlineAt(
      scope,
      buf,
      d,
      overrideIn(scope, cls, d.sym).getOrElse(d.sym),
      cls.flatName.toString,
      "allocated in the region"
    )

  private def superCall(scope: Scope, buf: mutable.ListBuffer[p.Stmt], d: Dispatch): Result[p.Expr] =
    inlineAt(scope, buf, d, d.sym, ownerName(d.sym), "reached through `super`")

  private def inlineAt(
      scope: Scope,
      buf: mutable.ListBuffer[p.Stmt],
      d: Dispatch,
      target: JSymbol.MethodSymbol,
      receiver: String,
      how: String
  ): Result[p.Expr] = {
    import d.*
    val refused = s"Cannot call `${fullName(sym)}` on `$receiver` $how"
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
      buf: mutable.ListBuffer[p.Stmt],
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

  private def construct(scope: Scope, buf: mutable.ListBuffer[p.Stmt], jc: JCTree.JCNewClass): Result[p.Expr] =
    for {
      tpe <- JavaTyper.typer(jcType(jc))
      struct <- tpe match {
        case s: p.Type.Struct => s.success
        case other            => s"Cannot allocate ${other.repr} with `new`".fail
      }
      ctor <- Option(jc.constructor)
        .collect { case m: JSymbol.MethodSymbol => m }
        .failIfEmpty(s"`$jc` names no resolved constructor")
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

  // an enum constant travels as its ordinal, so the state `Enum` declares never reaches a layout
  private val EnumOrObject = Set("java.lang.Object", "java.lang.Enum")

  private def superclassOf(deps: DepCollector, c: TypeElement): Result[Option[TypeElement]] =
    Option(c.getSuperclass)
      .flatMap(t => Option(deps.types.asElement(t)))
      .collect { case s: TypeElement if !EnumOrObject(s.getQualifiedName.toString) => s }
      .success

  // the nearest class from `from` up whose instances carry fields a subclass layout would have to inherit
  private def stateFrom(deps: DepCollector, from: TypeElement): Result[Option[TypeElement]] =
    ClassLayouts.stateAlong(from)(c => JavaTyper.stateFields(c).nonEmpty.success)(superclassOf(deps, _))

  private def stateAbove(deps: DepCollector, elem: TypeElement): Result[Option[TypeElement]] =
    superclassOf(deps, elem).flatMap(_.fold(Option.empty[TypeElement].success)(stateFrom(deps, _)))

  private def stateNames(c: TypeElement): String = JavaTyper.stateFields(c).map(_.getSimpleName).mkString(", ")

  private def superFieldsOf(scope: Scope, elem: JSymbol.ClassSymbol): Result[Unit] =
    stateAbove(scope.deps, elem).flatMap {
      case Some(sup) =>
        (s"Cannot allocate `${elem.flatName}`: its supertype `${sup.getQualifiedName}` carries instance " +
          s"state (${stateNames(sup)}), which this layout has no room for").fail
      case None => ().success
    }

  private def inlineCtor(
      scope: Scope,
      buf: mutable.ListBuffer[p.Stmt],
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

  private def samCall(scope: Scope, buf: mutable.ListBuffer[p.Stmt], mi: MethodInvocationTree): Option[Result[p.Expr]] =
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

  private def invoke(scope: Scope, buf: mutable.ListBuffer[p.Stmt], mi: MethodInvocationTree): Result[p.Expr] =
    dynamicSym(mi) match {
      case Some(d) if metafactoryImpl(d).isDefined =>
        ("A functional value has no device representation, so it only lowers where it is called " +
          "directly in the region; it cannot be stored in a field, returned, or handed to a method " +
          "whose body is not available").fail
      case Some(d) => s"Unsupported dynamic call site `${d.name}` (bootstrapped by ${Option(d.bsm).orNull})".fail
      case None    => samCall(scope, buf, mi).getOrElse(plainInvoke(scope, buf, mi))
    }

  private def candidateClasses(deps: DepCollector, owner: TypeElement): List[TypeElement] = {
    val ownerTpe = deps.types.erasure(owner.asType)
    (deps.classTable.values.toList ++ deps.witnessed.toList)
      .filter(c => !c.getModifiers.contains(Modifier.ABSTRACT))
      .filter(c => c.getKind == ElementKind.CLASS || c.getKind == ElementKind.ENUM || c.getKind == ElementKind.RECORD)
      .filter(c => deps.types.isSubtype(deps.types.erasure(c.asType), ownerTpe))
      .distinctBy(JavaTyper.structName(_))
      .sortBy(JavaTyper.structName(_))
  }

  private def declaredImplementation(
      deps: DepCollector,
      c: TypeElement,
      sym: JSymbol.MethodSymbol
  ): Option[JSymbol.MethodSymbol] =
    c.getEnclosedElements.asScala.collectFirst {
      case m: JSymbol.MethodSymbol
          if !m.getModifiers.contains(Modifier.ABSTRACT) && (m == sym || deps.elements.overrides(m, sym, c)) =>
        m
    }

  // a concrete method not redeclared by a class runs as declared in the owner, which is viewed through
  private def implementationIn(
      deps: DepCollector,
      c: TypeElement,
      sym: JSymbol.MethodSymbol
  ): Option[JSymbol.MethodSymbol] =
    declaredImplementation(deps, c, sym).orElse {
      Option
        .when(!sym.getModifiers.contains(Modifier.ABSTRACT))(
          LazyList
            .unfold(superclassOf(deps, c).toOption.flatten)(_.map(t => t -> superclassOf(deps, t).toOption.flatten))
            .flatMap(t => declaredImplementation(deps, t, sym).map(t -> _))
            .headOption
        )
        .flatten
        .collect { case (t, m) if t == sym.owner => m }
    }

  private def isOverridden(deps: DepCollector, sym: JSymbol.MethodSymbol): Boolean = sym.owner match {
    case owner: JSymbol.ClassSymbol =>
      candidateClasses(deps, owner).exists(c => c != owner && declaredImplementation(deps, c, sym).exists(_ != sym))
    case _ => false
  }

  // the candidate set is a lower bound: a class outside this compilation can implement the method, hence the host-written tag
  private def implementationsOf(
      deps: DepCollector,
      sym: JSymbol.MethodSymbol,
      within: Option[JSymbol.ClassSymbol] = None
  ): Result[List[(TypeElement, JSymbol.MethodSymbol)]] = sym.owner match {
    case declaring: JSymbol.ClassSymbol =>
      candidateClasses(deps, within.getOrElse(declaring)).traverse { c =>
        for {
          above <- stateAbove(deps, c)
          _ <- above.fold(().success)(s =>
            (s"`${c.getQualifiedName}` is a candidate receiver of `${fullName(sym)}` but its superclass " +
              s"`${s.getQualifiedName}` carries instance state (${stateNames(s)}), which its layout has no room for").fail
          )
          impl <- implementationIn(deps, c, sym)
            .map(c -> _)
            .failIfEmpty(
              s"`${c.getQualifiedName}` is a candidate receiver of `${fullName(sym)}` but inherits its " +
                "implementation rather than declaring one, which this lowering cannot dispatch to"
            )
        } yield impl
      }
    case _ => s"Cannot resolve implementations of `${fullName(sym)}`: it is not owned by a class".fail
  }

  private def emitDispatch(
      scope: Scope,
      buf: mutable.ListBuffer[p.Stmt],
      d: Dispatch,
      argLengths: List[Option[p.Term]]
  ): Result[p.Expr] = {
    import d.*
    for {
      self <- asSelect(p.Expr.Alias(byReference(scope, buf, recv)))
      refArgs = args.map(byReference(scope, buf, _))
      candidates <- implementationsOf(scope.deps, sym)
      _ <-
        if (candidates.nonEmpty) ().success
        else
          (s"Cannot devirtualise `${fullName(sym)}`: this compilation sees no class implementing it, " +
            "so there is nothing to compile into the kernel").fail
      // the owner's layout is replaced by the tag and the room for its largest implementation
      owner <- (sym.owner match {
        case c: JSymbol.ClassSymbol => Some(c)
        case _                      => None
      }).failIfEmpty(s"`${fullName(sym)}` is not owned by a class")
      _ <- stateFrom(scope.deps, owner).flatMap {
        case None => ().success
        case Some(c) =>
          (s"Cannot devirtualise `${fullName(sym)}`: `${c.getSimpleName}` carries instance state " +
            s"(${stateNames(c)}), which the layout shared by its implementations has no room for").fail
      }
      ownerSym = p.Sym(JavaTyper.structName(owner))
      _        = scope.deps.dispatchClasses += owner
      _ = scope.deps.owners(ownerSym) =
        (scope.deps.owners.getOrElse(ownerSym, Nil) ++ candidates.map(c => p.Sym(JavaTyper.structName(c._1)))).distinct
      arms <- candidates.traverse { (cls, impl) =>
        scope.deps.dispatchClasses += cls
        scope.deps.fnSyms += impl
        scope.calls += impl
        (lengthArgsFor(scope, impl, argLengths), staticArgsFor(scope, impl)).mapN((ls, ss) =>
          (
            p.Sym(JavaTyper.structName(cls)),
            p.Sym(fullName(impl)),
            ls ++ ss,
            Option(impl.owner).collect { case o: JSymbol.ClassSymbol if o != cls => p.Sym(JavaTyper.structName(o)) }
          )
        )
      }
      // implementations differ in the lengths and statics they take, so the call passes their union and each arm picks
      extras   = arms.flatMap(_._3).distinct
      callee   = p.Sym(fullName(sym))
      prefix   = refArgs.indices.toList
      refCount = refArgs.size
      slot     = extras.zipWithIndex.map((t, i) => t -> (refCount + i)).toMap
      armOf    = arms.map((cls, impl, own, implReceiver) => Arm(cls, impl, prefix ::: own.map(slot), implReceiver))
      _ <- scope.deps.dispatch.get(callee) match {
        case Some(known) if known != armOf =>
          s"Call sites of `${fullName(sym)}` disagree on the arguments its implementations take".fail
        case _ => scope.deps.dispatch(callee) = armOf; ().success
      }
      result = scope.fresh(rtn)
      call   = p.Expr.Invoke(p.Type.FnRef(callee), Nil, Some(self), refArgs ++ extras, rtn)
      _      = buf += p.Stmt.Var(result, Some(call), isMutable = false)
    } yield p.Expr.Alias(if (rtn == p.Type.Unit0) p.Term.Unit0Const else p.Term.Select(result, Nil, rtn))
  }

  private def methodOf(deps: DepCollector, owner: TypeElement, site: OpenSite): Option[JSymbol.MethodSymbol] =
    deps.elements.getAllMembers(owner).asScala.collectFirst {
      case m: JSymbol.MethodSymbol
          if !m.isStatic && m.getSimpleName.contentEquals(site.name) && descriptorOf(deps, m) == site.desc =>
        m
    }

  // the classes the compilation declares that a call from bytecode can reach, brought into the region as they are found
  private def sourceArms(deps: DepCollector, plan: Option[Plan])(site: OpenSite): Result[List[Arm]] =
    ClassLayouts.structName(site.owner, deps.classBytes).flatMap { name =>
      (for {
        owner <- Option(deps.elements.getTypeElement(name)).collect { case c: JSymbol.ClassSymbol => c }
        sym   <- methodOf(deps, owner, site)
      } yield implementationsOf(deps, sym, Some(owner)).flatMap(_.traverse { (cls, impl) =>
        val extras =
          plan.exists(pl => pl.statics.get(impl).exists(_.nonEmpty) || pl.lengths.get(impl).exists(_.nonEmpty))
        if (extras)
          (s"`${fullName(impl)}` is a candidate receiver of `${site.repr}` but takes array lengths or static state " +
            "that a call from bytecode cannot pass").fail
        else {
          deps.dispatchClasses += cls
          deps.fnSyms += impl
          Arm(p.Sym(JavaTyper.structName(cls)), p.Sym(fullName(impl)), List.range(0, site.arity)).success
        }
      })).getOrElse(Nil.success)
    }

  private def regionAllocated(scope: Scope, tpe: p.Type): Option[JSymbol.ClassSymbol] = referent(tpe) match {
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

  private def refuseStub(sym: JSymbol.MethodSymbol): Result[Unit] =
    InterfaceStub.identityOf(sym).traverse_ { identity =>
      InterfaceStub.refusal(s"${ownerName(sym)}.${sym.name}", identity).fail
    }

  private def plainInvoke(scope: Scope, buf: mutable.ListBuffer[p.Stmt], mi: MethodInvocationTree): Result[p.Expr] =
    boxIdentity(scope, buf, mi).getOrElse {
      for {
        symAndRecv <- mi.getMethodSelect match {
          case fa: MemberSelectTree =>
            methodSymOf(fa).flatMap { msym =>
              if (msym.isStatic) (msym, Option.empty[p.Term]).success
              else mapExpr(scope, buf, fa.getExpression).map(r => (msym, Option(bind(scope, buf, r))))
            }
          case id: IdentifierTree =>
            methodSymOf(id).flatMap { msym =>
              if (msym.isStatic) (msym, Option.empty[p.Term]).success
              else
                scope.thisOpt
                  .map(self => (msym, Option(selectTerm(Nil, self): p.Term)))
                  .failIfEmpty(s"Cannot resolve receiver for `${id.getName}` outside an instance method")
            }
          case other => s"Unsupported method selector ${other.getKind}".fail
        }
        (sym, recv) = symAndRecv
        _   <- refuseStub(sym)
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
      buf: mutable.ListBuffer[p.Stmt],
      sym: JSymbol.MethodSymbol,
      recv: Option[p.Term],
      argTrees: List[? <: ExpressionTree],
      rtn: p.Type
  ): Result[p.Expr] = scope.deps.treeOf(sym) match {
    case Some(tree) =>
      argValues(scope, buf, argTrees).flatMap(vs => inlineImpl(scope, buf, ImplCall(sym, recv, vs, rtn), tree))
    case None =>
      (s"`${fullName(sym)}` is handed a functional value, which has no device representation, and " +
        "its body is not available to lower into the call site").fail
  }

  // an object element must stay an lvalue path into its array: Index would copy it before the callee can take its address
  private def callArg(scope: Scope, buf: mutable.ListBuffer[p.Stmt], a: ExpressionTree): Result[p.Expr] =
    a match {
      case par: ParenthesizedTree => callArg(scope, buf, par.getExpression)
      case aa: ArrayAccessTree =>
        JavaTyper.typer(jcType(aa)).flatMap {
          case _: p.Type.Struct => selectPath(scope, buf, aa).map(p.Expr.Alias(_))
          case _                => mapExpr(scope, buf, a)
        }
      case _ => mapExpr(scope, buf, a)
    }

  private def directCall(
      scope: Scope,
      buf: mutable.ListBuffer[p.Stmt],
      mi: MethodInvocationTree,
      sym: JSymbol.MethodSymbol,
      recv: Option[p.Term],
      rtn: p.Type
  ): Result[p.Expr] =
    for {
      paramTpes <- sym.asType.getParameterTypes.asScala.toList.traverse(JavaTyper.typer)
      argTrees = mi.getArguments.asScala.toList
      lowered <- argTrees.traverse { a =>
        callArg(scope, buf, a).map { e =>
          val len = symOf(a)
            .flatMap(scope.arrayLengths.get)
            .orElse(a match { case _: NewArrayTree => scope.newArrayLengths.get(a); case _ => None })
          (bind(scope, buf, e), len)
        }
      }
      raw = lowered.map(_._1)
      _ = argTrees.zipWithIndex.foreach { (a, i) =>
        symOf(a).flatMap(scope.paramOf.get).foreach(j => scope.argFlows += ((sym, i, j)))
      }
      argLengths  = lowered.map(_._2)
      viaBytecode = recv.isEmpty && isBytecodeCallee(scope.deps, sym)
      args = if (paramTpes.size == raw.size) raw.zip(paramTpes).map(coerceArg(scope, buf, _, _, viaBytecode)) else raw
      e <- intrinsicOf(scope.deps, sym, args).fold(
        invokeOf(scope, buf, sym, recv, args, argLengths, rtn, !isSuperCall(mi))
      )(
        _.success
      )
    } yield e

  private def isSuperCall(mi: MethodInvocationTree): Boolean = mi.getMethodSelect match {
    case fa: MemberSelectTree =>
      fa.getExpression match {
        case id: IdentifierTree      => id.getName.contentEquals("super")
        case outer: MemberSelectTree => outer.getIdentifier.contentEquals("super")
        case _                       => false
      }
    case _ => false
  }

  private def overridable(sym: JSymbol.MethodSymbol): Boolean =
    isUserMethod(sym) && !sym.isStatic && !sym.isConstructor &&
      (sym.getModifiers.asScala.toSet & Set(Modifier.PRIVATE, Modifier.FINAL)).isEmpty &&
      !sym.owner.getModifiers.contains(Modifier.FINAL)

  private def invokeOf(
      scope: Scope,
      buf: mutable.ListBuffer[p.Stmt],
      sym: JSymbol.MethodSymbol,
      recv: Option[p.Term],
      args: List[p.Term],
      argLengths: List[Option[p.Term]],
      rtn: p.Type,
      virtual: Boolean
  ): Result[p.Expr] =
    if (recv.isEmpty && isBytecodeCallee(scope.deps, sym))
      bytecodeCall(scope.deps, sym, args.map(byReference(scope, buf, _)), rtn).success
    else if (sym.name.toString == "<init>")
      if (args.isEmpty) (p.Expr.Alias(p.Term.Unit0Const): p.Expr).success
      else s"Unsupported constructor delegation to `${fullName(sym)}`".fail
    else if (ownerName(sym) == EnumOwner && recv.isDefined)
      if (sym.name.toString == "ordinal" && args.isEmpty) p.Expr.Alias(recv.get).success
      else s"Unsupported `${fullName(sym)}`: an enum value travels as its ordinal alone".fail
    else if (isRecordAccessor(sym) && args.isEmpty && recv.isDefined)
      asSelect(p.Expr.Alias(recv.get))
        .map(sel => p.Expr.Alias(fieldOf(sel, sym.name.toString, rtn)))
    else if (recv.isDefined && sym.getModifiers.contains(Modifier.ABSTRACT))
      regionAllocated(scope, recv.get.tpe) match {
        case Some(cls) =>
          overrideIn(scope, cls, sym)
            .failIfEmpty(s"`${cls.flatName}` declares no override of `${fullName(sym)}` to call")
            .flatMap { impl =>
              scope.deps.fnSyms += impl
              scope.calls += impl
              (lengthArgsFor(scope, impl, argLengths), staticArgsFor(scope, impl)).mapN((ls, ss) =>
                p.Expr.Invoke(
                  p.Type.FnRef(p.Sym(fullName(impl))),
                  Nil,
                  recv.map(byReference(scope, buf, _)),
                  args.map(byReference(scope, buf, _)) ++ ls ++ ss,
                  rtn
                )
              )
            }
        case None => emitDispatch(scope, buf, Dispatch(sym, recv.get, args, rtn), argLengths)
      }
    else
      recv.filter(_ => virtual).flatMap(r => regionAllocated(scope, r.tpe)).filter(_ != sym.owner) match {
        case Some(cls) => inheritedCall(scope, buf, Dispatch(sym, recv.get, args, rtn), cls)
        case None if recv.isDefined && !virtual && isUserMethod(sym) =>
          superCall(scope, buf, Dispatch(sym, recv.get, args, rtn))
        case None if recv.isDefined && virtual && overridable(sym) && isOverridden(scope.deps, sym) =>
          emitDispatch(scope, buf, Dispatch(sym, recv.get, args, rtn), argLengths)
        case None =>
          val receiverTpeArgs =
            recv.map(r => referent(r.tpe)).collect { case p.Type.Struct(_, ts) => ts }.getOrElse(Nil)
          val user = isUserMethod(sym)
          if (user) { scope.deps.fnSyms += sym; scope.calls += sym }
          val passed  = if (user) recv.map(byReference(scope, buf, _)) else recv
          val passedA = if (user) args.map(byReference(scope, buf, _)) else args
          (lengthArgsFor(scope, sym, argLengths), staticArgsFor(scope, sym)).mapN((ls, ss) =>
            p.Expr.Invoke(p.Type.FnRef(p.Sym(fullName(sym))), receiverTpeArgs, passed, passedA ++ ls ++ ss, rtn)
          )
      }

  private def mapStmt(scope: Scope, buf: mutable.ListBuffer[p.Stmt], t: StatementTree): Result[Unit] = t match {
    case b: BlockTree => b.getStatements.asScala.toList.traverse_(mapStmt(scope, buf, _))
    case v: VariableTree if Option(v.getInitializer).exists(isClosureTree(scope, _)) =>
      closureOf(scope, buf, v.getInitializer).get.map { c =>
        scope.closures.put(declOf(v), c)
        ()
      }
    case v: VariableTree =>
      for {
        declared <- JavaTyper.typer(jcType(v))
        value    <- Option(v.getInitializer).traverse(mapExpr(scope, buf, _).map(bind(scope, buf, _)))
        // a local declared at a supertype but built with new keeps the concrete type, widening drops its fields
        tpe = (declared, value.map(_.tpe)) match {
          case (p.Type.Struct(_, _), Some(actual @ p.Type.Struct(_, _))) => actual
          case (p.Type.Struct(name, _), Some(actual @ p.Type.Ptr(p.Type.Struct(pointee, _), _))) if pointee == name =>
            actual
          case _ => declared
        }
        named = p.Named(scope.uniqueName(v.getName.toString), tpe)
        _     = scope.locals.put(declOf(v), named)
        init  = value.map(t => p.Expr.Alias(coerce(scope, buf, t, tpe)))
        // aliasing keeps the length: enhanced-for binds the subject to a local before measuring it
        aliased = Option(v.getInitializer).flatMap(symOf(_))
        _ = aliased
          .flatMap(scope.arrayLengths.get)
          .orElse(Option(v.getInitializer).map(unparenthesised(_)).flatMap(scope.newArrayLengths.get))
          .foreach(scope.arrayLengths.put(declOf(v), _))
        _ = aliased
          .flatMap(scope.rowLengths.get)
          .orElse(Option(v.getInitializer).map(unparenthesised(_)).flatMap(scope.newRowLengths.get))
          .foreach(scope.rowLengths.put(declOf(v), _))
        _ = aliased.flatMap(scope.paramOf.get).foreach(scope.paramOf.put(declOf(v), _))
      } yield { buf += p.Stmt.Var(named, init, isMutable = true); () }
    case e: ExpressionStatementTree => mapExpr(scope, buf, e.getExpression).map(_ => ())
    case i: IfTree =>
      for {
        cond <- mapExpr(scope, buf, i.getCondition).map(bind(scope, buf, _))
        thenBuf = mutable.ListBuffer.empty[p.Stmt]
        _ <- mapStmt(scope, thenBuf, i.getThenStatement)
        elseBuf = mutable.ListBuffer.empty[p.Stmt]
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
        case nc: NewClassTree =>
          namedRaise(scope, nc).map { raised =>
            bind(scope, buf, raised); ()
          }
        case _ => "Unsupported `throw`; PolyRegion kernels cannot throw".fail
      }
    case _: SynchronizedTree => "Unsupported `synchronized`; PolyRegion kernels have no monitors".fail
    case other               => s"Unsupported statement: ${other.getKind} `$other`".fail
  }

  private def allocRows(
      scope: Scope,
      buf: mutable.ListBuffer[p.Stmt],
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

  private def propagateLabels(scope: Scope, buf: mutable.ListBuffer[p.Stmt]): Unit =
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
    val condBuf = mutable.ListBuffer.empty[p.Stmt]
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
      buf: mutable.ListBuffer[p.Stmt],
      cond: Option[ExpressionTree],
      body: List[StatementTree],
      update: List[ExpressionStatementTree]
  ): Result[Unit] = for {
    anf <- condAnf(scope, cond)
    (condTerm, hoisted, reeval) = anf
    bodyBuf                     = mutable.ListBuffer.empty[p.Stmt]
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

  private def emitLabelled(scope: Scope, buf: mutable.ListBuffer[p.Stmt], l: LabeledStatementTree): Result[Unit] = {
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
        val bodyBuf = mutable.ListBuffer.empty[p.Stmt]
        scope.withLabel(ctx)(mapStmt(scope, bodyBuf, other)).map { _ =>
          buf ++= runOnce(scope, bodyBuf.toList)
          propagateLabels(scope, buf)
          ()
        }
    }
  }

  private def emitLabelledLoop(
      scope: Scope,
      buf: mutable.ListBuffer[p.Stmt],
      ctx: LabelCtx,
      cond: Option[ExpressionTree],
      body: StatementTree,
      update: List[ExpressionStatementTree]
  ): Result[Unit] = for {
    anf <- condAnf(scope, cond)
    (condTerm, hoisted, reeval) = anf
    bodyBuf                     = mutable.ListBuffer.empty[p.Stmt]
    _ <- scope.withLabel(ctx) {
      scope.inLoop(LoopCtx(update, reeval, Some(ctx)))(mapStmt(scope, bodyBuf, body))
    }
    updBuf = mutable.ListBuffer.empty[p.Stmt]
    _ <- scope.inLoop(LoopCtx(update, reeval))(update.traverse_(mapStmt(scope, updBuf, _)))
  } yield {
    val escapeBuf = mutable.ListBuffer.empty[p.Stmt]
    propagateLabels(scope, escapeBuf)
    val wrapped = (p.Stmt.Mut(ctx.cnt, p.Expr.Alias(p.Term.Bool1Const(false))) ::
      runOnce(scope, bodyBuf.toList)) :+ p.Stmt.Cond(ctx.brk, List(p.Stmt.Break), Nil)
    buf ++= hoisted
    buf ++= reeval
    buf += p.Stmt.While(condTerm, wrapped ++ escapeBuf.toList ++ updBuf.toList ++ reeval)
    propagateLabels(scope, buf)
    ()
  }

  private def emitDoWhile(scope: Scope, buf: mutable.ListBuffer[p.Stmt], d: DoWhileLoopTree): Result[Unit] = {
    val flag                   = scope.fresh(p.Type.Bool1)
    val flagSel: p.Term.Select = p.Term.Select(flag, Nil, p.Type.Bool1)
    for {
      anf <- condAnf(scope, Some(d.getCondition))
      (condTerm, hoisted, reeval) = anf
      tail                        = reeval :+ p.Stmt.Mut(flagSel, p.Expr.Alias(condTerm))
      bodyBuf                     = mutable.ListBuffer.empty[p.Stmt]
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
      buf: mutable.ListBuffer[p.Stmt],
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
      buf: mutable.ListBuffer[p.Stmt],
      selector: p.Term,
      cases: List[CaseTree],
      result: Option[p.Term.Select]
  ): Result[Unit] = for {
    guards <- caseGuards(scope, buf, selector, cases)
    (matchOf, noMatch)        = guards
    started                   = scope.fresh(p.Type.Bool1)
    startedSel: p.Term.Select = p.Term.Select(started, Nil, p.Type.Bool1)
    setStarted                = p.Stmt.Mut(startedSel, p.Expr.Alias(p.Term.Bool1Const(true)))
    bodyBuf                   = mutable.ListBuffer.empty[p.Stmt]
    _ = bodyBuf += p.Stmt.Var(started, Some(p.Expr.Alias(p.Term.Bool1Const(false))), isMutable = true)
    _ <- cases.traverse_ { c =>
      bodyBuf += p.Stmt.Cond(matchOf.getOrElse(c, noMatch), List(setStarted), Nil)
      val armBuf = mutable.ListBuffer.empty[p.Stmt]
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
      buf: mutable.ListBuffer[p.Stmt],
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

  private def switchExpression(scope: Scope, buf: mutable.ListBuffer[p.Stmt], s: SwitchExpressionTree): Result[p.Expr] =
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

  private def emitSwitch(scope: Scope, buf: mutable.ListBuffer[p.Stmt], s: SwitchTree): Result[Unit] =
    for {
      selector <- mapExpr(scope, buf, s.getExpression).map(bind(scope, buf, _))
      _        <- emitCases(scope, buf, selector, s.getCases.asScala.toList, None)
    } yield ()

  private def passed(isEntry: Boolean, byReference: Boolean, tpe: p.Type): p.Type =
    if (!isEntry || byReference) referenceTo(tpe) else tpe

  private enum Role {
    case Entry
    case ReferencingEntry
    case Callee
  }

  private def compileMethod(
      m: MethodTree,
      deps: DepCollector,
      role: Role,
      plan: Option[Plan]
  ): Result[(p.Function, List[JSymbol.VarSymbol], List[Captured])] = m match {
    case md: JCTree.JCMethodDecl =>
      JavaTyper
        .typer(md.sym.getReturnType)
        .flatMap(compileBody(md, deps, role != Role.Callee, role != Role.Entry, plan, _))
  }

  private def compileBody(
      md: JCTree.JCMethodDecl,
      deps: DepCollector,
      isEntry: Boolean,
      byReference: Boolean,
      plan: Option[Plan],
      rtn: p.Type
  ): Result[(p.Function, List[JSymbol.VarSymbol], List[Captured])] = {
    val receiver =
      if (md.sym.isStatic) None
      else
        Some(p.Arg(p.Named("this", passed(isEntry, byReference, p.Type.Struct(p.Sym(ownerName(md.sym)), Nil)))))
    val scope      = Scope(deps, isEntry, plan, rtn, receiver.map(_.named))
    val params     = md.getParameters.asScala.toVector
    val shift      = if (receiver.isDefined) 1 else 0
    val argSources = mutable.ListBuffer.empty[Captured]
    for {
      typed <- params.toList.traverse { v =>
        deps.capturedClosures.get(declOf(v)).filter(_ => isEntry) match {
          case Some(c) => (Left(c): Either[Closure, p.Type]).success
          case None    => JavaTyper.typer(jcType(v)).map(tpe => Right(passed(isEntry, byReference, tpe)))
        }
      }
      args = {
        val out = mutable.ListBuffer.empty[p.Arg]
        params.zip(typed).zipWithIndex.foreach {
          case ((v, Left(c)), _) => scope.closures.put(declOf(v), c)
          case ((v, Right(tpe)), i) =>
            val named = p.Named(scope.uniqueName(v.getName.toString), tpe)
            scope.locals.put(declOf(v), named)
            scope.paramOf.put(declOf(v), i)
            argSources += Captured(named.symbol, Capture.Arg(i + shift))
            out += p.Arg(named)
        }
        out.toList
      }
      staticTypes <- scope.staticParamsOf(md.sym).traverse(v => JavaTyper.typer(v.asType).map(v -> _))
      staticArgs = {
        val out = mutable.ListBuffer.empty[p.Arg]
        staticTypes.foreach { (v, tpe) =>
          val named = p.Named(s"#static${scope.staticFields.size}", p.Type.Ptr(tpe, p.Type.Space.Global))
          scope.staticFields.put(v, named)
          out += p.Arg(named)
        }
        out.toList
      }
      lengths =
        if (isEntry) params.zipWithIndex.collect { case (v, i) if jcType(v).getKind == TypeKind.ARRAY => (v, i) }
        else scope.lengthParamsOf(md.sym).map(i => (params(i), i))
      lengthArgs = {
        val out = mutable.ListBuffer.empty[p.Arg]
        lengths.foreach { (v, i) =>
          val named = p.Named(s"#len$i", p.Type.IntS32)
          scope.arrayLengths.put(declOf(v), p.Term.Select(named, Nil, p.Type.IntS32))
          argSources += Captured(named.symbol, Capture.Length(i + shift))
          out += p.Arg(named)
        }
        out.toList
      }
      rowLengthTpe = p.Type.Ptr(p.Type.IntS32, p.Type.Space.Global)
      rowLengths = {
        val out = mutable.ListBuffer.empty[(Int, p.Arg)]
        if (isEntry)
          params.zipWithIndex.foreach { (v, i) =>
            if (isArrayOfArrays(jcType(v))) {
              val named = p.Named(s"#rowlen$i", rowLengthTpe)
              scope.rowLengths.put(declOf(v), RowLength.PerRow(p.Term.Select(named, Nil, named.tpe)))
              argSources += Captured(named.symbol, Capture.RowLengths(i + shift))
              out += ((i, p.Arg(named)))
            }
          }
        out.toList
      }
      buf = mutable.ListBuffer.empty[p.Stmt]
      _ <- Option(md.getBody).traverse_(b => mapStmt(scope, buf, b))
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
      scope.staticFields.keys.toList,
      receiver.map(r => Captured(r.named.symbol, Capture.Receiver)).toList ++ argSources ++
        scope.staticFields.map((v, named) => Captured(named.symbol, Capture.StaticField(ownerName(v), v.name.toString)))
    )
  }

  // TODO overloads collapse on fullName
  private def compileTransitive(
      trees: Trees,
      deps: DepCollector,
      entrySym: JSymbol.MethodSymbol,
      plan: Option[Plan]
  ): Result[List[p.Function]] = {
    // GENERATE-phase trees are already lowered so getTree is null, callees resolve from methodTable
    def calleeTree(sym: JSymbol.MethodSymbol): Option[MethodTree] =
      deps.treeOf(sym).orElse(Option(trees.getTree(sym)).collect { case mt: MethodTree => mt })
    @tailrec def go(seen: Set[JSymbol.MethodSymbol], compiled: List[p.Function]): Result[List[p.Function]] =
      deps.fnSyms.find(!seen(_)) match {
        case None => compiled.reverse.success
        case Some(sym) =>
          calleeTree(sym).filter(bodied(_)) match {
            case Some(mt) =>
              compileMethod(mt, deps, Role.Callee, plan) match {
                case Right((fn, _, _)) => go(seen + sym, fn :: compiled)
                case Left(e)           => Left(e)
              }
            case None =>
              debug(s"[PolyAST] skip dependency (no reachable body): ${fullName(sym)}")
              go(seen + sym, compiled)
          }
      }
    go(Set(entrySym), Nil)
  }

  // TODO collect struct refs nested in bodies too
  private def structSyms(fns: List[p.Function]): List[p.Sym] = {
    def sig(fn: p.Function) =
      (fn.receiver.toList ++ fn.args ++ fn.moduleCaptures ++ fn.termCaptures).map(_.named.tpe) :+ fn.rtn
    fns.flatMap(sig(_)).flatMap(JavaTyper.structSyms).distinct
  }

  private def saturate[A, B: Ordering](
      seed: Iterable[(JSymbol.MethodSymbol, Iterable[A])]
  )(grow: Map[JSymbol.MethodSymbol, ListSet[A]] => Map[JSymbol.MethodSymbol, ListSet[A]])(
      key: A => B
  ): Map[JSymbol.MethodSymbol, List[A]] = {
    @tailrec def fix(acc: Map[JSymbol.MethodSymbol, ListSet[A]]): Map[JSymbol.MethodSymbol, ListSet[A]] = {
      val next = grow(acc)
      if (next == acc) acc else fix(next)
    }
    val start = seed.foldLeft(Map.empty[JSymbol.MethodSymbol, ListSet[A]]) { case (acc, (f, as)) =>
      acc.updated(f, acc.getOrElse(f, ListSet.empty[A]) ++ as)
    }
    fix(start).map((f, as) => f -> as.toList.sortBy(key))
  }

  // only the entry can grow a capture, so a function touching a static takes the buffer as a parameter its caller forwards
  private def planStatics(deps: DepCollector): Map[JSymbol.MethodSymbol, List[JSymbol.VarSymbol]] =
    saturate(
      (deps.directStatics.keys ++ deps.callees.keys).map(f => f -> deps.directStatics.getOrElse(f, Nil))
    ) { reaches =>
      reaches.map { (f, fs) =>
        f -> deps.callees.getOrElse(f, Nil).foldLeft(fs)((acc, g) => acc ++ reaches.getOrElse(g, ListSet.empty))
      }
    }(fullName(_))

  private def planLengths(deps: DepCollector): Map[JSymbol.MethodSymbol, List[Int]] =
    saturate(deps.measuredParams) { needs =>
      deps.lengthFlows.foldLeft(needs) { case (outer, (caller, flows)) =>
        flows.foldLeft(outer) { case (acc, (callee, calleeIdx, callerIdx)) =>
          if (acc.get(callee).exists(_.contains(calleeIdx)))
            acc.updated(caller, acc.getOrElse(caller, ListSet.empty[Int]) + callerIdx)
          else acc
        }
      }
    }(identity)

  case class Sources(
      env: ProcessingEnvironment,
      trees: Trees,
      classTable: Map[String, TypeElement],
      methodTable: Map[String, MethodTree],
      localTrees: Map[JSymbol.MethodSymbol, MethodTree]
  )

  // classpath jars through javac's own file manager, the running JDK for everything it ships
  private def classBytesOf(env: ProcessingEnvironment): BytecodeResolver.ClassBytes = {
    val manager = env match {
      case j: JavacProcessingEnvironment => Option(j.getContext.get(classOf[JavaFileManager]))
      case _                             => None
    }
    owner =>
      manager
        .flatMap(m => Option(m.getJavaFileForInput(StandardLocation.CLASS_PATH, owner, JavaFileObject.Kind.CLASS)))
        .map { f =>
          val in = f.openInputStream()
          try in.readAllBytes()
          finally in.close()
        }
        .orElse(BytecodeResolver.runtimeClasses(owner))
  }

  // lowering a candidate can reach more bytecode, and bytecode can reach more candidates, so both run until neither grows
  private def settle(
      trees: Trees,
      deps: DepCollector,
      entrySym: JSymbol.MethodSymbol,
      plan: Option[Plan]
  ): Result[(List[p.Function], BytecodeResolver.Resolved)] =
    compileTransitive(trees, deps, entrySym, plan).flatMap { fns =>
      val before = deps.fnSyms.size
      BytecodeResolver.resolve(deps.bytecodeRoots.toList, deps.classBytes, sourceArms(deps, plan)).flatMap { resolved =>
        if (deps.fnSyms.size > before) settle(trees, deps, entrySym, plan) else (fns, resolved).success
      }
    }

  def compileRegion(sources: Sources, entry: MethodTree, byReference: Boolean = false): Result[Region] = {
    val role = if (byReference) Role.ReferencingEntry else Role.Entry
    import sources.*
    val entrySym = methodDeclSym(entry)
    val deps = DepCollector(
      localTrees,
      methodTable,
      classTable,
      env.getElementUtils,
      env.getTypeUtils,
      capturedClosures(localTrees, entry),
      classBytesOf(env)
    )
    for {
      // exploratory lowering, discarded: learns the call graph and statics needed to size signatures
      _ <- compileMethod(entry, deps, role, None)
      _ <- settle(trees, deps, entrySym, plan = None)
      plan = Some(Plan(planStatics(deps), planLengths(deps)))
      _    = deps.dispatch.clear()
      entryAndStatics <- compileMethod(entry, deps, role, plan)
      (entryFn, _, captures) = entryAndStatics
      settled <- settle(trees, deps, entrySym, plan = plan)
      (sourceFns, bytecode) = settled
      _                     = bytecode.dispatch.foreach((callee, arms) => deps.dispatch(callee) = arms)
      _ = bytecode.owners.foreach((owner, impls) =>
        deps.owners(owner) = (deps.owners.getOrElse(owner, Nil) ++ impls).distinct
      )
      depFns = IntrinsicTable.dedupe(sourceFns ++ bytecode.functions ++ deps.nativeFns.toList)
      signatureStructs <- structSyms(entryFn :: depFns).flatTraverse { sym =>
        classTable.get(sym.fqn.mkString(".")) match {
          case Some(elem) => JavaTyper.lowerStruct(elem, deps.outerRefs.toSet).map(List(_))
          case None       => List.empty[p.StructDef].success
        }
      }
      allocatedStructs <- (deps.witnessed.toList ++ deps.dispatchClasses.toList).distinct.traverse { elem =>
        JavaTyper.lowerStruct(elem, deps.outerRefs.toSet)
      }
      structs <- ClassLayouts.merge((signatureStructs ++ allocatedStructs).distinct, bytecode.structs)
      tags = Devirtualise.tagsOf(deps.dispatch.values.flatten.map(_.cls))
      candidates = deps.dispatch.view.mapValues { arms =>
        arms.map(a =>
          Devirtualise.Candidate(
            p.Type.Struct(a.cls, Nil),
            a.impl,
            tags(a.cls),
            Some(a.args),
            a.implReceiver.map(p.Type.Struct(_, Nil))
          )
        )
      }.toMap
    } yield Region(
      p.Program(Some(entryFn), depFns, structs.distinct),
      captures,
      tags.map((cls, tag) => cls.fqn.mkString(".") -> tag),
      candidates,
      deps.owners.toMap,
      deps.outerRefs.toSet
    )
  }
}
