package polyregion.scalalang

import polyregion.ast.pass.Devirtualise
import polyregion.ast.PolyAST as p

import scala.util.Try

object Dispatch {

  case class Site(method: p.Sym, receivers: Set[p.Sym], arms: List[Arm])
  case class Arm(cls: p.Sym, impl: p.Sym)

  def sealedDescendants(using q: Quoted)(sym: q.Symbol): List[q.Symbol] =
    Try(sym.children).getOrElse(Nil).flatMap(c => c :: sealedDescendants(c))

  def compilationUnit(using q: Quoted): List[q.Symbol] = {
    val source   = Try(q.Position.ofMacroExpansion.sourceFile.path).toOption
    val top      = Retyper.ownersList(q.Symbol.spliceOwner).find(o => o.isClassDef && o.maybeOwner.isPackageDef)
    val inSource = (s: q.Symbol) => s.pos.map(_.sourceFile.path).zip(source).exists(_ == _)
    val siblings = top.toList ++ top.toList.flatMap { t =>
      Try(t.maybeOwner.declarations)
        .getOrElse(Nil)
        .flatMap(s => List(s, s.moduleClass))
        .filter(s => !s.isNoSymbol && s.isClassDef && inSource(s))
    }
    siblings.distinct.flatMap { s =>
      Try(q.collectTree(s.tree) {
        case cd: q.ClassDef => cd.symbol :: Nil
        case _              => Nil
      }).getOrElse(Nil)
    }.distinct
  }

  def isGeneric(using q: Quoted)(cls: q.Symbol): Boolean = Retyper.clsTypeCtorNames(cls).nonEmpty

  private def isConcrete(using q: Quoted)(c: q.Symbol): Boolean =
    c.isClassDef && !c.flags.is(q.Flags.Abstract) && !c.flags.is(q.Flags.Trait) && !c.flags.is(q.Flags.Module)

  private def subtype(using q: Quoted)(c: q.Symbol, owner: q.Symbol): Boolean =
    Try(c.typeRef.baseClasses.contains(owner)).getOrElse(false)

  private def implementation(using q: Quoted)(method: q.Symbol, cls: q.Symbol): Option[q.Symbol] =
    Some(method.overridingSymbol(cls))
      .filter(s => !s.isNoSymbol && s.isDefDef && s.maybeOwner == cls)
      .orElse(Option.when(method.maybeOwner == cls && isConcrete(cls) && method.isDefDef)(method))

  def classesOf(using q: Quoted)(method: q.Symbol, known: List[q.Symbol]): List[(q.Symbol, q.Symbol)] = {
    val owner    = method.maybeOwner
    val universe = (sealedDescendants(owner) ::: known).distinct
    universe
      .filter(c => isConcrete(c) && subtype(c, owner))
      .flatMap(c => implementation(method, c).map(c -> _))
      .sortBy(_._1.fullName)
  }

  def isVirtual(using q: Quoted)(method: q.Symbol, arms: List[(q.Symbol, q.Symbol)]): Boolean = {
    val ownerFlags = method.maybeOwner.flags
    arms.nonEmpty && method.isDefDef && !method.flags.is(q.Flags.Private) && !method.flags.is(q.Flags.Final) &&
    (ownerFlags.is(q.Flags.Trait) || ownerFlags.is(q.Flags.Abstract)) &&
    (method.flags.is(q.Flags.Deferred) || arms.exists(_._2 != method))
  }

  def tagsOf(sites: List[Site]): Map[p.Sym, Int] = Devirtualise.tagsOf(sites.flatMap(_.arms.map(_.cls)))

  def table(sites: List[Site], tags: Map[p.Sym, Int]): Map[p.Sym, List[Devirtualise.Candidate]] =
    sites.map { site =>
      site.method -> site.arms.map(a => Devirtualise.Candidate(p.Type.Struct(a.cls, Nil), a.impl, tags(a.cls)))
    }.toMap

  def isRoot(sdef: p.StructDef, tags: Map[p.Sym, Int]): Boolean =
    !tags.contains(sdef.name) && sdef.members.exists(_.symbol == p.Conventions.DispatchTag)

  def candidateReprs(using q: Quoted)(root: q.TypeRepr, tags: Map[p.Sym, Int]): List[(p.Sym, q.TypeRepr)] = {
    val owner = root.widenTermRefByName.typeSymbol
    (sealedDescendants(owner) ::: compilationUnit).distinct
      .filter(c => isConcrete(c) && subtype(c, owner))
      .map(c => Retyper.structName0(c) -> c.typeRef)
      .filter((name, _) => tags.contains(name))
  }

  def hostMembers(sdef: p.StructDef): List[p.Named] = sdef.members.filterNot(_.symbol.startsWith("#"))

  def tagMetadata(tags: Map[p.Sym, Int]): List[p.MetaEntry] =
    tags.toList.sortBy(_._2).map((cls, tag) => p.MetaEntry(p.Conventions.DispatchTagMeta + cls.fqcn, tag.toString))

  def tagsFrom(program: p.Program): Map[p.Sym, Int] =
    program.metadata.collect {
      case p.MetaEntry(key, value) if key.startsWith(p.Conventions.DispatchTagMeta) =>
        p.Sym(key.stripPrefix(p.Conventions.DispatchTagMeta)) -> value.toInt
    }.toMap

  def decorate(defs: List[p.StructDef], sites: List[Site], tags: Map[p.Sym, Int]): List[p.StructDef] =
    Devirtualise.decorate(
      defs,
      sites
        .flatMap { site =>
          val classes = site.arms.map(_.cls)
          site.receivers.map(_ -> classes)
        }
        .groupMap(_._1)(_._2)
        .view
        .mapValues(_.flatten.distinct)
        .toMap,
      tags
    )
}
