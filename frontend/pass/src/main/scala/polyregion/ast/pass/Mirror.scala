package polyregion.ast.pass

import polyregion.ast.{Log, PolyAST as p, *}
import polyregion.ast.PolyAST.Conventions.RuntimeAbi

// appends host prelude/postlude functions that deep-mirror a pointer capture to the device and read mutations
// back, walking the capture struct field-by-field and emitting SMA mirror ABI calls; every pointer field is
// null-guarded so only live pointers are mirrored/patched
// examples:
//   no capture struct       ->  prelude: alloc(capture); postlude: read_alloc(capture)
//   struct { int *p; }      ->  d = ensure_min(p, sizeof int); patch(remote, offsetof p, d)     // read_alloc(p)
//   struct { float **pp; }  ->  d = ensure_deep(pp, depth); patch(...)                          // read_deep(pp, depth)
//   struct { Node *n; }     ->  pool_reset; pool_ptr(self-offsets); mirror_graph(n); patch(...) // read_graph(n)
//   struct { Inner *q; }    ->  ensure(q); patch(...); for k in pointee_bytes/sizeof Inner: mirror q[k] fields
//   struct { Inner v; }     ->  recurse v's pointer fields at offsetof v, no patch (inline value)
// edge cases:
//   null pointer field          ->  guarded out, not mirrored or patched
//   already-seen struct sym     ->  not recursed (cycle/dup guard via seen set)
//   scalar pointee, O3-reused   ->  ensure_min floors at AST size so a stale-small reflect record can't truncate
//   array-of-structs pointee    ->  element count = max(pointee_bytes, sizeof elem) / sizeof elem
//   postlude read-alloc dedup   ->  visit_clear resets the visited set per postlude
final case class Mirror(id: String = "") extends ProgramPass derives PassArgCodec {

  override def phase: p.PassPhase = p.PassPhase.PostMono

  private def add(a: p.Term, b: p.Term): p.Expr                = p.Expr.IntrOp(p.Intr.Add(a, b, U64))
  private def u64v(name: String, e: p.Expr): (p.Named, p.Stmt) = vlet(name, U64, e)

  // shared per-pointer-field preamble: bind fp/fpi/nn/fp8 for `hostRoot.steps.mn` and null-guard `inner`
  // (which gets the field pointer + its byte-ptr cast) in a Cond on non-null
  private def ptrFieldGuard(
      hostRoot: p.Named,
      hostSteps: List[p.PathStep],
      mn: String,
      tag: String,
      comp: p.Type,
      space: p.Type.Space
  )(inner: (p.Named, p.Named) => List[p.Stmt]): List[p.Stmt] = {
    val ptrTpe = p.Type.Ptr(comp, space)
    val (fp, fpDecl) =
      vlet(s"fp_$tag", ptrTpe, p.Expr.Alias(p.Term.Select(hostRoot, hostSteps :+ p.PathStep.Field(mn), ptrTpe)))
    val (fpi, fpiDecl) = u64v(s"fpi_$tag", p.Expr.Cast(sel(fp), U64))
    val (nn, nnDecl) = vlet(s"nn_$tag", p.Type.Bool1, p.Expr.IntrOp(p.Intr.LogicNeq(sel(fpi), p.Term.IntU64Const(0L))))
    val (fp8, fp8Decl) = vlet(s"fp8_$tag", BytePtr, p.Expr.Cast(sel(fp), BytePtr))
    List(fpDecl, fpiDecl, nnDecl, p.Stmt.Cond(sel(nn), fp8Decl :: inner(fp, fp8), Nil))
  }

  private def captureStruct(program: p.Program): Option[(p.Type.Struct, p.StructDef)] =
    captureRoot(program.entry).flatMap((_, s) => program.defs.find(_.name == s.name).map(s -> _))

  def apply(program: p.Program, log: Log): p.Program = {
    val capture = p.Named("capture", BytePtr)
    val size    = p.Named("size", U64)

    def structDefOf(sym: p.Sym): Option[p.StructDef] = program.defs.find(_.name == sym)

    def hasPointers(sym: p.Sym, seen: Set[p.Sym]): Boolean = structHasPointers(program, sym, seen)
    def pointeeBearing(comp: p.Type): Boolean = comp match {
      case s: p.Type.Struct => hasPointers(s.name, Set.empty)
      case _                => false
    }
    def ptrDepth(t: p.Type): Int = t match {
      case p.Type.Ptr(comp, _) => 1 + ptrDepth(comp)
      case _                   => 0
    }

    def mirror(
        remote: p.Term,
        offAcc: p.Term,
        hostRoot: p.Named,
        hostSteps: List[p.PathStep], //
        sdef: p.StructDef,
        structTpe: p.Type,
        prefix: String,
        seen: Set[p.Sym]
    ): List[p.Stmt] = sdef.members.flatMap { m =>
      val mn  = m.symbol
      val tag = s"${prefix}_$mn"
      m.tpe match {
        case p.Type.Ptr(comp, space) =>
          ptrFieldGuard(hostRoot, hostSteps, mn, tag, comp, space) { (fp, fp8) =>
            val ptrTpe         = p.Type.Ptr(comp, space)
            val (ofn, ofDecl)  = u64v(s"of_$tag", p.Expr.OffsetOf(structTpe, mn))
            val (off, offDecl) = u64v(s"off_$tag", add(offAcc, sel(ofn)))
            def patchTo(dev: p.Named): p.Stmt =
              vlet(
                s"p_$tag",
                p.Type.Unit0,
                call(RuntimeAbi.SmaPatch, List(remote, sel(off), sel(dev)), p.Type.Unit0)
              )._2
            val body = comp match {
              // straight-line code can't unroll a runtime-length chain; mirror the whole graph at runtime
              case s: p.Type.Struct if structDefOf(s.name).exists(isPoolableNode) =>
                val nd    = structDefOf(s.name).get
                val reset = vlet(s"#grreset_$tag", p.Type.Unit0, call(RuntimeAbi.SmaPoolReset, Nil, p.Type.Unit0))._2
                val ptrDecls = selfPtrMembers(nd).flatMap { sp =>
                  val (o, od) = u64v(s"#gro_${tag}_${sp.symbol}", p.Expr.OffsetOf(comp, sp.symbol))
                  List(
                    od,
                    vlet(
                      s"#grp_${tag}_${sp.symbol}",
                      p.Type.Unit0,
                      call(RuntimeAbi.SmaPoolPtr, List(sel(o)), p.Type.Unit0)
                    )._2
                  )
                }
                val (dev, devDecl) = u64v(s"dev_$tag", call(RuntimeAbi.SmaMirrorGraph, List(sel(fp8)), U64))
                reset :: ptrDecls ::: List(devDecl, patchTo(dev))
              case _ =>
                val depth = ptrDepth(comp)
                val ensureCall =
                  if (depth > 0) call(RuntimeAbi.SmaEnsureDeep, List(sel(fp8), p.Term.IntU64Const(depth.toLong)), U64)
                  // floor a scalar pointee at its AST size; O3 stack-slot reuse leaves polyreflect's
                  // record stale-small, truncating the mirror
                  else
                    scalarBytes(comp) match {
                      case Some(sz) => call(RuntimeAbi.SmaEnsureMin, List(sel(fp8), p.Term.IntU64Const(sz)), U64)
                      case None     => call(RuntimeAbi.SmaEnsure, List(sel(fp8)), U64)
                    }
                val (dev, devDecl) = u64v(s"dev_$tag", ensureCall)
                val sub = comp match {
                  case s: p.Type.Struct if !seen(s.name) && pointeeBearing(comp) =>
                    structDefOf(s.name).toList.flatMap { d =>
                      // a Ptr(struct-with-pointers) may address an array; mirror each element's pointer
                      // fields. count = pointee bytes / sizeof(elem), floored at 1 for a single object
                      val elemSize       = sizeAlignOf(program, comp)._1.toLong
                      val (psz, pszDecl) = u64v(s"#psz_$tag", call(RuntimeAbi.SmaPointeeSize, List(sel(fp8)), U64))
                      val (clamp, clampDecl) =
                        u64v(s"#pcl_$tag", p.Expr.IntrOp(p.Intr.Max(sel(psz), p.Term.IntU64Const(elemSize), U64)))
                      val (cnt, cntDecl) =
                        u64v(s"#cnt_$tag", p.Expr.IntrOp(p.Intr.Div(sel(clamp), p.Term.IntU64Const(elemSize), U64)))
                      val k = p.Named(s"#k_$tag", U64)
                      val (eh, ehDecl) = vlet(
                        s"#eh_$tag",
                        ptrTpe,
                        p.Expr.RefTo(p.Term.Select(fp, Nil, ptrTpe), Some(sel(k)), comp, space, p.Region.Opaque)
                      )
                      val (offK, offKDecl) =
                        u64v(s"#offk_$tag", p.Expr.IntrOp(p.Intr.Mul(sel(k), p.Term.IntU64Const(elemSize), U64)))
                      val loopBody =
                        ehDecl :: offKDecl :: mirror(sel(dev), sel(offK), eh, Nil, d, comp, tag, seen + s.name)
                      List(
                        pszDecl,
                        clampDecl,
                        cntDecl,
                        p.Stmt.ForRange(k, p.Term.IntU64Const(0L), sel(cnt), p.Term.IntU64Const(1L), loopBody)
                      )
                    }
                  case _ => Nil
                }
                devDecl :: patchTo(dev) :: sub
            }
            ofDecl :: offDecl :: body
          }
        case s: p.Type.Struct if !seen(s.name) && hasPointers(s.name, Set.empty) =>
          val (ofn, ofDecl)  = u64v(s"of_$tag", p.Expr.OffsetOf(structTpe, mn))
          val (off, offDecl) = u64v(s"off_$tag", add(offAcc, sel(ofn)))
          structDefOf(s.name).toList.flatMap(d =>
            ofDecl :: offDecl :: mirror(
              remote,
              sel(off),
              hostRoot,
              hostSteps :+ p.PathStep.Field(mn),
              d,
              m.tpe,
              tag,
              seen + s.name
            )
          )
        case _ => Nil
      }
    }

    def readback(
        hostRoot: p.Named,
        hostSteps: List[p.PathStep],
        sdef: p.StructDef,
        prefix: String,
        seen: Set[p.Sym]
    ): List[p.Stmt] =
      sdef.members.flatMap { m =>
        val mn  = m.symbol
        val tag = s"${prefix}_$mn"
        m.tpe match {
          case p.Type.Ptr(comp, space) =>
            ptrFieldGuard(hostRoot, hostSteps, mn, tag, comp, space) { (fp, fp8) =>
              val depth = ptrDepth(comp)
              comp match {
                case s: p.Type.Struct if structDefOf(s.name).exists(isPoolableNode) =>
                  List(vlet(s"r_$tag", p.Type.Unit0, call(RuntimeAbi.SmaReadGraph, List(sel(fp8)), p.Type.Unit0))._2)
                case _ if depth > 0 =>
                  List(
                    vlet(
                      s"r_$tag",
                      p.Type.Unit0,
                      call(RuntimeAbi.SmaReadDeep, List(sel(fp8), p.Term.IntU64Const(depth.toLong)), p.Type.Unit0)
                    )._2
                  )
                case s: p.Type.Struct if !seen(s.name) && pointeeBearing(comp) =>
                  structDefOf(s.name).toList.flatMap(d => readback(fp, Nil, d, tag, seen + s.name))
                case _ =>
                  List(vlet(s"r_$tag", p.Type.Unit0, call(RuntimeAbi.SmaReadAlloc, List(sel(fp8)), p.Type.Unit0))._2)
              }
            }
          case s: p.Type.Struct if !seen(s.name) && hasPointers(s.name, Set.empty) =>
            structDefOf(s.name).toList.flatMap(d =>
              readback(hostRoot, hostSteps :+ p.PathStep.Field(mn), d, tag, seen + s.name)
            )
          case _ => Nil
        }
      }

    val (preludeFn, postludeFn) = captureStruct(program) match {
      case None =>
        val (remote, rDecl) =
          u64v("remote", call(RuntimeAbi.SmaAlloc, List(sel(capture), sel(size), p.Term.IntS32Const(0)), U64))
        val m           = hostFn(preludeName(id), U64, List(rDecl, p.Stmt.Return(p.Expr.Alias(sel(remote)))))
        val (_, rdDecl) = vlet("#d", p.Type.Unit0, call(RuntimeAbi.SmaReadAlloc, List(sel(capture)), p.Type.Unit0))
        val u = hostFn(postludeName(id), p.Type.Unit0, List(rdDecl, p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))))
        (m, u)

      case Some((structTpe, sdef)) =>
        val capturePtr = p.Type.Ptr(structTpe, p.Type.Space.Global)

        val (remote, remoteDecl) =
          u64v("remote", call(RuntimeAbi.SmaAlloc, List(sel(capture), sel(size), p.Term.IntS32Const(0)), U64))
        val (typed, typedDecl) = typedCapture(capture, capturePtr)
        val mirrorFields =
          mirror(sel(remote), p.Term.IntU64Const(0L), typed, Nil, sdef, structTpe, "c", Set(sdef.name))
        val m = hostFn(
          preludeName(id),
          U64,
          remoteDecl :: typedDecl :: mirrorFields ::: List(p.Stmt.Return(p.Expr.Alias(sel(remote))))
        )

        val (utyped, utypedDecl) = typedCapture(capture, capturePtr)
        // clear the visited set so read_alloc dedups within this postlude only
        val (_, vcDecl)    = vlet("#vc", p.Type.Unit0, call(RuntimeAbi.SmaVisitClear, Nil, p.Type.Unit0))
        val readbackFields = readback(utyped, Nil, sdef, "c", Set(sdef.name))
        val u = hostFn(
          postludeName(id),
          p.Type.Unit0,
          utypedDecl :: vcDecl :: readbackFields ::: List(p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const)))
        )
        (m, u)
    }

    program.copy(functions = program.functions ::: List(preludeFn, postludeFn))
  }
}

object Mirror {
  val PreludeName  = p.Conventions.Reflect.MirrorPrelude
  val PostludeName = p.Conventions.Reflect.MirrorPostlude
}

def preludeName(id: String): String  = if (id.isEmpty) Mirror.PreludeName else s"${Mirror.PreludeName}_$id"
def postludeName(id: String): String = if (id.isEmpty) Mirror.PostludeName else s"${Mirror.PostludeName}_$id"

// runtime::Type tag, kept in sync with the generated C enum via p.Enums.Type.value
private def runtimeTag(t: p.Type): Long = (t match {
  case _: p.Type.Ptr  => Some(p.Enums.Type.Ptr)
  case p.Type.Bool1   => Some(p.Enums.Type.Bool1)
  case p.Type.IntU8   => Some(p.Enums.Type.IntU8); case p.Type.IntU16    => Some(p.Enums.Type.IntU16)
  case p.Type.IntU32  => Some(p.Enums.Type.IntU32); case p.Type.IntU64   => Some(p.Enums.Type.IntU64)
  case p.Type.IntS8   => Some(p.Enums.Type.IntS8); case p.Type.IntS16    => Some(p.Enums.Type.IntS16)
  case p.Type.IntS32  => Some(p.Enums.Type.IntS32); case p.Type.IntS64   => Some(p.Enums.Type.IntS64)
  case p.Type.Float16 => Some(p.Enums.Type.Float16); case p.Type.Float32 => Some(p.Enums.Type.Float32)
  case p.Type.Float64 => Some(p.Enums.Type.Float64)
  case _              => None
}).fold(0L)(_.value.toLong)
def scalarTagOf(t: p.Type): Long = t match { case _: p.Type.Ptr => 0L; case _ => runtimeTag(t) }

def sizeAlignOf(program: p.Program, t: p.Type): (Int, Int) = t match {
  case p.Type.Struct(n, _) =>
    program.defs.find(_.name == n) match {
      case None => (0, 1)
      // union members overlap: widest member, not the running sum
      case Some(d) if d.isUnion =>
        val (maxS, maxA) = d.members.foldLeft((0, 1)) { case ((maxS, maxA), m) =>
          val (s, a) = sizeAlignOf(program, m.tpe)
          (math.max(maxS, s), math.max(maxA, a))
        }
        ((maxS + maxA - 1) / maxA * maxA, maxA)
      case Some(d) =>
        val (off, maxA) = d.members.foldLeft((0, 1)) { case ((off, maxA), m) =>
          val (s, a) = sizeAlignOf(program, m.tpe)
          ((off + a - 1) / a * a + s, math.max(maxA, a))
        }
        ((off + maxA - 1) / maxA * maxA, maxA)
    }
  case p.Type.Arr(c, len, _) => val (s, a) = sizeAlignOf(program, c); (s * len, a)
  case other                 => val s = scalarBytesOr8(other); (s, s)
}

def hostFn(name: String, rtn: p.Type, body: List[p.Stmt]): p.Function =
  p.Function(
    p.Sym(List(name)),
    Nil,
    None,
    List(p.Arg(p.Named("capture", BytePtr)), p.Arg(p.Named("size", U64))),
    Nil,
    Nil,
    rtn,
    body,
    p.Function.Visibility.Exported,
    p.Function.FpMode.Relaxed,
    isEntry = false,
    affinity = p.Function.Affinity.Host
  )

def selfPtrMembers(sdef: p.StructDef): List[p.Named] = sdef.members.filter { m =>
  m.tpe match { case p.Type.Ptr(p.Type.Struct(n, _), _) => n == sdef.name; case _ => false }
}
def isPoolableNode(sdef: p.StructDef): Boolean =
  selfPtrMembers(sdef).nonEmpty && sdef.members.forall { m =>
    m.tpe match {
      case p.Type.Ptr(p.Type.Struct(n, _), _) => n == sdef.name
      case t                                  => scalarTagOf(t) != 0
    }
  }

def structHasPointers(program: p.Program, sym: p.Sym, seen: Set[p.Sym]): Boolean =
  !seen(sym) && program.defs
    .find(_.name == sym)
    .exists(_.members.exists { m =>
      m.tpe match {
        case _: p.Type.Ptr    => true
        case s: p.Type.Struct => structHasPointers(program, s.name, seen + sym)
        case _                => false
      }
    })
