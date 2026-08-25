package polyregion.ast.pass

import polyregion.ast.{PolyAST as p, *, given}
import polyregion.ast.Traversal.*
import PassTest.*

class SubgroupLowerSuite extends munit.FunSuite {

  private val u32 = p.Type.IntU32
  private val f32 = p.Type.Float32

  private def run(program: p.Program, width: Int, maxGroupSize: Int): p.Program =
    SubgroupLower(width, maxGroupSize)(program, NoopLog)

  private def specs(program: p.Program): List[p.Spec] =
    program.entry.collectAll[p.Expr].collect { case p.Expr.SpecOp(op) => op }

  private def localArrays(program: p.Program): List[p.Named] =
    program.entry.collectAll[p.Stmt].collect {
      case p.Stmt.Var(name @ p.Named(_, p.Type.Arr(_, _, p.Type.Space.Local), _), None, _) => name
    }

  test("uses the configured subgroup width") {
    val body = List(
      p.Stmt.Var(named("size", u32), Some(p.Expr.SpecOp(p.Spec.GpuSubgroupSize))),
      p.Stmt.Var(named("lane", u32), Some(p.Expr.SpecOp(p.Spec.GpuLaneIdx)))
    )

    List(8, 16).foreach { width =>
      val out = run(program(entry(body = body)), width, 256)
      assert(!specs(out).contains(p.Spec.GpuSubgroupSize))
      assert(!specs(out).contains(p.Spec.GpuLaneIdx))
      assert(out.entry.collectAll[p.Term].contains(p.Term.IntU32Const(width)))
      assert(out.entry.collectAll[p.Expr].exists {
        case p.Expr.IntrOp(p.Intr.BAnd(_, p.Term.IntU32Const(mask), p.Type.IntU32)) => mask == width - 1
        case _                                                                      => false
      })
    }
  }

  test("sizes shuffle scratch to the configured group ceiling") {
    val value = named("value", f32)
    val shuffle = p.Expr.SpecOp(
      p.Spec.GpuShuffleDown(selectT(value), p.Term.IntU32Const(1), p.Term.IntU32Const(7), p.Term.IntU32Const(-1), f32)
    )
    val out = run(
      program(entry(body = List(p.Stmt.Var(named("result", f32), Some(shuffle), isMutable = true)))),
      width = 8,
      maxGroupSize = 256
    )

    assert(!specs(out).exists(_.isInstanceOf[p.Spec.GpuShuffleDown]))
    assert(out.entry.collectAll[p.Stmt].exists {
      case p.Stmt.Var(name, None, _) => name.tpe == p.Type.Arr(f32, 256, p.Type.Space.Local)
      case _                         => false
    })
  }

  test("is available as a configured pipeline step") {
    val built = PassPipelineParser
      .parseStep("SubgroupLower(width=8,maxGroupSize=256,lowerSubgroups=false,lowerGroups=true)")
      .flatMap(PassRegistry.build)
    assert(built.isRight, built.toString)
  }

  test("rejects an impractical scratch ceiling without overflowing scan construction") {
    val error = intercept[IllegalArgumentException] {
      SubgroupLower(width = 1, maxGroupSize = 1073741825)(program(entry()), NoopLog)
    }
    assert(error.getMessage.contains("1024"), error.getMessage)
  }

  test("can lower work-group operations without replacing native subgroup operations") {
    val value = named("value", p.Type.IntS32)
    val body = List(
      p.Stmt.Var(named("lane", u32), Some(p.Expr.SpecOp(p.Spec.GpuLaneIdx))),
      p.Stmt.Var(
        named("reduce", p.Type.IntS32),
        Some(p.Expr.SpecOp(p.Spec.GpuGroupReduce(p.AtomicOp.Add, selectT(value), p.Type.IntS32)))
      )
    )
    val out = SubgroupLower(width = 8, maxGroupSize = 256, lowerSubgroups = false, lowerGroups = true)(
      program(entry(body = body)),
      NoopLog
    )

    assert(specs(out).contains(p.Spec.GpuLaneIdx))
    assert(!specs(out).exists(_.isInstanceOf[p.Spec.GpuGroupReduce]))
    assertEquals(localArrays(out).size, 1)
  }

  test("optionally lowers multidimensional work-group collectives through local scratch") {
    val value = named("value", p.Type.IntS64)
    val body = List(
      p.Stmt.Var(
        named("reduce", p.Type.IntS64),
        Some(p.Expr.SpecOp(p.Spec.GpuGroupReduce(p.AtomicOp.Add, selectT(value), p.Type.IntS64)))
      ),
      p.Stmt.Var(
        named("inclusive", p.Type.IntS64),
        Some(p.Expr.SpecOp(p.Spec.GpuGroupInclusiveScan(p.AtomicOp.Add, selectT(value), p.Type.IntS64)))
      ),
      p.Stmt.Var(
        named("exclusive", p.Type.IntS64),
        Some(p.Expr.SpecOp(p.Spec.GpuGroupExclusiveScan(p.AtomicOp.Add, selectT(value), p.Type.IntS64)))
      )
    )
    val out = SubgroupLower(width = 8, maxGroupSize = 256, lowerGroups = true)(program(entry(body = body)), NoopLog)

    assert(!specs(out).exists {
      case _: p.Spec.GpuGroupReduce | _: p.Spec.GpuGroupInclusiveScan | _: p.Spec.GpuGroupExclusiveScan => true
      case _                                                                                            => false
    })
    assertEquals(localArrays(out).size, 2)
    assert(out.entry.collectAll[p.Expr].exists {
      case p.Expr.SpecOp(p.Spec.GpuLocalIdx(p.Term.IntU32Const(2))) => true
      case _                                                        => false
    })
    assert(out.entry.collectAll[p.Expr].exists {
      case p.Expr.SpecOp(p.Spec.GpuLocalSize(p.Term.IntU32Const(2))) => true
      case _                                                         => false
    })
    val assignments = out.entry
      .collectAll[p.Stmt]
      .collect { case p.Stmt.Var(name, Some(value), _) =>
        name.symbol -> value
      }
      .toMap
    assert(assignments.exists {
      case (name, p.Expr.IntrOp(p.Intr.Add(p.Term.Select(z, Nil, _), p.Term.Select(szyx, Nil, _), _))) =>
        name.contains("group_local_id") && z.symbol.contains("group_z") && szyx.symbol.contains("group_szyx")
      case _ => false
    })
    assert(assignments.exists {
      case (name, p.Expr.IntrOp(p.Intr.Mul(p.Term.Select(sz, Nil, _), p.Term.Select(yx, Nil, _), _))) =>
        name.contains("group_szyx") && sz.symbol.contains("group_sz") && yx.symbol.contains("group_yx")
      case _ => false
    })
  }

  test("fences a materialised reduction result before reusing its scratch") {
    val value = named("value", p.Type.IntS32)
    val reduce = (name: String) =>
      p.Stmt.Var(
        named(name, p.Type.IntS32),
        Some(p.Expr.SpecOp(p.Spec.GpuGroupReduce(p.AtomicOp.Add, selectT(value), p.Type.IntS32)))
      )
    val out = SubgroupLower(width = 8, maxGroupSize = 32, lowerGroups = true)(
      program(entry(body = List(reduce("first"), reduce("second")))),
      NoopLog
    )
    val resultIndices = out.entry.body.zipWithIndex.collect {
      case (p.Stmt.Var(p.Named(symbol, _, _), _, _), index) if symbol.contains("group_result") => index
    }
    assertEquals(resultIndices.size, 2)
    assert(resultIndices.forall { index =>
      out.entry.body
        .drop(index + 1)
        .collectFirst {
          case p.Stmt.Var(_, Some(p.Expr.SpecOp(p.Spec.GpuBarrierLocal)), _) => true
          case _: p.Stmt.Update                                              => false
        }
        .contains(true)
    })
  }

  test("fences materialised scan results before reusing their scratch") {
    val value = named("value", p.Type.IntS32)
    val scan = (name: String) =>
      p.Stmt.Var(
        named(name, p.Type.IntS32),
        Some(p.Expr.SpecOp(p.Spec.GpuGroupExclusiveScan(p.AtomicOp.Add, selectT(value), p.Type.IntS32)))
      )
    val out = SubgroupLower(width = 8, maxGroupSize = 32, lowerGroups = true)(
      program(entry(body = List(scan("first"), scan("second")))),
      NoopLog
    )
    val resultIndices = out.entry.body.zipWithIndex.collect {
      case (p.Stmt.Var(p.Named(symbol, _, _), _, _), index) if symbol.contains("group_result") => index
    }
    assertEquals(resultIndices.size, 2)
    assert(resultIndices.forall { index =>
      out.entry.body
        .drop(index + 1)
        .collectFirst {
          case p.Stmt.Var(_, Some(p.Expr.SpecOp(p.Spec.GpuBarrierLocal)), _) => true
          case _: p.Stmt.Update                                              => false
        }
        .contains(true)
    })
  }

  test("lowers every shuffle form and subgroup barrier while keeping unrelated operations visible") {
    val value = named("value", u32)
    val mask  = p.Term.IntU32Const(-1)
    val width = p.Term.IntU32Const(7)
    val operations = List[p.Spec](
      p.Spec.GpuShuffleDown(selectT(value), p.Term.IntU32Const(1), width, mask, u32),
      p.Spec.GpuShuffleUp(selectT(value), p.Term.IntU32Const(1), width, mask, u32),
      p.Spec.GpuShuffleIdx(selectT(value), p.Term.IntU32Const(3), width, mask, u32),
      p.Spec.GpuShuffleXor(selectT(value), p.Term.IntU32Const(1), width, mask, u32)
    )
    val body = operations.zipWithIndex.map { case (operation, index) =>
      p.Stmt.Var(named(s"result$index", u32), Some(p.Expr.SpecOp(operation)))
    } ::: List(
      p.Stmt.Var(named("barrier", p.Type.Unit0), Some(p.Expr.SpecOp(p.Spec.GpuSubgroupBarrier(mask)))),
      p.Stmt.Var(named("fence", p.Type.Unit0), Some(p.Expr.SpecOp(p.Spec.GpuFenceLocal)))
    )
    val remaining = specs(run(program(entry(body = body)), 8, 256))

    assert(!remaining.exists {
      case _: p.Spec.GpuShuffleDown | _: p.Spec.GpuShuffleUp | _: p.Spec.GpuShuffleIdx | _: p.Spec.GpuShuffleXor => true
      case _ => false
    })
    assert(!remaining.exists(_.isInstanceOf[p.Spec.GpuSubgroupBarrier]))
    assert(remaining.contains(p.Spec.GpuBarrierLocal))
    assert(remaining.contains(p.Spec.GpuFenceLocal))
  }

  test("uses distinct aggregate buffers within a site and reuses them across sites") {
    val pairSym = sym("Pair")
    val pairTpe = p.Type.Struct(pairSym, Nil)
    val pairDef = p.StructDef(pairSym, Nil, List(named("left", f32), named("right", f32)), Nil)
    val value   = named("value", pairTpe)
    val shuffle = (delta: Int) =>
      p.Expr.SpecOp(
        p.Spec.GpuShuffleDown(
          selectT(value),
          p.Term.IntU32Const(delta),
          p.Term.IntU32Const(7),
          p.Term.IntU32Const(-1),
          pairTpe
        )
      )
    val body = List(
      p.Stmt.Var(named("first", pairTpe), Some(shuffle(1))),
      p.Stmt.Var(named("second", pairTpe), Some(shuffle(2)))
    )
    val arrays = localArrays(run(program(entry(body = body), defs = List(pairDef)), 8, 256))

    assertEquals(arrays.size, 2)
    assert(arrays.forall(_.tpe == p.Type.Arr(f32, 256, p.Type.Space.Local)))
  }

  test("rejects masked barriers that a work-group barrier cannot emulate") {
    val barrier = p.Stmt.Var(
      named("barrier", p.Type.Unit0),
      Some(p.Expr.SpecOp(p.Spec.GpuSubgroupBarrier(p.Term.IntU32Const(0xffff))))
    )
    val error = intercept[IllegalArgumentException](run(program(entry(body = List(barrier))), 8, 256))
    assert(error.getMessage.contains("Masked subgroup barriers"))
  }

  test("materialises a by-pointer aggregate shuffle operand before selecting its leaves") {
    val pairSym = sym("PairByPointer")
    val pairTpe = p.Type.Struct(pairSym, Nil)
    val pairDef = p.StructDef(pairSym, Nil, List(named("left", f32), named("right", f32)), Nil)
    val pointer = named("value", p.Type.Ptr(pairTpe, p.Type.Space.Private))
    val shuffle = p.Expr.SpecOp(
      p.Spec.GpuShuffleDown(
        selectT(pointer),
        p.Term.IntU32Const(1),
        p.Term.IntU32Const(7),
        p.Term.IntU32Const(-1),
        pairTpe
      )
    )
    val out = run(
      program(entry(body = List(p.Stmt.Var(named("result", pairTpe), Some(shuffle)))), defs = List(pairDef)),
      width = 8,
      maxGroupSize = 256
    )

    assert(out.entry.collectAll[p.Expr].exists {
      case p.Expr.Index(p.Term.Select(root, Nil, _), p.Term.IntU32Const(0), tpe) => root == pointer && tpe == pairTpe
      case _                                                                     => false
    })
    assert(!out.entry.collectAll[p.Term].exists {
      case p.Term.Select(root, steps, _) => root == pointer && steps.nonEmpty
      case _                             => false
    })
  }

  test("rejects subgroup emulation under loop control flow") {
    val value = named("value", f32)
    val shuffle = p.Expr.SpecOp(
      p.Spec.GpuShuffleDown(selectT(value), p.Term.IntU32Const(1), p.Term.IntU32Const(7), p.Term.IntU32Const(-1), f32)
    )
    val loop = p.Stmt.While(
      p.Term.Bool1Const(true),
      List(p.Stmt.Var(named("result", f32), Some(shuffle), isMutable = true))
    )
    val error = intercept[IllegalArgumentException](run(program(entry(body = List(loop))), 8, 256))
    assert(error.getMessage.contains("whole-workgroup-uniform participation"), error.getMessage)
  }

  test("rejects conditional calls whose closure synchronises a subgroup") {
    val helper = fn(
      "helper",
      rtn = f32,
      body = List(
        p.Stmt.Return(
          p.Expr.SpecOp(
            p.Spec.GpuShuffleDown(
              p.Term.Float32Const(1),
              p.Term.IntU32Const(1),
              p.Term.IntU32Const(7),
              p.Term.IntU32Const(-1),
              f32
            )
          )
        )
      )
    )
    val call = p.Expr.Invoke(p.Type.FnRef(helper.decl.name), Nil, None, Nil, f32)
    val conditional = p.Stmt.Cond(
      p.Term.Bool1Const(true),
      List(p.Stmt.Var(named("result", f32), Some(call))),
      Nil
    )
    val error = intercept[IllegalArgumentException](run(program(entry(body = List(conditional)), List(helper)), 8, 256))
    assert(error.getMessage.contains("conditional control flow"), error.getMessage)
  }

  test("rejects conditional work-group collectives in group-only mode") {
    val reduce = p.Expr.SpecOp(p.Spec.GpuGroupReduce(p.AtomicOp.Add, p.Term.IntS32Const(1), p.Type.IntS32))
    val conditional = p.Stmt.Cond(
      p.Term.Bool1Const(true),
      List(p.Stmt.Var(named("result", p.Type.IntS32), Some(reduce))),
      Nil
    )
    val error = intercept[IllegalArgumentException] {
      SubgroupLower(width = 8, maxGroupSize = 256, lowerSubgroups = false, lowerGroups = true)(
        program(entry(body = List(conditional))),
        NoopLog
      )
    }
    assert(error.getMessage.contains("conditional control flow"), error.getMessage)
  }

  test("rejects conditional calls whose closure synchronises a work-group in group-only mode") {
    val helper = fn(
      "groupHelper",
      rtn = p.Type.IntS32,
      body = List(
        p.Stmt.Return(
          p.Expr.SpecOp(p.Spec.GpuGroupReduce(p.AtomicOp.Add, p.Term.IntS32Const(1), p.Type.IntS32))
        )
      )
    )
    val call = p.Expr.Invoke(p.Type.FnRef(helper.decl.name), Nil, None, Nil, p.Type.IntS32)
    val conditional = p.Stmt.Cond(
      p.Term.Bool1Const(true),
      List(p.Stmt.Var(named("result", p.Type.IntS32), Some(call))),
      Nil
    )
    val error = intercept[IllegalArgumentException] {
      SubgroupLower(width = 8, maxGroupSize = 256, lowerSubgroups = false, lowerGroups = true)(
        program(entry(body = List(conditional)), List(helper)),
        NoopLog
      )
    }
    assert(error.getMessage.contains("conditional control flow"), error.getMessage)
  }

  test("rejects conditional exits before a work-group collective in group-only mode") {
    val body = List(
      p.Stmt.Cond(p.Term.Bool1Const(true), List(p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))), Nil),
      p.Stmt.Var(
        named("result", p.Type.IntS32),
        Some(p.Expr.SpecOp(p.Spec.GpuGroupReduce(p.AtomicOp.Add, p.Term.IntS32Const(1), p.Type.IntS32)))
      )
    )
    val error = intercept[IllegalArgumentException] {
      SubgroupLower(width = 8, maxGroupSize = 256, lowerSubgroups = false, lowerGroups = true)(
        program(entry(body = body)),
        NoopLog
      )
    }
    assert(error.getMessage.contains("early exit"), error.getMessage)
  }

  test("rejects conditional exits before a synchronising subgroup operation") {
    val value = named("value", f32)
    val body = List(
      p.Stmt.Cond(p.Term.Bool1Const(true), List(p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))), Nil),
      p.Stmt.Var(
        named("result", f32),
        Some(
          p.Expr.SpecOp(
            p.Spec
              .GpuShuffleDown(selectT(value), p.Term.IntU32Const(1), p.Term.IntU32Const(7), p.Term.IntU32Const(-1), f32)
          )
        )
      )
    )
    val error = intercept[IllegalArgumentException](run(program(entry(body = body)), 8, 256))
    assert(error.getMessage.contains("early exit"), error.getMessage)
  }

  test("gives votes and ballots per-lane scratch") {
    val predicate = named("predicate", p.Type.Bool1)
    val mask      = p.Term.IntU32Const(-1)
    val body = List(
      p.Stmt.Var(named("any", p.Type.Bool1), Some(p.Expr.SpecOp(p.Spec.GpuVoteAny(mask, selectT(predicate))))),
      p.Stmt.Var(named("all", p.Type.Bool1), Some(p.Expr.SpecOp(p.Spec.GpuVoteAll(mask, selectT(predicate))))),
      p.Stmt.Var(named("ballot", u32), Some(p.Expr.SpecOp(p.Spec.GpuBallot(mask, selectT(predicate)))))
    )
    val out   = run(program(entry(body = body)), 8, 256)
    val types = localArrays(out).map(_.tpe).toSet

    assertEquals(localArrays(out).size, 1)
    assertEquals(types, Set[p.Type](p.Type.Arr(p.Type.IntU32, 256, p.Type.Space.Local)))
    assert(!specs(out).exists {
      case _: p.Spec.GpuVoteAny | _: p.Spec.GpuVoteAll | _: p.Spec.GpuBallot => true
      case _                                                                 => false
    })
    assertEquals(out.entry.collectAll[p.Stmt].count(_.isInstanceOf[p.Stmt.ForRange]), 3)
  }

  test("preserves shuffle clamps and masks while guarding partial ballots") {
    val value     = named("value", u32)
    val clamp     = named("clamp", u32)
    val mask      = named("mask", u32)
    val predicate = named("predicate", p.Type.Bool1)
    val body = List(
      p.Stmt.Var(
        named("shuffled", u32),
        Some(
          p.Expr.SpecOp(
            p.Spec.GpuShuffleIdx(selectT(value), p.Term.IntU32Const(3), selectT(clamp), selectT(mask), u32)
          )
        )
      ),
      p.Stmt.Var(named("ballot", u32), Some(p.Expr.SpecOp(p.Spec.GpuBallot(selectT(mask), selectT(predicate)))))
    )
    val out        = run(program(entry(body = body)), width = 8, maxGroupSize = 256)
    val intrinsics = out.entry.collectAll[p.Expr].collect { case p.Expr.IntrOp(op) => op }

    assert(intrinsics.exists {
      case p.Intr.BNot(p.Term.Select(root, Nil, _), p.Type.IntU32) => root == clamp
      case _                                                       => false
    })
    assert(intrinsics.exists {
      case p.Intr.BSR(p.Term.Select(root, Nil, _), _, p.Type.IntU32) => root == mask
      case _                                                         => false
    })
    assertEquals(
      specs(out).collect { case p.Spec.GpuLocalSize(p.Term.IntU32Const(dimension)) => dimension }.toSet,
      Set(0, 1, 2)
    )
    assert(!specs(out).exists(_.isInstanceOf[p.Spec.GpuGlobalSize]))
    assert(out.entry.collectAll[p.Stmt].exists {
      case p.Stmt.Cond(_, List(p.Stmt.Mut(_, p.Expr.Index(_, _, p.Type.IntU32))), Nil) => true
      case _                                                                           => false
    })
  }

  test("rejects invalid subgroup configurations") {
    val input = program(entry())
    List((0, 256), (3, 256), (64, 256), (8, 4), (8, 258)).foreach { case (width, ceiling) =>
      intercept[IllegalArgumentException](SubgroupLower(width, ceiling)(input, NoopLog))
    }
  }

  test("is deterministic across repeated runs") {
    val value = named("value", f32)
    val input = program(
      entry(body =
        List(
          p.Stmt.Var(
            named("result", f32),
            Some(
              p.Expr.SpecOp(
                p.Spec.GpuShuffleDown(
                  selectT(value),
                  p.Term.IntU32Const(1),
                  p.Term.IntU32Const(7),
                  p.Term.IntU32Const(-1),
                  f32
                )
              )
            )
          )
        )
      )
    )

    assertEquals(run(input, 8, 256), run(input, 8, 256))
  }
}
