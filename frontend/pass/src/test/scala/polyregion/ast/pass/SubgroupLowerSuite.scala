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
    val built = PassPipelineParser.parseStep("SubgroupLower(width=8,maxGroupSize=256)").flatMap(PassRegistry.build)
    assert(built.isRight, built.toString)
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

  test("keeps nested shuffle synchronization inside its control-flow scope") {
    val value = named("value", f32)
    val shuffle = p.Expr.SpecOp(
      p.Spec.GpuShuffleDown(selectT(value), p.Term.IntU32Const(1), p.Term.IntU32Const(7), p.Term.IntU32Const(-1), f32)
    )
    val loop = p.Stmt.While(
      p.Term.Bool1Const(true),
      List(p.Stmt.Var(named("result", f32), Some(shuffle), isMutable = true))
    )
    val out = run(program(entry(body = List(loop))), 8, 256)

    assertEquals(localArrays(out).size, 1)
    val loopBody = out.entry.body.collectFirst { case statement: p.Stmt.While => statement.body }.get
    assert(loopBody.count {
      case p.Stmt.Var(_, Some(p.Expr.SpecOp(p.Spec.GpuBarrierLocal)), _) => true
      case _                                                             => false
    } >= 2)
    assert(!loopBody.exists {
      case p.Stmt.Var(p.Named(_, p.Type.Arr(_, _, p.Type.Space.Local), _), None, _) => true
      case _                                                                        => false
    })
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
    assert(specs(out).exists(_.isInstanceOf[p.Spec.GpuLocalSize]))
    assert(specs(out).exists(_.isInstanceOf[p.Spec.GpuGlobalSize]))
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
