package polyregion

import polyregion.ast.{CodeGen, MsgPack, PolyAST as p}
import polyregion.jvm.compiler as ct
import polyregion.jvm.runtime as rt
import polyregion.scalalang.Platforms

import java.nio.{ByteBuffer, ByteOrder}

class EscapeRunSuite extends munit.FunSuite {

  private val i32    = p.Type.IntS32
  private val g      = p.Type.Space.Global
  private val ptrI32 = p.Type.Ptr(i32, g)
  private val Dirt   = 0x7f7f7f7f

  private def select(tpe: p.Type, n: String): (p.Named, p.Term.Select) = {
    val named = p.Named(n, tpe)
    named -> p.Term.Select(named, Nil, tpe)
  }

  private val ctx = p.Named("ctx", p.Spec.ContextType)

  private def host(name: String, args: List[p.Named], body: List[p.Stmt]): p.Program =
    p.Program(
      Some(
        p.Function(
          p.FunctionDecl(
            p.Sym(name),
            Nil,
            None,
            (ctx :: args).map(p.Arg(_)),
            Nil,
            Nil,
            p.Type.Unit0,
            p.Function.Affinity.Host
          ),
          body,
          p.Function.Visibility.Exported,
          p.Function.FpMode.Relaxed,
          p.CallConvention.RegularCall
        )
      ),
      Nil,
      Nil
    )

  private def compile(program: p.Program): ct.Compilation = {
    val compiler = ct.Compiler.create()
    try
      compiler.compile(
        MsgPack.encode(CodeGen.programVersioned(program)),
        true,
        ct.Options.of(ct.Target.LLVM_HOST, "native"),
        ct.Opt.O2.value
      )
    finally compiler.close()
  }

  private def assembly(c: ct.Compilation): String = {
    def go(e: ct.Event): List[String] =
      (if (e.name == "llvm_to_asm") List(e.data) else Nil) ::: e.items.toList.flatMap(go)
    c.events.toList.flatMap(go).mkString("\n")
  }

  private def load(name: String, program: p.Program): ct.Compilation = {
    val c = compile(program)
    rt.Context.loadHostProgram(name, c.program)
    c
  }

  private def direct(bytes: Int): ByteBuffer = ByteBuffer.allocateDirect(bytes).order(ByteOrder.nativeOrder)

  private def call(module: String, fn: String, out: ByteBuffer, n: Int): Unit = {
    val data = ByteBuffer
      .allocate(12)
      .order(ByteOrder.nativeOrder)
      .putLong(Platforms.pointerOfDirectBuffer(out))
      .putInt(n)
      .array
    rt.Context.current().callHost(Array.empty, module, fn, Array(rt.Type.PTR.value, rt.Type.INT.value), data)
  }

  private def dirty: p.Program = {
    val (n, nRef)     = select(i32, "n")
    val (blk, blkRef) = select(ptrI32, "blk")
    val (scr, scrRef) = select(ptrI32, "scr")
    val (len, lenRef) = select(p.Type.IntU64, "len")
    val (out, _)      = select(ptrI32, "out")
    val (nu, nuRef)   = select(p.Type.IntU64, "nu")
    host(
      "dirty",
      List(out, n),
      List(
        p.Stmt.Var(blk, Some(p.Expr.Alloc(i32, nRef, g, p.Region.Opaque))),
        p.Stmt.Var(nu, Some(p.Expr.Cast(nRef, p.Type.IntU64))),
        p.Stmt.Var(len, Some(p.Expr.IntrOp(p.Intr.Mul(nuRef, p.Term.IntU64Const(4), p.Type.IntU64)))),
        p.Stmt.Var(scr, Some(p.Expr.ForeignCall("memset", List(blkRef, p.Term.IntS32Const(0x7f), lenRef), ptrI32))),
        p.Stmt.Var(p.Named("freed", p.Type.Unit0), Some(p.Expr.ForeignCall("free", List(blkRef), p.Type.Unit0))),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )
  }

  private def readUnwritten: p.Program = {
    val (n, nRef)     = select(i32, "n")
    val (out, outRef) = select(ptrI32, "out")
    val (a, aRef)     = select(ptrI32, "a")
    val (x, xRef)     = select(i32, "x")
    host(
      "readUnwritten",
      List(out, n),
      List(
        p.Stmt.Var(a, Some(p.Expr.Alloc(i32, nRef, g, p.Region.Opaque))),
        p.Stmt.Update(aRef, p.Term.IntS32Const(0), p.Term.IntS32Const(1)),
        p.Stmt.Var(x, Some(p.Expr.Index(aRef, p.Term.IntS32Const(1), i32))),
        p.Stmt.Update(outRef, p.Term.IntS32Const(0), xRef),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )
  }

  private def readCell: p.Program = {
    val (n, _)        = select(i32, "n")
    val (out, outRef) = select(ptrI32, "out")
    val (a, aRef)     = select(ptrI32, "a")
    val (x, xRef)     = select(i32, "x")
    val (y, yRef)     = select(i32, "y")
    val (r, rRef)     = select(i32, "r")
    host(
      "readCell",
      List(out, n),
      List(
        p.Stmt.Var(a, Some(p.Expr.Alloc(i32, p.Term.IntS32Const(16), g, p.Region.Opaque))),
        p.Stmt.Update(aRef, p.Term.IntS32Const(0), p.Term.IntS32Const(1)),
        p.Stmt.Var(x, Some(p.Expr.Index(aRef, p.Term.IntS32Const(0), i32))),
        p.Stmt.Var(y, Some(p.Expr.Index(aRef, p.Term.IntS32Const(1), i32))),
        p.Stmt.Var(r, Some(p.Expr.IntrOp(p.Intr.Add(xRef, yRef, i32)))),
        p.Stmt.Update(outRef, p.Term.IntS32Const(0), rRef),
        p.Stmt.Return(p.Expr.Alias(p.Term.Unit0Const))
      )
    )
  }

  test("an unwritten element of a heap array reads zero even on recycled memory") {
    load("dirty@host", dirty)
    load("readUnwritten@host", readUnwritten)
    val out = direct(4)
    call("dirty@host", "dirty", out, 16)
    out.putInt(0, Dirt)
    call("readUnwritten@host", "readUnwritten", out, 16)
    assertEquals(out.getInt(0), 0)
  }

  test("a constant-size array with constant indices is replaced and its unwritten cell reads zero") {
    val compiled = load("readCell@host", readCell)
    val out      = direct(4)
    call("dirty@host", "dirty", out, 16)
    call("readCell@host", "readCell", out, 16)
    assertEquals(out.getInt(0), 1)
    assert(!assembly(compiled).contains("malloc"), assembly(compiled))
  }

  test("a dynamically sized array keeps its allocation") {
    val compiled = compile(readUnwritten)
    assert(assembly(compiled).contains("malloc"), assembly(compiled))
  }
}
