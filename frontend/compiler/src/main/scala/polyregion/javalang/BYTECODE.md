# Tier 3: lowering JVM bytecode to PolyAST

`scalalang` reads Scala 3 quotes and `javalang` reads attributed javac trees, so both need the source
of the region being offloaded. Tier 3 removes that requirement: it lowers a compiled method to
PolyAST from its class file, so a third-party jar, or a virtual call target only known once the
receiver's concrete class is observed at run time, can still be offloaded.

`Vineflower.scala` and `BytecodeRemapper.scala` implement everything under "Supported subset".
Everything under "Not yet built" is design only. The frontend lowers methods over primitives and
primitive arrays, is differentially tested against the JVM on both javac and scalac output, and is
wired into both frontends as the fallback resolver for a callee with no javac tree and no TASTy.

## Supported subset

- `static` methods, and instance methods that never read their receiver. The second case is not a
  convenience: every method on a Scala `object` has exactly that shape, so without it no scalac
  output could be lowered at all.
- Parameters and return types that are primitives or arrays of primitives.
- Numeric constants of every primitive width.
- Locals, including slots reused at different types and the temporaries Vineflower synthesises.
- Arithmetic: `+ - * / %` and unary `-` for `int`, `long`, `float`, `double`.
- Bitwise and shifts: `& | ^ ~ << >> >>>` for `int` and `long`; `& | ^` on `boolean`.
- Conversions: every `x2y`, including the `i2b`/`i2c`/`i2s` truncations and explicit casts.
- Comparisons: `== != < <= > >=` on any primitive, plus `lcmp`, `fcmpl`, `fcmpg`, `dcmpl`, `dcmpg`
  where the three-way result survives as a value.
- `&&`, `||`, `!` and `?:`, including in value position and including their short-circuiting.
- `++` and `--` in either fixity, in statement or value position, on a local or an array element.
- Arrays: element read and write at any primitive component type, and arrays of arrays.
- `while`, `do..while`, `for` and infinite loops, `break`, `continue`, and their labelled forms.
- Any control flow Vineflower can structure, which includes irreducible graphs (it duplicates nodes
  until the graph reduces) and the `goto` chains scalac emits for tail recursion.

Deliberate deviations from the JVM, consistent with the rest of PolyRegion:

- Division by zero is undefined rather than throwing `ArithmeticException`.
- `f2i`, `d2i`, `f2l` and `d2l` follow the target's semantics rather than Java's saturation on NaN
  and out-of-range values.
- Array bounds are not checked.

Two places where the JVM's semantics are reproduced exactly, because getting them wrong is silent:

- Shift distances are masked (`& 31` or `& 63`) before the shift, since C and LLVM leave wider
  distances undefined while the JVM specifies the mask.
- `>>>` is lowered as a cast to `IntU32`/`IntU64`, a `BZSR`, and a cast back, so the sign bit cannot
  be dragged into the result.

- Static calls. A call to a static method is lowered to an `Invoke` of `owner.name`, and the callee is
  lowered from its own class file in turn (`BytecodeResolver.resolve` walks the closure, deduplicated by
  owner, name and descriptor). Overloads share a symbol and differ by signature.
- Intrinsics win. `Natives.resolve` asks `IntrinsicTable` before a call is recorded, so `Math.sqrt` and
  friends become `MathOp`s instead of being lowered from their native bodies. A miss goes on to the bytecode
  tier. When the bytecode of a `java/lang/Math` or `StrictMath` method cannot be lowered, the refusal names
  the intrinsic (`unsupported JVM intrinsic java/lang/StrictMath.IEEEremainder(DD)D: ...`) and the construct it reached.

## Resolver wiring

- `javalang`: `TreeRemapper.directCall` sends a static callee with no tree in `localTrees` or `methodTable`
  to `BytecodeResolver`. Class bytes come from javac's class path file manager, then from the running JDK.
- `scalalang`: a Java static method (`JavaDefined` and `JavaStatic`) is invoked without its class's module as
  a receiver, and `Compiler.bytecodeFallback` lowers it when its signature is primitives and primitive arrays.
  The class is found through the macro class loader. A Java static with any other signature is refused.
- A refusal names the method and the construct: ``Cannot lower `Owner.m(I)I` from its class file: Unsupported `throw`...``.

## Getting the tree out of Vineflower

`org.vineflower:vineflower:1.12.0:slim` is Apache-2.0, 1.16 MiB, 432 classes, and its POM declares no
dependencies at all. It needs Java 17 at run time, which is already true here.

The documented `BaseDecompiler` only emits text through an `IResultSaver`, so `Vineflower.scala`
calls `MethodProcessor.codeToJava` directly. That is public and static and returns the
`RootStatement`. What it needs in return is the ambient state `ClassWrapper.init` would otherwise
have set up:

- `StructGeneralAttribute.init()`, which populates the attribute registry. Without it the `Code`
  attribute is skipped and every method silently reads as abstract.
- A `DecompilerContext` over a `StructContext`, a `ClassesProcessor` with the class registered in
  `getMapRootClasses`, `CURRENT_CLASS`/`CURRENT_CLASS_NODE`/`CURRENT_CLASS_WRAPPER`, an
  `ImportCollector`, and an `IdentityRenamerFactory`.
- `DECOMPILE_ASSERTIONS` off, because `assert` rewriting reads `<clinit>` properties only a full
  `ClassWrapper.init` computes, and `MAX_PROCESSING_METHOD` at 0, because the timeout path decompiles
  on a second thread whose context would have to be replicated.

`DecompilerContext` is a thread local and the attribute registry is a process-wide map, so the whole
thing runs under one lock. The tree is only valid while that context is current - a `VarExprent`
reads decompiler options to answer `getVarType` - so `Vineflower.structure` takes the lowering as a
callback rather than returning the tree.

Nothing here produces or consumes Java source text: Scala-emitted identifiers such as `$plus$plus`
do not survive a round trip through javac.

## Statements to PolyAST

`RootStatement`, `SequenceStatement`, `BasicBlockStatement`, `IfStatement` and `DoStatement` are the
five shapes the subset covers. `SwitchStatement`, `CatchStatement`, `CatchAllStatement`,
`SynchronizedStatement` and `GeneralStatement` are refused by name.

Control flow lives on the edges rather than in the tree. Every statement carries at most one direct
successor edge, and Vineflower's own text generator emits a `break` or `continue` for it exactly when
the edge is `explicit`, is not `TYPE_REGULAR`, and does not lead to the `DummyExitStatement`. The
lowering uses the same three-part test, so a jump is emitted here if and only if the decompiler would
have printed one.

Loops are emitted uniformly as `while (true)` with the test materialised as `if (!c) break`, because
PolyAST's `While` takes a `Term` and a condition that needs statements to compute cannot be spelled
any other way. A `for`'s increment and a `do..while`'s test go into a tail after the body.

## Labelled jumps and the `#jmp` register

Vineflower keeps labelled `break` and `continue`: `StatEdge` carries `TYPE_BREAK` and `TYPE_CONTINUE`
with a `closure` naming the statement they leave. PolyAST has `Stmt.Break` and `Stmt.Cont` and both
target the innermost `While`. Bridging that is confined to three functions:

- A non-loop statement that jumps close on becomes `var once = true; while (once) { once = false; ... }`,
  since a run-once loop is the only thing `Break` can leave.
- A jump that targets the innermost wrapper is a plain `Break` (or `Cont`).
- If any jump in the function targets something other than the innermost wrapper, the whole function
  switches to routing jumps through a `#jmp` register: the jump writes the target's id and breaks, and
  each wrapper it passes through checks whether the id is its own, clearing it and resuming, or breaks
  again to keep travelling outward. A loop gains a run-once wrapper inside its `while (true)` so that a
  continue arriving from a nested wrapper lands at the loop's tail rather than outside it.

Functions that need the register are detected up front and only they pay for it. Two rules decide:

- A jump whose closure is not the innermost enclosing loop or block, which in practice means a
  labelled `break` or `continue` that leaves more than one statement.
- An explicit `continue` to a `for` or a `do..while`. PolyAST's `Cont` goes to the loop head and would
  skip the increment or the test sitting in the tail; the register form's continue is a `Break` out of
  the run-once wrapper, which lands on the tail instead.

`TreeRemapper` solves the same problem for Java's labelled `break` with a per-label boolean flag; the
single-register form here is a generalisation of that and could replace it.

## Expressions and the type discipline

Vineflower hands over an expression tree, so the operand stack problem is gone. What replaces it is
that Vineflower *infers* types where the class file only had computational ones: a `ConstExprent` of
`1` bound to a `boolean` is its judgement, not a fact.

The rule adopted is to take Vineflower's type for a variable or a constant, since it is the only
source for one, and to ignore it everywhere an operation's own semantics answer the question:

- Binary arithmetic promotes its two lowered operands the way the JVM does - `double`, then `float`,
  then `long`, then `int` - and coerces both. The result type never comes from `getExprType`.
- Shifts take their type from the left operand alone.
- `&`, `|` and `^` on two `Bool1` operands become `LogicAnd`, `LogicOr` and `LogicNeq`; anything else
  is integral.
- Comparisons promote both sides and yield `Bool1`.
- Casts, including the `I2B`/`I2C`/`I2S` truncations, take the target from the node.

Every intermediate is bound to a fresh `Stmt.Var`, which is what makes the side-effecting cases fall
out: `?:` becomes a temporary plus a `Cond` with each arm's evaluation inside its own branch, `&&`
becomes `r = a; if (a) { r = b }`, and `i++` in value position snapshots the old value first.
Snapshotting is a real operation rather than an alias, since aliasing a slot that is about to be
written gives the new value. Redundant temporaries cost nothing, because `Intrinsify`, `PartialEval`
and `DeadArgElimination` run over the result.

Local slots are named `l<index>_<version>_<type>`, so a slot reused at two types produces two
variables, which is correct and needs no liveness analysis. Parameters are copied from the ABI-visible
argument into whichever slot variable the body actually reads, since Vineflower may have narrowed the
parameter to `byte` or `boolean` where the descriptor says `int`.

## What is refused, and with what diagnostic

Every refusal names the construct in the user's vocabulary, not as an opcode number.

| Construct | Diagnostic |
| --- | --- |
| instance method using `this` | ``` `Owner.m` is not static and uses its receiver; instance methods need a receiver struct ``` |
| non-static call | ``` Unsupported call `Owner.callee(I)I` ``` |
| field access | ``` Unsupported field access `Owner.field` ``` |
| allocation | ``` Unsupported array allocation of `int[]` ``` |
| `xs.length` | ``` Unsupported `length` of an array; PolyRegion carries array lengths as separate arguments ``` |
| `switch` | ``` Unsupported `switch` ``` |
| `try`/`catch`, `finally` | ``` Unsupported `try`/`catch`; PolyRegion kernels cannot throw ``` |
| `synchronized` | ``` Unsupported `synchronized`; PolyRegion kernels have no monitors ``` |
| `throw` | ``` Unsupported `throw`; PolyRegion kernels cannot throw ``` |
| `String` constant or concatenation | ``` Unsupported `String` constant; PolyRegion has no device-side String ``` |
| `instanceof` | ``` Unsupported `instanceof`; PolyRegion has no device heap ``` |
| null comparison | ``` Unsupported `null`; PolyRegion has no device heap ``` |
| non-primitive type | ``` Cannot lower the JVM type `java.util.List`; only primitives and their arrays travel to a device ``` |
| unstructurable method | ``` Vineflower could not recover structured control flow for `Owner.m`; the method keeps an unstructured region ``` |
| decompiler warning | ``` Vineflower reported <comment> while decompiling `Owner.m` ``` |

A lambda is refused at its functional interface type rather than at its bootstrap method, because the
`invokedynamic` has already become a value of that type by the time the tree is handed over.

The last two rows are the ones that matter for trusting the rest. Vineflower reports a structuring
failure by leaving a `GeneralStatement` in the tree, and reports anything else it is unhappy about by
hanging a comment off the `RootStatement`. Both are checked before a single statement is lowered, so a
method it could not handle is refused rather than lowered optimistically.

## Testing

`JavaInMemory` compiles Java with the running JDK's javac and `ScalaInMemory` compiles Scala with the
Scala 3 compiler this build already uses; both keep the class files in memory. `Differential` lowers a
named method, runs it in `Interpreter.Vm`, and compares the result and every array argument against
the same method invoked reflectively over many inputs.

The scalac half exists because non-javac bytecode was the largest identified risk, and it is written
so as not to look like Java: `while` over `var`s because Scala's `for` desugars into calls, and
`@tailrec` methods because the Scala backend lowers those into a `goto` back to the method entry that
no javac fixture would produce. `fact` in particular is asserted to produce a loop, since the class
file contains no loop header for one.

`Float32` arithmetic in `Interpreter` is evaluated at `Double` width, so the differential tests
deliberately avoid `float`. That is an interpreter limitation, not a lowering one, but it means the
`float` paths are unexercised.

## The census

`Census`, `BytecodeCensusSuite` and `BytecodeJarDifferentialSuite` run over three jars nobody here
compiled: `scala-library` and `scala3-compiler` are scalac output, `guava` is javac output pinned at
`33.5.0-jre`. They are three separate things and the difference matters.

**The no-crash invariant is the gate.** Every method of a jar is lowered, and the frontend must either
lower it or refuse it with a diagnostic. `Result` makes the distinction exact: a refusal is a
`CompilerException` and nothing else produces one, so any other `Throwable` arriving in the `Left` is
an exception that escaped and fails the suite naming the class, method and descriptor. A stack
overflow counts, since `Vineflower.withContext` catches `Throwable` and would otherwise launder it
into a refusal. So does a hang: the scan runs on a daemon thread publishing a heartbeat and the caller
gives up on it after two minutes without progress, because Vineflower serialises decompilation on a
process-wide monitor and a wedged method cannot be abandoned and retried.

**The census is a measurement, not an assertion.** Only two things are asserted: no crashes, and a
floor on the lowered count set at roughly two thirds of what was observed. The refusal reasons are
printed ranked by frequency, bucketed by masking the symbol out of each diagnostic while keeping the
construct name, so `` Unsupported `switch` `` and `` Unsupported `instanceof` `` stay apart while ten
thousand distinct callees collapse into one line.

A full pass is 147,981 methods and about a minute, so the default `test` walks a 500-class sample
drawn with a fixed seed off the sorted entry list and `sbt compiler/censusFull` walks the whole jar.
The floors apply to the sample only.

| Corpus | Classes | Methods | Lowered | Crashed |
| --- | --- | --- | --- | --- |
| scala-library 3.8.3 | 3658 | 47230 | 767 (1.62%) | 0 |
| scala3-compiler 3.8.3 | 5393 | 85157 | 2864 (3.36%) | 0 |
| guava 33.5.0-jre | 1961 | 15594 | 270 (1.73%) | 0 |

Refusals over all three, ranked:

| Count | Reason |
| --- | --- |
| 113326 | non-primitive parameter or return type |
| 17486 | call |
| 7336 | instance method using its receiver |
| 5515 | field access |
| 97 | `arraylength` |
| 86 | `throw` |
| 57 | `switch` |
| 42 | `instanceof` |
| 40 | `try`/`catch` |
| 36 | Vineflower could not simplify a switch on an enum |
| 34 | allocation |
| 9 | `finally` |
| 9 | `++`/`--` on a field |
| 4 | `null` |
| 3 | other Vineflower warnings |

The shape of that list is the roadmap: the head is the long tail of types, calls, fields and
receivers, and every structural construct together is under 300 methods. Two entries are new
information rather than confirmation:

- `++`/`--` on a field arrives as an `IPP` over a `FieldExprent`, which `step` rejects with its own
  message rather than with the field diagnostic. It disappears when fields land, but it is a distinct
  path through the lowering that no fixture reaches.
- Vineflower's own `switch`-on-enum failures show up as decompiler warnings rather than as a
  `SwitchStatement`, so `switch` support will not by itself recover those 36 methods.

**Differential correctness runs on the subset that can be executed.** Candidates are discovered by
signature rather than hand-picked: every method whose parameters and return are scalar primitives, that
is either static or a `this`-free method on a class with a `MODULE$`, and that actually lowers. Each is
then run in `Interpreter.Vm` and invoked reflectively over the same generated inputs. 331 methods were
discovered (115 in scala-library, 162 in scala3-compiler, 54 in guava) and about 25,000 inputs compared
with no disagreement.

Arrays are excluded from discovery, not from the frontend: with `arraylength` refused, a lowered method
taking an array must be indexed from some other argument and no signature says which.

The one disagreement discovery did find is the documented one. `scala.runtime.RichDouble.intValue` is a
bare `d2i`, and on an infinity the JVM saturates to `2147483647` where PolyRegion gives `-1`. That is
the deviation listed under "Supported subset" rather than a lowering bug, and it is reachable from
ordinary library code rather than only from a deliberately hostile fixture. The infinities are kept out
of the generated pool for that reason; `NaN` agrees and stays in.

## Not yet built

Nothing new is needed downstream. `BytecodeRemapper.lower` produces a `PolyAST.Function`, and
`lowerToProgram` wraps it in a `PolyAST.Program`. From there the existing runtime path applies
unchanged:

```scala
val program = BytecodeRemapper.lowerToProgram(classBytes, name, descriptor)
val ast     = MsgPack.encode(CodeGen.polyASTVersioned(program))
val binary  = ct.Compiler.create().compile(ast, false, ct.Options.of(target, arch), opt)
```

That is the same sequence `compiletime.scala` runs on its `jit` branch and that `polyc_jit.cpp`
services. The class bytes come from `ClassLoader.getResourceAsStream` on the binary name, or from
ByteBuddy's `TypePool` when the class must not be loaded. Neither needs the class initialised, which
matters for the open-world case: a call site can be specialised for a receiver class the moment its
concrete type is first observed.

The pieces missing before this is dispatchable are the capture plan and the entry convention, both of
which `TreeRemapper.compileRegion` and `Offloader` already implement for tier 2 and neither of which is
bytecode-specific:

- Array parameters need the companion `#len<i>` argument, since `Ptr` carries no length.
- The entry's parameters have to move into `termCaptures` in the order the runtime packs them.
- A non-`Unit0` return has to be routed through the `#result` output buffer.

### invokedynamic

A lambda's call site names `LambdaMetafactory.metafactory` or `altMetafactory` as its bootstrap, and
the class file's `BootstrapMethods` attribute already carries the implementation `MethodHandle` as a
static argument, so the target is resolvable without running the bootstrap. Under Vineflower the
resolution has already happened: `LambdaProcessor` turns the call site into a `ClassNode` of type
`LAMBDA` carrying `content_method_name` and `content_method_descriptor`, which is the method to lower
and inline. That processor is part of `ClassesProcessor`, not of `codeToJava`, so taking this path
means running enough of the class-level pipeline to have the lambda nodes.

Refused by name, with the reason:

- `StringConcatFactory.makeConcatWithConstants` - PolyRegion has no device-side `String`.
- `ObjectMethods.bootstrap` - record `equals`/`hashCode`/`toString`, which need object identity.
- `SwitchBootstraps.typeSwitch` and `enumSwitch` - pattern matching on reference types.
- `ConstantBootstraps.*` - condy; the primitive cases could be folded later but are not needed.

## Cost

Estimates assume the tail is written to `TreeRemapper`'s standard.

| Piece | Estimate |
| --- | --- |
| `switch` statements onto PolyAST | 2-3 days; `SwitchStatement` is already recovered, this is the lowering only |
| Fields and structs | 1 week, mostly reconstructing `StructDef`s from descriptors and resolving them across class files |
| Calls and the transitive worklist | 1 week, mirroring `compileTransitive` |
| Allocation | 3-4 days, reusing `TreeRemapper.allocZeroed` and `allocRows` |
| `arraylength` and the `#len` capture plan | 3-4 days |
| `invokedynamic` lambda resolution | 3-4 days, mostly running enough of `ClassesProcessor` to have the lambda nodes |
| Entry convention, capture plan, `#result` routing, processor wiring | 1 week |
| Exception handlers onto PolyAST | open question; the structure is recovered, but PolyAST has no unwinding |
| Instance methods and virtual dispatch for the open-world case | multi-week; this is a devirtualisation problem, not a bytecode problem |

Roughly four to five weeks to a tier-3 frontend matching what `TreeRemapper` accepts today.

## Risks

**Vineflower's `Statement`/`Exprent` tree is not a stable API.** The only compatibility promise covers
`...main.extern`, and this frontend reaches well past that: `MethodProcessor.codeToJava`,
`StructGeneralAttribute.init`, `DecompilerContext`'s property keys, and the whole `stats`/`exps`
hierarchy. The version is pinned exactly and the mapping should be expected to need work on upgrade.
The differential suite is what turns that from a silent risk into a build failure.

**Vineflower folds expressions.** `(int)(double)x` comes back as `(int)x`. That instance is sound, but
it is a reminder that the tree is the decompiler's opinion of the method rather than a transcription
of it. Anywhere the opinion is load-bearing the descriptor is preferred.

**Non-javac bytecode** was the largest identified risk and is now largely retired. scalac's tail
recursion, module classes and `while`-over-`var` loops all lower correctly, and the census puts 148,000
methods of scalac and javac output through the frontend without one escaping exception. What remains
untested is bytecode from a third kind of producer. Kotlin's coroutine state machines are
`tableswitch`-based and would be refused at `switch` rather than mislowered, which is a prediction
rather than a measurement. Obfuscated bytecode is untested.

**The lock.** One decompilation at a time, process-wide, because `DecompilerContext` is a thread local
and the attribute registry is a static map. That is fine for a JIT path that compiles one region at a
time and would need revisiting if tier 3 ever ran during a parallel build.

Smaller ones:

- Overload collapsing: `Function.name` is `owner.method`, so two overloads of the same name collide.
  The descriptor has to be folded into the symbol before calls are supported, exactly as
  `TreeRemapper` notes for its own case.
- `Float32` paths are unexercised, because `Interpreter` evaluates them at `Double` width.
