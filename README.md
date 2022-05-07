# Polyregion


Polyregion is a macro **library** that compiles normal and idiomatic Scala code fragments to machine code, targeting both CPUs and GPUs. 


## Supported target

* CPU

TODO

* OpenCL (1.2 minimum)
* CUDA
* Level Zero
* HIP

## Supported execution mode

* JVM

TODO

* Scala Native
* ScalaJS (WASM)
* Kotlin Multiplatform

## Supported source language

* Scala

TODO

* Java
* Kotlin

## Supported constructs in offload region

* All Scala/Java's primitive types, including `Unit`.
* All primitive control flows (e.g. `if then else` , `while`, `return`).
* Arbitrary function calls to definitions outside of the offload region, regardless of whether marked with `inline`.
* Generics and arbitrary classes/objects, including tuples, enums, and case classes.
* Dynamic memory allocation/deallocation for supported targets (e.g. instances allocated via `new`)
* Calls to `scala.math`/`java.lang.Math` methods replaced with calls to intrinsic where possible or `libm` as fallback.
* Instantiation and access of `Array[A]`
* Numeric conversions (e.g. `42.toByte`, `0.3f.toDouble`)
* Implicit conversions/extensions (e.g. `RichInt`, `ArrayOps`, etc)

* Heap memory access via `polyregion.Buffer`,
  where `trait polyregion.Buffer[A] extends scala.collection.mutable.IndexedSeq[A]`.

TODO

* Nested `def`
* `lazy val`
* `try catch` & `throw`
* Pattern matching
* Delegation of `Console.println` to `print` in the runtime
* String
* Tuples
* Casting (`asInstanceOf`)

# FAQ

## Is this some kind of DSL library that generates C-like source (e.g. CUDA/OpenCL)?

No, Polyregion transparently compiles annotated code blocks containing normal Scala code to high-performance machine code.
The generated code is then directly embedded in-place at call site where possible along with support for captures (e.g. class fields, local variables, and functions).
At runtime, calls to the code block will invoke the embedded code through JNI and serialise any captures and deserialise any return values or side effects.

## Does Polyregion use reflection?

Only compile-time reflection (i.e. macros) in Scala 3.

## Is is fast?

Boundary-crossing (i.e. FFI into native code and back) performance at runtime should be equivalent to hand written JNI bindings.
The actual performance of the code block is dependent on the generated machine code by LLVM for the selected platform.

## Doesn't Scala Native already exists? How does this compare to Scala Native?

Scala Native requires your whole Scala code base to be compiled to native code much like in C/C++.
Currently, this means you either get a fully native Scala program or you have to fallback to the JVM
if you need features such as multithreading, full-blown reflection, or anything that only works on
the JVM essentially (e.g Java-only libraries). Scala Native also doesn't have support for
accelerators (GPUs) even though LLVM is the underlying backend.

Polyregion, on the other hand, has a different execution model where your program still runs on the
JVM but select methods can be placed behind a by-name function `offload` for which the body of
method then gets compiled to native code for either the host CPU or an accelerator through LLVM's
codegen. Polyregion handles the embedding of the compiled native code and the invocation in a
transparent way so minimal change is needed to obtain a high-performance native method.

## How does this compare to Aparapi/TornadoVM?

Polyregion operates at the language AST level where we have a better view of the program. Aparapi
operates on the Java Bytecode directly and has limited support for advanced constructs, the project
also only supports OpenCL which may be a limiting factor.

TornadoVM also operates on the Bytecode but introduces advanced optimisation passes by modifying the JVM itself.
This is a significant restriction as TornadoVM will only work with specific versions of the JVM.

Polyregion only uses features supported under the JavaSE specification such as JNI and does not use
any VM specific features. Currently, we have verified correct operation on common OpenJDK builds,
GraalVM CE/EE, and also Eclipse OpenJ9.







