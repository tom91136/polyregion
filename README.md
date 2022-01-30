

# Polyregion

Your favourite language now runs the GPU!

## Supported target

 * CPU

 TODO
  * OpenCL (1.2 minimum)
  * CUDA

 ## Supported execution mode

 * JVM

 TODO 
  * Scala Native
  * ScalaJS (WASM)

## Supported source language

 * Scala 

TODO 
 * Java
 * Kotlin


## Supported constructs

 * All Scala/Java's primitive types, including `Unit`.
 * All primitive control flows (e.g `if else` , `while`, `return`).
 * Arbitrary case classes via derivation of the `NativeStruct` typeclass.
 * Heap memory access via `polyregion.Buffer`, where `trait polyregion.Buffer[A] extends scala.collection.mutable.IndexedSeq[A]`.
 * Delegation of all `scala.math`/`java.lang.Math` calls to `libm`.
 * Inline functions work as expected.
 * Math calls on primitives work as expected.
 * General constant propagation (e.g `Int.MaxValue`)
 
TODO

 * Nested `def`
 * `lazy val`
 * `try catch`
 * Instantiation and access of Arrays in offload 
 * Instantiation and access of Arrays as parameter
 * Case class construction 
 * Pattern matching
 * Delegation of `Console.println` to `print` in the runtime
 * Offload dynamic memory allocation
 * Generics
 * String
 * Tuples
 * Arbitrary method calls 
 * Casting (`asInstanceOf`)
 * Numeric conversion (`to<Type>`)




FAQ:

## Doesn't Scala Native already exists? How does this compare to Scala Native?

Scala Native requires your whole Scala code base to be compiled to native code much like in C/C++.
Currently, this means you either get a fully native Scala program or you have to fallback to the JVM if you need features such as multithreading, full-blown reflection, or anything that only works on the JVM essentially (e.g Java-only libraries).
Scala Native also doesn't have support for accelerators (GPUs) even though LLVM is the underlying backend.

Polyregion, on the other hand, has a different execution model where your program still runs on the JVM but select methods can be placed behind a by-name function `offload` for which the body of method then gets compiled to native code for either the host CPU or an accelerator through LLVM's codegen.
Polyregion handles the embedding of the compiled native code and the invocation in a transparent way so minimal change is needed to obtain a high-performance native method.


## How does this compare to Aparapi/TornadoVM?

Polyregion operates at the language AST level where we have a better view of the program.
Aparapi operates on the Java Bytecode directly and has limited support for advanced constructs, the project also only supports OpenCL which may be a limiting factor.

TornadoVM also operates on the Bytecode but introduces advanced optimisation passes as a JVM plugin that is specific for Oracle's GraalVM.
This is a significant restriction as TornadoVM will only work with specific versions of GraalVM.

Polyregion only uses features supported under the JavaSE specification (e.g JNI) and does not use any VM specific features.
Currently, we have verified correct operation on common OpenJDK builds, GraalVM CE and EE, and also Eclipse OpenJ9.







