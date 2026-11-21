package polyregion

import scala.annotation.compileTimeOnly

object offload {

  @compileTimeOnly("polyregion_interface:polyregion.core:polyregion.core.for_each")
  inline def for_each(n: Long, inline body: Long => Unit): Unit =
    polyregion.scalalang.launch.call[Unit](
      "polyregion.core:polyregion.core.for_each",
      polyregion.jvm.runtime.Context.current(),
      n,
      body
    )

  @compileTimeOnly("polyregion_interface:polyregion.core:polyregion.core.for_each")
  inline def for_each(context: polyregion.jvm.runtime.Context, n: Long, inline body: Long => Unit): Unit =
    polyregion.scalalang.launch.call[Unit]("polyregion.core:polyregion.core.for_each", context, n, body)

  @compileTimeOnly("polyregion_interface:polyregion.core:polyregion.core.transform_reduce")
  inline def transform_reduce[T](n: Long, init: T, inline map: Long => T, inline combine: (T, T) => T): T =
    polyregion.scalalang.launch.call[T](
      "polyregion.core:polyregion.core.transform_reduce",
      polyregion.jvm.runtime.Context.current(),
      n,
      init,
      map,
      combine
    )

  @compileTimeOnly("polyregion_interface:polyregion.core:polyregion.core.transform_reduce")
  inline def transform_reduce[T](
      context: polyregion.jvm.runtime.Context,
      n: Long,
      init: T,
      inline map: Long => T,
      inline combine: (T, T) => T
  ): T =
    polyregion.scalalang.launch.call[T](
      "polyregion.core:polyregion.core.transform_reduce",
      context,
      n,
      init,
      map,
      combine
    )

  inline def for_each(n0: Long, n1: Long, inline body: (Long, Long) => Unit): Unit =
    for_each(Math.multiplyExact(n0, n1), (i: Long) => body(i / n1, i % n1))

  inline def for_each(n0: Long, n1: Long, n2: Long, inline body: (Long, Long, Long) => Unit): Unit =
    for_each(Math.multiplyExact(Math.multiplyExact(n0, n1), n2), (i: Long) => body(i / (n1 * n2), i / n2 % n1, i % n2))

  inline def for_each(
      context: polyregion.jvm.runtime.Context,
      n0: Long,
      n1: Long,
      inline body: (Long, Long) => Unit
  ): Unit =
    for_each(context, Math.multiplyExact(n0, n1), (i: Long) => body(i / n1, i % n1))

  inline def for_each(
      context: polyregion.jvm.runtime.Context,
      n0: Long,
      n1: Long,
      n2: Long,
      inline body: (Long, Long, Long) => Unit
  ): Unit =
    for_each(
      context,
      Math.multiplyExact(Math.multiplyExact(n0, n1), n2),
      (i: Long) => body(i / (n1 * n2), i / n2 % n1, i % n2)
    )

  inline def transform_reduce[T](
      n0: Long,
      n1: Long,
      init: T,
      inline map: (Long, Long) => T,
      inline combine: (T, T) => T
  ): T =
    transform_reduce(Math.multiplyExact(n0, n1), init, (i: Long) => map(i / n1, i % n1), combine)

  inline def transform_reduce[T](
      n0: Long,
      n1: Long,
      n2: Long,
      init: T,
      inline map: (Long, Long, Long) => T,
      inline combine: (T, T) => T
  ): T =
    transform_reduce(
      Math.multiplyExact(Math.multiplyExact(n0, n1), n2),
      init,
      (i: Long) => map(i / (n1 * n2), i / n2 % n1, i % n2),
      combine
    )

  inline def transform_reduce[T](
      context: polyregion.jvm.runtime.Context,
      n0: Long,
      n1: Long,
      init: T,
      inline map: (Long, Long) => T,
      inline combine: (T, T) => T
  ): T =
    transform_reduce(context, Math.multiplyExact(n0, n1), init, (i: Long) => map(i / n1, i % n1), combine)

  inline def transform_reduce[T](
      context: polyregion.jvm.runtime.Context,
      n0: Long,
      n1: Long,
      n2: Long,
      init: T,
      inline map: (Long, Long, Long) => T,
      inline combine: (T, T) => T
  ): T =
    transform_reduce(
      context,
      Math.multiplyExact(Math.multiplyExact(n0, n1), n2),
      init,
      (i: Long) => map(i / (n1 * n2), i / n2 % n1, i % n2),
      combine
    )

  inline def task(inline body: () => Unit): Unit =
    for_each(1L, (_: Long) => body())

  inline def task(context: polyregion.jvm.runtime.Context, inline body: () => Unit): Unit =
    for_each(context, 1L, (_: Long) => body())
}
