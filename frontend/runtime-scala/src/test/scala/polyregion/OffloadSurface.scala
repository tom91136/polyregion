package polyregion

object OffloadSurface {
  def compile(context: polyregion.jvm.runtime.Context): Unit = InterfaceSurfaceCheck.erase {
    // for_each
    offload.for_each(0L, (arg0: Long) => ())

    // for_each: context
    offload.for_each(context, 0L, (arg0: Long) => ())

    // transform_reduce: T=Int
    offload.transform_reduce(0L, 0, (arg0: Long) => 0, (arg0: Int, arg1: Int) => 0)

    // transform_reduce: T=Int: context
    offload.transform_reduce(context, 0L, 0, (arg0: Long) => 0, (arg0: Int, arg1: Int) => 0)

    // transform_reduce: T=Long
    offload.transform_reduce(0L, 0L, (arg0: Long) => 0L, (arg0: Long, arg1: Long) => 0L)

    // transform_reduce: T=Long: context
    offload.transform_reduce(context, 0L, 0L, (arg0: Long) => 0L, (arg0: Long, arg1: Long) => 0L)

    // transform_reduce: T=Float
    offload.transform_reduce(0L, 0.0f, (arg0: Long) => 0.0f, (arg0: Float, arg1: Float) => 0.0f)

    // transform_reduce: T=Float: context
    offload.transform_reduce(context, 0L, 0.0f, (arg0: Long) => 0.0f, (arg0: Float, arg1: Float) => 0.0f)

    // transform_reduce: T=Double
    offload.transform_reduce(0L, 0.0, (arg0: Long) => 0.0, (arg0: Double, arg1: Double) => 0.0)

    // transform_reduce: T=Double: context
    offload.transform_reduce(context, 0L, 0.0, (arg0: Long) => 0.0, (arg0: Double, arg1: Double) => 0.0)

    // for_each: rank 2
    offload.for_each(0L, 0L, (arg0: Long, arg1: Long) => ())

    // for_each: rank 2
    offload.for_each(context, 0L, 0L, (arg0: Long, arg1: Long) => ())

    // for_each: rank 3
    offload.for_each(0L, 0L, 0L, (arg0: Long, arg1: Long, arg2: Long) => ())

    // for_each: rank 3
    offload.for_each(context, 0L, 0L, 0L, (arg0: Long, arg1: Long, arg2: Long) => ())

    // transform_reduce: rank 2
    offload.transform_reduce(0L, 0L, 0, (arg0: Long, arg1: Long) => 0, (arg0: Int, arg1: Int) => 0)

    // transform_reduce: rank 2
    offload.transform_reduce(context, 0L, 0L, 0, (arg0: Long, arg1: Long) => 0, (arg0: Int, arg1: Int) => 0)

    // transform_reduce: rank 3
    offload.transform_reduce(0L, 0L, 0L, 0, (arg0: Long, arg1: Long, arg2: Long) => 0, (arg0: Int, arg1: Int) => 0)

    // transform_reduce: rank 3
    offload.transform_reduce(
      context,
      0L,
      0L,
      0L,
      0,
      (arg0: Long, arg1: Long, arg2: Long) => 0,
      (arg0: Int, arg1: Int) => 0
    )

    // task
    offload.task(() => ())

    // task
    offload.task(context, () => ())
  }
}
