package polyregion;

public final class OffloadSurface {

  private static final class Value {}

  public static void compile() {
    int n = 4;
    int[] ints = new int[n];
    long[] longs = new long[n];
    float[] floats = new float[n];
    double[] doubles = new double[n];
    Value[] values = new Value[n];
    polyregion.jvm.runtime.Context context = null;

    // for_each
    Offload.for_each(0L, (arg0) -> { });

    // for_each: context
    Offload.for_each(context, 0L, (arg0) -> { });

    // transform_reduce: T=int
    Offload.transform_reduce(0L, 0, (arg0) -> 0, (arg0, arg1) -> 0);

    // transform_reduce: T=int: context
    Offload.transform_reduce(context, 0L, 0, (arg0) -> 0, (arg0, arg1) -> 0);

    // transform_reduce: T=long
    Offload.transform_reduce(0L, 0L, (arg0) -> 0L, (arg0, arg1) -> 0L);

    // transform_reduce: T=long: context
    Offload.transform_reduce(context, 0L, 0L, (arg0) -> 0L, (arg0, arg1) -> 0L);

    // transform_reduce: T=float
    Offload.transform_reduce(0L, 0.0f, (arg0) -> 0.0f, (arg0, arg1) -> 0.0f);

    // transform_reduce: T=float: context
    Offload.transform_reduce(context, 0L, 0.0f, (arg0) -> 0.0f, (arg0, arg1) -> 0.0f);

    // transform_reduce: T=double
    Offload.transform_reduce(0L, 0.0, (arg0) -> 0.0, (arg0, arg1) -> 0.0);

    // transform_reduce: T=double: context
    Offload.transform_reduce(context, 0L, 0.0, (arg0) -> 0.0, (arg0, arg1) -> 0.0);
  }
}
