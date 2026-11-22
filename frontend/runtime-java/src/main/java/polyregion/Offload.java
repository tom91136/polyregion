package polyregion;

import polyregion.javalang.Interface;

public final class Offload {

  private Offload() {}

  @FunctionalInterface
  public interface FnDD_D {
    double apply(double arg0, double arg1);
  }

  @FunctionalInterface
  public interface FnFF_F {
    float apply(float arg0, float arg1);
  }

  @FunctionalInterface
  public interface FnII_I {
    int apply(int arg0, int arg1);
  }

  @FunctionalInterface
  public interface FnJJ_J {
    long apply(long arg0, long arg1);
  }

  @FunctionalInterface
  public interface FnJ_D {
    double apply(long arg0);
  }

  @FunctionalInterface
  public interface FnJ_F {
    float apply(long arg0);
  }

  @FunctionalInterface
  public interface FnJ_I {
    int apply(long arg0);
  }

  @FunctionalInterface
  public interface FnJ_J {
    long apply(long arg0);
  }

  @FunctionalInterface
  public interface FnJ_V {
    void apply(long arg0);
  }

  @Interface("polyregion.core:polyregion.core.for_each")
  public static void for_each(long n, FnJ_V body) {
    throw new UnsupportedOperationException("compiler did not replace polyregion.core.for_each");
  }

  @Interface("polyregion.core:polyregion.core.for_each")
  public static void for_each(polyregion.jvm.runtime.Context context, long n, FnJ_V body) {
    throw new UnsupportedOperationException("compiler did not replace polyregion.core.for_each");
  }

  @Interface("polyregion.core:polyregion.core.transform_reduce")
  public static int transform_reduce(long n, Integer init, FnJ_I map, FnII_I combine) {
    throw new UnsupportedOperationException("compiler did not replace polyregion.core.transform_reduce");
  }

  @Interface("polyregion.core:polyregion.core.transform_reduce")
  public static int transform_reduce(
      polyregion.jvm.runtime.Context context,
      long n,
      Integer init,
      FnJ_I map,
      FnII_I combine) {
    throw new UnsupportedOperationException("compiler did not replace polyregion.core.transform_reduce");
  }

  @Interface("polyregion.core:polyregion.core.transform_reduce")
  public static long transform_reduce(long n, Long init, FnJ_J map, FnJJ_J combine) {
    throw new UnsupportedOperationException("compiler did not replace polyregion.core.transform_reduce");
  }

  @Interface("polyregion.core:polyregion.core.transform_reduce")
  public static long transform_reduce(
      polyregion.jvm.runtime.Context context,
      long n,
      Long init,
      FnJ_J map,
      FnJJ_J combine) {
    throw new UnsupportedOperationException("compiler did not replace polyregion.core.transform_reduce");
  }

  @Interface("polyregion.core:polyregion.core.transform_reduce")
  public static float transform_reduce(long n, Float init, FnJ_F map, FnFF_F combine) {
    throw new UnsupportedOperationException("compiler did not replace polyregion.core.transform_reduce");
  }

  @Interface("polyregion.core:polyregion.core.transform_reduce")
  public static float transform_reduce(
      polyregion.jvm.runtime.Context context,
      long n,
      Float init,
      FnJ_F map,
      FnFF_F combine) {
    throw new UnsupportedOperationException("compiler did not replace polyregion.core.transform_reduce");
  }

  @Interface("polyregion.core:polyregion.core.transform_reduce")
  public static double transform_reduce(long n, Double init, FnJ_D map, FnDD_D combine) {
    throw new UnsupportedOperationException("compiler did not replace polyregion.core.transform_reduce");
  }

  @Interface("polyregion.core:polyregion.core.transform_reduce")
  public static double transform_reduce(
      polyregion.jvm.runtime.Context context,
      long n,
      Double init,
      FnJ_D map,
      FnDD_D combine) {
    throw new UnsupportedOperationException("compiler did not replace polyregion.core.transform_reduce");
  }

  @Interface("polyregion.core:polyregion.core.transform_reduce_unordered")
  public static int transform_reduce_unordered(long n, Integer init, FnJ_I map, FnII_I combine) {
    throw new UnsupportedOperationException("compiler did not replace polyregion.core.transform_reduce_unordered");
  }

  @Interface("polyregion.core:polyregion.core.transform_reduce_unordered")
  public static int transform_reduce_unordered(
      polyregion.jvm.runtime.Context context,
      long n,
      Integer init,
      FnJ_I map,
      FnII_I combine) {
    throw new UnsupportedOperationException("compiler did not replace polyregion.core.transform_reduce_unordered");
  }

  @Interface("polyregion.core:polyregion.core.transform_reduce_unordered")
  public static long transform_reduce_unordered(long n, Long init, FnJ_J map, FnJJ_J combine) {
    throw new UnsupportedOperationException("compiler did not replace polyregion.core.transform_reduce_unordered");
  }

  @Interface("polyregion.core:polyregion.core.transform_reduce_unordered")
  public static long transform_reduce_unordered(
      polyregion.jvm.runtime.Context context,
      long n,
      Long init,
      FnJ_J map,
      FnJJ_J combine) {
    throw new UnsupportedOperationException("compiler did not replace polyregion.core.transform_reduce_unordered");
  }

  @Interface("polyregion.core:polyregion.core.transform_reduce_unordered")
  public static float transform_reduce_unordered(long n, Float init, FnJ_F map, FnFF_F combine) {
    throw new UnsupportedOperationException("compiler did not replace polyregion.core.transform_reduce_unordered");
  }

  @Interface("polyregion.core:polyregion.core.transform_reduce_unordered")
  public static float transform_reduce_unordered(
      polyregion.jvm.runtime.Context context,
      long n,
      Float init,
      FnJ_F map,
      FnFF_F combine) {
    throw new UnsupportedOperationException("compiler did not replace polyregion.core.transform_reduce_unordered");
  }

  @Interface("polyregion.core:polyregion.core.transform_reduce_unordered")
  public static double transform_reduce_unordered(long n, Double init, FnJ_D map, FnDD_D combine) {
    throw new UnsupportedOperationException("compiler did not replace polyregion.core.transform_reduce_unordered");
  }

  @Interface("polyregion.core:polyregion.core.transform_reduce_unordered")
  public static double transform_reduce_unordered(
      polyregion.jvm.runtime.Context context,
      long n,
      Double init,
      FnJ_D map,
      FnDD_D combine) {
    throw new UnsupportedOperationException("compiler did not replace polyregion.core.transform_reduce_unordered");
  }
}
