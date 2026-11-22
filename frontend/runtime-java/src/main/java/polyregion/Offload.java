package polyregion;

import polyregion.javalang.Interface;
import polyregion.javalang.Launch;

public final class Offload {

  private Offload() {}

  @FunctionalInterface
  public interface FnDD_D extends java.io.Serializable {
    double apply(double arg0, double arg1);
  }

  @FunctionalInterface
  public interface FnFF_F extends java.io.Serializable {
    float apply(float arg0, float arg1);
  }

  @FunctionalInterface
  public interface FnII_I extends java.io.Serializable {
    int apply(int arg0, int arg1);
  }

  @FunctionalInterface
  public interface FnJJ_J extends java.io.Serializable {
    long apply(long arg0, long arg1);
  }

  @FunctionalInterface
  public interface FnJ_D extends java.io.Serializable {
    double apply(long arg0);
  }

  @FunctionalInterface
  public interface FnJ_F extends java.io.Serializable {
    float apply(long arg0);
  }

  @FunctionalInterface
  public interface FnJ_I extends java.io.Serializable {
    int apply(long arg0);
  }

  @FunctionalInterface
  public interface FnJ_J extends java.io.Serializable {
    long apply(long arg0);
  }

  @FunctionalInterface
  public interface FnJ_V extends java.io.Serializable {
    void apply(long arg0);
  }

  @Interface("polyregion.core:polyregion.core.for_each")
  public static void for_each(long n, FnJ_V body) {
    Launch.call("polyregion.core:polyregion.core.for_each", null, new Object[] {n, body});
  }

  @Interface("polyregion.core:polyregion.core.for_each")
  public static void for_each(polyregion.jvm.runtime.Context context, long n, FnJ_V body) {
    Launch.call("polyregion.core:polyregion.core.for_each", context, new Object[] {n, body});
  }

  @Interface("polyregion.core:polyregion.core.transform_reduce")
  public static int transform_reduce(long n, Integer init, FnJ_I map, FnII_I combine) {
    return (Integer) Launch.call("polyregion.core:polyregion.core.transform_reduce", null, new Object[] {n, init, map, combine});
  }

  @Interface("polyregion.core:polyregion.core.transform_reduce")
  public static int transform_reduce(
      polyregion.jvm.runtime.Context context,
      long n,
      Integer init,
      FnJ_I map,
      FnII_I combine) {
    return (Integer) Launch.call("polyregion.core:polyregion.core.transform_reduce", context, new Object[] {n, init, map, combine});
  }

  @Interface("polyregion.core:polyregion.core.transform_reduce")
  public static long transform_reduce(long n, Long init, FnJ_J map, FnJJ_J combine) {
    return (Long) Launch.call("polyregion.core:polyregion.core.transform_reduce", null, new Object[] {n, init, map, combine});
  }

  @Interface("polyregion.core:polyregion.core.transform_reduce")
  public static long transform_reduce(
      polyregion.jvm.runtime.Context context,
      long n,
      Long init,
      FnJ_J map,
      FnJJ_J combine) {
    return (Long) Launch.call("polyregion.core:polyregion.core.transform_reduce", context, new Object[] {n, init, map, combine});
  }

  @Interface("polyregion.core:polyregion.core.transform_reduce")
  public static float transform_reduce(long n, Float init, FnJ_F map, FnFF_F combine) {
    return (Float) Launch.call("polyregion.core:polyregion.core.transform_reduce", null, new Object[] {n, init, map, combine});
  }

  @Interface("polyregion.core:polyregion.core.transform_reduce")
  public static float transform_reduce(
      polyregion.jvm.runtime.Context context,
      long n,
      Float init,
      FnJ_F map,
      FnFF_F combine) {
    return (Float) Launch.call("polyregion.core:polyregion.core.transform_reduce", context, new Object[] {n, init, map, combine});
  }

  @Interface("polyregion.core:polyregion.core.transform_reduce")
  public static double transform_reduce(long n, Double init, FnJ_D map, FnDD_D combine) {
    return (Double) Launch.call("polyregion.core:polyregion.core.transform_reduce", null, new Object[] {n, init, map, combine});
  }

  @Interface("polyregion.core:polyregion.core.transform_reduce")
  public static double transform_reduce(
      polyregion.jvm.runtime.Context context,
      long n,
      Double init,
      FnJ_D map,
      FnDD_D combine) {
    return (Double) Launch.call("polyregion.core:polyregion.core.transform_reduce", context, new Object[] {n, init, map, combine});
  }

  @Interface("polyregion.core:polyregion.core.transform_reduce_unordered")
  public static int transform_reduce_unordered(long n, Integer init, FnJ_I map, FnII_I combine) {
    return (Integer) Launch.call("polyregion.core:polyregion.core.transform_reduce_unordered", null, new Object[] {n, init, map, combine});
  }

  @Interface("polyregion.core:polyregion.core.transform_reduce_unordered")
  public static int transform_reduce_unordered(
      polyregion.jvm.runtime.Context context,
      long n,
      Integer init,
      FnJ_I map,
      FnII_I combine) {
    return (Integer) Launch.call("polyregion.core:polyregion.core.transform_reduce_unordered", context, new Object[] {n, init, map, combine});
  }

  @Interface("polyregion.core:polyregion.core.transform_reduce_unordered")
  public static long transform_reduce_unordered(long n, Long init, FnJ_J map, FnJJ_J combine) {
    return (Long) Launch.call("polyregion.core:polyregion.core.transform_reduce_unordered", null, new Object[] {n, init, map, combine});
  }

  @Interface("polyregion.core:polyregion.core.transform_reduce_unordered")
  public static long transform_reduce_unordered(
      polyregion.jvm.runtime.Context context,
      long n,
      Long init,
      FnJ_J map,
      FnJJ_J combine) {
    return (Long) Launch.call("polyregion.core:polyregion.core.transform_reduce_unordered", context, new Object[] {n, init, map, combine});
  }

  @Interface("polyregion.core:polyregion.core.transform_reduce_unordered")
  public static float transform_reduce_unordered(long n, Float init, FnJ_F map, FnFF_F combine) {
    return (Float) Launch.call("polyregion.core:polyregion.core.transform_reduce_unordered", null, new Object[] {n, init, map, combine});
  }

  @Interface("polyregion.core:polyregion.core.transform_reduce_unordered")
  public static float transform_reduce_unordered(
      polyregion.jvm.runtime.Context context,
      long n,
      Float init,
      FnJ_F map,
      FnFF_F combine) {
    return (Float) Launch.call("polyregion.core:polyregion.core.transform_reduce_unordered", context, new Object[] {n, init, map, combine});
  }

  @Interface("polyregion.core:polyregion.core.transform_reduce_unordered")
  public static double transform_reduce_unordered(long n, Double init, FnJ_D map, FnDD_D combine) {
    return (Double) Launch.call("polyregion.core:polyregion.core.transform_reduce_unordered", null, new Object[] {n, init, map, combine});
  }

  @Interface("polyregion.core:polyregion.core.transform_reduce_unordered")
  public static double transform_reduce_unordered(
      polyregion.jvm.runtime.Context context,
      long n,
      Double init,
      FnJ_D map,
      FnDD_D combine) {
    return (Double) Launch.call("polyregion.core:polyregion.core.transform_reduce_unordered", context, new Object[] {n, init, map, combine});
  }
}
