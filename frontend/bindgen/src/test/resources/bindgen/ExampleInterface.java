package example.bindings;

import polyregion.javalang.Interface;

public final class ExampleInterface {

  private ExampleInterface() {}

  @FunctionalInterface
  public interface FnD_D {
    double apply(double arg0);
  }

  @FunctionalInterface
  public interface FnD_F {
    float apply(double arg0);
  }

  @FunctionalInterface
  public interface FnD_I {
    int apply(double arg0);
  }

  @FunctionalInterface
  public interface FnD_J {
    long apply(double arg0);
  }

  @FunctionalInterface
  public interface FnF_D {
    double apply(float arg0);
  }

  @FunctionalInterface
  public interface FnF_F {
    float apply(float arg0);
  }

  @FunctionalInterface
  public interface FnF_I {
    int apply(float arg0);
  }

  @FunctionalInterface
  public interface FnF_J {
    long apply(float arg0);
  }

  @FunctionalInterface
  public interface FnI_D {
    double apply(int arg0);
  }

  @FunctionalInterface
  public interface FnI_F {
    float apply(int arg0);
  }

  @FunctionalInterface
  public interface FnI_I {
    int apply(int arg0);
  }

  @FunctionalInterface
  public interface FnI_J {
    long apply(int arg0);
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
  public interface FnL_L<T0, R> {
    R apply(T0 arg0);
  }

  @Interface("example:example.count")
  public static int count(int[] in, int n) {
    throw new UnsupportedOperationException("compiler did not replace example.count");
  }

  @Interface("example:example.count")
  public static int count(long[] in, int n) {
    throw new UnsupportedOperationException("compiler did not replace example.count");
  }

  @Interface("example:example.count")
  public static int count(float[] in, int n) {
    throw new UnsupportedOperationException("compiler did not replace example.count");
  }

  @Interface("example:example.count")
  public static int count(double[] in, int n) {
    throw new UnsupportedOperationException("compiler did not replace example.count");
  }

  @Interface("example:example.count")
  public static <T> int count(T[] in, int n) {
    throw new UnsupportedOperationException("compiler did not replace example.count");
  }

  @Interface("example:example.transform")
  public static void transform(int[] in, int[] out, int n, FnI_I op) {
    throw new UnsupportedOperationException("compiler did not replace example.transform");
  }

  @Interface("example:example.transform")
  public static void transform(int[] in, long[] out, int n, FnI_J op) {
    throw new UnsupportedOperationException("compiler did not replace example.transform");
  }

  @Interface("example:example.transform")
  public static void transform(int[] in, float[] out, int n, FnI_F op) {
    throw new UnsupportedOperationException("compiler did not replace example.transform");
  }

  @Interface("example:example.transform")
  public static void transform(int[] in, double[] out, int n, FnI_D op) {
    throw new UnsupportedOperationException("compiler did not replace example.transform");
  }

  @Interface("example:example.transform")
  public static void transform(long[] in, int[] out, int n, FnJ_I op) {
    throw new UnsupportedOperationException("compiler did not replace example.transform");
  }

  @Interface("example:example.transform")
  public static void transform(long[] in, long[] out, int n, FnJ_J op) {
    throw new UnsupportedOperationException("compiler did not replace example.transform");
  }

  @Interface("example:example.transform")
  public static void transform(long[] in, float[] out, int n, FnJ_F op) {
    throw new UnsupportedOperationException("compiler did not replace example.transform");
  }

  @Interface("example:example.transform")
  public static void transform(long[] in, double[] out, int n, FnJ_D op) {
    throw new UnsupportedOperationException("compiler did not replace example.transform");
  }

  @Interface("example:example.transform")
  public static void transform(float[] in, int[] out, int n, FnF_I op) {
    throw new UnsupportedOperationException("compiler did not replace example.transform");
  }

  @Interface("example:example.transform")
  public static void transform(float[] in, long[] out, int n, FnF_J op) {
    throw new UnsupportedOperationException("compiler did not replace example.transform");
  }

  @Interface("example:example.transform")
  public static void transform(float[] in, float[] out, int n, FnF_F op) {
    throw new UnsupportedOperationException("compiler did not replace example.transform");
  }

  @Interface("example:example.transform")
  public static void transform(float[] in, double[] out, int n, FnF_D op) {
    throw new UnsupportedOperationException("compiler did not replace example.transform");
  }

  @Interface("example:example.transform")
  public static void transform(double[] in, int[] out, int n, FnD_I op) {
    throw new UnsupportedOperationException("compiler did not replace example.transform");
  }

  @Interface("example:example.transform")
  public static void transform(double[] in, long[] out, int n, FnD_J op) {
    throw new UnsupportedOperationException("compiler did not replace example.transform");
  }

  @Interface("example:example.transform")
  public static void transform(double[] in, float[] out, int n, FnD_F op) {
    throw new UnsupportedOperationException("compiler did not replace example.transform");
  }

  @Interface("example:example.transform")
  public static void transform(double[] in, double[] out, int n, FnD_D op) {
    throw new UnsupportedOperationException("compiler did not replace example.transform");
  }

  @Interface("example:example.transform")
  public static <T, U> void transform(T[] in, U[] out, int n, FnL_L<T, U> op) {
    throw new UnsupportedOperationException("compiler did not replace example.transform");
  }
}
