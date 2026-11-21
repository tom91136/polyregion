package example.bindings;

import polyregion.javalang.Interface;
import polyregion.javalang.Launch;

public final class ExampleInterface {

  private ExampleInterface() {}

  @FunctionalInterface
  public interface FnD_D extends java.io.Serializable {
    double apply(double arg0);
  }

  @FunctionalInterface
  public interface FnD_F extends java.io.Serializable {
    float apply(double arg0);
  }

  @FunctionalInterface
  public interface FnD_I extends java.io.Serializable {
    int apply(double arg0);
  }

  @FunctionalInterface
  public interface FnD_J extends java.io.Serializable {
    long apply(double arg0);
  }

  @FunctionalInterface
  public interface FnF_D extends java.io.Serializable {
    double apply(float arg0);
  }

  @FunctionalInterface
  public interface FnF_F extends java.io.Serializable {
    float apply(float arg0);
  }

  @FunctionalInterface
  public interface FnF_I extends java.io.Serializable {
    int apply(float arg0);
  }

  @FunctionalInterface
  public interface FnF_J extends java.io.Serializable {
    long apply(float arg0);
  }

  @FunctionalInterface
  public interface FnI_D extends java.io.Serializable {
    double apply(int arg0);
  }

  @FunctionalInterface
  public interface FnI_F extends java.io.Serializable {
    float apply(int arg0);
  }

  @FunctionalInterface
  public interface FnI_I extends java.io.Serializable {
    int apply(int arg0);
  }

  @FunctionalInterface
  public interface FnI_J extends java.io.Serializable {
    long apply(int arg0);
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
  public interface FnL_L<T0, R> extends java.io.Serializable {
    R apply(T0 arg0);
  }

  @Interface("example:example.count")
  public static int count(int[] in, int n) {
    return (Integer) Launch.call("example:example.count", null, new Object[] {in, n});
  }

  @Interface("example:example.count")
  public static int count(long[] in, int n) {
    return (Integer) Launch.call("example:example.count", null, new Object[] {in, n});
  }

  @Interface("example:example.count")
  public static int count(float[] in, int n) {
    return (Integer) Launch.call("example:example.count", null, new Object[] {in, n});
  }

  @Interface("example:example.count")
  public static int count(double[] in, int n) {
    return (Integer) Launch.call("example:example.count", null, new Object[] {in, n});
  }

  @Interface("example:example.count")
  public static <T> int count(T[] in, int n) {
    return (Integer) Launch.call("example:example.count", null, new Object[] {in, n});
  }

  @Interface("example:example.transform")
  public static void transform(int[] in, int[] out, int n, FnI_I op) {
    Launch.call("example:example.transform", null, new Object[] {in, out, n, op});
  }

  @Interface("example:example.transform")
  public static void transform(int[] in, long[] out, int n, FnI_J op) {
    Launch.call("example:example.transform", null, new Object[] {in, out, n, op});
  }

  @Interface("example:example.transform")
  public static void transform(int[] in, float[] out, int n, FnI_F op) {
    Launch.call("example:example.transform", null, new Object[] {in, out, n, op});
  }

  @Interface("example:example.transform")
  public static void transform(int[] in, double[] out, int n, FnI_D op) {
    Launch.call("example:example.transform", null, new Object[] {in, out, n, op});
  }

  @Interface("example:example.transform")
  public static void transform(long[] in, int[] out, int n, FnJ_I op) {
    Launch.call("example:example.transform", null, new Object[] {in, out, n, op});
  }

  @Interface("example:example.transform")
  public static void transform(long[] in, long[] out, int n, FnJ_J op) {
    Launch.call("example:example.transform", null, new Object[] {in, out, n, op});
  }

  @Interface("example:example.transform")
  public static void transform(long[] in, float[] out, int n, FnJ_F op) {
    Launch.call("example:example.transform", null, new Object[] {in, out, n, op});
  }

  @Interface("example:example.transform")
  public static void transform(long[] in, double[] out, int n, FnJ_D op) {
    Launch.call("example:example.transform", null, new Object[] {in, out, n, op});
  }

  @Interface("example:example.transform")
  public static void transform(float[] in, int[] out, int n, FnF_I op) {
    Launch.call("example:example.transform", null, new Object[] {in, out, n, op});
  }

  @Interface("example:example.transform")
  public static void transform(float[] in, long[] out, int n, FnF_J op) {
    Launch.call("example:example.transform", null, new Object[] {in, out, n, op});
  }

  @Interface("example:example.transform")
  public static void transform(float[] in, float[] out, int n, FnF_F op) {
    Launch.call("example:example.transform", null, new Object[] {in, out, n, op});
  }

  @Interface("example:example.transform")
  public static void transform(float[] in, double[] out, int n, FnF_D op) {
    Launch.call("example:example.transform", null, new Object[] {in, out, n, op});
  }

  @Interface("example:example.transform")
  public static void transform(double[] in, int[] out, int n, FnD_I op) {
    Launch.call("example:example.transform", null, new Object[] {in, out, n, op});
  }

  @Interface("example:example.transform")
  public static void transform(double[] in, long[] out, int n, FnD_J op) {
    Launch.call("example:example.transform", null, new Object[] {in, out, n, op});
  }

  @Interface("example:example.transform")
  public static void transform(double[] in, float[] out, int n, FnD_F op) {
    Launch.call("example:example.transform", null, new Object[] {in, out, n, op});
  }

  @Interface("example:example.transform")
  public static void transform(double[] in, double[] out, int n, FnD_D op) {
    Launch.call("example:example.transform", null, new Object[] {in, out, n, op});
  }

  @Interface("example:example.transform")
  public static <T, U> void transform(T[] in, U[] out, int n, FnL_L<T, U> op) {
    Launch.call("example:example.transform", null, new Object[] {in, out, n, op});
  }
}
