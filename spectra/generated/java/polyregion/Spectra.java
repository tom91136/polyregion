package polyregion;

import polyregion.javalang.Interface;
import polyregion.javalang.Launch;

public final class Spectra {

  private Spectra() {}

  @FunctionalInterface
  public interface FnDDI_D extends java.io.Serializable {
    double apply(double arg0, double arg1, int arg2);
  }

  @FunctionalInterface
  public interface FnDD_D extends java.io.Serializable {
    double apply(double arg0, double arg1);
  }

  @FunctionalInterface
  public interface FnDD_F extends java.io.Serializable {
    float apply(double arg0, double arg1);
  }

  @FunctionalInterface
  public interface FnDD_I extends java.io.Serializable {
    int apply(double arg0, double arg1);
  }

  @FunctionalInterface
  public interface FnDD_J extends java.io.Serializable {
    long apply(double arg0, double arg1);
  }

  @FunctionalInterface
  public interface FnDD_Z extends java.io.Serializable {
    boolean apply(double arg0, double arg1);
  }

  @FunctionalInterface
  public interface FnDF_D extends java.io.Serializable {
    double apply(double arg0, float arg1);
  }

  @FunctionalInterface
  public interface FnDF_F extends java.io.Serializable {
    float apply(double arg0, float arg1);
  }

  @FunctionalInterface
  public interface FnDF_I extends java.io.Serializable {
    int apply(double arg0, float arg1);
  }

  @FunctionalInterface
  public interface FnDF_J extends java.io.Serializable {
    long apply(double arg0, float arg1);
  }

  @FunctionalInterface
  public interface FnDI_D extends java.io.Serializable {
    double apply(double arg0, int arg1);
  }

  @FunctionalInterface
  public interface FnDI_F extends java.io.Serializable {
    float apply(double arg0, int arg1);
  }

  @FunctionalInterface
  public interface FnDI_I extends java.io.Serializable {
    int apply(double arg0, int arg1);
  }

  @FunctionalInterface
  public interface FnDI_J extends java.io.Serializable {
    long apply(double arg0, int arg1);
  }

  @FunctionalInterface
  public interface FnDJ_D extends java.io.Serializable {
    double apply(double arg0, long arg1);
  }

  @FunctionalInterface
  public interface FnDJ_F extends java.io.Serializable {
    float apply(double arg0, long arg1);
  }

  @FunctionalInterface
  public interface FnDJ_I extends java.io.Serializable {
    int apply(double arg0, long arg1);
  }

  @FunctionalInterface
  public interface FnDJ_J extends java.io.Serializable {
    long apply(double arg0, long arg1);
  }

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
  public interface FnD_Z extends java.io.Serializable {
    boolean apply(double arg0);
  }

  @FunctionalInterface
  public interface FnFD_D extends java.io.Serializable {
    double apply(float arg0, double arg1);
  }

  @FunctionalInterface
  public interface FnFD_F extends java.io.Serializable {
    float apply(float arg0, double arg1);
  }

  @FunctionalInterface
  public interface FnFD_I extends java.io.Serializable {
    int apply(float arg0, double arg1);
  }

  @FunctionalInterface
  public interface FnFD_J extends java.io.Serializable {
    long apply(float arg0, double arg1);
  }

  @FunctionalInterface
  public interface FnFFI_F extends java.io.Serializable {
    float apply(float arg0, float arg1, int arg2);
  }

  @FunctionalInterface
  public interface FnFF_D extends java.io.Serializable {
    double apply(float arg0, float arg1);
  }

  @FunctionalInterface
  public interface FnFF_F extends java.io.Serializable {
    float apply(float arg0, float arg1);
  }

  @FunctionalInterface
  public interface FnFF_I extends java.io.Serializable {
    int apply(float arg0, float arg1);
  }

  @FunctionalInterface
  public interface FnFF_J extends java.io.Serializable {
    long apply(float arg0, float arg1);
  }

  @FunctionalInterface
  public interface FnFF_Z extends java.io.Serializable {
    boolean apply(float arg0, float arg1);
  }

  @FunctionalInterface
  public interface FnFI_D extends java.io.Serializable {
    double apply(float arg0, int arg1);
  }

  @FunctionalInterface
  public interface FnFI_F extends java.io.Serializable {
    float apply(float arg0, int arg1);
  }

  @FunctionalInterface
  public interface FnFI_I extends java.io.Serializable {
    int apply(float arg0, int arg1);
  }

  @FunctionalInterface
  public interface FnFI_J extends java.io.Serializable {
    long apply(float arg0, int arg1);
  }

  @FunctionalInterface
  public interface FnFJ_D extends java.io.Serializable {
    double apply(float arg0, long arg1);
  }

  @FunctionalInterface
  public interface FnFJ_F extends java.io.Serializable {
    float apply(float arg0, long arg1);
  }

  @FunctionalInterface
  public interface FnFJ_I extends java.io.Serializable {
    int apply(float arg0, long arg1);
  }

  @FunctionalInterface
  public interface FnFJ_J extends java.io.Serializable {
    long apply(float arg0, long arg1);
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
  public interface FnF_Z extends java.io.Serializable {
    boolean apply(float arg0);
  }

  @FunctionalInterface
  public interface FnID_D extends java.io.Serializable {
    double apply(int arg0, double arg1);
  }

  @FunctionalInterface
  public interface FnID_F extends java.io.Serializable {
    float apply(int arg0, double arg1);
  }

  @FunctionalInterface
  public interface FnID_I extends java.io.Serializable {
    int apply(int arg0, double arg1);
  }

  @FunctionalInterface
  public interface FnID_J extends java.io.Serializable {
    long apply(int arg0, double arg1);
  }

  @FunctionalInterface
  public interface FnIF_D extends java.io.Serializable {
    double apply(int arg0, float arg1);
  }

  @FunctionalInterface
  public interface FnIF_F extends java.io.Serializable {
    float apply(int arg0, float arg1);
  }

  @FunctionalInterface
  public interface FnIF_I extends java.io.Serializable {
    int apply(int arg0, float arg1);
  }

  @FunctionalInterface
  public interface FnIF_J extends java.io.Serializable {
    long apply(int arg0, float arg1);
  }

  @FunctionalInterface
  public interface FnIII_I extends java.io.Serializable {
    int apply(int arg0, int arg1, int arg2);
  }

  @FunctionalInterface
  public interface FnII_D extends java.io.Serializable {
    double apply(int arg0, int arg1);
  }

  @FunctionalInterface
  public interface FnII_F extends java.io.Serializable {
    float apply(int arg0, int arg1);
  }

  @FunctionalInterface
  public interface FnII_I extends java.io.Serializable {
    int apply(int arg0, int arg1);
  }

  @FunctionalInterface
  public interface FnII_J extends java.io.Serializable {
    long apply(int arg0, int arg1);
  }

  @FunctionalInterface
  public interface FnII_Z extends java.io.Serializable {
    boolean apply(int arg0, int arg1);
  }

  @FunctionalInterface
  public interface FnIJ_D extends java.io.Serializable {
    double apply(int arg0, long arg1);
  }

  @FunctionalInterface
  public interface FnIJ_F extends java.io.Serializable {
    float apply(int arg0, long arg1);
  }

  @FunctionalInterface
  public interface FnIJ_I extends java.io.Serializable {
    int apply(int arg0, long arg1);
  }

  @FunctionalInterface
  public interface FnIJ_J extends java.io.Serializable {
    long apply(int arg0, long arg1);
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
  public interface FnI_L<R> extends java.io.Serializable {
    R apply(int arg0);
  }

  @FunctionalInterface
  public interface FnI_Z extends java.io.Serializable {
    boolean apply(int arg0);
  }

  @FunctionalInterface
  public interface FnJD_D extends java.io.Serializable {
    double apply(long arg0, double arg1);
  }

  @FunctionalInterface
  public interface FnJD_F extends java.io.Serializable {
    float apply(long arg0, double arg1);
  }

  @FunctionalInterface
  public interface FnJD_I extends java.io.Serializable {
    int apply(long arg0, double arg1);
  }

  @FunctionalInterface
  public interface FnJD_J extends java.io.Serializable {
    long apply(long arg0, double arg1);
  }

  @FunctionalInterface
  public interface FnJF_D extends java.io.Serializable {
    double apply(long arg0, float arg1);
  }

  @FunctionalInterface
  public interface FnJF_F extends java.io.Serializable {
    float apply(long arg0, float arg1);
  }

  @FunctionalInterface
  public interface FnJF_I extends java.io.Serializable {
    int apply(long arg0, float arg1);
  }

  @FunctionalInterface
  public interface FnJF_J extends java.io.Serializable {
    long apply(long arg0, float arg1);
  }

  @FunctionalInterface
  public interface FnJI_D extends java.io.Serializable {
    double apply(long arg0, int arg1);
  }

  @FunctionalInterface
  public interface FnJI_F extends java.io.Serializable {
    float apply(long arg0, int arg1);
  }

  @FunctionalInterface
  public interface FnJI_I extends java.io.Serializable {
    int apply(long arg0, int arg1);
  }

  @FunctionalInterface
  public interface FnJI_J extends java.io.Serializable {
    long apply(long arg0, int arg1);
  }

  @FunctionalInterface
  public interface FnJJI_J extends java.io.Serializable {
    long apply(long arg0, long arg1, int arg2);
  }

  @FunctionalInterface
  public interface FnJJ_D extends java.io.Serializable {
    double apply(long arg0, long arg1);
  }

  @FunctionalInterface
  public interface FnJJ_F extends java.io.Serializable {
    float apply(long arg0, long arg1);
  }

  @FunctionalInterface
  public interface FnJJ_I extends java.io.Serializable {
    int apply(long arg0, long arg1);
  }

  @FunctionalInterface
  public interface FnJJ_J extends java.io.Serializable {
    long apply(long arg0, long arg1);
  }

  @FunctionalInterface
  public interface FnJJ_Z extends java.io.Serializable {
    boolean apply(long arg0, long arg1);
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
  public interface FnJ_Z extends java.io.Serializable {
    boolean apply(long arg0);
  }

  @FunctionalInterface
  public interface FnLLI_L<T0, T1, R> extends java.io.Serializable {
    R apply(T0 arg0, T1 arg1, int arg2);
  }

  @FunctionalInterface
  public interface FnLL_L<T0, T1, R> extends java.io.Serializable {
    R apply(T0 arg0, T1 arg1);
  }

  @FunctionalInterface
  public interface FnLL_Z<T0, T1> extends java.io.Serializable {
    boolean apply(T0 arg0, T1 arg1);
  }

  @FunctionalInterface
  public interface FnL_L<T0, R> extends java.io.Serializable {
    R apply(T0 arg0);
  }

  @FunctionalInterface
  public interface FnL_Z<T0> extends java.io.Serializable {
    boolean apply(T0 arg0);
  }

  @FunctionalInterface
  public interface Fn_D extends java.io.Serializable {
    double apply();
  }

  @FunctionalInterface
  public interface Fn_F extends java.io.Serializable {
    float apply();
  }

  @FunctionalInterface
  public interface Fn_I extends java.io.Serializable {
    int apply();
  }

  @FunctionalInterface
  public interface Fn_J extends java.io.Serializable {
    long apply();
  }

  @FunctionalInterface
  public interface Fn_L<R> extends java.io.Serializable {
    R apply();
  }

  @Interface("spectra:spectra.adjacent_difference")
  public static void adjacent_difference(int[] in, int[] out, int n, FnII_I op) {
    Launch.call("spectra:spectra.adjacent_difference", null, new Object[] {in, out, n, op});
  }

  @Interface("spectra:spectra.adjacent_difference")
  public static void adjacent_difference(long[] in, long[] out, int n, FnJJ_J op) {
    Launch.call("spectra:spectra.adjacent_difference", null, new Object[] {in, out, n, op});
  }

  @Interface("spectra:spectra.adjacent_difference")
  public static void adjacent_difference(float[] in, float[] out, int n, FnFF_F op) {
    Launch.call("spectra:spectra.adjacent_difference", null, new Object[] {in, out, n, op});
  }

  @Interface("spectra:spectra.adjacent_difference")
  public static void adjacent_difference(double[] in, double[] out, int n, FnDD_D op) {
    Launch.call("spectra:spectra.adjacent_difference", null, new Object[] {in, out, n, op});
  }

  @Interface("spectra:spectra.adjacent_difference")
  public static <T> void adjacent_difference(T[] in, T[] out, int n, FnLL_L<T, T, T> op) {
    Launch.call("spectra:spectra.adjacent_difference", null, new Object[] {in, out, n, op});
  }

  @Interface("spectra:spectra.all_of")
  public static boolean all_of(int[] in, int n, FnI_Z op) {
    return (Boolean) Launch.call("spectra:spectra.all_of", null, new Object[] {in, n, op});
  }

  @Interface("spectra:spectra.all_of")
  public static boolean all_of(long[] in, int n, FnJ_Z op) {
    return (Boolean) Launch.call("spectra:spectra.all_of", null, new Object[] {in, n, op});
  }

  @Interface("spectra:spectra.all_of")
  public static boolean all_of(float[] in, int n, FnF_Z op) {
    return (Boolean) Launch.call("spectra:spectra.all_of", null, new Object[] {in, n, op});
  }

  @Interface("spectra:spectra.all_of")
  public static boolean all_of(double[] in, int n, FnD_Z op) {
    return (Boolean) Launch.call("spectra:spectra.all_of", null, new Object[] {in, n, op});
  }

  @Interface("spectra:spectra.all_of")
  public static <T> boolean all_of(T[] in, int n, FnL_Z<T> op) {
    return (Boolean) Launch.call("spectra:spectra.all_of", null, new Object[] {in, n, op});
  }

  @Interface("spectra:spectra.any_of")
  public static boolean any_of(int[] in, int n, FnI_Z op) {
    return (Boolean) Launch.call("spectra:spectra.any_of", null, new Object[] {in, n, op});
  }

  @Interface("spectra:spectra.any_of")
  public static boolean any_of(long[] in, int n, FnJ_Z op) {
    return (Boolean) Launch.call("spectra:spectra.any_of", null, new Object[] {in, n, op});
  }

  @Interface("spectra:spectra.any_of")
  public static boolean any_of(float[] in, int n, FnF_Z op) {
    return (Boolean) Launch.call("spectra:spectra.any_of", null, new Object[] {in, n, op});
  }

  @Interface("spectra:spectra.any_of")
  public static boolean any_of(double[] in, int n, FnD_Z op) {
    return (Boolean) Launch.call("spectra:spectra.any_of", null, new Object[] {in, n, op});
  }

  @Interface("spectra:spectra.any_of")
  public static <T> boolean any_of(T[] in, int n, FnL_Z<T> op) {
    return (Boolean) Launch.call("spectra:spectra.any_of", null, new Object[] {in, n, op});
  }

  @Interface("spectra:spectra.binary_search")
  public static boolean binary_search(int[] in, int n, int value, FnII_Z op) {
    return (Boolean) Launch.call("spectra:spectra.binary_search", null, new Object[] {in, n, value, op});
  }

  @Interface("spectra:spectra.binary_search")
  public static boolean binary_search(long[] in, int n, long value, FnJJ_Z op) {
    return (Boolean) Launch.call("spectra:spectra.binary_search", null, new Object[] {in, n, value, op});
  }

  @Interface("spectra:spectra.binary_search")
  public static boolean binary_search(float[] in, int n, float value, FnFF_Z op) {
    return (Boolean) Launch.call("spectra:spectra.binary_search", null, new Object[] {in, n, value, op});
  }

  @Interface("spectra:spectra.binary_search")
  public static boolean binary_search(double[] in, int n, double value, FnDD_Z op) {
    return (Boolean) Launch.call("spectra:spectra.binary_search", null, new Object[] {in, n, value, op});
  }

  @Interface("spectra:spectra.binary_search")
  public static <T> boolean binary_search(T[] in, int n, T value, FnLL_Z<T, T> op) {
    return (Boolean) Launch.call("spectra:spectra.binary_search", null, new Object[] {in, n, value, op});
  }

  @Interface("spectra:spectra.copy")
  public static void copy(int[] in, int n, int[] out) {
    Launch.call("spectra:spectra.copy", null, new Object[] {in, n, out});
  }

  @Interface("spectra:spectra.copy")
  public static void copy(long[] in, int n, long[] out) {
    Launch.call("spectra:spectra.copy", null, new Object[] {in, n, out});
  }

  @Interface("spectra:spectra.copy")
  public static void copy(float[] in, int n, float[] out) {
    Launch.call("spectra:spectra.copy", null, new Object[] {in, n, out});
  }

  @Interface("spectra:spectra.copy")
  public static void copy(double[] in, int n, double[] out) {
    Launch.call("spectra:spectra.copy", null, new Object[] {in, n, out});
  }

  @Interface("spectra:spectra.copy")
  public static <T> void copy(T[] in, int n, T[] out) {
    Launch.call("spectra:spectra.copy", null, new Object[] {in, n, out});
  }

  @Interface("spectra:spectra.copy_if")
  public static int copy_if(int[] in, int n, int[] out, FnI_Z op) {
    return (Integer) Launch.call("spectra:spectra.copy_if", null, new Object[] {in, n, out, op});
  }

  @Interface("spectra:spectra.copy_if")
  public static int copy_if(long[] in, int n, long[] out, FnJ_Z op) {
    return (Integer) Launch.call("spectra:spectra.copy_if", null, new Object[] {in, n, out, op});
  }

  @Interface("spectra:spectra.copy_if")
  public static int copy_if(float[] in, int n, float[] out, FnF_Z op) {
    return (Integer) Launch.call("spectra:spectra.copy_if", null, new Object[] {in, n, out, op});
  }

  @Interface("spectra:spectra.copy_if")
  public static int copy_if(double[] in, int n, double[] out, FnD_Z op) {
    return (Integer) Launch.call("spectra:spectra.copy_if", null, new Object[] {in, n, out, op});
  }

  @Interface("spectra:spectra.copy_if")
  public static <T> int copy_if(T[] in, int n, T[] out, FnL_Z<T> op) {
    return (Integer) Launch.call("spectra:spectra.copy_if", null, new Object[] {in, n, out, op});
  }

  @Interface("spectra:spectra.copy_n")
  public static void copy_n(int[] in, int n, int[] out) {
    Launch.call("spectra:spectra.copy_n", null, new Object[] {in, n, out});
  }

  @Interface("spectra:spectra.copy_n")
  public static void copy_n(long[] in, int n, long[] out) {
    Launch.call("spectra:spectra.copy_n", null, new Object[] {in, n, out});
  }

  @Interface("spectra:spectra.copy_n")
  public static void copy_n(float[] in, int n, float[] out) {
    Launch.call("spectra:spectra.copy_n", null, new Object[] {in, n, out});
  }

  @Interface("spectra:spectra.copy_n")
  public static void copy_n(double[] in, int n, double[] out) {
    Launch.call("spectra:spectra.copy_n", null, new Object[] {in, n, out});
  }

  @Interface("spectra:spectra.copy_n")
  public static <T> void copy_n(T[] in, int n, T[] out) {
    Launch.call("spectra:spectra.copy_n", null, new Object[] {in, n, out});
  }

  @Interface("spectra:spectra.count")
  public static int count(int[] in, int n, int value, FnII_Z eq) {
    return (Integer) Launch.call("spectra:spectra.count", null, new Object[] {in, n, value, eq});
  }

  @Interface("spectra:spectra.count")
  public static int count(long[] in, int n, long value, FnJJ_Z eq) {
    return (Integer) Launch.call("spectra:spectra.count", null, new Object[] {in, n, value, eq});
  }

  @Interface("spectra:spectra.count")
  public static int count(float[] in, int n, float value, FnFF_Z eq) {
    return (Integer) Launch.call("spectra:spectra.count", null, new Object[] {in, n, value, eq});
  }

  @Interface("spectra:spectra.count")
  public static int count(double[] in, int n, double value, FnDD_Z eq) {
    return (Integer) Launch.call("spectra:spectra.count", null, new Object[] {in, n, value, eq});
  }

  @Interface("spectra:spectra.count")
  public static <T> int count(T[] in, int n, T value, FnLL_Z<T, T> eq) {
    return (Integer) Launch.call("spectra:spectra.count", null, new Object[] {in, n, value, eq});
  }

  @Interface("spectra:spectra.count_if")
  public static int count_if(int[] in, int n, FnI_Z op) {
    return (Integer) Launch.call("spectra:spectra.count_if", null, new Object[] {in, n, op});
  }

  @Interface("spectra:spectra.count_if")
  public static int count_if(long[] in, int n, FnJ_Z op) {
    return (Integer) Launch.call("spectra:spectra.count_if", null, new Object[] {in, n, op});
  }

  @Interface("spectra:spectra.count_if")
  public static int count_if(float[] in, int n, FnF_Z op) {
    return (Integer) Launch.call("spectra:spectra.count_if", null, new Object[] {in, n, op});
  }

  @Interface("spectra:spectra.count_if")
  public static int count_if(double[] in, int n, FnD_Z op) {
    return (Integer) Launch.call("spectra:spectra.count_if", null, new Object[] {in, n, op});
  }

  @Interface("spectra:spectra.count_if")
  public static <T> int count_if(T[] in, int n, FnL_Z<T> op) {
    return (Integer) Launch.call("spectra:spectra.count_if", null, new Object[] {in, n, op});
  }

  @Interface("spectra:spectra.equal")
  public static boolean equal(int[] a, int[] b, int n, FnII_Z op) {
    return (Boolean) Launch.call("spectra:spectra.equal", null, new Object[] {a, b, n, op});
  }

  @Interface("spectra:spectra.equal")
  public static boolean equal(long[] a, long[] b, int n, FnJJ_Z op) {
    return (Boolean) Launch.call("spectra:spectra.equal", null, new Object[] {a, b, n, op});
  }

  @Interface("spectra:spectra.equal")
  public static boolean equal(float[] a, float[] b, int n, FnFF_Z op) {
    return (Boolean) Launch.call("spectra:spectra.equal", null, new Object[] {a, b, n, op});
  }

  @Interface("spectra:spectra.equal")
  public static boolean equal(double[] a, double[] b, int n, FnDD_Z op) {
    return (Boolean) Launch.call("spectra:spectra.equal", null, new Object[] {a, b, n, op});
  }

  @Interface("spectra:spectra.equal")
  public static <T> boolean equal(T[] a, T[] b, int n, FnLL_Z<T, T> op) {
    return (Boolean) Launch.call("spectra:spectra.equal", null, new Object[] {a, b, n, op});
  }

  @Interface("spectra:spectra.exclusive_scan")
  public static void exclusive_scan(int[] in, int[] out, int n, int init, FnII_I op) {
    Launch.call("spectra:spectra.exclusive_scan", null, new Object[] {in, out, n, init, op});
  }

  @Interface("spectra:spectra.exclusive_scan")
  public static void exclusive_scan(long[] in, long[] out, int n, long init, FnJJ_J op) {
    Launch.call("spectra:spectra.exclusive_scan", null, new Object[] {in, out, n, init, op});
  }

  @Interface("spectra:spectra.exclusive_scan")
  public static void exclusive_scan(float[] in, float[] out, int n, float init, FnFF_F op) {
    Launch.call("spectra:spectra.exclusive_scan", null, new Object[] {in, out, n, init, op});
  }

  @Interface("spectra:spectra.exclusive_scan")
  public static void exclusive_scan(double[] in, double[] out, int n, double init, FnDD_D op) {
    Launch.call("spectra:spectra.exclusive_scan", null, new Object[] {in, out, n, init, op});
  }

  @Interface("spectra:spectra.exclusive_scan")
  public static <T> void exclusive_scan(T[] in, T[] out, int n, T init, FnLL_L<T, T, T> op) {
    Launch.call("spectra:spectra.exclusive_scan", null, new Object[] {in, out, n, init, op});
  }

  @Interface("spectra:spectra.exclusive_scan_by_key")
  public static void exclusive_scan_by_key(
      int[] keys,
      int[] vals,
      int[] out,
      int n,
      int init,
      FnII_Z eq,
      FnII_I op) {
    Launch.call("spectra:spectra.exclusive_scan_by_key", null, new Object[] {keys, vals, out, n, init, eq, op});
  }

  @Interface("spectra:spectra.exclusive_scan_by_key")
  public static void exclusive_scan_by_key(
      int[] keys,
      long[] vals,
      long[] out,
      int n,
      long init,
      FnII_Z eq,
      FnJJ_J op) {
    Launch.call("spectra:spectra.exclusive_scan_by_key", null, new Object[] {keys, vals, out, n, init, eq, op});
  }

  @Interface("spectra:spectra.exclusive_scan_by_key")
  public static void exclusive_scan_by_key(
      int[] keys,
      float[] vals,
      float[] out,
      int n,
      float init,
      FnII_Z eq,
      FnFF_F op) {
    Launch.call("spectra:spectra.exclusive_scan_by_key", null, new Object[] {keys, vals, out, n, init, eq, op});
  }

  @Interface("spectra:spectra.exclusive_scan_by_key")
  public static void exclusive_scan_by_key(
      int[] keys,
      double[] vals,
      double[] out,
      int n,
      double init,
      FnII_Z eq,
      FnDD_D op) {
    Launch.call("spectra:spectra.exclusive_scan_by_key", null, new Object[] {keys, vals, out, n, init, eq, op});
  }

  @Interface("spectra:spectra.exclusive_scan_by_key")
  public static void exclusive_scan_by_key(
      long[] keys,
      int[] vals,
      int[] out,
      int n,
      int init,
      FnJJ_Z eq,
      FnII_I op) {
    Launch.call("spectra:spectra.exclusive_scan_by_key", null, new Object[] {keys, vals, out, n, init, eq, op});
  }

  @Interface("spectra:spectra.exclusive_scan_by_key")
  public static void exclusive_scan_by_key(
      long[] keys,
      long[] vals,
      long[] out,
      int n,
      long init,
      FnJJ_Z eq,
      FnJJ_J op) {
    Launch.call("spectra:spectra.exclusive_scan_by_key", null, new Object[] {keys, vals, out, n, init, eq, op});
  }

  @Interface("spectra:spectra.exclusive_scan_by_key")
  public static void exclusive_scan_by_key(
      long[] keys,
      float[] vals,
      float[] out,
      int n,
      float init,
      FnJJ_Z eq,
      FnFF_F op) {
    Launch.call("spectra:spectra.exclusive_scan_by_key", null, new Object[] {keys, vals, out, n, init, eq, op});
  }

  @Interface("spectra:spectra.exclusive_scan_by_key")
  public static void exclusive_scan_by_key(
      long[] keys,
      double[] vals,
      double[] out,
      int n,
      double init,
      FnJJ_Z eq,
      FnDD_D op) {
    Launch.call("spectra:spectra.exclusive_scan_by_key", null, new Object[] {keys, vals, out, n, init, eq, op});
  }

  @Interface("spectra:spectra.exclusive_scan_by_key")
  public static void exclusive_scan_by_key(
      float[] keys,
      int[] vals,
      int[] out,
      int n,
      int init,
      FnFF_Z eq,
      FnII_I op) {
    Launch.call("spectra:spectra.exclusive_scan_by_key", null, new Object[] {keys, vals, out, n, init, eq, op});
  }

  @Interface("spectra:spectra.exclusive_scan_by_key")
  public static void exclusive_scan_by_key(
      float[] keys,
      long[] vals,
      long[] out,
      int n,
      long init,
      FnFF_Z eq,
      FnJJ_J op) {
    Launch.call("spectra:spectra.exclusive_scan_by_key", null, new Object[] {keys, vals, out, n, init, eq, op});
  }

  @Interface("spectra:spectra.exclusive_scan_by_key")
  public static void exclusive_scan_by_key(
      float[] keys,
      float[] vals,
      float[] out,
      int n,
      float init,
      FnFF_Z eq,
      FnFF_F op) {
    Launch.call("spectra:spectra.exclusive_scan_by_key", null, new Object[] {keys, vals, out, n, init, eq, op});
  }

  @Interface("spectra:spectra.exclusive_scan_by_key")
  public static void exclusive_scan_by_key(
      float[] keys,
      double[] vals,
      double[] out,
      int n,
      double init,
      FnFF_Z eq,
      FnDD_D op) {
    Launch.call("spectra:spectra.exclusive_scan_by_key", null, new Object[] {keys, vals, out, n, init, eq, op});
  }

  @Interface("spectra:spectra.exclusive_scan_by_key")
  public static void exclusive_scan_by_key(
      double[] keys,
      int[] vals,
      int[] out,
      int n,
      int init,
      FnDD_Z eq,
      FnII_I op) {
    Launch.call("spectra:spectra.exclusive_scan_by_key", null, new Object[] {keys, vals, out, n, init, eq, op});
  }

  @Interface("spectra:spectra.exclusive_scan_by_key")
  public static void exclusive_scan_by_key(
      double[] keys,
      long[] vals,
      long[] out,
      int n,
      long init,
      FnDD_Z eq,
      FnJJ_J op) {
    Launch.call("spectra:spectra.exclusive_scan_by_key", null, new Object[] {keys, vals, out, n, init, eq, op});
  }

  @Interface("spectra:spectra.exclusive_scan_by_key")
  public static void exclusive_scan_by_key(
      double[] keys,
      float[] vals,
      float[] out,
      int n,
      float init,
      FnDD_Z eq,
      FnFF_F op) {
    Launch.call("spectra:spectra.exclusive_scan_by_key", null, new Object[] {keys, vals, out, n, init, eq, op});
  }

  @Interface("spectra:spectra.exclusive_scan_by_key")
  public static void exclusive_scan_by_key(
      double[] keys,
      double[] vals,
      double[] out,
      int n,
      double init,
      FnDD_Z eq,
      FnDD_D op) {
    Launch.call("spectra:spectra.exclusive_scan_by_key", null, new Object[] {keys, vals, out, n, init, eq, op});
  }

  @Interface("spectra:spectra.exclusive_scan_by_key")
  public static <K, V> void exclusive_scan_by_key(
      K[] keys,
      V[] vals,
      V[] out,
      int n,
      V init,
      FnLL_Z<K, K> eq,
      FnLL_L<V, V, V> op) {
    Launch.call("spectra:spectra.exclusive_scan_by_key", null, new Object[] {keys, vals, out, n, init, eq, op});
  }

  @Interface("spectra:spectra.fill")
  public static void fill(int[] out, int n, int v) {
    Launch.call("spectra:spectra.fill", null, new Object[] {out, n, v});
  }

  @Interface("spectra:spectra.fill")
  public static void fill(long[] out, int n, long v) {
    Launch.call("spectra:spectra.fill", null, new Object[] {out, n, v});
  }

  @Interface("spectra:spectra.fill")
  public static void fill(float[] out, int n, float v) {
    Launch.call("spectra:spectra.fill", null, new Object[] {out, n, v});
  }

  @Interface("spectra:spectra.fill")
  public static void fill(double[] out, int n, double v) {
    Launch.call("spectra:spectra.fill", null, new Object[] {out, n, v});
  }

  @Interface("spectra:spectra.fill")
  public static <T> void fill(T[] out, int n, T v) {
    Launch.call("spectra:spectra.fill", null, new Object[] {out, n, v});
  }

  @Interface("spectra:spectra.fill_n")
  public static void fill_n(int[] out, int n, int v) {
    Launch.call("spectra:spectra.fill_n", null, new Object[] {out, n, v});
  }

  @Interface("spectra:spectra.fill_n")
  public static void fill_n(long[] out, int n, long v) {
    Launch.call("spectra:spectra.fill_n", null, new Object[] {out, n, v});
  }

  @Interface("spectra:spectra.fill_n")
  public static void fill_n(float[] out, int n, float v) {
    Launch.call("spectra:spectra.fill_n", null, new Object[] {out, n, v});
  }

  @Interface("spectra:spectra.fill_n")
  public static void fill_n(double[] out, int n, double v) {
    Launch.call("spectra:spectra.fill_n", null, new Object[] {out, n, v});
  }

  @Interface("spectra:spectra.fill_n")
  public static <T> void fill_n(T[] out, int n, T v) {
    Launch.call("spectra:spectra.fill_n", null, new Object[] {out, n, v});
  }

  @Interface("spectra:spectra.find")
  public static int find(int[] in, int n, int value, FnII_Z eq) {
    return (Integer) Launch.call("spectra:spectra.find", null, new Object[] {in, n, value, eq});
  }

  @Interface("spectra:spectra.find")
  public static int find(long[] in, int n, long value, FnJJ_Z eq) {
    return (Integer) Launch.call("spectra:spectra.find", null, new Object[] {in, n, value, eq});
  }

  @Interface("spectra:spectra.find")
  public static int find(float[] in, int n, float value, FnFF_Z eq) {
    return (Integer) Launch.call("spectra:spectra.find", null, new Object[] {in, n, value, eq});
  }

  @Interface("spectra:spectra.find")
  public static int find(double[] in, int n, double value, FnDD_Z eq) {
    return (Integer) Launch.call("spectra:spectra.find", null, new Object[] {in, n, value, eq});
  }

  @Interface("spectra:spectra.find")
  public static <T> int find(T[] in, int n, T value, FnLL_Z<T, T> eq) {
    return (Integer) Launch.call("spectra:spectra.find", null, new Object[] {in, n, value, eq});
  }

  @Interface("spectra:spectra.find_if")
  public static int find_if(int[] in, int n, FnI_Z op) {
    return (Integer) Launch.call("spectra:spectra.find_if", null, new Object[] {in, n, op});
  }

  @Interface("spectra:spectra.find_if")
  public static int find_if(long[] in, int n, FnJ_Z op) {
    return (Integer) Launch.call("spectra:spectra.find_if", null, new Object[] {in, n, op});
  }

  @Interface("spectra:spectra.find_if")
  public static int find_if(float[] in, int n, FnF_Z op) {
    return (Integer) Launch.call("spectra:spectra.find_if", null, new Object[] {in, n, op});
  }

  @Interface("spectra:spectra.find_if")
  public static int find_if(double[] in, int n, FnD_Z op) {
    return (Integer) Launch.call("spectra:spectra.find_if", null, new Object[] {in, n, op});
  }

  @Interface("spectra:spectra.find_if")
  public static <T> int find_if(T[] in, int n, FnL_Z<T> op) {
    return (Integer) Launch.call("spectra:spectra.find_if", null, new Object[] {in, n, op});
  }

  @Interface("spectra:spectra.find_if_not")
  public static int find_if_not(int[] in, int n, FnI_Z op) {
    return (Integer) Launch.call("spectra:spectra.find_if_not", null, new Object[] {in, n, op});
  }

  @Interface("spectra:spectra.find_if_not")
  public static int find_if_not(long[] in, int n, FnJ_Z op) {
    return (Integer) Launch.call("spectra:spectra.find_if_not", null, new Object[] {in, n, op});
  }

  @Interface("spectra:spectra.find_if_not")
  public static int find_if_not(float[] in, int n, FnF_Z op) {
    return (Integer) Launch.call("spectra:spectra.find_if_not", null, new Object[] {in, n, op});
  }

  @Interface("spectra:spectra.find_if_not")
  public static int find_if_not(double[] in, int n, FnD_Z op) {
    return (Integer) Launch.call("spectra:spectra.find_if_not", null, new Object[] {in, n, op});
  }

  @Interface("spectra:spectra.find_if_not")
  public static <T> int find_if_not(T[] in, int n, FnL_Z<T> op) {
    return (Integer) Launch.call("spectra:spectra.find_if_not", null, new Object[] {in, n, op});
  }

  @Interface("spectra:spectra.for_each")
  public static void for_each(int[] data, int n, FnI_I op) {
    Launch.call("spectra:spectra.for_each", null, new Object[] {data, n, op});
  }

  @Interface("spectra:spectra.for_each")
  public static void for_each(long[] data, int n, FnJ_J op) {
    Launch.call("spectra:spectra.for_each", null, new Object[] {data, n, op});
  }

  @Interface("spectra:spectra.for_each")
  public static void for_each(float[] data, int n, FnF_F op) {
    Launch.call("spectra:spectra.for_each", null, new Object[] {data, n, op});
  }

  @Interface("spectra:spectra.for_each")
  public static void for_each(double[] data, int n, FnD_D op) {
    Launch.call("spectra:spectra.for_each", null, new Object[] {data, n, op});
  }

  @Interface("spectra:spectra.for_each")
  public static <T> void for_each(T[] data, int n, FnL_L<T, T> op) {
    Launch.call("spectra:spectra.for_each", null, new Object[] {data, n, op});
  }

  @Interface("spectra:spectra.for_each_n")
  public static void for_each_n(int[] data, int n, FnI_I op) {
    Launch.call("spectra:spectra.for_each_n", null, new Object[] {data, n, op});
  }

  @Interface("spectra:spectra.for_each_n")
  public static void for_each_n(long[] data, int n, FnJ_J op) {
    Launch.call("spectra:spectra.for_each_n", null, new Object[] {data, n, op});
  }

  @Interface("spectra:spectra.for_each_n")
  public static void for_each_n(float[] data, int n, FnF_F op) {
    Launch.call("spectra:spectra.for_each_n", null, new Object[] {data, n, op});
  }

  @Interface("spectra:spectra.for_each_n")
  public static void for_each_n(double[] data, int n, FnD_D op) {
    Launch.call("spectra:spectra.for_each_n", null, new Object[] {data, n, op});
  }

  @Interface("spectra:spectra.for_each_n")
  public static <T> void for_each_n(T[] data, int n, FnL_L<T, T> op) {
    Launch.call("spectra:spectra.for_each_n", null, new Object[] {data, n, op});
  }

  @Interface("spectra:spectra.gather")
  public static void gather(int[] map, int n, int[] in, int in_n, int[] out) {
    Launch.call("spectra:spectra.gather", null, new Object[] {map, n, in, in_n, out});
  }

  @Interface("spectra:spectra.gather")
  public static void gather(int[] map, int n, long[] in, int in_n, long[] out) {
    Launch.call("spectra:spectra.gather", null, new Object[] {map, n, in, in_n, out});
  }

  @Interface("spectra:spectra.gather")
  public static void gather(int[] map, int n, float[] in, int in_n, float[] out) {
    Launch.call("spectra:spectra.gather", null, new Object[] {map, n, in, in_n, out});
  }

  @Interface("spectra:spectra.gather")
  public static void gather(int[] map, int n, double[] in, int in_n, double[] out) {
    Launch.call("spectra:spectra.gather", null, new Object[] {map, n, in, in_n, out});
  }

  @Interface("spectra:spectra.gather")
  public static <T> void gather(int[] map, int n, T[] in, int in_n, T[] out) {
    Launch.call("spectra:spectra.gather", null, new Object[] {map, n, in, in_n, out});
  }

  @Interface("spectra:spectra.generate")
  public static void generate(int[] out, int n, Fn_I op) {
    Launch.call("spectra:spectra.generate", null, new Object[] {out, n, op});
  }

  @Interface("spectra:spectra.generate")
  public static void generate(long[] out, int n, Fn_J op) {
    Launch.call("spectra:spectra.generate", null, new Object[] {out, n, op});
  }

  @Interface("spectra:spectra.generate")
  public static void generate(float[] out, int n, Fn_F op) {
    Launch.call("spectra:spectra.generate", null, new Object[] {out, n, op});
  }

  @Interface("spectra:spectra.generate")
  public static void generate(double[] out, int n, Fn_D op) {
    Launch.call("spectra:spectra.generate", null, new Object[] {out, n, op});
  }

  @Interface("spectra:spectra.generate")
  public static <T> void generate(T[] out, int n, Fn_L<T> op) {
    Launch.call("spectra:spectra.generate", null, new Object[] {out, n, op});
  }

  @Interface("spectra:spectra.generate_n")
  public static void generate_n(int[] out, int n, Fn_I op) {
    Launch.call("spectra:spectra.generate_n", null, new Object[] {out, n, op});
  }

  @Interface("spectra:spectra.generate_n")
  public static void generate_n(long[] out, int n, Fn_J op) {
    Launch.call("spectra:spectra.generate_n", null, new Object[] {out, n, op});
  }

  @Interface("spectra:spectra.generate_n")
  public static void generate_n(float[] out, int n, Fn_F op) {
    Launch.call("spectra:spectra.generate_n", null, new Object[] {out, n, op});
  }

  @Interface("spectra:spectra.generate_n")
  public static void generate_n(double[] out, int n, Fn_D op) {
    Launch.call("spectra:spectra.generate_n", null, new Object[] {out, n, op});
  }

  @Interface("spectra:spectra.generate_n")
  public static <T> void generate_n(T[] out, int n, Fn_L<T> op) {
    Launch.call("spectra:spectra.generate_n", null, new Object[] {out, n, op});
  }

  @Interface("spectra:spectra.includes")
  public static boolean includes(int[] a, int na, int[] b, int nb, FnII_Z op) {
    return (Boolean) Launch.call("spectra:spectra.includes", null, new Object[] {a, na, b, nb, op});
  }

  @Interface("spectra:spectra.includes")
  public static boolean includes(long[] a, int na, long[] b, int nb, FnJJ_Z op) {
    return (Boolean) Launch.call("spectra:spectra.includes", null, new Object[] {a, na, b, nb, op});
  }

  @Interface("spectra:spectra.includes")
  public static boolean includes(float[] a, int na, float[] b, int nb, FnFF_Z op) {
    return (Boolean) Launch.call("spectra:spectra.includes", null, new Object[] {a, na, b, nb, op});
  }

  @Interface("spectra:spectra.includes")
  public static boolean includes(double[] a, int na, double[] b, int nb, FnDD_Z op) {
    return (Boolean) Launch.call("spectra:spectra.includes", null, new Object[] {a, na, b, nb, op});
  }

  @Interface("spectra:spectra.includes")
  public static <T> boolean includes(T[] a, int na, T[] b, int nb, FnLL_Z<T, T> op) {
    return (Boolean) Launch.call("spectra:spectra.includes", null, new Object[] {a, na, b, nb, op});
  }

  @Interface("spectra:spectra.inclusive_scan")
  public static void inclusive_scan(int[] in, int[] out, int n, FnII_I op) {
    Launch.call("spectra:spectra.inclusive_scan", null, new Object[] {in, out, n, op});
  }

  @Interface("spectra:spectra.inclusive_scan")
  public static void inclusive_scan(long[] in, long[] out, int n, FnJJ_J op) {
    Launch.call("spectra:spectra.inclusive_scan", null, new Object[] {in, out, n, op});
  }

  @Interface("spectra:spectra.inclusive_scan")
  public static void inclusive_scan(float[] in, float[] out, int n, FnFF_F op) {
    Launch.call("spectra:spectra.inclusive_scan", null, new Object[] {in, out, n, op});
  }

  @Interface("spectra:spectra.inclusive_scan")
  public static void inclusive_scan(double[] in, double[] out, int n, FnDD_D op) {
    Launch.call("spectra:spectra.inclusive_scan", null, new Object[] {in, out, n, op});
  }

  @Interface("spectra:spectra.inclusive_scan")
  public static <T> void inclusive_scan(T[] in, T[] out, int n, FnLL_L<T, T, T> op) {
    Launch.call("spectra:spectra.inclusive_scan", null, new Object[] {in, out, n, op});
  }

  @Interface("spectra:spectra.inclusive_scan_by_key")
  public static void inclusive_scan_by_key(
      int[] keys,
      int[] vals,
      int[] out,
      int n,
      FnII_Z eq,
      FnII_I op) {
    Launch.call("spectra:spectra.inclusive_scan_by_key", null, new Object[] {keys, vals, out, n, eq, op});
  }

  @Interface("spectra:spectra.inclusive_scan_by_key")
  public static void inclusive_scan_by_key(
      int[] keys,
      long[] vals,
      long[] out,
      int n,
      FnII_Z eq,
      FnJJ_J op) {
    Launch.call("spectra:spectra.inclusive_scan_by_key", null, new Object[] {keys, vals, out, n, eq, op});
  }

  @Interface("spectra:spectra.inclusive_scan_by_key")
  public static void inclusive_scan_by_key(
      int[] keys,
      float[] vals,
      float[] out,
      int n,
      FnII_Z eq,
      FnFF_F op) {
    Launch.call("spectra:spectra.inclusive_scan_by_key", null, new Object[] {keys, vals, out, n, eq, op});
  }

  @Interface("spectra:spectra.inclusive_scan_by_key")
  public static void inclusive_scan_by_key(
      int[] keys,
      double[] vals,
      double[] out,
      int n,
      FnII_Z eq,
      FnDD_D op) {
    Launch.call("spectra:spectra.inclusive_scan_by_key", null, new Object[] {keys, vals, out, n, eq, op});
  }

  @Interface("spectra:spectra.inclusive_scan_by_key")
  public static void inclusive_scan_by_key(
      long[] keys,
      int[] vals,
      int[] out,
      int n,
      FnJJ_Z eq,
      FnII_I op) {
    Launch.call("spectra:spectra.inclusive_scan_by_key", null, new Object[] {keys, vals, out, n, eq, op});
  }

  @Interface("spectra:spectra.inclusive_scan_by_key")
  public static void inclusive_scan_by_key(
      long[] keys,
      long[] vals,
      long[] out,
      int n,
      FnJJ_Z eq,
      FnJJ_J op) {
    Launch.call("spectra:spectra.inclusive_scan_by_key", null, new Object[] {keys, vals, out, n, eq, op});
  }

  @Interface("spectra:spectra.inclusive_scan_by_key")
  public static void inclusive_scan_by_key(
      long[] keys,
      float[] vals,
      float[] out,
      int n,
      FnJJ_Z eq,
      FnFF_F op) {
    Launch.call("spectra:spectra.inclusive_scan_by_key", null, new Object[] {keys, vals, out, n, eq, op});
  }

  @Interface("spectra:spectra.inclusive_scan_by_key")
  public static void inclusive_scan_by_key(
      long[] keys,
      double[] vals,
      double[] out,
      int n,
      FnJJ_Z eq,
      FnDD_D op) {
    Launch.call("spectra:spectra.inclusive_scan_by_key", null, new Object[] {keys, vals, out, n, eq, op});
  }

  @Interface("spectra:spectra.inclusive_scan_by_key")
  public static void inclusive_scan_by_key(
      float[] keys,
      int[] vals,
      int[] out,
      int n,
      FnFF_Z eq,
      FnII_I op) {
    Launch.call("spectra:spectra.inclusive_scan_by_key", null, new Object[] {keys, vals, out, n, eq, op});
  }

  @Interface("spectra:spectra.inclusive_scan_by_key")
  public static void inclusive_scan_by_key(
      float[] keys,
      long[] vals,
      long[] out,
      int n,
      FnFF_Z eq,
      FnJJ_J op) {
    Launch.call("spectra:spectra.inclusive_scan_by_key", null, new Object[] {keys, vals, out, n, eq, op});
  }

  @Interface("spectra:spectra.inclusive_scan_by_key")
  public static void inclusive_scan_by_key(
      float[] keys,
      float[] vals,
      float[] out,
      int n,
      FnFF_Z eq,
      FnFF_F op) {
    Launch.call("spectra:spectra.inclusive_scan_by_key", null, new Object[] {keys, vals, out, n, eq, op});
  }

  @Interface("spectra:spectra.inclusive_scan_by_key")
  public static void inclusive_scan_by_key(
      float[] keys,
      double[] vals,
      double[] out,
      int n,
      FnFF_Z eq,
      FnDD_D op) {
    Launch.call("spectra:spectra.inclusive_scan_by_key", null, new Object[] {keys, vals, out, n, eq, op});
  }

  @Interface("spectra:spectra.inclusive_scan_by_key")
  public static void inclusive_scan_by_key(
      double[] keys,
      int[] vals,
      int[] out,
      int n,
      FnDD_Z eq,
      FnII_I op) {
    Launch.call("spectra:spectra.inclusive_scan_by_key", null, new Object[] {keys, vals, out, n, eq, op});
  }

  @Interface("spectra:spectra.inclusive_scan_by_key")
  public static void inclusive_scan_by_key(
      double[] keys,
      long[] vals,
      long[] out,
      int n,
      FnDD_Z eq,
      FnJJ_J op) {
    Launch.call("spectra:spectra.inclusive_scan_by_key", null, new Object[] {keys, vals, out, n, eq, op});
  }

  @Interface("spectra:spectra.inclusive_scan_by_key")
  public static void inclusive_scan_by_key(
      double[] keys,
      float[] vals,
      float[] out,
      int n,
      FnDD_Z eq,
      FnFF_F op) {
    Launch.call("spectra:spectra.inclusive_scan_by_key", null, new Object[] {keys, vals, out, n, eq, op});
  }

  @Interface("spectra:spectra.inclusive_scan_by_key")
  public static void inclusive_scan_by_key(
      double[] keys,
      double[] vals,
      double[] out,
      int n,
      FnDD_Z eq,
      FnDD_D op) {
    Launch.call("spectra:spectra.inclusive_scan_by_key", null, new Object[] {keys, vals, out, n, eq, op});
  }

  @Interface("spectra:spectra.inclusive_scan_by_key")
  public static <K, V> void inclusive_scan_by_key(
      K[] keys,
      V[] vals,
      V[] out,
      int n,
      FnLL_Z<K, K> eq,
      FnLL_L<V, V, V> op) {
    Launch.call("spectra:spectra.inclusive_scan_by_key", null, new Object[] {keys, vals, out, n, eq, op});
  }

  @Interface("spectra:spectra.inner_product")
  public static int inner_product(
      int[] a,
      int[] b,
      int n,
      Integer init,
      FnII_I op_reduce,
      FnII_I op_product) {
    return (Integer) Launch.call("spectra:spectra.inner_product", null, new Object[] {a, b, n, init, op_reduce, op_product});
  }

  @Interface("spectra:spectra.inner_product")
  public static long inner_product(
      int[] a,
      int[] b,
      int n,
      Long init,
      FnJJ_J op_reduce,
      FnII_J op_product) {
    return (Long) Launch.call("spectra:spectra.inner_product", null, new Object[] {a, b, n, init, op_reduce, op_product});
  }

  @Interface("spectra:spectra.inner_product")
  public static float inner_product(
      int[] a,
      int[] b,
      int n,
      Float init,
      FnFF_F op_reduce,
      FnII_F op_product) {
    return (Float) Launch.call("spectra:spectra.inner_product", null, new Object[] {a, b, n, init, op_reduce, op_product});
  }

  @Interface("spectra:spectra.inner_product")
  public static double inner_product(
      int[] a,
      int[] b,
      int n,
      Double init,
      FnDD_D op_reduce,
      FnII_D op_product) {
    return (Double) Launch.call("spectra:spectra.inner_product", null, new Object[] {a, b, n, init, op_reduce, op_product});
  }

  @Interface("spectra:spectra.inner_product")
  public static int inner_product(
      int[] a,
      long[] b,
      int n,
      Integer init,
      FnII_I op_reduce,
      FnIJ_I op_product) {
    return (Integer) Launch.call("spectra:spectra.inner_product", null, new Object[] {a, b, n, init, op_reduce, op_product});
  }

  @Interface("spectra:spectra.inner_product")
  public static long inner_product(
      int[] a,
      long[] b,
      int n,
      Long init,
      FnJJ_J op_reduce,
      FnIJ_J op_product) {
    return (Long) Launch.call("spectra:spectra.inner_product", null, new Object[] {a, b, n, init, op_reduce, op_product});
  }

  @Interface("spectra:spectra.inner_product")
  public static float inner_product(
      int[] a,
      long[] b,
      int n,
      Float init,
      FnFF_F op_reduce,
      FnIJ_F op_product) {
    return (Float) Launch.call("spectra:spectra.inner_product", null, new Object[] {a, b, n, init, op_reduce, op_product});
  }

  @Interface("spectra:spectra.inner_product")
  public static double inner_product(
      int[] a,
      long[] b,
      int n,
      Double init,
      FnDD_D op_reduce,
      FnIJ_D op_product) {
    return (Double) Launch.call("spectra:spectra.inner_product", null, new Object[] {a, b, n, init, op_reduce, op_product});
  }

  @Interface("spectra:spectra.inner_product")
  public static int inner_product(
      int[] a,
      float[] b,
      int n,
      Integer init,
      FnII_I op_reduce,
      FnIF_I op_product) {
    return (Integer) Launch.call("spectra:spectra.inner_product", null, new Object[] {a, b, n, init, op_reduce, op_product});
  }

  @Interface("spectra:spectra.inner_product")
  public static long inner_product(
      int[] a,
      float[] b,
      int n,
      Long init,
      FnJJ_J op_reduce,
      FnIF_J op_product) {
    return (Long) Launch.call("spectra:spectra.inner_product", null, new Object[] {a, b, n, init, op_reduce, op_product});
  }

  @Interface("spectra:spectra.inner_product")
  public static float inner_product(
      int[] a,
      float[] b,
      int n,
      Float init,
      FnFF_F op_reduce,
      FnIF_F op_product) {
    return (Float) Launch.call("spectra:spectra.inner_product", null, new Object[] {a, b, n, init, op_reduce, op_product});
  }

  @Interface("spectra:spectra.inner_product")
  public static double inner_product(
      int[] a,
      float[] b,
      int n,
      Double init,
      FnDD_D op_reduce,
      FnIF_D op_product) {
    return (Double) Launch.call("spectra:spectra.inner_product", null, new Object[] {a, b, n, init, op_reduce, op_product});
  }

  @Interface("spectra:spectra.inner_product")
  public static int inner_product(
      int[] a,
      double[] b,
      int n,
      Integer init,
      FnII_I op_reduce,
      FnID_I op_product) {
    return (Integer) Launch.call("spectra:spectra.inner_product", null, new Object[] {a, b, n, init, op_reduce, op_product});
  }

  @Interface("spectra:spectra.inner_product")
  public static long inner_product(
      int[] a,
      double[] b,
      int n,
      Long init,
      FnJJ_J op_reduce,
      FnID_J op_product) {
    return (Long) Launch.call("spectra:spectra.inner_product", null, new Object[] {a, b, n, init, op_reduce, op_product});
  }

  @Interface("spectra:spectra.inner_product")
  public static float inner_product(
      int[] a,
      double[] b,
      int n,
      Float init,
      FnFF_F op_reduce,
      FnID_F op_product) {
    return (Float) Launch.call("spectra:spectra.inner_product", null, new Object[] {a, b, n, init, op_reduce, op_product});
  }

  @Interface("spectra:spectra.inner_product")
  public static double inner_product(
      int[] a,
      double[] b,
      int n,
      Double init,
      FnDD_D op_reduce,
      FnID_D op_product) {
    return (Double) Launch.call("spectra:spectra.inner_product", null, new Object[] {a, b, n, init, op_reduce, op_product});
  }

  @Interface("spectra:spectra.inner_product")
  public static int inner_product(
      long[] a,
      int[] b,
      int n,
      Integer init,
      FnII_I op_reduce,
      FnJI_I op_product) {
    return (Integer) Launch.call("spectra:spectra.inner_product", null, new Object[] {a, b, n, init, op_reduce, op_product});
  }

  @Interface("spectra:spectra.inner_product")
  public static long inner_product(
      long[] a,
      int[] b,
      int n,
      Long init,
      FnJJ_J op_reduce,
      FnJI_J op_product) {
    return (Long) Launch.call("spectra:spectra.inner_product", null, new Object[] {a, b, n, init, op_reduce, op_product});
  }

  @Interface("spectra:spectra.inner_product")
  public static float inner_product(
      long[] a,
      int[] b,
      int n,
      Float init,
      FnFF_F op_reduce,
      FnJI_F op_product) {
    return (Float) Launch.call("spectra:spectra.inner_product", null, new Object[] {a, b, n, init, op_reduce, op_product});
  }

  @Interface("spectra:spectra.inner_product")
  public static double inner_product(
      long[] a,
      int[] b,
      int n,
      Double init,
      FnDD_D op_reduce,
      FnJI_D op_product) {
    return (Double) Launch.call("spectra:spectra.inner_product", null, new Object[] {a, b, n, init, op_reduce, op_product});
  }

  @Interface("spectra:spectra.inner_product")
  public static int inner_product(
      long[] a,
      long[] b,
      int n,
      Integer init,
      FnII_I op_reduce,
      FnJJ_I op_product) {
    return (Integer) Launch.call("spectra:spectra.inner_product", null, new Object[] {a, b, n, init, op_reduce, op_product});
  }

  @Interface("spectra:spectra.inner_product")
  public static long inner_product(
      long[] a,
      long[] b,
      int n,
      Long init,
      FnJJ_J op_reduce,
      FnJJ_J op_product) {
    return (Long) Launch.call("spectra:spectra.inner_product", null, new Object[] {a, b, n, init, op_reduce, op_product});
  }

  @Interface("spectra:spectra.inner_product")
  public static float inner_product(
      long[] a,
      long[] b,
      int n,
      Float init,
      FnFF_F op_reduce,
      FnJJ_F op_product) {
    return (Float) Launch.call("spectra:spectra.inner_product", null, new Object[] {a, b, n, init, op_reduce, op_product});
  }

  @Interface("spectra:spectra.inner_product")
  public static double inner_product(
      long[] a,
      long[] b,
      int n,
      Double init,
      FnDD_D op_reduce,
      FnJJ_D op_product) {
    return (Double) Launch.call("spectra:spectra.inner_product", null, new Object[] {a, b, n, init, op_reduce, op_product});
  }

  @Interface("spectra:spectra.inner_product")
  public static int inner_product(
      long[] a,
      float[] b,
      int n,
      Integer init,
      FnII_I op_reduce,
      FnJF_I op_product) {
    return (Integer) Launch.call("spectra:spectra.inner_product", null, new Object[] {a, b, n, init, op_reduce, op_product});
  }

  @Interface("spectra:spectra.inner_product")
  public static long inner_product(
      long[] a,
      float[] b,
      int n,
      Long init,
      FnJJ_J op_reduce,
      FnJF_J op_product) {
    return (Long) Launch.call("spectra:spectra.inner_product", null, new Object[] {a, b, n, init, op_reduce, op_product});
  }

  @Interface("spectra:spectra.inner_product")
  public static float inner_product(
      long[] a,
      float[] b,
      int n,
      Float init,
      FnFF_F op_reduce,
      FnJF_F op_product) {
    return (Float) Launch.call("spectra:spectra.inner_product", null, new Object[] {a, b, n, init, op_reduce, op_product});
  }

  @Interface("spectra:spectra.inner_product")
  public static double inner_product(
      long[] a,
      float[] b,
      int n,
      Double init,
      FnDD_D op_reduce,
      FnJF_D op_product) {
    return (Double) Launch.call("spectra:spectra.inner_product", null, new Object[] {a, b, n, init, op_reduce, op_product});
  }

  @Interface("spectra:spectra.inner_product")
  public static int inner_product(
      long[] a,
      double[] b,
      int n,
      Integer init,
      FnII_I op_reduce,
      FnJD_I op_product) {
    return (Integer) Launch.call("spectra:spectra.inner_product", null, new Object[] {a, b, n, init, op_reduce, op_product});
  }

  @Interface("spectra:spectra.inner_product")
  public static long inner_product(
      long[] a,
      double[] b,
      int n,
      Long init,
      FnJJ_J op_reduce,
      FnJD_J op_product) {
    return (Long) Launch.call("spectra:spectra.inner_product", null, new Object[] {a, b, n, init, op_reduce, op_product});
  }

  @Interface("spectra:spectra.inner_product")
  public static float inner_product(
      long[] a,
      double[] b,
      int n,
      Float init,
      FnFF_F op_reduce,
      FnJD_F op_product) {
    return (Float) Launch.call("spectra:spectra.inner_product", null, new Object[] {a, b, n, init, op_reduce, op_product});
  }

  @Interface("spectra:spectra.inner_product")
  public static double inner_product(
      long[] a,
      double[] b,
      int n,
      Double init,
      FnDD_D op_reduce,
      FnJD_D op_product) {
    return (Double) Launch.call("spectra:spectra.inner_product", null, new Object[] {a, b, n, init, op_reduce, op_product});
  }

  @Interface("spectra:spectra.inner_product")
  public static int inner_product(
      float[] a,
      int[] b,
      int n,
      Integer init,
      FnII_I op_reduce,
      FnFI_I op_product) {
    return (Integer) Launch.call("spectra:spectra.inner_product", null, new Object[] {a, b, n, init, op_reduce, op_product});
  }

  @Interface("spectra:spectra.inner_product")
  public static long inner_product(
      float[] a,
      int[] b,
      int n,
      Long init,
      FnJJ_J op_reduce,
      FnFI_J op_product) {
    return (Long) Launch.call("spectra:spectra.inner_product", null, new Object[] {a, b, n, init, op_reduce, op_product});
  }

  @Interface("spectra:spectra.inner_product")
  public static float inner_product(
      float[] a,
      int[] b,
      int n,
      Float init,
      FnFF_F op_reduce,
      FnFI_F op_product) {
    return (Float) Launch.call("spectra:spectra.inner_product", null, new Object[] {a, b, n, init, op_reduce, op_product});
  }

  @Interface("spectra:spectra.inner_product")
  public static double inner_product(
      float[] a,
      int[] b,
      int n,
      Double init,
      FnDD_D op_reduce,
      FnFI_D op_product) {
    return (Double) Launch.call("spectra:spectra.inner_product", null, new Object[] {a, b, n, init, op_reduce, op_product});
  }

  @Interface("spectra:spectra.inner_product")
  public static int inner_product(
      float[] a,
      long[] b,
      int n,
      Integer init,
      FnII_I op_reduce,
      FnFJ_I op_product) {
    return (Integer) Launch.call("spectra:spectra.inner_product", null, new Object[] {a, b, n, init, op_reduce, op_product});
  }

  @Interface("spectra:spectra.inner_product")
  public static long inner_product(
      float[] a,
      long[] b,
      int n,
      Long init,
      FnJJ_J op_reduce,
      FnFJ_J op_product) {
    return (Long) Launch.call("spectra:spectra.inner_product", null, new Object[] {a, b, n, init, op_reduce, op_product});
  }

  @Interface("spectra:spectra.inner_product")
  public static float inner_product(
      float[] a,
      long[] b,
      int n,
      Float init,
      FnFF_F op_reduce,
      FnFJ_F op_product) {
    return (Float) Launch.call("spectra:spectra.inner_product", null, new Object[] {a, b, n, init, op_reduce, op_product});
  }

  @Interface("spectra:spectra.inner_product")
  public static double inner_product(
      float[] a,
      long[] b,
      int n,
      Double init,
      FnDD_D op_reduce,
      FnFJ_D op_product) {
    return (Double) Launch.call("spectra:spectra.inner_product", null, new Object[] {a, b, n, init, op_reduce, op_product});
  }

  @Interface("spectra:spectra.inner_product")
  public static int inner_product(
      float[] a,
      float[] b,
      int n,
      Integer init,
      FnII_I op_reduce,
      FnFF_I op_product) {
    return (Integer) Launch.call("spectra:spectra.inner_product", null, new Object[] {a, b, n, init, op_reduce, op_product});
  }

  @Interface("spectra:spectra.inner_product")
  public static long inner_product(
      float[] a,
      float[] b,
      int n,
      Long init,
      FnJJ_J op_reduce,
      FnFF_J op_product) {
    return (Long) Launch.call("spectra:spectra.inner_product", null, new Object[] {a, b, n, init, op_reduce, op_product});
  }

  @Interface("spectra:spectra.inner_product")
  public static float inner_product(
      float[] a,
      float[] b,
      int n,
      Float init,
      FnFF_F op_reduce,
      FnFF_F op_product) {
    return (Float) Launch.call("spectra:spectra.inner_product", null, new Object[] {a, b, n, init, op_reduce, op_product});
  }

  @Interface("spectra:spectra.inner_product")
  public static double inner_product(
      float[] a,
      float[] b,
      int n,
      Double init,
      FnDD_D op_reduce,
      FnFF_D op_product) {
    return (Double) Launch.call("spectra:spectra.inner_product", null, new Object[] {a, b, n, init, op_reduce, op_product});
  }

  @Interface("spectra:spectra.inner_product")
  public static int inner_product(
      float[] a,
      double[] b,
      int n,
      Integer init,
      FnII_I op_reduce,
      FnFD_I op_product) {
    return (Integer) Launch.call("spectra:spectra.inner_product", null, new Object[] {a, b, n, init, op_reduce, op_product});
  }

  @Interface("spectra:spectra.inner_product")
  public static long inner_product(
      float[] a,
      double[] b,
      int n,
      Long init,
      FnJJ_J op_reduce,
      FnFD_J op_product) {
    return (Long) Launch.call("spectra:spectra.inner_product", null, new Object[] {a, b, n, init, op_reduce, op_product});
  }

  @Interface("spectra:spectra.inner_product")
  public static float inner_product(
      float[] a,
      double[] b,
      int n,
      Float init,
      FnFF_F op_reduce,
      FnFD_F op_product) {
    return (Float) Launch.call("spectra:spectra.inner_product", null, new Object[] {a, b, n, init, op_reduce, op_product});
  }

  @Interface("spectra:spectra.inner_product")
  public static double inner_product(
      float[] a,
      double[] b,
      int n,
      Double init,
      FnDD_D op_reduce,
      FnFD_D op_product) {
    return (Double) Launch.call("spectra:spectra.inner_product", null, new Object[] {a, b, n, init, op_reduce, op_product});
  }

  @Interface("spectra:spectra.inner_product")
  public static int inner_product(
      double[] a,
      int[] b,
      int n,
      Integer init,
      FnII_I op_reduce,
      FnDI_I op_product) {
    return (Integer) Launch.call("spectra:spectra.inner_product", null, new Object[] {a, b, n, init, op_reduce, op_product});
  }

  @Interface("spectra:spectra.inner_product")
  public static long inner_product(
      double[] a,
      int[] b,
      int n,
      Long init,
      FnJJ_J op_reduce,
      FnDI_J op_product) {
    return (Long) Launch.call("spectra:spectra.inner_product", null, new Object[] {a, b, n, init, op_reduce, op_product});
  }

  @Interface("spectra:spectra.inner_product")
  public static float inner_product(
      double[] a,
      int[] b,
      int n,
      Float init,
      FnFF_F op_reduce,
      FnDI_F op_product) {
    return (Float) Launch.call("spectra:spectra.inner_product", null, new Object[] {a, b, n, init, op_reduce, op_product});
  }

  @Interface("spectra:spectra.inner_product")
  public static double inner_product(
      double[] a,
      int[] b,
      int n,
      Double init,
      FnDD_D op_reduce,
      FnDI_D op_product) {
    return (Double) Launch.call("spectra:spectra.inner_product", null, new Object[] {a, b, n, init, op_reduce, op_product});
  }

  @Interface("spectra:spectra.inner_product")
  public static int inner_product(
      double[] a,
      long[] b,
      int n,
      Integer init,
      FnII_I op_reduce,
      FnDJ_I op_product) {
    return (Integer) Launch.call("spectra:spectra.inner_product", null, new Object[] {a, b, n, init, op_reduce, op_product});
  }

  @Interface("spectra:spectra.inner_product")
  public static long inner_product(
      double[] a,
      long[] b,
      int n,
      Long init,
      FnJJ_J op_reduce,
      FnDJ_J op_product) {
    return (Long) Launch.call("spectra:spectra.inner_product", null, new Object[] {a, b, n, init, op_reduce, op_product});
  }

  @Interface("spectra:spectra.inner_product")
  public static float inner_product(
      double[] a,
      long[] b,
      int n,
      Float init,
      FnFF_F op_reduce,
      FnDJ_F op_product) {
    return (Float) Launch.call("spectra:spectra.inner_product", null, new Object[] {a, b, n, init, op_reduce, op_product});
  }

  @Interface("spectra:spectra.inner_product")
  public static double inner_product(
      double[] a,
      long[] b,
      int n,
      Double init,
      FnDD_D op_reduce,
      FnDJ_D op_product) {
    return (Double) Launch.call("spectra:spectra.inner_product", null, new Object[] {a, b, n, init, op_reduce, op_product});
  }

  @Interface("spectra:spectra.inner_product")
  public static int inner_product(
      double[] a,
      float[] b,
      int n,
      Integer init,
      FnII_I op_reduce,
      FnDF_I op_product) {
    return (Integer) Launch.call("spectra:spectra.inner_product", null, new Object[] {a, b, n, init, op_reduce, op_product});
  }

  @Interface("spectra:spectra.inner_product")
  public static long inner_product(
      double[] a,
      float[] b,
      int n,
      Long init,
      FnJJ_J op_reduce,
      FnDF_J op_product) {
    return (Long) Launch.call("spectra:spectra.inner_product", null, new Object[] {a, b, n, init, op_reduce, op_product});
  }

  @Interface("spectra:spectra.inner_product")
  public static float inner_product(
      double[] a,
      float[] b,
      int n,
      Float init,
      FnFF_F op_reduce,
      FnDF_F op_product) {
    return (Float) Launch.call("spectra:spectra.inner_product", null, new Object[] {a, b, n, init, op_reduce, op_product});
  }

  @Interface("spectra:spectra.inner_product")
  public static double inner_product(
      double[] a,
      float[] b,
      int n,
      Double init,
      FnDD_D op_reduce,
      FnDF_D op_product) {
    return (Double) Launch.call("spectra:spectra.inner_product", null, new Object[] {a, b, n, init, op_reduce, op_product});
  }

  @Interface("spectra:spectra.inner_product")
  public static int inner_product(
      double[] a,
      double[] b,
      int n,
      Integer init,
      FnII_I op_reduce,
      FnDD_I op_product) {
    return (Integer) Launch.call("spectra:spectra.inner_product", null, new Object[] {a, b, n, init, op_reduce, op_product});
  }

  @Interface("spectra:spectra.inner_product")
  public static long inner_product(
      double[] a,
      double[] b,
      int n,
      Long init,
      FnJJ_J op_reduce,
      FnDD_J op_product) {
    return (Long) Launch.call("spectra:spectra.inner_product", null, new Object[] {a, b, n, init, op_reduce, op_product});
  }

  @Interface("spectra:spectra.inner_product")
  public static float inner_product(
      double[] a,
      double[] b,
      int n,
      Float init,
      FnFF_F op_reduce,
      FnDD_F op_product) {
    return (Float) Launch.call("spectra:spectra.inner_product", null, new Object[] {a, b, n, init, op_reduce, op_product});
  }

  @Interface("spectra:spectra.inner_product")
  public static double inner_product(
      double[] a,
      double[] b,
      int n,
      Double init,
      FnDD_D op_reduce,
      FnDD_D op_product) {
    return (Double) Launch.call("spectra:spectra.inner_product", null, new Object[] {a, b, n, init, op_reduce, op_product});
  }

  @Interface("spectra:spectra.inner_product")
  @SuppressWarnings("unchecked")
  public static <T, U, V> V inner_product(
      T[] a,
      U[] b,
      int n,
      V init,
      FnLL_L<V, V, V> op_reduce,
      FnLL_L<T, U, V> op_product) {
    return (V) Launch.call("spectra:spectra.inner_product", null, new Object[] {a, b, n, init, op_reduce, op_product});
  }

  @Interface("spectra:spectra.is_partitioned")
  public static boolean is_partitioned(int[] data, int n, FnI_Z op) {
    return (Boolean) Launch.call("spectra:spectra.is_partitioned", null, new Object[] {data, n, op});
  }

  @Interface("spectra:spectra.is_partitioned")
  public static boolean is_partitioned(long[] data, int n, FnJ_Z op) {
    return (Boolean) Launch.call("spectra:spectra.is_partitioned", null, new Object[] {data, n, op});
  }

  @Interface("spectra:spectra.is_partitioned")
  public static boolean is_partitioned(float[] data, int n, FnF_Z op) {
    return (Boolean) Launch.call("spectra:spectra.is_partitioned", null, new Object[] {data, n, op});
  }

  @Interface("spectra:spectra.is_partitioned")
  public static boolean is_partitioned(double[] data, int n, FnD_Z op) {
    return (Boolean) Launch.call("spectra:spectra.is_partitioned", null, new Object[] {data, n, op});
  }

  @Interface("spectra:spectra.is_partitioned")
  public static <T> boolean is_partitioned(T[] data, int n, FnL_Z<T> op) {
    return (Boolean) Launch.call("spectra:spectra.is_partitioned", null, new Object[] {data, n, op});
  }

  @Interface("spectra:spectra.is_sorted")
  public static boolean is_sorted(int[] data, int n, FnII_Z op) {
    return (Boolean) Launch.call("spectra:spectra.is_sorted", null, new Object[] {data, n, op});
  }

  @Interface("spectra:spectra.is_sorted")
  public static boolean is_sorted(long[] data, int n, FnJJ_Z op) {
    return (Boolean) Launch.call("spectra:spectra.is_sorted", null, new Object[] {data, n, op});
  }

  @Interface("spectra:spectra.is_sorted")
  public static boolean is_sorted(float[] data, int n, FnFF_Z op) {
    return (Boolean) Launch.call("spectra:spectra.is_sorted", null, new Object[] {data, n, op});
  }

  @Interface("spectra:spectra.is_sorted")
  public static boolean is_sorted(double[] data, int n, FnDD_Z op) {
    return (Boolean) Launch.call("spectra:spectra.is_sorted", null, new Object[] {data, n, op});
  }

  @Interface("spectra:spectra.is_sorted")
  public static <T> boolean is_sorted(T[] data, int n, FnLL_Z<T, T> op) {
    return (Boolean) Launch.call("spectra:spectra.is_sorted", null, new Object[] {data, n, op});
  }

  @Interface("spectra:spectra.is_sorted_until")
  public static int is_sorted_until(int[] data, int n, FnII_Z op) {
    return (Integer) Launch.call("spectra:spectra.is_sorted_until", null, new Object[] {data, n, op});
  }

  @Interface("spectra:spectra.is_sorted_until")
  public static int is_sorted_until(long[] data, int n, FnJJ_Z op) {
    return (Integer) Launch.call("spectra:spectra.is_sorted_until", null, new Object[] {data, n, op});
  }

  @Interface("spectra:spectra.is_sorted_until")
  public static int is_sorted_until(float[] data, int n, FnFF_Z op) {
    return (Integer) Launch.call("spectra:spectra.is_sorted_until", null, new Object[] {data, n, op});
  }

  @Interface("spectra:spectra.is_sorted_until")
  public static int is_sorted_until(double[] data, int n, FnDD_Z op) {
    return (Integer) Launch.call("spectra:spectra.is_sorted_until", null, new Object[] {data, n, op});
  }

  @Interface("spectra:spectra.is_sorted_until")
  public static <T> int is_sorted_until(T[] data, int n, FnLL_Z<T, T> op) {
    return (Integer) Launch.call("spectra:spectra.is_sorted_until", null, new Object[] {data, n, op});
  }

  @Interface("spectra:spectra.lower_bound")
  public static int lower_bound(int[] in, int n, int value, FnII_Z op) {
    return (Integer) Launch.call("spectra:spectra.lower_bound", null, new Object[] {in, n, value, op});
  }

  @Interface("spectra:spectra.lower_bound")
  public static int lower_bound(long[] in, int n, long value, FnJJ_Z op) {
    return (Integer) Launch.call("spectra:spectra.lower_bound", null, new Object[] {in, n, value, op});
  }

  @Interface("spectra:spectra.lower_bound")
  public static int lower_bound(float[] in, int n, float value, FnFF_Z op) {
    return (Integer) Launch.call("spectra:spectra.lower_bound", null, new Object[] {in, n, value, op});
  }

  @Interface("spectra:spectra.lower_bound")
  public static int lower_bound(double[] in, int n, double value, FnDD_Z op) {
    return (Integer) Launch.call("spectra:spectra.lower_bound", null, new Object[] {in, n, value, op});
  }

  @Interface("spectra:spectra.lower_bound")
  public static <T> int lower_bound(T[] in, int n, T value, FnLL_Z<T, T> op) {
    return (Integer) Launch.call("spectra:spectra.lower_bound", null, new Object[] {in, n, value, op});
  }

  @Interface("spectra:spectra.max_element")
  public static int max_element(int[] in, int n, FnII_Z op) {
    return (Integer) Launch.call("spectra:spectra.max_element", null, new Object[] {in, n, op});
  }

  @Interface("spectra:spectra.max_element")
  public static int max_element(long[] in, int n, FnJJ_Z op) {
    return (Integer) Launch.call("spectra:spectra.max_element", null, new Object[] {in, n, op});
  }

  @Interface("spectra:spectra.max_element")
  public static int max_element(float[] in, int n, FnFF_Z op) {
    return (Integer) Launch.call("spectra:spectra.max_element", null, new Object[] {in, n, op});
  }

  @Interface("spectra:spectra.max_element")
  public static int max_element(double[] in, int n, FnDD_Z op) {
    return (Integer) Launch.call("spectra:spectra.max_element", null, new Object[] {in, n, op});
  }

  @Interface("spectra:spectra.max_element")
  public static <T> int max_element(T[] in, int n, FnLL_Z<T, T> op) {
    return (Integer) Launch.call("spectra:spectra.max_element", null, new Object[] {in, n, op});
  }

  @Interface("spectra:spectra.merge")
  public static void merge(int[] a, int na, int[] b, int nb, int[] out, FnII_Z op) {
    Launch.call("spectra:spectra.merge", null, new Object[] {a, na, b, nb, out, op});
  }

  @Interface("spectra:spectra.merge")
  public static void merge(long[] a, int na, long[] b, int nb, long[] out, FnJJ_Z op) {
    Launch.call("spectra:spectra.merge", null, new Object[] {a, na, b, nb, out, op});
  }

  @Interface("spectra:spectra.merge")
  public static void merge(float[] a, int na, float[] b, int nb, float[] out, FnFF_Z op) {
    Launch.call("spectra:spectra.merge", null, new Object[] {a, na, b, nb, out, op});
  }

  @Interface("spectra:spectra.merge")
  public static void merge(double[] a, int na, double[] b, int nb, double[] out, FnDD_Z op) {
    Launch.call("spectra:spectra.merge", null, new Object[] {a, na, b, nb, out, op});
  }

  @Interface("spectra:spectra.merge")
  public static <T> void merge(T[] a, int na, T[] b, int nb, T[] out, FnLL_Z<T, T> op) {
    Launch.call("spectra:spectra.merge", null, new Object[] {a, na, b, nb, out, op});
  }

  @Interface("spectra:spectra.min_element")
  public static int min_element(int[] in, int n, FnII_Z op) {
    return (Integer) Launch.call("spectra:spectra.min_element", null, new Object[] {in, n, op});
  }

  @Interface("spectra:spectra.min_element")
  public static int min_element(long[] in, int n, FnJJ_Z op) {
    return (Integer) Launch.call("spectra:spectra.min_element", null, new Object[] {in, n, op});
  }

  @Interface("spectra:spectra.min_element")
  public static int min_element(float[] in, int n, FnFF_Z op) {
    return (Integer) Launch.call("spectra:spectra.min_element", null, new Object[] {in, n, op});
  }

  @Interface("spectra:spectra.min_element")
  public static int min_element(double[] in, int n, FnDD_Z op) {
    return (Integer) Launch.call("spectra:spectra.min_element", null, new Object[] {in, n, op});
  }

  @Interface("spectra:spectra.min_element")
  public static <T> int min_element(T[] in, int n, FnLL_Z<T, T> op) {
    return (Integer) Launch.call("spectra:spectra.min_element", null, new Object[] {in, n, op});
  }

  @Interface("spectra:spectra.minmax_element")
  public static void minmax_element(int[] in, int n, int[] min_out, int[] max_out, FnII_Z op) {
    Launch.call("spectra:spectra.minmax_element", null, new Object[] {in, n, min_out, max_out, op});
  }

  @Interface("spectra:spectra.minmax_element")
  public static void minmax_element(long[] in, int n, int[] min_out, int[] max_out, FnJJ_Z op) {
    Launch.call("spectra:spectra.minmax_element", null, new Object[] {in, n, min_out, max_out, op});
  }

  @Interface("spectra:spectra.minmax_element")
  public static void minmax_element(float[] in, int n, int[] min_out, int[] max_out, FnFF_Z op) {
    Launch.call("spectra:spectra.minmax_element", null, new Object[] {in, n, min_out, max_out, op});
  }

  @Interface("spectra:spectra.minmax_element")
  public static void minmax_element(double[] in, int n, int[] min_out, int[] max_out, FnDD_Z op) {
    Launch.call("spectra:spectra.minmax_element", null, new Object[] {in, n, min_out, max_out, op});
  }

  @Interface("spectra:spectra.minmax_element")
  public static <T> void minmax_element(
      T[] in,
      int n,
      int[] min_out,
      int[] max_out,
      FnLL_Z<T, T> op) {
    Launch.call("spectra:spectra.minmax_element", null, new Object[] {in, n, min_out, max_out, op});
  }

  @Interface("spectra:spectra.mismatch")
  public static int mismatch(int[] a, int[] b, int n, FnII_Z op) {
    return (Integer) Launch.call("spectra:spectra.mismatch", null, new Object[] {a, b, n, op});
  }

  @Interface("spectra:spectra.mismatch")
  public static int mismatch(long[] a, long[] b, int n, FnJJ_Z op) {
    return (Integer) Launch.call("spectra:spectra.mismatch", null, new Object[] {a, b, n, op});
  }

  @Interface("spectra:spectra.mismatch")
  public static int mismatch(float[] a, float[] b, int n, FnFF_Z op) {
    return (Integer) Launch.call("spectra:spectra.mismatch", null, new Object[] {a, b, n, op});
  }

  @Interface("spectra:spectra.mismatch")
  public static int mismatch(double[] a, double[] b, int n, FnDD_Z op) {
    return (Integer) Launch.call("spectra:spectra.mismatch", null, new Object[] {a, b, n, op});
  }

  @Interface("spectra:spectra.mismatch")
  public static <T> int mismatch(T[] a, T[] b, int n, FnLL_Z<T, T> op) {
    return (Integer) Launch.call("spectra:spectra.mismatch", null, new Object[] {a, b, n, op});
  }

  @Interface("spectra:spectra.none_of")
  public static boolean none_of(int[] in, int n, FnI_Z op) {
    return (Boolean) Launch.call("spectra:spectra.none_of", null, new Object[] {in, n, op});
  }

  @Interface("spectra:spectra.none_of")
  public static boolean none_of(long[] in, int n, FnJ_Z op) {
    return (Boolean) Launch.call("spectra:spectra.none_of", null, new Object[] {in, n, op});
  }

  @Interface("spectra:spectra.none_of")
  public static boolean none_of(float[] in, int n, FnF_Z op) {
    return (Boolean) Launch.call("spectra:spectra.none_of", null, new Object[] {in, n, op});
  }

  @Interface("spectra:spectra.none_of")
  public static boolean none_of(double[] in, int n, FnD_Z op) {
    return (Boolean) Launch.call("spectra:spectra.none_of", null, new Object[] {in, n, op});
  }

  @Interface("spectra:spectra.none_of")
  public static <T> boolean none_of(T[] in, int n, FnL_Z<T> op) {
    return (Boolean) Launch.call("spectra:spectra.none_of", null, new Object[] {in, n, op});
  }

  @Interface("spectra:spectra.partition")
  public static int partition(int[] data, int n, FnI_Z op) {
    return (Integer) Launch.call("spectra:spectra.partition", null, new Object[] {data, n, op});
  }

  @Interface("spectra:spectra.partition")
  public static int partition(long[] data, int n, FnJ_Z op) {
    return (Integer) Launch.call("spectra:spectra.partition", null, new Object[] {data, n, op});
  }

  @Interface("spectra:spectra.partition")
  public static int partition(float[] data, int n, FnF_Z op) {
    return (Integer) Launch.call("spectra:spectra.partition", null, new Object[] {data, n, op});
  }

  @Interface("spectra:spectra.partition")
  public static int partition(double[] data, int n, FnD_Z op) {
    return (Integer) Launch.call("spectra:spectra.partition", null, new Object[] {data, n, op});
  }

  @Interface("spectra:spectra.partition")
  public static <T> int partition(T[] data, int n, FnL_Z<T> op) {
    return (Integer) Launch.call("spectra:spectra.partition", null, new Object[] {data, n, op});
  }

  @Interface("spectra:spectra.partition_point")
  public static int partition_point(int[] in, int n, FnI_Z op) {
    return (Integer) Launch.call("spectra:spectra.partition_point", null, new Object[] {in, n, op});
  }

  @Interface("spectra:spectra.partition_point")
  public static int partition_point(long[] in, int n, FnJ_Z op) {
    return (Integer) Launch.call("spectra:spectra.partition_point", null, new Object[] {in, n, op});
  }

  @Interface("spectra:spectra.partition_point")
  public static int partition_point(float[] in, int n, FnF_Z op) {
    return (Integer) Launch.call("spectra:spectra.partition_point", null, new Object[] {in, n, op});
  }

  @Interface("spectra:spectra.partition_point")
  public static int partition_point(double[] in, int n, FnD_Z op) {
    return (Integer) Launch.call("spectra:spectra.partition_point", null, new Object[] {in, n, op});
  }

  @Interface("spectra:spectra.partition_point")
  public static <T> int partition_point(T[] in, int n, FnL_Z<T> op) {
    return (Integer) Launch.call("spectra:spectra.partition_point", null, new Object[] {in, n, op});
  }

  @Interface("spectra:spectra.reduce")
  public static int reduce(int[] in, int n, int init, FnII_I op) {
    return (Integer) Launch.call("spectra:spectra.reduce", null, new Object[] {in, n, init, op});
  }

  @Interface("spectra:spectra.reduce")
  public static long reduce(long[] in, int n, long init, FnJJ_J op) {
    return (Long) Launch.call("spectra:spectra.reduce", null, new Object[] {in, n, init, op});
  }

  @Interface("spectra:spectra.reduce")
  public static float reduce(float[] in, int n, float init, FnFF_F op) {
    return (Float) Launch.call("spectra:spectra.reduce", null, new Object[] {in, n, init, op});
  }

  @Interface("spectra:spectra.reduce")
  public static double reduce(double[] in, int n, double init, FnDD_D op) {
    return (Double) Launch.call("spectra:spectra.reduce", null, new Object[] {in, n, init, op});
  }

  @Interface("spectra:spectra.reduce")
  @SuppressWarnings("unchecked")
  public static <T> T reduce(T[] in, int n, T init, FnLL_L<T, T, T> op) {
    return (T) Launch.call("spectra:spectra.reduce", null, new Object[] {in, n, init, op});
  }

  @Interface("spectra:spectra.reduce_by_key")
  public static int reduce_by_key(
      int[] keys,
      int[] vals,
      int[] kout,
      int[] vout,
      int n,
      FnII_Z eq,
      FnII_I op) {
    return (Integer) Launch.call("spectra:spectra.reduce_by_key", null, new Object[] {keys, vals, kout, vout, n, eq, op});
  }

  @Interface("spectra:spectra.reduce_by_key")
  public static int reduce_by_key(
      int[] keys,
      long[] vals,
      int[] kout,
      long[] vout,
      int n,
      FnII_Z eq,
      FnJJ_J op) {
    return (Integer) Launch.call("spectra:spectra.reduce_by_key", null, new Object[] {keys, vals, kout, vout, n, eq, op});
  }

  @Interface("spectra:spectra.reduce_by_key")
  public static int reduce_by_key(
      int[] keys,
      float[] vals,
      int[] kout,
      float[] vout,
      int n,
      FnII_Z eq,
      FnFF_F op) {
    return (Integer) Launch.call("spectra:spectra.reduce_by_key", null, new Object[] {keys, vals, kout, vout, n, eq, op});
  }

  @Interface("spectra:spectra.reduce_by_key")
  public static int reduce_by_key(
      int[] keys,
      double[] vals,
      int[] kout,
      double[] vout,
      int n,
      FnII_Z eq,
      FnDD_D op) {
    return (Integer) Launch.call("spectra:spectra.reduce_by_key", null, new Object[] {keys, vals, kout, vout, n, eq, op});
  }

  @Interface("spectra:spectra.reduce_by_key")
  public static int reduce_by_key(
      long[] keys,
      int[] vals,
      long[] kout,
      int[] vout,
      int n,
      FnJJ_Z eq,
      FnII_I op) {
    return (Integer) Launch.call("spectra:spectra.reduce_by_key", null, new Object[] {keys, vals, kout, vout, n, eq, op});
  }

  @Interface("spectra:spectra.reduce_by_key")
  public static int reduce_by_key(
      long[] keys,
      long[] vals,
      long[] kout,
      long[] vout,
      int n,
      FnJJ_Z eq,
      FnJJ_J op) {
    return (Integer) Launch.call("spectra:spectra.reduce_by_key", null, new Object[] {keys, vals, kout, vout, n, eq, op});
  }

  @Interface("spectra:spectra.reduce_by_key")
  public static int reduce_by_key(
      long[] keys,
      float[] vals,
      long[] kout,
      float[] vout,
      int n,
      FnJJ_Z eq,
      FnFF_F op) {
    return (Integer) Launch.call("spectra:spectra.reduce_by_key", null, new Object[] {keys, vals, kout, vout, n, eq, op});
  }

  @Interface("spectra:spectra.reduce_by_key")
  public static int reduce_by_key(
      long[] keys,
      double[] vals,
      long[] kout,
      double[] vout,
      int n,
      FnJJ_Z eq,
      FnDD_D op) {
    return (Integer) Launch.call("spectra:spectra.reduce_by_key", null, new Object[] {keys, vals, kout, vout, n, eq, op});
  }

  @Interface("spectra:spectra.reduce_by_key")
  public static int reduce_by_key(
      float[] keys,
      int[] vals,
      float[] kout,
      int[] vout,
      int n,
      FnFF_Z eq,
      FnII_I op) {
    return (Integer) Launch.call("spectra:spectra.reduce_by_key", null, new Object[] {keys, vals, kout, vout, n, eq, op});
  }

  @Interface("spectra:spectra.reduce_by_key")
  public static int reduce_by_key(
      float[] keys,
      long[] vals,
      float[] kout,
      long[] vout,
      int n,
      FnFF_Z eq,
      FnJJ_J op) {
    return (Integer) Launch.call("spectra:spectra.reduce_by_key", null, new Object[] {keys, vals, kout, vout, n, eq, op});
  }

  @Interface("spectra:spectra.reduce_by_key")
  public static int reduce_by_key(
      float[] keys,
      float[] vals,
      float[] kout,
      float[] vout,
      int n,
      FnFF_Z eq,
      FnFF_F op) {
    return (Integer) Launch.call("spectra:spectra.reduce_by_key", null, new Object[] {keys, vals, kout, vout, n, eq, op});
  }

  @Interface("spectra:spectra.reduce_by_key")
  public static int reduce_by_key(
      float[] keys,
      double[] vals,
      float[] kout,
      double[] vout,
      int n,
      FnFF_Z eq,
      FnDD_D op) {
    return (Integer) Launch.call("spectra:spectra.reduce_by_key", null, new Object[] {keys, vals, kout, vout, n, eq, op});
  }

  @Interface("spectra:spectra.reduce_by_key")
  public static int reduce_by_key(
      double[] keys,
      int[] vals,
      double[] kout,
      int[] vout,
      int n,
      FnDD_Z eq,
      FnII_I op) {
    return (Integer) Launch.call("spectra:spectra.reduce_by_key", null, new Object[] {keys, vals, kout, vout, n, eq, op});
  }

  @Interface("spectra:spectra.reduce_by_key")
  public static int reduce_by_key(
      double[] keys,
      long[] vals,
      double[] kout,
      long[] vout,
      int n,
      FnDD_Z eq,
      FnJJ_J op) {
    return (Integer) Launch.call("spectra:spectra.reduce_by_key", null, new Object[] {keys, vals, kout, vout, n, eq, op});
  }

  @Interface("spectra:spectra.reduce_by_key")
  public static int reduce_by_key(
      double[] keys,
      float[] vals,
      double[] kout,
      float[] vout,
      int n,
      FnDD_Z eq,
      FnFF_F op) {
    return (Integer) Launch.call("spectra:spectra.reduce_by_key", null, new Object[] {keys, vals, kout, vout, n, eq, op});
  }

  @Interface("spectra:spectra.reduce_by_key")
  public static int reduce_by_key(
      double[] keys,
      double[] vals,
      double[] kout,
      double[] vout,
      int n,
      FnDD_Z eq,
      FnDD_D op) {
    return (Integer) Launch.call("spectra:spectra.reduce_by_key", null, new Object[] {keys, vals, kout, vout, n, eq, op});
  }

  @Interface("spectra:spectra.reduce_by_key")
  public static <K, V> int reduce_by_key(
      K[] keys,
      V[] vals,
      K[] kout,
      V[] vout,
      int n,
      FnLL_Z<K, K> eq,
      FnLL_L<V, V, V> op) {
    return (Integer) Launch.call("spectra:spectra.reduce_by_key", null, new Object[] {keys, vals, kout, vout, n, eq, op});
  }

  @Interface("spectra:spectra.reduce_unordered")
  public static int reduce_unordered(int[] in, int n, int init, FnII_I op) {
    return (Integer) Launch.call("spectra:spectra.reduce_unordered", null, new Object[] {in, n, init, op});
  }

  @Interface("spectra:spectra.reduce_unordered")
  public static long reduce_unordered(long[] in, int n, long init, FnJJ_J op) {
    return (Long) Launch.call("spectra:spectra.reduce_unordered", null, new Object[] {in, n, init, op});
  }

  @Interface("spectra:spectra.reduce_unordered")
  public static float reduce_unordered(float[] in, int n, float init, FnFF_F op) {
    return (Float) Launch.call("spectra:spectra.reduce_unordered", null, new Object[] {in, n, init, op});
  }

  @Interface("spectra:spectra.reduce_unordered")
  public static double reduce_unordered(double[] in, int n, double init, FnDD_D op) {
    return (Double) Launch.call("spectra:spectra.reduce_unordered", null, new Object[] {in, n, init, op});
  }

  @Interface("spectra:spectra.reduce_unordered")
  @SuppressWarnings("unchecked")
  public static <T> T reduce_unordered(T[] in, int n, T init, FnLL_L<T, T, T> op) {
    return (T) Launch.call("spectra:spectra.reduce_unordered", null, new Object[] {in, n, init, op});
  }

  @Interface("spectra:spectra.remove")
  public static int remove(int[] data, int n, int value, FnII_Z eq) {
    return (Integer) Launch.call("spectra:spectra.remove", null, new Object[] {data, n, value, eq});
  }

  @Interface("spectra:spectra.remove")
  public static int remove(long[] data, int n, long value, FnJJ_Z eq) {
    return (Integer) Launch.call("spectra:spectra.remove", null, new Object[] {data, n, value, eq});
  }

  @Interface("spectra:spectra.remove")
  public static int remove(float[] data, int n, float value, FnFF_Z eq) {
    return (Integer) Launch.call("spectra:spectra.remove", null, new Object[] {data, n, value, eq});
  }

  @Interface("spectra:spectra.remove")
  public static int remove(double[] data, int n, double value, FnDD_Z eq) {
    return (Integer) Launch.call("spectra:spectra.remove", null, new Object[] {data, n, value, eq});
  }

  @Interface("spectra:spectra.remove")
  public static <T> int remove(T[] data, int n, T value, FnLL_Z<T, T> eq) {
    return (Integer) Launch.call("spectra:spectra.remove", null, new Object[] {data, n, value, eq});
  }

  @Interface("spectra:spectra.remove_copy")
  public static int remove_copy(int[] in, int n, int[] out, int value, FnII_Z eq) {
    return (Integer) Launch.call("spectra:spectra.remove_copy", null, new Object[] {in, n, out, value, eq});
  }

  @Interface("spectra:spectra.remove_copy")
  public static int remove_copy(long[] in, int n, long[] out, long value, FnJJ_Z eq) {
    return (Integer) Launch.call("spectra:spectra.remove_copy", null, new Object[] {in, n, out, value, eq});
  }

  @Interface("spectra:spectra.remove_copy")
  public static int remove_copy(float[] in, int n, float[] out, float value, FnFF_Z eq) {
    return (Integer) Launch.call("spectra:spectra.remove_copy", null, new Object[] {in, n, out, value, eq});
  }

  @Interface("spectra:spectra.remove_copy")
  public static int remove_copy(double[] in, int n, double[] out, double value, FnDD_Z eq) {
    return (Integer) Launch.call("spectra:spectra.remove_copy", null, new Object[] {in, n, out, value, eq});
  }

  @Interface("spectra:spectra.remove_copy")
  public static <T> int remove_copy(T[] in, int n, T[] out, T value, FnLL_Z<T, T> eq) {
    return (Integer) Launch.call("spectra:spectra.remove_copy", null, new Object[] {in, n, out, value, eq});
  }

  @Interface("spectra:spectra.remove_copy_if")
  public static int remove_copy_if(int[] in, int n, int[] out, FnI_Z op) {
    return (Integer) Launch.call("spectra:spectra.remove_copy_if", null, new Object[] {in, n, out, op});
  }

  @Interface("spectra:spectra.remove_copy_if")
  public static int remove_copy_if(long[] in, int n, long[] out, FnJ_Z op) {
    return (Integer) Launch.call("spectra:spectra.remove_copy_if", null, new Object[] {in, n, out, op});
  }

  @Interface("spectra:spectra.remove_copy_if")
  public static int remove_copy_if(float[] in, int n, float[] out, FnF_Z op) {
    return (Integer) Launch.call("spectra:spectra.remove_copy_if", null, new Object[] {in, n, out, op});
  }

  @Interface("spectra:spectra.remove_copy_if")
  public static int remove_copy_if(double[] in, int n, double[] out, FnD_Z op) {
    return (Integer) Launch.call("spectra:spectra.remove_copy_if", null, new Object[] {in, n, out, op});
  }

  @Interface("spectra:spectra.remove_copy_if")
  public static <T> int remove_copy_if(T[] in, int n, T[] out, FnL_Z<T> op) {
    return (Integer) Launch.call("spectra:spectra.remove_copy_if", null, new Object[] {in, n, out, op});
  }

  @Interface("spectra:spectra.remove_if")
  public static int remove_if(int[] data, int n, FnI_Z op) {
    return (Integer) Launch.call("spectra:spectra.remove_if", null, new Object[] {data, n, op});
  }

  @Interface("spectra:spectra.remove_if")
  public static int remove_if(long[] data, int n, FnJ_Z op) {
    return (Integer) Launch.call("spectra:spectra.remove_if", null, new Object[] {data, n, op});
  }

  @Interface("spectra:spectra.remove_if")
  public static int remove_if(float[] data, int n, FnF_Z op) {
    return (Integer) Launch.call("spectra:spectra.remove_if", null, new Object[] {data, n, op});
  }

  @Interface("spectra:spectra.remove_if")
  public static int remove_if(double[] data, int n, FnD_Z op) {
    return (Integer) Launch.call("spectra:spectra.remove_if", null, new Object[] {data, n, op});
  }

  @Interface("spectra:spectra.remove_if")
  public static <T> int remove_if(T[] data, int n, FnL_Z<T> op) {
    return (Integer) Launch.call("spectra:spectra.remove_if", null, new Object[] {data, n, op});
  }

  @Interface("spectra:spectra.replace")
  public static void replace(int[] io, int n, int oldv, int newv, FnII_Z eq) {
    Launch.call("spectra:spectra.replace", null, new Object[] {io, n, oldv, newv, eq});
  }

  @Interface("spectra:spectra.replace")
  public static void replace(long[] io, int n, long oldv, long newv, FnJJ_Z eq) {
    Launch.call("spectra:spectra.replace", null, new Object[] {io, n, oldv, newv, eq});
  }

  @Interface("spectra:spectra.replace")
  public static void replace(float[] io, int n, float oldv, float newv, FnFF_Z eq) {
    Launch.call("spectra:spectra.replace", null, new Object[] {io, n, oldv, newv, eq});
  }

  @Interface("spectra:spectra.replace")
  public static void replace(double[] io, int n, double oldv, double newv, FnDD_Z eq) {
    Launch.call("spectra:spectra.replace", null, new Object[] {io, n, oldv, newv, eq});
  }

  @Interface("spectra:spectra.replace")
  public static <T> void replace(T[] io, int n, T oldv, T newv, FnLL_Z<T, T> eq) {
    Launch.call("spectra:spectra.replace", null, new Object[] {io, n, oldv, newv, eq});
  }

  @Interface("spectra:spectra.replace_copy")
  public static void replace_copy(int[] in, int[] out, int n, int oldv, int newv, FnII_Z eq) {
    Launch.call("spectra:spectra.replace_copy", null, new Object[] {in, out, n, oldv, newv, eq});
  }

  @Interface("spectra:spectra.replace_copy")
  public static void replace_copy(long[] in, long[] out, int n, long oldv, long newv, FnJJ_Z eq) {
    Launch.call("spectra:spectra.replace_copy", null, new Object[] {in, out, n, oldv, newv, eq});
  }

  @Interface("spectra:spectra.replace_copy")
  public static void replace_copy(
      float[] in,
      float[] out,
      int n,
      float oldv,
      float newv,
      FnFF_Z eq) {
    Launch.call("spectra:spectra.replace_copy", null, new Object[] {in, out, n, oldv, newv, eq});
  }

  @Interface("spectra:spectra.replace_copy")
  public static void replace_copy(
      double[] in,
      double[] out,
      int n,
      double oldv,
      double newv,
      FnDD_Z eq) {
    Launch.call("spectra:spectra.replace_copy", null, new Object[] {in, out, n, oldv, newv, eq});
  }

  @Interface("spectra:spectra.replace_copy")
  public static <T> void replace_copy(T[] in, T[] out, int n, T oldv, T newv, FnLL_Z<T, T> eq) {
    Launch.call("spectra:spectra.replace_copy", null, new Object[] {in, out, n, oldv, newv, eq});
  }

  @Interface("spectra:spectra.replace_copy_if")
  public static void replace_copy_if(int[] in, int[] out, int n, int new_value, FnI_Z op) {
    Launch.call("spectra:spectra.replace_copy_if", null, new Object[] {in, out, n, new_value, op});
  }

  @Interface("spectra:spectra.replace_copy_if")
  public static void replace_copy_if(long[] in, long[] out, int n, long new_value, FnJ_Z op) {
    Launch.call("spectra:spectra.replace_copy_if", null, new Object[] {in, out, n, new_value, op});
  }

  @Interface("spectra:spectra.replace_copy_if")
  public static void replace_copy_if(float[] in, float[] out, int n, float new_value, FnF_Z op) {
    Launch.call("spectra:spectra.replace_copy_if", null, new Object[] {in, out, n, new_value, op});
  }

  @Interface("spectra:spectra.replace_copy_if")
  public static void replace_copy_if(double[] in, double[] out, int n, double new_value, FnD_Z op) {
    Launch.call("spectra:spectra.replace_copy_if", null, new Object[] {in, out, n, new_value, op});
  }

  @Interface("spectra:spectra.replace_copy_if")
  public static <T> void replace_copy_if(T[] in, T[] out, int n, T new_value, FnL_Z<T> op) {
    Launch.call("spectra:spectra.replace_copy_if", null, new Object[] {in, out, n, new_value, op});
  }

  @Interface("spectra:spectra.replace_if")
  public static void replace_if(int[] data, int n, int new_value, FnI_Z op) {
    Launch.call("spectra:spectra.replace_if", null, new Object[] {data, n, new_value, op});
  }

  @Interface("spectra:spectra.replace_if")
  public static void replace_if(long[] data, int n, long new_value, FnJ_Z op) {
    Launch.call("spectra:spectra.replace_if", null, new Object[] {data, n, new_value, op});
  }

  @Interface("spectra:spectra.replace_if")
  public static void replace_if(float[] data, int n, float new_value, FnF_Z op) {
    Launch.call("spectra:spectra.replace_if", null, new Object[] {data, n, new_value, op});
  }

  @Interface("spectra:spectra.replace_if")
  public static void replace_if(double[] data, int n, double new_value, FnD_Z op) {
    Launch.call("spectra:spectra.replace_if", null, new Object[] {data, n, new_value, op});
  }

  @Interface("spectra:spectra.replace_if")
  public static <T> void replace_if(T[] data, int n, T new_value, FnL_Z<T> op) {
    Launch.call("spectra:spectra.replace_if", null, new Object[] {data, n, new_value, op});
  }

  @Interface("spectra:spectra.reverse")
  public static void reverse(int[] data, int n) {
    Launch.call("spectra:spectra.reverse", null, new Object[] {data, n});
  }

  @Interface("spectra:spectra.reverse")
  public static void reverse(long[] data, int n) {
    Launch.call("spectra:spectra.reverse", null, new Object[] {data, n});
  }

  @Interface("spectra:spectra.reverse")
  public static void reverse(float[] data, int n) {
    Launch.call("spectra:spectra.reverse", null, new Object[] {data, n});
  }

  @Interface("spectra:spectra.reverse")
  public static void reverse(double[] data, int n) {
    Launch.call("spectra:spectra.reverse", null, new Object[] {data, n});
  }

  @Interface("spectra:spectra.reverse")
  public static <T> void reverse(T[] data, int n) {
    Launch.call("spectra:spectra.reverse", null, new Object[] {data, n});
  }

  @Interface("spectra:spectra.reverse_copy")
  public static void reverse_copy(int[] in, int n, int[] out) {
    Launch.call("spectra:spectra.reverse_copy", null, new Object[] {in, n, out});
  }

  @Interface("spectra:spectra.reverse_copy")
  public static void reverse_copy(long[] in, int n, long[] out) {
    Launch.call("spectra:spectra.reverse_copy", null, new Object[] {in, n, out});
  }

  @Interface("spectra:spectra.reverse_copy")
  public static void reverse_copy(float[] in, int n, float[] out) {
    Launch.call("spectra:spectra.reverse_copy", null, new Object[] {in, n, out});
  }

  @Interface("spectra:spectra.reverse_copy")
  public static void reverse_copy(double[] in, int n, double[] out) {
    Launch.call("spectra:spectra.reverse_copy", null, new Object[] {in, n, out});
  }

  @Interface("spectra:spectra.reverse_copy")
  public static <T> void reverse_copy(T[] in, int n, T[] out) {
    Launch.call("spectra:spectra.reverse_copy", null, new Object[] {in, n, out});
  }

  @Interface("spectra:spectra.scatter")
  public static void scatter(int[] in, int n, int[] map, int[] out, int out_n) {
    Launch.call("spectra:spectra.scatter", null, new Object[] {in, n, map, out, out_n});
  }

  @Interface("spectra:spectra.scatter")
  public static void scatter(long[] in, int n, int[] map, long[] out, int out_n) {
    Launch.call("spectra:spectra.scatter", null, new Object[] {in, n, map, out, out_n});
  }

  @Interface("spectra:spectra.scatter")
  public static void scatter(float[] in, int n, int[] map, float[] out, int out_n) {
    Launch.call("spectra:spectra.scatter", null, new Object[] {in, n, map, out, out_n});
  }

  @Interface("spectra:spectra.scatter")
  public static void scatter(double[] in, int n, int[] map, double[] out, int out_n) {
    Launch.call("spectra:spectra.scatter", null, new Object[] {in, n, map, out, out_n});
  }

  @Interface("spectra:spectra.scatter")
  public static <T> void scatter(T[] in, int n, int[] map, T[] out, int out_n) {
    Launch.call("spectra:spectra.scatter", null, new Object[] {in, n, map, out, out_n});
  }

  @Interface("spectra:spectra.search")
  public static int search(int[] in, int n, int[] sub, int m, FnII_Z op) {
    return (Integer) Launch.call("spectra:spectra.search", null, new Object[] {in, n, sub, m, op});
  }

  @Interface("spectra:spectra.search")
  public static int search(long[] in, int n, long[] sub, int m, FnJJ_Z op) {
    return (Integer) Launch.call("spectra:spectra.search", null, new Object[] {in, n, sub, m, op});
  }

  @Interface("spectra:spectra.search")
  public static int search(float[] in, int n, float[] sub, int m, FnFF_Z op) {
    return (Integer) Launch.call("spectra:spectra.search", null, new Object[] {in, n, sub, m, op});
  }

  @Interface("spectra:spectra.search")
  public static int search(double[] in, int n, double[] sub, int m, FnDD_Z op) {
    return (Integer) Launch.call("spectra:spectra.search", null, new Object[] {in, n, sub, m, op});
  }

  @Interface("spectra:spectra.search")
  public static <T> int search(T[] in, int n, T[] sub, int m, FnLL_Z<T, T> op) {
    return (Integer) Launch.call("spectra:spectra.search", null, new Object[] {in, n, sub, m, op});
  }

  @Interface("spectra:spectra.search_n")
  public static int search_n(int[] in, int n, int count, int value, FnII_Z op) {
    return (Integer) Launch.call("spectra:spectra.search_n", null, new Object[] {in, n, count, value, op});
  }

  @Interface("spectra:spectra.search_n")
  public static int search_n(long[] in, int n, int count, long value, FnJJ_Z op) {
    return (Integer) Launch.call("spectra:spectra.search_n", null, new Object[] {in, n, count, value, op});
  }

  @Interface("spectra:spectra.search_n")
  public static int search_n(float[] in, int n, int count, float value, FnFF_Z op) {
    return (Integer) Launch.call("spectra:spectra.search_n", null, new Object[] {in, n, count, value, op});
  }

  @Interface("spectra:spectra.search_n")
  public static int search_n(double[] in, int n, int count, double value, FnDD_Z op) {
    return (Integer) Launch.call("spectra:spectra.search_n", null, new Object[] {in, n, count, value, op});
  }

  @Interface("spectra:spectra.search_n")
  public static <T> int search_n(T[] in, int n, int count, T value, FnLL_Z<T, T> op) {
    return (Integer) Launch.call("spectra:spectra.search_n", null, new Object[] {in, n, count, value, op});
  }

  @Interface("spectra:spectra.sequence")
  public static void sequence(int[] out, int n, int init, int step, FnIII_I value_at) {
    Launch.call("spectra:spectra.sequence", null, new Object[] {out, n, init, step, value_at});
  }

  @Interface("spectra:spectra.sequence")
  public static void sequence(long[] out, int n, long init, long step, FnJJI_J value_at) {
    Launch.call("spectra:spectra.sequence", null, new Object[] {out, n, init, step, value_at});
  }

  @Interface("spectra:spectra.sequence")
  public static void sequence(float[] out, int n, float init, float step, FnFFI_F value_at) {
    Launch.call("spectra:spectra.sequence", null, new Object[] {out, n, init, step, value_at});
  }

  @Interface("spectra:spectra.sequence")
  public static void sequence(double[] out, int n, double init, double step, FnDDI_D value_at) {
    Launch.call("spectra:spectra.sequence", null, new Object[] {out, n, init, step, value_at});
  }

  @Interface("spectra:spectra.sequence")
  public static <T> void sequence(T[] out, int n, T init, T step, FnLLI_L<T, T, T> value_at) {
    Launch.call("spectra:spectra.sequence", null, new Object[] {out, n, init, step, value_at});
  }

  @Interface("spectra:spectra.set_difference")
  public static int set_difference(int[] a, int na, int[] b, int nb, int[] out, FnII_Z op) {
    return (Integer) Launch.call("spectra:spectra.set_difference", null, new Object[] {a, na, b, nb, out, op});
  }

  @Interface("spectra:spectra.set_difference")
  public static int set_difference(long[] a, int na, long[] b, int nb, long[] out, FnJJ_Z op) {
    return (Integer) Launch.call("spectra:spectra.set_difference", null, new Object[] {a, na, b, nb, out, op});
  }

  @Interface("spectra:spectra.set_difference")
  public static int set_difference(float[] a, int na, float[] b, int nb, float[] out, FnFF_Z op) {
    return (Integer) Launch.call("spectra:spectra.set_difference", null, new Object[] {a, na, b, nb, out, op});
  }

  @Interface("spectra:spectra.set_difference")
  public static int set_difference(
      double[] a,
      int na,
      double[] b,
      int nb,
      double[] out,
      FnDD_Z op) {
    return (Integer) Launch.call("spectra:spectra.set_difference", null, new Object[] {a, na, b, nb, out, op});
  }

  @Interface("spectra:spectra.set_difference")
  public static <T> int set_difference(T[] a, int na, T[] b, int nb, T[] out, FnLL_Z<T, T> op) {
    return (Integer) Launch.call("spectra:spectra.set_difference", null, new Object[] {a, na, b, nb, out, op});
  }

  @Interface("spectra:spectra.set_intersection")
  public static int set_intersection(int[] a, int na, int[] b, int nb, int[] out, FnII_Z op) {
    return (Integer) Launch.call("spectra:spectra.set_intersection", null, new Object[] {a, na, b, nb, out, op});
  }

  @Interface("spectra:spectra.set_intersection")
  public static int set_intersection(long[] a, int na, long[] b, int nb, long[] out, FnJJ_Z op) {
    return (Integer) Launch.call("spectra:spectra.set_intersection", null, new Object[] {a, na, b, nb, out, op});
  }

  @Interface("spectra:spectra.set_intersection")
  public static int set_intersection(float[] a, int na, float[] b, int nb, float[] out, FnFF_Z op) {
    return (Integer) Launch.call("spectra:spectra.set_intersection", null, new Object[] {a, na, b, nb, out, op});
  }

  @Interface("spectra:spectra.set_intersection")
  public static int set_intersection(
      double[] a,
      int na,
      double[] b,
      int nb,
      double[] out,
      FnDD_Z op) {
    return (Integer) Launch.call("spectra:spectra.set_intersection", null, new Object[] {a, na, b, nb, out, op});
  }

  @Interface("spectra:spectra.set_intersection")
  public static <T> int set_intersection(T[] a, int na, T[] b, int nb, T[] out, FnLL_Z<T, T> op) {
    return (Integer) Launch.call("spectra:spectra.set_intersection", null, new Object[] {a, na, b, nb, out, op});
  }

  @Interface("spectra:spectra.set_union")
  public static int set_union(int[] a, int na, int[] b, int nb, int[] out, FnII_Z op) {
    return (Integer) Launch.call("spectra:spectra.set_union", null, new Object[] {a, na, b, nb, out, op});
  }

  @Interface("spectra:spectra.set_union")
  public static int set_union(long[] a, int na, long[] b, int nb, long[] out, FnJJ_Z op) {
    return (Integer) Launch.call("spectra:spectra.set_union", null, new Object[] {a, na, b, nb, out, op});
  }

  @Interface("spectra:spectra.set_union")
  public static int set_union(float[] a, int na, float[] b, int nb, float[] out, FnFF_Z op) {
    return (Integer) Launch.call("spectra:spectra.set_union", null, new Object[] {a, na, b, nb, out, op});
  }

  @Interface("spectra:spectra.set_union")
  public static int set_union(double[] a, int na, double[] b, int nb, double[] out, FnDD_Z op) {
    return (Integer) Launch.call("spectra:spectra.set_union", null, new Object[] {a, na, b, nb, out, op});
  }

  @Interface("spectra:spectra.set_union")
  public static <T> int set_union(T[] a, int na, T[] b, int nb, T[] out, FnLL_Z<T, T> op) {
    return (Integer) Launch.call("spectra:spectra.set_union", null, new Object[] {a, na, b, nb, out, op});
  }

  @Interface("spectra:spectra.sort")
  public static void sort(int[] data, int n, FnII_Z op) {
    Launch.call("spectra:spectra.sort", null, new Object[] {data, n, op});
  }

  @Interface("spectra:spectra.sort")
  public static void sort(long[] data, int n, FnJJ_Z op) {
    Launch.call("spectra:spectra.sort", null, new Object[] {data, n, op});
  }

  @Interface("spectra:spectra.sort")
  public static void sort(float[] data, int n, FnFF_Z op) {
    Launch.call("spectra:spectra.sort", null, new Object[] {data, n, op});
  }

  @Interface("spectra:spectra.sort")
  public static void sort(double[] data, int n, FnDD_Z op) {
    Launch.call("spectra:spectra.sort", null, new Object[] {data, n, op});
  }

  @Interface("spectra:spectra.sort")
  public static <T> void sort(T[] data, int n, FnLL_Z<T, T> op) {
    Launch.call("spectra:spectra.sort", null, new Object[] {data, n, op});
  }

  @Interface("spectra:spectra.sort_by_key")
  public static void sort_by_key(int[] keys, int[] values, int n, FnII_Z op) {
    Launch.call("spectra:spectra.sort_by_key", null, new Object[] {keys, values, n, op});
  }

  @Interface("spectra:spectra.sort_by_key")
  public static void sort_by_key(int[] keys, long[] values, int n, FnII_Z op) {
    Launch.call("spectra:spectra.sort_by_key", null, new Object[] {keys, values, n, op});
  }

  @Interface("spectra:spectra.sort_by_key")
  public static void sort_by_key(int[] keys, float[] values, int n, FnII_Z op) {
    Launch.call("spectra:spectra.sort_by_key", null, new Object[] {keys, values, n, op});
  }

  @Interface("spectra:spectra.sort_by_key")
  public static void sort_by_key(int[] keys, double[] values, int n, FnII_Z op) {
    Launch.call("spectra:spectra.sort_by_key", null, new Object[] {keys, values, n, op});
  }

  @Interface("spectra:spectra.sort_by_key")
  public static void sort_by_key(long[] keys, int[] values, int n, FnJJ_Z op) {
    Launch.call("spectra:spectra.sort_by_key", null, new Object[] {keys, values, n, op});
  }

  @Interface("spectra:spectra.sort_by_key")
  public static void sort_by_key(long[] keys, long[] values, int n, FnJJ_Z op) {
    Launch.call("spectra:spectra.sort_by_key", null, new Object[] {keys, values, n, op});
  }

  @Interface("spectra:spectra.sort_by_key")
  public static void sort_by_key(long[] keys, float[] values, int n, FnJJ_Z op) {
    Launch.call("spectra:spectra.sort_by_key", null, new Object[] {keys, values, n, op});
  }

  @Interface("spectra:spectra.sort_by_key")
  public static void sort_by_key(long[] keys, double[] values, int n, FnJJ_Z op) {
    Launch.call("spectra:spectra.sort_by_key", null, new Object[] {keys, values, n, op});
  }

  @Interface("spectra:spectra.sort_by_key")
  public static void sort_by_key(float[] keys, int[] values, int n, FnFF_Z op) {
    Launch.call("spectra:spectra.sort_by_key", null, new Object[] {keys, values, n, op});
  }

  @Interface("spectra:spectra.sort_by_key")
  public static void sort_by_key(float[] keys, long[] values, int n, FnFF_Z op) {
    Launch.call("spectra:spectra.sort_by_key", null, new Object[] {keys, values, n, op});
  }

  @Interface("spectra:spectra.sort_by_key")
  public static void sort_by_key(float[] keys, float[] values, int n, FnFF_Z op) {
    Launch.call("spectra:spectra.sort_by_key", null, new Object[] {keys, values, n, op});
  }

  @Interface("spectra:spectra.sort_by_key")
  public static void sort_by_key(float[] keys, double[] values, int n, FnFF_Z op) {
    Launch.call("spectra:spectra.sort_by_key", null, new Object[] {keys, values, n, op});
  }

  @Interface("spectra:spectra.sort_by_key")
  public static void sort_by_key(double[] keys, int[] values, int n, FnDD_Z op) {
    Launch.call("spectra:spectra.sort_by_key", null, new Object[] {keys, values, n, op});
  }

  @Interface("spectra:spectra.sort_by_key")
  public static void sort_by_key(double[] keys, long[] values, int n, FnDD_Z op) {
    Launch.call("spectra:spectra.sort_by_key", null, new Object[] {keys, values, n, op});
  }

  @Interface("spectra:spectra.sort_by_key")
  public static void sort_by_key(double[] keys, float[] values, int n, FnDD_Z op) {
    Launch.call("spectra:spectra.sort_by_key", null, new Object[] {keys, values, n, op});
  }

  @Interface("spectra:spectra.sort_by_key")
  public static void sort_by_key(double[] keys, double[] values, int n, FnDD_Z op) {
    Launch.call("spectra:spectra.sort_by_key", null, new Object[] {keys, values, n, op});
  }

  @Interface("spectra:spectra.sort_by_key")
  public static <K, V> void sort_by_key(K[] keys, V[] values, int n, FnLL_Z<K, K> op) {
    Launch.call("spectra:spectra.sort_by_key", null, new Object[] {keys, values, n, op});
  }

  @Interface("spectra:spectra.stable_partition")
  public static int stable_partition(int[] data, int n, FnI_Z op) {
    return (Integer) Launch.call("spectra:spectra.stable_partition", null, new Object[] {data, n, op});
  }

  @Interface("spectra:spectra.stable_partition")
  public static int stable_partition(long[] data, int n, FnJ_Z op) {
    return (Integer) Launch.call("spectra:spectra.stable_partition", null, new Object[] {data, n, op});
  }

  @Interface("spectra:spectra.stable_partition")
  public static int stable_partition(float[] data, int n, FnF_Z op) {
    return (Integer) Launch.call("spectra:spectra.stable_partition", null, new Object[] {data, n, op});
  }

  @Interface("spectra:spectra.stable_partition")
  public static int stable_partition(double[] data, int n, FnD_Z op) {
    return (Integer) Launch.call("spectra:spectra.stable_partition", null, new Object[] {data, n, op});
  }

  @Interface("spectra:spectra.stable_partition")
  public static <T> int stable_partition(T[] data, int n, FnL_Z<T> op) {
    return (Integer) Launch.call("spectra:spectra.stable_partition", null, new Object[] {data, n, op});
  }

  @Interface("spectra:spectra.stable_sort")
  public static void stable_sort(int[] data, int n, FnII_Z op) {
    Launch.call("spectra:spectra.stable_sort", null, new Object[] {data, n, op});
  }

  @Interface("spectra:spectra.stable_sort")
  public static void stable_sort(long[] data, int n, FnJJ_Z op) {
    Launch.call("spectra:spectra.stable_sort", null, new Object[] {data, n, op});
  }

  @Interface("spectra:spectra.stable_sort")
  public static void stable_sort(float[] data, int n, FnFF_Z op) {
    Launch.call("spectra:spectra.stable_sort", null, new Object[] {data, n, op});
  }

  @Interface("spectra:spectra.stable_sort")
  public static void stable_sort(double[] data, int n, FnDD_Z op) {
    Launch.call("spectra:spectra.stable_sort", null, new Object[] {data, n, op});
  }

  @Interface("spectra:spectra.stable_sort")
  public static <T> void stable_sort(T[] data, int n, FnLL_Z<T, T> op) {
    Launch.call("spectra:spectra.stable_sort", null, new Object[] {data, n, op});
  }

  @Interface("spectra:spectra.swap_ranges")
  public static void swap_ranges(int[] a, int n, int[] b) {
    Launch.call("spectra:spectra.swap_ranges", null, new Object[] {a, n, b});
  }

  @Interface("spectra:spectra.swap_ranges")
  public static void swap_ranges(long[] a, int n, long[] b) {
    Launch.call("spectra:spectra.swap_ranges", null, new Object[] {a, n, b});
  }

  @Interface("spectra:spectra.swap_ranges")
  public static void swap_ranges(float[] a, int n, float[] b) {
    Launch.call("spectra:spectra.swap_ranges", null, new Object[] {a, n, b});
  }

  @Interface("spectra:spectra.swap_ranges")
  public static void swap_ranges(double[] a, int n, double[] b) {
    Launch.call("spectra:spectra.swap_ranges", null, new Object[] {a, n, b});
  }

  @Interface("spectra:spectra.swap_ranges")
  public static <T> void swap_ranges(T[] a, int n, T[] b) {
    Launch.call("spectra:spectra.swap_ranges", null, new Object[] {a, n, b});
  }

  @Interface("spectra:spectra.tabulate")
  public static void tabulate(int[] out, int n, FnI_I op) {
    Launch.call("spectra:spectra.tabulate", null, new Object[] {out, n, op});
  }

  @Interface("spectra:spectra.tabulate")
  public static void tabulate(long[] out, int n, FnI_J op) {
    Launch.call("spectra:spectra.tabulate", null, new Object[] {out, n, op});
  }

  @Interface("spectra:spectra.tabulate")
  public static void tabulate(float[] out, int n, FnI_F op) {
    Launch.call("spectra:spectra.tabulate", null, new Object[] {out, n, op});
  }

  @Interface("spectra:spectra.tabulate")
  public static void tabulate(double[] out, int n, FnI_D op) {
    Launch.call("spectra:spectra.tabulate", null, new Object[] {out, n, op});
  }

  @Interface("spectra:spectra.tabulate")
  public static <T> void tabulate(T[] out, int n, FnI_L<T> op) {
    Launch.call("spectra:spectra.tabulate", null, new Object[] {out, n, op});
  }

  @Interface("spectra:spectra.transform")
  public static void transform(int[] in, int[] out, int n, FnI_I op) {
    Launch.call("spectra:spectra.transform", null, new Object[] {in, out, n, op});
  }

  @Interface("spectra:spectra.transform")
  public static void transform(int[] in, long[] out, int n, FnI_J op) {
    Launch.call("spectra:spectra.transform", null, new Object[] {in, out, n, op});
  }

  @Interface("spectra:spectra.transform")
  public static void transform(int[] in, float[] out, int n, FnI_F op) {
    Launch.call("spectra:spectra.transform", null, new Object[] {in, out, n, op});
  }

  @Interface("spectra:spectra.transform")
  public static void transform(int[] in, double[] out, int n, FnI_D op) {
    Launch.call("spectra:spectra.transform", null, new Object[] {in, out, n, op});
  }

  @Interface("spectra:spectra.transform")
  public static void transform(long[] in, int[] out, int n, FnJ_I op) {
    Launch.call("spectra:spectra.transform", null, new Object[] {in, out, n, op});
  }

  @Interface("spectra:spectra.transform")
  public static void transform(long[] in, long[] out, int n, FnJ_J op) {
    Launch.call("spectra:spectra.transform", null, new Object[] {in, out, n, op});
  }

  @Interface("spectra:spectra.transform")
  public static void transform(long[] in, float[] out, int n, FnJ_F op) {
    Launch.call("spectra:spectra.transform", null, new Object[] {in, out, n, op});
  }

  @Interface("spectra:spectra.transform")
  public static void transform(long[] in, double[] out, int n, FnJ_D op) {
    Launch.call("spectra:spectra.transform", null, new Object[] {in, out, n, op});
  }

  @Interface("spectra:spectra.transform")
  public static void transform(float[] in, int[] out, int n, FnF_I op) {
    Launch.call("spectra:spectra.transform", null, new Object[] {in, out, n, op});
  }

  @Interface("spectra:spectra.transform")
  public static void transform(float[] in, long[] out, int n, FnF_J op) {
    Launch.call("spectra:spectra.transform", null, new Object[] {in, out, n, op});
  }

  @Interface("spectra:spectra.transform")
  public static void transform(float[] in, float[] out, int n, FnF_F op) {
    Launch.call("spectra:spectra.transform", null, new Object[] {in, out, n, op});
  }

  @Interface("spectra:spectra.transform")
  public static void transform(float[] in, double[] out, int n, FnF_D op) {
    Launch.call("spectra:spectra.transform", null, new Object[] {in, out, n, op});
  }

  @Interface("spectra:spectra.transform")
  public static void transform(double[] in, int[] out, int n, FnD_I op) {
    Launch.call("spectra:spectra.transform", null, new Object[] {in, out, n, op});
  }

  @Interface("spectra:spectra.transform")
  public static void transform(double[] in, long[] out, int n, FnD_J op) {
    Launch.call("spectra:spectra.transform", null, new Object[] {in, out, n, op});
  }

  @Interface("spectra:spectra.transform")
  public static void transform(double[] in, float[] out, int n, FnD_F op) {
    Launch.call("spectra:spectra.transform", null, new Object[] {in, out, n, op});
  }

  @Interface("spectra:spectra.transform")
  public static void transform(double[] in, double[] out, int n, FnD_D op) {
    Launch.call("spectra:spectra.transform", null, new Object[] {in, out, n, op});
  }

  @Interface("spectra:spectra.transform")
  public static <T, U> void transform(T[] in, U[] out, int n, FnL_L<T, U> op) {
    Launch.call("spectra:spectra.transform", null, new Object[] {in, out, n, op});
  }

  @Interface("spectra:spectra.transform_binary")
  public static void transform_binary(int[] a, int[] b, int[] out, int n, FnII_I op) {
    Launch.call("spectra:spectra.transform_binary", null, new Object[] {a, b, out, n, op});
  }

  @Interface("spectra:spectra.transform_binary")
  public static void transform_binary(int[] a, int[] b, long[] out, int n, FnII_J op) {
    Launch.call("spectra:spectra.transform_binary", null, new Object[] {a, b, out, n, op});
  }

  @Interface("spectra:spectra.transform_binary")
  public static void transform_binary(int[] a, int[] b, float[] out, int n, FnII_F op) {
    Launch.call("spectra:spectra.transform_binary", null, new Object[] {a, b, out, n, op});
  }

  @Interface("spectra:spectra.transform_binary")
  public static void transform_binary(int[] a, int[] b, double[] out, int n, FnII_D op) {
    Launch.call("spectra:spectra.transform_binary", null, new Object[] {a, b, out, n, op});
  }

  @Interface("spectra:spectra.transform_binary")
  public static void transform_binary(int[] a, long[] b, int[] out, int n, FnIJ_I op) {
    Launch.call("spectra:spectra.transform_binary", null, new Object[] {a, b, out, n, op});
  }

  @Interface("spectra:spectra.transform_binary")
  public static void transform_binary(int[] a, long[] b, long[] out, int n, FnIJ_J op) {
    Launch.call("spectra:spectra.transform_binary", null, new Object[] {a, b, out, n, op});
  }

  @Interface("spectra:spectra.transform_binary")
  public static void transform_binary(int[] a, long[] b, float[] out, int n, FnIJ_F op) {
    Launch.call("spectra:spectra.transform_binary", null, new Object[] {a, b, out, n, op});
  }

  @Interface("spectra:spectra.transform_binary")
  public static void transform_binary(int[] a, long[] b, double[] out, int n, FnIJ_D op) {
    Launch.call("spectra:spectra.transform_binary", null, new Object[] {a, b, out, n, op});
  }

  @Interface("spectra:spectra.transform_binary")
  public static void transform_binary(int[] a, float[] b, int[] out, int n, FnIF_I op) {
    Launch.call("spectra:spectra.transform_binary", null, new Object[] {a, b, out, n, op});
  }

  @Interface("spectra:spectra.transform_binary")
  public static void transform_binary(int[] a, float[] b, long[] out, int n, FnIF_J op) {
    Launch.call("spectra:spectra.transform_binary", null, new Object[] {a, b, out, n, op});
  }

  @Interface("spectra:spectra.transform_binary")
  public static void transform_binary(int[] a, float[] b, float[] out, int n, FnIF_F op) {
    Launch.call("spectra:spectra.transform_binary", null, new Object[] {a, b, out, n, op});
  }

  @Interface("spectra:spectra.transform_binary")
  public static void transform_binary(int[] a, float[] b, double[] out, int n, FnIF_D op) {
    Launch.call("spectra:spectra.transform_binary", null, new Object[] {a, b, out, n, op});
  }

  @Interface("spectra:spectra.transform_binary")
  public static void transform_binary(int[] a, double[] b, int[] out, int n, FnID_I op) {
    Launch.call("spectra:spectra.transform_binary", null, new Object[] {a, b, out, n, op});
  }

  @Interface("spectra:spectra.transform_binary")
  public static void transform_binary(int[] a, double[] b, long[] out, int n, FnID_J op) {
    Launch.call("spectra:spectra.transform_binary", null, new Object[] {a, b, out, n, op});
  }

  @Interface("spectra:spectra.transform_binary")
  public static void transform_binary(int[] a, double[] b, float[] out, int n, FnID_F op) {
    Launch.call("spectra:spectra.transform_binary", null, new Object[] {a, b, out, n, op});
  }

  @Interface("spectra:spectra.transform_binary")
  public static void transform_binary(int[] a, double[] b, double[] out, int n, FnID_D op) {
    Launch.call("spectra:spectra.transform_binary", null, new Object[] {a, b, out, n, op});
  }

  @Interface("spectra:spectra.transform_binary")
  public static void transform_binary(long[] a, int[] b, int[] out, int n, FnJI_I op) {
    Launch.call("spectra:spectra.transform_binary", null, new Object[] {a, b, out, n, op});
  }

  @Interface("spectra:spectra.transform_binary")
  public static void transform_binary(long[] a, int[] b, long[] out, int n, FnJI_J op) {
    Launch.call("spectra:spectra.transform_binary", null, new Object[] {a, b, out, n, op});
  }

  @Interface("spectra:spectra.transform_binary")
  public static void transform_binary(long[] a, int[] b, float[] out, int n, FnJI_F op) {
    Launch.call("spectra:spectra.transform_binary", null, new Object[] {a, b, out, n, op});
  }

  @Interface("spectra:spectra.transform_binary")
  public static void transform_binary(long[] a, int[] b, double[] out, int n, FnJI_D op) {
    Launch.call("spectra:spectra.transform_binary", null, new Object[] {a, b, out, n, op});
  }

  @Interface("spectra:spectra.transform_binary")
  public static void transform_binary(long[] a, long[] b, int[] out, int n, FnJJ_I op) {
    Launch.call("spectra:spectra.transform_binary", null, new Object[] {a, b, out, n, op});
  }

  @Interface("spectra:spectra.transform_binary")
  public static void transform_binary(long[] a, long[] b, long[] out, int n, FnJJ_J op) {
    Launch.call("spectra:spectra.transform_binary", null, new Object[] {a, b, out, n, op});
  }

  @Interface("spectra:spectra.transform_binary")
  public static void transform_binary(long[] a, long[] b, float[] out, int n, FnJJ_F op) {
    Launch.call("spectra:spectra.transform_binary", null, new Object[] {a, b, out, n, op});
  }

  @Interface("spectra:spectra.transform_binary")
  public static void transform_binary(long[] a, long[] b, double[] out, int n, FnJJ_D op) {
    Launch.call("spectra:spectra.transform_binary", null, new Object[] {a, b, out, n, op});
  }

  @Interface("spectra:spectra.transform_binary")
  public static void transform_binary(long[] a, float[] b, int[] out, int n, FnJF_I op) {
    Launch.call("spectra:spectra.transform_binary", null, new Object[] {a, b, out, n, op});
  }

  @Interface("spectra:spectra.transform_binary")
  public static void transform_binary(long[] a, float[] b, long[] out, int n, FnJF_J op) {
    Launch.call("spectra:spectra.transform_binary", null, new Object[] {a, b, out, n, op});
  }

  @Interface("spectra:spectra.transform_binary")
  public static void transform_binary(long[] a, float[] b, float[] out, int n, FnJF_F op) {
    Launch.call("spectra:spectra.transform_binary", null, new Object[] {a, b, out, n, op});
  }

  @Interface("spectra:spectra.transform_binary")
  public static void transform_binary(long[] a, float[] b, double[] out, int n, FnJF_D op) {
    Launch.call("spectra:spectra.transform_binary", null, new Object[] {a, b, out, n, op});
  }

  @Interface("spectra:spectra.transform_binary")
  public static void transform_binary(long[] a, double[] b, int[] out, int n, FnJD_I op) {
    Launch.call("spectra:spectra.transform_binary", null, new Object[] {a, b, out, n, op});
  }

  @Interface("spectra:spectra.transform_binary")
  public static void transform_binary(long[] a, double[] b, long[] out, int n, FnJD_J op) {
    Launch.call("spectra:spectra.transform_binary", null, new Object[] {a, b, out, n, op});
  }

  @Interface("spectra:spectra.transform_binary")
  public static void transform_binary(long[] a, double[] b, float[] out, int n, FnJD_F op) {
    Launch.call("spectra:spectra.transform_binary", null, new Object[] {a, b, out, n, op});
  }

  @Interface("spectra:spectra.transform_binary")
  public static void transform_binary(long[] a, double[] b, double[] out, int n, FnJD_D op) {
    Launch.call("spectra:spectra.transform_binary", null, new Object[] {a, b, out, n, op});
  }

  @Interface("spectra:spectra.transform_binary")
  public static void transform_binary(float[] a, int[] b, int[] out, int n, FnFI_I op) {
    Launch.call("spectra:spectra.transform_binary", null, new Object[] {a, b, out, n, op});
  }

  @Interface("spectra:spectra.transform_binary")
  public static void transform_binary(float[] a, int[] b, long[] out, int n, FnFI_J op) {
    Launch.call("spectra:spectra.transform_binary", null, new Object[] {a, b, out, n, op});
  }

  @Interface("spectra:spectra.transform_binary")
  public static void transform_binary(float[] a, int[] b, float[] out, int n, FnFI_F op) {
    Launch.call("spectra:spectra.transform_binary", null, new Object[] {a, b, out, n, op});
  }

  @Interface("spectra:spectra.transform_binary")
  public static void transform_binary(float[] a, int[] b, double[] out, int n, FnFI_D op) {
    Launch.call("spectra:spectra.transform_binary", null, new Object[] {a, b, out, n, op});
  }

  @Interface("spectra:spectra.transform_binary")
  public static void transform_binary(float[] a, long[] b, int[] out, int n, FnFJ_I op) {
    Launch.call("spectra:spectra.transform_binary", null, new Object[] {a, b, out, n, op});
  }

  @Interface("spectra:spectra.transform_binary")
  public static void transform_binary(float[] a, long[] b, long[] out, int n, FnFJ_J op) {
    Launch.call("spectra:spectra.transform_binary", null, new Object[] {a, b, out, n, op});
  }

  @Interface("spectra:spectra.transform_binary")
  public static void transform_binary(float[] a, long[] b, float[] out, int n, FnFJ_F op) {
    Launch.call("spectra:spectra.transform_binary", null, new Object[] {a, b, out, n, op});
  }

  @Interface("spectra:spectra.transform_binary")
  public static void transform_binary(float[] a, long[] b, double[] out, int n, FnFJ_D op) {
    Launch.call("spectra:spectra.transform_binary", null, new Object[] {a, b, out, n, op});
  }

  @Interface("spectra:spectra.transform_binary")
  public static void transform_binary(float[] a, float[] b, int[] out, int n, FnFF_I op) {
    Launch.call("spectra:spectra.transform_binary", null, new Object[] {a, b, out, n, op});
  }

  @Interface("spectra:spectra.transform_binary")
  public static void transform_binary(float[] a, float[] b, long[] out, int n, FnFF_J op) {
    Launch.call("spectra:spectra.transform_binary", null, new Object[] {a, b, out, n, op});
  }

  @Interface("spectra:spectra.transform_binary")
  public static void transform_binary(float[] a, float[] b, float[] out, int n, FnFF_F op) {
    Launch.call("spectra:spectra.transform_binary", null, new Object[] {a, b, out, n, op});
  }

  @Interface("spectra:spectra.transform_binary")
  public static void transform_binary(float[] a, float[] b, double[] out, int n, FnFF_D op) {
    Launch.call("spectra:spectra.transform_binary", null, new Object[] {a, b, out, n, op});
  }

  @Interface("spectra:spectra.transform_binary")
  public static void transform_binary(float[] a, double[] b, int[] out, int n, FnFD_I op) {
    Launch.call("spectra:spectra.transform_binary", null, new Object[] {a, b, out, n, op});
  }

  @Interface("spectra:spectra.transform_binary")
  public static void transform_binary(float[] a, double[] b, long[] out, int n, FnFD_J op) {
    Launch.call("spectra:spectra.transform_binary", null, new Object[] {a, b, out, n, op});
  }

  @Interface("spectra:spectra.transform_binary")
  public static void transform_binary(float[] a, double[] b, float[] out, int n, FnFD_F op) {
    Launch.call("spectra:spectra.transform_binary", null, new Object[] {a, b, out, n, op});
  }

  @Interface("spectra:spectra.transform_binary")
  public static void transform_binary(float[] a, double[] b, double[] out, int n, FnFD_D op) {
    Launch.call("spectra:spectra.transform_binary", null, new Object[] {a, b, out, n, op});
  }

  @Interface("spectra:spectra.transform_binary")
  public static void transform_binary(double[] a, int[] b, int[] out, int n, FnDI_I op) {
    Launch.call("spectra:spectra.transform_binary", null, new Object[] {a, b, out, n, op});
  }

  @Interface("spectra:spectra.transform_binary")
  public static void transform_binary(double[] a, int[] b, long[] out, int n, FnDI_J op) {
    Launch.call("spectra:spectra.transform_binary", null, new Object[] {a, b, out, n, op});
  }

  @Interface("spectra:spectra.transform_binary")
  public static void transform_binary(double[] a, int[] b, float[] out, int n, FnDI_F op) {
    Launch.call("spectra:spectra.transform_binary", null, new Object[] {a, b, out, n, op});
  }

  @Interface("spectra:spectra.transform_binary")
  public static void transform_binary(double[] a, int[] b, double[] out, int n, FnDI_D op) {
    Launch.call("spectra:spectra.transform_binary", null, new Object[] {a, b, out, n, op});
  }

  @Interface("spectra:spectra.transform_binary")
  public static void transform_binary(double[] a, long[] b, int[] out, int n, FnDJ_I op) {
    Launch.call("spectra:spectra.transform_binary", null, new Object[] {a, b, out, n, op});
  }

  @Interface("spectra:spectra.transform_binary")
  public static void transform_binary(double[] a, long[] b, long[] out, int n, FnDJ_J op) {
    Launch.call("spectra:spectra.transform_binary", null, new Object[] {a, b, out, n, op});
  }

  @Interface("spectra:spectra.transform_binary")
  public static void transform_binary(double[] a, long[] b, float[] out, int n, FnDJ_F op) {
    Launch.call("spectra:spectra.transform_binary", null, new Object[] {a, b, out, n, op});
  }

  @Interface("spectra:spectra.transform_binary")
  public static void transform_binary(double[] a, long[] b, double[] out, int n, FnDJ_D op) {
    Launch.call("spectra:spectra.transform_binary", null, new Object[] {a, b, out, n, op});
  }

  @Interface("spectra:spectra.transform_binary")
  public static void transform_binary(double[] a, float[] b, int[] out, int n, FnDF_I op) {
    Launch.call("spectra:spectra.transform_binary", null, new Object[] {a, b, out, n, op});
  }

  @Interface("spectra:spectra.transform_binary")
  public static void transform_binary(double[] a, float[] b, long[] out, int n, FnDF_J op) {
    Launch.call("spectra:spectra.transform_binary", null, new Object[] {a, b, out, n, op});
  }

  @Interface("spectra:spectra.transform_binary")
  public static void transform_binary(double[] a, float[] b, float[] out, int n, FnDF_F op) {
    Launch.call("spectra:spectra.transform_binary", null, new Object[] {a, b, out, n, op});
  }

  @Interface("spectra:spectra.transform_binary")
  public static void transform_binary(double[] a, float[] b, double[] out, int n, FnDF_D op) {
    Launch.call("spectra:spectra.transform_binary", null, new Object[] {a, b, out, n, op});
  }

  @Interface("spectra:spectra.transform_binary")
  public static void transform_binary(double[] a, double[] b, int[] out, int n, FnDD_I op) {
    Launch.call("spectra:spectra.transform_binary", null, new Object[] {a, b, out, n, op});
  }

  @Interface("spectra:spectra.transform_binary")
  public static void transform_binary(double[] a, double[] b, long[] out, int n, FnDD_J op) {
    Launch.call("spectra:spectra.transform_binary", null, new Object[] {a, b, out, n, op});
  }

  @Interface("spectra:spectra.transform_binary")
  public static void transform_binary(double[] a, double[] b, float[] out, int n, FnDD_F op) {
    Launch.call("spectra:spectra.transform_binary", null, new Object[] {a, b, out, n, op});
  }

  @Interface("spectra:spectra.transform_binary")
  public static void transform_binary(double[] a, double[] b, double[] out, int n, FnDD_D op) {
    Launch.call("spectra:spectra.transform_binary", null, new Object[] {a, b, out, n, op});
  }

  @Interface("spectra:spectra.transform_binary")
  public static <T, U, V> void transform_binary(T[] a, U[] b, V[] out, int n, FnLL_L<T, U, V> op) {
    Launch.call("spectra:spectra.transform_binary", null, new Object[] {a, b, out, n, op});
  }

  @Interface("spectra:spectra.transform_exclusive_scan")
  public static void transform_exclusive_scan(
      int[] in,
      int[] out,
      int n,
      int init,
      FnI_I map,
      FnII_I op) {
    Launch.call("spectra:spectra.transform_exclusive_scan", null, new Object[] {in, out, n, init, map, op});
  }

  @Interface("spectra:spectra.transform_exclusive_scan")
  public static void transform_exclusive_scan(
      int[] in,
      long[] out,
      int n,
      long init,
      FnI_J map,
      FnJJ_J op) {
    Launch.call("spectra:spectra.transform_exclusive_scan", null, new Object[] {in, out, n, init, map, op});
  }

  @Interface("spectra:spectra.transform_exclusive_scan")
  public static void transform_exclusive_scan(
      int[] in,
      float[] out,
      int n,
      float init,
      FnI_F map,
      FnFF_F op) {
    Launch.call("spectra:spectra.transform_exclusive_scan", null, new Object[] {in, out, n, init, map, op});
  }

  @Interface("spectra:spectra.transform_exclusive_scan")
  public static void transform_exclusive_scan(
      int[] in,
      double[] out,
      int n,
      double init,
      FnI_D map,
      FnDD_D op) {
    Launch.call("spectra:spectra.transform_exclusive_scan", null, new Object[] {in, out, n, init, map, op});
  }

  @Interface("spectra:spectra.transform_exclusive_scan")
  public static void transform_exclusive_scan(
      long[] in,
      int[] out,
      int n,
      int init,
      FnJ_I map,
      FnII_I op) {
    Launch.call("spectra:spectra.transform_exclusive_scan", null, new Object[] {in, out, n, init, map, op});
  }

  @Interface("spectra:spectra.transform_exclusive_scan")
  public static void transform_exclusive_scan(
      long[] in,
      long[] out,
      int n,
      long init,
      FnJ_J map,
      FnJJ_J op) {
    Launch.call("spectra:spectra.transform_exclusive_scan", null, new Object[] {in, out, n, init, map, op});
  }

  @Interface("spectra:spectra.transform_exclusive_scan")
  public static void transform_exclusive_scan(
      long[] in,
      float[] out,
      int n,
      float init,
      FnJ_F map,
      FnFF_F op) {
    Launch.call("spectra:spectra.transform_exclusive_scan", null, new Object[] {in, out, n, init, map, op});
  }

  @Interface("spectra:spectra.transform_exclusive_scan")
  public static void transform_exclusive_scan(
      long[] in,
      double[] out,
      int n,
      double init,
      FnJ_D map,
      FnDD_D op) {
    Launch.call("spectra:spectra.transform_exclusive_scan", null, new Object[] {in, out, n, init, map, op});
  }

  @Interface("spectra:spectra.transform_exclusive_scan")
  public static void transform_exclusive_scan(
      float[] in,
      int[] out,
      int n,
      int init,
      FnF_I map,
      FnII_I op) {
    Launch.call("spectra:spectra.transform_exclusive_scan", null, new Object[] {in, out, n, init, map, op});
  }

  @Interface("spectra:spectra.transform_exclusive_scan")
  public static void transform_exclusive_scan(
      float[] in,
      long[] out,
      int n,
      long init,
      FnF_J map,
      FnJJ_J op) {
    Launch.call("spectra:spectra.transform_exclusive_scan", null, new Object[] {in, out, n, init, map, op});
  }

  @Interface("spectra:spectra.transform_exclusive_scan")
  public static void transform_exclusive_scan(
      float[] in,
      float[] out,
      int n,
      float init,
      FnF_F map,
      FnFF_F op) {
    Launch.call("spectra:spectra.transform_exclusive_scan", null, new Object[] {in, out, n, init, map, op});
  }

  @Interface("spectra:spectra.transform_exclusive_scan")
  public static void transform_exclusive_scan(
      float[] in,
      double[] out,
      int n,
      double init,
      FnF_D map,
      FnDD_D op) {
    Launch.call("spectra:spectra.transform_exclusive_scan", null, new Object[] {in, out, n, init, map, op});
  }

  @Interface("spectra:spectra.transform_exclusive_scan")
  public static void transform_exclusive_scan(
      double[] in,
      int[] out,
      int n,
      int init,
      FnD_I map,
      FnII_I op) {
    Launch.call("spectra:spectra.transform_exclusive_scan", null, new Object[] {in, out, n, init, map, op});
  }

  @Interface("spectra:spectra.transform_exclusive_scan")
  public static void transform_exclusive_scan(
      double[] in,
      long[] out,
      int n,
      long init,
      FnD_J map,
      FnJJ_J op) {
    Launch.call("spectra:spectra.transform_exclusive_scan", null, new Object[] {in, out, n, init, map, op});
  }

  @Interface("spectra:spectra.transform_exclusive_scan")
  public static void transform_exclusive_scan(
      double[] in,
      float[] out,
      int n,
      float init,
      FnD_F map,
      FnFF_F op) {
    Launch.call("spectra:spectra.transform_exclusive_scan", null, new Object[] {in, out, n, init, map, op});
  }

  @Interface("spectra:spectra.transform_exclusive_scan")
  public static void transform_exclusive_scan(
      double[] in,
      double[] out,
      int n,
      double init,
      FnD_D map,
      FnDD_D op) {
    Launch.call("spectra:spectra.transform_exclusive_scan", null, new Object[] {in, out, n, init, map, op});
  }

  @Interface("spectra:spectra.transform_exclusive_scan")
  public static <T, U> void transform_exclusive_scan(
      T[] in,
      U[] out,
      int n,
      U init,
      FnL_L<T, U> map,
      FnLL_L<U, U, U> op) {
    Launch.call("spectra:spectra.transform_exclusive_scan", null, new Object[] {in, out, n, init, map, op});
  }

  @Interface("spectra:spectra.transform_inclusive_scan")
  public static void transform_inclusive_scan(int[] in, int[] out, int n, FnI_I map, FnII_I op) {
    Launch.call("spectra:spectra.transform_inclusive_scan", null, new Object[] {in, out, n, map, op});
  }

  @Interface("spectra:spectra.transform_inclusive_scan")
  public static void transform_inclusive_scan(int[] in, long[] out, int n, FnI_J map, FnJJ_J op) {
    Launch.call("spectra:spectra.transform_inclusive_scan", null, new Object[] {in, out, n, map, op});
  }

  @Interface("spectra:spectra.transform_inclusive_scan")
  public static void transform_inclusive_scan(int[] in, float[] out, int n, FnI_F map, FnFF_F op) {
    Launch.call("spectra:spectra.transform_inclusive_scan", null, new Object[] {in, out, n, map, op});
  }

  @Interface("spectra:spectra.transform_inclusive_scan")
  public static void transform_inclusive_scan(int[] in, double[] out, int n, FnI_D map, FnDD_D op) {
    Launch.call("spectra:spectra.transform_inclusive_scan", null, new Object[] {in, out, n, map, op});
  }

  @Interface("spectra:spectra.transform_inclusive_scan")
  public static void transform_inclusive_scan(long[] in, int[] out, int n, FnJ_I map, FnII_I op) {
    Launch.call("spectra:spectra.transform_inclusive_scan", null, new Object[] {in, out, n, map, op});
  }

  @Interface("spectra:spectra.transform_inclusive_scan")
  public static void transform_inclusive_scan(long[] in, long[] out, int n, FnJ_J map, FnJJ_J op) {
    Launch.call("spectra:spectra.transform_inclusive_scan", null, new Object[] {in, out, n, map, op});
  }

  @Interface("spectra:spectra.transform_inclusive_scan")
  public static void transform_inclusive_scan(long[] in, float[] out, int n, FnJ_F map, FnFF_F op) {
    Launch.call("spectra:spectra.transform_inclusive_scan", null, new Object[] {in, out, n, map, op});
  }

  @Interface("spectra:spectra.transform_inclusive_scan")
  public static void transform_inclusive_scan(
      long[] in,
      double[] out,
      int n,
      FnJ_D map,
      FnDD_D op) {
    Launch.call("spectra:spectra.transform_inclusive_scan", null, new Object[] {in, out, n, map, op});
  }

  @Interface("spectra:spectra.transform_inclusive_scan")
  public static void transform_inclusive_scan(float[] in, int[] out, int n, FnF_I map, FnII_I op) {
    Launch.call("spectra:spectra.transform_inclusive_scan", null, new Object[] {in, out, n, map, op});
  }

  @Interface("spectra:spectra.transform_inclusive_scan")
  public static void transform_inclusive_scan(float[] in, long[] out, int n, FnF_J map, FnJJ_J op) {
    Launch.call("spectra:spectra.transform_inclusive_scan", null, new Object[] {in, out, n, map, op});
  }

  @Interface("spectra:spectra.transform_inclusive_scan")
  public static void transform_inclusive_scan(
      float[] in,
      float[] out,
      int n,
      FnF_F map,
      FnFF_F op) {
    Launch.call("spectra:spectra.transform_inclusive_scan", null, new Object[] {in, out, n, map, op});
  }

  @Interface("spectra:spectra.transform_inclusive_scan")
  public static void transform_inclusive_scan(
      float[] in,
      double[] out,
      int n,
      FnF_D map,
      FnDD_D op) {
    Launch.call("spectra:spectra.transform_inclusive_scan", null, new Object[] {in, out, n, map, op});
  }

  @Interface("spectra:spectra.transform_inclusive_scan")
  public static void transform_inclusive_scan(double[] in, int[] out, int n, FnD_I map, FnII_I op) {
    Launch.call("spectra:spectra.transform_inclusive_scan", null, new Object[] {in, out, n, map, op});
  }

  @Interface("spectra:spectra.transform_inclusive_scan")
  public static void transform_inclusive_scan(
      double[] in,
      long[] out,
      int n,
      FnD_J map,
      FnJJ_J op) {
    Launch.call("spectra:spectra.transform_inclusive_scan", null, new Object[] {in, out, n, map, op});
  }

  @Interface("spectra:spectra.transform_inclusive_scan")
  public static void transform_inclusive_scan(
      double[] in,
      float[] out,
      int n,
      FnD_F map,
      FnFF_F op) {
    Launch.call("spectra:spectra.transform_inclusive_scan", null, new Object[] {in, out, n, map, op});
  }

  @Interface("spectra:spectra.transform_inclusive_scan")
  public static void transform_inclusive_scan(
      double[] in,
      double[] out,
      int n,
      FnD_D map,
      FnDD_D op) {
    Launch.call("spectra:spectra.transform_inclusive_scan", null, new Object[] {in, out, n, map, op});
  }

  @Interface("spectra:spectra.transform_inclusive_scan")
  public static <T, U> void transform_inclusive_scan(
      T[] in,
      U[] out,
      int n,
      FnL_L<T, U> map,
      FnLL_L<U, U, U> op) {
    Launch.call("spectra:spectra.transform_inclusive_scan", null, new Object[] {in, out, n, map, op});
  }

  @Interface("spectra:spectra.transform_reduce")
  public static int transform_reduce(int[] in, int n, Integer init, FnI_I map, FnII_I op) {
    return (Integer) Launch.call("spectra:spectra.transform_reduce", null, new Object[] {in, n, init, map, op});
  }

  @Interface("spectra:spectra.transform_reduce")
  public static long transform_reduce(int[] in, int n, Long init, FnI_J map, FnJJ_J op) {
    return (Long) Launch.call("spectra:spectra.transform_reduce", null, new Object[] {in, n, init, map, op});
  }

  @Interface("spectra:spectra.transform_reduce")
  public static float transform_reduce(int[] in, int n, Float init, FnI_F map, FnFF_F op) {
    return (Float) Launch.call("spectra:spectra.transform_reduce", null, new Object[] {in, n, init, map, op});
  }

  @Interface("spectra:spectra.transform_reduce")
  public static double transform_reduce(int[] in, int n, Double init, FnI_D map, FnDD_D op) {
    return (Double) Launch.call("spectra:spectra.transform_reduce", null, new Object[] {in, n, init, map, op});
  }

  @Interface("spectra:spectra.transform_reduce")
  public static int transform_reduce(long[] in, int n, Integer init, FnJ_I map, FnII_I op) {
    return (Integer) Launch.call("spectra:spectra.transform_reduce", null, new Object[] {in, n, init, map, op});
  }

  @Interface("spectra:spectra.transform_reduce")
  public static long transform_reduce(long[] in, int n, Long init, FnJ_J map, FnJJ_J op) {
    return (Long) Launch.call("spectra:spectra.transform_reduce", null, new Object[] {in, n, init, map, op});
  }

  @Interface("spectra:spectra.transform_reduce")
  public static float transform_reduce(long[] in, int n, Float init, FnJ_F map, FnFF_F op) {
    return (Float) Launch.call("spectra:spectra.transform_reduce", null, new Object[] {in, n, init, map, op});
  }

  @Interface("spectra:spectra.transform_reduce")
  public static double transform_reduce(long[] in, int n, Double init, FnJ_D map, FnDD_D op) {
    return (Double) Launch.call("spectra:spectra.transform_reduce", null, new Object[] {in, n, init, map, op});
  }

  @Interface("spectra:spectra.transform_reduce")
  public static int transform_reduce(float[] in, int n, Integer init, FnF_I map, FnII_I op) {
    return (Integer) Launch.call("spectra:spectra.transform_reduce", null, new Object[] {in, n, init, map, op});
  }

  @Interface("spectra:spectra.transform_reduce")
  public static long transform_reduce(float[] in, int n, Long init, FnF_J map, FnJJ_J op) {
    return (Long) Launch.call("spectra:spectra.transform_reduce", null, new Object[] {in, n, init, map, op});
  }

  @Interface("spectra:spectra.transform_reduce")
  public static float transform_reduce(float[] in, int n, Float init, FnF_F map, FnFF_F op) {
    return (Float) Launch.call("spectra:spectra.transform_reduce", null, new Object[] {in, n, init, map, op});
  }

  @Interface("spectra:spectra.transform_reduce")
  public static double transform_reduce(float[] in, int n, Double init, FnF_D map, FnDD_D op) {
    return (Double) Launch.call("spectra:spectra.transform_reduce", null, new Object[] {in, n, init, map, op});
  }

  @Interface("spectra:spectra.transform_reduce")
  public static int transform_reduce(double[] in, int n, Integer init, FnD_I map, FnII_I op) {
    return (Integer) Launch.call("spectra:spectra.transform_reduce", null, new Object[] {in, n, init, map, op});
  }

  @Interface("spectra:spectra.transform_reduce")
  public static long transform_reduce(double[] in, int n, Long init, FnD_J map, FnJJ_J op) {
    return (Long) Launch.call("spectra:spectra.transform_reduce", null, new Object[] {in, n, init, map, op});
  }

  @Interface("spectra:spectra.transform_reduce")
  public static float transform_reduce(double[] in, int n, Float init, FnD_F map, FnFF_F op) {
    return (Float) Launch.call("spectra:spectra.transform_reduce", null, new Object[] {in, n, init, map, op});
  }

  @Interface("spectra:spectra.transform_reduce")
  public static double transform_reduce(double[] in, int n, Double init, FnD_D map, FnDD_D op) {
    return (Double) Launch.call("spectra:spectra.transform_reduce", null, new Object[] {in, n, init, map, op});
  }

  @Interface("spectra:spectra.transform_reduce")
  @SuppressWarnings("unchecked")
  public static <T, U> U transform_reduce(
      T[] in,
      int n,
      U init,
      FnL_L<T, U> map,
      FnLL_L<U, U, U> op) {
    return (U) Launch.call("spectra:spectra.transform_reduce", null, new Object[] {in, n, init, map, op});
  }

  @Interface("spectra:spectra.transform_reduce_unordered")
  public static int transform_reduce_unordered(
      int[] in,
      int n,
      Integer init,
      FnI_I map,
      FnII_I op) {
    return (Integer) Launch.call("spectra:spectra.transform_reduce_unordered", null, new Object[] {in, n, init, map, op});
  }

  @Interface("spectra:spectra.transform_reduce_unordered")
  public static long transform_reduce_unordered(int[] in, int n, Long init, FnI_J map, FnJJ_J op) {
    return (Long) Launch.call("spectra:spectra.transform_reduce_unordered", null, new Object[] {in, n, init, map, op});
  }

  @Interface("spectra:spectra.transform_reduce_unordered")
  public static float transform_reduce_unordered(
      int[] in,
      int n,
      Float init,
      FnI_F map,
      FnFF_F op) {
    return (Float) Launch.call("spectra:spectra.transform_reduce_unordered", null, new Object[] {in, n, init, map, op});
  }

  @Interface("spectra:spectra.transform_reduce_unordered")
  public static double transform_reduce_unordered(
      int[] in,
      int n,
      Double init,
      FnI_D map,
      FnDD_D op) {
    return (Double) Launch.call("spectra:spectra.transform_reduce_unordered", null, new Object[] {in, n, init, map, op});
  }

  @Interface("spectra:spectra.transform_reduce_unordered")
  public static int transform_reduce_unordered(
      long[] in,
      int n,
      Integer init,
      FnJ_I map,
      FnII_I op) {
    return (Integer) Launch.call("spectra:spectra.transform_reduce_unordered", null, new Object[] {in, n, init, map, op});
  }

  @Interface("spectra:spectra.transform_reduce_unordered")
  public static long transform_reduce_unordered(long[] in, int n, Long init, FnJ_J map, FnJJ_J op) {
    return (Long) Launch.call("spectra:spectra.transform_reduce_unordered", null, new Object[] {in, n, init, map, op});
  }

  @Interface("spectra:spectra.transform_reduce_unordered")
  public static float transform_reduce_unordered(
      long[] in,
      int n,
      Float init,
      FnJ_F map,
      FnFF_F op) {
    return (Float) Launch.call("spectra:spectra.transform_reduce_unordered", null, new Object[] {in, n, init, map, op});
  }

  @Interface("spectra:spectra.transform_reduce_unordered")
  public static double transform_reduce_unordered(
      long[] in,
      int n,
      Double init,
      FnJ_D map,
      FnDD_D op) {
    return (Double) Launch.call("spectra:spectra.transform_reduce_unordered", null, new Object[] {in, n, init, map, op});
  }

  @Interface("spectra:spectra.transform_reduce_unordered")
  public static int transform_reduce_unordered(
      float[] in,
      int n,
      Integer init,
      FnF_I map,
      FnII_I op) {
    return (Integer) Launch.call("spectra:spectra.transform_reduce_unordered", null, new Object[] {in, n, init, map, op});
  }

  @Interface("spectra:spectra.transform_reduce_unordered")
  public static long transform_reduce_unordered(
      float[] in,
      int n,
      Long init,
      FnF_J map,
      FnJJ_J op) {
    return (Long) Launch.call("spectra:spectra.transform_reduce_unordered", null, new Object[] {in, n, init, map, op});
  }

  @Interface("spectra:spectra.transform_reduce_unordered")
  public static float transform_reduce_unordered(
      float[] in,
      int n,
      Float init,
      FnF_F map,
      FnFF_F op) {
    return (Float) Launch.call("spectra:spectra.transform_reduce_unordered", null, new Object[] {in, n, init, map, op});
  }

  @Interface("spectra:spectra.transform_reduce_unordered")
  public static double transform_reduce_unordered(
      float[] in,
      int n,
      Double init,
      FnF_D map,
      FnDD_D op) {
    return (Double) Launch.call("spectra:spectra.transform_reduce_unordered", null, new Object[] {in, n, init, map, op});
  }

  @Interface("spectra:spectra.transform_reduce_unordered")
  public static int transform_reduce_unordered(
      double[] in,
      int n,
      Integer init,
      FnD_I map,
      FnII_I op) {
    return (Integer) Launch.call("spectra:spectra.transform_reduce_unordered", null, new Object[] {in, n, init, map, op});
  }

  @Interface("spectra:spectra.transform_reduce_unordered")
  public static long transform_reduce_unordered(
      double[] in,
      int n,
      Long init,
      FnD_J map,
      FnJJ_J op) {
    return (Long) Launch.call("spectra:spectra.transform_reduce_unordered", null, new Object[] {in, n, init, map, op});
  }

  @Interface("spectra:spectra.transform_reduce_unordered")
  public static float transform_reduce_unordered(
      double[] in,
      int n,
      Float init,
      FnD_F map,
      FnFF_F op) {
    return (Float) Launch.call("spectra:spectra.transform_reduce_unordered", null, new Object[] {in, n, init, map, op});
  }

  @Interface("spectra:spectra.transform_reduce_unordered")
  public static double transform_reduce_unordered(
      double[] in,
      int n,
      Double init,
      FnD_D map,
      FnDD_D op) {
    return (Double) Launch.call("spectra:spectra.transform_reduce_unordered", null, new Object[] {in, n, init, map, op});
  }

  @Interface("spectra:spectra.transform_reduce_unordered")
  @SuppressWarnings("unchecked")
  public static <T, U> U transform_reduce_unordered(
      T[] in,
      int n,
      U init,
      FnL_L<T, U> map,
      FnLL_L<U, U, U> op) {
    return (U) Launch.call("spectra:spectra.transform_reduce_unordered", null, new Object[] {in, n, init, map, op});
  }

  @Interface("spectra:spectra.unique")
  public static int unique(int[] data, int n, FnII_Z eq) {
    return (Integer) Launch.call("spectra:spectra.unique", null, new Object[] {data, n, eq});
  }

  @Interface("spectra:spectra.unique")
  public static int unique(long[] data, int n, FnJJ_Z eq) {
    return (Integer) Launch.call("spectra:spectra.unique", null, new Object[] {data, n, eq});
  }

  @Interface("spectra:spectra.unique")
  public static int unique(float[] data, int n, FnFF_Z eq) {
    return (Integer) Launch.call("spectra:spectra.unique", null, new Object[] {data, n, eq});
  }

  @Interface("spectra:spectra.unique")
  public static int unique(double[] data, int n, FnDD_Z eq) {
    return (Integer) Launch.call("spectra:spectra.unique", null, new Object[] {data, n, eq});
  }

  @Interface("spectra:spectra.unique")
  public static <T> int unique(T[] data, int n, FnLL_Z<T, T> eq) {
    return (Integer) Launch.call("spectra:spectra.unique", null, new Object[] {data, n, eq});
  }

  @Interface("spectra:spectra.unique_copy")
  public static int unique_copy(int[] in, int n, int[] out, FnII_Z eq) {
    return (Integer) Launch.call("spectra:spectra.unique_copy", null, new Object[] {in, n, out, eq});
  }

  @Interface("spectra:spectra.unique_copy")
  public static int unique_copy(long[] in, int n, long[] out, FnJJ_Z eq) {
    return (Integer) Launch.call("spectra:spectra.unique_copy", null, new Object[] {in, n, out, eq});
  }

  @Interface("spectra:spectra.unique_copy")
  public static int unique_copy(float[] in, int n, float[] out, FnFF_Z eq) {
    return (Integer) Launch.call("spectra:spectra.unique_copy", null, new Object[] {in, n, out, eq});
  }

  @Interface("spectra:spectra.unique_copy")
  public static int unique_copy(double[] in, int n, double[] out, FnDD_Z eq) {
    return (Integer) Launch.call("spectra:spectra.unique_copy", null, new Object[] {in, n, out, eq});
  }

  @Interface("spectra:spectra.unique_copy")
  public static <T> int unique_copy(T[] in, int n, T[] out, FnLL_Z<T, T> eq) {
    return (Integer) Launch.call("spectra:spectra.unique_copy", null, new Object[] {in, n, out, eq});
  }

  @Interface("spectra:spectra.upper_bound")
  public static int upper_bound(int[] in, int n, int value, FnII_Z op) {
    return (Integer) Launch.call("spectra:spectra.upper_bound", null, new Object[] {in, n, value, op});
  }

  @Interface("spectra:spectra.upper_bound")
  public static int upper_bound(long[] in, int n, long value, FnJJ_Z op) {
    return (Integer) Launch.call("spectra:spectra.upper_bound", null, new Object[] {in, n, value, op});
  }

  @Interface("spectra:spectra.upper_bound")
  public static int upper_bound(float[] in, int n, float value, FnFF_Z op) {
    return (Integer) Launch.call("spectra:spectra.upper_bound", null, new Object[] {in, n, value, op});
  }

  @Interface("spectra:spectra.upper_bound")
  public static int upper_bound(double[] in, int n, double value, FnDD_Z op) {
    return (Integer) Launch.call("spectra:spectra.upper_bound", null, new Object[] {in, n, value, op});
  }

  @Interface("spectra:spectra.upper_bound")
  public static <T> int upper_bound(T[] in, int n, T value, FnLL_Z<T, T> op) {
    return (Integer) Launch.call("spectra:spectra.upper_bound", null, new Object[] {in, n, value, op});
  }
}
