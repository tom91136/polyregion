package polyregion.javalang;

import java.util.function.DoubleUnaryOperator;
import java.util.function.IntToDoubleFunction;
import java.util.function.IntToLongFunction;
import java.util.function.IntUnaryOperator;
import java.util.function.IntBinaryOperator;
import java.util.function.LongBinaryOperator;
import java.util.function.DoubleBinaryOperator;
import polyregion.jvm.runtime.Context;

/**
 * Parallel pipelines over primitive arrays and ranges that the PolyRegion compiler lowers onto
 * Spectra and the core interface.
 *
 * <p>A chain must be a single expression that ends in a terminal operation, for example {@code
 * Par.ints(xs).map(x -> x * 2).filter(x -> x > 0).sum()}. The compiler replaces the whole
 * expression, so a chain that is stored in a variable, or that the compiler cannot lower, is a
 * compile error naming the operation. The methods of this package never run: reaching one means
 * the compiler did not process the call.
 *
 * <p>Lambdas must be pure. {@code reduce} applies its operation in encounter order with left
 * association, as the JDK does for an ordered stream, so the operation only needs to be
 * associative. {@code sum}, {@code count}, {@code min}, {@code max} and the match operations use
 * any order, which is exact for them. Every element runs through every stage, so {@code anyMatch},
 * {@code allMatch} and {@code noneMatch} evaluate their predicate on all elements rather than
 * stopping at the first decisive one.
 *
 * <p>{@code double} sums and averages use the JDK's compensated summation. Elements are summed in
 * contiguous blocks of {@link Ops#BLOCK}, each block with the JDK's sequential accumulation, and
 * the blocks are combined in encounter order with the JDK's combiner. A chain of at most {@code
 * Ops.BLOCK} elements gives exactly the result of the sequential {@code DoubleStream.sum}; a longer
 * one gives the result of a parallel JDK stream whose leaves are blocks of that size.
 */
public final class Par {

  private Par() {
    throw new AssertionError();
  }

  static UnsupportedOperationException unlowered(String operation) {
    return new UnsupportedOperationException(
        "`"
            + operation
            + "` was not lowered: a .par chain must be a single expression ending in a terminal"
            + " operation, compiled with the polyregion annotation processor");
  }

  public static ParInt ints(int[] xs) {
    throw unlowered("Par.ints");
  }

  public static ParLong longs(long[] xs) {
    throw unlowered("Par.longs");
  }

  public static ParDouble doubles(double[] xs) {
    throw unlowered("Par.doubles");
  }

  public static ParInt range(int lo, int hi) {
    throw unlowered("Par.range");
  }

  public static ParInt rangeClosed(int lo, int hi) {
    throw unlowered("Par.rangeClosed");
  }

  public static ParLong range(long lo, long hi) {
    throw unlowered("Par.range");
  }

  public static ParLong rangeClosed(long lo, long hi) {
    throw unlowered("Par.rangeClosed");
  }

  public static void setAll(int[] xs, IntUnaryOperator f) {
    throw unlowered("Par.setAll");
  }

  public static void setAll(Context context, int[] xs, IntUnaryOperator f) {
    throw unlowered("Par.setAll");
  }

  public static void setAll(long[] xs, IntToLongFunction f) {
    throw unlowered("Par.setAll");
  }

  public static void setAll(Context context, long[] xs, IntToLongFunction f) {
    throw unlowered("Par.setAll");
  }

  public static void setAll(double[] xs, IntToDoubleFunction f) {
    throw unlowered("Par.setAll");
  }

  public static void setAll(Context context, double[] xs, IntToDoubleFunction f) {
    throw unlowered("Par.setAll");
  }

  public static void prefix(int[] xs, IntBinaryOperator op) {
    throw unlowered("Par.prefix");
  }

  public static void prefix(Context context, int[] xs, IntBinaryOperator op) {
    throw unlowered("Par.prefix");
  }

  public static void prefix(long[] xs, LongBinaryOperator op) {
    throw unlowered("Par.prefix");
  }

  public static void prefix(Context context, long[] xs, LongBinaryOperator op) {
    throw unlowered("Par.prefix");
  }

  public static void prefix(double[] xs, DoubleBinaryOperator op) {
    throw unlowered("Par.prefix");
  }

  public static void prefix(Context context, double[] xs, DoubleBinaryOperator op) {
    throw unlowered("Par.prefix");
  }

  public static void sort(int[] xs) {
    throw unlowered("Par.sort");
  }

  public static void sort(Context context, int[] xs) {
    throw unlowered("Par.sort");
  }

  public static void sort(long[] xs) {
    throw unlowered("Par.sort");
  }

  public static void sort(Context context, long[] xs) {
    throw unlowered("Par.sort");
  }

  public static void sort(float[] xs) {
    throw unlowered("Par.sort");
  }

  public static void sort(Context context, float[] xs) {
    throw unlowered("Par.sort");
  }

  public static void sort(double[] xs) {
    throw unlowered("Par.sort");
  }

  public static void sort(Context context, double[] xs) {
    throw unlowered("Par.sort");
  }
}
