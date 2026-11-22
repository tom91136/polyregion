package polyregion.javalang;

import java.util.OptionalDouble;
import java.util.OptionalInt;
import java.util.function.IntBinaryOperator;
import java.util.function.IntConsumer;
import java.util.function.IntPredicate;
import java.util.function.IntToDoubleFunction;
import java.util.function.IntToLongFunction;
import java.util.function.IntUnaryOperator;
import polyregion.jvm.runtime.Context;

/** A int pipeline lowered by the compiler; see {@link Par}. */
public final class ParInt {

  private ParInt() {
    throw new AssertionError();
  }

  public ParInt on(Context context) {
    throw Par.unlowered("ParInt.on");
  }

  public ParInt map(IntUnaryOperator f) {
    throw Par.unlowered("ParInt.map");
  }

  public ParInt filter(IntPredicate p) {
    throw Par.unlowered("ParInt.filter");
  }

  public ParLong mapToLong(IntToLongFunction f) {
    throw Par.unlowered("ParInt.mapToLong");
  }

  public ParDouble mapToDouble(IntToDoubleFunction f) {
    throw Par.unlowered("ParInt.mapToDouble");
  }

  public ParInt sorted() {
    throw Par.unlowered("ParInt.sorted");
  }

  public int sum() {
    throw Par.unlowered("ParInt.sum");
  }

  public long count() {
    throw Par.unlowered("ParInt.count");
  }

  public OptionalInt min() {
    throw Par.unlowered("ParInt.min");
  }

  public OptionalInt max() {
    throw Par.unlowered("ParInt.max");
  }

  public OptionalDouble average() {
    throw Par.unlowered("ParInt.average");
  }

  public boolean anyMatch(IntPredicate p) {
    throw Par.unlowered("ParInt.anyMatch");
  }

  public boolean allMatch(IntPredicate p) {
    throw Par.unlowered("ParInt.allMatch");
  }

  public boolean noneMatch(IntPredicate p) {
    throw Par.unlowered("ParInt.noneMatch");
  }

  public int[] toArray() {
    throw Par.unlowered("ParInt.toArray");
  }

  public void forEach(IntConsumer f) {
    throw Par.unlowered("ParInt.forEach");
  }

  public int reduce(int identity, IntBinaryOperator op) {
    throw Par.unlowered("ParInt.reduce");
  }

  public OptionalInt reduce(IntBinaryOperator op) {
    throw Par.unlowered("ParInt.reduce");
  }
}
