package polyregion.javalang;

import java.util.OptionalDouble;
import java.util.OptionalLong;
import java.util.function.LongBinaryOperator;
import java.util.function.LongConsumer;
import java.util.function.LongPredicate;
import java.util.function.LongToDoubleFunction;
import java.util.function.LongToIntFunction;
import java.util.function.LongUnaryOperator;
import polyregion.jvm.runtime.Context;

/** A long pipeline lowered by the compiler; see {@link Par}. */
public final class ParLong {

  private ParLong() {
    throw new AssertionError();
  }

  public ParLong on(Context context) {
    throw Par.unlowered("ParLong.on");
  }

  public ParLong map(LongUnaryOperator f) {
    throw Par.unlowered("ParLong.map");
  }

  public ParLong filter(LongPredicate p) {
    throw Par.unlowered("ParLong.filter");
  }

  public ParInt mapToInt(LongToIntFunction f) {
    throw Par.unlowered("ParLong.mapToInt");
  }

  public ParDouble mapToDouble(LongToDoubleFunction f) {
    throw Par.unlowered("ParLong.mapToDouble");
  }

  public ParLong sorted() {
    throw Par.unlowered("ParLong.sorted");
  }

  public long sum() {
    throw Par.unlowered("ParLong.sum");
  }

  public long count() {
    throw Par.unlowered("ParLong.count");
  }

  public OptionalLong min() {
    throw Par.unlowered("ParLong.min");
  }

  public OptionalLong max() {
    throw Par.unlowered("ParLong.max");
  }

  public OptionalDouble average() {
    throw Par.unlowered("ParLong.average");
  }

  public boolean anyMatch(LongPredicate p) {
    throw Par.unlowered("ParLong.anyMatch");
  }

  public boolean allMatch(LongPredicate p) {
    throw Par.unlowered("ParLong.allMatch");
  }

  public boolean noneMatch(LongPredicate p) {
    throw Par.unlowered("ParLong.noneMatch");
  }

  public long[] toArray() {
    throw Par.unlowered("ParLong.toArray");
  }

  public void forEach(LongConsumer f) {
    throw Par.unlowered("ParLong.forEach");
  }

  public long reduce(long identity, LongBinaryOperator op) {
    throw Par.unlowered("ParLong.reduce");
  }

  public OptionalLong reduce(LongBinaryOperator op) {
    throw Par.unlowered("ParLong.reduce");
  }
}
