package polyregion.javalang;

import java.util.OptionalDouble;
import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleConsumer;
import java.util.function.DoublePredicate;
import java.util.function.DoubleToIntFunction;
import java.util.function.DoubleToLongFunction;
import java.util.function.DoubleUnaryOperator;
import polyregion.jvm.runtime.Context;

/** A double pipeline lowered by the compiler; see {@link Par}. */
public final class ParDouble {

  private ParDouble() {
    throw new AssertionError();
  }

  public ParDouble on(Context context) {
    throw Par.unlowered("ParDouble.on");
  }

  public ParDouble map(DoubleUnaryOperator f) {
    throw Par.unlowered("ParDouble.map");
  }

  public ParDouble filter(DoublePredicate p) {
    throw Par.unlowered("ParDouble.filter");
  }

  public ParInt mapToInt(DoubleToIntFunction f) {
    throw Par.unlowered("ParDouble.mapToInt");
  }

  public ParLong mapToLong(DoubleToLongFunction f) {
    throw Par.unlowered("ParDouble.mapToLong");
  }

  public ParDouble sorted() {
    throw Par.unlowered("ParDouble.sorted");
  }

  public double sum() {
    throw Par.unlowered("ParDouble.sum");
  }

  public long count() {
    throw Par.unlowered("ParDouble.count");
  }

  public OptionalDouble min() {
    throw Par.unlowered("ParDouble.min");
  }

  public OptionalDouble max() {
    throw Par.unlowered("ParDouble.max");
  }

  public OptionalDouble average() {
    throw Par.unlowered("ParDouble.average");
  }

  public boolean anyMatch(DoublePredicate p) {
    throw Par.unlowered("ParDouble.anyMatch");
  }

  public boolean allMatch(DoublePredicate p) {
    throw Par.unlowered("ParDouble.allMatch");
  }

  public boolean noneMatch(DoublePredicate p) {
    throw Par.unlowered("ParDouble.noneMatch");
  }

  public double[] toArray() {
    throw Par.unlowered("ParDouble.toArray");
  }

  public void forEach(DoubleConsumer f) {
    throw Par.unlowered("ParDouble.forEach");
  }

  public double reduce(double identity, DoubleBinaryOperator op) {
    throw Par.unlowered("ParDouble.reduce");
  }

  public OptionalDouble reduce(DoubleBinaryOperator op) {
    throw Par.unlowered("ParDouble.reduce");
  }
}
