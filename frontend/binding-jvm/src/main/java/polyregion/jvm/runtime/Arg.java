package polyregion.jvm.runtime;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Objects;
import java.util.function.Consumer;

public final class Arg<T> {
  public final Type type;
  public final T value;
  final Consumer<ByteBuffer> drain;

  Arg(Type type, T value, Consumer<ByteBuffer> drain) {
    this.type = Objects.requireNonNull(type);
    this.value = value;
    this.drain = Objects.requireNonNull(drain);
  }

  void drainTo(ByteBuffer b) {
    drain.accept(b);
  }

  @Override
  public String toString() {
    return "Arg{" + "type=" + type + ", value=" + value + '}';
  }

  public static Arg<Void> of() {
    return new Arg<>(Type.VOID, null, b -> {});
  }

  public static Arg<Boolean> of(boolean x) {
    return new Arg<>(Type.BOOL, x, bb -> bb.put(x ? (byte) 1 : (byte) 0));
  }

  public static Arg<Byte> of(byte x) {
    return new Arg<>(Type.BYTE, x, bb -> bb.put(x));
  }

  public static Arg<Character> of(char x) {
    return new Arg<>(Type.CHAR, x, bb -> bb.putChar(x));
  }

  public static Arg<Short> of(short x) {
    return new Arg<>(Type.SHORT, x, bb -> bb.putShort(x));
  }

  public static Arg<Integer> of(int x) {
    return new Arg<>(Type.INT, x, bb -> bb.putInt(x));
  }

  public static Arg<Long> of(long x) {
    return new Arg<>(Type.LONG, x, bb -> bb.putLong(x));
  }

  public static Arg<Float> of(float x) {
    return new Arg<>(Type.FLOAT, x, bb -> bb.putFloat(x));
  }

  public static Arg<Double> of(double x) {
    return new Arg<>(Type.DOUBLE, x, bb -> bb.putDouble(x));
  }

  public static Arg<Long> ptr(long x) {
    return new Arg<>(Type.PTR, x, bb -> bb.putLong(x));
  }
}
