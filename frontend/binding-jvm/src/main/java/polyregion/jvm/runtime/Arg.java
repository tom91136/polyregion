package polyregion.jvm.runtime;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Objects;

public final class Arg {
  public final Type type;
  public final ByteBuffer data;

  public Arg(Type type, ByteBuffer data) {
    this.type = Objects.requireNonNull(type);
    this.data = Objects.requireNonNull(data);
  }

  @Override
  public String toString() {
    return "Arg{" + "type=" + type + ", data=" + data + '}';
  }

  private static ByteBuffer allocate(int byteSize) {
    return ByteBuffer.allocate(byteSize).order(ByteOrder.nativeOrder());
  }

  public static Arg of(boolean x) {
    return new Arg(Type.BOOL, allocate(Byte.BYTES).put((byte) (x ? 1 : 0)));
  }

  public static Arg of(byte x) {
    return new Arg(Type.BYTE, allocate(Byte.BYTES).put(x));
  }

  public static Arg of(char x) {
    return new Arg(Type.CHAR, allocate(Byte.BYTES).putChar(x));
  }

  public static Arg of(short x) {
    return new Arg(Type.SHORT, allocate(Short.BYTES).putShort(x));
  }

  public static Arg of(int x) {
    return new Arg(Type.INT, allocate(Integer.BYTES).putInt(x));
  }

  public static Arg of(long x) {
    return new Arg(Type.LONG, allocate(Long.BYTES).putLong(x));
  }

  public static Arg of(float x) {
    return new Arg(Type.FLOAT, allocate(Float.BYTES).putFloat(x));
  }

  public static Arg of(double x) {
    return new Arg(Type.DOUBLE, allocate(Double.BYTES).putDouble(x));
  }
}
