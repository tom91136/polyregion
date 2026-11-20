package polyregion.jvm.twin;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

enum Prim {
  BOOL('Z', boolean.class, 1) {
    Object get(ByteBuffer b, int at) {
      return b.get(at) != 0;
    }

    void put(ByteBuffer b, int at, Object v) {
      b.put(at, (byte) ((Boolean) v ? 1 : 0));
    }

    void writeArray(Object xs, ByteBuffer b) {
      boolean[] a = (boolean[]) xs;
      for (int i = 0; i < a.length; i++) b.put(i, (byte) (a[i] ? 1 : 0));
    }

    void readArray(ByteBuffer b, Object xs) {
      boolean[] a = (boolean[]) xs;
      for (int i = 0; i < a.length; i++) a[i] = b.get(i) != 0;
    }
  },
  BYTE('B', byte.class, 1) {
    Object get(ByteBuffer b, int at) {
      return b.get(at);
    }

    void put(ByteBuffer b, int at, Object v) {
      b.put(at, (Byte) v);
    }

    void writeArray(Object xs, ByteBuffer b) {
      b.duplicate().put((byte[]) xs);
    }

    void readArray(ByteBuffer b, Object xs) {
      b.duplicate().get((byte[]) xs);
    }
  },
  CHAR('C', char.class, 2) {
    Object get(ByteBuffer b, int at) {
      return b.getChar(at);
    }

    void put(ByteBuffer b, int at, Object v) {
      b.putChar(at, (Character) v);
    }

    void writeArray(Object xs, ByteBuffer b) {
      b.asCharBuffer().put((char[]) xs);
    }

    void readArray(ByteBuffer b, Object xs) {
      b.asCharBuffer().get((char[]) xs);
    }
  },
  SHORT('S', short.class, 2) {
    Object get(ByteBuffer b, int at) {
      return b.getShort(at);
    }

    void put(ByteBuffer b, int at, Object v) {
      b.putShort(at, (Short) v);
    }

    void writeArray(Object xs, ByteBuffer b) {
      b.asShortBuffer().put((short[]) xs);
    }

    void readArray(ByteBuffer b, Object xs) {
      b.asShortBuffer().get((short[]) xs);
    }
  },
  INT('I', int.class, 4) {
    Object get(ByteBuffer b, int at) {
      return b.getInt(at);
    }

    void put(ByteBuffer b, int at, Object v) {
      b.putInt(at, (Integer) v);
    }

    void writeArray(Object xs, ByteBuffer b) {
      b.asIntBuffer().put((int[]) xs);
    }

    void readArray(ByteBuffer b, Object xs) {
      b.asIntBuffer().get((int[]) xs);
    }
  },
  LONG('J', long.class, 8) {
    Object get(ByteBuffer b, int at) {
      return b.getLong(at);
    }

    void put(ByteBuffer b, int at, Object v) {
      b.putLong(at, (Long) v);
    }

    void writeArray(Object xs, ByteBuffer b) {
      b.asLongBuffer().put((long[]) xs);
    }

    void readArray(ByteBuffer b, Object xs) {
      b.asLongBuffer().get((long[]) xs);
    }
  },
  FLOAT('F', float.class, 4) {
    Object get(ByteBuffer b, int at) {
      return b.getFloat(at);
    }

    void put(ByteBuffer b, int at, Object v) {
      b.putFloat(at, (Float) v);
    }

    void writeArray(Object xs, ByteBuffer b) {
      b.asFloatBuffer().put((float[]) xs);
    }

    void readArray(ByteBuffer b, Object xs) {
      b.asFloatBuffer().get((float[]) xs);
    }
  },
  DOUBLE('D', double.class, 8) {
    Object get(ByteBuffer b, int at) {
      return b.getDouble(at);
    }

    void put(ByteBuffer b, int at, Object v) {
      b.putDouble(at, (Double) v);
    }

    void writeArray(Object xs, ByteBuffer b) {
      b.asDoubleBuffer().put((double[]) xs);
    }

    void readArray(ByteBuffer b, Object xs) {
      b.asDoubleBuffer().get((double[]) xs);
    }
  };

  final char descriptor;
  final Class<?> type;
  final int size;

  Prim(char descriptor, Class<?> type, int size) {
    this.descriptor = descriptor;
    this.type = type;
    this.size = size;
  }

  abstract Object get(ByteBuffer b, int at);

  abstract void put(ByteBuffer b, int at, Object v);

  abstract void writeArray(Object xs, ByteBuffer b);

  abstract void readArray(ByteBuffer b, Object xs);

  static Prim of(Class<?> type) {
    for (Prim p : values()) if (p.type == type) return p;
    return null;
  }

  static Prim ofDescriptor(String descriptor) {
    if (descriptor.length() != 1) return null;
    for (Prim p : values()) if (p.descriptor == descriptor.charAt(0)) return p;
    return null;
  }

  static ByteBuffer native0(ByteBuffer b) {
    return b.duplicate().order(ByteOrder.nativeOrder());
  }
}
