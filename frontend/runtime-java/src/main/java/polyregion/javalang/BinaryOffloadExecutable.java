package polyregion.javalang;

import java.lang.reflect.Array;
import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Base64;
import java.util.List;
import polyregion.jvm.compiler.Target;
import polyregion.jvm.runtime.Context;
import polyregion.jvm.runtime.Dim3;
import polyregion.jvm.runtime.ModuleImage;
import polyregion.jvm.runtime.Policy;
import polyregion.jvm.runtime.Type;
import polyregion.jvm.twin.Marshal;
import polyregion.jvm.twin.Twin;
import polyregion.jvm.twin.TwinException;
import polyregion.jvm.twin.Twins;

public abstract class BinaryOffloadExecutable implements OffloadExecutable {

  public abstract byte[] binary();

  // `<module>|<symbol>|<ordinals>|<result>|<plan>|<layouts>|<twins>|<features>`, see
  // Offloader.descriptorOf
  public abstract String descriptor();

  // chunked as a constant pool utf8 entry caps at 65535 bytes
  public static byte[] decodeBinary(String[] chunks) {
    StringBuilder sb = new StringBuilder();
    for (String chunk : chunks) sb.append(chunk);
    return Base64.getDecoder().decode(sb.toString());
  }

  private static final class Runtime0 {
    static final Context CONTEXT = Context.current();
  }

  // test seam: runs once every capture is packed and before the kernel is launched
  static Runnable afterPack;

  private ModuleImage image;

  private ModuleImage imageOf(String moduleName, String features) {
    ModuleImage cached = image;
    if (cached == null)
      image =
          cached =
              new ModuleImage(
                  moduleName,
                  Target.LLVM_HOST,
                  features.isEmpty() ? new String[0] : features.split(","),
                  binary());
    return cached;
  }

  private static Type typeOf(byte ordinal) {
    for (Type t : Type.VALUES) if (t.value() == ordinal) return t;
    throw new IllegalArgumentException("Unknown runtime type ordinal " + ordinal);
  }

  private enum Prim {
    BOOL(boolean.class, Type.BOOL, 1) {
      Object get(ByteBuffer b, int at) {
        return b.get(at) != 0;
      }

      void put(ByteBuffer b, int at, Object v) {
        b.put(at, (byte) ((Boolean) v ? 1 : 0));
      }

    },
    BYTE(byte.class, Type.BYTE, Byte.BYTES) {
      Object get(ByteBuffer b, int at) {
        return b.get(at);
      }

      void put(ByteBuffer b, int at, Object v) {
        b.put(at, (Byte) v);
      }

    },
    CHAR(char.class, Type.CHAR, Character.BYTES) {
      Object get(ByteBuffer b, int at) {
        return b.getChar(at);
      }

      void put(ByteBuffer b, int at, Object v) {
        b.putChar(at, (Character) v);
      }

    },
    SHORT(short.class, Type.SHORT, Short.BYTES) {
      Object get(ByteBuffer b, int at) {
        return b.getShort(at);
      }

      void put(ByteBuffer b, int at, Object v) {
        b.putShort(at, (Short) v);
      }

    },
    INT(int.class, Type.INT, Integer.BYTES) {
      Object get(ByteBuffer b, int at) {
        return b.getInt(at);
      }

      void put(ByteBuffer b, int at, Object v) {
        b.putInt(at, (Integer) v);
      }

    },
    LONG(long.class, Type.LONG, Long.BYTES) {
      Object get(ByteBuffer b, int at) {
        return b.getLong(at);
      }

      void put(ByteBuffer b, int at, Object v) {
        b.putLong(at, (Long) v);
      }

    },
    FLOAT(float.class, Type.FLOAT, Float.BYTES) {
      Object get(ByteBuffer b, int at) {
        return b.getFloat(at);
      }

      void put(ByteBuffer b, int at, Object v) {
        b.putFloat(at, (Float) v);
      }

    },
    DOUBLE(double.class, Type.DOUBLE, Double.BYTES) {
      Object get(ByteBuffer b, int at) {
        return b.getDouble(at);
      }

      void put(ByteBuffer b, int at, Object v) {
        b.putDouble(at, (Double) v);
      }

    };

    private static final Prim[] ALL = values();

    final Class<?> cls;
    final Type type;
    final int bytes;

    Prim(Class<?> cls, Type type, int bytes) {
      this.cls = cls;
      this.type = type;
      this.bytes = bytes;
    }

    abstract Object get(ByteBuffer b, int at);

    abstract void put(ByteBuffer b, int at, Object v);

    final void append(ByteBuffer b, Object v) {
      put(b, b.position(), v);
      // XXX cast to Buffer: JDK 9 overrides position(int) covariantly, which JDK 8 has no
      // descriptor for
      ((Buffer) b).position(b.position() + bytes);
    }

    static Prim of(Class<?> cls) {
      for (Prim p : ALL) if (p.cls == cls) return p;
      return null;
    }

    static Prim of(Type type) {
      for (Prim p : ALL) if (p.type == type) return p;
      return null;
    }
  }

  private static void writeScalar(Type type, Object value, ByteBuffer b) {
    if (type == Type.VOID) return;
    Prim p = Prim.of(type);
    if (p == null) throw new IllegalArgumentException("Cannot pack capture of type " + type);
    p.append(b, value);
  }

  private static Object readScalar(Type type, ByteBuffer b, int offset) {
    if (type == Type.VOID) return null;
    Prim p = Prim.of(type);
    if (p == null) throw new IllegalArgumentException("Cannot unpack return of type " + type);
    return p.get(b, offset);
  }

  private Twins twins;

  // the generated twins are classes of the region's loader, anything else travels through a reflective twin
  private Twins twinsOf(String layouts, String generated) {
    Twins cached = twins;
    if (cached == null) {
      Twins created = Twins.decode(layouts);
      if (!generated.isEmpty())
        for (String name : generated.split(",")) created.add(newTwin(name, created));
      twins = cached = created;
    }
    return cached;
  }

  private Twin newTwin(String name, Twins twins) {
    try {
      return (Twin)
          Class.forName(name, true, getClass().getClassLoader())
              .getConstructor(Twins.class)
              .newInstance(twins);
    } catch (ReflectiveOperationException e) {
      throw new TwinException("Cannot create the generated twin " + name, e);
    }
  }

  @Override
  public Object invoke(Object[] args) {
    if (descriptor().startsWith("!"))
      throw new UnsupportedOperationException(descriptor().substring(1));

    String[] parts = descriptor().split("\\|", 8);
    if (parts.length != 8) throw new IllegalStateException("Malformed descriptor: " + descriptor());
    String moduleName = parts[0];
    String symbol = parts[1];

    String[] ordinals = parts[2].split(",");
    byte[] launchTypes = new byte[ordinals.length];
    for (int i = 0; i < ordinals.length; i++) launchTypes[i] = Byte.parseByte(ordinals[i].trim());
    Type resultType = typeOf(Byte.parseByte(parts[3].trim()));
    String[] plan = parts[4].isEmpty() ? new String[0] : parts[4].split(",");
    Twins twins = twinsOf(parts[5], parts[6]);

    if (plan.length != launchTypes.length - 2)
      throw new IllegalStateException(
          "Entry "
              + symbol
              + " has "
              + (launchTypes.length - 2)
              + " captures but a "
              + plan.length
              + " entry plan");

    // the context supplies the thread id slot, the error block and the void return
    byte[] argTypes = Arrays.copyOfRange(launchTypes, 1, launchTypes.length - 1);
    ModuleImage image = imageOf(moduleName, parts[7]);
    Context context = Runtime0.CONTEXT;

    int total = 0;
    for (byte t : argTypes) total += typeOf(t).sizeInBytes();
    ByteBuffer values = ByteBuffer.allocate(total).order(ByteOrder.nativeOrder());

    byte[] result = null;
    List<StaticBinding> statics = new ArrayList<>();
    try (Marshal m = new Marshal(context.queue(), twins)) {
      for (int slot = 0; slot < plan.length; slot++) {
        String source = plan[slot].trim();
        Type type = typeOf(argTypes[slot]);
        if (source.equals("r")) {
          result = new byte[resultType.sizeInBytes()];
          values.putLong(m.array(result));
          continue;
        }
        if (source.equals("t")) {
          values.putLong(instanceOf(m, args[0]));
          continue;
        }
        if (source.charAt(0) == 's') {
          StaticBinding binding = StaticBinding.of(source.substring(1));
          statics.add(binding);
          values.putLong(m.array(binding.box));
          continue;
        }
        int i = Integer.parseInt(source.substring(1));
        if (i >= args.length)
          throw new IllegalStateException(
              "Entry "
                  + symbol
                  + " wants capture "
                  + i
                  + " but the lambda supplied "
                  + args.length);
        Object arg = args[i];
        switch (source.charAt(0)) {
          case 'a':
            if (type == Type.PTR) {
              if (arg == null) values.putLong(0L);
              else if (arg.getClass().isArray()) values.putLong(m.array(arg));
              else values.putLong(instanceOf(m, arg));
            } else if (arg instanceof Enum) writeScalar(type, ((Enum<?>) arg).ordinal(), values);
            else writeScalar(type, arg, values);
            break;
          case 'l':
            values.putInt(arg == null ? 0 : Array.getLength(arg));
            break;
          case 'L':
            if (arg == null) values.putLong(0L);
            else {
              Object[] rows = (Object[]) arg;
              int[] lengths = new int[rows.length];
              for (int r = 0; r < rows.length; r++)
                lengths[r] = rows[r] == null ? 0 : Array.getLength(rows[r]);
              values.putLong(m.array(lengths));
            }
            break;
          default:
            throw new IllegalStateException("Unknown capture source `" + source + "`");
        }
      }

      if (afterPack != null) afterPack.run();

      context.launch(
          new ModuleImage[] {image},
          moduleName,
          symbol,
          argTypes,
          values.array(),
          new Policy(new Dim3(1, 1, 1)));

      m.settle();
      for (StaticBinding binding : statics) binding.writeBack();

      if (result == null) return null;
      return readScalar(resultType, ByteBuffer.wrap(result).order(ByteOrder.nativeOrder()), 0);
    }
  }

  private static long instanceOf(Marshal m, Object target) {
    if (target == null) throw new TwinException("Captured object is null");
    return m.instance(target);
  }

  // a static travels as a one-element array copied back into the field after the kernel
  private static final class StaticBinding {
    final Field field;
    final Object box;

    private StaticBinding(Field field, Object box) {
      this.field = field;
      this.box = box;
    }

    static StaticBinding of(String spec) {
      int hash = spec.indexOf('#');
      String owner = spec.substring(0, hash);
      String name = spec.substring(hash + 1);
      Field f;
      try {
        f = Class.forName(owner).getDeclaredField(name);
        f.setAccessible(true);
      } catch (ClassNotFoundException | NoSuchFieldException | SecurityException e) {
        throw new IllegalStateException("Cannot resolve static field " + owner + "." + name, e);
      }
      Class<?> t = f.getType();
      if (Prim.of(t) == null)
        throw new IllegalStateException("Cannot offload static field of type " + t);
      try {
        Object box = Array.newInstance(t, 1);
        Array.set(box, 0, f.get(null));
        return new StaticBinding(f, box);
      } catch (IllegalAccessException e) {
        throw new IllegalStateException("Cannot read static field " + owner + "." + name, e);
      }
    }

    void writeBack() {
      if (Modifier.isFinal(field.getModifiers())) return;
      try {
        field.set(null, Array.get(box, 0));
      } catch (IllegalAccessException e) {
        throw new IllegalStateException("Cannot write back static field " + field, e);
      }
    }
  }
}
