package polyregion.javalang;

import java.lang.reflect.Array;
import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Base64;
import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicReference;
import polyregion.jvm.compiler.Target;
import polyregion.jvm.runtime.Completion;
import polyregion.jvm.runtime.Context;
import polyregion.jvm.runtime.Dim3;
import polyregion.jvm.runtime.ModuleImage;
import polyregion.jvm.runtime.Policy;
import polyregion.jvm.runtime.PolyregionRuntimeException;
import polyregion.jvm.runtime.Queue;
import polyregion.jvm.runtime.Type;

public abstract class BinaryOffloadExecutable implements OffloadExecutable {

  public abstract byte[] binary();

  // `<module>|<symbol>|<ordinals>|<result>|<plan>|<layouts>|<tags>|<features>`, see
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

      void writeArray(Object xs, ByteBuffer b) {
        boolean[] a = (boolean[]) xs;
        for (int i = 0; i < a.length; i++) b.put(i, (byte) (a[i] ? 1 : 0));
      }

      void readArray(ByteBuffer b, Object xs) {
        boolean[] a = (boolean[]) xs;
        for (int i = 0; i < a.length; i++) a[i] = b.get(i) != 0;
      }
    },
    BYTE(byte.class, Type.BYTE, Byte.BYTES) {
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
    CHAR(char.class, Type.CHAR, Character.BYTES) {
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
    SHORT(short.class, Type.SHORT, Short.BYTES) {
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
    INT(int.class, Type.INT, Integer.BYTES) {
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
    LONG(long.class, Type.LONG, Long.BYTES) {
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
    FLOAT(float.class, Type.FLOAT, Float.BYTES) {
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
    DOUBLE(double.class, Type.DOUBLE, Double.BYTES) {
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

    abstract void writeArray(Object xs, ByteBuffer b);

    abstract void readArray(ByteBuffer b, Object xs);

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

    static Prim ofArray(Object array, String what) {
      Prim p = of(array.getClass().getComponentType());
      if (p == null)
        throw new IllegalArgumentException("Cannot " + what + " array of type " + array.getClass());
      return p;
    }
  }

  private static int sizeOf(Object array) {
    Prim p = Prim.of(array.getClass().getComponentType());
    if (p == null)
      throw new IllegalArgumentException(
          "Cannot offload reference capture of type " + array.getClass());
    return Array.getLength(array) * p.bytes;
  }

  // must not move the position, the binding reads it back on a shared address space
  private static void writeArray(Object array, ByteBuffer b) {
    Prim.ofArray(array, "write").writeArray(array, b);
  }

  private static void readArray(ByteBuffer b, Object array) {
    Prim.ofArray(array, "read").readArray(b, array);
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
    Map<String, ObjectLayout> layouts = ObjectLayout.parse(parts[5]);
    Map<String, Integer> tags = parseTags(parts[6]);

    if (plan.length != launchTypes.length - 2)
      throw new IllegalStateException(
          "Entry "
              + symbol
              + " has "
              + (launchTypes.length - 2)
              + " captures but a "
              + plan.length
              + " entry plan");

    // the context supplies the thread id slot and the void return
    byte[] argTypes = Arrays.copyOfRange(launchTypes, 1, launchTypes.length - 1);
    ModuleImage image = imageOf(moduleName, parts[7]);
    Context context = Runtime0.CONTEXT;

    int total = 0;
    for (byte t : argTypes) total += typeOf(t).sizeInBytes();
    ByteBuffer values = ByteBuffer.allocate(total).order(ByteOrder.nativeOrder());

    byte[] result = null;
    byte[] error = null;
    List<StaticBinding> statics = new ArrayList<>();
    Marshal m = new Marshal(context.queue(), layouts, tags, new ArrayList<>());
    try {
      for (int slot = 0; slot < plan.length; slot++) {
        String source = plan[slot].trim();
        Type type = typeOf(argTypes[slot]);
        if (source.equals("r")) {
          result = new byte[resultType.sizeInBytes()];
          values.putLong(register(m, result));
          continue;
        }
        if (source.equals("e")) {
          error = new byte[ASSERT_CODE_BYTES + ASSERT_MESSAGE_LIMIT];
          values.putLong(register(m, error));
          continue;
        }
        if (source.equals("t")) {
          values.putLong(registerObject(m, args[0]));
          continue;
        }
        if (source.charAt(0) == 's') {
          StaticBinding binding = StaticBinding.of(source.substring(1));
          statics.add(binding);
          values.putLong(register(m, binding.box));
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
              else if (arg.getClass().isArray()) values.putLong(registerDeep(m, arg));
              else values.putLong(registerObject(m, arg));
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
              values.putLong(register(m, lengths));
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

      // a lane that asserted drained without finishing, so every other result is meaningless
      if (error != null) {
        sync(m, error);
        raiseIfAsserted(error);
      }

      for (Object buffer : m.toSync) sync(m, buffer);
      for (BoxedObject b : m.boxed) {
        sync(m, b.buffer);
        b.writeBack();
      }

      for (StaticBinding binding : statics) {
        sync(m, binding.box);
        binding.writeBack();
      }

      if (result == null) return null;
      sync(m, result);
      return readScalar(resultType, ByteBuffer.wrap(result).order(ByteOrder.nativeOrder()), 0);
    } finally {
      m.release();
    }
  }

  // rows are registered before the outer array: registering from its write callback would modify
  // the map being walked
  private static long registerDeep(Marshal m, Object array) {
    if (!(array instanceof Object[])) {
      m.toSync.add(array);
      return register(m, array);
    }
    Object[] rows = (Object[]) array;
    // the element type decides, as an empty array has no first element
    if (!array.getClass().getComponentType().isArray()) return registerObjectArray(m, rows);
    long[] ptrs = new long[rows.length];
    for (int i = 0; i < rows.length; i++) {
      if (rows[i] == null)
        throw new IllegalArgumentException("Row " + i + " of a captured array is null");
      ptrs[i] = registerDeep(m, rows[i]);
    }
    return register(m, ptrs);
  }

  private static final class Staged {
    final ByteBuffer buffer;
    final long pointer;

    Staged(ByteBuffer buffer, long pointer) {
      this.buffer = buffer;
      this.pointer = pointer;
    }
  }

  // a queue rejects an empty buffer
  private static ByteBuffer stagingFor(Object array) {
    return ByteBuffer.allocateDirect(Math.max(sizeOf(array), 1)).order(ByteOrder.nativeOrder());
  }

  // an array staged twice in one launch keeps its first upload and device pointer
  private static long register(Marshal m, Object array) {
    Staged staged = m.staged.get(array);
    if (staged != null) return staged.pointer;
    ByteBuffer buffer = stagingFor(array);
    writeArray(array, buffer);
    Settled settled = new Settled();
    long pointer = m.queue.invalidate(buffer, settled);
    settled.await();
    m.staged.put(array, new Staged(buffer, pointer));
    return pointer;
  }

  // field offsets as the compiler laid them out, keyed by binary name; tagOffset is -1 when the
  // class takes no part in dispatch
  private static final class ObjectLayout {
    static final String TAG = "#tag";

    final int size;
    final String[] fields;
    final int[] offsets;
    final int tagOffset;

    private ObjectLayout(int size, String[] fields, int[] offsets, int tagOffset) {
      this.size = size;
      this.fields = fields;
      this.offsets = offsets;
      this.tagOffset = tagOffset;
    }

    static Map<String, ObjectLayout> parse(String spec) {
      Map<String, ObjectLayout> out = new HashMap<>();
      if (spec.isEmpty()) return out;
      for (String entry : spec.split("~")) {
        String[] parts = entry.split(":", 3);
        String name = parts[0];
        int size = Integer.parseInt(parts[1]);
        String members = parts.length > 2 ? parts[2] : "";
        List<String> fields = new ArrayList<>();
        List<Integer> offsets = new ArrayList<>();
        int tagOffset = -1;
        if (!members.isEmpty()) {
          for (String m : members.split(",")) {
            int at = m.indexOf('@');
            int colon = m.indexOf(':', at);
            String field = m.substring(0, at);
            int offset = Integer.parseInt(m.substring(at + 1, colon));
            if (field.equals(TAG)) tagOffset = offset;
            else {
              fields.add(field);
              offsets.add(offset);
            }
          }
        }
        int[] os = new int[offsets.size()];
        for (int i = 0; i < os.length; i++) os[i] = offsets.get(i);
        out.put(name, new ObjectLayout(size, fields.toArray(new String[0]), os, tagOffset));
      }
      return out;
    }
  }

  private static Map<String, Integer> parseTags(String spec) {
    Map<String, Integer> out = new HashMap<>();
    if (spec.isEmpty()) return out;
    for (String entry : spec.split("~")) {
      int eq = entry.lastIndexOf('=');
      out.put(entry.substring(0, eq), Integer.valueOf(entry.substring(eq + 1)));
    }
    return out;
  }

  // the class hierarchy is only seen per compilation, so a class without a tag is refused rather
  // than dispatched blindly
  private static void writeTag(
      ObjectLayout layout, Map<String, Integer> tags, Object target, byte[] buffer, int base) {
    if (layout.tagOffset < 0) return;
    Integer tag = tags.get(target.getClass().getName());
    if (tag == null)
      throw new IllegalArgumentException(
          "Cannot offload a receiver of type "
              + target.getClass().getName()
              + ": the region was compiled to dispatch over "
              + tags.keySet()
              + " and holds no code for anything else");
    ByteBuffer.wrap(buffer).order(ByteOrder.nativeOrder()).putInt(base + layout.tagOffset, tag);
  }

  private static final class Marshal {
    final Queue queue;
    final Map<String, ObjectLayout> layouts;
    final Map<String, Integer> tags;
    final List<Object> toSync;
    final List<BoxedObject> boxed = new ArrayList<>();
    final Map<Object, Staged> staged = new IdentityHashMap<>();

    Marshal(
        Queue queue,
        Map<String, ObjectLayout> layouts,
        Map<String, Integer> tags,
        List<Object> toSync) {
      this.queue = queue;
      this.layouts = layouts;
      this.tags = tags;
      this.toSync = toSync;
    }

    // the queue is shared by every launch on the context, so only this launch's buffers go
    void release() {
      RuntimeException failure = null;
      for (Staged s : staged.values())
        try {
          queue.release(s.buffer);
        } catch (RuntimeException e) {
          if (failure == null) failure = e;
          else failure.addSuppressed(e);
        }
      if (failure != null) throw failure;
    }

    ObjectLayout layoutOf(Object target) {
      ObjectLayout layout = layouts.get(target.getClass().getName());
      if (layout == null)
        throw new IllegalArgumentException(
            "No device layout for " + target.getClass().getName() + "; known: " + layouts.keySet());
      return layout;
    }
  }

  private static final class BoxedObject {
    final Object target;
    final ObjectLayout layout;
    final byte[] buffer;
    final int base;

    private BoxedObject(Object target, ObjectLayout layout, byte[] buffer, int base) {
      this.target = target;
      this.layout = layout;
      this.buffer = buffer;
      this.base = base;
    }

    private Field fieldOf(String name) {
      for (Class<?> c = target.getClass(); c != null; c = c.getSuperclass()) {
        try {
          Field f = c.getDeclaredField(name);
          f.setAccessible(true);
          return f;
        } catch (NoSuchFieldException ignored) {
        }
      }
      throw new IllegalStateException("No field `" + name + "` on " + target.getClass());
    }

    // an array field is registered separately and the struct stores its device pointer; an object
    // field sits inline
    void pack(Marshal m) {
      ByteBuffer b = ByteBuffer.wrap(buffer).order(ByteOrder.nativeOrder());
      try {
        for (int i = 0; i < layout.fields.length; i++) {
          Field f = fieldOf(layout.fields[i]);
          int at = base + layout.offsets[i];
          Class<?> t = f.getType();
          Prim prim = Prim.of(t);
          if (prim != null) prim.put(b, at, f.get(target));
          else if (t.isArray()) {
            Object row = f.get(target);
            b.putLong(at, row == null ? 0L : registerDeep(m, row));
          } else {
            Object nested = f.get(target);
            if (nested == null)
              throw new IllegalArgumentException(
                  "Field `"
                      + f.getName()
                      + "` of "
                      + target.getClass()
                      + " is null; a nested object travels by value and has no null to stand for");
            ObjectLayout declared = m.layouts.get(t.getName());
            if (declared == null)
              throw new IllegalArgumentException(
                  "No device layout for " + t.getName() + "; known: " + m.layouts.keySet());
            if (m.layouts.get(nested.getClass().getName()) != declared)
              throw new IllegalArgumentException(
                  "Field `"
                      + f.getName()
                      + "` declares "
                      + t.getName()
                      + " but holds a "
                      + nested.getClass().getName()
                      + "; a nested object travels by value at the declared class's layout");
            box(m, nested, declared, buffer, at);
          }
        }
      } catch (IllegalAccessException e) {
        throw new IllegalStateException("Cannot read fields of " + target.getClass(), e);
      }
    }

    void writeBack() {
      ByteBuffer b = ByteBuffer.wrap(buffer).order(ByteOrder.nativeOrder());
      try {
        for (int i = 0; i < layout.fields.length; i++) {
          Field f = fieldOf(layout.fields[i]);
          if (Modifier.isFinal(f.getModifiers())) continue;
          Prim prim = Prim.of(f.getType());
          if (prim != null) f.set(target, prim.get(b, base + layout.offsets[i]));
        }
      } catch (IllegalAccessException e) {
        throw new IllegalStateException("Cannot write back fields of " + target.getClass(), e);
      }
    }
  }

  // the tag is stamped after the fields and the box recorded after both, so `boxed` never holds a
  // half-filled buffer
  private static void box(Marshal m, Object target, ObjectLayout layout, byte[] buffer, int base) {
    BoxedObject b = new BoxedObject(target, layout, buffer, base);
    b.pack(m);
    writeTag(layout, m.tags, target, buffer, base);
    m.boxed.add(b);
  }

  private static long registerObject(Marshal m, Object target) {
    if (target == null) throw new IllegalArgumentException("Captured object is null");
    ObjectLayout layout = m.layoutOf(target);
    byte[] buffer = new byte[layout.size];
    box(m, target, layout, buffer, 0);
    return register(m, buffer);
  }

  // the elements are structs travelling by value at the layout's stride
  private static long registerObjectArray(Marshal m, Object[] rows) {
    ObjectLayout layout = null;
    for (Object row : rows) {
      if (row == null)
        throw new IllegalArgumentException("Element of a captured object array is null");
      ObjectLayout l = m.layoutOf(row);
      if (layout != null && l != layout)
        throw new IllegalArgumentException("A captured object array must hold one concrete type");
      layout = l;
    }
    byte[] buffer = new byte[layout == null ? 0 : layout.size * rows.length];
    for (int i = 0; i < rows.length; i++) box(m, rows[i], layout, buffer, i * layout.size);
    return register(m, buffer);
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

  private static final int ASSERT_CODE_BYTES = 4;

  private static final int ASSERT_MESSAGE_LIMIT = 1024;

  // the kernel writes [code:u32 le][message: NUL-terminated] into a zeroed buffer, so all zeros
  // means no lane asserted
  private static void raiseIfAsserted(byte[] error) {
    ByteBuffer b = ByteBuffer.wrap(error).order(ByteOrder.LITTLE_ENDIAN);
    int code = b.getInt(0);
    int end = ASSERT_CODE_BYTES;
    while (end < error.length && error[end] != 0) end++;
    String message =
        new String(error, ASSERT_CODE_BYTES, end - ASSERT_CODE_BYTES, StandardCharsets.UTF_8);
    if (code == 0 && message.isEmpty()) return;
    throw new AssertionError(
        message.isEmpty() ? "device trap " + Integer.toHexString(code) : message);
  }

  private static void sync(Marshal m, Object array) {
    Staged staged = m.staged.get(array);
    Settled settled = new Settled();
    m.queue.sync(staged.buffer, settled);
    settled.await();
    readArray(staged.buffer, array);
  }

  private static final class Settled implements Completion {
    private final CountDownLatch latch = new CountDownLatch(1);
    private final AtomicReference<Throwable> failure = new AtomicReference<>();

    @Override
    public void complete(Throwable failure) {
      this.failure.set(failure);
      latch.countDown();
    }

    void await() {
      try {
        latch.await();
      } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
        throw new PolyregionRuntimeException("Interrupted while waiting for the device");
      }
      Throwable t = failure.get();
      if (t instanceof RuntimeException) throw (RuntimeException) t;
      if (t instanceof Error) throw (Error) t;
      if (t != null) throw new PolyregionRuntimeException(t.toString());
    }
  }
}
