package polyregion.javalang;

import java.io.ByteArrayInputStream;
import java.io.DataInputStream;
import java.io.IOException;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.SerializedLambda;
import java.lang.reflect.Array;
import java.lang.reflect.Method;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.locks.ReentrantLock;
import polyregion.jvm.compiler.Target;
import polyregion.jvm.runtime.CapabilityGapException;
import polyregion.jvm.runtime.Context;
import polyregion.jvm.runtime.LoadOutcome;
import polyregion.jvm.runtime.ModuleImage;
import polyregion.jvm.runtime.Platforms;
import polyregion.jvm.runtime.PolyregionRuntimeException;
import polyregion.jvm.runtime.Type;
import polyregion.jvm.twin.Layout;
import polyregion.jvm.twin.Marshal;
import polyregion.jvm.twin.Twins;

public final class Launch {

  private Launch() {
    throw new AssertionError();
  }

  private static final class Natives {
    static final Platforms PLATFORMS = Platforms.create();
  }

  static PolyregionRuntimeException refusal(String message, Iterable<LoadOutcome> outcomes) {
    LoadOutcome worst = null;
    for (LoadOutcome outcome : outcomes)
      if (worst == null || outcome.value() > worst.value()) worst = outcome;
    return worst == LoadOutcome.LACKS ? new CapabilityGapException(message) : new PolyregionRuntimeException(message);
  }

  private static final class Group {
    final String program;
    final byte[] hostObject;
    final ModuleImage[] modules;

    Group(String program, byte[] hostObject, ModuleImage[] modules) {
      this.program = program;
      this.hostObject = hostObject;
      this.modules = modules;
    }
  }

  private static final class Field {
    final int offset;
    final char kind;
    final int capture;

    Field(int offset, char kind, int capture) {
      this.offset = offset;
      this.kind = kind;
      this.capture = capture;
    }
  }

  private static final class Spec {
    final char shape;
    final char code;
    final int size;
    final Field[] fields;
    final long value;

    Spec(char shape, char code, int size, Field[] fields, long value) {
      this.shape = shape;
      this.code = code;
      this.size = size;
      this.fields = fields;
      this.value = value;
    }
  }

  private static final class Plan {
    final String unsupported;
    final String entry;
    final char result;
    final char regionResult;
    final Spec[] specs;
    final Group[] groups;
    final Twins twins = Twins.decode(Layout.encodeAll(Collections.<Layout>emptyList()));

    Plan(
        String unsupported,
        String entry,
        char result,
        char regionResult,
        Spec[] specs,
        Group[] groups) {
      this.unsupported = unsupported;
      this.entry = entry;
      this.result = result;
      this.regionResult = regionResult;
      this.specs = specs;
      this.groups = groups;
    }

    Group select(Context context) {
      Group chosen = null;
      List<LoadOutcome> refusals = new ArrayList<>();
      for (Group group : groups)
        if (group.modules.length == 0) {
          if (chosen == null) chosen = group;
        } else {
          int found = context.select(group.modules);
          if (found >= 0) return group;
          refusals.add(LoadOutcome.VALUES[-found]);
        }
      if (chosen == null)
        throw refusal("No compiled variant of `" + entry + "` accepts " + context, refusals);
      return chosen;
    }
  }

  private static final ClassValue<Plan> PLANS =
      new ClassValue<Plan>() {
        @Override
        protected Plan computeValue(Class<?> site) {
          return decode(create(site));
        }
      };

  private static final ClassValue<ConcurrentHashMap<String, Class<?>>> SITES =
      new ClassValue<ConcurrentHashMap<String, Class<?>>>() {
        @Override
        protected ConcurrentHashMap<String, Class<?>> computeValue(Class<?> home) {
          return new ConcurrentHashMap<>();
        }
      };

  private static final Set<String> LOADED = ConcurrentHashMap.newKeySet();

  private static final ReentrantLock LOAD_LOCK = new ReentrantLock();

  private static final ClassValue<MethodHandle> REPLACE =
      new ClassValue<MethodHandle>() {
        @Override
        protected MethodHandle computeValue(Class<?> callable) {
          return writeReplace(callable);
        }
      };

  private static Target target(byte value) {
    for (Target t : Target.VALUES) if (t.value() == value) return t;
    throw new IllegalStateException("Unknown target " + value);
  }

  private static Group[] decodeGroups(byte[] bytes) {
    try (DataInputStream in = new DataInputStream(new ByteArrayInputStream(bytes))) {
      Group[] groups = new Group[in.readInt()];
      for (int g = 0; g < groups.length; g++) {
        String program = in.readUTF();
        byte[] host = new byte[in.readInt()];
        in.readFully(host);
        ModuleImage[] modules = new ModuleImage[in.readInt()];
        for (int m = 0; m < modules.length; m++) {
          String name = in.readUTF();
          byte target = in.readByte();
          String[] features = new String[in.readInt()];
          for (int f = 0; f < features.length; f++) features[f] = in.readUTF();
          byte[] image = new byte[in.readInt()];
          in.readFully(image);
          modules[m] = new ModuleImage(name, target(target), features, image);
        }
        groups[g] = new Group(program, host, modules);
      }
      return groups;
    } catch (IOException e) {
      throw new IllegalStateException("Malformed launch groups", e);
    }
  }

  private static Plan decode(LaunchSite site) {
    String descriptor = site.descriptor();
    if (descriptor.startsWith("!"))
      return new Plan(descriptor.substring(1), "", 'V', 'V', new Spec[0], new Group[0]);
    String[] parts = descriptor.split("\\|", 4);
    if (parts.length < 3) throw new IllegalStateException("Malformed descriptor: " + descriptor);
    String[] args = parts[2].isEmpty() ? new String[0] : parts[2].split(";");
    Spec[] specs = new Spec[args.length];
    for (int i = 0; i < args.length; i++) {
      String arg = args[i];
      switch (arg.charAt(0)) {
        case 'P':
        case 'A':
          specs[i] = new Spec(arg.charAt(0), arg.charAt(1), 0, new Field[0], 0);
          break;
        case 'L':
          specs[i] = new Spec('L', arg.charAt(1), 0, new Field[0], Long.parseLong(arg.substring(2)));
          break;
        case 'C':
          int colon = arg.indexOf(':');
          int size = Integer.parseInt(arg.substring(1, colon));
          String body = arg.substring(colon + 1);
          String[] fields = body.isEmpty() ? new String[0] : body.split(",");
          Field[] decoded = new Field[fields.length];
          for (int f = 0; f < fields.length; f++) {
            String field = fields[f];
            int kind = 0;
            while (Character.isDigit(field.charAt(kind))) kind++;
            decoded[f] =
                new Field(
                    Integer.parseInt(field.substring(0, kind)),
                    field.charAt(kind),
                    Integer.parseInt(field.substring(kind + 1)));
          }
          specs[i] = new Spec('C', 'V', size, decoded, 0);
          break;
        default:
          throw new IllegalStateException("Malformed descriptor: " + descriptor);
      }
    }
    return new Plan(
        null,
        parts[0],
        parts[1].charAt(0),
        parts.length == 4 ? parts[3].charAt(0) : 'V',
        specs,
        decodeGroups(site.binary()));
  }

  private static Class<?> siteClass(String name, ClassLoader loader) {
    try {
      return Class.forName(name, true, loader);
    } catch (ClassNotFoundException e) {
      throw new RuntimeException(
          "Cannot find the launch site class "
              + name
              + "; is the polyregion annotation processor on the compiler's processor path, and is the call inside"
              + " an ordinary method?",
          e);
    }
  }

  private static LaunchSite create(Class<?> site) {
    try {
      return site.asSubclass(LaunchSite.class).getDeclaredConstructor().newInstance();
    } catch (ReflectiveOperationException e) {
      throw new RuntimeException("Cannot create the launch site " + site.getName(), e);
    }
  }

  private static MethodHandle writeReplace(Class<?> callable) {
    Method found = null;
    for (Class<?> c = callable; c != null && found == null; c = c.getSuperclass()) {
      try {
        found = c.getDeclaredMethod("writeReplace");
      } catch (NoSuchMethodException ignored) {
      }
    }
    if (found == null)
      throw new IllegalArgumentException(
          "A launch callable must be a lambda expression, got " + callable.getName());
    try {
      found.setAccessible(true);
      return MethodHandles.lookup().unreflect(found);
    } catch (IllegalAccessException | SecurityException e) {
      throw new RuntimeException(e);
    }
  }

  private static SerializedLambda serialised(Object callable) {
    MethodHandle handle = REPLACE.get(callable.getClass());
    try {
      Object replaced = handle.invoke(callable);
      if (!(replaced instanceof SerializedLambda))
        throw new IllegalArgumentException(
            "A launch callable must be a lambda expression, got " + callable.getClass().getName());
      return (SerializedLambda) replaced;
    } catch (RuntimeException | Error e) {
      throw e;
    } catch (Throwable e) {
      throw new AssertionError(e);
    }
  }

  private static void put(ByteBuffer b, int at, char code, Object v) {
    switch (code) {
      case 'I':
        b.putInt(at, (Integer) v);
        break;
      case 'J':
        b.putLong(at, (Long) v);
        break;
      case 'F':
        b.putFloat(at, (Float) v);
        break;
      case 'D':
        b.putDouble(at, (Double) v);
        break;
      case 'S':
        b.putShort(at, (Short) v);
        break;
      case 'B':
        b.put(at, (Byte) v);
        break;
      case 'C':
        b.putChar(at, (Character) v);
        break;
      case 'Z':
        b.put(at, (byte) ((Boolean) v ? 1 : 0));
        break;
      default:
        throw new IllegalStateException("Unknown scalar code `" + code + "`");
    }
  }

  private static int sizeOf(char code) {
    switch (code) {
      case 'I':
      case 'F':
        return 4;
      case 'J':
      case 'D':
        return 8;
      case 'S':
      case 'C':
        return 2;
      case 'B':
      case 'Z':
        return 1;
      case 'V':
        return 0;
      default:
        throw new IllegalStateException("Unknown scalar code `" + code + "`");
    }
  }

  private static Object get(ByteBuffer b, char code) {
    switch (code) {
      case 'I':
        return b.getInt(0);
      case 'J':
        return b.getLong(0);
      case 'F':
        return b.getFloat(0);
      case 'D':
        return b.getDouble(0);
      case 'S':
        return b.getShort(0);
      case 'B':
        return b.get(0);
      case 'C':
        return b.getChar(0);
      case 'Z':
        return b.get(0) != 0;
      default:
        throw new IllegalStateException("Unknown scalar code `" + code + "`");
    }
  }

  private static void ensureLoaded(Group group) {
    if (LOADED.contains(group.program)) return;
    LOAD_LOCK.lock();
    try {
      if (!LOADED.contains(group.program)) {
        if (!Context.hostProgramLoaded(group.program))
          Context.loadHostProgram(group.program, group.hostObject);
        LOADED.add(group.program);
      }
    } finally {
      LOAD_LOCK.unlock();
    }
  }

  private static ClassLoader defaultLoader() {
    ClassLoader context = Thread.currentThread().getContextClassLoader();
    return context != null ? context : Launch.class.getClassLoader();
  }

  public static Object call(String identity, Context context, Object[] args) {
    SerializedLambda[] lambdas = new SerializedLambda[args.length];
    List<String> parts = new ArrayList<>();
    ClassLoader loader = null;
    for (int i = 0; i < args.length; i++) {
      Object arg = args[i];
      if (arg == null)
        throw new IllegalArgumentException("Argument " + i + " of `" + identity + "` is null");
      String code = LaunchSite.codeOf(arg.getClass());
      if (!code.isEmpty()) parts.add(code);
      else {
        lambdas[i] = serialised(arg);
        parts.add("@" + lambdas[i].getImplClass() + "." + lambdas[i].getImplMethodName());
        if (loader == null) loader = arg.getClass().getClassLoader();
      }
    }
    String name = LaunchSite.nameOf(identity, parts);
    Plan plan = PLANS.get(siteClass(name, loader != null ? loader : defaultLoader()));
    if (plan.unsupported != null) throw new UnsupportedOperationException(plan.unsupported);
    if (plan.specs.length != args.length)
      throw new IllegalStateException(
          "`" + identity + "` was compiled for " + plan.specs.length + " arguments, called with " + args.length);

    Object[][] captured = new Object[args.length][];
    for (int i = 0; i < args.length; i++)
      if (lambdas[i] != null) {
        captured[i] = new Object[lambdas[i].getCapturedArgCount()];
        for (int c = 0; c < captured[i].length; c++) captured[i][c] = lambdas[i].getCapturedArg(c);
      }
    return execute(plan, context, args, captured);
  }

  static Object step(Class<?> home, String name, Context context, Object[] args, Object[] captured) {
    Plan plan =
        PLANS.get(SITES.get(home).computeIfAbsent(name, site -> siteClass(site, home.getClassLoader())));
    if (plan.unsupported != null) throw new UnsupportedOperationException(plan.unsupported);
    if (plan.specs.length != args.length)
      throw new IllegalStateException(
          "`" + name + "` was compiled for " + plan.specs.length + " arguments, called with " + args.length);
    Object[][] each = new Object[args.length][];
    java.util.Arrays.fill(each, captured);
    return execute(plan, context, args, each);
  }

  static Object region(LaunchSite site, Object[] captured) {
    Plan plan = PLANS.get(site.getClass());
    if (plan.unsupported != null) throw new UnsupportedOperationException(plan.unsupported);
    Object[][] each = new Object[plan.specs.length][];
    each[plan.specs.length - 1] = captured;
    return execute(plan, null, new Object[plan.specs.length], each);
  }

  private static Object execute(Plan plan, Context context, Object[] args, Object[][] captured) {
    Context ctx = context != null ? context : Context.current();
    Group group = plan.select(ctx);
    ensureLoaded(group);

    List<Long> pointers = new ArrayList<>();
    ByteBuffer cell = null;
    ByteBuffer regionCell = null;
    try (Marshal m =
        Marshal.inPlace(
            plan.twins,
            new Marshal.HostView() {
              @Override
              public ByteBuffer view(long pointer, long size) {
                return Natives.PLATFORMS.directBufferFromPointer(pointer, size);
              }
            })) {
      for (int i = 0; i < plan.specs.length; i++) {
        Spec spec = plan.specs[i];
        switch (spec.shape) {
          case 'P':
          case 'L':
            {
              ByteBuffer scalar = Marshal.allocate(sizeOf(spec.code));
              put(scalar, 0, spec.code, spec.shape == 'L' ? literal(spec) : args[i]);
              pointers.add(m.stage(scalar));
              break;
            }
          case 'A':
            pointers.add(m.array(args[i]));
            break;
          default:
            if (spec.size > 0) {
              ByteBuffer env = Marshal.allocate(spec.size);
              for (Field field : spec.fields) {
                switch (field.kind) {
                  case 'r':
                    regionCell = Marshal.allocate(sizeOf(plan.regionResult));
                    env.putLong(field.offset, m.stage(regionCell));
                    break;
                  case 'a':
                    {
                      Object value = captured[i][field.capture];
                      env.putLong(field.offset, value == null ? 0L : m.array(value));
                      break;
                    }
                  case 'l':
                    {
                      Object value = captured[i][field.capture];
                      env.putInt(field.offset, value == null ? 0 : Array.getLength(value));
                      break;
                    }
                  default:
                    {
                      Object value = captured[i][field.capture];
                      put(env, field.offset, field.kind, value instanceof Enum ? ((Enum<?>) value).ordinal() : value);
                    }
                }
              }
              pointers.add(m.stage(env));
            }
        }
      }
      if (plan.result != 'V') {
        cell = Marshal.allocate(sizeOf(plan.result));
        pointers.add(m.stage(cell));
      }
      byte[] types = new byte[pointers.size()];
      java.util.Arrays.fill(types, Type.PTR.value());
      ByteBuffer data = ByteBuffer.allocate(8 * types.length).order(ByteOrder.nativeOrder());
      for (long pointer : pointers) data.putLong(pointer);
      ctx.callHost(group.modules, group.program, plan.entry, types, data.array());
      m.settle();
    }
    if (regionCell != null) return get(regionCell, plan.regionResult);
    return cell == null ? null : get(cell, plan.result);
  }

  private static Object literal(Spec spec) {
    switch (spec.code) {
      case 'I':
        return (int) spec.value;
      case 'J':
        return spec.value;
      default:
        throw new IllegalStateException("Unsupported literal code `" + spec.code + "`");
    }
  }
}
