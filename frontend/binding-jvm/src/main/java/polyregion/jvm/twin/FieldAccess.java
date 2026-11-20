package polyregion.jvm.twin;

import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.Field;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Reads and writes one host field through a {@code VarHandle}, whatever its visibility.
 *
 * <p>The handle comes from {@code MethodHandles.privateLookupIn}, so fields of JDK classes need the
 * owning package opened to the caller ({@code --add-opens java.base/java.util=ALL-UNNAMED}). The
 * lookup goes through reflection so this class loads on a JDK without {@code VarHandle}, where
 * {@link #of} fails with a {@link TwinException} instead.
 */
public final class FieldAccess {

  public final Field field;
  private final MethodHandle getter;
  private final MethodHandle setter;

  private FieldAccess(Field field, MethodHandle getter, MethodHandle setter) {
    this.field = field;
    this.getter = getter;
    this.setter = setter;
  }

  public boolean writable() {
    return setter != null;
  }

  public Object get(Object target) {
    try {
      return getter.invoke(target);
    } catch (Throwable t) {
      throw new TwinException("Cannot read " + field, t);
    }
  }

  public void set(Object target, Object value) {
    try {
      setter.invoke(target, value);
    } catch (Throwable t) {
      throw new TwinException("Cannot write " + field, t);
    }
  }

  static String descriptorOf(Class<?> c) {
    if (c.isArray()) return "[" + descriptorOf(c.getComponentType());
    Prim p = Prim.of(c);
    if (p != null) return String.valueOf(p.descriptor);
    return "L" + c.getName().replace('.', '/') + ";";
  }

  static Field find(Class<?> owner, String name) {
    for (Class<?> c = owner; c != null; c = c.getSuperclass())
      try {
        return c.getDeclaredField(name);
      } catch (NoSuchFieldException ignored) {
      }
    throw new TwinException(
        "Class " + owner.getName() + " has no field `" + name + "`, the layout does not match this class");
  }

  private static final ClassValue<ConcurrentHashMap<String, FieldAccess>> CACHE =
      new ClassValue<ConcurrentHashMap<String, FieldAccess>>() {
        @Override
        protected ConcurrentHashMap<String, FieldAccess> computeValue(Class<?> type) {
          return new ConcurrentHashMap<>();
        }
      };

  public static FieldAccess of(Class<?> owner, String name, String descriptor) {
    ConcurrentHashMap<String, FieldAccess> known = CACHE.get(owner);
    String key = name + ":" + descriptor;
    FieldAccess cached = known.get(key);
    if (cached != null) return cached;
    FieldAccess created = create(owner, name, descriptor);
    FieldAccess raced = known.putIfAbsent(key, created);
    return raced != null ? raced : created;
  }

  private static FieldAccess create(Class<?> owner, String name, String descriptor) {
    Field field = find(owner, name);
    String actual = descriptorOf(field.getType());
    if (!descriptor.isEmpty() && !descriptor.equals(actual))
      throw new TwinException(
          "Field "
              + owner.getName()
              + "."
              + name
              + " has type "
              + actual
              + " but the layout expects "
              + descriptor);
    if (Modifier.isStatic(field.getModifiers()))
      throw new TwinException("Field " + owner.getName() + "." + name + " is static");
    return open(field);
  }

  private static TwinException unreachable(Field field, Throwable cause) {
    Class<?> c = field.getDeclaringClass();
    String module = moduleName(c);
    String pkg = c.getName().substring(0, Math.max(c.getName().lastIndexOf('.'), 0));
    String hint =
        module == null
            ? ""
            : ". Launch the JVM with --add-opens " + module + "/" + pkg + "=ALL-UNNAMED";
    return new TwinException(
        "Cannot reach private field " + c.getName() + "." + field.getName() + hint, cause);
  }

  private static String moduleName(Class<?> c) {
    try {
      Object module = Class.class.getMethod("getModule").invoke(c);
      return (String) module.getClass().getMethod("getName").invoke(module);
    } catch (ReflectiveOperationException e) {
      return null;
    }
  }

  private static FieldAccess open(Field field) {
    Class<?> owner = field.getDeclaringClass();
    Class<?> varHandle;
    Class<?> accessMode;
    Method privateLookupIn;
    try {
      varHandle = Class.forName("java.lang.invoke.VarHandle");
      accessMode = Class.forName("java.lang.invoke.VarHandle$AccessMode");
      privateLookupIn =
          MethodHandles.class.getMethod(
              "privateLookupIn", Class.class, MethodHandles.Lookup.class);
    } catch (ReflectiveOperationException e) {
      throw new TwinException(
          "Reaching private fields needs a JDK with VarHandle (9 or newer), running on "
              + System.getProperty("java.version"),
          e);
    }
    try {
      MethodHandles.Lookup lookup =
          (MethodHandles.Lookup) privateLookupIn.invoke(null, owner, MethodHandles.lookup());
      Object handle =
          MethodHandles.Lookup.class
              .getMethod("findVarHandle", Class.class, String.class, Class.class)
              .invoke(lookup, owner, field.getName(), field.getType());
      Method toMethodHandle = varHandle.getMethod("toMethodHandle", accessMode);
      MethodHandle getter =
          (MethodHandle) toMethodHandle.invoke(handle, enumConstant(accessMode, "GET"));
      MethodHandle setter =
          Modifier.isFinal(field.getModifiers())
              ? null
              : (MethodHandle) toMethodHandle.invoke(handle, enumConstant(accessMode, "SET"));
      return new FieldAccess(field, getter, setter);
    } catch (InvocationTargetException e) {
      throw unreachable(field, e.getCause());
    } catch (ReflectiveOperationException e) {
      throw unreachable(field, e);
    }
  }

  @SuppressWarnings({"unchecked", "rawtypes"})
  private static Object enumConstant(Class<?> type, String name) {
    return Enum.valueOf((Class) type, name);
  }
}
