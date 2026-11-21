package polyregion.javalang;

import java.io.Serializable;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.SerializedLambda;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.Collections;
import java.util.Map;
import java.util.WeakHashMap;

public final class Runtime {

  private Runtime() {
    throw new AssertionError();
  }

  private static MethodHandle extractWriteReplaceSerializedLambda(
      final Class<? extends Serializable> base) {
    for (Class<?> cl = base; cl != null; cl = cl.getSuperclass()) {
      try {
        Method m = cl.getDeclaredMethod("writeReplace");
        m.setAccessible(true);
        return MethodHandles.lookup().unreflect(m);
      } catch (NoSuchMethodException ignored) {
        // keep going
      } catch (IllegalAccessException | SecurityException e) {
        throw new RuntimeException(e);
      }
    }
    throw new AssertionError(
        "Missing writeReplace method for lambda that implements Serializable!");
  }

  // don't prevent the key class from unloading
  private static final Map<Class<? extends OffloadRegion>, MethodHandle> regionWriteReplaceMap =
      Collections.synchronizedMap(new WeakHashMap<Class<? extends OffloadRegion>, MethodHandle>());

  private static final Map<Class<? extends OffloadRegion>, OffloadExecutable> regionExecutableMap =
      Collections.synchronizedMap(
          new WeakHashMap<Class<? extends OffloadRegion>, OffloadExecutable>());

  private static MethodHandle resolveWriteReplace(final OffloadRegion region) {
    Class<? extends OffloadRegion> cls = region.getClass();
    MethodHandle handle = regionWriteReplaceMap.get(cls);
    if (handle == null) {
      handle = extractWriteReplaceSerializedLambda(cls);
      regionWriteReplaceMap.put(cls, handle);
    }
    return handle;
  }

  private static OffloadExecutable instantiateOffloadExecutable(
      final OffloadRegion region, final SerializedLambda lambda) {
    Class<? extends OffloadRegion> key = region.getClass();
    OffloadExecutable cached = regionExecutableMap.get(key);
    if (cached != null) return cached;

    final String name =
        "polyregion.$gen$."
            + lambda.getImplClass().replace('/', '$')
            + "$"
            + lambda.getImplMethodName();
    OffloadExecutable exe;
    try {
      Class<? extends OffloadExecutable> cls =
          Class.forName(name, true, region.getClass().getClassLoader())
              .asSubclass(OffloadExecutable.class);
      try {
        exe = cls.getDeclaredConstructor().newInstance();
      } catch (InstantiationException
          | IllegalAccessException
          | InvocationTargetException
          | NoSuchMethodException e) {
        throw new RuntimeException(
            "Cannot create a new OffloadExecutable instance of class " + cls, e);
      }
    } catch (ClassCastException e) {
      throw new RuntimeException(
          "Discovered OffloadExecutable class "
              + name
              + " does not appear to implement"
              + OffloadExecutable.class,
          e);
    } catch (ClassNotFoundException e) {
      throw new RuntimeException(
          "Cannot find OffloadExecutable for class "
              + name
              + "; is the polyregion annotation processor on the compiler's processor path?",
          e);
    }
    regionExecutableMap.put(key, exe);
    return exe;
  }

  private static Object dispatch(final OffloadRegion f) {
    MethodHandle handle = resolveWriteReplace(f);
    SerializedLambda lambda;
    try {
      lambda = (SerializedLambda) handle.invoke(f);
    } catch (Throwable e) {
      throw new AssertionError(e);
    }

    final Object[] args = new Object[lambda.getCapturedArgCount()];
    for (int i = 0, n = args.length; i < n; i++) args[i] = lambda.getCapturedArg(i);

    return instantiateOffloadExecutable(f, lambda).invoke(args);
  }

  public static void offload(OffloadRunnable f) {
    dispatch(f);
  }

  @SuppressWarnings("unchecked")
  public static <R> R offload(OffloadFunction<R> f) {
    return (R) dispatch(f);
  }
}
