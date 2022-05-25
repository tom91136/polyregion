package polyregion.java;

import java.io.Serializable;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.SerializedLambda;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.Arrays;
import java.util.WeakHashMap;

public final class Runtime {

  private Runtime() {
    throw new AssertionError();
  }

  private static MethodHandle extractWriteReplaceSerializedLambda(
      final Class<? extends Serializable> base) {
    for (Class<?> cl = base; cl != null; cl = cl.getSuperclass()) {
      try {
        System.out.println("in > " + cl);
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

  public static void offload(OffloadRunnable f) {
    //		SerializedLambda lambda = extractSerializedLambda(f);
    //		System.out.println("f0=" + f + " " + lambda.toString() + " === " +
    // lambda.getImplMethodName());
    f.run();
  }

  enum Backend {
    LLVM
  }

  // don't prevent the key class from unloading
  private static final WeakHashMap<Class<? extends OffloadRegion>, MethodHandle>
      regionWriteReplaceMap = new WeakHashMap<>();

  private static final WeakHashMap<Class<? extends OffloadRegion>, OffloadExecutable>
      regionExecutableMap = new WeakHashMap<>();

  private static MethodHandle resolveWriteReplace(final OffloadRegion region) {
    return regionWriteReplaceMap.computeIfAbsent(
        region.getClass(), Runtime::extractWriteReplaceSerializedLambda);
  }

  private static OffloadExecutable instantiateOffloadExecutable(
      final OffloadRegion region, final SerializedLambda lambda) {
    final String name =
        "polyregion.$gen$."
            + lambda.getImplClass().replace('/', '$')
            + "$"
            + lambda.getImplMethodName();
    return regionExecutableMap.computeIfAbsent(
        region.getClass(),
        x -> {
          try {
            Class<? extends OffloadExecutable> cls =
                Class.forName(name).asSubclass(OffloadExecutable.class);
            try {
              return cls.getDeclaredConstructor().newInstance();
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
            throw new RuntimeException("Cannot find OffloadExecutable for class " + name, e);
          }
        });
  }

  public static <R> R offload(OffloadFunction<R> f) {
    MethodHandle handle = resolveWriteReplace(f);
    SerializedLambda lambda;
    try {
      lambda = (SerializedLambda) handle.invoke(f);
    } catch (Throwable e) {
      throw new AssertionError(e);
    }

    final Object[] args = new Object[lambda.getCapturedArgCount()];
    for (int i = 0, argsLength = args.length; i < argsLength; i++)
      args[i] = lambda.getCapturedArg(i);

    OffloadExecutable exe = instantiateOffloadExecutable(f, lambda);

    System.out.println(
        "f1=" + f + "  => " + lambda.getImplClass() + "." + lambda.getImplMethodName());
    System.out.println("  args=" + Arrays.toString(args) + " ins=" + f.getClass());
    System.out.println("  exe=" + exe);
    exe.invoke(args);

    //		a.computeIfAbsent(f, r -> "");

    return f.run();
  }
}
