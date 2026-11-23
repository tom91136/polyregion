package polyregion;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.fail;

import java.lang.ref.WeakReference;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.ByteBuffer;
import java.util.Collections;
import java.util.HashSet;
import java.util.Set;
import org.junit.Test;
import polyregion.javalang.Launch;
import polyregion.javalang.LaunchSite;
import polyregion.jvm.runtime.Context;

public class LaunchPlanTest {

  private static final String IDENTITY = "fake:fake.plan";

  private static final String SITE = LaunchSite.nameOf(IDENTITY, Collections.<String>emptyList());

  private static final class Isolated extends URLClassLoader {
    private final String label;

    private final Set<String> own = new HashSet<>();

    Isolated(String label, String... own) {
      super(
          new URL[] {LaunchPlanTest.class.getProtectionDomain().getCodeSource().getLocation()},
          LaunchPlanTest.class.getClassLoader());
      this.label = label;
      this.own.add(SITE);
      Collections.addAll(this.own, own);
    }

    @Override
    protected Class<?> loadClass(String name, boolean resolve) throws ClassNotFoundException {
      synchronized (getClassLoadingLock(name)) {
        if (!own.contains(name)) return super.loadClass(name, resolve);
        Class<?> loaded = findLoadedClass(name);
        return loaded != null ? loaded : findClass(name);
      }
    }

    @Override
    public String toString() {
      return label;
    }
  }

  private static final class Blind extends ClassLoader {
    Blind() {
      super(LaunchPlanTest.class.getClassLoader());
    }

    @Override
    protected Class<?> loadClass(String name, boolean resolve) throws ClassNotFoundException {
      if (name.equals(SITE)) throw new ClassNotFoundException(name);
      return super.loadClass(name, resolve);
    }
  }

  public static final class Stub {
    public static void launch() {
      Launch.call(IDENTITY, null, new Object[0]);
    }
  }

  public static final class Caller implements Runnable {
    @Override
    public void run() {
      Stub.launch();
    }
  }

  public static final class Probe implements java.util.concurrent.Callable<Object> {
    @Override
    public Object call() {
      return (Runnable & java.io.Serializable) () -> {};
    }
  }

  private static String launchedFrom(Isolated loader, ClassLoader context) throws Exception {
    Runnable user = (Runnable) loader.loadClass(Caller.class.getName()).getDeclaredConstructor().newInstance();
    Thread thread = Thread.currentThread();
    ClassLoader previous = thread.getContextClassLoader();
    thread.setContextClassLoader(context);
    try {
      user.run();
    } catch (UnsupportedOperationException e) {
      return e.getMessage();
    } finally {
      thread.setContextClassLoader(previous);
    }
    fail("the fake site must refuse the launch");
    return null;
  }

  @Test
  public void theSiteNameMatchesTheFixture() {
    assertEquals("polyregion.$gen$.Launch$fake_fake_plan$1e05ab1f314ca3cc", SITE);
  }

  @Test
  public void eachLoaderGetsItsOwnPlan() throws Exception {
    String caller = Caller.class.getName();
    try (Isolated first = new Isolated("first", caller);
        Isolated second = new Isolated("second", caller)) {
      assertEquals("first", launchedFrom(first, null));
      assertEquals("second", launchedFrom(second, null));
      assertEquals("first", launchedFrom(first, null));
    }
  }

  private static final class Point {
    int x;
  }

  @Test
  public void theKeyVocabularyCarriesScalarsAndPrimitiveArraysOnly() {
    Class<?>[] scalars = {
      int.class, long.class, float.class, double.class, short.class, byte.class, char.class, boolean.class,
      Integer.class, Long.class, Float.class, Double.class, Short.class, Byte.class, Character.class, Boolean.class
    };
    for (Class<?> c : scalars) assertEquals(c.getName(), 1, LaunchSite.codeOf(c).length());
    Class<?>[] arrays = {
      int[].class, long[].class, float[].class, double[].class, short[].class, byte[].class, char[].class, boolean[].class
    };
    for (Class<?> c : arrays) assertEquals(c.getName(), 2, LaunchSite.codeOf(c).length());
    Class<?>[] outside = {
      Object.class, String.class, Object[].class, Integer[].class, int[][].class, Point.class, Point[].class,
      Enum.class, ByteBuffer.class, Context.class, Runnable.class
    };
    for (Class<?> c : outside)
      assertEquals(c.getName() + " has no key code, the plan carries no struct layouts", "", LaunchSite.codeOf(c));
  }

  @Test
  public void aCallableLessCallResolvesTheSiteFromItsCallerNotTheContextLoader() throws Exception {
    try (Isolated child = new Isolated("child", Caller.class.getName())) {
      assertEquals("child", launchedFrom(child, new Blind()));
    }
  }

  private static WeakReference<ClassLoader> launchedFromACollectableLoader() throws Exception {
    String probe = Probe.class.getName();
    Isolated child = new Isolated("collectable", probe);
    Object lambda = ((java.util.concurrent.Callable<?>) child.loadClass(probe).getDeclaredConstructor().newInstance()).call();
    try {
      Launch.call(IDENTITY, null, new Object[] {lambda});
      fail("the lambda has no generated site");
    } catch (RuntimeException expected) {
    }
    return new WeakReference<ClassLoader>(child);
  }

  @Test
  public void aLoaderThatRanALaunchCanBeCollected() throws Exception {
    WeakReference<ClassLoader> ref = launchedFromACollectableLoader();
    for (int i = 0; i < 50 && ref.get() != null; i++) {
      System.gc();
      Thread.sleep(20);
    }
    assertNull("the loader stays reachable through the launch caches", ref.get());
  }
}
