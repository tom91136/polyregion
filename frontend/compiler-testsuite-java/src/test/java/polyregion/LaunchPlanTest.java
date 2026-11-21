package polyregion;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

import java.net.URL;
import java.net.URLClassLoader;
import java.nio.ByteBuffer;
import java.util.Collections;
import org.junit.Test;
import polyregion.javalang.Launch;
import polyregion.javalang.LaunchSite;
import polyregion.jvm.runtime.Context;

public class LaunchPlanTest {

  private static final String IDENTITY = "fake:fake.plan";

  private static final String SITE = LaunchSite.nameOf(IDENTITY, Collections.<String>emptyList());

  private static final class Isolated extends URLClassLoader {
    private final String label;

    Isolated(String label) {
      super(
          new URL[] {LaunchPlanTest.class.getProtectionDomain().getCodeSource().getLocation()},
          LaunchPlanTest.class.getClassLoader());
      this.label = label;
    }

    @Override
    protected Class<?> loadClass(String name, boolean resolve) throws ClassNotFoundException {
      synchronized (getClassLoadingLock(name)) {
        if (!name.equals(SITE)) return super.loadClass(name, resolve);
        Class<?> loaded = findLoadedClass(name);
        return loaded != null ? loaded : findClass(name);
      }
    }

    @Override
    public String toString() {
      return label;
    }
  }

  private static String launchedUnder(ClassLoader loader) {
    Thread thread = Thread.currentThread();
    ClassLoader previous = thread.getContextClassLoader();
    thread.setContextClassLoader(loader);
    try {
      Launch.call(IDENTITY, null, new Object[0]);
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
    try (Isolated first = new Isolated("first");
        Isolated second = new Isolated("second")) {
      assertEquals("first", launchedUnder(first));
      assertEquals("second", launchedUnder(second));
      assertEquals("first", launchedUnder(first));
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
}
