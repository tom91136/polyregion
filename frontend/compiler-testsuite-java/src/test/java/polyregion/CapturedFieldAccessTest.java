package polyregion;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;
import static polyregion.javalang.Runtime.offload;

import java.io.File;
import java.net.JarURLConnection;
import java.net.URL;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Enumeration;
import java.util.List;
import java.util.jar.JarEntry;
import java.util.jar.JarFile;
import org.junit.Test;
import polyregion.javalang.OffloadFunction;
import polyregion.javalang.OffloadRunnable;
import polyregion.jvm.twin.TwinException;

public class CapturedFieldAccessTest {

  static final class Open {
    int count = 1;
    long wide = 2L;
    double ratio = 1.5;
    boolean flag = true;
    final int fixed;
    int[] data = {1, 2, 3};

    Open() {
      fixed = 9;
    }
  }

  static final class Hidden {
    private int secret = 3;
    private int[] cells = {4, 5};
    private final int fixed;

    Hidden() {
      fixed = 6;
    }
  }

  static final class Inner {
    int v = 2;
  }

  static final class Outer {
    Inner inner = new Inner();
    int k = 1;
  }

  static final class Boxed {
    Integer count = 1;
  }

  private static List<String> twinClasses(String region) throws Exception {
    URL root = CapturedFieldAccessTest.class.getResource("/polyregion/");
    List<String> names = new ArrayList<>();
    if (root.getProtocol().equals("jar")) {
      JarFile jar = ((JarURLConnection) root.openConnection()).getJarFile();
      for (Enumeration<JarEntry> es = jar.entries(); es.hasMoreElements(); ) {
        String name = es.nextElement().getName();
        if (name.startsWith("polyregion/") && name.indexOf('/', 11) < 0) names.add(name.substring(11));
      }
    } else names.addAll(Arrays.asList(new File(root.toURI()).list()));
    List<String> out = new ArrayList<>();
    for (String name : names) if (name.contains("$Twin$") && name.contains(region)) out.add(name);
    return out;
  }

  @Test
  public void generatedTwinRoundTrips() throws Exception {
    Open o = new Open();
    OffloadRunnable r =
        () -> {
          o.count += o.fixed;
          o.wide *= 3;
          o.ratio += 1;
          o.flag = !o.flag;
          o.data[1] = 20;
        };
    offload(r);
    assertEquals(10, o.count);
    assertEquals(6L, o.wide);
    assertEquals(2.5, o.ratio, 0.0);
    assertFalse(o.flag);
    assertEquals(9, o.fixed);
    assertArrayEquals(new int[] {1, 20, 3}, o.data);
    assertEquals(1, twinClasses("generatedTwinRoundTrips").size());
  }

  @Test
  public void privateFieldsGoThroughTheReflectiveTwin() throws Exception {
    Hidden h = new Hidden();
    OffloadRunnable r =
        () -> {
          h.secret += h.fixed;
          h.cells[0] = h.secret;
        };
    offload(r);
    assertEquals(9, h.secret);
    assertArrayEquals(new int[] {9, 5}, h.cells);
    assertTrue(twinClasses("privateFieldsGoThroughTheReflectiveTwin").isEmpty());
  }

  @Test
  public void nestedObjectsRoundTrip() throws Exception {
    Outer o = new Outer();
    OffloadRunnable r =
        () -> {
          o.inner.v += 5;
          o.k += o.inner.v;
        };
    offload(r);
    assertEquals(7, o.inner.v);
    assertEquals(8, o.k);
    assertEquals(2, twinClasses("nestedObjectsRoundTrip").size());
  }

  @Test
  public void aBoxedFieldIsRefusedByName() {
    Boxed b = new Boxed();
    OffloadFunction<Integer> f = () -> b.count + 1;
    try {
      offload(f);
      fail("expected a boxed field to be refused");
    } catch (TwinException e) {
      assertTrue(e.getMessage(), e.getMessage().contains("count"));
    }
  }
}
