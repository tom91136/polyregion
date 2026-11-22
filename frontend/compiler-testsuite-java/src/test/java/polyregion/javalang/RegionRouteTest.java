package polyregion.javalang;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.io.File;
import java.io.IOException;
import java.net.JarURLConnection;
import java.net.URISyntaxException;
import java.net.URL;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.List;
import java.util.jar.JarEntry;
import org.junit.Test;

public class RegionRouteTest {

  private static final String PACKAGE = "polyregion/$gen$/";

  private static List<String> generated() throws IOException, URISyntaxException {
    URL directory = RegionRouteTest.class.getClassLoader().getResource(PACKAGE);
    assertTrue("no generated classes found", directory != null);
    List<String> names = new ArrayList<>();
    if (directory.getProtocol().equals("jar")) {
      Enumeration<JarEntry> entries = ((JarURLConnection) directory.openConnection()).getJarFile().entries();
      while (entries.hasMoreElements()) {
        String name = entries.nextElement().getName();
        if (name.startsWith(PACKAGE) && name.endsWith(".class")) names.add(name);
      }
    } else {
      File[] files = new File(directory.toURI()).listFiles((dir, name) -> name.endsWith(".class"));
      if (files != null) for (File file : files) names.add(PACKAGE + file.getName());
    }
    return names;
  }

  @Test
  public void everyGeneratedClassLaunchesThroughTheCore() throws Exception {
    List<String> names = generated();
    assertTrue("no generated classes found", !names.isEmpty());
    int regions = 0;
    for (String name : names) {
      Class<?> generated = Class.forName(name.substring(0, name.length() - ".class".length()).replace('/', '.'));
      assertTrue(generated + " is not a launch site", LaunchSite.class.isAssignableFrom(generated));
      if (LaunchRegion.class.isAssignableFrom(generated)) regions++;
    }
    assertTrue("no region classes among the generated ones", regions > 0);
  }

  @Test
  public void theBinaryRegionPipelineIsGone() {
    try {
      Class.forName("polyregion.javalang.BinaryOffloadExecutable");
    } catch (ClassNotFoundException expected) {
      assertEquals("polyregion.javalang.BinaryOffloadExecutable", expected.getMessage());
      return;
    }
    fail("BinaryOffloadExecutable is still on the classpath");
  }
}
