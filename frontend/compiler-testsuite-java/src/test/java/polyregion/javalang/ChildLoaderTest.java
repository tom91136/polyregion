package polyregion.javalang;

import static org.junit.Assert.assertEquals;

import java.io.IOException;
import java.io.InputStream;
import java.io.ByteArrayOutputStream;
import java.util.concurrent.Callable;
import org.junit.Test;

public class ChildLoaderTest {

  private static final class ChildFirst extends ClassLoader {
    ChildFirst(ClassLoader parent) {
      super(parent);
    }

    @Override
    protected Class<?> loadClass(String name, boolean resolve) throws ClassNotFoundException {
      synchronized (getClassLoadingLock(name)) {
        Class<?> c = findLoadedClass(name);
        if (c == null && (name.contains("ChildLoaderHarness") || name.contains("ChildLoaderCell"))) {
          try (InputStream in =
              getParent().getResourceAsStream(name.replace('.', '/') + ".class")) {
            ByteArrayOutputStream out = new ByteArrayOutputStream();
            byte[] buffer = new byte[8192];
            for (int n = in.read(buffer); n >= 0; n = in.read(buffer)) out.write(buffer, 0, n);
            c = defineClass(name, out.toByteArray(), 0, out.size());
          } catch (IOException e) {
            throw new ClassNotFoundException(name, e);
          }
        }
        return c != null ? c : super.loadClass(name, resolve);
      }
    }
  }

  @Test
  public void aRegionOfAChildLoaderLaunchesThroughItsOwnClasses() throws Exception {
    ClassLoader child = new ChildFirst(ChildLoaderTest.class.getClassLoader());
    Class<?> harness = child.loadClass(ChildLoaderHarness.class.getName());
    assertEquals(child, harness.getClassLoader());
    Callable<?> call = (Callable<?>) harness.getDeclaredConstructor().newInstance();
    assertEquals(7, call.call());
  }
}
