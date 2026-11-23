package polyregion.javalang;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import javax.tools.JavaCompiler;
import javax.tools.ToolProvider;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;
import polyregion.jvm.runtime.PolyregionRuntimeException;

public class RuntimeLoweringTest {

  private static final String SOURCE =
      String.join(
          "\n",
          "package external;",
          "import polyregion.javalang.OffloadFunction;",
          "import polyregion.javalang.OffloadRunnable;",
          "public final class Kernels {",
          "  public static final class Box { public int v = 1; }",
          "  public static OffloadRunnable scale(int[] xs, int k) {",
          "    return () -> { for (int i = 0; i < 8; i++) xs[i] = xs[i] * k + i; };",
          "  }",
          "  public static OffloadRunnable saxpy(double[] ys, double[] xs, double a, int n) {",
          "    return () -> { for (int i = 0; i < n; i++) ys[i] = a * xs[i] + ys[i]; };",
          "  }",
          "  public static OffloadFunction<Integer> total(int[] xs, int n) {",
          "    return () -> { int s = 0; for (int i = 0; i < n; i++) s += xs[i]; return s; };",
          "  }",
          "  public static OffloadRunnable touch(Box box) {",
          "    return () -> { box.v = box.v + 1; };",
          "  }",
          "  public static OffloadRunnable branchy(int[] out, int n) {",
          "    return () -> { for (int i = 0; i < n; i++) out[i] = (i % 3 == 0) ? helper(i) : -i;"
              + " };",
          "  }",
          "  static int helper(int i) { return i * i + 1; }",
          "}",
          "");

  private static Path work;
  private static URLClassLoader loader;

  @BeforeClass
  public static void compileWithoutTheProcessor() throws IOException {
    work = Files.createTempDirectory("polyregion-runtime-lowering");
    Path source = work.resolve("external").resolve("Kernels.java");
    Files.createDirectories(source.getParent());
    Files.write(source, SOURCE.getBytes("UTF-8"));
    Path classes = Files.createDirectory(work.resolve("classes"));
    JavaCompiler javac = ToolProvider.getSystemJavaCompiler();
    assertNotNull("No system Java compiler", javac);
    int status =
        javac.run(
            null,
            null,
            null,
            "-proc:none",
            "-nowarn",
            "-cp",
            System.getProperty("polyregion.hostClasspath"),
            "-d",
            classes.toString(),
            source.toString());
    assertEquals("javac failed", 0, status);
    loader =
        new URLClassLoader(
            new URL[] {classes.toUri().toURL()}, RuntimeLoweringTest.class.getClassLoader());
  }

  @AfterClass
  public static void cleanUp() throws IOException {
    loader.close();
  }

  private static Object call(String method, Class<?>[] types, Object... args) throws Exception {
    try {
      return loader
          .loadClass("external.Kernels")
          .getDeclaredMethod(method, types)
          .invoke(null, args);
    } catch (InvocationTargetException e) {
      throw (Exception) e.getCause();
    }
  }

  private static OffloadRunnable runnable(String method, Class<?>[] types, Object... args)
      throws Exception {
    return (OffloadRunnable) call(method, types, args);
  }

  @Test
  public void aLambdaFromAClassCompiledWithoutTheProcessorLaunches() throws Exception {
    assertEquals(
        "the class must come from the loader that compiled it without the processor",
        loader,
        loader.loadClass("external.Kernels").getClassLoader());
    int[] xs = {1, 2, 3, 4, 5, 6, 7, 8};
    int[] expected = new int[8];
    for (int i = 0; i < 8; i++) expected[i] = xs[i] * 3 + i;
    Runtime.offload(runnable("scale", new Class<?>[] {int[].class, int.class}, xs, 3));
    assertArrayEquals(expected, xs);
  }

  @Test
  public void doublesAndMixedScalarsLaunch() throws Exception {
    double[] xs = {1, 2, 3, 4, 5, 6};
    double[] ys = {10, 20, 30, 40, 50, 60};
    double[] expected = new double[6];
    for (int i = 0; i < 6; i++) expected[i] = 2.5 * xs[i] + ys[i];
    Runtime.offload(
        runnable(
            "saxpy",
            new Class<?>[] {double[].class, double[].class, double.class, int.class},
            ys,
            xs,
            2.5,
            6));
    assertArrayEquals(expected, ys, 0.0);
  }

  @Test
  public void aHelperCallAndABranchLaunch() throws Exception {
    int[] out = new int[10];
    int[] expected = new int[10];
    for (int i = 0; i < 10; i++) expected[i] = (i % 3 == 0) ? i * i + 1 : -i;
    Runtime.offload(runnable("branchy", new Class<?>[] {int[].class, int.class}, out, 10));
    assertArrayEquals(expected, out);
  }

  @Test
  @SuppressWarnings("unchecked")
  public void aResultComesBack() throws Exception {
    int[] xs = {1, 2, 3, 4, 5};
    OffloadFunction<Integer> total =
        (OffloadFunction<Integer>) call("total", new Class<?>[] {int[].class, int.class}, xs, 5);
    assertEquals(15, (int) Runtime.offload(total));
  }

  @Test
  public void aStoredLambdaCompilesOnceAndSeesNewCaptures() throws Exception {
    int[] xs = {1, 1, 1, 1, 1, 1, 1, 1};
    OffloadRunnable first = runnable("scale", new Class<?>[] {int[].class, int.class}, xs, 2);
    long before = Launch.LOWERINGS.get();
    Runtime.offload(first);
    Runtime.offload(first);
    int[] ys = {2, 2, 2, 2, 2, 2, 2, 2};
    Runtime.offload(runnable("scale", new Class<?>[] {int[].class, int.class}, ys, 5));
    long lowered = Launch.LOWERINGS.get() - before;
    assertTrue("`scale` was lowered " + lowered + " times, expected at most once", lowered <= 1);
    int[] twice = new int[8];
    int[] once = new int[8];
    for (int i = 0; i < 8; i++) {
      twice[i] = (1 * 2 + i) * 2 + i;
      once[i] = 2 * 5 + i;
    }
    assertArrayEquals(twice, xs);
    assertArrayEquals(once, ys);
  }

  @Test
  public void anObjectCaptureIsRefusedWithADiagnostic() throws Exception {
    Class<?> box = loader.loadClass("external.Kernels$Box");
    OffloadRunnable touch =
        runnable("touch", new Class<?>[] {box}, box.getDeclaredConstructor().newInstance());
    try {
      Runtime.offload(touch);
      fail("an object capture should be refused");
    } catch (PolyregionRuntimeException e) {
      assertTrue(e.getMessage(), e.getMessage().contains("not supported at run time"));
    }
    try {
      Runtime.offload(touch);
      fail("the refusal should repeat");
    } catch (PolyregionRuntimeException e) {
      assertTrue(Arrays.asList(e.getMessage()).toString(), e.getMessage().contains("touch"));
    }
  }
}
