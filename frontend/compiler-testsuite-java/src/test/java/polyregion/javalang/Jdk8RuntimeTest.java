package polyregion.javalang;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeTrue;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.junit.Test;

// binding-jvm and runtime-java must load on a real Java 8, which the class-file version alone does
// not prove
public class Jdk8RuntimeTest {

  private static final String DEFAULT_JDK8 =
      System.getProperty("user.home") + "/.jdks/temurin-1.8.0_502";

  private static final String PROBE = "Jdk8Probe";

  private static final String SOURCE =
      String.join(
          "\n",
          "import polyregion.javalang.OffloadFunction;",
          "import polyregion.javalang.OffloadRunnable;",
          "import polyregion.javalang.Runtime;",
          "",
          "public final class " + PROBE + " {",
          "  public static void main(String[] args) {",
          "    int a = 3;",
          "    int b = 4;",
          "    OffloadFunction<Integer> scalar = () -> a * b + 1;",
          "    int[][] grid = {{1, 2}, {3, 4}};",
          "    OffloadFunction<Integer> nested = () -> grid[0][0] + grid[1][1];",
          "    int[] xs = new int[4];",
          "    OffloadRunnable write = () -> { for (int i = 0; i < 4; i++) xs[i] = i * 2; };",
          "    Runtime.offload(write);",
          "    StringBuilder sb = new StringBuilder();",
          "    for (int i = 0; i < xs.length; i++) sb.append(i == 0 ? \"\" : \",\").append(xs[i]);",
          "    System.out.println(\"probe scalar=\" + Runtime.offload(scalar)",
          "        + \" nested=\" + Runtime.offload(nested) + \" written=\" + sb);",
          "  }",
          "}",
          "");

  private static String jdk8Home() {
    String property = System.getProperty("polyregion.jdk8Home");
    if (property != null && !property.isEmpty()) return property;
    String env = System.getenv("POLYREGION_JDK8_HOME");
    return env != null && !env.isEmpty() ? env : DEFAULT_JDK8;
  }

  private static String run(List<String> command) throws IOException, InterruptedException {
    ProcessBuilder pb = new ProcessBuilder(command).redirectErrorStream(true);
    Process p = pb.start();
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    byte[] buffer = new byte[8192];
    try (InputStream in = p.getInputStream()) {
      for (int n = in.read(buffer); n >= 0; n = in.read(buffer)) out.write(buffer, 0, n);
    }
    String output = new String(out.toByteArray(), StandardCharsets.UTF_8);
    assertEquals("`" + command.get(0) + "` failed:\n" + output, 0, p.waitFor());
    return output;
  }

  @Test
  public void regionRunsOnJava8() throws IOException, InterruptedException {
    Path java8 = Paths.get(jdk8Home(), "bin", "java");
    assumeTrue(
        "No Java 8 at " + java8 + "; set -Dpolyregion.jdk8Home or POLYREGION_JDK8_HOME to run this",
        Files.isExecutable(java8));

    String artefacts = System.getProperty("polyregion.runtimeArtefacts");
    assertTrue(
        "polyregion.runtimeArtefacts is unset, so the Java 8 classpath cannot be built",
        artefacts != null && !artefacts.isEmpty());

    Path work = Files.createTempDirectory("polyregion-jdk8");
    Path source = work.resolve(PROBE + ".java");
    Path classes = Files.createDirectory(work.resolve("classes"));
    Files.write(source, SOURCE.getBytes(StandardCharsets.UTF_8));

    String hostClasspath = System.getProperty("polyregion.hostClasspath");
    Path javac = Paths.get(System.getProperty("java.home"), "bin", "javac");
    assertTrue("No javac at " + javac, Files.isExecutable(javac));
    run(
        Arrays.asList(
            javac.toString(),
            "--release",
            "8",
            "-proc:full",
            "-nowarn",
            "-cp",
            hostClasspath,
            "-processorpath",
            hostClasspath,
            "-d",
            classes.toString(),
            source.toString()));

    List<String> command =
        new ArrayList<>(
            Arrays.asList(
                java8.toString(), "-cp", classes + File.pathSeparator + artefacts, PROBE));
    String output = run(command);
    assertTrue(
        "Java 8 run did not produce the expected values:\n" + output,
        output.contains("probe scalar=13 nested=5 written=0,2,4,6"));
  }
}
