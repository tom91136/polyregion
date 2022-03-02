package polyregion.loader;

import java.io.IOException;
import java.nio.file.FileAlreadyExistsException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.LinkedHashSet;
import java.util.Set;
import java.util.stream.Stream;
import java.util.Comparator;
import java.util.concurrent.atomic.AtomicBoolean;

public class Loader {

  private Loader() {
    throw new AssertionError();
  }

  public static final Path HOME_DIR = Paths.get(System.getProperty("user.home")).toAbsolutePath();
  private static final int LOAD_DIRECT_MAX_LINK_ATTEMPT = 10;

  enum UnlinkOnVmExitHook {
    INSTANCE;
    private final Set<Path> toDelete = new LinkedHashSet<>();

    void deleteSilently(Path p) {
      try {
        System.out.println(" - " + p);
        Files.deleteIfExists(p);
      } catch (IOException ignored) {
      }
    }

    void deleteAllSilently(Path root) {
      System.out.println("Deleting " + root);
      try (Stream<Path> walk = Files.walk(root)) {
        walk.sorted(Comparator.reverseOrder()).forEach(this::deleteSilently);
      } catch (IOException ignored) {

      }
    }

    {
      java.lang.Runtime.getRuntime()
          .addShutdownHook(
              new Thread(
                  () -> {
                    toDelete.forEach(this::deleteAllSilently);
                  }));
    }

    void mark(Path path) {
      toDelete.add(path);
    }
  }

  public static void loadResource(String resource, Path destination) throws IOException {
    var stream = ClassLoader.getSystemResourceAsStream(resource);
    Files.copy(stream, destination);
  }

  public static void loadDirect(Path path, Path workDir) {
    System.out.println(">>" + Loader.class);
    try {
      Files.createDirectories(workDir);
    } catch (IOException ioe) {
      throw new RuntimeException(
          "Cannot create work directory " + workDir + " to extract native libraries", ioe);
    }

    Path absolute = path.toAbsolutePath();
    int attempt = 0;

    Path libPath = absolute;
    while (attempt < LOAD_DIRECT_MAX_LINK_ATTEMPT) {
      try {
        System.out.println("load native:" + libPath);
        System.load(libPath.toString());
        System.out.println("load native OK");
        return;
      } catch (UnsatisfiedLinkError e) {
        System.out.println(e.getMessage());
        if (e.getMessage().endsWith("already loaded in another classloader")) {
          var link =
              workDir.resolve(
                  absolute.getFileName()
                      + "."
                      + System.currentTimeMillis()
                      + ".hardlink."
                      + attempt);
          try {
            link = Files.createLink(link, absolute);
          } catch (FileAlreadyExistsException ignored) {
            attempt++;
            continue;
          } catch (IOException ioe) {
            throw new RuntimeException("Cannot load native library " + path, ioe);
          }
          UnlinkOnVmExitHook.INSTANCE.mark(link);
          System.out.println("creating symlink: " + link);
          libPath = link;
        } else {
          throw e;
        }
        attempt++;
      }
    }
    throw new RuntimeException(
        "Unable to load native library " + path + " after " + attempt + " attempts");
  }
}
