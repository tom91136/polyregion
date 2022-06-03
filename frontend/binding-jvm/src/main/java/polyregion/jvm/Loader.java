package polyregion.jvm;

import java.io.IOException;
import java.io.InputStream;
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

  // Taken from https://github.com/bytedeco/javacpp/blob/3d0256c2cea4e931b655b2dc90c9f5f0d2100eeb/src/main/java/org/bytedeco/javacpp/Loader.java#L97-L136
  static String resolvePlatform() {
    String jvmName = System.getProperty("java.vm.name", "").toLowerCase();
    String osName = System.getProperty("os.name", "").toLowerCase();
    String osArch = System.getProperty("os.arch", "").toLowerCase();
    String abiType = System.getProperty("sun.arch.abi", "").toLowerCase();
    String libPath = System.getProperty("sun.boot.library.path", "").toLowerCase();
    if (jvmName.startsWith("dalvik") && osName.startsWith("linux")) {
      osName = "android";
    } else if (jvmName.startsWith("robovm") && osName.startsWith("darwin")) {
      osName = "ios";
      osArch = "arm";
    } else if (osName.startsWith("mac os x") || osName.startsWith("darwin")) {
      osName = "macosx";
    } else {
      int spaceIndex = osName.indexOf(' ');
      if (spaceIndex > 0) {
        osName = osName.substring(0, spaceIndex);
      }
    }
    if (osArch.equals("i386")
        || osArch.equals("i486")
        || osArch.equals("i586")
        || osArch.equals("i686")) {
      osArch = "x86";
    } else if (osArch.equals("amd64") || osArch.equals("x86-64") || osArch.equals("x64")) {
      osArch = "x86_64";
    } else if (osArch.startsWith("aarch64")
        || osArch.startsWith("armv8")
        || osArch.startsWith("arm64")) {
      osArch = "arm64";
    } else if ((osArch.startsWith("arm"))
        && ((abiType.equals("gnueabihf")) || (libPath.contains("openjdk-armhf")))) {
      osArch = "armhf";
    } else if (osArch.startsWith("arm")) {
      osArch = "arm";
    }
    return osName + "-" + osArch;
  }

  public static void loadResource(String resource, Path destination) throws IOException {
    InputStream stream = ClassLoader.getSystemResourceAsStream(resource);
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
          Path link =
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
