package polyregion.jvm;

import java.io.IOException;
import java.net.URL;
import java.nio.file.FileAlreadyExistsException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Stream;

@SuppressWarnings("unused")
public final class Loader {

  private Loader() {
    throw new AssertionError();
  }

  static final Path HOME_DIR = Paths.get(System.getProperty("user.home")).toAbsolutePath();
  static final Path RESOURCE_DIR = HOME_DIR.resolve(".polyregion");

  public static void touch() {
    try {
      Class.forName("polyregion.jvm.Loader");
    } catch (ClassNotFoundException e) {
      throw new AssertionError(e);
    }
  }

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
      Runtime.getRuntime()
          .addShutdownHook(new Thread(() -> toDelete.forEach(this::deleteAllSilently)));
    }

    void mark(Path path) {
      toDelete.add(path);
    }
  }

  // Taken from
  // https://github.com/bytedeco/javacpp/blob/3d0256c2cea4e931b655b2dc90c9f5f0d2100eeb/src/main/java/org/bytedeco/javacpp/Loader.java#L97-L136
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

  public static Optional<Path> searchAndCopyResourceIfNeeded(String filename, Path pwd) {

    String platform = resolvePlatform();

    // TODO remove for prod
    List<Path> paths =
        new ArrayList<>(
            Arrays.asList(
                //      Paths.get("../native/build-" + platform + "/bindings/jvm/"),
                Paths.get("../native/cmake-build-debug-clang/bindings/jvm/"),
                Paths.get("../native/cmake-build-release-clang/bindings/jvm/"),
                Paths.get("../../native/build-" + platform + "/bindings/jvm/"),
                Paths.get("../../native/cmake-build-release-clang/bindings/jvm/"),
                Paths.get("../../native/cmake-build-debug-clang/bindings/jvm/")));

    // TODO remove for prod; use debug first for compiler
    if (filename.contains("compiler")) {
      paths.add(0, Paths.get("../native/cmake-build-debug-clang/bindings/jvm/"));
    }

    String[] resourcePaths = {
      platform + "/", "",
    };

    for (Path p : paths) {
      Path name = pwd.resolve(p).resolve(filename);
      if (Files.isRegularFile(name)) {
        return Optional.of(name.normalize().toAbsolutePath());
      }
    }
    try {
      for (String resourcePath : resourcePaths) {
        String resource = resourcePath + filename;
        URL url = ClassLoader.getSystemResource(resource);
        if (url != null) {
          Path destination =
              Files.createDirectories(RESOURCE_DIR).resolve(filename).normalize().toAbsolutePath();
          Files.copy(url.openStream(), destination, StandardCopyOption.REPLACE_EXISTING);
          return Optional.of(destination);
        }
      }
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
    return Optional.empty();
  }

  // This is needed for incremental *compile* environments like SBT where VM don't exit after
  // compilation but instead loads all the classes again with a new ClassLoader.
  // When a new ClassLoader is used, we get a UnsatisfiedLinkError if old ClassLoader has not been
  // GC'ed.
  static Path loadLibrary(Path path) {
    // XXX We need to suggest the VM to GC here because it increases the chance JNI_OnUnload
    // gets called.
    System.gc();
    Path absolute = path.toAbsolutePath();
    Path libPath = absolute;
    int attempt = 0;
    while (attempt < 10) {
      try {
        System.out.println("[Loader] Load native:" + libPath);
        System.load(libPath.toString());
        System.out.println("[Loader] Loaded native:" + libPath);
        return libPath;
      } catch (UnsatisfiedLinkError e) {
        if (!e.getMessage().endsWith("already loaded in another classloader")) throw e;

        Path link =
            RESOURCE_DIR.resolve(
                absolute.getFileName() + "." + System.currentTimeMillis() + ".hardlink." + attempt);
        try {
          try {
            // This could fail if link and absolute is on different FS/drives.
            Files.createLink(link, absolute);
          } catch (IOException ignored) {
            // If it does fail, we just create a copy instead.
            Files.copy(absolute, link, StandardCopyOption.REPLACE_EXISTING);
          }
        } catch (FileAlreadyExistsException ignored) {
          attempt++;
          continue;
        } catch (IOException ioe) {
          throw new RuntimeException(
              "Cannot load native library even with hardlink workaround " + path, ioe);
        }
        UnlinkOnVmExitHook.INSTANCE.mark(link);
        System.out.println("Created symlink: " + link);
        libPath = link;
        attempt++;
      }
    }
    throw new RuntimeException(
        "Unable to load native library " + path + " after " + attempt + " attempts");
  }

  static {
    // Load the basic dl native library first so that we can load and unload other native libraries.
    String name = "libpolyregion-shim-jvm.so";
    Path pwd = Paths.get(".").toAbsolutePath();
    Path path =
        searchAndCopyResourceIfNeeded(name, pwd)
            .orElseThrow(
                () ->
                    new RuntimeException(
                        "Cannot load library "
                            + name
                            + ", search paths exhausted, working directory is "
                            + pwd));

    Path actualPath = loadLibrary(path);
    if (path != actualPath) {
      Natives.registerFilesToDropOnUnload0(actualPath.toFile());
    }

    System.out.println(name + " loaded (" + path + ")");
  }
}
