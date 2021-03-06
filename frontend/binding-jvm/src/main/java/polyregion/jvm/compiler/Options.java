package polyregion.jvm.compiler;

import java.util.Objects;

@SuppressWarnings("unused")
public final class Options {

  public final byte target;
  public final String arch;

  public static Options of(Target target, String arch) {
    return new Options(target.value, arch);
  }

  Options(byte target, String arch) {
    this.target = target;
    this.arch = Objects.requireNonNull(arch);
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;
    Options options = (Options) o;
    return target == options.target && Objects.equals(arch, options.arch);
  }

  @Override
  public int hashCode() {
    return Objects.hash(target, arch);
  }

  @Override
  public String toString() {
    return "Options{" + "target=" + target + ", arch='" + arch + '\'' + '}';
  }
}
