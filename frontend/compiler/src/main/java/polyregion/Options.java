package polyregion;

public final class Options {

  public final byte target;
  public final String arch;

  public Options(byte target, String arch) {
    this.target = target;
    this.arch = arch;
  }

  @Override
  public String toString() {
    return "Options{" + "target=" + target + ", arch='" + arch + '\'' + '}';
  }
}
