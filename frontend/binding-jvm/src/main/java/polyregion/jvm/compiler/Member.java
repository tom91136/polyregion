package polyregion.jvm.compiler;

import java.util.Objects;

public final class Member {

  public final String name;
  public final long offsetInBytes, sizeInBytes;

  Member(String name, long offsetInBytes, long sizeInBytes) {
    this.name = Objects.requireNonNull(name);
    this.offsetInBytes = offsetInBytes;
    this.sizeInBytes = sizeInBytes;
  }

  @Override
  public String toString() {
    return "Member{" + name + ", offset=" + offsetInBytes + ", size=" + sizeInBytes + "}";
  }
}
