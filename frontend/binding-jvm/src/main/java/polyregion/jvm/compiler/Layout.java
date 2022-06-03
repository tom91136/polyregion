package polyregion.jvm.compiler;

import java.util.Arrays;
import java.util.Objects;

public final class Layout {

  public static final class Member {

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

  public final String[] name;
  public final long sizeInBytes;
  public final long alignment;
  public final Member[] members;

  Layout(String[] name, long sizeInBytes, long alignment, Member[] members) {
    this.name = Objects.requireNonNull(name);
    this.sizeInBytes = sizeInBytes;
    this.alignment = alignment;
    this.members = Objects.requireNonNull(members);
  }

  @Override
  public String toString() {
    return "Layout{"
        + "name="
        + Arrays.toString(members)
        + ", size="
        + sizeInBytes
        + ", alignment="
        + alignment
        + ", members="
        + Arrays.toString(members)
        + "}";
  }
}
