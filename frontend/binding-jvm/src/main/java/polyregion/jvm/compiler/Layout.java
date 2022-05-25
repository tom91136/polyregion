package polyregion.jvm.compiler;

import java.util.Arrays;

public final class Layout {

  public final String[] name;
  public final long sizeInBytes;
  public final long alignment;
  public final Member[] members;

  Layout(String[] name, long sizeInBytes, long alignment, Member[] members) {
    this.name = name;
    this.sizeInBytes = sizeInBytes;
    this.alignment = alignment;
    this.members = members;
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
