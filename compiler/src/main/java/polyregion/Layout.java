package polyregion;

import java.util.Arrays;

public final class Layout {
  public long sizeInBytes;
  public long alignment;
  public Member[] members;

  @Override
  public String toString() {
    return "Layout{"
        + "size="
        + sizeInBytes
        + ", alignment="
        + alignment
        + ", members="
        + Arrays.toString(members)
        + "}";
  }
}
