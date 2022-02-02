package polyregion;

import java.util.Arrays;

public final class Layout {
  public String[] name;
  public long sizeInBytes;
  public long alignment;
  public Member[] members;

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
