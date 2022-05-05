package polyregion;

public final class Member {
  public String name;
  public long offsetInBytes;
  public long sizeInBytes;

  @Override
  public String toString() {
    return "Member{" + name + ", offset=" + offsetInBytes + ", size=" + sizeInBytes + "}";
  }
}
