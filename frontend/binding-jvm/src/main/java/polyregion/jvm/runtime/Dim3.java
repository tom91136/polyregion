package polyregion.jvm.runtime;

import java.util.Objects;

public final class Dim3 {
  public final long x, y, z;

  public Dim3(long x, long y, long z) {
    this.x = x;
    this.y = y;
    this.z = z;
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;
    Dim3 dim3 = (Dim3) o;
    return x == dim3.x && y == dim3.y && z == dim3.z;
  }

  @Override
  public int hashCode() {
    return Objects.hash(x, y, z);
  }

  @Override
  public String toString() {
    return "Dim3{" + "x=" + x + ", y=" + y + ", z=" + z + '}';
  }
}
