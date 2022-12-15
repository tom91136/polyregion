package polyregion.jvm.runtime;

import java.util.Objects;
import java.util.Optional;

@SuppressWarnings("unused")
public final class Policy {
  public final Dim3 global;
  final Dim3 local;
  public final int localMemoryBytes;

  public Policy(Dim3 global, Dim3 local, int localMemoryBytes) {
    this.global = Objects.requireNonNull(global);
    this.local = Objects.requireNonNull(local);
    this.localMemoryBytes = localMemoryBytes;
  }

  public Policy(Dim3 global, Dim3 local ) {
    this(global, local, 0);
  }

  public Policy(Dim3 global) {
    this.global = Objects.requireNonNull(global);
    this.local = null;
    this.localMemoryBytes = 0;
  }

  public Optional<Dim3> local() {
    return Optional.ofNullable(local);
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;
    Policy policy = (Policy) o;
    return global.equals(policy.global) && Objects.equals(local, policy.local);
  }

  @Override
  public int hashCode() {
    return Objects.hash(global, local);
  }

  @Override
  public String toString() {
    return "Policy{"
        + "global="
        + global
        + ", local="
        + (local == null ? "(none)" : local)
        + ", localMemoryBytes="
        + localMemoryBytes
        + '}';
  }
}
