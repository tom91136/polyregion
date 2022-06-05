package polyregion.jvm.runtime;

import java.util.Objects;

@SuppressWarnings("unused")
public class PolyregionRuntimeException extends RuntimeException {
  private static final long serialVersionUID = 3947179116958396648L;

  public PolyregionRuntimeException(String message) {
    super(Objects.requireNonNull(message));
  }
}
