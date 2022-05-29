package polyregion;

import java.util.Objects;

public class PolyregionRuntimeException extends RuntimeException {
  public PolyregionRuntimeException(String message) {
    super(Objects.requireNonNull(message));
  }
}
