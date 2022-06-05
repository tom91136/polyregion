package polyregion.jvm;

import java.util.Objects;

@SuppressWarnings("unused")
public class PolyregionLoaderException extends RuntimeException {
  private static final long serialVersionUID = -5103414923372376140L;

  public PolyregionLoaderException(String message) {
    super(Objects.requireNonNull(message));
  }
}
