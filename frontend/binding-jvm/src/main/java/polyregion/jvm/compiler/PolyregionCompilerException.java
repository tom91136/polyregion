package polyregion.jvm.compiler;

import java.util.Objects;

public class PolyregionCompilerException extends RuntimeException {
  public PolyregionCompilerException(String message) {
    super(Objects.requireNonNull(message));
  }
}
