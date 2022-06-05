package polyregion.jvm.compiler;

import java.util.Objects;

@SuppressWarnings("unused")
public class PolyregionCompilerException extends RuntimeException {
  private static final long serialVersionUID = 2500790194596475117L;

  public PolyregionCompilerException(String message) {
    super(Objects.requireNonNull(message));
  }
}
