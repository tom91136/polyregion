package polyregion.jvm.twin;

public class TwinException extends RuntimeException {
  private static final long serialVersionUID = 1L;

  public TwinException(String message) {
    super(message);
  }

  public TwinException(String message, Throwable cause) {
    super(message, cause);
  }
}
