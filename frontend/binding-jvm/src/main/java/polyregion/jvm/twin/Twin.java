package polyregion.jvm.twin;

import java.nio.ByteBuffer;

/**
 * Copies one host class to and from its device image.
 *
 * <p>Frontends emit one subclass per class at compile time, reading and writing fields directly.
 * {@link ReflectiveTwin} stands in for a class the generated code cannot reach.
 */
public abstract class Twin {

  public abstract Layout layout();

  /**
   * Writes the fields of {@code root} into {@code dst} at {@code at}, at the offsets of {@link
   * #layout}. Arrays are staged through {@code m} and stored as device pointers, nested objects are
   * written in place.
   */
  public abstract void write(Marshal m, Object root, ByteBuffer dst, int at);

  /** Copies the fields the device may have changed from {@code src} at {@code at} back. */
  public abstract void read(Marshal m, Object root, ByteBuffer src, int at);
}
