package polyregion.javalang;

import java.lang.invoke.SerializedLambda;

/**
 * Lowers a lambda that has no generated launch site, from the class file of its implementation
 * method, at run time. Implementations are discovered with {@link java.util.ServiceLoader} and ship
 * in the optional {@code runtime-java-jit} artefact; without one on the class path, such a lambda
 * cannot launch.
 */
public interface RuntimeLowering {

  /** The compiled form of one lambda: what a generated launch site class would carry. */
  final class Compiled {
    public final String descriptor;
    public final byte[] binary;

    public Compiled(String descriptor, byte[] binary) {
      this.descriptor = descriptor;
      this.binary = binary;
    }
  }

  /** Everything besides the lambda that changes the compiled result, e.g. targets and libraries. */
  String configuration();

  /**
   * @throws polyregion.jvm.runtime.PolyregionRuntimeException when the method cannot be lowered
   */
  Compiled lower(SerializedLambda lambda, ClassLoader loader);
}
