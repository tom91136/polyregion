package polyregion.javalang;

import java.io.Serializable;
import java.lang.invoke.SerializedLambda;

/**
 * Carries an indexed region callable and its extents to the launch. The serialised form names the
 * user's implementation method, so the launch site generated at compile time is found unchanged;
 * the extents follow the user's captures in the environment.
 */
public final class SugarCallable implements Serializable {

  private final SerializedLambda original;
  private final int rank;
  private final Object[] extents;

  public SugarCallable(SerializedLambda original, int rank, Object[] extents) {
    this.original = original;
    this.rank = rank;
    this.extents = extents;
  }

  private Object writeReplace() {
    int count = original.getCapturedArgCount();
    Object[] all = new Object[count + extents.length];
    for (int i = 0; i < count; i++) all[i] = original.getCapturedArg(i);
    System.arraycopy(extents, 0, all, count, extents.length);
    return new SerializedLambda(
        SugarCallable.class,
        original.getFunctionalInterfaceClass(),
        original.getFunctionalInterfaceMethodName(),
        original.getFunctionalInterfaceMethodSignature(),
        original.getImplMethodKind(),
        original.getImplClass(),
        original.getImplMethodName(),
        withoutIndices(original.getImplMethodSignature(), rank),
        withoutIndices(original.getInstantiatedMethodType(), rank),
        all);
  }

  private static String withoutIndices(String signature, int rank) {
    if (rank == 0) return signature;
    int close = signature.lastIndexOf(')');
    StringBuilder params = new StringBuilder(signature.substring(1, close));
    for (int i = 0; i < rank; i++) {
      int end = params.length();
      if (end == 0 || params.charAt(end - 1) != 'J') return signature;
      params.setLength(end - 1);
    }
    return "(" + params + "J)" + signature.substring(close + 1);
  }
}
