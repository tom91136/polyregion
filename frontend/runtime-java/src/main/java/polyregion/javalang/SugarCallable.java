package polyregion.javalang;

import java.io.Serializable;
import java.lang.invoke.SerializedLambda;
import java.util.ArrayList;
import java.util.List;

/**
 * Carries an indexed region callable and its extents to the launch. The serialised form keeps the
 * user's implementation signature: the compile-time site key needs only the implementation class and
 * name, and the environment reads the captures followed by the extents.
 */
public final class SugarCallable implements Serializable {

  private static final long serialVersionUID = 1L;

  private final SerializedLambda original;
  private final Object[] extents;

  public SugarCallable(SerializedLambda original, int rank, Object[] extents) {
    List<String> parameters = parameters(original.getImplMethodSignature());
    if (rank < 0 || parameters.size() != rank + original.getCapturedArgCount()) refuse(original, rank);
    for (int i = 0; i < rank; i++)
      if (!"J".equals(parameters.get(parameters.size() - rank + i))) refuse(original, rank);
    this.original = original;
    this.extents = extents;
  }

  private Object writeReplace() {
    int captures = original.getCapturedArgCount();
    Object[] all = new Object[captures + extents.length];
    for (int i = 0; i < captures; i++) all[i] = original.getCapturedArg(i);
    System.arraycopy(extents, 0, all, captures, extents.length);
    return new SerializedLambda(
        SugarCallable.class,
        original.getFunctionalInterfaceClass(),
        original.getFunctionalInterfaceMethodName(),
        original.getFunctionalInterfaceMethodSignature(),
        original.getImplMethodKind(),
        original.getImplClass(),
        original.getImplMethodName(),
        original.getImplMethodSignature(),
        original.getInstantiatedMethodType(),
        all);
  }

  private static void refuse(SerializedLambda lambda, int rank) {
    throw new IllegalArgumentException(
        "unsupported sugar callable shape: "
            + lambda.getImplMethodSignature()
            + " with rank "
            + rank
            + " and "
            + lambda.getCapturedArgCount()
            + " captures");
  }

  private static List<String> parameters(String signature) {
    List<String> found = new ArrayList<>();
    int at = signature.indexOf('(') + 1;
    while (at < signature.length() && signature.charAt(at) != ')') {
      int start = at;
      while (signature.charAt(at) == '[') at++;
      if (signature.charAt(at) == 'L') {
        while (signature.charAt(at) != ';') at++;
        at++;
      } else {
        at++;
      }
      found.add(signature.substring(start, at));
    }
    return found;
  }
}
