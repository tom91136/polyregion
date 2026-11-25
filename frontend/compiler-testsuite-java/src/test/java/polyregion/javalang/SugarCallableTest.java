package polyregion.javalang;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

import java.lang.invoke.SerializedLambda;
import java.lang.reflect.Method;
import org.junit.Test;

public class SugarCallableTest {

  private static SerializedLambda lambda(String signature, Object[] captures) {
    return new SerializedLambda(
        SugarCallableTest.class,
        "polyregion/Offload$FnJJ_V",
        "apply",
        "(JJ)V",
        6,
        "polyregion/NdHelper",
        "lambda$fill$1",
        signature,
        "(JJ)V",
        captures);
  }

  private static SerializedLambda replace(SugarCallable callable) throws Exception {
    Method writeReplace = SugarCallable.class.getDeclaredMethod("writeReplace");
    writeReplace.setAccessible(true);
    return (SerializedLambda) writeReplace.invoke(callable);
  }

  @Test
  public void captureArrayAndExtentsKeepTheUserSignature() throws Exception {
    int[] out = new int[6];
    SerializedLambda original = lambda("([IJJJ)V", new Object[] {out, 3L});
    SerializedLambda rewritten = replace(new SugarCallable(original, 2, new Object[] {3L, 2L}));
    assertEquals(original.getImplMethodSignature(), rewritten.getImplMethodSignature());
    assertEquals(original.getInstantiatedMethodType(), rewritten.getInstantiatedMethodType());
    assertEquals(4, rewritten.getCapturedArgCount());
    assertArrayEquals(out, (int[]) rewritten.getCapturedArg(0));
    assertEquals(3L, rewritten.getCapturedArg(1));
    assertEquals(3L, rewritten.getCapturedArg(2));
    assertEquals(2L, rewritten.getCapturedArg(3));
  }

  @Test
  public void anExtentsShapeThatDoesNotMatchTheCallablesImplIsRefused() {
    SerializedLambda unsupported = lambda("([IJJ)V", new Object[] {new int[6], 3L});
    try {
      new SugarCallable(unsupported, 2, new Object[] {3L, 2L});
      fail("an unsupported shape must be refused");
    } catch (IllegalArgumentException expected) {
      assertEquals(
          "unsupported sugar callable shape: ([IJJ)V with rank 2 and 2 captures",
          expected.getMessage());
    }
  }
}
