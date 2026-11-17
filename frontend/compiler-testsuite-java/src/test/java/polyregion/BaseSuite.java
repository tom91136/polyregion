package polyregion;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;
import static polyregion.javalang.Runtime.offload;

import polyregion.javalang.OffloadFunction;

public abstract class BaseSuite {

  protected static <R> void assertMatchesJdk(OffloadFunction<R> f) {
    assertEquals(f.run(), offload(f));
  }

  protected static <R> void assertMatchesJdk(String what, OffloadFunction<R> f) {
    assertEquals(what, f.run(), offload(f));
  }

  protected static <R> void assertRejected(String construct, OffloadFunction<R> f) {
    try {
      offload(f);
    } catch (UnsupportedOperationException e) {
      String message = String.valueOf(e.getMessage());
      assertTrue(
          "Rejected, but the diagnostic does not name `" + construct + "`: " + message,
          message.contains(construct));
      return;
    }
    fail("Expected offloading to be refused because of " + construct);
  }

  protected static <R> void assertTraps(String message, OffloadFunction<R> f) {
    String seen = null;
    try {
      offload(f);
    } catch (AssertionError e) {
      seen = String.valueOf(e.getMessage());
    }
    assertEquals("the device should have trapped", message, seen);
  }
}
