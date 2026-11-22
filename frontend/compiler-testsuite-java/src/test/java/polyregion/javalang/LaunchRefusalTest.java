package polyregion.javalang;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.util.Arrays;
import java.util.Collections;
import org.junit.Test;
import polyregion.jvm.runtime.CapabilityGapException;
import polyregion.jvm.runtime.LoadOutcome;

public class LaunchRefusalTest {

  @Test
  public void lacksOnlyIsACapabilityGap() {
    assertTrue(
        Launch.refusal("refused", Collections.singletonList(LoadOutcome.LACKS)) instanceof CapabilityGapException);
  }

  @Test
  public void formatAndLacksIsACapabilityGap() {
    assertTrue(
        Launch.refusal("refused", Arrays.asList(LoadOutcome.FORMAT, LoadOutcome.LACKS))
            instanceof CapabilityGapException);
  }

  @Test
  public void formatAloneIsPlain() {
    assertFalse(
        Launch.refusal("refused", Collections.singletonList(LoadOutcome.FORMAT)) instanceof CapabilityGapException);
  }

  @Test
  public void lacksAndUnpreparedIsPlain() {
    assertFalse(
        Launch.refusal("refused", Arrays.asList(LoadOutcome.LACKS, LoadOutcome.UNPREPARED))
            instanceof CapabilityGapException);
  }

  @Test
  public void noDeviceIsPlain() {
    assertFalse(
        Launch.refusal("refused", Collections.singletonList(LoadOutcome.NO_DEVICE)) instanceof CapabilityGapException);
  }

  @Test
  public void emptyRefusalsArePlain() {
    assertFalse(Launch.refusal("refused", Collections.<LoadOutcome>emptyList()) instanceof CapabilityGapException);
  }
}
