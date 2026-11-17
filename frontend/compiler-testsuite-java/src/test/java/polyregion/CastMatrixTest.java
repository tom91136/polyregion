package polyregion;

import org.junit.Test;

public class CastMatrixTest extends BaseSuite {

  @Test
  public void fromByte() {
    byte v = -128;
    assertMatchesJdk("byte->short", () -> (short) v);
    assertMatchesJdk("byte->char", () -> (char) v);
    assertMatchesJdk("byte->int", () -> (int) v);
    assertMatchesJdk("byte->long", () -> (long) v);
    assertMatchesJdk("byte->float", () -> (float) v);
    assertMatchesJdk("byte->double", () -> (double) v);
  }

  @Test
  public void fromShort() {
    short v = -32768;
    assertMatchesJdk("short->byte", () -> (byte) v);
    assertMatchesJdk("short->char", () -> (char) v);
    assertMatchesJdk("short->int", () -> (int) v);
    assertMatchesJdk("short->long", () -> (long) v);
    assertMatchesJdk("short->float", () -> (float) v);
    assertMatchesJdk("short->double", () -> (double) v);
  }

  @Test
  public void fromChar() {
    char v = '\uffff';
    assertMatchesJdk("char->byte", () -> (byte) v);
    assertMatchesJdk("char->short", () -> (short) v);
    assertMatchesJdk("char->int", () -> (int) v);
    assertMatchesJdk("char->long", () -> (long) v);
    assertMatchesJdk("char->float", () -> (float) v);
    assertMatchesJdk("char->double", () -> (double) v);
  }

  @Test
  public void fromInt() {
    int v = Integer.MIN_VALUE;
    assertMatchesJdk("int->byte", () -> (byte) v);
    assertMatchesJdk("int->short", () -> (short) v);
    assertMatchesJdk("int->char", () -> (char) v);
    assertMatchesJdk("int->long", () -> (long) v);
    assertMatchesJdk("int->float", () -> (float) v);
    assertMatchesJdk("int->double", () -> (double) v);
  }

  @Test
  public void fromLong() {
    long v = Long.MIN_VALUE;
    assertMatchesJdk("long->byte", () -> (byte) v);
    assertMatchesJdk("long->short", () -> (short) v);
    assertMatchesJdk("long->char", () -> (char) v);
    assertMatchesJdk("long->int", () -> (int) v);
    assertMatchesJdk("long->float", () -> (float) v);
    assertMatchesJdk("long->double", () -> (double) v);
  }

  @Test
  public void fromFloat() {
    float v = 3.4028235e+38f;
    assertMatchesJdk("float->byte", () -> (byte) v);
    assertMatchesJdk("float->short", () -> (short) v);
    assertMatchesJdk("float->char", () -> (char) v);
    assertMatchesJdk("float->int", () -> (int) v);
    assertMatchesJdk("float->long", () -> (long) v);
    assertMatchesJdk("float->double", () -> (double) v);
  }

  @Test
  public void fromDouble() {
    double v = 1.7976931348623157e+308d;
    assertMatchesJdk("double->byte", () -> (byte) v);
    assertMatchesJdk("double->short", () -> (short) v);
    assertMatchesJdk("double->char", () -> (char) v);
    assertMatchesJdk("double->int", () -> (int) v);
    assertMatchesJdk("double->long", () -> (long) v);
    assertMatchesJdk("double->float", () -> (float) v);
  }

  @Test
  public void fromNegativeDouble() {
    double v = -2.5;
    assertMatchesJdk("negdouble->int", () -> (int) v);
    assertMatchesJdk("negdouble->long", () -> (long) v);
    assertMatchesJdk("negdouble->byte", () -> (byte) v);
    assertMatchesJdk("negdouble->float", () -> (float) v);
  }
}
