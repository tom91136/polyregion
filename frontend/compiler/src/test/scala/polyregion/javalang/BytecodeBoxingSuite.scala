package polyregion.javalang

import polyregion.ast.PolyAST as p

class BytecodeBoxingSuite extends Differential {

  private val Source =
    """
      |public final class Boxing {
      |  public static Integer boxedInt(int a) { return a * 2 + 1; }
      |  public static Long boxedLong(long a) { return a - 3L; }
      |  public static Double boxedDouble(double a) { return a / 2; }
      |  public static Boolean boxedBoolean(int a) { return a > 0; }
      |  public static Character boxedChar(int a) { return (char) (a + 'a'); }
      |  public static Integer fromArray(int[] xs, int n) {
      |    int s = 0;
      |    for (int i = 0; i < n; i++) s += xs[i];
      |    return s;
      |  }
      |}
      |""".stripMargin

  private lazy val boxing             = JavaInMemory.compile("Boxing", Source)
  override protected lazy val subject = Differential.staticsOf(boxing, "Boxing")

  test("a boxed int result is the int")(check("boxedInt", List(0, 1, -5, 1000).map(List(_))))
  test("a boxed long result is the long")(check("boxedLong", List(0L, 9L, -4L).map(List(_))))
  test("a boxed double result is the double")(check("boxedDouble", List(0d, 3.5, -8d).map(List(_))))
  test("a boxed boolean result is the boolean")(check("boxedBoolean", List(-1, 0, 4).map(List(_))))
  test("a boxed char result is the char")(check("boxedChar", List(0, 1, 5).map(List(_))))
  test("a boxed result of a loop")(check("fromArray", List(List(Array(1, 2, 3, 4), 4), List(Array(5), 0))))

  test("a boxed result lowers to its primitive") {
    assertEquals(lower("boxedInt").rtn, p.Type.IntS32)
    assertEquals(lower("boxedDouble").rtn, p.Type.Float64)
  }
}
