package polyregion;

import static polyregion.javalang.Runtime.offload;

public class AnotherClass {

  static {
    offload(() -> "1");
  }

  public static final void a() {
    offload(() -> System.out.println("A"));
  }
}
