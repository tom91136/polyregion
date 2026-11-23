package polyregion;

import org.junit.Test;

public class DispatchConcreteTest extends BaseSuite {

  static class Parent {
    int m() {
      return 1;
    }
  }

  static final class Child extends Parent {
    @Override
    int m() {
      return 2;
    }
  }

  static class Plain extends Parent {}

  static final class Sup extends Parent {
    @Override
    int m() {
      return super.m() + 20;
    }
  }

  static class Base {
    int m() {
      return 0;
    }
  }

  static final class Trio extends Base {
    final int a;
    final int b;
    final int c;

    Trio(int a, int b, int c) {
      this.a = a;
      this.b = b;
      this.c = c;
    }

    @Override
    int m() {
      return a * 100 + b * 10 + c;
    }
  }

  static int viaParent(Parent p) {
    return p.m();
  }

  @Test
  public void concreteOwnerNotAllocatedInRegion() {
    assertMatchesJdk("base-typed source subclass", () -> viaParent(new Child()));
  }

  @Test
  public void concreteOwnerAllocatedInSource() {
    assertMatchesJdk("both allocated", () -> viaParent(new Parent()) * 10 + viaParent(new Child()));
  }

  @Test
  public void aSubclassInheritingTheImplementation() {
    assertMatchesJdk("inheriting", () -> viaParent(new Plain()) * 100 + viaParent(new Parent()) * 10 + viaParent(new Child()));
  }

  @Test
  public void aSuperCallStaysBoundToTheParent() {
    assertMatchesJdk("super", () -> viaParent(new Sup()) * 10 + viaParent(new Parent()));
  }

  @Test
  public void aTwelveByteCandidateCapturedAtItsOwner() {
    Base held = new Trio(1, 2, 3);
    assertMatchesJdk("captured", () -> held.m());
  }

  @Test
  public void aTwelveByteCandidateBuiltBesideItsOwner() {
    assertMatchesJdk("built", () -> new Base().m() + new Trio(4, 5, 6).m());
  }
}
