package polyregion;

import java.util.function.Predicate;
import org.junit.AssumptionViolatedException;
import org.junit.rules.TestRule;
import org.junit.runner.Description;
import org.junit.runners.model.Statement;
import polyregion.jvm.runtime.CapabilityGapException;
import polyregion.jvm.runtime.Context;

public final class LaunchRule {

  private LaunchRule() {
    throw new AssertionError();
  }

  public static TestRule skipOnGap() {
    return (base, description) ->
        new Statement() {
          @Override
          public void evaluate() throws Throwable {
            try {
              base.evaluate();
            } catch (Throwable failure) {
              Throwable cause = failure instanceof ExceptionInInitializerError ? failure.getCause() : failure;
              if (cause instanceof CapabilityGapException)
                throw new AssumptionViolatedException(cause.getMessage(), cause);
              throw failure;
            }
          }
        };
  }

  public static TestRule requireLaunch(Predicate<String> exempt) {
    return (base, description) ->
        skipOnGap()
            .apply(
                new Statement() {
                  @Override
                  public void evaluate() throws Throwable {
                    long before = Context.current().stats().launches;
                    base.evaluate();
                    if (!exempt.test(description.getMethodName())
                        && Context.current().stats().launches <= before)
                      throw new AssertionError(description.getMethodName() + " ran no kernel");
                  }
                },
                description);
  }
}
