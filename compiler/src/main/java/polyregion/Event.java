package polyregion;

import java.time.Duration;

public final class Event {
  public long epochMillis;
  public String name;
  public long elapsedNanos;

  @Override
  public String toString() {
    return "Event{"
        + "@"
        + epochMillis
        + " "
        + name
        + " : "
        + String.format("%.3f", ((double) elapsedNanos) / 1.0e6)
        + "ms}";
  }
}
