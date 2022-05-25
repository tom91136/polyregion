package polyregion.jvm.compiler;

import java.time.Duration;
import java.util.Arrays;
import java.util.Objects;
import java.util.stream.Collectors;

public final class Event {

  public final long epochMillis, elapsedNanos;
  public final String name, data;

  Event(long epochMillis, long elapsedNanos, String name, String data) {
    this.epochMillis = epochMillis;
    this.elapsedNanos = elapsedNanos;
    this.name = Objects.requireNonNull(name);
    this.data = Objects.requireNonNull(data);
  }

  @Override
  public String toString() {
    return "[@"
        + epochMillis
        + ",+"
        + String.format("%.3f", (double) elapsedNanos / 1.0e6)
        + "ms] "
        + name
        + (data.isEmpty()
            ? ""
            : ":\n"
                + Arrays.stream(data.split("\\n"))
                    .map(l -> " │" + l)
                    .collect(Collectors.joining("\n"))
                + "\n ╰───");
  }
}
