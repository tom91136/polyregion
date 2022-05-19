package polyregion;

import java.time.Duration;
import java.util.Arrays;
import java.util.stream.Collectors;

public final class Event {
  public long epochMillis, elapsedNanos;
  public String name, data;

  @Override
  public String toString() {
    return "[@"
        + epochMillis
        + ",+"
        + String.format("%.3f", ((double) elapsedNanos) / 1.0e6)
        + "ms] "
        + name
        + (data == null || data.isEmpty()
            ? ""
            : ":\n"
                + Arrays.stream(data.split("\\n"))
                    .map(l -> " │" + l)
                    .collect(Collectors.joining("\n"))
                + "\n ╰───");
  }
}
