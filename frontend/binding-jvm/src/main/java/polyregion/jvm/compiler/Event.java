package polyregion.jvm.compiler;

import java.util.Arrays;
import java.util.Objects;
import java.util.stream.Collectors;

@SuppressWarnings("unused")
public final class Event {

  public final long epochMillis, elapsedNanos;
  public final String name, data;
  public final Event[] items;

  Event(long epochMillis, long elapsedNanos, String name, String data, Event[] items) {
    this.epochMillis = epochMillis;
    this.elapsedNanos = elapsedNanos;
    this.name = Objects.requireNonNull(name);
    this.data = Objects.requireNonNull(data);
    this.items = Objects.requireNonNull(items);
  }

  @Override
  public String toString() {
    String self = "[@"
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
    if (items.length == 0) return self;
    return self
        + "\n"
        + Arrays.stream(items)
            .map(Event::toString)
            .map(s -> Arrays.stream(s.split("\\n")).map(l -> "  " + l).collect(Collectors.joining("\n")))
            .collect(Collectors.joining("\n"));
  }
}
