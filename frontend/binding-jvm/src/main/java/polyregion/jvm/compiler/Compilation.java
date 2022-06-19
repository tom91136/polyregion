package polyregion.jvm.compiler;

import java.util.Arrays;
import java.util.Objects;

@SuppressWarnings("unused")
public final class Compilation {

  public final byte[] program;
  public final String[] features;
  public final Event[] events;
  public final Layout[] layouts;
  public final String messages;

  Compilation(
      byte[] program, String[] features, Event[] events, Layout[] layouts, String messages) {
    this.program = Objects.requireNonNull(program);
    this.features = Objects.requireNonNull(features);
    this.events = Objects.requireNonNull(events);
    this.layouts = Objects.requireNonNull(layouts);
    this.messages = Objects.requireNonNull(messages);
  }

  @Override
  public String toString() {
    return "Compilation{"
        + "program="
        + (program.length / 1000)
        + "KB"
        + ", features="
        + Arrays.toString(features)
        + ", events="
        + Arrays.toString(events)
        + ", layouts="
        + Arrays.toString(layouts)
        + ", messages='"
        + messages
        + "'"
        + '}';
  }
}
