package polyregion.jvm.compiler;

import java.util.Arrays;
import java.util.Objects;

@SuppressWarnings("unused")
public final class Compilation {

  public final byte[] program;
  public final String[] features;
  public final String[] entryArgs;
  public final Event[] events;
  public final Layout[] layouts;
  public final String messages;

  Compilation(
      byte[] program,
      String[] features,
      String[] entryArgs,
      Event[] events,
      Layout[] layouts,
      String messages) {
    this.program = Objects.requireNonNull(program);
    this.features = Objects.requireNonNull(features);
    this.entryArgs = Objects.requireNonNull(entryArgs);
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
        + ", entryArgs="
        + Arrays.toString(entryArgs)
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
