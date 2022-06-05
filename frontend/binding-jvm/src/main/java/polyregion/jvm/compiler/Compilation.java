package polyregion.jvm.compiler;

import java.util.Arrays;
import java.util.Objects;

@SuppressWarnings("unused")
public final class Compilation {

  public final byte[] program;
  public final Event[] events;
  public final Layout[] layouts;
  public final String messages;

  Compilation(byte[] program, Event[] events, Layout[] layouts, String messages) {
    this.program = Objects.requireNonNull(program);
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
