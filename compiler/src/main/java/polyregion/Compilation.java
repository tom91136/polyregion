package polyregion;

import java.util.Arrays;

public final class Compilation {
  public byte[] program;
  public Event[] events;
  public Layout[] layouts;
  public String messages;
  public String disassembly;


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
        + '\''
        + ", disassembly='"
        + disassembly
        + '\''
        + '}';
  }
}
