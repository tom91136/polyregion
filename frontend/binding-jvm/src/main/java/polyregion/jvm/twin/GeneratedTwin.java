package polyregion.jvm.twin;

import java.nio.ByteBuffer;

/**
 * The base of a twin a frontend generates for one class. The subclass reads and writes primitive
 * fields directly and calls the helpers here for arrays, nested objects and the dispatch tag, so
 * the checks match {@link ReflectiveTwin}. Offsets and sizes come from the layout, never from the
 * generated code.
 */
public abstract class GeneratedTwin extends Twin {

  private final Layout layout;
  private final Twins twins;

  protected final int[] offsets;

  protected GeneratedTwin(Layout layout, Twins twins) {
    this.layout = layout;
    this.twins = twins;
    this.offsets = new int[layout.fields.length];
    for (int i = 0; i < offsets.length; i++) offsets[i] = layout.fields[i].offset;
  }

  @Override
  public final Layout layout() {
    return layout;
  }

  protected final void putArray(Marshal m, Object value, ByteBuffer dst, int at, int field) {
    Layout.Field f = layout.fields[field];
    dst.putLong(
        at + f.offset, value == null ? 0L : m.array(value, layout.name + "." + f.name, f.element));
  }

  protected final void putStruct(
      Marshal m, Object root, Object value, ByteBuffer dst, int at, int field) {
    Layout.Field f = layout.fields[field];
    String label = layout.name + "." + f.name;
    if (value == null)
      throw new TwinException(
          "Field `"
              + f.name
              + "` of "
              + layout.name
              + " is null; a nested object travels by value and has no null to stand for");
    Twin nested = twins.of(value.getClass());
    if (!f.element.isEmpty() && !nested.layout().name.equals(f.element))
      throw new TwinException(
          "Field `"
              + f.name
              + "` of "
              + layout.name
              + " declares "
              + f.element
              + " but holds a "
              + value.getClass().getName()
              + "; a nested object travels by value at the declared class's layout");
    m.child(root, f.name, value);
    m.inline(value, nested, dst, at + f.offset, label);
  }

  protected final void readStruct(Marshal m, Object root, ByteBuffer src, int at, int field) {
    Layout.Field f = layout.fields[field];
    Object child = m.child(root, f.name);
    twins.of(child.getClass()).read(m, child, src, at + f.offset);
  }

  protected final void putTag(ByteBuffer dst, int at) {
    if (layout.tagOffset >= 0) dst.putInt(at + layout.tagOffset, layout.tag);
  }

  protected static boolean getBool(ByteBuffer src, int at) {
    return src.get(at) != 0;
  }
}
