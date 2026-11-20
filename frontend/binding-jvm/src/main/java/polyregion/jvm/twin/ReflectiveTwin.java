package polyregion.jvm.twin;

import java.nio.ByteBuffer;

/**
 * A twin that walks the layout and reaches fields through {@link FieldAccess}. The field names and
 * types are checked against the class when the twin is created, so a class that has drifted from
 * the layout fails there rather than corrupting data.
 */
public final class ReflectiveTwin extends Twin {

  private final Layout layout;
  private final Twins twins;
  private final FieldAccess[] access;
  private final Prim[] prims;

  public ReflectiveTwin(Layout layout, Class<?> type, Twins twins) {
    this.layout = layout;
    this.twins = twins;
    Layout.Field[] fields = layout.fields;
    access = new FieldAccess[fields.length];
    prims = new Prim[fields.length];
    for (int i = 0; i < fields.length; i++) {
      Layout.Field f = fields[i];
      access[i] = FieldAccess.of(type, f.name, f.descriptor);
      Class<?> t = access[i].field.getType();
      switch (f.kind) {
        case PRIM:
          prims[i] = Prim.of(t);
          if (prims[i] == null || prims[i].size != f.size)
            throw new TwinException(
                "Field "
                    + layout.name
                    + "."
                    + f.name
                    + " is "
                    + f.size
                    + " bytes in the layout but "
                    + t.getName()
                    + " on the host");
          break;
        case ARRAY:
          if (!t.isArray())
            throw new TwinException(
                "Field " + layout.name + "." + f.name + " is an array in the layout but " + t.getName() + " on the host");
          break;
        case STRUCT:
          if (t.isPrimitive() || t.isArray())
            throw new TwinException(
                "Field " + layout.name + "." + f.name + " is an object in the layout but " + t.getName() + " on the host");
          break;
      }
    }
  }

  @Override
  public Layout layout() {
    return layout;
  }

  @Override
  public void write(Marshal m, Object root, ByteBuffer dst, int at) {
    Layout.Field[] fields = layout.fields;
    for (int i = 0; i < fields.length; i++) {
      Layout.Field f = fields[i];
      Object value = access[i].get(root);
      int where = at + f.offset;
      switch (f.kind) {
        case PRIM:
          prims[i].put(dst, where, value);
          break;
        case ARRAY:
          dst.putLong(where, value == null ? 0L : m.array(value));
          break;
        case STRUCT:
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
          nested.write(m, value, dst, where);
          break;
      }
    }
    if (layout.tagOffset >= 0) dst.putInt(at + layout.tagOffset, layout.tag);
  }

  @Override
  public void read(Marshal m, Object root, ByteBuffer src, int at) {
    Layout.Field[] fields = layout.fields;
    for (int i = 0; i < fields.length; i++) {
      Layout.Field f = fields[i];
      int where = at + f.offset;
      switch (f.kind) {
        case PRIM:
          if (access[i].writable()) access[i].set(root, prims[i].get(src, where));
          break;
        case STRUCT:
          Object value = access[i].get(root);
          twins.of(value.getClass()).read(m, value, src, where);
          break;
        case ARRAY:
          break;
      }
    }
  }
}
