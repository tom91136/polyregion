package polyregion.jvm.twin;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * The device layout of one host class: field offsets, sizes and the structure alignment, as the
 * native compiler computed them for a target.
 *
 * <p>A layout is a plain value. A frontend builds it from a struct definition and the compiler's
 * layout, {@link #encode} turns it into text it can embed as a string constant, and the runtime
 * reads it back with {@link #decode}.
 */
public final class Layout {

  public enum Kind {
    /** A primitive stored inline. */
    PRIM('p'),
    /** A nested object stored inline, by value, at the layout named by {@link Field#element}. */
    STRUCT('s'),
    /**
     * A device pointer to a separately staged array. The elements are primitives of the field's
     * array descriptor, or objects stored by value at {@link Field#element} when that is not empty.
     */
    ARRAY('a');

    final char code;

    Kind(char code) {
      this.code = code;
    }

    static Kind of(char code) {
      for (Kind k : values()) if (k.code == code) return k;
      throw new TwinException("Unknown layout member kind `" + code + "`");
    }
  }

  public static final class Field {
    public final String name;

    /** The JVM descriptor of the host field, e.g. {@code I}, {@code [D} or {@code Lfoo/Bar;}. */
    public final String descriptor;

    public final Kind kind;
    public final int offset;
    public final int size;

    /** The layout name of the nested object or array element, empty for primitives. */
    public final String element;

    /** Whether the kernel never changes the field, so a host {@code final} field is allowed. */
    public final boolean readOnly;

    public Field(
        String name, String descriptor, Kind kind, int offset, int size, String element) {
      this(name, descriptor, kind, offset, size, element, false);
    }

    public Field(
        String name,
        String descriptor,
        Kind kind,
        int offset,
        int size,
        String element,
        boolean readOnly) {
      this.name = Objects.requireNonNull(name);
      this.descriptor = Objects.requireNonNull(descriptor);
      this.kind = Objects.requireNonNull(kind);
      this.offset = offset;
      this.size = size;
      this.element = Objects.requireNonNull(element);
      this.readOnly = readOnly;
    }

    @Override
    public String toString() {
      return name + "@" + offset + ":" + size + " " + kind + " " + descriptor;
    }
  }

  /** The key twins refer to this layout by. */
  public final String name;

  /** The binary name of the host class, empty when the struct has no host class of its own. */
  public final String className;

  public final int size;
  public final int alignment;

  /** The offset of the dispatch tag, or -1 when the class takes no part in dispatch. */
  public final int tagOffset;

  /** The dispatch tag written at {@link #tagOffset}, meaningless when that is -1. */
  public final int tag;

  public final Field[] fields;

  public Layout(
      String name,
      String className,
      int size,
      int alignment,
      int tagOffset,
      int tag,
      Field[] fields) {
    this.name = Objects.requireNonNull(name);
    this.className = Objects.requireNonNull(className);
    this.size = size;
    this.alignment = alignment;
    this.tagOffset = tagOffset;
    this.tag = tag;
    this.fields = fields.clone();
  }

  public Field field(String name) {
    for (Field f : fields) if (f.name.equals(name)) return f;
    throw new TwinException("Layout " + this.name + " has no field `" + name + "`");
  }

  @Override
  public String toString() {
    return "Layout{" + name + ", size=" + size + ", fields=" + java.util.Arrays.toString(fields) + "}";
  }

  private static final String VERSION = "TL2";

  private static String escape(String s) {
    StringBuilder b = new StringBuilder(s.length());
    for (int i = 0; i < s.length(); i++) {
      char c = s.charAt(i);
      if (c == '%' || c == '~' || c == ':' || c == ',' || c == '@' || c == ';' || c == '|' || c < 33 || c > 126)
        b.append('%').append(String.format("%04x", (int) c));
      else b.append(c);
    }
    return b.toString();
  }

  private static String unescape(String s) {
    if (s.indexOf('%') < 0) return s;
    StringBuilder b = new StringBuilder(s.length());
    for (int i = 0; i < s.length(); i++) {
      char c = s.charAt(i);
      if (c == '%') {
        if (i + 5 > s.length()) throw new TwinException("Truncated escape in `" + s + "`");
        b.append((char) parseInt(s.substring(i + 1, i + 5), 16));
        i += 4;
      } else b.append(c);
    }
    return b.toString();
  }

  private static int parseInt(String text, int radix) {
    try {
      return Integer.parseInt(text, radix);
    } catch (NumberFormatException e) {
      throw new TwinException("Malformed number `" + text + "` in a layout", e);
    }
  }

  /** {@code name:className:size:alignment:tagOffset:tag:member,member,...} with {@code member = name@offset:size:kind:descriptor:element:flags}, flags being {@code r} for a read-only field. */
  public String encode() {
    StringBuilder b = new StringBuilder();
    b.append(escape(name)).append(':').append(escape(className)).append(':');
    b.append(size).append(':').append(alignment).append(':');
    b.append(tagOffset).append(':').append(tag).append(':');
    for (int i = 0; i < fields.length; i++) {
      Field f = fields[i];
      if (i > 0) b.append(',');
      b.append(escape(f.name)).append('@').append(f.offset).append(':').append(f.size).append(':');
      b.append(f.kind.code).append(':').append(escape(f.descriptor)).append(':').append(escape(f.element));
      b.append(':').append(f.readOnly ? "r" : "");
    }
    return b.toString();
  }

  /** {@code TL1~layout~layout...} */
  public static String encodeAll(Iterable<Layout> layouts) {
    StringBuilder b = new StringBuilder(VERSION);
    for (Layout l : layouts) b.append('~').append(l.encode());
    return b.toString();
  }

  public static Layout decode(String text) {
    String[] head = text.split(":", 7);
    if (head.length != 7) throw new TwinException("Malformed layout `" + text + "`");
    List<Field> fields = new ArrayList<>();
    if (!head[6].isEmpty())
      for (String member : head[6].split(",")) {
        int at = member.indexOf('@');
        if (at < 0) throw new TwinException("Malformed layout member `" + member + "`");
        String[] rest = member.substring(at + 1).split(":", 6);
        if (rest.length != 6 || rest[2].length() != 1)
          throw new TwinException("Malformed layout member `" + member + "`");
        fields.add(
            new Field(
                unescape(member.substring(0, at)),
                unescape(rest[3]),
                Kind.of(rest[2].charAt(0)),
                parseInt(rest[0], 10),
                parseInt(rest[1], 10),
                unescape(rest[4]),
                rest[5].equals("r")));
      }
    return new Layout(
        unescape(head[0]),
        unescape(head[1]),
        parseInt(head[2], 10),
        parseInt(head[3], 10),
        parseInt(head[4], 10),
        parseInt(head[5], 10),
        fields.toArray(new Field[0]));
  }

  public static Map<String, Layout> decodeAll(String text) {
    String[] parts = text.split("~", -1);
    if (!parts[0].equals(VERSION))
      throw new TwinException("Unsupported layout encoding `" + parts[0] + "`, expected " + VERSION);
    Map<String, Layout> out = new LinkedHashMap<>();
    for (int i = 1; i < parts.length; i++) {
      if (parts[i].isEmpty()) continue;
      Layout l = decode(parts[i]);
      if (out.put(l.name, l) != null)
        throw new TwinException("Duplicate layout name `" + l.name + "`");
    }
    return out;
  }
}
