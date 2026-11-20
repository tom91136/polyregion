package polyregion.jvm.twin;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/** The twins of one launch, keyed by the binary name of the host class. */
public final class Twins {

  private final Map<String, Layout> layouts;
  private final Map<String, Twin> twins = new ConcurrentHashMap<>();

  public Twins(Map<String, Layout> layouts, Twin... generated) {
    this.layouts = Collections.unmodifiableMap(new HashMap<>(layouts));
    for (Twin t : generated) twins.put(t.layout().className, t);
  }

  public static Twins decode(String layouts, Twin... generated) {
    return new Twins(Layout.decodeAll(layouts), generated);
  }

  public Layout layout(String className) {
    Layout l = layouts.get(className);
    if (l == null)
      throw new TwinException(
          "No device layout for " + className + "; known: " + layouts.keySet());
    return l;
  }

  public Twin of(Class<?> type) {
    Twin t = twins.get(type.getName());
    if (t != null) return t;
    Twin created = new ReflectiveTwin(layout(type.getName()), type, this);
    Twin raced = twins.putIfAbsent(type.getName(), created);
    return raced != null ? raced : created;
  }
}
