package polyregion.javalang;

import java.lang.reflect.Array;
import java.lang.reflect.Field;
import java.lang.reflect.Modifier;

final class StaticBinding {

  final Field field;
  final Object box;

  private StaticBinding(Field field, Object box) {
    this.field = field;
    this.box = box;
  }

  static StaticBinding of(String spec) {
    int hash = spec.indexOf('#');
    String owner = spec.substring(0, hash);
    String name = spec.substring(hash + 1);
    Field f;
    try {
      f = Class.forName(owner).getDeclaredField(name);
      f.setAccessible(true);
    } catch (ClassNotFoundException | NoSuchFieldException | SecurityException e) {
      throw new IllegalStateException("Cannot resolve static field " + owner + "." + name, e);
    }
    Class<?> t = f.getType();
    if (!t.isPrimitive()) throw new IllegalStateException("Cannot offload static field of type " + t);
    try {
      Object box = Array.newInstance(t, 1);
      Array.set(box, 0, f.get(null));
      return new StaticBinding(f, box);
    } catch (IllegalAccessException e) {
      throw new IllegalStateException("Cannot read static field " + owner + "." + name, e);
    }
  }

  void writeBack() {
    if (Modifier.isFinal(field.getModifiers())) return;
    try {
      field.set(null, Array.get(box, 0));
    } catch (IllegalAccessException e) {
      throw new IllegalStateException("Cannot write back static field " + field, e);
    }
  }
}
