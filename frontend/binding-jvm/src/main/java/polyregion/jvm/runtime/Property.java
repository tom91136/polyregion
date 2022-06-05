package polyregion.jvm.runtime;

import java.util.Objects;

@SuppressWarnings("unused")
public final class Property {
  public final String key;
  public final String value;

  public Property(String key, String value) {
    this.key = Objects.requireNonNull(key);
    this.value = Objects.requireNonNull(value);
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;
    Property property = (Property) o;
    return key.equals(property.key) && value.equals(property.value);
  }

  @Override
  public int hashCode() {
    return Objects.hash(key, value);
  }

  @Override
  public String toString() {
    return "Property{" + "key='" + key + '\'' + ", value='" + value + '\'' + '}';
  }
}
