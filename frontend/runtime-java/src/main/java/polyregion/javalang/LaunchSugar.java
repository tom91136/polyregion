package polyregion.javalang;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * Marks a generated region API sugar method. The processor linearises calls to it onto the
 * one-dimensional interface named by {@code value}. {@code rank} extents lead the arguments;
 * {@code map} is the position of the callable to wrap among the remaining arguments ({@code 0} for a
 * task, whose rank is zero).
 */
@Retention(RetentionPolicy.CLASS)
@Target(ElementType.METHOD)
public @interface LaunchSugar {
  String value();

  int rank();

  int map();
}
