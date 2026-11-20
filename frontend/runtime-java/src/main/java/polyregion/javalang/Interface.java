package polyregion.javalang;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * Marks a static stub as the Java projection of a PolyRegion interface declaration.
 *
 * <p>The value is the interface identity, {@code <interface>:<declaration>}, for example {@code
 * spectra:spectra.transform}. It matches the identity carried by the C++ {@code
 * [[clang::annotate("polyregion_interface:...")]]} and the Scala {@code @compileTimeOnly} stubs.
 * The annotation is retained in class files so that the compiler can recognise calls to stubs in
 * dependencies.
 */
@Retention(RetentionPolicy.CLASS)
@Target(ElementType.METHOD)
public @interface Interface {
  String value();
}
