package polyregion.jvm.runtime;

import java.util.Objects;
import polyregion.jvm.compiler.Target;

@SuppressWarnings("unused")
public final class ModuleImage {
  public final String name;
  public final Target target;
  public final String[] features;
  public final byte[] image;

  public ModuleImage(String name, Target target, String[] features, byte[] image) {
    this.name = Objects.requireNonNull(name);
    this.target = Objects.requireNonNull(target);
    for (String feature : features) Objects.requireNonNull(feature);
    this.features = Objects.requireNonNull(features);
    this.image = Objects.requireNonNull(image);
  }

  static String[] names(ModuleImage[] images) {
    String[] xs = new String[images.length];
    for (int i = 0; i < xs.length; ++i) xs[i] = images[i].name;
    return xs;
  }

  static byte[] targets(ModuleImage[] images) {
    byte[] xs = new byte[images.length];
    for (int i = 0; i < xs.length; ++i) xs[i] = images[i].target.value();
    return xs;
  }

  static String[][] features(ModuleImage[] images) {
    String[][] xs = new String[images.length][];
    for (int i = 0; i < xs.length; ++i) xs[i] = images[i].features;
    return xs;
  }

  static byte[][] bytes(ModuleImage[] images) {
    byte[][] xs = new byte[images.length][];
    for (int i = 0; i < xs.length; ++i) xs[i] = images[i].image;
    return xs;
  }
}
