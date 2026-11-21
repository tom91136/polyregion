package polyregion.javalang;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.List;

public abstract class LaunchSite {

  public static final String PACKAGE = "polyregion.$gen$";

  // `<entry>|<result>|<arguments>`, see LaunchSites.descriptorOf
  public abstract String descriptor();

  // the compiled groups, see LaunchSites.encodeGroups
  public abstract byte[] binary();

  public static String nameOf(String identity, List<String> parts) {
    String raw = identity + "|" + String.join(",", parts);
    StringBuilder name = new StringBuilder("Launch$");
    for (char c : identity.toCharArray()) name.append(Character.isLetterOrDigit(c) ? c : '_');
    byte[] digest;
    try {
      digest = MessageDigest.getInstance("SHA-1").digest(raw.getBytes(StandardCharsets.UTF_8));
    } catch (NoSuchAlgorithmException e) {
      throw new AssertionError(e);
    }
    name.append('$');
    for (int i = 0; i < 8; i++) name.append(String.format("%02x", digest[i]));
    return PACKAGE + "." + name;
  }

  public static String codeOf(Class<?> cls) {
    if (cls == Integer.class || cls == int.class) return "I";
    if (cls == Long.class || cls == long.class) return "J";
    if (cls == Float.class || cls == float.class) return "F";
    if (cls == Double.class || cls == double.class) return "D";
    if (cls == Short.class || cls == short.class) return "S";
    if (cls == Byte.class || cls == byte.class) return "B";
    if (cls == Character.class || cls == char.class) return "C";
    if (cls == Boolean.class || cls == boolean.class) return "Z";
    if (cls.isArray() && cls.getComponentType().isPrimitive())
      return "[" + codeOf(cls.getComponentType());
    return "";
  }
}
