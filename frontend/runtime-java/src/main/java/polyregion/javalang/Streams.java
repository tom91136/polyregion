package polyregion.javalang;

import java.lang.reflect.Array;
import java.util.Arrays;
import java.util.OptionalDouble;
import java.util.OptionalInt;
import java.util.OptionalLong;
import polyregion.jvm.runtime.Context;

/**
 * Run-time half of the stream lowering. The compiler replaces a recognised chain with a call to one
 * of the typed entries below; the chain's launches are compiled ahead of time into generated
 * launch-site classes, and this class orders them.
 *
 * <p>{@code values} holds the context (or {@code null}), the source values, the terminal's extra
 * values, and finally the captured values of every lambda in the chain, in chain order.
 */
public final class Streams {

  private Streams() {
    throw new AssertionError();
  }

  public enum Op {
    SUM(0),
    COUNT(0),
    MIN(0),
    MAX(0),
    AVERAGE(0),
    ANY(0),
    ALL(0),
    NONE(0),
    REDUCE_ID(1),
    REDUCE(0),
    FOR_EACH(0),
    TO_ARRAY(0),
    SORTED(0),
    SORT(0),
    SET_ALL(0),
    PREFIX(0);

    final int extras;

    Op(int extras) {
      this.extras = extras;
    }
  }

  public static final String MAIN = "main";
  public static final String COUNT = "count";
  public static final String VALIDITY = "valid";
  public static final String VALUES = "values";
  public static final String FILL = "fill";
  public static final String INDICES = "indices";
  public static final String SORT = "sort";
  public static final String SCAN = "scan";
  public static final String TAIL = "tail";
  public static final String EACH = "each";

  private static final char ARRAY = 'A';
  private static final char RANGE = 'R';
  private static final char CLOSED = 'C';

  public static String shape(Op op, char source, char in, char out, boolean filtered, boolean mapped) {
    return op.name() + ":" + source + ":" + in + ":" + out + ":" + (filtered ? 1 : 0) + ":" + (mapped ? 1 : 0);
  }

  public static String siteName(String key, String role) {
    return LaunchSite.PACKAGE + ".Stream$" + key + "$" + role;
  }

  public static int i(Class<?> home, String key, String shape, Object[] values) {
    return (Integer) run(home, key, shape, values);
  }

  public static long j(Class<?> home, String key, String shape, Object[] values) {
    return (Long) run(home, key, shape, values);
  }

  public static double d(Class<?> home, String key, String shape, Object[] values) {
    return (Double) run(home, key, shape, values);
  }

  public static boolean z(Class<?> home, String key, String shape, Object[] values) {
    return (Boolean) run(home, key, shape, values);
  }

  public static void v(Class<?> home, String key, String shape, Object[] values) {
    run(home, key, shape, values);
  }

  public static int[] ia(Class<?> home, String key, String shape, Object[] values) {
    return (int[]) run(home, key, shape, values);
  }

  public static long[] ja(Class<?> home, String key, String shape, Object[] values) {
    return (long[]) run(home, key, shape, values);
  }

  public static double[] da(Class<?> home, String key, String shape, Object[] values) {
    return (double[]) run(home, key, shape, values);
  }

  public static OptionalInt oi(Class<?> home, String key, String shape, Object[] values) {
    return (OptionalInt) run(home, key, shape, values);
  }

  public static OptionalLong oj(Class<?> home, String key, String shape, Object[] values) {
    return (OptionalLong) run(home, key, shape, values);
  }

  public static OptionalDouble od(Class<?> home, String key, String shape, Object[] values) {
    return (OptionalDouble) run(home, key, shape, values);
  }

  private static Object run(Class<?> home, String key, String shape, Object[] values) {
    return new Chain(home, key, shape, values).run();
  }

  private static final class Chain {
    final Class<?> home;
    final String key;
    final Op op;
    final char source;
    final char in;
    final char out;
    final boolean filtered;
    final boolean mapped;
    final Context context;
    final Object origin;
    final Object extra;
    final Object[] captured;
    final long n;

    Chain(Class<?> home, String key, String shape, Object[] values) {
      String[] parts = shape.split(":");
      this.home = home;
      this.key = key;
      this.op = Op.valueOf(parts[0]);
      this.source = parts[1].charAt(0);
      this.in = parts[2].charAt(0);
      this.out = parts[3].charAt(0);
      this.filtered = parts[4].equals("1");
      this.mapped = parts[5].equals("1");
      this.context = (Context) values[0];
      int at = 1;
      if (source == ARRAY) {
        origin = values[at++];
        if (origin == null) throw new NullPointerException();
        n = Array.getLength(origin);
      } else {
        long lo = ((Number) values[at++]).longValue();
        long hi = ((Number) values[at++]).longValue();
        origin = coerce(lo, in);
        n = hi < lo ? 0 : hi - lo + (source == CLOSED ? 1 : 0);
      }
      extra = op.extras > 0 ? coerce(values[at], out) : null;
      at += op.extras;
      captured = Arrays.copyOfRange(values, at, values.length);
    }

    static Object coerce(Object value, char type) {
      Number number = (Number) value;
      switch (type) {
        case 'I':
          return number.intValue();
        case 'J':
          return number.longValue();
        default:
          return number.doubleValue();
      }
    }

    Object run() {
      switch (op) {
        case SUM:
        case REDUCE_ID:
        case MIN:
        case MAX:
        case COUNT:
        case AVERAGE:
        case ANY:
        case ALL:
        case NONE:
          return reduction();
        case TO_ARRAY:
          return toArray();
        case SORTED:
          {
            Object sorted = toArray();
            sort(sorted, Array.getLength(sorted));
            return sorted;
          }
        case REDUCE:
          return tailReduction();
        case FOR_EACH:
          if (n > 0) launch(EACH, null, n, null);
          return null;
        case SET_ALL:
          if (n > 0) launch(FILL, null, origin, (int) n, null);
          return null;
        case PREFIX:
          if (n > 1) {
            Object scanned = newArray(out, (int) n);
            launch(SCAN, null, origin, scanned, (int) n, null);
            System.arraycopy(scanned, 0, origin, 0, (int) n);
          }
          return null;
        case SORT:
          sort(origin, (int) n);
          return null;
        default:
          throw new AssertionError(op);
      }
    }

    static Object newArray(char type, int length) {
      switch (type) {
        case 'I':
          return new int[length];
        case 'J':
          return new long[length];
        default:
          return new double[length];
      }
    }

    static Object copyOf(Object array, char type, int from, int to) {
      switch (type) {
        case 'I':
          return Arrays.copyOfRange((int[]) array, from, to);
        case 'J':
          return Arrays.copyOfRange((long[]) array, from, to);
        default:
          return Arrays.copyOfRange((double[]) array, from, to);
      }
    }

    void sort(Object array, int length) {
      if (length > 1) launch(SORT, null, array, length, null);
    }

    Object filtered(Object source, char kind) {
      Object survivors = newArray(kind, (int) n);
      int kept = (Integer) launch(VALIDITY, null, source, (int) n, survivors, null);
      if (kept == 0) return newArray(out, 0);
      if (!mapped) return copyOf(survivors, kind, 0, kept);
      Object result = newArray(out, kept);
      launch(VALUES, null, survivors, result, kept, null);
      return result;
    }

    Object toArray() {
      if (n == 0) return newArray(out, 0);
      if (source == ARRAY) {
        if (filtered) return filtered(origin, in);
        if (!mapped) return copyOf(origin, in, 0, (int) n);
        Object result = newArray(out, (int) n);
        launch(VALUES, null, origin, result, (int) n, null);
        return result;
      }
      if (!filtered) {
        Object result = newArray(out, (int) n);
        launch(FILL, null, result, (int) n, null);
        return result;
      }
      Object indices = newArray(in, (int) n);
      launch(INDICES, null, indices, (int) n, null);
      return filtered(indices, in);
    }

    Object tailReduction() {
      Object all = toArray();
      int length = Array.getLength(all);
      if (length == 0) return empty(out);
      Object first = Array.get(all, 0);
      if (length == 1) return present(out, first);
      Object tail = copyOf(all, out, 1, length);
      return present(out, launch(TAIL, null, tail, length - 1, first, null));
    }

    String site(String role) {
      return siteName(key, role);
    }

    Object[] flat(Object neutral) {
      Object[] all = new Object[captured.length + 2];
      all[0] = origin;
      all[1] = neutral == null ? (Object) 0 : neutral;
      System.arraycopy(captured, 0, all, 2, captured.length);
      return all;
    }

    Object launch(String role, Object neutral, Object... args) {
      return Launch.step(home, site(role), context, args, flat(neutral));
    }

    static Object zero(char type) {
      switch (type) {
        case 'I':
          return 0;
        case 'J':
          return 0L;
        case 'D':
          return 0.0;
        default:
          throw new IllegalStateException(String.valueOf(type));
      }
    }

    Object extremum(char type, boolean min) {
      switch (type) {
        case 'I':
          return min ? Integer.MAX_VALUE : Integer.MIN_VALUE;
        case 'J':
          return min ? Long.MAX_VALUE : Long.MIN_VALUE;
        case 'D':
          return min ? Double.POSITIVE_INFINITY : Double.NEGATIVE_INFINITY;
        default:
          throw new IllegalStateException(String.valueOf(type));
      }
    }

    Object reduce(String role, char type, Object neutral) {
      return source == ARRAY
          ? launch(role, neutral, origin, (int) n, neutral, null, null)
          : launch(role, neutral, n, neutral, null, null);
    }

    Object reduction() {
      switch (op) {
        case COUNT:
          if (!filtered) return n;
          return n == 0 ? (Object) 0L : reduce(MAIN, 'J', 0L);
        case SUM:
          return n == 0 ? zero(out) : reduce(MAIN, out, zero(out));
        case REDUCE_ID:
          return n == 0 ? extra : reduce(MAIN, out, extra);
        case ANY:
          return n != 0 && (Integer) reduce(MAIN, 'I', 0) > 0;
        case ALL:
          return n == 0 || (Integer) reduce(MAIN, 'I', 0) == 0;
        case NONE:
          return n == 0 || (Integer) reduce(MAIN, 'I', 0) == 0;
        case AVERAGE:
          {
            if (n == 0) return OptionalDouble.empty();
            char wide = out == 'D' ? 'D' : 'J';
            Object sum = reduce(MAIN, wide, zero(wide));
            long count = filtered ? (Long) reduce(COUNT, 'J', 0L) : n;
            if (count == 0) return OptionalDouble.empty();
            return OptionalDouble.of(((Number) sum).doubleValue() / count);
          }
        default:
          {
            boolean min = op == Op.MIN;
            if (n == 0) return empty(out);
            Object neutral = extremum(out, min);
            Object best = reduce(MAIN, out, neutral);
            if (filtered && best.equals(neutral) && (Long) reduce(COUNT, 'J', 0L) == 0) return empty(out);
            return present(out, best);
          }
      }
    }

    static Object empty(char type) {
      switch (type) {
        case 'I':
          return OptionalInt.empty();
        case 'J':
          return OptionalLong.empty();
        default:
          return OptionalDouble.empty();
      }
    }

    static Object present(char type, Object value) {
      switch (type) {
        case 'I':
          return OptionalInt.of((Integer) value);
        case 'J':
          return OptionalLong.of((Long) value);
        default:
          return OptionalDouble.of((Double) value);
      }
    }
  }
}
