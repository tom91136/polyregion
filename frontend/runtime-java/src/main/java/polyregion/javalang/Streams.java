package polyregion.javalang;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.OptionalDouble;
import java.util.OptionalInt;
import java.util.OptionalLong;
import polyregion.jvm.runtime.Context;

/**
 * Run-time half of the stream lowering. The compiler replaces a recognised chain with a call to one
 * of the typed entries below; the chain's launches are compiled ahead of time into generated
 * launch-site classes, and this class orders them.
 *
 * <p>{@code values} holds the context (or {@code null}) and the source values in the order the
 * chain evaluates them, then the terminal's extra values, and finally the captured values of every
 * lambda in the chain, in chain order.
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

  public enum Kind {
    TABULATE('t'),
    TRANSFORM('m'),
    FILTER('f');

    final char tag;

    Kind(char tag) {
      this.tag = tag;
    }
  }

  public static final class Segment {
    public final Kind kind;
    public final int from;
    public final int to;

    Segment(Kind kind, int from, int to) {
      this.kind = kind;
      this.from = from;
      this.to = to;
    }

    public String role() {
      return kind.tag + "" + from;
    }
  }

  public static final String MAIN = "main";
  public static final String ORDER = "order";
  public static final String FILL = "fill";
  public static final String SCAN = "scan";
  public static final String TAIL = "tail";
  public static final String EACH = "each";
  public static final String SUMS = "sums";
  public static final String EXT = "ext";

  public static final char FILTER_STAGE = 'F';
  public static final char NO_NEUTRAL = 'I';

  private static final char ARRAY = 'A';
  private static final char RANGE = 'R';
  private static final char CLOSED = 'C';

  static long indexChunk = 1L << 30;
  static long arrayChunk = 1L << 24;
  private static final long MAX_ARRAY_SIZE = Integer.MAX_VALUE - 8;

  public static String shape(
      Op op, char source, char in, char out, String stages, boolean contextFirst) {
    return op.name()
        + ":"
        + source
        + ":"
        + in
        + ":"
        + out
        + ":"
        + stages
        + ":"
        + (contextFirst ? 1 : 0);
  }

  public static List<Segment> segments(String stages, boolean range) {
    List<Segment> out = new ArrayList<>();
    boolean fromRange = range;
    int start = 0;
    int i = 0;
    while (i < stages.length()) {
      if (stages.charAt(i) != FILTER_STAGE) {
        i++;
        continue;
      }
      if (fromRange) {
        out.add(new Segment(Kind.TABULATE, start, i));
        fromRange = false;
      } else if (i > start) {
        out.add(new Segment(Kind.TRANSFORM, start, i));
      }
      int j = i;
      while (j < stages.length() && stages.charAt(j) == FILTER_STAGE) j++;
      out.add(new Segment(Kind.FILTER, i, j));
      i = j;
      start = j;
    }
    if (fromRange) out.add(new Segment(Kind.TABULATE, start, stages.length()));
    else if (start < stages.length()) out.add(new Segment(Kind.TRANSFORM, start, stages.length()));
    return out;
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

  private enum How {
    SUM,
    MIN,
    MAX,
    DEVICE
  }

  private static final class Materialised {
    final Object array;
    final int length;
    final boolean fresh;

    Materialised(Object array, int length, boolean fresh) {
      this.array = array;
      this.length = length;
      this.fresh = fresh;
    }
  }

  private static final class Chain {
    final Class<?> home;
    final String key;
    final Op op;
    final char source;
    final char in;
    final char out;
    final String stages;
    final char[] kinds;
    final boolean filtered;
    final Context context;
    final Object origin;
    final long lo;
    final Object extra;
    final Object[] captured;
    final long n;

    Chain(Class<?> home, String key, String shape, Object[] values) {
      String[] parts = shape.split(":", -1);
      this.home = home;
      this.key = key;
      this.op = Op.valueOf(parts[0]);
      this.source = parts[1].charAt(0);
      this.in = parts[2].charAt(0);
      this.out = parts[3].charAt(0);
      this.stages = parts[4];
      this.kinds = new char[stages.length() + 1];
      kinds[0] = in;
      for (int i = 0; i < stages.length(); i++)
        kinds[i + 1] = stages.charAt(i) == FILTER_STAGE ? kinds[i] : stages.charAt(i);
      this.filtered = stages.indexOf(FILTER_STAGE) >= 0;
      boolean contextFirst = parts[5].equals("1");
      int at = 0;
      Context selected = contextFirst ? (Context) values[at++] : null;
      if (source == ARRAY) {
        origin = values[at++];
        n = Array.getLength(origin);
        lo = 0;
      } else {
        lo = ((Number) values[at++]).longValue();
        long hi = ((Number) values[at++]).longValue();
        origin = null;
        boolean empty = source == CLOSED ? hi < lo : hi <= lo;
        n = empty ? 0 : source == CLOSED ? Math.addExact(Math.subtractExact(hi, lo), 1) : Math.subtractExact(hi, lo);
      }
      this.context = contextFirst ? selected : (Context) values[at++];
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

    Object base(long offset) {
      return source == ARRAY ? origin : coerce(lo + offset, in);
    }

    long step(long chunk) {
      return source == ARRAY ? n : chunk;
    }

    int length(long offset, long step) {
      return (int) Math.min(step, n - offset);
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
          {
            long step = step(indexChunk);
            for (long offset = 0; offset < n; offset += step)
              launch(EACH, base(offset), null, (long) length(offset, step), null);
            return null;
          }
        case SET_ALL:
          if (n > 0) launch(FILL, origin, null, origin, (int) n, null);
          return null;
        case PREFIX:
          if (n > 1) {
            Object scanned = newArray(out, (int) n);
            launch(SCAN, origin, null, origin, scanned, (int) n, null);
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
      if (length > 1) launch(ORDER, base(0), null, array, length, null);
    }

    Materialised materialise(Object base, int length) {
      Object current = source == ARRAY ? base : null;
      boolean fresh = false;
      int size = length;
      for (Segment segment : segments(stages, source != ARRAY)) {
        char to = kinds[segment.to];
        char from = kinds[segment.from];
        switch (segment.kind) {
          case TABULATE:
            {
              Object result = newArray(to, size);
              launch(segment.role(), base, null, result, size, null);
              current = result;
              break;
            }
          case TRANSFORM:
            {
              Object result = newArray(to, size);
              launch(segment.role(), base, null, current, result, size, null);
              current = result;
              break;
            }
          default:
            {
              Object survivors = newArray(from, size);
              int kept = (Integer) launch(segment.role(), base, null, current, size, survivors, null);
              if (kept == 0) return new Materialised(newArray(out, 0), 0, true);
              current = survivors;
              size = kept;
            }
        }
        fresh = true;
      }
      return new Materialised(current, size, fresh);
    }

    Object trimmed(Materialised m) {
      if (m.fresh && Array.getLength(m.array) == m.length) return m.array;
      return copyOf(m.array, out, 0, m.length);
    }

    static IllegalArgumentException tooLarge() {
      return new IllegalArgumentException("Stream size exceeds max array size");
    }

    Object toArray() {
      if (n == 0) return newArray(out, 0);
      if (source == ARRAY) return trimmed(materialise(origin, (int) n));
      if (!filtered) {
        if (n >= MAX_ARRAY_SIZE) throw tooLarge();
        return trimmed(materialise(base(0), (int) n));
      }
      List<Object> parts = new ArrayList<>();
      long total = 0;
      for (long offset = 0; offset < n; offset += arrayChunk) {
        Materialised m = materialise(base(offset), length(offset, arrayChunk));
        if (m.length == 0) continue;
        total += m.length;
        if (total >= MAX_ARRAY_SIZE) throw tooLarge();
        parts.add(trimmed(m));
      }
      if (parts.size() == 1) return parts.get(0);
      Object all = newArray(out, (int) total);
      int at = 0;
      for (Object part : parts) {
        int size = Array.getLength(part);
        System.arraycopy(part, 0, all, at, size);
        at += size;
      }
      return all;
    }

    Object tail(Object array, int length) {
      Object first = Array.get(array, 0);
      if (length == 1) return first;
      return launch(TAIL, base(0), null, copyOf(array, out, 1, length), length - 1, first, null);
    }

    Object tailReduction() {
      List<Object> parts = new ArrayList<>();
      long step = source == ARRAY ? n : arrayChunk;
      for (long offset = 0; offset < n; offset += step) {
        Materialised m = materialise(base(offset), length(offset, step));
        if (m.length > 0) parts.add(tail(m.array, m.length));
      }
      if (parts.isEmpty()) return empty(out);
      if (parts.size() == 1) return present(out, parts.get(0));
      Object partials = newArray(out, parts.size());
      for (int i = 0; i < parts.size(); i++) Array.set(partials, i, parts.get(i));
      return present(out, tail(partials, parts.size()));
    }

    String site(String role) {
      return siteName(key, role);
    }

    Object[] flat(Object base, Object neutral) {
      Object[] all = new Object[captured.length + 2];
      all[0] = base;
      all[1] = neutral == null ? zero(NO_NEUTRAL) : neutral;
      System.arraycopy(captured, 0, all, 2, captured.length);
      return all;
    }

    Ops.Sum compensated(long[] kept) {
      Ops.Sum total = null;
      long step = step(indexChunk);
      for (long offset = 0; offset < n; offset += step) {
        long length = length(offset, step);
        int blocks = (int) ((length + Ops.BLOCK - 1) / Ops.BLOCK);
        double[] out = new double[4 * blocks];
        Object[] all = java.util.Arrays.copyOf(flat(base(offset), length), captured.length + 3);
        all[all.length - 1] = out;
        Launch.step(home, site(SUMS), context, new Object[] {(long) blocks, null}, all);
        for (int b = 0; b < blocks; b++) {
          long count = (long) out[4 * b + 3];
          if (count == 0) continue;
          kept[0] += count;
          Ops.Sum block = new Ops.Sum(out[4 * b], out[4 * b + 1], out[4 * b + 2]);
          total = total == null ? block : Ops.sumMerge(total, block);
        }
      }
      return total;
    }

    Object extremumPartials(How how, long[] count) {
      Object best = null;
      long total = 0;
      long step = step(indexChunk);
      int maxBlocks = (int) ((step + Ops.BLOCK - 1) / Ops.BLOCK);
      Object values = newArray(out, 2 * maxBlocks);
      long[] counts = new long[2 * maxBlocks];
      for (long offset = 0; offset < n; offset += step) {
        long length = length(offset, step);
        int blocks = (int) ((length + Ops.BLOCK - 1) / Ops.BLOCK);
        Object[] all = Arrays.copyOf(flat(base(offset), length), captured.length + 4);
        all[captured.length + 2] = values;
        all[captured.length + 3] = counts;
        Launch.step(home, site(EXT), context, new Object[] {(long) blocks, null}, all);
        for (int b = 0; b < blocks; b++) {
          long kept = counts[2 * b + 1];
          if (kept == 0) continue;
          total += kept;
          Object value = Array.get(values, 2 * b);
          best = best == null ? value : combine(how, out, best, value);
        }
      }
      count[0] = total;
      return best;
    }

    Object launch(String role, Object base, Object neutral, Object... args) {
      return Launch.step(home, site(role), context, args, flat(base, neutral));
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

    static Object extremum(char type, boolean min) {
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

    static Object combine(How how, char type, Object a, Object b) {
      switch (type) {
        case 'I':
          return how == How.SUM
              ? Ops.sumI((Integer) a, (Integer) b)
              : how == How.MIN ? Ops.minI((Integer) a, (Integer) b) : Ops.maxI((Integer) a, (Integer) b);
        case 'J':
          return how == How.SUM
              ? Ops.sumJ((Long) a, (Long) b)
              : how == How.MIN ? Ops.minJ((Long) a, (Long) b) : Ops.maxJ((Long) a, (Long) b);
        default:
          return how == How.SUM
              ? Ops.sumD((Double) a, (Double) b)
              : how == How.MIN ? Ops.minD((Double) a, (Double) b) : Ops.maxD((Double) a, (Double) b);
      }
    }

    List<Object> partials(Object neutral) {
      List<Object> parts = new ArrayList<>();
      long step = step(indexChunk);
      for (long offset = 0; offset < n; offset += step) {
        int length = length(offset, step);
        parts.add(
            source == ARRAY
                ? launch(MAIN, origin, neutral, origin, length, neutral, null, null)
                : launch(MAIN, base(offset), neutral, (long) length, neutral, null, null));
      }
      return parts;
    }

    Object merge(List<Object> parts, How how, char type, Object identity) {
      if (parts.size() > 1 && how == How.DEVICE) {
        Object all = newArray(type, parts.size());
        for (int i = 0; i < parts.size(); i++) Array.set(all, i, parts.get(i));
        return launch(TAIL, base(0), null, all, parts.size(), identity, null);
      }
      Object acc = parts.get(0);
      for (int i = 1; i < parts.size(); i++) acc = combine(how, type, acc, parts.get(i));
      return acc;
    }

    Object total(char type, Object neutral, How how) {
      return merge(partials(neutral), how, type, neutral);
    }

    List<Object> survivorPartials(Object neutral, long[] count) {
      List<Object> parts = new ArrayList<>();
      long step = source == ARRAY ? n : arrayChunk;
      for (long offset = 0; offset < n; offset += step) {
        Materialised m = materialise(base(offset), length(offset, step));
        if (m.length == 0) continue;
        count[0] += m.length;
        parts.add(launch(MAIN, base(offset), neutral, m.array, m.length, neutral, null, null));
      }
      return parts;
    }

    Object reduction() {
      switch (op) {
        case COUNT:
          if (!filtered) return n;
          return n == 0 ? (Object) 0L : total('J', 0L, How.SUM);
        case SUM:
          {
            if (n == 0) return zero(out);
            if (out != 'D') return total(out, zero(out), How.SUM);
            Ops.Sum sum = compensated(new long[1]);
            return sum == null ? zero(out) : (Object) Ops.sumValue(sum);
          }
        case REDUCE_ID:
          return n == 0 ? extra : total(out, extra, How.DEVICE);
        case ANY:
          return n != 0 && (Long) total('J', 0L, How.SUM) > 0;
        case ALL:
        case NONE:
          return n == 0 || (Long) total('J', 0L, How.SUM) == 0;
        case AVERAGE:
          {
            if (n == 0) return OptionalDouble.empty();
            if (out == 'D') {
              long[] count = {0};
              Ops.Sum sum = compensated(count);
              return sum == null ? OptionalDouble.empty() : OptionalDouble.of(Ops.sumValue(sum) / count[0]);
            }
            char wide = 'J';
            if (!filtered) {
              Object sum = total(wide, zero(wide), How.SUM);
              return OptionalDouble.of(((Number) sum).doubleValue() / n);
            }
            long[] count = {0};
            List<Object> parts = survivorPartials(zero(wide), count);
            if (count[0] == 0) return OptionalDouble.empty();
            Object sum = merge(parts, How.SUM, wide, zero(wide));
            return OptionalDouble.of(((Number) sum).doubleValue() / count[0]);
          }
        default:
          {
            boolean min = op == Op.MIN;
            How how = min ? How.MIN : How.MAX;
            if (n == 0) return empty(out);
            if (!filtered) return present(out, total(out, extremum(out, min), how));
            long[] count = {0};
            Object value = extremumPartials(how, count);
            if (count[0] == 0) return empty(out);
            return present(out, value);
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
