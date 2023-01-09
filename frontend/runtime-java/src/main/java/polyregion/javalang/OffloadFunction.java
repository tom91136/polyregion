package polyregion.javalang;

public interface OffloadFunction<R> extends OffloadRegion {
  R run();
}
