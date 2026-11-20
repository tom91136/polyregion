package polyregion.jvm.runtime;

@FunctionalInterface
public interface Completion {
  void complete(Throwable failure);
}
