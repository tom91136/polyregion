package polyregion.javalang;

public abstract class LaunchRegion extends LaunchSite implements OffloadExecutable {

  @Override
  public final Object invoke(Object[] captured) {
    return Launch.region(this, captured);
  }
}
