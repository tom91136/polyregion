package polyregion.jvm.runtime;

import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

public final class Device {

  public static class Queue {

    final long nativePeer;

    Queue(long nativePeer) {
      this.nativePeer = nativePeer;
    }

    public void enqueueHostToDeviceAsync(ByteBuffer src, long dst, int size, Runnable cb) {
      Runtimes.enqueueHostToDeviceAsync0(nativePeer, src, dst, size, cb);
    }

    public void enqueueDeviceToHostAsync(long src, ByteBuffer dst, int size, Runnable cb) {
      Runtimes.enqueueDeviceToHostAsync0(nativePeer, src, dst, size, cb);
    }

    public void enqueueInvokeAsync(
        String moduleName, String symbol, List<Arg> args, Arg rtn, Policy policy, Runnable cb) {
      byte[] argTys = new byte[args.size()];
      ByteBuffer[] argBuffers = new ByteBuffer[args.size()];
      for (int i = 0; i < args.size(); i++) {
        argTys[i] = args.get(i).type.value;
        argBuffers[i] = args.get(i).data;
      }
      Runtimes.enqueueInvokeAsync0(
          nativePeer, moduleName, symbol, argTys, argBuffers, rtn.type.value, rtn.data, policy, cb);
    }

    void enqueueInvokeAsync(
        String moduleName,
        String symbol,
        byte[] argTys,
        ByteBuffer[] argBuffers,
        byte rtnTy,
        ByteBuffer rtnBuffer,
        Policy policy,
        Runnable cb) {
      Runtimes.enqueueInvokeAsync0(
          nativePeer, moduleName, symbol, argTys, argBuffers, rtnTy, rtnBuffer, policy, cb);
    }
  }

  final long nativePeer;
  public final long id;
  public final String name;
  public final Property[] properties;

  Device(long nativePeer, long id, String name, Property[] properties) {
    this.nativePeer = nativePeer;
    this.id = id;
    this.name = Objects.requireNonNull(name);
    this.properties = Objects.requireNonNull(properties);
  }

  public Queue createQueue() {
    return Runtimes.createQueue0(nativePeer);
  }

  public void loadModule(String name, byte[] image) {
    Runtimes.loadModule0(nativePeer, name, image);
  }

  public long malloc(long size, Access access) {
    return Runtimes.malloc0(nativePeer, size, access.value);
  }

  public void free(long data) {
    Runtimes.free0(nativePeer, data);
  }

  @Override
  public String toString() {
    return "Device{"
        + "id="
        + id
        + ", name='"
        + name
        + '\''
        + ", properties="
        + Arrays.toString(properties)
        + '}';
  }
}
