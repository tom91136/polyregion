package polyregion.jvm.runtime;

import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

public final class Device implements AutoCloseable {

  public static class Queue implements AutoCloseable {

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
      long[] argPtrs = new long[args.size()];
      for (int i = 0; i < args.size(); i++) {
        argTys[i] = args.get(i).type.value;
        argPtrs[i] = args.get(i).data;
      }
      Runtimes.enqueueInvokeAsync0(
          nativePeer, moduleName, symbol, argTys, argPtrs, rtn.type.value, rtn.data, policy, cb);
    }

    public void enqueueInvokeAsync(
        String moduleName,
        String symbol,
        byte[] argTys,
        long[] argPtrs,
        byte rtnTy,
        long rtnPtr,
        Policy policy,
        Runnable cb) {
      Runtimes.enqueueInvokeAsync0(
          nativePeer, moduleName, symbol, argTys, argPtrs, rtnTy, rtnPtr, policy, cb);
    }

    @Override
    public void close() {
      Runtimes.deleteQueuePeer(nativePeer);
    }
  }

  final long nativePeer;
  public final long id;
  public final String name;

  Device(long nativePeer, long id, String name) {
    this.nativePeer = nativePeer;
    this.id = id;
    this.name = Objects.requireNonNull(name);
  }

  public Property[] properties() {
    return Runtimes.deviceProperties(nativePeer);
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
  public void close() {
    Runtimes.deleteDevicePeer(nativePeer);
  }

  @Override
  public String toString() {
    return "Device{" + "nativePeer=" + nativePeer + ", id=" + id + ", name='" + name + '\'' + '}';
  }
}
