package polyregion.jvm.runtime;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
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
      try {

        Runtimes.enqueueHostToDeviceAsync0(nativePeer, src, dst, size,  cb);
      }catch (Throwable t){

        throw t;
      }

    }

    public void enqueueDeviceToHostAsync(long src, ByteBuffer dst, int size, Runnable cb) {
      Runtimes.enqueueDeviceToHostAsync0(nativePeer, src, dst, size, cb);
    }

    public void enqueueInvokeAsync(
        String moduleName,
        String symbol,
        List<Arg<?>> args,
        Arg<?> rtn,
        Policy policy,
        Runnable cb) {

      int bytes = 0;
      byte[] argTys = new byte[args.size() + 1];
      for (int i = 0; i < args.size(); ++i) {
        bytes += args.get(i).type.sizeInBytes;
        argTys[i] = args.get(i).type.value;
      }
      bytes += rtn.type.sizeInBytes;
      argTys[args.size()] = rtn.type.value;

      ByteBuffer buffer = ByteBuffer.allocate(bytes).order(ByteOrder.nativeOrder());
      for (Arg<?> arg : args) arg.drainTo(buffer);
      rtn.drainTo(buffer);

      Runtimes.enqueueInvokeAsync0(
          nativePeer, moduleName, symbol, argTys, buffer.array(), policy, cb);
    }

    public void enqueueInvokeAsync(
        String moduleName,
        String symbol,
        byte[] argTypes,
        byte[] argData,
        Policy policy,
        Runnable cb) {
      Runtimes.enqueueInvokeAsync0(
          nativePeer, moduleName, symbol, argTypes, argData, policy, cb);
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
