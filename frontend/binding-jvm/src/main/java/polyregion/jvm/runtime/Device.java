package polyregion.jvm.runtime;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.List;
import java.util.Objects;

public final class Device implements AutoCloseable {

  public static class Queue implements AutoCloseable {

    final long nativePeer;

    Queue(long nativePeer) {
      this.nativePeer = nativePeer;
    }

    public void enqueueHostToDeviceAsync(ByteBuffer src, long dst, int size, Runnable cb) {
      Runtime.enqueueHostToDeviceAsync0(nativePeer, src, dst, size, cb);
    }

    public void enqueueDeviceToHostAsync(long src, ByteBuffer dst, int size, Runnable cb) {
      Runtime.enqueueDeviceToHostAsync0(nativePeer, src, dst, size, cb);
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

      Runtime.enqueueInvokeAsync0(
          nativePeer, moduleName, symbol, argTys, buffer.array(), policy, cb);
    }

    public void enqueueInvokeAsync(
        String moduleName,
        String symbol,
        byte[] argTypes,
        byte[] argData,
        Policy policy,
        Runnable cb) {
      Runtime.enqueueInvokeAsync0(nativePeer, moduleName, symbol, argTypes, argData, policy, cb);
    }

    @Override
    public void close() {
      Runtime.deleteQueuePeer0(nativePeer);
    }
  }

  final long nativePeer;
  public final long id;
  public final String name;
  public final boolean sharedAddressSpace;

  Device(long nativePeer, long id, String name, boolean sharedAddressSpace) {
    this.nativePeer = nativePeer;
    this.id = id;
    this.name = Objects.requireNonNull(name);
    this.sharedAddressSpace = sharedAddressSpace;
  }

  public Property[] properties() {
    return Runtime.deviceProperties0(nativePeer);
  }

  public Queue createQueue() {
    return Runtime.createQueue0(nativePeer);
  }

  public void loadModule(String name, byte[] image) {
    Runtime.loadModule0(nativePeer, name, image);
  }

  public boolean moduleLoaded(String name) {
    return Runtime.moduleLoaded0(nativePeer, name);
  }

  public long malloc(long size, Access access) {
    return Runtime.malloc0(nativePeer, size, access.value);
  }

  public void free(long data) {
    Runtime.free0(nativePeer, data);
  }

  @Override
  public void close() {
    Runtime.deleteDevicePeer0(nativePeer);
  }

  @Override
  public String toString() {
    return "Device{" + "nativePeer=" + nativePeer + ", id=" + id + ", name='" + name + '\'' + '}';
  }
}
