package polyregion;

import java.nio.Buffer;
import java.util.concurrent.atomic.AtomicBoolean;

public class PolyregionRuntime {

    public static byte TYPE_BOOL = 1;
    public static byte TYPE_BYTE = 2;
    public static byte TYPE_CHAR = 3;
    public static byte TYPE_SHORT = 4;
    public static byte TYPE_INT = 5;
    public static byte TYPE_LONG = 6;
    public static byte TYPE_FLOAT = 7;
    public static byte TYPE_DOUBLE = 8;
    public static byte TYPE_PTR = 9;
    public static byte TYPE_VOID = 10;

    public static native void invoke(
            byte[] object, String symbol, //
            byte returnType, Buffer returnPtr, //
            byte[] paramTypes, Buffer[] paramPtrs //
    );

    private static AtomicBoolean loaded = new AtomicBoolean();

    public static void load() {
        if (!loaded.getAndSet(true)) {
            try {

                System.load("/home/tom/polyregion/native/cmake-build-release-clang/bindings/libjava-runtime.so");
            } catch (Throwable e) {
                e.printStackTrace();
            }
        }
    }

}
