package polyregion;

import java.nio.Buffer;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;

public class PolyregionCompiler {

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

    public static final short BACKEND_LLVM = 0;

    public static native Compilation compile(byte[] ast, boolean emitAssembly, short backend);

    private static AtomicBoolean loaded = new AtomicBoolean();

    public static void load() {
        if (!loaded.getAndSet(true)) {
            try {
                System.load("/home/tom/polyregion/native/cmake-build-debug-clang/bindings/libjava-compiler.so");
            } catch (Throwable e) {
                e.printStackTrace();
            }
        }
    }

}
