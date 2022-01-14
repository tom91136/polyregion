package polyregion;

import java.nio.Buffer;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;

public class PolyregionCompiler {

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
