package polyregion;

import java.util.Optional;
import java.nio.Buffer;

public class PolyregionRuntime {

    // enum Types {
    // INT, FLOAT
    // }

    public static native Optional<String> invoke(byte[] object, Buffer[] b);
}
