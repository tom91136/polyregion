package polyregion.java;

import java.io.Serializable;

public interface OffloadFunction<R> extends OffloadRegion {
	R run();
}