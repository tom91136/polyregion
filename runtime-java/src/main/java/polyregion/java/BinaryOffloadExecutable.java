package polyregion.java;

import java.util.Arrays;

public abstract class BinaryOffloadExecutable implements OffloadExecutable {

	public abstract byte[] binary();

	@Override
	public Object invoke(Object[] args) {
		byte[] bin = binary();
		System.out.println("In offload, arg=" + Arrays.toString(args) + " bin=" + (bin == null ? "null" : bin.length));
		return null;
	}

}
