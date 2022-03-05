package polyregion.java;

import java.io.Serializable;
import java.lang.invoke.SerializedLambda;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;

public final class Runtime {

	private Runtime() {
		throw new AssertionError();
	}

	private static SerializedLambda extractSerializedLambda(Serializable lambda) {
		for (Class<?> cl = lambda.getClass(); cl != null; cl = cl.getSuperclass()) {
			try {
				Method m = cl.getDeclaredMethod("writeReplace");
				m.setAccessible(true);
				Object replacement = m.invoke(lambda);
				if (!(replacement instanceof SerializedLambda))
					throw new AssertionError("Unexpected writeReplace instance (" + replacement.getClass() + "), lambda object does not implement Serializable");
				return (SerializedLambda) replacement;
			} catch (NoSuchMethodException ignored) {
				// keep going
			} catch (IllegalAccessException | InvocationTargetException | SecurityException e) {
				throw new RuntimeException(e);
			}
		}
		throw new AssertionError("Missing writeReplace method for lambda that implements Serializable!");
	}


	public static void offload(OffloadRunnable f) {
		var lambda = extractSerializedLambda(f);
		System.out.println("f0=" + f + " " + lambda.toString() + " === " + lambda.getImplMethodName());
		f.run();
	}

	public static <R> R offload(OffloadFunction<R> f) {
		var lambda = extractSerializedLambda(f);
		System.out.println("f1=" + f + " " + lambda.toString() + " === " + lambda.getImplMethodName());
		return f.run();
	}

}
