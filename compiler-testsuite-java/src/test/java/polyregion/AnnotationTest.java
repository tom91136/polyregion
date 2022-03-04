package polyregion;

import org.junit.Test;

import java.io.Serializable;
import java.lang.invoke.LambdaMetafactory;
import java.lang.invoke.SerializedLambda;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.stream.Collectors;
import java.util.stream.Stream;


import polyregion.java.Offload;

public class AnnotationTest {

	//	@Offload
	record A(int a) {
	}


	public static final class Foo {
		//		@Offload
		public void fn() {
			// a
			int a = 0;
//			var x = new A(2);
			int c = a + a;
		}

		//		@Offload
		public static void fn2(int x) {
			// a
			int a = 0;
//			var x = new A(2);
			int c = a + a;
		}
	}


	private static SerializedLambda getSerializedLambda(Serializable lambda) {
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

	interface F0 extends Serializable {
		default String ident() {
			return getClass().getDeclaredMethods()[1]
					.getName();
		}

		void call();
	}


	static void offload(F0 f) {
		System.out.println(StackWalker
				.getInstance(StackWalker.Option.SHOW_HIDDEN_FRAMES).walk(xs -> xs.map(x -> x.toString()).collect(Collectors.joining("\n"))).toString());


		var ctx = getSerializedLambda(f);

		//
		System.out.println("t1=" + f + " " + ctx.toString() + " === " + ctx.getImplMethodName() + " that = " + f.ident());
	}

//	public static void main(String[] args) {
//		Foo ff = new Foo();
////		offload(ff::fn);
//
//		int a = new Object().hashCode();
//		offload(( x) -> {  int b = x;}, a);
//
//		System.out.println("Hey!!");
//	}

	@Test
	public void test1() {

//		Foo ff = new Foo();
//		offload(ff::fn);

		int a = new Object().hashCode();


		offload((
				() -> System.out.println(a)));
//		offload(( () -> System.out.println(a)) );


		System.out.println("Hey!!");

	}

}
