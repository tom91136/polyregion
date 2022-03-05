package polyregion;

import org.junit.Test;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

import polyregion.java.OffloadFunction;
import polyregion.java.OffloadRunnable;

import static polyregion.java.Runtime.offload;

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


//	public static void main(String[] args) {
//		Foo ff = new Foo();
////		offload(ff::fn);
//
//		int a = new Object().hashCode();
//		offload(( x) -> {  int b = x;}, a);
//
//		System.out.println("Hey!!");
//	}

	static String x = "a";
	OffloadRunnable f0 = () -> System.out.println(x);


	interface Bad extends Serializable, Runnable {

	}

	@Test
	public void test1() {

//		Foo ff = new Foo();
//		offload(ff::fn);

		int a = new Object().hashCode();
		int b = 2;


		var xs = new ArrayList<Integer>();
		xs.add(32);

		OffloadRunnable f0 = () -> System.out.println(a + b);
		OffloadFunction<Integer> f1 = () -> (a + b);

		var str = "b";

		Foo foo = new Foo();
		offload(foo::fn);

		var r = offload(() -> (a + b));


		Runnable rr1 = () -> {
			System.out.println("AAA");
		};

		Bad rr2 = () -> {
			System.out.println("BBB");
		};

		List.of(1, 2).stream().map(x -> x + 1).collect(Collectors.toList());
		offload((
				() -> System.out.println(a)));
		offload((
				() -> System.out.println(a)));
//		offload(( () -> System.out.println(a)) );


		System.out.println("Hey!!");

	}

	OffloadRunnable f1 = () -> System.out.println("aa");
	OffloadRunnable f11 = () -> System.out.println("b");


}
