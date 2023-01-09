package polyregion;

import org.junit.Test;

import java.io.Serializable;
import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import polyregion.javalang.OffloadFunction;
import polyregion.javalang.OffloadRunnable;

import static polyregion.javalang.Runtime.offload;

public class AnnotationTest {

	@Retention(RetentionPolicy.RUNTIME)
	@Target(ElementType.TYPE_USE)
	@interface X {

	}

	//	@Offload
	class A {
		int a;
	}


	public static final class Foo {

		public static final class Bar {

		}
		//		@Offload

		public void fn() {
			// a
			int a = 0;
//			var x = new A(2);
			int c = a + a;
		}
		public int f1() {
			return 2;
		}

		//		@Offload
		public static void fn2(int x) {
			// a
			int a = 0;
//			var x = new A(2);
			int c = a + a;
		}
	}



	static String x = "ab";
	OffloadRunnable f0 = () -> System.out.println(x);


	interface Bad extends Serializable, Runnable {

	}

	interface Arch{}
	interface X86Arch extends Arch{}
	static class Haswell implements X86Arch {}

	interface CUDAArch extends Arch{
		@Target(ElementType.TYPE_USE)
		@interface Config{
			Class<? extends CUDAArch> value();
			OptA opt() default OptA.O1;
			String flags() default "A";
		}
	}
	static class SM51 implements CUDAArch {}
	static class Many implements CUDAArch {
		@Target(ElementType.TYPE_USE)
		@interface Support{
			CUDAArch.Config[] value();
		}





	}

	interface Opt{
		interface O1 extends Opt	{}
		interface O2 extends Opt	{}
		interface O3 extends Opt	{}
	}


	interface Config<A extends Arch, O extends Opt>{}

	interface Device<A extends Arch>{
		<C  extends A > void offload( OffloadRunnable r);
	}

	enum OptA{
		O1, O2, O3
	}

	@Retention(RetentionPolicy.RUNTIME)
	@Target(ElementType.TYPE_USE)
	@interface ConfigA{
		 OptA opt();
	}



	@Test
	public void test1() {

		Device<CUDAArch> d = null;

//		new Config<@X SM51, Opt.O1>(){}
		d.<@Many.Support(SM51.class) Many>offload(  () -> System.out.println("A"));
		d.<@CUDAArch.Config(SM51.class) SM51>offload(  () -> System.out.println("A"));

//		d.<Config<Haswell, Opt.O1>>offload(() -> System.out.println("A"));


//		Foo ff = new Foo();
//		offload(ff::fn);


		int a = new Object().hashCode();
		int b = 2;


		List<Integer> xs = new ArrayList<>();
		xs.add(32);

		OffloadRunnable f0 = () -> System.out.println(a + b);
		OffloadFunction<Integer> f1 = () -> (a + b);

		String str = "b";

		Foo foo = new Foo();
		offload(foo::fn);
		offload(foo::fn);
		offload(foo::f1);

		Integer r = offload(() -> (a + b));


		Runnable rr1 = () -> {
			System.out.println("AAAAA");
		};

		Bad rr2 = () -> {
			System.out.println("BBB");
		};


		Arrays.asList(1, 2).stream().map(x -> x + 1).collect(Collectors.toList());
		offload((
				() -> System.out.println(a)));
		offload((
				() -> System.out.println(a)));
//		offload(( () -> System.out.println(a)) );

		polyregion.p0.AnotherClass c = new polyregion.p0.AnotherClass();
		offload(c::f0);


		System.out.println("Hey!!");


		for (int i = 0; i < 10; i++) {
			int finalI = i;
//			int x = offload(() -> finalI);

		}

		// d.aot.task(opt, )

		int x = offload(() -> 0);
		int y = offload(() -> 10);
		int z = offload(() -> 5);
		int w = offload(() -> 5);

	}

	OffloadRunnable f1 = () -> System.out.println("aa");
	OffloadRunnable f11 = () -> System.out.println("b");


}
