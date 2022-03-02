package polyregion;

import org.junit.Test;

import polyregion.java.Offload;

public class AnnotationTest {

	@Offload
	public static final class Foo {
		void fn() {
			int a = 0;
			System.out.println("a" + a);
		}
	}


	@Test
	void test1() {

		System.out.println("Hey!");

	}

}
