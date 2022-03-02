package polyregion;

import org.junit.Test;

import polyregion.java.Offload;

public class AnnotationTest {

	public static final class Foo {
		@Offload
		void fn() {
			int a = 0;
			System.out.println(" a   " + a);
		}
	}


	@Test
	public void test1() {

		System.out.println("Hey!!");

	}

}
