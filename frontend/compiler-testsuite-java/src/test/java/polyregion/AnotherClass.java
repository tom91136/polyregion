package polyregion;

import static polyregion.java.Runtime.offload;

public class AnotherClass {

	static {
		offload(() -> "1");
	}


	public  final static void a(){
		offload(() -> System.out.println("A"));
	}
}
