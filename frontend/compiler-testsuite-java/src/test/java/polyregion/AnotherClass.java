package polyregion;

import static polyregion.javalang.Runtime.offload;

public class AnotherClass {

	static {
		offload(() -> "1");
	}


	public  final static void a(){
		offload(() -> System.out.println("A"));
	}
}
