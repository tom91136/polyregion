package polyregion.p0;

import static polyregion.java.Runtime.offload;

public class AnotherClass {

	public void f0(){
		return;
	}

	{
		offload(() -> 43);
	}
}
