package polyregion.p0;

import static polyregion.javalang.Runtime.offload;

public class AnotherClass {

	public void f0(){
		return;
	}

	{
		offload(() -> 43);
	}
}
