import static polyregion.javalang.Runtime.offload;

public class NoPackageTest {

//	class B{}

	public static void main(String[] args) {
		offload(() -> 42);
	}

}
