package polyregion.foo;

import java.lang.ref.Reference;
import java.lang.ref.ReferenceQueue;
import java.lang.ref.WeakReference;
import java.util.*;

public class C<A> {

    interface I<T>{}

    final class B extends I<A>{}

}

