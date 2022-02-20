package polyregion.examples

inline def time[A](inline f: => A): (Long, A) = {
  val start = System.nanoTime()
  val a     = f
  (System.nanoTime() - start, a)
}
