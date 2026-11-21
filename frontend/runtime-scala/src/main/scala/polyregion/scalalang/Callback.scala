package polyregion.scalalang

// invoked exactly once, possibly on the launching thread before the launch returns
type Callback[A] = Either[Throwable, A] => Unit
