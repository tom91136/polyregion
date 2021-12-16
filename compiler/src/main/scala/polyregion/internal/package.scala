package polyregion

import cats.Eval
import cats.data.EitherT
import polyregion.PolyAst

package object internal {

  def VNil[A] = Vector.empty[A]

  type Result[A] = Either[Throwable, A]

  type Deferred[A] = EitherT[Eval, Throwable, A]

  extension [A](a: Result[A]) {
    def deferred: Deferred[A] = EitherT.fromEither[Eval](a)
  }

  extension [A](a: Deferred[A]) {
    def resolve: Result[A]          = a.value.value
    def withFilter(p: A => Boolean) = a.subflatMap(x => (if (p(x)) Right(x) else Left(new MatchError(x))))
  }

  extension [A](a: A) {
    def success: Result[A] = Right(a)
  }
  extension (message: => String) {
    def fail[A]: Result[A] = Left(new Exception(message))
  }
  extension (e: => Throwable) {
    def failE[A]: Result[A] = Left(e)
  }

  extension (e: => PolyAst.Sym.type) {
    def apply(raw: String): PolyAst.Sym = {
      require(!raw.isBlank)
      // normalise dollar
      PolyAst.Sym(raw.split('.').toList)
    }
  }

}
