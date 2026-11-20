package polyregion.scalalang

import polyregion.jvm.runtime as rt

object Residents {
  def pointerOf(ctx: rt.Context, root: Any): Long = root match {
    case b: Buffer[?] =>
      rt.Queue.ownerOf(b.backing) match {
        case null                        => 0L
        case owner if owner == ctx.queue => owner.devicePointer(b.backing)
        case owner =>
          throw IllegalStateException(
            s"Buffer is resident on ${owner.context} but launched on $ctx, leave the scope or launch on that context"
          )
      }
    case _ => 0L
  }
}
