package polyregion.ast.pass

import polyregion.ast.{PolyAST as p, *}

// runs Verify.validateRegions over the program; strict throws on any opaque-origin access, else logs
// examples:
//   no opaque access, strict             ->  pass through unchanged
//   q[i] with q opaque, strict           ->  throw "binding-slot region check: 1 access(es) ..."
//   q[i] with q opaque, non-strict       ->  log.info, program unchanged
//   p Global rooted at a Local           ->  log.info "region-space drift: ..."
case class VerifyAnchors(strict: Boolean = false) extends ProgramPass derives PassArgCodec {
  override def apply(program: p.Program, log: Log): p.Program = {
    val errs = Verify.validateRegions(program)
    if (errs.nonEmpty) {
      val msg =
        s"binding-slot region check: ${errs.size} access(es) through opaque-origin pointers:\n  ${errs.mkString("\n  ")}"
      if (strict) throw RuntimeException(msg) else log.info(msg)
    }
    // space drift stays observational even under strict
    val drift = Verify.validateRegionSpaces(program)
    if (drift.nonEmpty)
      log.info(
        s"region-space drift: ${drift.size} pointer(s) declared in a different space to their root:\n  ${drift.mkString("\n  ")}"
      )
    program
  }
}
