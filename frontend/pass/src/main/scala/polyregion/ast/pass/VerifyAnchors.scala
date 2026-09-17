package polyregion.ast.pass

import polyregion.ast.{PolyAST as p, *}

// runs address-refinement validation over the program; strict throws on unresolved or inconsistent use
// examples:
//   all pointer uses solved, strict       ->  pass through unchanged
//   unresolved q[i], strict              ->  throw "address refinement check: 1 error(s) ..."
//   unresolved q[i], non-strict          ->  log.info, program unchanged
//   p Global rooted at a Local           ->  log.info "region-space drift: ..."
case class VerifyAnchors(strict: Boolean = false) extends ProgramPass derives PassArgCodec {
  override def apply(program: p.Program, log: Log): p.Program = {
    val poison = Verify.validatePoison(program)
    if (poison.nonEmpty)
      log.info(s"poison reaches backend: ${poison.size} unlowered value(s):\n  ${poison.mkString("\n  ")}")
    val errs = Verify.validateRegions(program)
    if (errs.nonEmpty) {
      val msg =
        s"address refinement check: ${errs.size} error(s):\n  ${errs.mkString("\n  ")}"
      if (strict) throw RuntimeException(msg) else log.info(msg)
    }
    // space drift stays observational even under strict
    val drift = Verify.validateRegionSpaces(program)
    if (drift.nonEmpty)
      log.info(
        s"region-space drift: ${drift.size} pointer(s) declared in a different space to their inferred address:\n  ${drift
            .mkString("\n  ")}"
      )
    program
  }
}
