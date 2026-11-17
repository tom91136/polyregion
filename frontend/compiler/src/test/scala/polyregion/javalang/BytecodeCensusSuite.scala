package polyregion.javalang

class BytecodeCensusSuite extends munit.FunSuite {

  private val full  = sys.props.get("polyregion.bytecode.census").contains("full")
  private val limit = if (full) Int.MaxValue else 500

  override def munitTimeout: scala.concurrent.duration.Duration =
    scala.concurrent.duration.Duration(20, "min")

  private def census(name: String, marker: Class[?], floor: Int): Unit =
    test(s"every method of $name either lowers or is refused by name") {
      val report = Census.scan(Census.Corpus(name, Census.locate(marker)), limit)
      println(report.render)
      assert(report.crashes.isEmpty, s"${report.crashes.size} method(s) crashed the frontend:\n${report.render}")
      assert(report.stalled.isEmpty, s"the scan stalled:\n${report.render}")
      assert(report.attempted > 0, s"the corpus is empty:\n${report.render}")
      // a ratchet over the sample, not a target
      if (!full) assert(report.lowered >= floor, s"lowered ${report.lowered}, floor $floor:\n${report.render}")
    }

  census("scala-library", classOf[scala.collection.immutable.List[?]], floor = 55)
  census("scala3-compiler", classOf[dotty.tools.dotc.Driver], floor = 170)
  census("guava", classOf[com.google.common.math.IntMath], floor = 50)
}
