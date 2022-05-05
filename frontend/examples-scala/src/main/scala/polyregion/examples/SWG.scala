package polyregion.examples

import better.files.File

import java.nio.file.{FileSystems, Path}
import scala.collection.immutable.ArraySeq
import scala.collection.mutable.ArrayBuffer
import scala.collection.parallel.CollectionConverters.*

object SWG {

  def main(args: Array[String]): Unit = {

    val pwd               = File.currentWorkingDirectory
    val compressedCorpuss = pwd / "src" / "main" / "resources" / "swg" / "corpus.zip"

    val zipfs = FileSystems.newFileSystem(compressedCorpuss.path)
    import scala.jdk.CollectionConverters.*

    val lines = zipfs.getRootDirectories.asScala.toVector
      .flatMap(File(_).listRecursively)
      .par
      .map(_.lines.to(ArraySeq))
      .reduceOption(_ ++ _)
      .getOrElse(ArraySeq.empty[String])

    println(s"Corpus size = ${lines.size} lines")

    for (_ <- 0 until 10) run(lines, "foobarbaz")

  }

  def run(xs: ArraySeq[String], needle: String): Unit = {

    val results = Array.ofDim[(Int, Double)](xs.size)
    val (swgNs, _) = time {
      for (i <- results.indices.par) results(i) = i -> smithWatermanGotoh(xs(i), needle)
    }

    val (sortNs, matches) = time {
      results
        .sortBy(a => -a._2)
        .take(10)
        .zipWithIndex
    }

    println(s"SWG:  ${swgNs.toDouble / 1e6} ms")
    println(s"Sort: ${sortNs.toDouble / 1e6} ms")

    println(
      matches
        .map { case ((i, s), n) => s"  [$n] $s -> `${xs(i)}`" }
        .mkString("\n")
    )
  }

  def smithWatermanGotoh(haystack: String, needle: String, gap: Int = -1, max: Int = 1, min: Int = -2): Double = {

    inline def max4(a: Int, b: Int, c: Int, d: Int): Int = math.max(math.max(math.max(a, b), c), d)
    inline def max3(a: Int, b: Int, c: Int): Int         = math.max(math.max(a, b), c)
    inline def max2(a: Int, b: Int): Int                 = math.max(a, b)

    inline def cmp(a: String, ai: Int, b: String, bi: Int): Int = if (a(ai) == b(bi)) max else min

    if (haystack.isEmpty && needle.isEmpty) 1d
    else if (haystack.isEmpty || needle.isEmpty) 0d
    else {
      var v0 = new Array[Int](needle.length)
      var m  = max3(0, gap, cmp(haystack, 0, needle, 0))
      v0(0) = m
      var j = 1
      while (j < needle.length) {
        // if (needle(j) != ' ' && haystack.indexOf(needle(j)) < 0) return 0
        v0(j) = max3(0, v0(j - 1) + gap, cmp(haystack, 0, needle, j))
        m = max2(m, v0(j))
        j += 1
      }

      var v1 = new Array[Int](needle.length)
      var i  = 1
      while (i < haystack.length) {
        v1(0) = max3(0, v0(0) + gap, cmp(haystack, i, needle, 0))

        m = max2(m, v1(0))
        var j = 1
        while (j < needle.length) {
          v1(j) = max4(0, v0(j) + gap, v1(j - 1) + gap, v0(j - 1) + cmp(haystack, i, needle, j))
          m = max2(m, v1(j))
          j += 1
        }
        val tmp = v0
        v0 = v1
        v1 = tmp
        i += 1
      }
      m.toDouble / math.min(haystack.length, needle.length) * math.max(max, gap)
    }
  }
}
