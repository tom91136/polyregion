package polyregion.javalang

import java.io.File
import java.nio.file.{Files, Path, Paths}
import scala.jdk.CollectionConverters.*

object ScalaInMemory {

  private def locationOf(c: Class[?]): Option[String] =
    Option(c.getProtectionDomain)
      .flatMap(d => Option(d.getCodeSource))
      .flatMap(s => Option(s.getLocation))
      .map(u => Paths.get(u.toURI).toString)

  private lazy val classPath: String = List(
    classOf[scala.collection.immutable.List[?]],
    classOf[scala.deriving.Mirror],
    classOf[dotty.tools.dotc.Driver]
  ).flatMap(locationOf).distinct.mkString(File.pathSeparator)

  def compile(fileName: String, code: String): JavaInMemory.Loaded = {
    val work = Files.createTempDirectory("polyregion-scalac")
    try {
      val src = work.resolve(s"$fileName.scala")
      val out = Files.createDirectory(work.resolve("classes"))
      Files.writeString(src, code)
      val reporter = dotty.tools.dotc.Main.process(
        Array("-classpath", classPath, "-d", out.toString, "-nowarn", "-usejavacp:false", src.toString)
      )
      if (reporter.hasErrors) sys.error(s"scalac failed on `$fileName.scala`")
      val classes = Files
        .walk(out)
        .iterator
        .asScala
        .filter(p => p.getFileName.toString.endsWith(".class"))
        .map { p =>
          val relative = out.relativize(p).toString.dropRight(".class".length)
          relative.replace(File.separatorChar, '.') -> Files.readAllBytes(p)
        }
        .toMap
      new JavaInMemory.Loaded(classes)
    } finally delete(work)
  }

  private def delete(root: Path): Unit =
    Files.walk(root).sorted(java.util.Comparator.reverseOrder).forEach(Files.deleteIfExists(_))
}
