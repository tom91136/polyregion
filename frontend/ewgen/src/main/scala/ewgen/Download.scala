package ewgen

import java.io.BufferedInputStream
import java.io.InputStream
import java.nio.file.Files
import java.nio.file.Path
import java.security.MessageDigest
import java.util.HexFormat

import scala.jdk.CollectionConverters.*
import scala.util.Using

import org.apache.commons.compress.archivers.ar.ArArchiveInputStream
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream
import org.apache.commons.compress.compressors.gzip.GzipCompressorInputStream
import sttp.client4.*

object Download {

  private lazy val backend: SyncBackend = DefaultSyncBackend()

  def fetch(cacheDir: Path, url: String): Path = {
    val md = MessageDigest.getInstance("SHA-1")
    md.update(url.getBytes("UTF-8"))
    val hash = HexFormat.of().formatHex(md.digest()).take(16)
    val out  = cacheDir.resolve(s"$hash-${url.split('/').last}")
    if (Files.exists(out)) {
      println(s"  [cache] $url -> $out")
      out
    } else {
      println(s"  [fetch] $url")
      Option(out.getParent).foreach(Files.createDirectories(_))
      val r = basicRequest.get(uri"$url").response(asByteArrayAlways).send(backend)
      if (!r.code.isSuccess) sys.error(s"GET $url -> HTTP ${r.code.code}")
      Files.write(out, r.body)
      out
    }
  }

  private def strippedPath(name: String, stripComponents: Int): Option[Path] = {
    val parts = name.split('/').filter(_.nonEmpty)
    val kept  = parts.drop(stripComponents).filter(_ != ".")
    if (kept.isEmpty) None else Some(Path.of(kept.head, kept.tail*))
  }

  private def targetPath(destDir: Path, relative: Path): Path = {
    val root = destDir.toAbsolutePath.normalize()
    val out  = root.resolve(relative).normalize()
    if (!out.startsWith(root)) sys.error(s"Refusing to extract path outside destination: $relative")
    out
  }

  private def entries[A](next: => A | Null): Iterator[A] =
    Iterator.continually(next).takeWhile(_ != null).map(_.nn)

  private def extractTarStream(in: InputStream, destDir: Path, stripComponents: Int): Unit =
    Using.resource(new TarArchiveInputStream(in)) { tar =>
      entries(tar.getNextEntry()).foreach { entry =>
        if (!entry.isGlobalPaxHeader && !entry.isPaxHeader) {
          strippedPath(entry.getName, stripComponents).foreach { relative =>
            val out = targetPath(destDir, relative)
            if (entry.isDirectory) Files.createDirectories(out)
            else if (entry.isSymbolicLink) {
              Files.createDirectories(out.getParent)
              Files.deleteIfExists(out)
              Files.createSymbolicLink(out, Path.of(entry.getLinkName))
            } else if (entry.isFile) {
              Files.createDirectories(out.getParent)
              Using.resource(Files.newOutputStream(out))(tar.transferTo)
            }
          }
        }
      }
    }

  private def extractDeb(archive: Path, destDir: Path, stripComponents: Int): Unit =
    Using.resource(new ArArchiveInputStream(new BufferedInputStream(Files.newInputStream(archive)))) { ar =>
      entries(ar.getNextEntry()).find(_.getName.startsWith("data.tar.")) match {
        case Some(entry) =>
          val payload =
            if (entry.getName.endsWith(".gz")) new GzipCompressorInputStream(ar)
            else if (entry.getName.endsWith(".tar")) ar
            else sys.error(s"Unsupported deb payload: ${entry.getName}")
          extractTarStream(payload, destDir, stripComponents)
        case None => sys.error(s"No data.tar payload found in $archive")
      }
    }

  private def deleteTree(path: Path): Unit =
    if (Files.exists(path))
      Using.resource(Files.walk(path)) { paths =>
        paths
          .iterator()
          .asScala
          .toVector
          .sortBy(_.getNameCount)(using Ordering.Int.reverse)
          .foreach(Files.deleteIfExists)
      }

  def extract(archive: Path, destDir: Path, stripComponents: Int): Path = {
    if (Files.exists(destDir)) deleteTree(destDir)
    Files.createDirectories(destDir)
    println(s"  [extract] $archive -> $destDir (strip=$stripComponents)")
    val name = archive.getFileName.toString
    if (name.endsWith(".deb")) extractDeb(archive, destDir, stripComponents)
    else {
      val in = new BufferedInputStream(Files.newInputStream(archive))
      val stream =
        if (name.endsWith(".tar.gz") || name.endsWith(".tgz")) new GzipCompressorInputStream(in)
        else if (name.endsWith(".tar")) in
        else sys.error(s"Unsupported archive extension: $name")
      extractTarStream(stream, destDir, stripComponents)
    }
    destDir
  }
}
