package polyregion

import java.io.File
import java.lang.management.ManagementFactory
import java.nio.file.{Files, Path}
import scala.jdk.CollectionConverters.*

object ChildJvm {

  private def classpath: String =
    List[Class[?]](
      classOf[ChildJvm.type],
      classOf[polyregion.jvm.runtime.Context],
      classOf[polyregion.scalalang.Platforms.type],
      classOf[scala.Predef.type],
      classOf[scala.collection.Seq[?]],
      classOf[scala.runtime.Scala3RunTime.type]
    )
      .map(c => Path.of(c.getProtectionDomain.getCodeSource.getLocation.toURI).toString)
      .distinct
      .mkString(File.pathSeparator)

  def run(main: String, env: Map[String, String], flags: List[String] = Nil): (Int, String) = {
    val java      = Path.of(System.getProperty("java.home"), "bin", "java").toString
    val jvmFlags  = ManagementFactory.getRuntimeMXBean.getInputArguments.asScala.filterNot(_.startsWith("-agentlib"))
    val errorFile = Files.createTempFile("hs_err", ".log")
    val command = List(java) ++ jvmFlags ++ flags ++
      List(s"-XX:ErrorFile=$errorFile", "-XX:-CreateCoredumpOnCrash", "-cp", classpath, main)
    val builder = new ProcessBuilder(command.asJava).redirectErrorStream(true)
    builder.environment().remove("POLYRT_PLATFORM")
    env.foreach((k, v) => builder.environment().put(k, v))
    val process = builder.start()
    val output  = new String(process.getInputStream.readAllBytes())
    val code    = process.waitFor()
    Files.deleteIfExists(errorFile)
    (code, output)
  }

  def pocl(): Map[String, String] = {
    val directory = java.nio.file.Files.createTempDirectory("polyrt-icd")
    val system    = Path.of("/etc/OpenCL/vendors/pocl.icd")
    java.nio.file.Files.writeString(
      directory.resolve("pocl.icd"),
      if (java.nio.file.Files.exists(system)) java.nio.file.Files.readString(system) else "libpocl.so.2\n"
    )
    Map("OCL_ICD_VENDORS" -> (directory.toString + File.separator))
  }
}
