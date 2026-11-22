package polyregion.javalang

import polyregion.javalang.Interop.absent

import java.io.ByteArrayOutputStream
import java.net.URI
import javax.annotation.processing.Processor as JavacProcessor
import javax.tools.*
import scala.collection.mutable
import scala.jdk.CollectionConverters.*

object JavaInMemory {

  private class Source(name: String, code: String)
      extends SimpleJavaFileObject(
        URI.create(s"string:///${name.replace('.', '/')}.java"),
        JavaFileObject.Kind.SOURCE
      ) {
    override def getCharContent(ignoreEncodingErrors: Boolean): CharSequence = code
  }

  private class Output(name: String)
      extends SimpleJavaFileObject(URI.create(s"bytes:///$name"), JavaFileObject.Kind.CLASS) {
    private val buf                                        = ByteArrayOutputStream()
    override def openOutputStream(): ByteArrayOutputStream = buf
    def bytes: Array[Byte]                                 = buf.toByteArray
  }

  private class Manager(delegate: JavaFileManager, outputs: mutable.Map[String, Output])
      extends ForwardingJavaFileManager[JavaFileManager](delegate) {
    override def getJavaFileForOutput(
        location: JavaFileManager.Location,
        name: String,
        kind: JavaFileObject.Kind,
        sibling: FileObject
    ): JavaFileObject = outputs.getOrElseUpdate(name, Output(name))

    override def getFileForOutput(
        location: JavaFileManager.Location,
        packageName: String,
        relativeName: String,
        sibling: FileObject
    ): FileObject = {
      val name = (if (packageName.isEmpty) relativeName else s"$packageName.$relativeName").stripSuffix(".class")
      outputs.getOrElseUpdate(name, Output(name))
    }
  }

  class Loaded(val classes: Map[String, Array[Byte]], val messages: List[String] = Nil)
      extends ClassLoader(JavaInMemory.getClass.getClassLoader) {
    override def findClass(name: String): Class[?] =
      classes.get(name) match {
        case Some(bs) => defineClass(name, bs, 0, bs.length)
        case None     => throw ClassNotFoundException(name)
      }
    def bytesOf(name: String): Array[Byte] =
      classes.getOrElse(name, sys.error(s"javac produced no class `$name`, only ${classes.keys.mkString(", ")}"))
  }

  // Vineflower has no class file version ceiling, so fixtures track the running JDK
  val DefaultOptions: List[String] = List("-g")

  def compile(
      className: String,
      code: String,
      options: List[String] = DefaultOptions,
      processors: List[JavacProcessor] = Nil
  ): Loaded = {
    val javac = ToolProvider.getSystemJavaCompiler
    require(Option(javac).isDefined, "no system Java compiler; the tests need a JDK rather than a JRE")
    val outputs     = mutable.LinkedHashMap.empty[String, Output]
    val manager     = Manager(javac.getStandardFileManager(absent, absent, absent), outputs)
    val diagnostics = DiagnosticCollector[JavaFileObject]()
    val task = javac.getTask(
      absent,
      manager,
      diagnostics,
      options.asJava,
      absent,
      List[JavaFileObject](Source(className, code)).asJava
    )
    if (processors.nonEmpty) task.setProcessors(processors.asJava)
    if (!task.call())
      sys.error(diagnostics.getDiagnostics.asScala.map(_.toString).mkString("javac failed:\n", "\n", ""))
    Loaded(outputs.map((k, v) => k -> v.bytes).toMap, diagnostics.getDiagnostics.asScala.map(_.toString).toList)
  }
}
