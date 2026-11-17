package polyregion.javalang

import java.io.ByteArrayOutputStream
import java.net.URI
import javax.tools.*
import scala.collection.mutable
import scala.jdk.CollectionConverters.*

object JavaInMemory {

  private final class Source(name: String, code: String)
      extends SimpleJavaFileObject(
        URI.create(s"string:///${name.replace('.', '/')}.java"),
        JavaFileObject.Kind.SOURCE
      ) {
    override def getCharContent(ignoreEncodingErrors: Boolean): CharSequence = code
  }

  private final class Output(name: String)
      extends SimpleJavaFileObject(URI.create(s"bytes:///$name"), JavaFileObject.Kind.CLASS) {
    private val buf                                        = new ByteArrayOutputStream
    override def openOutputStream(): ByteArrayOutputStream = buf
    def bytes: Array[Byte]                                 = buf.toByteArray
  }

  final class Loaded(val classes: Map[String, Array[Byte]]) extends ClassLoader(JavaInMemory.getClass.getClassLoader) {
    override def findClass(name: String): Class[?] =
      classes.get(name) match {
        case Some(bs) => defineClass(name, bs, 0, bs.length)
        case None     => throw new ClassNotFoundException(name)
      }
    def bytesOf(name: String): Array[Byte] =
      classes.getOrElse(name, sys.error(s"javac produced no class `$name`, only ${classes.keys.mkString(", ")}"))
  }

  // Vineflower has no class file version ceiling, so fixtures track the running JDK
  val DefaultOptions: List[String] = List("-g")

  def compile(className: String, code: String, options: List[String] = DefaultOptions): Loaded = {
    val javac = ToolProvider.getSystemJavaCompiler
    require(javac != null, "no system Java compiler; the tests need a JDK rather than a JRE")
    val outputs = mutable.LinkedHashMap.empty[String, Output]
    val manager = new ForwardingJavaFileManager[JavaFileManager](javac.getStandardFileManager(null, null, null)) {
      override def getJavaFileForOutput(
          location: JavaFileManager.Location,
          name: String,
          kind: JavaFileObject.Kind,
          sibling: FileObject
      ): JavaFileObject = outputs.getOrElseUpdate(name, new Output(name))
    }
    val diagnostics = new DiagnosticCollector[JavaFileObject]
    val task = javac.getTask(
      null,
      manager,
      diagnostics,
      options.asJava,
      null,
      List[JavaFileObject](new Source(className, code)).asJava
    )
    if (!task.call())
      sys.error(diagnostics.getDiagnostics.asScala.map(_.toString).mkString("javac failed:\n", "\n", ""))
    new Loaded(outputs.map((k, v) => k -> v.bytes).toMap)
  }
}
