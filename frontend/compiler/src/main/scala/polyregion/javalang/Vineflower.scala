package polyregion.javalang

import org.jetbrains.java.decompiler.main.collectors.ImportCollector
import org.jetbrains.java.decompiler.main.decompiler.CancelationManager
import org.jetbrains.java.decompiler.main.extern.{IFernflowerLogger, IFernflowerPreferences, IResultSaver}
import org.jetbrains.java.decompiler.main.rels.{ClassWrapper, MethodProcessor}
import org.jetbrains.java.decompiler.main.{ClassesProcessor, DecompilerContext, IdentityRenamerFactory}
import org.jetbrains.java.decompiler.modules.decompiler.stats.{GeneralStatement, RootStatement, Statement}
import org.jetbrains.java.decompiler.modules.decompiler.vars.VarProcessor
import org.jetbrains.java.decompiler.struct.attr.StructGeneralAttribute
import org.jetbrains.java.decompiler.struct.gen.MethodDescriptor
import org.jetbrains.java.decompiler.struct.{IDecompiledData, StructClass, StructContext, StructMethod}
import org.jetbrains.java.decompiler.util.DataInputFullStream
import polyregion.ast.*

import java.util.jar.Manifest
import scala.jdk.CollectionConverters.*

// drives Vineflower far enough to obtain the structured tree for one method, see BYTECODE.md
object Vineflower {

  final case class Method(cl: StructClass, mt: StructMethod, md: MethodDescriptor, root: RootStatement)

  private object Silent extends IFernflowerLogger {
    override def writeMessage(message: String, severity: IFernflowerLogger.Severity): Unit               = ()
    override def writeMessage(message: String, severity: IFernflowerLogger.Severity, t: Throwable): Unit = ()
  }

  private object NoResults extends IResultSaver {
    override def saveFolder(path: String): Unit                                                 = ()
    override def copyFile(source: String, path: String, entryName: String): Unit                = ()
    override def saveClassFile(p: String, q: String, e: String, c: String, m: Array[Int]): Unit = ()
    override def createArchive(path: String, archiveName: String, manifest: Manifest): Unit     = ()
    override def saveDirEntry(path: String, archiveName: String, entryName: String): Unit       = ()
    override def copyEntry(s: String, path: String, archiveName: String, entry: String): Unit   = ()
    override def saveClassEntry(p: String, a: String, q: String, e: String, c: String): Unit    = ()
    override def closeArchive(path: String, archiveName: String): Unit                          = ()
  }

  private object NoData extends IDecompiledData {
    override def getClassEntryName(cl: StructClass, entryName: String): String = entryName
    override def processClass(cl: StructClass): Unit                           = ()
    override def getClassContent(cl: StructClass): String                      = ""
  }

  private def preferences: java.util.Map[String, Object] = {
    val props = new java.util.HashMap[String, Object](IFernflowerPreferences.getDefaults)
    // the timeout path decompiles on a second thread whose DecompilerContext would need replicating
    props.put(IFernflowerPreferences.MAX_PROCESSING_METHOD, "0")
    props.put(IFernflowerPreferences.BYTECODE_SOURCE_MAPPING, "0")
    // assert rewriting reads <clinit> properties that only a full ClassWrapper.init computes
    props.put(IFernflowerPreferences.DECOMPILE_ASSERTIONS, "0")
    props
  }

  // DecompilerContext is thread local and the attribute registry process-wide, so one decompilation at a time
  private val lock = new Object

  private def read(classBytes: Array[Byte]): StructClass = {
    StructGeneralAttribute.init()
    StructClass.create(new DataInputFullStream(classBytes), true)
  }

  private def withContext[A](classBytes: Array[Byte])(f: StructClass => A): Result[A] = lock.synchronized {
    val structContext = new StructContext(NoResults, NoData)
    val classes       = new ClassesProcessor(structContext)
    val context       = new DecompilerContext(preferences, Silent, structContext, classes, null)
    context.renamerFactory = new IdentityRenamerFactory
    DecompilerContext.setCurrentContext(context)
    try {
      val cl   = read(classBytes)
      val node = new ClassesProcessor.ClassNode(ClassesProcessor.ClassNode.Type.ROOT, cl)
      classes.getMapRootClasses.put(cl.qualifiedName, node)
      DecompilerContext.setProperty(DecompilerContext.CURRENT_CLASS, cl)
      DecompilerContext.setProperty(DecompilerContext.CURRENT_CLASS_NODE, node)
      DecompilerContext.setProperty(DecompilerContext.CURRENT_CLASS_WRAPPER, new ClassWrapper(node))
      DecompilerContext.startClass(new ImportCollector(node))
      f(cl).success
    } catch {
      case e: CancelationManager.CanceledException => s"Vineflower gave up: ${e.getMessage}".fail
      case e: Throwable                            => e.failE
    } finally DecompilerContext.setCurrentContext(null)
  }

  def classNameOf(classBytes: Array[Byte]): Result[String] =
    withContext(classBytes)(_.qualifiedName)

  // a VarExprent reads the decompiler's options, so the tree is only valid, and lowered, inside its context
  def structure[A](classBytes: Array[Byte], name: String, desc: String)(use: Method => Result[A]): Result[A] =
    withContext(classBytes) { cl =>
      val mt = cl.getMethod(name, desc)
      if (mt == null) {
        val declared = cl.getMethods.asScala.map(m => s"${m.getName}${m.getDescriptor}").toList.sorted
        s"No method `$name$desc`; the class declares ${declared.mkString(", ")}".fail
      } else if (!mt.containsCode) s"`${cl.qualifiedName}.$name` is abstract or native".fail
      else {
        val md      = MethodDescriptor.parseDescriptor(mt, null)
        val varProc = new VarProcessor(mt, md)
        DecompilerContext.startMethod(varProc)
        val m = Method(cl, mt, md, MethodProcessor.codeToJava(cl, mt, md, varProc, null))
        diagnose(m).flatMap(_ => use(m))
      }
    }.flatten

  private def walk(s: Statement): LazyList[Statement] =
    s #:: LazyList.from(s.getStats.asScala).flatMap(walk)

  // a structuring failure is a GeneralStatement in the tree, anything else a comment on the root
  private def diagnose(m: Method): Result[Unit] = {
    val comments = Option(m.root.commentLines).fold(List.empty[String])(_.asScala.toList)
    if (walk(m.root).exists(_.isInstanceOf[GeneralStatement]))
      (s"Vineflower could not recover structured control flow for `${m.cl.qualifiedName}.${m.mt.getName}`; " +
        "the method keeps an unstructured region").fail
    else if (comments.nonEmpty)
      s"Vineflower reported ${comments.mkString("; ")} while decompiling `${m.cl.qualifiedName}.${m.mt.getName}`".fail
    else ().success
  }
}
