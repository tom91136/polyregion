package ewgen

import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths

import mainargs.{ParserForMethods, TokensReader, arg, main}

import ewgen.Config.*

object Main {

  private inline val RocmVersion = "5.0.0"

  val All: Vector[Wrangler] = Vector(
    Wrangler(
      "hsaew",
      Vector(
        Group(
          name = "hsa",
          source = Source(
            url = s"https://github.com/ROCm/ROCR-Runtime/archive/refs/tags/rocm-$RocmVersion.tar.gz",
            stripComponents = 1,
            rootSubdir = Some("src")
          ),
          headers = Vector("inc/hsa.h", "inc/hsa_ext_amd.h"),
          includes = Vector("inc"),
          symPrefixes = Vector("hsa_", "HSA_")
        )
      )
    ),
    Wrangler(
      "clew",
      Vector(
        Group(
          name = "cl",
          source = Source(
            url = "https://github.com/KhronosGroup/OpenCL-Headers/archive/refs/tags/v2020.06.16.tar.gz",
            stripComponents = 1
          ),
          headers = Vector("CL/cl.h"),
          includes = Vector("."),
          defines = Vector(
            "CL_TARGET_OPENCL_VERSION=300",
            "CL_USE_DEPRECATED_OPENCL_1_0_APIS",
            "CL_USE_DEPRECATED_OPENCL_1_1_APIS",
            "CL_USE_DEPRECATED_OPENCL_1_2_APIS",
            "CL_USE_DEPRECATED_OPENCL_2_0_APIS",
            "CL_USE_DEPRECATED_OPENCL_2_1_APIS",
            "CL_USE_DEPRECATED_OPENCL_2_2_APIS"
          ),
          symPrefixes = Vector("cl", "CL_")
        )
      )
    ),
    Wrangler(
      "hipew",
      Vector(
        Group(
          name = "hip",
          source = Source(
            url = s"https://github.com/ROCm/HIP/archive/refs/tags/rocm-$RocmVersion.tar.gz",
            stripComponents = 1
          ),
          headers = Vector("include/hip/hip_runtime_api.h"),
          includes = Vector("include"),
          defines = Vector("__HIP_PLATFORM_AMD__=1"),
          symPrefixes = Vector("hip", "HIP_")
        ),
        Group(
          name = "hiprtc",
          source = Source(
            url = s"https://github.com/ROCm/HIP/archive/refs/tags/rocm-$RocmVersion.tar.gz",
            stripComponents = 1
          ),
          headers = Vector("include/hip/hiprtc.h"),
          includes = Vector("include"),
          defines = Vector("__HIP_PLATFORM_AMD__=1"),
          symPrefixes = Vector("hiprtc", "HIPRTC_")
        )
      )
    ),
    Wrangler(
      "cuew",
      Vector(
        Group(
          name = "cuda",
          source = Source(
            url =
              "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-misc-headers-8-0_8.0.61-1_amd64.deb",
            stripComponents = 6
          ),
          headers = Vector("include/cuda.h"),
          includes = Vector("include"),
          symPrefixes = Vector("cu", "CU")
        ),
        Group(
          name = "nvrtc",
          source = Source(
            url =
              "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-nvrtc-dev-8-0_8.0.61-1_amd64.deb",
            stripComponents = 6
          ),
          headers = Vector("include/nvrtc.h"),
          includes = Vector("include"),
          symPrefixes = Vector("nvrtc")
        )
      )
    ),
    Wrangler(
      "zeew",
      Vector(
        Group(
          name = "ze",
          source = Source(
            url = "https://github.com/oneapi-src/level-zero/archive/refs/tags/v1.0.tar.gz",
            stripComponents = 1
          ),
          headers = Vector("include/ze_api.h", "include/zes_api.h", "include/zet_api.h"),
          includes = Vector("include"),
          symPrefixes = Vector("ze", "ZE")
        )
      )
    )
  )

  given TokensReader.Simple[Path] with {
    def shortName = "path"
    def read(s: Seq[String]) = s.lastOption.toRight("missing value").map { v =>
      val p = Path.of(v)
      (if (p.isAbsolute) p else Paths.get("").toAbsolutePath.resolve(p)).normalize()
    }
  }

  @main(doc = "Regenerate the polyinvoke wrangler sources from pinned upstream headers.")
  def run(
      @arg(doc = "Output root: wranglers land under <out>/<name>/{include,src}/")
      out: Path = Paths.get("").toAbsolutePath.resolve("out").normalize(),
      @arg(doc = "Cache directory for downloaded SDK archives, keyed by URL hash")
      cache: Path = Paths.get("").toAbsolutePath.resolve(".cache").normalize(),
      @arg(doc = "Working tree for extracted archives, one subdir per group")
      work: Path = Paths.get("").toAbsolutePath.resolve("work").normalize(),
      @arg(doc = "Comma-separated wrangler names to build (default: all)")
      only: Option[String] = None
  ): Unit = {
    val selected = only.fold(All)(names => All.filter(w => names.split(',').toSet(w.name)))
    if (selected.isEmpty) println("No wranglers selected.")
    else {
      Vector(out, cache, work).foreach(Files.createDirectories(_))
      selected.foreach { w =>
        println(s"==> ${w.name}")
        try {
          val groupResults = w.groups.map { g =>
            val archive      = Download.fetch(cache, g.source.url)
            val groupWorkDir = work.resolve(w.name).resolve(g.name)
            Download.extract(archive, groupWorkDir, g.source.stripComponents)
            val sourceRoot = g.source.rootSubdir.fold(groupWorkDir)(sub => groupWorkDir.resolve(Path.of(sub)))
            val probe      = Clang.probe(g, sourceRoot)
            println(
              s"  [probed] ${g.name}: ${probe.functions.size} fns, ${probe.typeSlices.size} types, ${probe.macros.size} macros"
            )
            Generator.GroupResult(g, probe)
          }
          val output  = Generator.emit(w, groupResults)
          val outRoot = out.resolve(w.name)
          val incDir  = outRoot.resolve("include")
          val srcDir  = outRoot.resolve("src")
          Files.createDirectories(incDir)
          Files.createDirectories(srcDir)
          val hPath = incDir.resolve(s"${w.name}.h")
          val cPath = srcDir.resolve(s"${w.name}.c")
          Files.writeString(hPath, output.headerText, StandardCharsets.UTF_8)
          Files.writeString(cPath, output.sourceText, StandardCharsets.UTF_8)
          println(s"  [wrote] $hPath")
          println(s"  [wrote] $cPath")
        } catch {
          case e: Throwable =>
            println(s"  FAILED: ${e.getMessage}")
            e.printStackTrace()
        }
      }
    }
  }

  def main(args: Array[String]): Unit = ParserForMethods(this).runOrExit(args.toIndexedSeq)
}
