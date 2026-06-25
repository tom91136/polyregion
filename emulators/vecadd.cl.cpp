#define CL_TARGET_OPENCL_VERSION 300
#include <array>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <string_view>
#include <vector>

#include <CL/cl.h>

#ifdef _WIN32
  #define NULDEV "nul"
#else
  #define NULDEV "/dev/null"
#endif

namespace {

constexpr int N = 1024;

constexpr const char *kernel_src = "__kernel void vecadd(__global const float *a, __global const float *b, "
                                   "__global float *c) {\n"
                                   "  int i = get_global_id(0); c[i] = a[i] + b[i];\n"
                                   "}\n";

bool run(cl_context ctx, cl_command_queue queue, cl_device_id dev, cl_program prog, std::string_view tag) {
  if (auto e = clBuildProgram(prog, 1, &dev, nullptr, nullptr, nullptr); e != CL_SUCCESS) {
    std::array<char, 8192> log{};
    clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, log.size(), log.data(), nullptr);
    std::printf("    %.*s build FAIL (%d): %s\n", static_cast<int>(tag.size()), tag.data(), e, log.data());
    return false;
  }
  cl_int e{};
  auto kernel = clCreateKernel(prog, "vecadd", &e);
  std::array<float, N> a{}, b{}, c{};
  for (int i = 0; i < N; ++i) {
    a[i] = static_cast<float>(i);
    b[i] = static_cast<float>(2 * i);
  }
  auto da = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof a, a.data(), &e);
  auto db = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof b, b.data(), &e);
  auto dc = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, sizeof c, nullptr, &e);
  clSetKernelArg(kernel, 0, sizeof(cl_mem), &da);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), &db);
  clSetKernelArg(kernel, 2, sizeof(cl_mem), &dc);
  std::size_t global = N;
  if ((e = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr)) != CL_SUCCESS) {
    std::printf("    %.*s enqueue FAIL %d\n", static_cast<int>(tag.size()), tag.data(), e);
    return false;
  }
  clEnqueueReadBuffer(queue, dc, CL_TRUE, 0, sizeof c, c.data(), 0, nullptr, nullptr);
  int bad = 0;
  for (int i = 0; i < N; ++i)
    bad += c[i] != 3.0f * static_cast<float>(i);
  std::printf("    %.*s %s (c[1023]=%g, mismatches=%d)\n", static_cast<int>(tag.size()), tag.data(), bad ? "FAIL" : "PASS",
              static_cast<double>(c[1023]), bad);
  return bad == 0;
}

std::string spirv_from_kernel() {
  const char *td = std::getenv("TMPDIR");
  if (!td) td = std::getenv("TEMP");
  if (!td) td = std::getenv("TMP");
  if (!td) td = "/tmp";
  const std::string dir = td;
  const auto cl = dir + "/vecadd.cl", bc = dir + "/vecadd.bc", spv = dir + "/vecadd.spv";
  {
    std::ofstream f{cl};
    if (!f) return {};
    f << kernel_src;
  }
  // clang's native SPIR-V target, else the translator (llvm-spirv) for clang builds without it
  auto sys = [](const std::string &c) { return std::system(c.c_str()) == 0; };
  const bool ok = sys("clang -x cl -cl-std=CL1.2 -target spirv64 -c \"" + cl + "\" -o \"" + spv + "\" 2>" NULDEV) ||
                  sys("clang -x cl -cl-std=CL1.2 -emit-llvm -target spir64 -c \"" + cl + "\" -o \"" + bc +
                      "\" 2>" NULDEV " && llvm-spirv \"" + bc + "\" -o \"" + spv + "\" 2>" NULDEV);
  if (!ok) return {};
  std::ifstream f{spv, std::ios::binary};
  return {std::istreambuf_iterator<char>{f}, std::istreambuf_iterator<char>{}};
}

std::string name_of(cl_platform_id p) {
  std::array<char, 256> buf{};
  clGetPlatformInfo(p, CL_PLATFORM_NAME, buf.size(), buf.data(), nullptr);
  return buf.data();
}
std::string name_of(cl_device_id d) {
  std::array<char, 256> buf{};
  clGetDeviceInfo(d, CL_DEVICE_NAME, buf.size(), buf.data(), nullptr);
  return buf.data();
}

bool device_takes_il(cl_device_id d) {
  std::array<char, 256> il{};
  clGetDeviceInfo(d, CL_DEVICE_IL_VERSION, il.size(), il.data(), nullptr);
  return il[0] != '\0';
}

} // namespace

int main() {
  cl_uint count = 0;
  clGetPlatformIDs(0, nullptr, &count);
  if (count == 0) {
    std::fprintf(stderr, "no OpenCL platforms found\n");
    return 2;
  }
  std::vector<cl_platform_id> platforms(count);
  clGetPlatformIDs(count, platforms.data(), nullptr);

  const auto spirv = spirv_from_kernel();
  if (spirv.empty()) std::printf("  note: no SPIR-V generator available, subtest skipped\n");

  int fails = 0;
  for (auto platform : platforms) {
    cl_device_id dev{};
    cl_uint ndev = 0;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &dev, &ndev);
    if (ndev == 0) {
      std::printf("  '%s': no device, skip\n", name_of(platform).c_str());
      continue;
    }
    std::printf("  '%s' -> %s\n", name_of(platform).c_str(), name_of(dev).c_str());
    cl_int e{};
    auto ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &e);
    auto queue = clCreateCommandQueueWithProperties(ctx, dev, nullptr, &e);
    auto text = kernel_src;
    auto src = clCreateProgramWithSource(ctx, 1, &text, nullptr, &e);
    fails += !run(ctx, queue, dev, src, "source");
    if (!spirv.empty()) {
      if (!device_takes_il(dev)) {
        std::printf("    spirv  device has no IL version, subtest skipped\n");
      } else if (auto prog = clCreateProgramWithIL(ctx, spirv.data(), spirv.size(), &e); e != CL_SUCCESS) {
        std::printf("    spirv  clCreateProgramWithIL FAIL %d\n", e);
        ++fails;
      } else fails += !run(ctx, queue, dev, prog, "spirv");
    }
  }
  std::printf("opencl: %d failure(s)\n", fails);
  return fails == 0 ? 0 : 1;
}
