#include <array>
#include <cstdint>
#include <cstdio>
#include <vector>

#include <vulkan/vulkan.h>

namespace {

constexpr int N = 1024;

// glslangValidator -V --target-env vulkan1.0 --vn VECADD_SPV
constexpr uint32_t VECADD_SPV[] = {
    0x07230203, 0x00010000, 0x0008000b, 0x00000037, 0x00000000, 0x00020011, 0x00000001, 0x0006000b, 0x00000001, 0x4c534c47, 0x6474732e,
    0x3035342e, 0x00000000, 0x0003000e, 0x00000000, 0x00000001, 0x0006000f, 0x00000005, 0x00000004, 0x6e69616d, 0x00000000, 0x0000000b,
    0x00060010, 0x00000004, 0x00000011, 0x00000040, 0x00000001, 0x00000001, 0x00030003, 0x00000002, 0x000001c2, 0x00040005, 0x00000004,
    0x6e69616d, 0x00000000, 0x00030005, 0x00000008, 0x00000069, 0x00080005, 0x0000000b, 0x475f6c67, 0x61626f6c, 0x766e496c, 0x7461636f,
    0x496e6f69, 0x00000044, 0x00030005, 0x00000011, 0x00000050, 0x00040006, 0x00000011, 0x00000000, 0x0000006e, 0x00030005, 0x00000013,
    0x00006370, 0x00030005, 0x0000001f, 0x00000043, 0x00040006, 0x0000001f, 0x00000000, 0x00000063, 0x00030005, 0x00000021, 0x00000000,
    0x00030005, 0x00000024, 0x00000041, 0x00040006, 0x00000024, 0x00000000, 0x00000061, 0x00030005, 0x00000026, 0x00000000, 0x00030005,
    0x0000002c, 0x00000042, 0x00040006, 0x0000002c, 0x00000000, 0x00000062, 0x00030005, 0x0000002e, 0x00000000, 0x00040047, 0x0000000b,
    0x0000000b, 0x0000001c, 0x00030047, 0x00000011, 0x00000002, 0x00050048, 0x00000011, 0x00000000, 0x00000023, 0x00000000, 0x00040047,
    0x0000001e, 0x00000006, 0x00000004, 0x00030047, 0x0000001f, 0x00000003, 0x00040048, 0x0000001f, 0x00000000, 0x00000019, 0x00050048,
    0x0000001f, 0x00000000, 0x00000023, 0x00000000, 0x00030047, 0x00000021, 0x00000019, 0x00040047, 0x00000021, 0x00000021, 0x00000002,
    0x00040047, 0x00000021, 0x00000022, 0x00000000, 0x00040047, 0x00000023, 0x00000006, 0x00000004, 0x00030047, 0x00000024, 0x00000003,
    0x00040048, 0x00000024, 0x00000000, 0x00000018, 0x00050048, 0x00000024, 0x00000000, 0x00000023, 0x00000000, 0x00030047, 0x00000026,
    0x00000018, 0x00040047, 0x00000026, 0x00000021, 0x00000000, 0x00040047, 0x00000026, 0x00000022, 0x00000000, 0x00040047, 0x0000002b,
    0x00000006, 0x00000004, 0x00030047, 0x0000002c, 0x00000003, 0x00040048, 0x0000002c, 0x00000000, 0x00000018, 0x00050048, 0x0000002c,
    0x00000000, 0x00000023, 0x00000000, 0x00030047, 0x0000002e, 0x00000018, 0x00040047, 0x0000002e, 0x00000021, 0x00000001, 0x00040047,
    0x0000002e, 0x00000022, 0x00000000, 0x00040047, 0x00000036, 0x0000000b, 0x00000019, 0x00020013, 0x00000002, 0x00030021, 0x00000003,
    0x00000002, 0x00040015, 0x00000006, 0x00000020, 0x00000000, 0x00040020, 0x00000007, 0x00000007, 0x00000006, 0x00040017, 0x00000009,
    0x00000006, 0x00000003, 0x00040020, 0x0000000a, 0x00000001, 0x00000009, 0x0004003b, 0x0000000a, 0x0000000b, 0x00000001, 0x0004002b,
    0x00000006, 0x0000000c, 0x00000000, 0x00040020, 0x0000000d, 0x00000001, 0x00000006, 0x0003001e, 0x00000011, 0x00000006, 0x00040020,
    0x00000012, 0x00000009, 0x00000011, 0x0004003b, 0x00000012, 0x00000013, 0x00000009, 0x00040015, 0x00000014, 0x00000020, 0x00000001,
    0x0004002b, 0x00000014, 0x00000015, 0x00000000, 0x00040020, 0x00000016, 0x00000009, 0x00000006, 0x00020014, 0x00000019, 0x00030016,
    0x0000001d, 0x00000020, 0x0003001d, 0x0000001e, 0x0000001d, 0x0003001e, 0x0000001f, 0x0000001e, 0x00040020, 0x00000020, 0x00000002,
    0x0000001f, 0x0004003b, 0x00000020, 0x00000021, 0x00000002, 0x0003001d, 0x00000023, 0x0000001d, 0x0003001e, 0x00000024, 0x00000023,
    0x00040020, 0x00000025, 0x00000002, 0x00000024, 0x0004003b, 0x00000025, 0x00000026, 0x00000002, 0x00040020, 0x00000028, 0x00000002,
    0x0000001d, 0x0003001d, 0x0000002b, 0x0000001d, 0x0003001e, 0x0000002c, 0x0000002b, 0x00040020, 0x0000002d, 0x00000002, 0x0000002c,
    0x0004003b, 0x0000002d, 0x0000002e, 0x00000002, 0x0004002b, 0x00000006, 0x00000034, 0x00000040, 0x0004002b, 0x00000006, 0x00000035,
    0x00000001, 0x0006002c, 0x00000009, 0x00000036, 0x00000034, 0x00000035, 0x00000035, 0x00050036, 0x00000002, 0x00000004, 0x00000000,
    0x00000003, 0x000200f8, 0x00000005, 0x0004003b, 0x00000007, 0x00000008, 0x00000007, 0x00050041, 0x0000000d, 0x0000000e, 0x0000000b,
    0x0000000c, 0x0004003d, 0x00000006, 0x0000000f, 0x0000000e, 0x0003003e, 0x00000008, 0x0000000f, 0x0004003d, 0x00000006, 0x00000010,
    0x00000008, 0x00050041, 0x00000016, 0x00000017, 0x00000013, 0x00000015, 0x0004003d, 0x00000006, 0x00000018, 0x00000017, 0x000500b0,
    0x00000019, 0x0000001a, 0x00000010, 0x00000018, 0x000300f7, 0x0000001c, 0x00000000, 0x000400fa, 0x0000001a, 0x0000001b, 0x0000001c,
    0x000200f8, 0x0000001b, 0x0004003d, 0x00000006, 0x00000022, 0x00000008, 0x0004003d, 0x00000006, 0x00000027, 0x00000008, 0x00060041,
    0x00000028, 0x00000029, 0x00000026, 0x00000015, 0x00000027, 0x0004003d, 0x0000001d, 0x0000002a, 0x00000029, 0x0004003d, 0x00000006,
    0x0000002f, 0x00000008, 0x00060041, 0x00000028, 0x00000030, 0x0000002e, 0x00000015, 0x0000002f, 0x0004003d, 0x0000001d, 0x00000031,
    0x00000030, 0x00050081, 0x0000001d, 0x00000032, 0x0000002a, 0x00000031, 0x00060041, 0x00000028, 0x00000033, 0x00000021, 0x00000015,
    0x00000022, 0x0003003e, 0x00000033, 0x00000032, 0x000200f9, 0x0000001c, 0x000200f8, 0x0000001c, 0x000100fd, 0x00010038,
};

#define CHECK(x)                                                                                                                           \
  do {                                                                                                                                     \
    VkResult _r = (x);                                                                                                                     \
    if (_r) {                                                                                                                              \
      std::printf("    %s -> VkResult %d\n", #x, _r);                                                                                      \
      return 1;                                                                                                                            \
    }                                                                                                                                      \
  } while (0)

uint32_t mem_type(VkPhysicalDevice pd, uint32_t bits, VkMemoryPropertyFlags want) {
  VkPhysicalDeviceMemoryProperties mp;
  vkGetPhysicalDeviceMemoryProperties(pd, &mp);
  for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
    if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
  return UINT32_MAX;
}

int run_dev(VkPhysicalDevice pd) {
  VkPhysicalDeviceProperties props;
  vkGetPhysicalDeviceProperties(pd, &props);
  std::printf("  device '%s' (type=%u)\n", props.deviceName, props.deviceType);

  uint32_t qn = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, nullptr);
  std::vector<VkQueueFamilyProperties> qf(qn);
  vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, qf.data());
  uint32_t qfi = UINT32_MAX;
  for (uint32_t i = 0; i < qn; ++i)
    if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
      qfi = i;
      break;
    }
  if (qfi == UINT32_MAX) {
    std::printf("    no compute queue\n");
    return 1;
  }

  float prio = 1.0f;
  VkDeviceQueueCreateInfo qci = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  qci.queueFamilyIndex = qfi;
  qci.queueCount = 1;
  qci.pQueuePriorities = &prio;
  VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  dci.queueCreateInfoCount = 1;
  dci.pQueueCreateInfos = &qci;
  VkDevice dev;
  CHECK(vkCreateDevice(pd, &dci, nullptr, &dev));
  VkQueue queue;
  vkGetDeviceQueue(dev, qfi, 0, &queue);

  VkDeviceSize sz = N * sizeof(float);
  uint32_t mt = UINT32_MAX;
  std::array<VkBuffer, 3> buf{};
  std::array<VkDeviceMemory, 3> mem{};
  for (int i = 0; i < 3; ++i) {
    VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = sz;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    CHECK(vkCreateBuffer(dev, &bci, nullptr, &buf[i]));
    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(dev, buf[i], &mr);
    if (mt == UINT32_MAX) mt = mem_type(pd, mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (mt == UINT32_MAX) {
      std::printf("    no host-visible coherent memory\n");
      return 1;
    }
    VkMemoryAllocateInfo mai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = mr.size;
    mai.memoryTypeIndex = mt;
    CHECK(vkAllocateMemory(dev, &mai, nullptr, &mem[i]));
    CHECK(vkBindBufferMemory(dev, buf[i], mem[i], 0));
  }

  float *pa = nullptr, *pb = nullptr;
  CHECK(vkMapMemory(dev, mem[0], 0, sz, 0, reinterpret_cast<void **>(&pa)));
  CHECK(vkMapMemory(dev, mem[1], 0, sz, 0, reinterpret_cast<void **>(&pb)));
  for (int i = 0; i < N; ++i) {
    pa[i] = static_cast<float>(i);
    pb[i] = static_cast<float>(2 * i);
  }
  vkUnmapMemory(dev, mem[0]);
  vkUnmapMemory(dev, mem[1]);

  std::array<VkDescriptorSetLayoutBinding, 3> b{};
  for (int i = 0; i < 3; ++i) {
    b[i].binding = i;
    b[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    b[i].descriptorCount = 1;
    b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  }
  VkDescriptorSetLayoutCreateInfo dl = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  dl.bindingCount = 3;
  dl.pBindings = b.data();
  VkDescriptorSetLayout dsl;
  CHECK(vkCreateDescriptorSetLayout(dev, &dl, nullptr, &dsl));

  VkPushConstantRange pcr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t)};
  VkPipelineLayoutCreateInfo pl = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  pl.setLayoutCount = 1;
  pl.pSetLayouts = &dsl;
  pl.pushConstantRangeCount = 1;
  pl.pPushConstantRanges = &pcr;
  VkPipelineLayout playout;
  CHECK(vkCreatePipelineLayout(dev, &pl, nullptr, &playout));

  VkShaderModuleCreateInfo smci = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  smci.codeSize = sizeof(VECADD_SPV);
  smci.pCode = VECADD_SPV;
  VkShaderModule sm;
  CHECK(vkCreateShaderModule(dev, &smci, nullptr, &sm));
  VkComputePipelineCreateInfo cpci = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  cpci.stage.module = sm;
  cpci.stage.pName = "main";
  cpci.layout = playout;
  VkPipeline pipe;
  CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipe));

  VkDescriptorPoolSize psz = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3};
  VkDescriptorPoolCreateInfo dp = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  dp.maxSets = 1;
  dp.poolSizeCount = 1;
  dp.pPoolSizes = &psz;
  VkDescriptorPool pool;
  CHECK(vkCreateDescriptorPool(dev, &dp, nullptr, &pool));
  VkDescriptorSetAllocateInfo dsa = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  dsa.descriptorPool = pool;
  dsa.descriptorSetCount = 1;
  dsa.pSetLayouts = &dsl;
  VkDescriptorSet ds;
  CHECK(vkAllocateDescriptorSets(dev, &dsa, &ds));
  std::array<VkDescriptorBufferInfo, 3> bi{};
  std::array<VkWriteDescriptorSet, 3> w{};
  for (int i = 0; i < 3; ++i) {
    bi[i].buffer = buf[i];
    bi[i].offset = 0;
    bi[i].range = VK_WHOLE_SIZE;
    w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w[i].dstSet = ds;
    w[i].dstBinding = i;
    w[i].descriptorCount = 1;
    w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w[i].pBufferInfo = &bi[i];
  }
  vkUpdateDescriptorSets(dev, 3, w.data(), 0, nullptr);

  VkCommandPoolCreateInfo cp = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  cp.queueFamilyIndex = qfi;
  VkCommandPool cpool;
  CHECK(vkCreateCommandPool(dev, &cp, nullptr, &cpool));
  VkCommandBufferAllocateInfo cba = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cba.commandPool = cpool;
  cba.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cba.commandBufferCount = 1;
  VkCommandBuffer cmd;
  CHECK(vkAllocateCommandBuffers(dev, &cba, &cmd));
  VkCommandBufferBeginInfo cbb = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  cbb.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  CHECK(vkBeginCommandBuffer(cmd, &cbb));
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, playout, 0, 1, &ds, 0, nullptr);
  uint32_t n = N;
  vkCmdPushConstants(cmd, playout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(n), &n);
  vkCmdDispatch(cmd, (N + 63) / 64, 1, 1);
  CHECK(vkEndCommandBuffer(cmd));
  VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.commandBufferCount = 1;
  si.pCommandBuffers = &cmd;
  CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
  CHECK(vkQueueWaitIdle(queue));

  float *pc = nullptr;
  CHECK(vkMapMemory(dev, mem[2], 0, sz, 0, reinterpret_cast<void **>(&pc)));
  int mism = 0;
  for (int i = 0; i < N; ++i)
    mism += pc[i] != static_cast<float>(3 * i);
  int last = static_cast<int>(pc[N - 1]);
  vkUnmapMemory(dev, mem[2]);
  std::printf("    vecadd %s (c[%d]=%d, mismatches=%d)\n", mism ? "FAIL" : "PASS", N - 1, last, mism);

  vkDestroyCommandPool(dev, cpool, nullptr);
  vkDestroyDescriptorPool(dev, pool, nullptr);
  vkDestroyPipeline(dev, pipe, nullptr);
  vkDestroyShaderModule(dev, sm, nullptr);
  vkDestroyPipelineLayout(dev, playout, nullptr);
  vkDestroyDescriptorSetLayout(dev, dsl, nullptr);
  for (int i = 0; i < 3; ++i) {
    vkDestroyBuffer(dev, buf[i], nullptr);
    vkFreeMemory(dev, mem[i], nullptr);
  }
  vkDestroyDevice(dev, nullptr);
  return mism ? 1 : 0;
}

} // namespace

int main() {
  VkApplicationInfo ai = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
  ai.apiVersion = VK_API_VERSION_1_0;
  VkInstanceCreateInfo ci = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  ci.pApplicationInfo = &ai;
  VkInstance inst;
  if (vkCreateInstance(&ci, nullptr, &inst) != VK_SUCCESS) {
    std::printf("vkCreateInstance FAILED\n");
    return 1;
  }
  uint32_t n = 0;
  vkEnumeratePhysicalDevices(inst, &n, nullptr);
  if (!n) {
    std::printf("no Vulkan devices\n");
    return 2;
  }
  std::vector<VkPhysicalDevice> devs(n);
  vkEnumeratePhysicalDevices(inst, &n, devs.data());
  int fails = 0;
  for (auto pd : devs)
    fails += run_dev(pd);
  vkDestroyInstance(inst, nullptr);
  return fails ? 1 : 0;
}
