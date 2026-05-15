package ewgen

class SplitFnTypeSuite extends munit.FunSuite {

  test("simple") {
    val (ret, args) = Generator.splitFnType("hsa_status_t (uint16_t, const char **)")
    assertEquals(ret, "hsa_status_t")
    assertEquals(args, "(uint16_t, const char **)")
  }

  test("function pointer in return type stays attached") {
    val (ret, args) = Generator.splitFnType("void * (size_t)")
    assertEquals(ret, "void *")
    assertEquals(args, "(size_t)")
  }

  test("nested function pointer parameter") {
    val (ret, args) = Generator.splitFnType(
      "hsa_status_t (uint32_t, hsa_status_t (*)(hsa_amd_event_t *, void *), void *)"
    )
    assertEquals(ret, "hsa_status_t")
    assertEquals(args, "(uint32_t, hsa_status_t (*)(hsa_amd_event_t *, void *), void *)")
  }

  test("void params") {
    val (ret, args) = Generator.splitFnType("hsa_status_t (void)")
    assertEquals(ret, "hsa_status_t")
    assertEquals(args, "(void)")
  }

  test("typedef from source signature preserves typedef spelling") {
    val td = Generator.typedefFromSignature(
      "hsaew_hsa",
      "hsa_memory_allocate",
      """hsa_status_t hsa_memory_allocate(hsa_region_t region,
        |    size_t size,
        |    void** ptr);""".stripMargin
    )
    assertEquals(
      td,
      Some(
        """typedef hsa_status_t hsaew_hsa_hsa_memory_allocate(hsa_region_t region,
          |    size_t size,
          |    void** ptr);""".stripMargin
      )
    )
  }

  test("typedef from source signature drops storage and function attributes") {
    val td = Generator.typedefFromSignature(
      "clew_cl",
      "clCreateImage2D",
      """extern cl_mem
        |clCreateImage2D(cl_context context,
        |                size_t image_width) __attribute__((deprecated));""".stripMargin
    )
    assertEquals(
      td,
      Some(
        """typedef cl_mem
          |clew_cl_clCreateImage2D(cl_context context,
          |                size_t image_width);""".stripMargin
      )
    )
  }

  test("typedef from source signature can use a versioned public name") {
    val td = Generator.typedefFromSignature(
      "clew_cl",
      "clCreateBufferWithProperties",
      "clCreateBufferWithProperties_30",
      "cl_mem clCreateBufferWithProperties(cl_context context, const cl_mem_properties* properties);"
    )
    assertEquals(
      td,
      Some(
        "typedef cl_mem clew_cl_clCreateBufferWithProperties_30(cl_context context, const cl_mem_properties* properties);"
      )
    )
  }
}
