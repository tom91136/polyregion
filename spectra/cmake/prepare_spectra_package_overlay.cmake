cmake_minimum_required(VERSION 3.23)

foreach(required BACKEND VENDOR_ROOT OUT_DIR)
  if(NOT DEFINED ${required} OR NOT ${required})
    message(FATAL_ERROR "${required} is required")
  endif()
endforeach()

file(REMOVE_RECURSE "${OUT_DIR}")
file(MAKE_DIRECTORY "${OUT_DIR}")

function(copy_and_replace_from source_root relative_path needle replacement expected_count)
  set(source "${source_root}/${relative_path}")
  set(output "${OUT_DIR}/${relative_path}")
  if(NOT EXISTS "${source}")
    message(FATAL_ERROR "package overlay source is missing: ${source}")
  endif()
  if(EXISTS "${output}")
    file(READ "${output}" contents)
  else()
    file(READ "${source}" contents)
  endif()
  string(REGEX MATCHALL "${needle}" matches "${contents}")
  list(LENGTH matches actual_count)
  if(NOT actual_count EQUAL expected_count)
    message(FATAL_ERROR
      "package overlay anchor count changed for ${relative_path}: expected ${expected_count}, got ${actual_count}")
  endif()
  string(REGEX REPLACE "${needle}" "${replacement}" contents "${contents}")
  get_filename_component(output_parent "${output}" DIRECTORY)
  file(MAKE_DIRECTORY "${output_parent}")
  file(WRITE "${output}" "${contents}")
endfunction()

function(copy_and_replace relative_path needle replacement expected_count)
  copy_and_replace_from("${VENDOR_ROOT}" "${relative_path}" "${needle}" "${replacement}" "${expected_count}")
endfunction()

function(copy_and_replace_literal relative_path needle replacement expected_count)
  set(source "${VENDOR_ROOT}/${relative_path}")
  set(output "${OUT_DIR}/${relative_path}")
  if(NOT EXISTS "${source}")
    message(FATAL_ERROR "package overlay source is missing: ${source}")
  endif()
  if(EXISTS "${output}")
    file(READ "${output}" contents)
  else()
    file(READ "${source}" contents)
  endif()
  set(rest "${contents}")
  set(actual_count 0)
  string(LENGTH "${needle}" needle_length)
  while(TRUE)
    string(FIND "${rest}" "${needle}" offset)
    if(offset LESS 0)
      break()
    endif()
    math(EXPR next "${offset} + ${needle_length}")
    string(SUBSTRING "${rest}" ${next} -1 rest)
    math(EXPR actual_count "${actual_count} + 1")
  endwhile()
  if(NOT actual_count EQUAL expected_count)
    message(FATAL_ERROR
      "package overlay literal count changed for ${relative_path}: expected ${expected_count}, got ${actual_count}")
  endif()
  string(REPLACE "${needle}" "${replacement}" contents "${contents}")
  get_filename_component(output_parent "${output}" DIRECTORY)
  file(MAKE_DIRECTORY "${output_parent}")
  file(WRITE "${output}" "${contents}")
endfunction()

function(replace_region relative_path begin_marker end_marker replacement)
  set(source "${VENDOR_ROOT}/${relative_path}")
  set(output "${OUT_DIR}/${relative_path}")
  if(EXISTS "${output}")
    file(READ "${output}" contents)
  elseif(EXISTS "${source}")
    file(READ "${source}" contents)
  else()
    message(FATAL_ERROR "package overlay source is missing: ${source}")
  endif()
  string(FIND "${contents}" "${begin_marker}" begin_offset)
  string(FIND "${contents}" "${end_marker}" end_offset)
  if(begin_offset LESS 0 OR end_offset LESS 0 OR end_offset LESS_EQUAL begin_offset)
    message(FATAL_ERROR "package overlay region anchors changed for ${relative_path}")
  endif()
  string(SUBSTRING "${contents}" 0 ${begin_offset} prefix)
  string(SUBSTRING "${contents}" ${end_offset} -1 suffix)
  get_filename_component(output_parent "${output}" DIRECTORY)
  file(MAKE_DIRECTORY "${output_parent}")
  file(WRITE "${output}" "${prefix}${replacement}${suffix}")
endfunction()

if(BACKEND STREQUAL "cuda")
  # Package launches have one runtime-owned default stream. The Spectra producer
  # always selects thrust::device, but CCCL stores its null stream in a launcher
  # member and therefore obscures that fact from Clang's null-expression check.
  copy_and_replace(
    "thrust/system/cuda/detail/core/triple_chevron_launch.h"
    "k<<<grid, block, shared_mem, stream>>>"
    "k<<<grid, block, shared_mem, 0>>>"
    17)
  copy_and_replace(
    "thrust/system/cuda/detail/util.h"
    "cudaStreamSynchronize\\(stream\\(policy\\)\\)"
    "cudaDeviceSynchronize()"
    1)
  copy_and_replace(
    "thrust/system/cuda/detail/util.h"
    "cudaStreamSynchronize\\(stream\\)"
    "cudaDeviceSynchronize()"
    2)
  copy_and_replace(
    "thrust/system/cuda/detail/util.h"
    "cudaMemcpyDeviceToHost,\n                             stream"
    "cudaMemcpyDeviceToHost,\n                             0"
    1)
  copy_and_replace(
    "thrust/system/cuda/detail/util.h"
    "cudaMemcpyHostToDevice,\n                             stream"
    "cudaMemcpyHostToDevice,\n                             0"
    1)
  # Runtime failures are surfaced by Polyregion's remote operations; retaining
  # Thrust's host-only system_error construction makes an unreachable error arm
  # part of the package program and pulls in non-device exception machinery.
  replace_region(
    "thrust/system/cuda/detail/util.h"
    "__host__  __device__\ninline void throw_on_error(cudaError_t status)"
    "// FIXME: Move the iterators elsewhere."
    "__host__ __device__ inline void throw_on_error(cudaError_t) {}\n__host__ __device__ inline void throw_on_error(cudaError_t, char const *) {}\n\n")
  copy_and_replace(
    "thrust/system/cuda/detail/malloc_and_free.h"
    "throw thrust::system::detail::bad_alloc\\(thrust::cuda_category\\(\\)\\.message\\(status\\)\\.c_str\\(\\)\\)"
    "(void)status"
    1)
  copy_and_replace(
    "thrust/system/cuda/memory_resource.h"
    "throw thrust::system::detail::bad_alloc\\(thrust::cuda_category\\(\\)\\.message\\(status\\)\\.c_str\\(\\)\\)"
    "(void)status"
    1)
  copy_and_replace(
    "thrust/detail/allocator/temporary_allocator.inl"
    "throw thrust::system::detail::bad_alloc\\(\"temporary_buffer::allocate: get_temporary_buffer failed\"\\)"
    "(void)cnt"
    1)
  copy_and_replace(
    "thrust/system/cuda/detail/future.inl"
    "cudaStreamSynchronize\\(handle_\\.get\\(\\)\\)"
    "cudaDeviceSynchronize()"
    1)
  copy_and_replace(
    "thrust/system/cuda/detail/core/agent_launcher.h"
    "cudaStreamSynchronize\\(stream\\)"
    "cudaDeviceSynchronize()"
    1)
  copy_and_replace(
    "thrust/system/cuda/detail/core/util.h"
    "cudaStreamSynchronize\\(s\\)"
    "cudaDeviceSynchronize()"
    1)
  # CUB uses quoted paths throughout its tree; keep a single physical include
  # identity in the overlay so pragma-once headers cannot be reached twice.
  file(COPY "${VENDOR_ROOT}/cub" DESTINATION "${OUT_DIR}")
  copy_and_replace(
    "cub/util_device.cuh"
    "cudaStreamSynchronize\\(stream\\)"
    "cudaDeviceSynchronize()"
    1)
elseif(BACKEND STREQUAL "hip")
  if(NOT DEFINED ROCPRIM_ROOT OR NOT ROCPRIM_ROOT)
    message(FATAL_ERROR "ROCPRIM_ROOT is required for the HIP package overlay")
  endif()
  # Polyregion remote allocation reports its own runtime failures. rocThrust's
  # host-only bad_alloc arms are therefore unreachable package code, just as
  # their CCCL counterparts are in the CUDA overlay above.
  copy_and_replace(
    "thrust/system/hip/detail/malloc_and_free.h"
    "throw thrust::system::detail::bad_alloc\\(thrust::hip_category\\(\\)\\.message\\(status\\)\\.c_str\\(\\)\\)"
    "(void)status"
    1)
  copy_and_replace(
    "thrust/system/hip/memory_resource.h"
    "throw thrust::system::detail::bad_alloc\\(thrust::hip_category\\(\\)\\.message\\(status\\)\\.c_str\\(\\)\\)"
    "(void)status"
    1)
  copy_and_replace(
    "thrust/detail/allocator/temporary_allocator.inl"
    "throw thrust::system::detail::bad_alloc\\(\"temporary_buffer::allocate: get_temporary_buffer failed\"\\)"
    "(void)cnt"
    3)
  copy_and_replace(
    "thrust/system/hip/detail/util.h"
    "hipStreamSynchronize\\(stream\\(policy\\)\\)"
    "hipDeviceSynchronize()"
    1)
  copy_and_replace(
    "thrust/system/hip/detail/util.h"
    "hipStreamSynchronize\\(stream\\)"
    "hipDeviceSynchronize()"
    2)
  copy_and_replace(
    "thrust/system/hip/detail/util.h"
    "hipMemcpyDeviceToHost, stream"
    "hipMemcpyDeviceToHost, 0"
    2)
  copy_and_replace(
    "thrust/system/hip/detail/util.h"
    "hipMemcpyHostToDevice, stream"
    "hipMemcpyHostToDevice, 0"
    2)
  copy_and_replace(
    "thrust/system/hip/detail/util.h"
    "hipMemcpyDeviceToDevice, stream"
    "hipMemcpyDeviceToDevice, 0"
    1)
  copy_and_replace(
    "thrust/system/hip/detail/set_operations.h"
    "hipStreamSynchronize\\(stream\\)"
    "hipDeviceSynchronize()"
    1)
  # As in rocPRIM below, Spectra never enables rocThrust's optional synchronous
  # diagnostics. Keep error and synchronization behavior, but omit the
  # unmodelled iostream expressions from the derived package producer.
  set(rocthrust_set_operations "${OUT_DIR}/thrust/system/hip/detail/set_operations.h")
  file(READ "${rocthrust_set_operations}" contents)
  string(REGEX MATCHALL "std::cout" rocthrust_debug_streams "${contents}")
  list(LENGTH rocthrust_debug_streams rocthrust_debug_stream_count)
  if(NOT rocthrust_debug_stream_count EQUAL 7)
    message(FATAL_ERROR
      "rocThrust debug stream anchor count changed: expected 7, got ${rocthrust_debug_stream_count}")
  endif()
  string(REGEX REPLACE "std::cout[^;]*;" "(void)0;" contents "${contents}")
  file(WRITE "${rocthrust_set_operations}" "${contents}")
  copy_and_replace(
    "thrust/system/hip/detail/parallel_for.h"
    "\n      stream,\n"
    "\n      0,\n"
    1)
  copy_and_replace(
    "thrust/system/hip/detail/set_operations.h"
    "\n    stream,\n"
    "\n    0,\n"
    2)
  # rocThrust's device policy supplies only the default stream, but rocPRIM
  # stores it in launch_plan before issuing the tuned trampoline launch. Keep a
  # single derived rocPRIM include identity and make that known-null stream
  # explicit so package lowering can continue rejecting arbitrary user streams.
  file(COPY "${ROCPRIM_ROOT}/rocprim" DESTINATION "${OUT_DIR}")
  # Spectra never enables rocPRIM's optional debug-synchronous mode. Its
  # diagnostic stream expressions are nevertheless retained in the generic
  # host orchestration closure, where std::cout/std::cerr have no package ABI.
  # Remove only those dead diagnostics from the derived copy; synchronization,
  # error propagation, launches, and every algorithm body remain unchanged.
  file(GLOB_RECURSE rocprim_overlay_headers
       "${OUT_DIR}/rocprim/*.h" "${OUT_DIR}/rocprim/*.hpp")
  list(REMOVE_DUPLICATES rocprim_overlay_headers)
  set(rocprim_debug_stream_count 0)
  foreach(rocprim_overlay_header IN LISTS rocprim_overlay_headers)
    file(READ "${rocprim_overlay_header}" contents)
    string(REGEX MATCHALL "std::cout|std::cerr" debug_stream_statements "${contents}")
    list(LENGTH debug_stream_statements file_debug_stream_count)
    if(file_debug_stream_count GREATER 0)
      math(EXPR rocprim_debug_stream_count "${rocprim_debug_stream_count} + ${file_debug_stream_count}")
      string(REGEX REPLACE "std::cout[^;]*;|std::cerr[^;]*;" "(void)0;" contents "${contents}")
      file(WRITE "${rocprim_overlay_header}" "${contents}")
    endif()
  endforeach()
  if(NOT rocprim_debug_stream_count EQUAL 159)
    message(FATAL_ERROR
      "rocPRIM debug stream anchor count changed: expected 159, got ${rocprim_debug_stream_count}")
  endif()
  # The HIP producer is compiled for the single SPECTRA_ROCM_ARCH selected by
  # CMake. This rocPRIM revision does not list gfx1036, so its canonical lookup
  # selects the unknown-architecture fallback in the device phase. Encode that
  # same fixed result in the host phase instead of retaining the process-time
  # architecture-name table and lookup loop in every imported consumer closure.
  replace_region(
    "rocprim/device/config_types.hpp"
    "constexpr target_arch get_target_arch_from_name(const char* const arch_name, const std::size_t n)"
    "template<class F, std::size_t... Is>"
    "constexpr target_arch get_target_arch_from_name(const char* const, const std::size_t)\n{\n    return target_arch::unknown;\n}\n\n")
  # rocPRIM's value variant contains empty integral_constant alternatives. Its
  # canonical assignment loop reaches libstdc++'s checked std::get machinery,
  # even though selecting an empty alternative only needs to update the variant
  # discriminant. Construct the selected alternative directly so the packaged
  # host closure keeps the same dispatch semantics without exception scaffolding.
  replace_region(
    "rocprim/detail/various.hpp"
    "template<typename T, T... Vs>\nstruct constexpr_value_variant"
    "template<typename... T>\nstruct constexpr_type_variant"
    "template<typename T, T... Vs>\nstruct constexpr_value_variant\n{\n    using variant = std::variant<std::integral_constant<T, Vs>...>;\n\n    static variant create(T value)\n    {\n        static_assert(sizeof...(Vs) > 0, \"constexpr_value_variant requires an alternative\");\n        return create_impl<0, Vs...>(value);\n    }\n\nprivate:\n    template<std::size_t I, T V, T... Rest>\n    static variant create_impl(T value)\n    {\n        if(value == V)\n        {\n            return variant{std::in_place_index<I>};\n        }\n        if constexpr(sizeof...(Rest) > 0)\n        {\n            return create_impl<I + 1, Rest...>(value);\n        }\n        return variant{};\n    }\n};\n\n")
  # Each optional below is assigned either by the matching architecture arm or
  # by the immediately preceding unknown-architecture fallback. value_or keeps
  # both bad_optional_access and libstdc++'s operator* assertion machinery out
  # of package host orchestration; its fallback repeats rocPRIM's own fallback.
  copy_and_replace_from(
    "${ROCPRIM_ROOT}"
    "rocprim/device/config_types.hpp"
    "tuned_kernel\.value[(][)]"
    "tuned_kernel.value_or(trampoline_kernel<Config, target_arch::unknown, Kernel, LaunchSelector>)"
    1)
  copy_and_replace_from(
    "${ROCPRIM_ROOT}"
    "rocprim/device/device_partition.hpp"
    "vsmem_per_block\.value[(][)]"
    "vsmem_per_block.value_or(0)"
    1)
  copy_and_replace_from(
    "${ROCPRIM_ROOT}"
    "rocprim/device/device_merge.hpp"
    "vsmem_per_block\.value[(][)]"
    "vsmem_per_block.value_or(0)"
    1)
  copy_and_replace_from(
    "${ROCPRIM_ROOT}"
    "rocprim/device/device_merge_sort.hpp"
    "return [*]out"
    "return out.value_or(0)"
    1)
  # PolyAST exposes atomic scalar operations up to 64 bits. Select rocPRIM's
  # canonical split-value/flag lookback state, which it also uses on targets
  # without 128-bit atomics, instead of importing target-specific b128 asm or
  # relying on the host compilation phase's non-operative fallback bodies.
  copy_and_replace_from(
    "${ROCPRIM_ROOT}"
    "rocprim/config.hpp"
    "    #define ROCPRIM_MAX_ATOMIC_SIZE 16"
    "    #define ROCPRIM_MAX_ATOMIC_SIZE 8"
    1)
  copy_and_replace_from(
    "${ROCPRIM_ROOT}"
    "rocprim/device/config_types.hpp"
    "                           stream,\n                           device_callback"
    "                           0,\n                           device_callback"
    1)
  # The caught object is unused. Avoid importing std::exception's host-only
  # copy/destruction machinery solely for rocPRIM's environment parsing path.
  copy_and_replace_from(
    "${ROCPRIM_ROOT}"
    "rocprim/device/detail/ordered_block_id.hpp"
    "catch\\(std::exception\\)"
    "catch(const std::exception&)"
    1)
  # Host lowering must instantiate the same wave-specific trampoline types as
  # the device pass. ROCPRIM_NAVI already fixes min_size() for this package build.
  copy_and_replace_from(
    "${ROCPRIM_ROOT}"
    "rocprim/intrinsics/arch.hpp"
    "return target::dynamic"
    "return min_size() == ROCPRIM_WARP_SIZE_32 ? target::size32 : target::size64"
    1)
  # This producer targets gfx1036, so rocPRIM's early-revision gfx908 sleep
  # workaround is inapplicable. Avoid importing std::string just to rediscover
  # that fixed target fact in every lookback-based host closure.
  copy_and_replace_from(
    "${ROCPRIM_ROOT}"
    "rocprim/device/detail/lookback_scan_state.hpp"
    "use_sleep = std::string\\(prop\\.gcnArchName\\)\\.find\\(\"908\"\\) != std::string::npos && asicRevision < 2"
    "use_sleep = false"
    1)
  # Portable package execution uses rocPRIM's default hotfix policy. Its
  # process-global environment override and std::stoi exception path are not
  # part of Spectra's package contract.
  replace_region(
    "rocprim/device/detail/ordered_block_id.hpp"
    "        // Try to parse the env var, if it fails fall back to the default option."
    "        data.valid = true;"
    "        data.option = use_atomic_block_id::default_option;\n\n")
  foreach(sync_file_count
          "rocprim/common.hpp|1"
          "rocprim/device/device_reduce.hpp|1"
          "rocprim/device/device_merge_inplace.hpp|5"
          "rocprim/device/device_segmented_radix_sort.hpp|1"
          "rocprim/device/device_radix_sort.hpp|1"
          "rocprim/device/device_histogram.hpp|1"
          "rocprim/device/detail/device_nth_element.hpp|1")
    string(REPLACE "|" ";" sync_fields "${sync_file_count}")
    list(GET sync_fields 0 sync_file)
    list(GET sync_fields 1 sync_count)
    copy_and_replace_from(
      "${ROCPRIM_ROOT}"
      "${sync_file}"
      "hipStreamSynchronize\\(stream\\)"
      "hipDeviceSynchronize()"
      "${sync_count}")
  endforeach()
  # These asynchronous fills receive the same runtime-owned default stream as
  # the launches above. Make the null value explicit for the package runtime;
  # retaining the opaque host variable would falsely model user stream support.
  foreach(fill_file_count
          "rocprim/device/device_partition.hpp|1"
          "rocprim/device/device_histogram.hpp|1"
          "rocprim/device/device_radix_sort.hpp|2"
          "rocprim/device/detail/ordered_block_id.hpp|1"
          "rocprim/device/detail/device_nth_element.hpp|3"
          "rocprim/device/detail/device_search_n.hpp|2")
    string(REPLACE "|" ";" fill_fields "${fill_file_count}")
    list(GET fill_fields 0 fill_file)
    list(GET fill_fields 1 fill_count)
    copy_and_replace_from(
      "${ROCPRIM_ROOT}"
      "${fill_file}"
      "hipMemsetAsync\\(([^;]*)stream\\)"
      "hipMemsetAsync(\\1 0)"
      "${fill_count}")
  endforeach()
  # A handful of rocPRIM setup kernels bypass launch_plan and forward the same
  # default policy stream directly through triple-chevron syntax.
  foreach(launch_file_count
          "rocprim/device/device_partition.hpp|1"
          "rocprim/device/device_find_first_of.hpp|1"
          "rocprim/device/device_scan.hpp|1"
          "rocprim/device/device_reduce_by_key.hpp|1"
          "rocprim/device/device_adjacent_find.hpp|1"
          "rocprim/device/detail/device_batch_memcpy.hpp|1"
          "rocprim/device/device_merge_inplace.hpp|1"
          "rocprim/device/detail/device_search_n.hpp|3"
          "rocprim/device/detail/device_search.hpp|2"
          "rocprim/device/device_scan_by_key.hpp|2"
          "rocprim/device/device_run_length_encode.hpp|1")
    string(REPLACE "|" ";" launch_fields "${launch_file_count}")
    list(GET launch_fields 0 launch_file)
    list(GET launch_fields 1 launch_count)
    copy_and_replace_from(
      "${ROCPRIM_ROOT}"
      "${launch_file}"
      "stream>>>"
      "0>>>"
      "${launch_count}")
  endforeach()
elseif(BACKEND STREQUAL "dpl")
  # Package execution owns the selected queue. oneDPL recovers a native queue
  # from shared_ptr's custom deleter only to issue a result copy; use the
  # package-build shim's current queue instead of importing shared_ptr control-block
  # internals into the package representation.
  # oneDPL uses quoted relative includes throughout its tree, so keep one
  # physical include identity rooted in the overlay, as for CUB above.
  file(COPY "${VENDOR_ROOT}/oneapi" DESTINATION "${OUT_DIR}")
  copy_and_replace(
    "oneapi/dpl/pstl/hetero/dpcpp/parallel_backend_sycl_utils.h"
    "auto __q_proxy = std::get_deleter<__internal::__sycl_usm_free>\\(__scratch_buf\\)"
    "sycl::queue __q_proxy"
    1)
  copy_and_replace(
    "oneapi/dpl/pstl/hetero/dpcpp/parallel_backend_sycl_utils.h"
    "assert\\(__q_proxy != nullptr && __q_proxy->__q.has_value\\(\\)\\)"
    "(void)__q_proxy"
    1)
  copy_and_replace(
    "oneapi/dpl/pstl/hetero/dpcpp/parallel_backend_sycl_utils.h"
    "__q_proxy->__q->memcpy"
    "__q_proxy.memcpy"
    1)
  # Spectra's package ABI is direct/USM-only. Collapse oneDPL's runtime
  # host-USM/buffer selection to its device-USM path so unreachable allocator
  # and accessor machinery does not become part of the portable package.
  replace_region(
    "oneapi/dpl/pstl/hetero/dpcpp/parallel_backend_sycl_utils.h"
    "    __result_and_scratch_storage(sycl::queue __q, std::size_t __scratch_n)"
    "    __result_and_scratch_storage(__copyable_storage_state"
    [=[    __result_and_scratch_storage(sycl::queue __q, std::size_t __scratch_n)
        : __sycl_buf{nullptr, sycl::range{0}}, __scratch_n{__scratch_n}, __use_USM_host{false},
          __supports_USM_device{true}
    {
        const std::size_t __total_n = _NResults + __scratch_n;
        if (__total_n > 0)
        {
            auto __new_scratch_buf =
                std::shared_ptr<_T>(__internal::__sycl_usm_alloc<_T, sycl::usm::alloc::device>(__q, __total_n),
                                    __internal::__sycl_usm_free{__q});
            __scratch_buf.swap(__new_scratch_buf);
        }
    }

]=])
  # Metal has no 64-bit atomics, so the package keeps find state in oneDPL's own 32-bit FPGA atomic type
  copy_and_replace_literal(
    "oneapi/dpl/pstl/hetero/dpcpp/parallel_backend_sycl.h"
    [=[// FPGA devices don't support 64-bit atomics
#if _ONEDPL_FPGA_DEVICE
    using _AtomicType = uint32_t;
#else
    using _AtomicType = _IndexType;
#endif]=]
    [=[    using _AtomicType = uint32_t;]=]
    1)
  copy_and_replace_literal(
    "oneapi/dpl/pstl/hetero/dpcpp/parallel_backend_sycl.h"
    [=[// FPGA devices don't support 64-bit atomics
#if _ONEDPL_FPGA_DEVICE
    using _AtomicType = int32_t;
#else
    using _AtomicType = _IndexType;
#endif]=]
    [=[    using _AtomicType = int32_t;]=]
    1)
  # Device USM always backs combined accessors here, and a null-tested choice
  # between two buffers has no logical-addressing (Vulkan) form.
  copy_and_replace_literal(
    "oneapi/dpl/pstl/hetero/dpcpp/parallel_backend_sycl_utils.h"
    [=[        return __ptr ? __ptr : &__acc[0];]=]
    [=[        return __ptr;]=]
    1)
  # The direct-only storage has one owner. Keep it unique through the future
  # instead of importing libstdc++'s polymorphic shared-control machinery.
  copy_and_replace_literal(
    "oneapi/dpl/pstl/hetero/dpcpp/parallel_backend_sycl_utils.h"
    [=[    std::shared_ptr<_T> __result_buf;
    std::shared_ptr<_T> __scratch_buf;]=]
    [=[    std::unique_ptr<_T, __internal::__sycl_usm_free> __result_buf;
    std::unique_ptr<_T, __internal::__sycl_usm_free> __scratch_buf;]=]
    1)
  copy_and_replace_literal(
    "oneapi/dpl/pstl/hetero/dpcpp/parallel_backend_sycl_utils.h"
    [=[    std::shared_ptr<_T> __scratch_buf;
    std::shared_ptr<_T> __result_buf;]=]
    [=[    std::unique_ptr<_T, __internal::__sycl_usm_free> __scratch_buf;
    std::unique_ptr<_T, __internal::__sycl_usm_free> __result_buf;]=]
    1)
  copy_and_replace_literal(
    "oneapi/dpl/pstl/hetero/dpcpp/parallel_backend_sycl_utils.h"
    [=[                std::shared_ptr<_T>(__internal::__sycl_usm_alloc<_T, sycl::usm::alloc::device>(__q, __total_n),
                                    __internal::__sycl_usm_free{__q});]=]
    [=[                std::unique_ptr<_T, __internal::__sycl_usm_free>(
                    __internal::__sycl_usm_alloc<_T, sycl::usm::alloc::device>(__q, __total_n),
                    __internal::__sycl_usm_free{__q});]=]
    1)
  copy_and_replace_literal(
    "oneapi/dpl/pstl/hetero/dpcpp/parallel_backend_sycl_utils.h"
    [=[    __future(_Event __e, _Args... __args) : std::tuple<_Args...>(__args...), __my_event(__e) {}]=]
    [=[    __future(_Event __e, _Args... __args) : std::tuple<_Args...>(std::move(__args)...), __my_event(__e) {}]=]
    1)
  # oneDPL's large merge submitter copies the complete scratch owner into its
  # host-side submission lambdas. Only the accessor derived inside each lambda
  # enters a device kernel, so capture the owner by reference as merge-sort does.
  copy_and_replace_literal(
    "oneapi/dpl/pstl/hetero/dpcpp/parallel_backend_sycl_merge.h"
    [=[                           __base_diagonals_sp_global_storage, __n1, __n2,]=]
    [=[                           &__base_diagonals_sp_global_storage, __n1, __n2,]=]
    1)
  copy_and_replace_literal(
    "oneapi/dpl/pstl/hetero/dpcpp/parallel_backend_sycl_merge.h"
    [=[                           __base_diagonals_sp_global_storage, __n1, __n2](sycl::handler& __cgh) {]=]
    [=[                           &__base_diagonals_sp_global_storage, __n1, __n2](sycl::handler& __cgh) {]=]
    1)
  # Spectra's range lengths are signed 32-bit values, so even a two-range
  # merge fits oneDPL's uint32 index path. Retain both its small and large
  # algorithms, but give their scratch storage one concrete value owner. This
  # removes the type-erased heap/shared_ptr lifetime used only to unify the
  # library's uint32/uint64 return types.
  copy_and_replace_literal(
    "oneapi/dpl/pstl/hetero/dpcpp/parallel_backend_sycl_merge.h"
    [=[    __future<sycl::event, std::shared_ptr<__result_and_scratch_storage_base>>]=]
    [=[    auto]=]
    2)
  copy_and_replace_literal(
    "oneapi/dpl/pstl/hetero/dpcpp/parallel_backend_sycl_merge.h"
    [=[__future<sycl::event, std::shared_ptr<__result_and_scratch_storage_base>>]=]
    [=[auto]=]
    2)
  copy_and_replace_literal(
    "oneapi/dpl/pstl/hetero/dpcpp/parallel_backend_sycl_merge.h"
    [=[        __result_and_scratch_storage_t* __p_res_storage = nullptr;

        if constexpr (_OutSizeLimit{})
            __p_res_storage = new __result_and_scratch_storage_t(__q, 0);
        else
            assert(oneapi::dpl::__ranges::__size(__rng3) >= __n1 + __n2);

        std::shared_ptr<__result_and_scratch_storage_base> __p_result_and_scratch_storage_base(
            static_cast<__result_and_scratch_storage_base*>(__p_res_storage));]=]
    [=[        __result_and_scratch_storage_t __res_storage(__q, 0);
        __result_and_scratch_storage_t* __p_res_storage = nullptr;

        if constexpr (_OutSizeLimit{})
            __p_res_storage = &__res_storage;
        else
            assert(oneapi::dpl::__ranges::__size(__rng3) >= __n1 + __n2);]=]
    1)
  copy_and_replace_literal(
    "oneapi/dpl/pstl/hetero/dpcpp/parallel_backend_sycl_merge.h"
    [=[        // Save the raw pointer into a shared_ptr to return it in __future and extend the lifetime of the storage.
        // We should return the same thing in the second param of __future for compatibility
        // with the returning value in __parallel_merge_submitter_large::operator()
        return __future{std::move(__event), std::move(__p_result_and_scratch_storage_base)};]=]
    [=[        return __future{std::move(__event), std::move(__res_storage)};]=]
    1)
  copy_and_replace_literal(
    "oneapi/dpl/pstl/hetero/dpcpp/parallel_backend_sycl_merge.h"
    [=[        auto __p_base_diagonals_sp_global_storage =
            new __result_and_scratch_storage_t(__q, __nd_range_params.base_diag_count + 1);

        // Save the raw pointer into a shared_ptr to return it in __future and extend the lifetime of the storage.
        std::shared_ptr<__result_and_scratch_storage_base> __p_result_and_scratch_storage_base(
            static_cast<__result_and_scratch_storage_base*>(__p_base_diagonals_sp_global_storage));]=]
    [=[        __result_and_scratch_storage_t __base_diagonals_sp_global_storage(__q, __nd_range_params.base_diag_count + 1);]=]
    1)
  copy_and_replace_literal(
    "oneapi/dpl/pstl/hetero/dpcpp/parallel_backend_sycl_merge.h"
    [=[                                                           __nd_range_params,
                                                           *__p_base_diagonals_sp_global_storage);]=]
    [=[                                                           __nd_range_params, __base_diagonals_sp_global_storage);]=]
    1)
  copy_and_replace_literal(
    "oneapi/dpl/pstl/hetero/dpcpp/parallel_backend_sycl_merge.h"
    [=[                                     *__p_base_diagonals_sp_global_storage);

        return __future{std::move(__event), std::move(__p_result_and_scratch_storage_base)};]=]
    [=[                                     __base_diagonals_sp_global_storage);

        return __future{std::move(__event), std::move(__base_diagonals_sp_global_storage)};]=]
    1)
  copy_and_replace_literal(
    "oneapi/dpl/pstl/hetero/dpcpp/parallel_backend_sycl_merge.h"
    [=[                              oneapi::dpl::__ranges::__size(__rng3));
    if (__n < __get_starting_size_limit_for_large_submitter<__value_type>())
    {
        using _WiIndex = std::uint32_t;]=]
    [=[                              oneapi::dpl::__ranges::__size(__rng3));
    using _WiIndex = std::uint32_t;
    if (__n < __get_starting_size_limit_for_large_submitter<__value_type>())
    {]=]
    1)
  replace_region(
    "oneapi/dpl/pstl/hetero/dpcpp/parallel_backend_sycl_merge.h"
    [=[    else
    {
        if (__n <= std::numeric_limits<std::uint32_t>::max())]=]
    [=[    }
}

template <typename _OutSizeLimit = std::false_type, typename _ExecutionPolicy]=]
    [=[    else
    {
        using _DiagonalsKernelName = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
            __diagonals_kernel_name<_CustomName, _WiIndex>>;
        using _MergeKernelName = oneapi::dpl::__par_backend_hetero::__internal::__kernel_name_provider<
            __merge_kernel_name_large<_CustomName, _WiIndex>>;
        return __parallel_merge_submitter_large<_OutSizeLimit, _WiIndex, _CustomName, _DiagonalsKernelName,
                                                _MergeKernelName>()(
            __q, std::forward<_Range1>(__rng1), std::forward<_Range2>(__rng2), std::forward<_Range3>(__rng3), __comp,
            __proj1, __proj2);
]=])
  # Merge-sort has the same type-erasure solely to unify uint32/uint64 scratch
  # owners. Spectra's int32 length selects uint32 unconditionally, so carry the
  # concrete scratch value through the tuple/future and retain both of oneDPL's
  # normal and large merge stages.
  copy_and_replace_literal(
    "oneapi/dpl/pstl/hetero/dpcpp/parallel_backend_sycl_merge_sort.h"
    [=[    std::tuple<sycl::event, bool, std::shared_ptr<__result_and_scratch_storage_base>>]=]
    [=[    auto]=]
    1)
  copy_and_replace_literal(
    "oneapi/dpl/pstl/hetero/dpcpp/parallel_backend_sycl_merge_sort.h"
    [=[__future<sycl::event, std::shared_ptr<__result_and_scratch_storage_base>>]=]
    [=[auto]=]
    3)
  copy_and_replace_literal(
    "oneapi/dpl/pstl/hetero/dpcpp/parallel_backend_sycl_merge_sort.h"
    [=[        // Storage to save split-points on each base diagonal + 1 (for the right base diagonal in the last work-group)
        __base_diagonals_sp_storage_t* __p_base_diagonals_sp_global_storage = nullptr;

        // shared_ptr instance to return it in __future and extend the lifetime of the storage.
        std::shared_ptr<__result_and_scratch_storage_base> __p_result_and_scratch_storage_base;

        // Max amount of base diagonals
        const std::size_t __max_base_diags_count =
            get_max_base_diags_count(__q, __nd_range_params.chunk, __n) + __1_final_base_diag;]=]
    [=[        // Max amount of base diagonals
        const std::size_t __max_base_diags_count =
            get_max_base_diags_count(__q, __nd_range_params.chunk, __n) + __1_final_base_diag;
        __base_diagonals_sp_storage_t __base_diagonals_sp_global_storage(__q, __max_base_diags_count);]=]
    1)
  copy_and_replace_literal(
    "oneapi/dpl/pstl/hetero/dpcpp/parallel_backend_sycl_merge_sort.h"
    [=[                if (nullptr == __p_base_diagonals_sp_global_storage)
                {
                    // Create storage to save split-points on each base diagonal + 1 (for the right base diagonal in the last work-group)
                    __p_base_diagonals_sp_global_storage =
                        new __base_diagonals_sp_storage_t(__q, __max_base_diags_count);

                    // Save the raw pointer into a shared_ptr to return it in __future and extend the lifetime of the storage.
                    __p_result_and_scratch_storage_base.reset(
                        static_cast<__result_and_scratch_storage_base*>(__p_base_diagonals_sp_global_storage));
                }

]=]
    ""
    1)
  copy_and_replace_literal(
    "oneapi/dpl/pstl/hetero/dpcpp/parallel_backend_sycl_merge_sort.h"
    [=[*__p_base_diagonals_sp_global_storage]=]
    [=[__base_diagonals_sp_global_storage]=]
    2)
  copy_and_replace_literal(
    "oneapi/dpl/pstl/hetero/dpcpp/parallel_backend_sycl_merge_sort.h"
    [=[        return {std::move(__event_chain), __data_in_temp, std::move(__p_result_and_scratch_storage_base)};]=]
    [=[        return std::tuple{std::move(__event_chain), __data_in_temp, std::move(__base_diagonals_sp_global_storage)};]=]
    1)
  copy_and_replace_literal(
    "oneapi/dpl/pstl/hetero/dpcpp/parallel_backend_sycl_merge_sort.h"
    [=[    if (oneapi::dpl::__ranges::__size(__rng) <= std::numeric_limits<std::uint32_t>::max())
    {
        return __submit_selecting_leaf<_CustomName, std::uint32_t>(__q_local, std::forward<_Range>(__rng), __comp);
    }
    else
    {
        return __submit_selecting_leaf<_CustomName, std::uint64_t>(__q_local, std::forward<_Range>(__rng), __comp);
    }]=]
    [=[    return __submit_selecting_leaf<_CustomName, std::uint32_t>(__q_local, std::forward<_Range>(__rng), __comp);]=]
    1)
  copy_and_replace_literal(
    "oneapi/dpl/pstl/hetero/dpcpp/parallel_backend_sycl.h"
    [=[__future<sycl::event, std::shared_ptr<__result_and_scratch_storage_base>>
__parallel_stable_sort]=]
    [=[auto
__parallel_stable_sort]=]
    1)
  # Source backends expose a portable 32-lane subgroup model. oneDPL's CPU
  # merge-sort tuning queries host SIMD widths through std::vector and selects
  # a geometry that does not describe that package execution model. Keep CPU
  # device typing for scan/compaction, but use merge-sort's portable GPU
  # geometry as the ext-libs prototype validated across POCL and Rusticl.
  copy_and_replace_literal(
    "oneapi/dpl/pstl/hetero/dpcpp/parallel_backend_sycl_merge_sort.h"
    [=[        const bool __is_cpu = __q.get_device().is_cpu();]=]
    [=[        const bool __is_cpu = false;]=]
    1)
  copy_and_replace_literal(
    "oneapi/dpl/pstl/hetero/dpcpp/parallel_backend_sycl_merge_sort.h"
    [=[    const bool __is_cpu = __device.is_cpu();]=]
    [=[    const bool __is_cpu = false;]=]
    1)
  # The importer operates on the instantiated AST before ordinary dead-code
  # elimination, so remove the now-unreachable vector-valued subgroup query
  # rather than relying on `if (false)` to disappear later.
  copy_and_replace_literal(
    "oneapi/dpl/pstl/hetero/dpcpp/parallel_backend_sycl_merge_sort.h"
    [=[    const bool __is_cpu = false;
    std::uint32_t __max_sg_size{};
    if (__is_cpu)
    {
        const auto __sg_sizes = __device.template get_info<sycl::info::device::sub_group_sizes>();
        __max_sg_size = *std::max_element(__sg_sizes.begin(), __sg_sizes.end());
    }
    // Assume CPUs handle one sub-group (SIMD) per CU;
    // Assume GPUs handle multiple sub-groups per CU,
    // while the maximum work-group size takes hardware multithreading (occupancy) into account
    const std::size_t __max_hw_wg_size = __is_cpu ? __max_sg_size : __max_wg_size;]=]
    [=[    const std::size_t __max_hw_wg_size = __max_wg_size;]=]
    1)
  copy_and_replace_literal(
    "oneapi/dpl/pstl/hetero/dpcpp/unseq_backend_sycl.h"
    [=[            if (__adjusted_global_id == __n - 1)
                __wg_assigner(__wg_sums_ptr, __group_id, __local_acc, __local_id);
        }
]=]
    [=[            if (__adjusted_global_id == __n - 1)
                __wg_assigner(__wg_sums_ptr, __group_id, __local_acc, __local_id);
            __dpl_sycl::__group_barrier(__item);
        }
]=]
    2)
  # Express the one-dimensional work-group extent through SYCL's scalar
  # accessor. It is identical to range::size() here and maps directly to the
  # portable block-size intrinsic without materialising an opaque range.
  copy_and_replace(
    "oneapi/dpl/pstl/hetero/dpcpp/unseq_backend_sycl.h"
    "__item\\.get_local_range\\(\\)\\.size\\(\\)"
    "__item.get_local_range(0)"
    1)
  copy_and_replace(
    "oneapi/dpl/pstl/hetero/dpcpp/parallel_backend_sycl_reduce.h"
    "__item\\.get_local_range\\(\\)\\.size\\(\\)"
    "__item.get_local_range(0)"
    2)
  copy_and_replace(
    "oneapi/dpl/pstl/hetero/dpcpp/parallel_backend_sycl_reduce_then_scan.h"
    "__sub_group\\.get_max_local_range\\(\\)\\[0\\]"
    "__sub_group.get_local_linear_range()"
    3)
  # Package generation cannot ask the eventual import device which
  # optional fixed subgroup widths it advertises. Select oneDPL's portable
  # non-required-subgroup path; subgroup collectives themselves remain
  # available where an algorithm does not require a fixed width.
  replace_region(
    "oneapi/dpl/pstl/hetero/dpcpp/parallel_backend_sycl_utils.h"
    "inline bool\n__supports_sub_group_size"
    "//-----------------------------------------------------------------------------\n// Kernel run-time information helpers"
    [=[inline bool
__supports_sub_group_size(const sycl::queue&, std::size_t)
{
    return false;
}

]=])
  # Spectra packages require direct/USM memory. The remaining buffer-backed
  # host-access arm is therefore unreachable, but compiling it into the package would import a
  # temporary whose destructor performs a host/device transfer.
  replace_region(
    "oneapi/dpl/pstl/hetero/dpcpp/parallel_backend_sycl_utils.h"
    "        else\n        {\n            return __sycl_buf.get_host_access"
    "  private:\n    bool\n    is_USM() const"
    "        else\n        {\n            return *(__scratch_buf.get() + __scratch_n + _Idx);\n        }\n    }\n\n")
else()
  message(FATAL_ERROR "unsupported Spectra package overlay backend: ${BACKEND}")
endif()
