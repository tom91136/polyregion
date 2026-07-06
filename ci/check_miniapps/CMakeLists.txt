cmake_minimum_required(VERSION 3.20)
project(check_miniapps CXX)
enable_testing()
find_package(Git REQUIRED)
find_program(SYSTEM_CXX NAMES c++ g++ clang++)
find_program(BASH_PROGRAM bash REQUIRED)

if (NOT EXISTS "${POLYCPP_DIST}/bin/polycpp")
  message(FATAL_ERROR "pass -DPOLYCPP_DIST=<dist dir with bin/polycpp>, "
                      "e.g. native/out/polyregion-Release-<arch>-dylib-dist")
endif ()

# the apps fetch their own TBB; inject the resolved system package (their ccache compiler breaks the find)
find_package(TBB CONFIG QUIET)
set(TBB_SHIM "${CMAKE_CURRENT_BINARY_DIR}/tbb-shim.cmake")
file(WRITE "${TBB_SHIM}" "set(TBB_DIR \"${TBB_DIR}\")\nfind_package(TBB CONFIG REQUIRED)\n")
set(CLOVER_DECK "${CMAKE_CURRENT_BINARY_DIR}/clover_512.in")
file(WRITE "${CLOVER_DECK}"
  "*clover\n"
  " state 1 density=0.2 energy=1.0\n"
  " state 2 density=1.0 energy=2.5 geometry=rectangle xmin=0.0 xmax=5.0 ymin=0.0 ymax=2.0\n"
  " x_cells=384\n y_cells=384\n xmin=0.0\n ymin=0.0\n xmax=10.0\n ymax=10.0\n"
  " initial_timestep=0.04\n max_timestep=0.04\n end_step=40\n test_problem 1\n"
  "*endclover\n")
set(GOLD_CloverLeaf     "384 384 40 1.340475094048098e+00")
set(PROBLEMS_CloverLeaf clover.problems)
set(TEA_DECK "${CMAKE_CURRENT_BINARY_DIR}/tea_512.in")    # test_problem 2 makes it emit a deterministic qa value
file(WRITE "${TEA_DECK}"
  "*tea\n"
  "state 1 density=100.0 energy=0.0001\n"
  "state 2 density=0.1 energy=25.0 geometry=rectangle xmin=0.0 xmax=1.0 ymin=1.0 ymax=2.0\n"
  "state 3 density=0.1 energy=0.1 geometry=rectangle xmin=1.0 xmax=6.0 ymin=1.0 ymax=2.0\n"
  "state 4 density=0.1 energy=0.1 geometry=rectangle xmin=5.0 xmax=6.0 ymin=1.0 ymax=8.0\n"
  "state 5 density=0.1 energy=0.1 geometry=rectangle xmin=5.0 xmax=10.0 ymin=7.0 ymax=8.0\n"
  "x_cells=384\n y_cells=384\n xmin=0.0\n ymin=0.0\n xmax=10.0\n ymax=10.0\n"
  "initial_timestep=0.004\n end_step=5\n max_iters=10000\n use_cg\n eps 1.0e-15\n test_problem 2\n use_c_kernels\n"
  "*endtea\n")
set(GOLD_TeaLeaf     "384 384 5 9.884261537678871e+01")
set(PROBLEMS_TeaLeaf tea.problems)

# offload backends run on the emulator bundle; absent on macos/windows, those fall back to cpu-only.
# pass -DMINIAPP_EMULATORS=<bundle>/out, or it's taken from POLYREGION_EMULATORS_HOME.
if (NOT DEFINED MINIAPP_EMULATORS AND DEFINED ENV{POLYREGION_EMULATORS_HOME})
  set(MINIAPP_EMULATORS "$ENV{POLYREGION_EMULATORS_HOME}")
endif ()
set(HAVE_EMULATORS FALSE)
if (MINIAPP_EMULATORS AND EXISTS "${MINIAPP_EMULATORS}/env.sh" AND CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux")
  set(HAVE_EMULATORS TRUE)
endif ()

# cpu-tbb is the correctness reference (system compiler + oneTBB); emulators are added when the bundle is present
if (NOT DEFINED MINIAPP_BACKENDS)
  if (HAVE_EMULATORS)
    set(MINIAPP_BACKENDS cpu-tbb vk-lavapipe cl-llvmpipe cl-pocl cpu)
  else ()
    set(MINIAPP_BACKENDS cpu-tbb cpu)
  endif ()
endif ()

# polyregion backend -> -fstdpar-arch + the polyrt device selector. the bundle env (libs, ICDs, GALLIUM_DRIVER,
# ...) is sourced from the bundle's env.sh by run.sh, so it lives only in emulators/collect.sh.
set(ARCH_vk-lavapipe spirv_glcompute@)
set(ARCH_cl-llvmpipe opencl1_1@)
set(ARCH_cl-pocl     opencl1_1@)
set(ARCH_cpu         host@native)
set(RUN_WRAPPER "${CMAKE_CURRENT_SOURCE_DIR}/run.sh")
set(ENV_vk-lavapipe "POLYRT_PLATFORM=spirv_glcompute@!spirv_glcompute@*llvmpipe*;POLYRT_HOST_FALLBACK=0")
set(ENV_cl-llvmpipe "POLYRT_PLATFORM=opencl1_1@!opencl1_1@*llvmpipe*source*;POLYRT_HOST_FALLBACK=0")
set(ENV_cl-pocl     "POLYRT_PLATFORM=opencl1_1@!opencl1_1@*Portable*source*;POLYRT_HOST_FALLBACK=0")
set(ENV_cpu         "POLYRT_PLATFORM=host")
set(ENV_cpu-tbb     "")                                    # the reference runs natively, no polyrt
# bundle for run.sh to source per backend (`-` = none); only the emulator backends need it
set(BUNDLE_vk-lavapipe "${MINIAPP_EMULATORS}")
set(BUNDLE_cl-llvmpipe "${MINIAPP_EMULATORS}")
set(BUNDLE_cl-pocl     "${MINIAPP_EMULATORS}")
set(BUNDLE_cpu         -)
set(BUNDLE_cpu-tbb     -)

# app -> repo, pinned commit, stdpar model + its binary, the validating run, and the PASS_/FAIL_ banner its
# built-in self-validation prints (clover/tea validate against the GOLD_ injected beside their decks above)
set(REPO_miniBUDE     https://github.com/UoB-HPC/miniBUDE.git)
set(SHA_miniBUDE      570f66c1fd0c29b7f3ab6fa96fba87f4561aa44c)
set(MODEL_miniBUDE    std-indices)
set(EXE_miniBUDE      std-indices-bude)
set(RUN_miniBUDE      --deck data/bm1 -n 8192 -i 8)        # bm1 (900x27), 8192 poses, 8 iterations
set(PASS_miniBUDE     "valid: true")
set(REPO_CloverLeaf   https://github.com/UoB-HPC/CloverLeaf.git)
set(SHA_CloverLeaf    3e10ff9268d3704a744e5a2a4ea3c62831078082)
set(MODEL_CloverLeaf  std-indices)
set(EXE_CloverLeaf    std-indices-cloverleaf)
set(RUN_CloverLeaf    --file ${CLOVER_DECK})               # 384x384, 40 steps
set(PASS_CloverLeaf   "is considered PASSED")              # CloverLeaf's built-in QA banner
set(PATCH_CloverLeaf  cloverleaf-gold-from-file.patch)     # report.cpp reads its qa gold from clover.problems
set(REPO_TeaLeaf      https://github.com/UoB-HPC/TeaLeaf)
set(SHA_TeaLeaf       e70261c0be40537da75b258108ed2898f84f3c58)
set(MODEL_TeaLeaf     std-indices)
set(EXE_TeaLeaf       std-indices-tealeaf)
set(RUN_TeaLeaf       --file ${TEA_DECK})                  # 384x384, 5 steps
set(PASS_TeaLeaf      "This run PASSED")                   # TeaLeaf's built-in QA banner
set(REPO_BabelStream  https://github.com/UoB-HPC/BabelStream.git)
set(SHA_BabelStream   17ab377b0e919e14fd3df2b67268761fdac8abb3)
set(MODEL_BabelStream std)                                 # std (not std-indices); binary std-stream
set(EXE_BabelStream   std-stream)
set(RUN_BabelStream   -s 134217728 -n 25)                  # ~1GB arrays (2^27 doubles), 25 iterations
# XXX emulators cap device allocations at ~2GB (lavapipe), so they use 256MB arrays
set(RUN_BabelStream_vk-lavapipe -s 33554432 -n 25)
set(RUN_BabelStream_cl-llvmpipe -s 33554432 -n 25)
set(FAIL_BabelStream  "FAILED validation")                 # no PASS banner; fails loudly on bad results
set(EXTRA_BabelStream "set(STDIMPL INDICES CACHE STRING \"\")")   # the std model needs an explicit impl strategy

# Cross: set specific on target arch, not host
if (NOT MINIAPP_TARGET_ARCH)
  set(MINIAPP_TARGET_ARCH "$ENV{POLYREGION_ARCH}")
endif ()
if (NOT MINIAPP_TARGET_ARCH)
  set(MINIAPP_TARGET_ARCH "${CMAKE_HOST_SYSTEM_PROCESSOR}")
endif ()
# Cross: skip each app's -march=native
if (NOT MINIAPP_TARGET_ARCH STREQUAL CMAKE_HOST_SYSTEM_PROCESSOR)
  string(APPEND EXTRA_BabelStream "\nset(BUILD_NATIVE OFF CACHE BOOL \"\")")
  set(EXTRA_TeaLeaf "set(RELEASE_CXX_FLAGS \"-O3;-Wall\" CACHE STRING \"\")")
endif ()
if (MINIAPP_TARGET_ARCH STREQUAL riscv64 OR MINIAPP_TARGET_ARCH STREQUAL arm)
  set(RUN_BabelStream -s 16777216 -n 25) # aarch32 has 3GB user VA, so 128MB arrays
endif ()

set(APPS "${CMAKE_CURRENT_SOURCE_DIR}/apps")
foreach (app miniBUDE CloverLeaf TeaLeaf BabelStream)
  set(src "${APPS}/${app}")
  if (NOT EXISTS "${src}/.git")                            # fetch the pinned commit once (shallow), shared across backends
    message(STATUS "check_miniapps: fetching ${app} @ ${SHA_${app}}")
    execute_process(COMMAND ${GIT_EXECUTABLE} init -q "${src}" COMMAND_ERROR_IS_FATAL ANY)
    execute_process(COMMAND ${GIT_EXECUTABLE} -C "${src}" fetch -q --depth 1 ${REPO_${app}} ${SHA_${app}} COMMAND_ERROR_IS_FATAL ANY)
    execute_process(COMMAND ${GIT_EXECUTABLE} -C "${src}" -c advice.detachedHead=false checkout -q FETCH_HEAD COMMAND_ERROR_IS_FATAL ANY)
    if (DEFINED PATCH_${app})                              # local source tweak (read the qa gold from a file)
      execute_process(COMMAND ${GIT_EXECUTABLE} -C "${src}" apply "${CMAKE_CURRENT_SOURCE_DIR}/${PATCH_${app}}" COMMAND_ERROR_IS_FATAL ANY)
    endif ()
  endif ()
  if (DEFINED GOLD_${app})                                 # inject the deck's reference value for the app's built-in QA
    file(WRITE "${src}/${PROBLEMS_${app}}" "${GOLD_${app}}\n")
  endif ()
  foreach (backend IN LISTS MINIAPP_BACKENDS)
    set(bdir "${src}/build-${backend}")                    # one build dir per backend
    set(init "${bdir}/polyregion-init.cmake")
    if (backend STREQUAL cpu-tbb)                           # reference: system compiler + system oneTBB
      file(WRITE "${init}"
        "set(MODEL ${MODEL_${app}} CACHE STRING \"\")\n"
        "set(CMAKE_CXX_COMPILER \"${SYSTEM_CXX}\" CACHE STRING \"\")\n"
        "set(USE_TBB ON CACHE BOOL \"\")\n"
        "set(CMAKE_PROJECT_TOP_LEVEL_INCLUDES \"${TBB_SHIM}\" CACHE STRING \"\")\n")
    else ()                                                # polycpp offload backend
      set(cxxExtra "-fstdpar;-fstdpar-arch=${ARCH_${backend}};-fstdpar-mem=reflect;-fstdpar-rt=dynamic")
      file(WRITE "${init}"
        "set(MODEL ${MODEL_${app}} CACHE STRING \"\")\n"
        "set(CMAKE_CXX_COMPILER \"${POLYCPP_DIST}/bin/polycpp\" CACHE STRING \"\")\n"
        "set(CXX_EXTRA_FLAGS \"${cxxExtra}\" CACHE STRING \"\")\n")
    endif ()
    if (DEFINED EXTRA_${app})                              # per-app extra cache vars (e.g. BabelStream STDIMPL)
      file(APPEND "${init}" "${EXTRA_${app}}\n")
    endif ()
    # libstdc++15's pstl-through-TBB rejects iota_view iterators (int128 distance -> deleted __advance);
    # DATA17 uses plain pointer ranges. polycpp goes through clang and is fine.
    if (backend STREQUAL cpu-tbb AND app STREQUAL BabelStream AND MINIAPP_TARGET_ARCH STREQUAL riscv64)
      file(APPEND "${init}" "set(STDIMPL DATA17 CACHE STRING \"\" FORCE)\n")
    endif ()
    execute_process(COMMAND ${CMAKE_COMMAND} -S "${src}" -B "${bdir}" -C "${init}")
    add_test(NAME ${app}-${backend}-build COMMAND ${CMAKE_COMMAND} --build "${bdir}" --parallel)

    set(run ${RUN_${app}})                                 # per-backend run-arg override (e.g. smaller emulator size)
    if (DEFINED RUN_${app}_${backend})
      set(run ${RUN_${app}_${backend}})
    endif ()
    add_test(NAME ${app}-${backend}-run COMMAND ${BASH_PROGRAM} ${RUN_WRAPPER} "${BUNDLE_${backend}}"
             "${bdir}/${EXE_${app}}" ${run})

    set_tests_properties(${app}-${backend}-run PROPERTIES
      DEPENDS "${app}-${backend}-build"
      WORKING_DIRECTORY "${src}"
      ENVIRONMENT "${ENV_${backend}}"
      SKIP_RETURN_CODE 77
      RESOURCE_LOCK run-cpu # runs are CPU-bound (host threads / CPU software rasterisers)
      TIMEOUT 1200)
    if (DEFINED PASS_${app})
      set_tests_properties(${app}-${backend}-run PROPERTIES PASS_REGULAR_EXPRESSION "${PASS_${app}}")
    elseif (DEFINED FAIL_${app})
      set_tests_properties(${app}-${backend}-run PROPERTIES FAIL_REGULAR_EXPRESSION "${FAIL_${app}}")
    endif ()
  endforeach ()
endforeach ()
