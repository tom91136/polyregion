

# Build an optimized toolchain for an example set of targets.
set(CMAKE_BUILD_TYPE Release CACHE STRING "")
set(LLVM_TARGETS_TO_BUILD
        AArch64
        ARM
        X86
        ARM
        NVPTX
        AMDGPU
        CACHE STRING "")

# Enable the LLVM projects and runtimes.
set(LLVM_ENABLE_PROJECTS
        clang
        lld
        CACHE STRING "")
set(LLVM_ENABLE_RUNTIMES
        compiler-rt
        libcxx
        libcxxabi
        CACHE STRING "")

set(USE_STATIC_CXX_STDLIB ON)

# We'll build two distributions: Toolchain, which just holds the tools
# (intended for most end users), and Development, which has libraries (for end
# users who wish to develop their own tooling using those libraries). This will
# produce the install-toolchain-distribution and install-development-distribution
# targets to install the distributions.
set(LLVM_DISTRIBUTIONS
        Toolchain
        CACHE STRING "")

# We want to include the C++ headers in our distribution.
set(LLVM_RUNTIME_DISTRIBUTION_COMPONENTS
        cxx-headers
        CACHE STRING "")

# You likely want more tools; this is just an example :) Note that we need to
# include cxx-headers explicitly here (in addition to it being added to
# LLVM_RUNTIME_DISTRIBUTION_COMPONENTS above).
set(LLVM_Toolchain_DISTRIBUTION_COMPONENTS
        builtins
        clang
        clang-resource-headers
        cxx-headers
        lld
        llvm-objdump

        CACHE STRING "")

