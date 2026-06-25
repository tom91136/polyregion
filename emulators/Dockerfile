ARG LLVM_REF=llvmorg-21.1.8
ARG SPIRV_TRANSLATOR_REF=llvm_release_210
ARG SPIRV_TOOLS_REF=vulkan-sdk-1.4.350.0
ARG BOOST_VERSION=1.66.0
ARG MESA_REF=mesa-26.1.1
ARG GLSLANG_REF=vulkan-sdk-1.4.350.0
ARG VULKAN_REF=v1.3.280
ARG OPENCL_HEADERS_REF=v2026.05.29
ARG OCL_ICD_REF=v2.3.5
ARG POCL_REF=v7.1
ARG GPUOCELOT_COMMIT=b16039dc
ARG SWIFTSHADER_COMMIT=2843cbcc714fe111e1083127c048a18002bc10ed

FROM almalinux:8 AS base
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN dnf -y install epel-release dnf-plugins-core \
 && dnf config-manager --set-enabled powertools \
 && dnf -y install gcc-toolset-13 git make cmake ninja-build pkgconfig \
      python3.11 python3.11-pip flex bison wget xz tar which file patch \
      zlib-devel expat-devel libzstd-devel libffi-devel ncurses-devel \
      libxml2-devel libdrm-devel elfutils-libelf-devel \
      glew-devel hwloc-devel libtool-ltdl-devel \
 && dnf clean all
RUN pip3.11 install --no-cache-dir 'meson>=1.5' mako packaging pyyaml

ENV PATH=/opt/rh/gcc-toolset-13/root/usr/bin:/opt/llvm/bin:/usr/local/bin:/usr/bin:/bin
ENV LD_LIBRARY_PATH=/opt/rh/gcc-toolset-13/root/usr/lib64:/opt/llvm/lib
ENV CC=gcc CXX=g++

ARG NJOBS=
ARG LLVM_LINK_JOBS=8
ARG LLVM_REF
RUN git clone --depth 1 -b ${LLVM_REF} https://github.com/llvm/llvm-project /src/llvm \
 && cmake -S /src/llvm/llvm -B /src/llvm/build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DLLVM_ENABLE_PROJECTS=clang \
      -DLLVM_TARGETS_TO_BUILD="X86;AArch64;NVPTX" \
      -DLLVM_BUILD_LLVM_DYLIB=ON -DLLVM_LINK_LLVM_DYLIB=ON \
      -DLLVM_ENABLE_RTTI=OFF -DLLVM_ENABLE_TERMINFO=OFF -DLLVM_ENABLE_ZSTD=OFF \
      -DLLVM_ENABLE_DUMP=ON \
      -DLLVM_INCLUDE_TESTS=OFF -DLLVM_INCLUDE_BENCHMARKS=OFF \
      -DLLVM_INCLUDE_EXAMPLES=OFF -DLLVM_ENABLE_BINDINGS=OFF \
      -DLLVM_PARALLEL_LINK_JOBS=${LLVM_LINK_JOBS} -DCMAKE_INSTALL_PREFIX=/opt/llvm \
 && ninja ${NJOBS:+-j$NJOBS} -C /src/llvm/build install \
 && rm -rf /src/llvm/build
ENV LLVM_DIR=/opt/llvm/lib/cmake/llvm

ARG SPIRV_TRANSLATOR_REF
# XXX drop the static lib as rusticl and pocl must link the same DSO or the translator's cl::opts double-registers
RUN git clone -b ${SPIRV_TRANSLATOR_REF} --depth 1 https://github.com/KhronosGroup/SPIRV-LLVM-Translator /src/spirv-xlat \
 && cmake -S /src/spirv-xlat -B /src/spirv-xlat/build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DLLVM_DIR=${LLVM_DIR} -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=/opt/llvm \
 && ninja -C /src/spirv-xlat/build install \
 && rm -f /opt/llvm/lib/libLLVMSPIRVLib.a \
 && rm -rf /src/spirv-xlat

RUN cmake -S /src/llvm/libclc -B /src/libclc-build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DLLVM_CONFIG=/opt/llvm/bin/llvm-config \
      -DLLVM_SPIRV=/opt/llvm/bin/llvm-spirv \
      -DLIBCLC_TARGETS_TO_BUILD="spirv-mesa3d-;spirv64-mesa3d-" \
      -DCMAKE_INSTALL_PREFIX=/opt/llvm \
 && ninja -C /src/libclc-build install \
 && rm -rf /src/llvm /src/libclc-build

ARG SPIRV_TOOLS_REF
RUN git clone -b ${SPIRV_TOOLS_REF} --depth 1 https://github.com/KhronosGroup/SPIRV-Tools /src/spirv-tools \
 && python3.11 /src/spirv-tools/utils/git-sync-deps \
 && cmake -S /src/spirv-tools -B /src/spirv-tools/build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_SHARED_LIBS=ON -DSPIRV_SKIP_TESTS=ON \
      -DCMAKE_INSTALL_LIBDIR=lib -DCMAKE_INSTALL_PREFIX=/opt/spirv \
 && ninja -C /src/spirv-tools/build install \
 && rm -rf /src/spirv-tools
ENV PKG_CONFIG_PATH=/opt/spirv/lib/pkgconfig:/opt/llvm/share/pkgconfig:/opt/llvm/lib/pkgconfig

ARG BOOST_VERSION
# XXX EL8's boost-static is non-PIC and cannot go into a DSO
RUN BV_U=${BOOST_VERSION//./_} \
 && wget -q -O /tmp/boost.tar.gz https://archives.boost.io/release/${BOOST_VERSION}/source/boost_${BV_U}.tar.gz \
 && tar -C /tmp -xf /tmp/boost.tar.gz && cd /tmp/boost_${BV_U} \
 && ./bootstrap.sh --prefix=/opt/boost --with-libraries=thread,filesystem,system,chrono,atomic,date_time \
 && ./b2 -j"$(nproc)" cxxflags=-fPIC cflags=-fPIC link=static runtime-link=shared threading=multi variant=release install \
 && rm -rf /tmp/boost*

# Mesa = lavapipe (vk) + llvmpipe + rusticl
FROM base AS mesa
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
      | sh -s -- -y --default-toolchain stable --profile minimal \
 && source "$HOME/.cargo/env" && rustup component add rustfmt \
 && cargo install --locked bindgen-cli
ENV PATH=/root/.cargo/bin:$PATH
ENV LIBCLANG_PATH=/opt/llvm/lib
ARG GLSLANG_REF
# XXX lavapipe needs glslangValidator
RUN git clone -b ${GLSLANG_REF} --depth 1 https://github.com/KhronosGroup/glslang /src/glslang \
 && cmake -S /src/glslang -B /src/glslang/build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DENABLE_OPT=OFF -DGLSLANG_TESTS=OFF -DCMAKE_INSTALL_PREFIX=/usr/local \
 && ninja -C /src/glslang/build install \
 && rm -rf /src/glslang
ARG MESA_REF
# EL8 ninja 1.8.2 is too old for Mesa's multiple-output depslog; # XXX bindgen's clang needs gcc-toolset-13's libstdc++ headers
RUN TRIPLE=$(gcc -dumpmachine) \
 && export BINDGEN_EXTRA_CLANG_ARGS="-isystem /opt/rh/gcc-toolset-13/root/usr/include/c++/13 -isystem /opt/rh/gcc-toolset-13/root/usr/include/c++/13/$TRIPLE --gcc-install-dir=/opt/rh/gcc-toolset-13/root/usr/lib/gcc/$TRIPLE/13" \
 && pip3.11 install --no-cache-dir 'ninja>=1.11' \
 && git clone -b ${MESA_REF} --depth 1 https://gitlab.freedesktop.org/mesa/mesa /src/mesa \
 && meson setup /src/mesa /src/mesa/build \
      -Dbuildtype=release \
      -Dvulkan-drivers=swrast -Dgallium-drivers=llvmpipe \
      -Dgallium-rusticl=true -Drust_std=2021 -Dstatic-libclc=all \
      -Dmesa-clc-bundle-headers=enabled \
      -Dspirv-tools=enabled \
      -Dllvm=enabled -Dshared-llvm=enabled -Dcpp_rtti=false \
      -Dplatforms= -Dglx=disabled -Degl=disabled -Dgbm=disabled \
      -Dgles1=disabled -Dgles2=disabled -Dopengl=false \
      -Dvulkan-layers= -Dvideo-codecs= \
      --prefix=/opt/mesa \
 && ninja -C /src/mesa/build install

FROM base AS vkloader
ARG VULKAN_REF
RUN git clone -b ${VULKAN_REF} --depth 1 https://github.com/KhronosGroup/Vulkan-Headers /src/vkh \
 && cmake -S /src/vkh -B /src/vkh/b -DCMAKE_INSTALL_PREFIX=/usr/local && cmake --install /src/vkh/b \
 && git clone -b ${VULKAN_REF} --depth 1 https://github.com/KhronosGroup/Vulkan-Loader /src/vkl \
 && cmake -S /src/vkl -B /src/vkl/b -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/opt/vkloader -DVULKAN_HEADERS_INSTALL_DIR=/usr/local \
      -DBUILD_WSI_XLIB_SUPPORT=OFF -DBUILD_WSI_XCB_SUPPORT=OFF \
      -DBUILD_WSI_WAYLAND_SUPPORT=OFF -DBUILD_WSI_DIRECTFB_SUPPORT=OFF \
      -DBUILD_TESTS=OFF \
 && ninja -C /src/vkl/b install

# XXX PoCL builds a driver only against its pkg-config
FROM base AS icdloader
ARG OPENCL_HEADERS_REF
ARG OCL_ICD_REF
RUN dnf install -y -q ruby autoconf automake libtool && dnf clean all
RUN git clone -b ${OPENCL_HEADERS_REF} --depth 1 https://github.com/KhronosGroup/OpenCL-Headers /src/ocl-h \
 && cmake -S /src/ocl-h -B /src/ocl-h/b -DCMAKE_INSTALL_PREFIX=/opt/ocl && cmake --install /src/ocl-h/b \
 && git clone -b ${OCL_ICD_REF} --depth 1 https://github.com/OCL-dev/ocl-icd /src/ocl-icd \
 && cd /src/ocl-icd && ./bootstrap \
 && ./configure --prefix=/opt/ocl --libdir=/opt/ocl/lib CPPFLAGS=-I/opt/ocl/include \
 && make -j"$(nproc)" install

FROM base AS pocl
ARG TARGETARCH
ARG POCL_REF
COPY --from=icdloader /opt/ocl /opt/ocl
RUN CPU=$([ "$TARGETARCH" = arm64 ] && echo generic || echo x86-64) \
 && git clone -b ${POCL_REF} --depth 1 https://github.com/pocl/pocl /src/pocl \
 && export PKG_CONFIG_PATH=/opt/ocl/lib/pkgconfig:${PKG_CONFIG_PATH} \
 && cmake -S /src/pocl -B /src/pocl/build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DWITH_LLVM_CONFIG=/opt/llvm/bin/llvm-config -DCMAKE_PREFIX_PATH=/opt/ocl \
      -DENABLE_ICD=ON -DENABLE_HOST_CPU_DEVICES=ON -DENABLE_LOADABLE_DRIVERS=OFF \
      -DENABLE_SPIRV=ON -DLLVM_SPIRV=/opt/llvm/bin/llvm-spirv \
      -DSTATIC_LLVM=OFF -DLLC_HOST_CPU=$CPU -DKERNELLIB_HOST_CPU_VARIANTS=$CPU \
      -DENABLE_TESTS=OFF -DENABLE_EXAMPLES=OFF \
      -DCMAKE_INSTALL_PREFIX=/opt/pocl \
 && ninja -C /src/pocl/build install

FROM base AS ocelot
ARG GPUOCELOT_COMMIT
COPY gpuocelot.patch /src/gpuocelot.patch
# XXX g++ 13, not clang: gcc 16 miscompiles the emulator, gcc 13 predates that.
RUN git clone --recurse-submodules https://github.com/gpuocelot/gpuocelot /src/gpuocelot \
 && cd /src/gpuocelot && git checkout -q ${GPUOCELOT_COMMIT} && git submodule update --init --recursive \
 && git apply /src/gpuocelot.patch \
 && cd ocelot && cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
      -DBUILD_TOOLS=OFF -DLLVM_DIR=${LLVM_DIR} \
      -DBOOST_ROOT=/opt/boost -DBoost_USE_STATIC_LIBS=ON -DBoost_NO_SYSTEM_PATHS=ON \
      -DCMAKE_CXX_FLAGS_RELEASE="-O1 -DNDEBUG -fno-strict-aliasing -fwrapv" \
 && ninja -C build gpuocelot

FROM base AS swiftshader
ARG SWIFTSHADER_COMMIT
RUN TRIPLE=$(gcc -dumpmachine) \
 && GCC_DIR="--gcc-install-dir=/opt/rh/gcc-toolset-13/root/usr/lib/gcc/$TRIPLE/13" \
 && git clone --filter=blob:none https://github.com/google/swiftshader /src/swiftshader \
 && git -C /src/swiftshader checkout -q ${SWIFTSHADER_COMMIT} \
 && cmake -S /src/swiftshader -B /src/swiftshader/build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
      -DCMAKE_C_FLAGS="$GCC_DIR" -DCMAKE_CXX_FLAGS="$GCC_DIR" \
      -DCMAKE_SHARED_LINKER_FLAGS=-Wl,-z,noexecstack \
      -DSWIFTSHADER_BUILD_TESTS=OFF -DSWIFTSHADER_WARNINGS_AS_ERRORS=OFF \
 && ninja -C /src/swiftshader/build vk_swiftshader \
 && mkdir -p /opt/swiftshader \
 && cp "$(find /src/swiftshader/build -name libvk_swiftshader.so -type f | head -1)" /opt/swiftshader/ \
 && cp "$(find /src/swiftshader/build -name vk_swiftshader_icd.json | head -1)" /opt/swiftshader/

FROM base AS export
COPY --from=mesa     /opt/mesa     /opt/mesa
COPY --from=vkloader /opt/vkloader /opt/vkloader
COPY --from=vkloader /usr/local/include /opt/vkloader/include
COPY --from=icdloader /opt/ocl     /opt/ocl
COPY --from=pocl     /opt/pocl     /opt/pocl
COPY --from=ocelot   /src/gpuocelot/ocelot/build/libgpuocelot.so /opt/ocelot/libgpuocelot.so
COPY --from=swiftshader /opt/swiftshader /opt/swiftshader
COPY collect.sh /src/collect.sh
RUN dnf install -y -q patchelf && bash /src/collect.sh

FROM export AS check
COPY vecadd.cu vecadd.cl.cpp vecadd.vk.cpp run.sh /src/
RUN bash /src/run.sh /out
