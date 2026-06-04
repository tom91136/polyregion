FROM almalinux:8 AS base
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN dnf -y install epel-release dnf-plugins-core \
 && dnf config-manager --set-enabled powertools \
 && dnf -y install gcc-toolset-13 git make cmake ninja-build pkgconfig \
      python3.11 python3.11-pip flex bison wget xz tar which file patch \
      zlib-devel expat-devel libzstd-devel libffi-devel ncurses-devel \
      libxml2-devel libdrm-devel elfutils-libelf-devel \
      boost-devel glew-devel hwloc-devel libtool-ltdl-devel \
 && dnf clean all
RUN pip3.11 install --no-cache-dir 'meson>=1.3' mako packaging pyyaml

ENV PATH=/opt/rh/gcc-toolset-13/root/usr/bin:/opt/llvm15/bin:/usr/local/bin:/usr/bin:/bin
ENV LD_LIBRARY_PATH=/opt/rh/gcc-toolset-13/root/usr/lib64:/opt/llvm15/lib
ENV CC=gcc CXX=g++

ARG NJOBS=
RUN git clone --depth 1 -b llvmorg-15.0.6 https://github.com/llvm/llvm-project /src/llvm \
 && cmake -S /src/llvm/llvm -B /src/llvm/build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DLLVM_ENABLE_PROJECTS=clang \
      -DLLVM_TARGETS_TO_BUILD="X86;AArch64;NVPTX" \
      -DLLVM_BUILD_LLVM_DYLIB=ON -DLLVM_LINK_LLVM_DYLIB=ON \
      -DLLVM_ENABLE_RTTI=OFF -DLLVM_ENABLE_TERMINFO=OFF -DLLVM_ENABLE_ZSTD=OFF \
      -DLLVM_INCLUDE_TESTS=OFF -DLLVM_INCLUDE_BENCHMARKS=OFF \
      -DLLVM_INCLUDE_EXAMPLES=OFF -DLLVM_ENABLE_BINDINGS=OFF \
      -DLLVM_PARALLEL_LINK_JOBS=2 -DCMAKE_INSTALL_PREFIX=/opt/llvm15 \
 && ninja ${NJOBS:+-j$NJOBS} -C /src/llvm/build install \
 && rm -rf /src/llvm
ENV LLVM_DIR=/opt/llvm15/lib/cmake/llvm

RUN git clone -b llvm_release_150 --depth 1 https://github.com/KhronosGroup/SPIRV-LLVM-Translator /src/spirv-xlat \
 && cmake -S /src/spirv-xlat -B /src/spirv-xlat/build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DLLVM_DIR=${LLVM_DIR} -DCMAKE_INSTALL_PREFIX=/opt/llvm15 \
 && ninja -C /src/spirv-xlat/build install \
 && rm -rf /src/spirv-xlat

# XXX EL8's boost-static is non-PIC and cannot go into a DSO
RUN wget -q -O /tmp/boost.tar.gz https://archives.boost.io/release/1.66.0/source/boost_1_66_0.tar.gz \
 && tar -C /tmp -xf /tmp/boost.tar.gz && cd /tmp/boost_1_66_0 \
 && ./bootstrap.sh --prefix=/opt/boost --with-libraries=thread,filesystem,system,chrono,atomic,date_time \
 && ./b2 -j"$(nproc)" cxxflags=-fPIC cflags=-fPIC link=static runtime-link=shared threading=multi variant=release install \
 && rm -rf /tmp/boost*

FROM base AS lavapipe
RUN git clone -b mesa-24.0.9 --depth 1 https://gitlab.freedesktop.org/mesa/mesa /src/mesa \
 && meson setup /src/mesa /src/mesa/build \
      -Dbuildtype=release \
      -Dvulkan-drivers=swrast -Dgallium-drivers=swrast \
      -Dplatforms= -Dglx=disabled -Degl=disabled -Dgbm=disabled \
      -Dgles1=disabled -Dgles2=disabled -Dopengl=false \
      -Dllvm=enabled -Dshared-llvm=enabled -Dcpp_rtti=false \
      -Dvulkan-layers= -Dvideo-codecs= \
      --prefix=/opt/lavapipe \
 && ninja -C /src/mesa/build install

FROM base AS vkloader
RUN git clone -b v1.3.280 --depth 1 https://github.com/KhronosGroup/Vulkan-Headers /src/vkh \
 && cmake -S /src/vkh -B /src/vkh/b -DCMAKE_INSTALL_PREFIX=/usr/local && cmake --install /src/vkh/b \
 && git clone -b v1.3.280 --depth 1 https://github.com/KhronosGroup/Vulkan-Loader /src/vkl \
 && cmake -S /src/vkl -B /src/vkl/b -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/opt/vkloader -DVULKAN_HEADERS_INSTALL_DIR=/usr/local \
      -DBUILD_WSI_XLIB_SUPPORT=OFF -DBUILD_WSI_XCB_SUPPORT=OFF \
      -DBUILD_WSI_WAYLAND_SUPPORT=OFF -DBUILD_WSI_DIRECTFB_SUPPORT=OFF \
      -DBUILD_TESTS=OFF \
 && ninja -C /src/vkl/b install

FROM base AS pocl
ARG TARGETARCH
RUN CPU=$([ "$TARGETARCH" = arm64 ] && echo generic || echo x86-64) \
 && git clone -b v6.0 --depth 1 https://github.com/pocl/pocl /src/pocl \
 && cmake -S /src/pocl -B /src/pocl/build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DWITH_LLVM_CONFIG=/opt/llvm15/bin/llvm-config \
      -DENABLE_ICD=OFF -DENABLE_HOST_CPU_DEVICES=ON -DENABLE_LOADABLE_DRIVERS=OFF \
      -DENABLE_SPIRV=ON -DLLVM_SPIRV=/opt/llvm15/bin/llvm-spirv \
      -DSTATIC_LLVM=OFF -DLLC_HOST_CPU=$CPU -DKERNELLIB_HOST_CPU_VARIANTS=$CPU \
      -DENABLE_TESTS=OFF -DENABLE_EXAMPLES=OFF \
      -DCMAKE_INSTALL_PREFIX=/opt/pocl \
 && ninja -C /src/pocl/build install

FROM base AS ocelot
COPY gpuocelot.patch /src/gpuocelot.patch
# XXX g++ 13, not clang: gcc 16 miscompiles the emulator, gcc 13 predates that.
RUN git clone --recurse-submodules https://github.com/gpuocelot/gpuocelot /src/gpuocelot \
 && cd /src/gpuocelot && git checkout -q b16039dc && git submodule update --init --recursive \
 && git apply /src/gpuocelot.patch \
 && cd ocelot && cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
      -DBUILD_TOOLS=OFF -DLLVM_DIR=${LLVM_DIR} \
      -DBOOST_ROOT=/opt/boost -DBoost_USE_STATIC_LIBS=ON -DBoost_NO_SYSTEM_PATHS=ON \
      -DCMAKE_CXX_FLAGS_RELEASE="-O1 -DNDEBUG -fno-strict-aliasing -fwrapv" \
 && ninja -C build gpuocelot

FROM base AS export
COPY --from=lavapipe /opt/lavapipe /opt/lavapipe
COPY --from=vkloader /opt/vkloader /opt/vkloader
COPY --from=pocl     /opt/pocl     /opt/pocl
COPY --from=ocelot   /src/gpuocelot/ocelot/build/libgpuocelot.so /opt/ocelot/libgpuocelot.so
COPY collect.sh /src/collect.sh
RUN dnf install -y -q patchelf && bash /src/collect.sh
