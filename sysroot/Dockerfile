ARG SYSBASE=quay.io/almalinuxorg/almalinux:8
FROM ${SYSBASE} AS staging
ARG RHEL_TRIPLE
ARG GNU_TRIPLE
ARG EXTRA_PKGS

RUN dnf install -y dnf-plugins-core && dnf config-manager --set-enabled powertools

RUN mkdir -p /mnt/sys-root \
 && dnf --installroot /mnt/sys-root --releasever 8 --setopt install_weak_deps=false \
        --nodocs -y install \
        glibc glibc-devel glibc-headers glibc-static \
        libgcc libstdc++ libstdc++-devel libstdc++-static \
        libatomic libatomic-static \
        libgfortran \
        kernel-headers \
        gcc gcc-c++ \
        gcc-toolset-12-gcc gcc-toolset-12-gcc-c++ \
        gcc-toolset-12-libstdc++-devel gcc-toolset-12-libatomic-devel \
        ${EXTRA_PKGS} \
 && dnf --installroot /mnt/sys-root clean all

RUN rm -rf \
    /mnt/sys-root/usr/share/{locale,doc,man,info} \
    /mnt/sys-root/boot \
    /mnt/sys-root/var/{cache,log,lib}/{dnf,rpm,yum.*}

RUN cd /mnt/sys-root/usr/lib64 \
 && ln -sf libstdc++.so.6   libstdc++.so   \
 && ln -sf libgcc_s.so.1    libgcc_s.so    \
 && ln -sf libquadmath.so.0 libquadmath.so \
 && ln -sf libgfortran.so.5 libgfortran.so \
 && ln -sf libatomic.so.1   libatomic.so   \
 && cd "/mnt/sys-root/usr/lib/gcc/${RHEL_TRIPLE}/8" \
 && ln -sf ../../../../lib64/libstdc++.so.6 libstdc++.so

RUN ln -sfn "${RHEL_TRIPLE}" "/mnt/sys-root/usr/lib/gcc/${GNU_TRIPLE}" \
 && ln -sfn "${RHEL_TRIPLE}" "/mnt/sys-root/usr/include/c++/8/${GNU_TRIPLE}"

RUN chmod -R u+rwX /mnt/sys-root

FROM scratch
COPY --from=staging /mnt/sys-root/ /
# XXX Dummy CMD so `create` accepts the image; we only export its filesystem.
CMD ["/notused"]
