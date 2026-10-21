# Immutable source inputs used by the Spectra producer campaign.
#
# Upstream release check: 2026-08-26.
# - oneDPL 2022.13.0 remains the latest oneDPL release.
# - ROCm 7.2.4 / rocPRIM 4.2.0 is newer, but 7.1.1 is retained to match the
#   validated HIP compiler/runtime campaign.
# - CCCL 3.4.0 is newer, but the CUDA 11.6.55 redist is retained as the oldest
#   producer/toolkit combination supported by the validated backend matrix.

set(SPECTRA_ONEDPL_VERSION "2022.13.0")
set(SPECTRA_ONEDPL_COMMIT "7b0e7267ff81ba36259c834a52493ed6eac70d66")
set(SPECTRA_ONEDPL_SHA256 "5a5d926af29c98528a804bbdf57a2c7179ccda2caa0587f54719ee50182d4ea8")

set(SPECTRA_ROCM_VERSION "7.1.1")
set(SPECTRA_ROCTHRUST_COMMIT "b388186866089185616f97416d1af23569cc60b3")
set(SPECTRA_ROCTHRUST_SHA256 "f88d29c495e02848a9f8f11c20892e2440e4a236adb1bf1de7c1ba23eb3df485")
set(SPECTRA_ROCPRIM_COMMIT "8d8a6b57b5b0dfe38295c92adef2850a43fd35ff")
set(SPECTRA_ROCPRIM_SHA256 "fe2c9bd504d0715aadc508f3dc37609e9a9147754a484ab16f574354c876061d")

set(SPECTRA_CUDA_CCCL_VERSION "11.6.55")
set(SPECTRA_CUDA_CCCL_LINUX_X86_64_SHA256 "14aee04c86a47321b9fee1c86c9a640be7729ed8063663ed0c7dafdd77fb18f5")
set(SPECTRA_CUDA_CCCL_LINUX_SBSA_SHA256 "6de184be2cee64b278befc0839d0b08104f19f6d15358e3bd8a92cefca552e20")
# The matching runtime headers and libdevice form the producer CUDA toolkit, so no host install is needed.
set(SPECTRA_CUDA_CUDART_VERSION "11.6.55")
set(SPECTRA_CUDA_CUDART_LINUX_X86_64_SHA256 "734a77b3a26a9d08489d43afb74bad230c7c4a0ed2d17a6317a47cf363dca521")
set(SPECTRA_CUDA_CUDART_LINUX_SBSA_SHA256 "ae0ba7295280c4b2f2a217bc79cc604a3beb20484011dd2db604507f31fde0a7")
set(SPECTRA_CUDA_NVCC_VERSION "11.6.124")
set(SPECTRA_CUDA_NVCC_LINUX_X86_64_SHA256 "8c81199c5a096869a10c284197cefc1a958df8bf482322a0a48dff9cc82291b8")
set(SPECTRA_CUDA_NVCC_LINUX_SBSA_SHA256 "cdca0d230967ccf227063841ae984704a1812663bd0074e400eee57ba47d99a9")

# Official compiler-bearing images. Tags document the selected release while
# manifest digests make each CI and local smoke environment immutable.
set(SPECTRA_CUDA_CONTAINER
    "docker.io/nvidia/cuda:11.6.2-devel-ubuntu20.04@sha256:bf4c3628625d7d14e1a37a99fb57a91382fc3dcf9875665479ee177d59e93438")
set(SPECTRA_HIP_CONTAINER
    "docker.io/rocm/dev-ubuntu-24.04:7.1.1@sha256:383859242afc0ab21b8823265eeba8b148a9b8d09a4274b39d81e715f4816666")
set(SPECTRA_ONEDPL_CONTAINER
    "docker.io/intel/oneapi-toolkit:2026.1.0-devel-ubuntu24.04@sha256:e9db518398753434ee5aab9740a25f1d3134396a30be1569cfad8f8b0d90740c")
