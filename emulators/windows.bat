@echo off
rem   windows.bat [out]  build+stage  |  --collect re-stage+smoke  |  --check smoke
setlocal enabledelayedexpansion

set "DIR=%~dp0"
set "WORK=%EMU_WORK%"
if "%WORK%"=="" set "WORK=%USERPROFILE%\emu-build"
if not exist "%WORK%" mkdir "%WORK%"
set "LLVM_REF=llvmorg-21.1.8"
set "MESA_REF=mesa-26.1.1"
set "POCL_REF=v7.1"
set "VKL_REF=v1.3.280"
set "GLSLANG_REF=vulkan-sdk-1.4.350.0"
set "SWIFTSHADER_REF=2843cbcc714fe111e1083127c048a18002bc10ed"
set "WFB_URL=https://github.com/lexxmark/winflexbison/releases/download/v2.5.25/win_flex_bison-2.5.25.zip"
set "SED=C:\Program Files\Git\usr\bin\sed.exe"
rem LLVM21 ignores LLVM_USE_CRT_RELEASE; CMP0091 makes CMAKE_MSVC_RUNTIME_LIBRARY apply
set "MT_CMAKE=-DCMAKE_POLICY_DEFAULT_CMP0091=NEW -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded"

rem arch-aware toolchain/target (amd64 default; arm64 override)
rem POCL_CPU pins pocl's kernel-codegen target - runtime auto-detect picks micro-archs (znver4 etc.) it mishandles
set "VCARCH=64" & set "VCREQ=Microsoft.VisualStudio.Component.VC.Tools.x86.x64"
set "LLVM_TGT=X86" & set "ICDARCH=x86_64" & set "POCL_CPU=-DLLC_HOST_CPU=x86-64" & set "SS=1"
if /i "%PROCESSOR_ARCHITECTURE%"=="ARM64" (
  set "VCARCH=arm64" & set "VCREQ=Microsoft.VisualStudio.Component.VC.Tools.ARM64"
  set "LLVM_TGT=AArch64" & set "ICDARCH=aarch64" & set "POCL_CPU=-DLLC_HOST_CPU=generic"
)

set "MODE=build"
if /i "%~1"=="--collect" set "MODE=collect"
if /i "%~1"=="--check"   set "MODE=check"
if "%MODE%"=="build" (set "OUT=%~1") else (set "OUT=%~2")
if "%OUT%"=="" set "OUT=%DIR%out"

for /f "usebackq delims=" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires %VCREQ% -property installationPath`) do set "VS=%%i"
if not defined VS (echo no VC-capable Visual Studio found & exit /b 1)
call "%VS%\VC\Auxiliary\Build\vcvars%VCARCH%.bat" >nul || (echo VCVARS-FAIL & exit /b 1)
for /f "delims=" %%p in ('where python 2^>nul') do if not defined PY set "PY=%%p"
set "PATH=%WORK%\llvm\bin;%WORK%\winflexbison;%WORK%\glslang-prefix\bin;%PATH%"

if "%MODE%"=="check"   goto :check
if "%MODE%"=="collect" goto :collect

rem deps
"%PY%" -m pip install -q --upgrade pip || exit /b 1
"%PY%" -m pip install -q meson mako pyyaml packaging || exit /b 1
if not exist "%WORK%\winflexbison\win_flex.exe" (
  curl -fL -o "%WORK%\wfb.zip" "%WFB_URL%" || (echo WFB-DL-FAIL & exit /b 1)
  if not exist "%WORK%\winflexbison" mkdir "%WORK%\winflexbison"
  tar -xf "%WORK%\wfb.zip" -C "%WORK%\winflexbison" || (echo WFB-EXTRACT-FAIL & exit /b 1)
)

rem LLVM (clang, static CRT) -- winget LLVM has no dev libs
if not exist "%WORK%\llvm-project\.git" git clone --depth 1 -b %LLVM_REF% https://github.com/llvm/llvm-project "%WORK%\llvm-project" || exit /b 1
cmake -S "%WORK%\llvm-project\llvm" -B "%WORK%\build-llvm" -G Ninja -DCMAKE_BUILD_TYPE=Release %MT_CMAKE% ^
  -DLLVM_ENABLE_PROJECTS=clang -DLLVM_TARGETS_TO_BUILD=%LLVM_TGT% ^
  -DLLVM_ENABLE_RTTI=OFF -DLLVM_ENABLE_TERMINFO=OFF -DLLVM_ENABLE_ZSTD=OFF ^
  -DLLVM_INCLUDE_TESTS=OFF -DLLVM_INCLUDE_BENCHMARKS=OFF -DLLVM_INCLUDE_EXAMPLES=OFF ^
  -DLLVM_ENABLE_BINDINGS=OFF -DCMAKE_INSTALL_PREFIX="%WORK%\llvm" || exit /b 1
ninja -C "%WORK%\build-llvm" install || (echo LLVM-FAIL & exit /b 1)

rem glslang (build tool for lavapipe; invoked as a subprocess, so CRT is irrelevant)
if not exist "%WORK%\glslang-src\.git" git clone -b %GLSLANG_REF% --depth 1 https://github.com/KhronosGroup/glslang "%WORK%\glslang-src" || exit /b 1
cmake -S "%WORK%\glslang-src" -B "%WORK%\build-glslang" -G Ninja -DCMAKE_BUILD_TYPE=Release ^
  -DENABLE_OPT=OFF -DGLSLANG_TESTS=OFF -DENABLE_GLSLANG_BINARIES=ON -DCMAKE_INSTALL_PREFIX="%WORK%\glslang-prefix" || exit /b 1
ninja -C "%WORK%\build-glslang" install || (echo GLSLANG-FAIL & exit /b 1)
if not exist "%WORK%\glslang-prefix\bin\glslangValidator.exe" copy /y "%WORK%\glslang-prefix\bin\glslang.exe" "%WORK%\glslang-prefix\bin\glslangValidator.exe" >nul

rem Vulkan-Loader (vulkan-1.dll, static CRT)
if not exist "%WORK%\vkh\.git" git clone -b %VKL_REF% --depth 1 https://github.com/KhronosGroup/Vulkan-Headers "%WORK%\vkh" || exit /b 1
cmake -S "%WORK%\vkh" -B "%WORK%\vkh\b" -G Ninja -DCMAKE_INSTALL_PREFIX="%WORK%\vkinstall" && cmake --install "%WORK%\vkh\b" || exit /b 1
if not exist "%WORK%\vkl\.git" git clone -b %VKL_REF% --depth 1 https://github.com/KhronosGroup/Vulkan-Loader "%WORK%\vkl" || exit /b 1
cmake -S "%WORK%\vkl" -B "%WORK%\vkl\b" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="%WORK%\vkinstall" ^
  -DVULKAN_HEADERS_INSTALL_DIR="%WORK%\vkinstall" -DBUILD_TESTS=OFF -DUSE_GAS=OFF %MT_CMAKE% || exit /b 1
ninja -C "%WORK%\vkl\b" install || (echo VKLOADER-FAIL & exit /b 1)

rem Mesa lavapipe (static LLVM + CRT). drm.h assumes non-Linux==BSD and pulls sys/ioccom.h (idempotent guard);
rem cpp_rtti matches LLVM RTTI=OFF; expat off (POSIX xmlconfig); /wd* demote Mesa's warning-as-error;
rem platforms=windows declares vk_icdEnumerateAdapterPhysicalDevices
if not exist "%WORK%\mesa\.git" git clone -b %MESA_REF% --depth 1 https://gitlab.freedesktop.org/mesa/mesa "%WORK%\mesa" || exit /b 1
findstr /c:"!defined(_WIN32)" "%WORK%\mesa\include\drm-uapi\drm.h" >nul || "%SED%" -i "s|#include <sys/ioccom.h>|#if !defined(_WIN32)\n#include <sys/ioccom.h>\n#endif|" "%WORK%\mesa\include\drm-uapi\drm.h"
rmdir /s /q "%WORK%\build-mesa" 2>nul
rmdir /s /q "%WORK%\mesa-prefix" 2>nul
"%PY%" -m mesonbuild.mesonmain setup "%WORK%\mesa" "%WORK%\build-mesa" --prefix="%WORK%\mesa-prefix" ^
  -Dbuildtype=release -Db_vscrt=mt -Dvulkan-drivers=swrast -Dgallium-drivers=llvmpipe ^
  -Dllvm=enabled -Dshared-llvm=disabled -Dcpp_rtti=false ^
  -Dopengl=false -Dgles1=disabled -Dgles2=disabled -Degl=disabled ^
  -Dgbm=disabled -Dglx=disabled -Dplatforms=windows -Dexpat=disabled ^
  -Dc_args="/wd4189 /wd4020 /wd4024" -Dcpp_args="/wd4189 /wd4020 /wd4024" ^
  -Dvideo-codecs= -Dgallium-va=disabled || (echo MESA-CFG-FAIL & exit /b 1)
"%PY%" -m mesonbuild.mesonmain install -C "%WORK%\build-mesa" || (echo MESA-FAIL & exit /b 1)

rem pocl as a direct OpenCL.dll (static device + LLVM + CRT)
if not exist "%WORK%\pocl-src\.git" git clone -b %POCL_REF% --depth 1 https://github.com/pocl/pocl "%WORK%\pocl-src" || exit /b 1
rmdir /s /q "%WORK%\build-pocl" 2>nul
rmdir /s /q "%WORK%\pocl-prefix" 2>nul
cmake -S "%WORK%\pocl-src" -B "%WORK%\build-pocl" -G Ninja -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_C_COMPILER=clang-cl -DCMAKE_CXX_COMPILER=clang-cl %MT_CMAKE% ^
  -DWITH_LLVM_CONFIG="%WORK%\llvm\bin\llvm-config.exe" -DSTATIC_LLVM=ON %POCL_CPU% ^
  -DENABLE_ICD=ON -DENABLE_HOST_CPU_DEVICES=ON -DENABLE_LOADABLE_DRIVERS=OFF ^
  -DENABLE_SPIRV=OFF -DENABLE_HWLOC=OFF -DENABLE_TESTS=OFF -DENABLE_EXAMPLES=OFF ^
  -DCMAKE_INSTALL_PREFIX="%WORK%\pocl-prefix" || (echo POCL-CFG-FAIL & exit /b 1)
ninja -C "%WORK%\build-pocl" install || (echo POCL-FAIL & exit /b 1)

rem SwiftShader (second Vulkan; vendored static LLVM + static CRT, self-contained, hidden symbols)
if not defined SS goto :collect
if not exist "%WORK%\swiftshader\.git" git clone --filter=blob:none https://github.com/google/swiftshader "%WORK%\swiftshader" || exit /b 1
git -C "%WORK%\swiftshader" checkout -q %SWIFTSHADER_REF% || (echo SS-CHECKOUT-FAIL & exit /b 1)
rem Windows reports CMAKE_SYSTEM_PROCESSOR=ARM64 (uppercase); SwiftShader's case-sensitive arch regex
rem misses it, defaults ARCH=x86_64 and builds X86 LLVM the AArch64 Reactor JIT can't link - widen it
powershell -NoProfile -Command "$f='%WORK%\swiftshader\CMakeLists.txt'; (Get-Content -Raw $f) -replace 'MATCHES \"arm\" OR CMAKE_SYSTEM_PROCESSOR MATCHES \"aarch\"', 'MATCHES \"[Aa][Rr][Mm]\" OR CMAKE_SYSTEM_PROCESSOR MATCHES \"[Aa][Aa][Rr][Cc][Hh]\"' | Set-Content -NoNewline $f" || (echo SS-PATCH-FAIL & exit /b 1)
rmdir /s /q "%WORK%\swiftshader\build-ss" 2>nul
rem force cl (MSVC): else SwiftShader's vendored-LLVM-10 build picks the from-source clang++ on PATH and
rem feeds it GNU flags (-fPIC/-march) invalid for the *-pc-windows-msvc target
cmake -S "%WORK%\swiftshader" -B "%WORK%\swiftshader\build-ss" -G Ninja -DCMAKE_BUILD_TYPE=Release %MT_CMAKE% ^
  -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl ^
  -DSWIFTSHADER_BUILD_TESTS=OFF -DSWIFTSHADER_WARNINGS_AS_ERRORS=OFF || (echo SS-CFG-FAIL & exit /b 1)
cmake --build "%WORK%\swiftshader\build-ss" --target vk_swiftshader || (echo SS-FAIL & exit /b 1)

:collect
rmdir /s /q "%OUT%" 2>nul
mkdir "%OUT%\bin" "%OUT%\share\vulkan\icd.d" 2>nul
copy /y "%WORK%\mesa-prefix\bin\vulkan_lvp.dll" "%OUT%\bin\" >nul
copy /y "%WORK%\mesa-prefix\share\vulkan\icd.d\lvp_icd.%ICDARCH%.json" "%OUT%\share\vulkan\icd.d\" >nul
copy /y "%WORK%\vkinstall\bin\vulkan-1.dll" "%OUT%\bin\" >nul
copy /y "%WORK%\pocl-prefix\bin\OpenCL.dll" "%OUT%\bin\" >nul
if defined SS (
  copy /y "%WORK%\swiftshader\build-ss\Windows\vk_swiftshader.dll" "%OUT%\bin\" >nul
  copy /y "%WORK%\swiftshader\build-ss\Windows\vk_swiftshader_icd.json" "%OUT%\share\vulkan\icd.d\" >nul
)
rem pocl finds bitcode/headers at ..\share\pocl
xcopy /e /i /y /q "%WORK%\pocl-prefix\share\pocl" "%OUT%\share\pocl" >nul
rem point each ICD manifest at its driver in ..\..\..\bin
"%PY%" -c "import json;p=r'%OUT%\share\vulkan\icd.d\lvp_icd.%ICDARCH%.json';d=json.load(open(p));d['ICD']['library_path']=r'..\..\..\bin\vulkan_lvp.dll';json.dump(d,open(p,'w'),indent=4)"
if defined SS "%PY%" -c "import json;p=r'%OUT%\share\vulkan\icd.d\vk_swiftshader_icd.json';d=json.load(open(p));d['ICD']['library_path']=r'..\..\..\bin\vk_swiftshader.dll';json.dump(d,open(p,'w'),indent=4)"
> "%OUT%\env.bat" echo @echo off
>>"%OUT%\env.bat" echo set "HERE=%%~dp0"
>>"%OUT%\env.bat" echo set "PATH=%%HERE%%bin;%%PATH%%"
>>"%OUT%\env.bat" echo set "VK_DRIVER_FILES=%%HERE%%share\vulkan\icd.d\lvp_icd.%ICDARCH%.json"
if defined SS >>"%OUT%\env.bat" echo set "VK_DRIVER_FILES=%%VK_DRIVER_FILES%%;%%HERE%%share\vulkan\icd.d\vk_swiftshader_icd.json"
>>"%OUT%\env.bat" echo set "VK_ICD_FILENAMES=%%VK_DRIVER_FILES%%"
>>"%OUT%\env.bat" echo set "POLYINVOKE_OPENCL_CPU=1"
echo === bundle staged at %OUT% ===
dir /b "%OUT%\bin"

:check
echo === smoke ===
cl /nologo /MT /EHsc /std:c++17 /I"%WORK%\vkinstall\include" "%DIR%vecadd.vk.cpp" /Fe:"%OUT%\bin\vk_vecadd.exe" /link "%WORK%\vkinstall\lib\vulkan-1.lib" >nul || (echo VK-COMPILE-FAIL & exit /b 1)
cl /nologo /MT /EHsc /std:c++17 /I"%WORK%\pocl-prefix\include" "%DIR%vecadd.cl.cpp" /Fe:"%OUT%\bin\clvec.exe" /link "%WORK%\pocl-prefix\lib\OpenCL.lib" >nul || (echo CL-COMPILE-FAIL & exit /b 1)
call "%OUT%\env.bat"
rem registry ICDs may also appear; the VK_DRIVER_FILES override needs a non-elevated context
echo == Vulkan (lavapipe + swiftshader) ==
"%OUT%\bin\vk_vecadd.exe"
echo == OpenCL (pocl) ==
"%OUT%\bin\clvec.exe"
echo SMOKE DONE
