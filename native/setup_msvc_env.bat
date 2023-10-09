@echo off
::setlocal enabledelayedexpansion

if "%GITHUB_ENV%"=="" (
  set "VC=C:\Program Files\Microsoft Visual Studio\2022\Community"
) else (
  set "VC=C:\Program Files\Microsoft Visual Studio\2022\Enterprise"
)

echo Using VC=%VC%

if not "%VCPKG_ROOT%"=="" (
  set "VCPKG_ROOT_=%VCPKG_ROOT%"
)

set "PATH=%VC%\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe;%PATH%"
:: See https://docs.microsoft.com/en-us/cpp/build/building-on-the-command-line?view=msvc-170
call "%VC%\VC\Auxiliary\Build\vcvarsall.bat" %1
:: X86 | AMD64

if not "%VCPKG_ROOT_%"=="" (
  set "VCPKG_ROOT=%VCPKG_ROOT_%"
)

echo Done