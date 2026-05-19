@echo off

set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" set "VSWHERE=%ProgramFiles%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
  echo ERROR: vswhere.exe not found
  exit /b 1
)

for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -prerelease -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
  set "VC=%%i"
)

if "%VC%"=="" (
  echo ERROR: no Visual Studio with VC tools found
  exit /b 1
)

echo Using VC=%VC%

:: XXX Pin vcpkg to the activated VS install so its prebuilts (Catch2.lib) link against
:: the same MSVC stdlib as our objects.
set "VCPKG_VISUAL_STUDIO_PATH=%VC%"

if not "%VCPKG_ROOT%"=="" (
  set "VCPKG_ROOT_=%VCPKG_ROOT%"
)

set "PATH=%VC%\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;%PATH%"
:: See https://docs.microsoft.com/en-us/cpp/build/building-on-the-command-line?view=msvc-170
call "%VC%\VC\Auxiliary\Build\vcvarsall.bat" %1
:: X86 | AMD64

if not "%VCPKG_ROOT_%"=="" (
  set "VCPKG_ROOT=%VCPKG_ROOT_%"
)

echo Done
