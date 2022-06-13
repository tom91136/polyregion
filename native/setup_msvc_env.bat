@echo off
::setlocal enabledelayedexpansion

if "%GITHUB_ENV%"=="" (
  set "VC=C:\Program Files\Microsoft Visual Studio\2022\Community"
) else (
  set "VC=C:\Program Files\Microsoft Visual Studio\2022\Enterprise"
)

echo Using VC=%VC%

set "PATH=%VC%\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe;%PATH%"
:: See https://docs.microsoft.com/en-us/cpp/build/building-on-the-command-line?view=msvc-170
call "%VC%\VC\Auxiliary\Build\vcvarsall.bat" x64

echo Done