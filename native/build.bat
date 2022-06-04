@echo off
setlocal enabledelayedexpansion

set ACTION=%1
set ARCH=windows-x86_64
set BUILD=build-%ARCH%
rem Using build name %BUILD%

set LINKER="C:\\Program Files\\LLVM\\bin\\lld-link.exe"
set NINJA="C:\\Users\\Tom\\Downloads\\ninja-win\\ninja.exe"

:: See https://docs.microsoft.com/en-us/cpp/build/building-on-the-command-line?view=msvc-170
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64


if "%ACTION%"=="configure" (call :configure) 
if "%ACTION%"=="build" (call :build %2) 
rem Unknown action %ACTION%

goto:eof

:configure
  @echo on
  cmake -B %BUILD% -S . ^
    -DUSE_LINKER=%LINKER% ^
    -DCMAKE_BUILD_TYPE=Release ^
    -G "Ninja" ^
    -DCMAKE_MAKE_PROGRAM=%NINJA%
  @echo off  
goto:eof

:build 
  set TARGET=%1
  @echo on
  cmake --build %BUILD% --target "%TARGET%"
  @echo off
goto:eof
