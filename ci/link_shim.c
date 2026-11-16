#include <stdio.h>
#include <wchar.h>
#include <windows.h>

static wchar_t dir[MAX_PATH];

static void keepInputs(const DWORD pid) {
  int argc = 0;
  LPWSTR *argv = CommandLineToArgvW(GetCommandLineW(), &argc);
  if (!argv) return;
  for (int i = 1; i < argc; ++i) {
    const size_t len = wcslen(argv[i]);
    if (len < 3 || _wcsicmp(argv[i] + len - 2, L".o") != 0) continue;
    wchar_t dst[MAX_PATH];
    swprintf_s(dst, MAX_PATH, L"%slink-shim-%lu-%d.o", dir, pid, i);
    CopyFileW(argv[i], dst, FALSE);
  }
  LocalFree(argv);
}

int wmain(void) {
  const DWORD n = GetModuleFileNameW(NULL, dir, MAX_PATH);
  wchar_t *slash = wcsrchr(dir, L'\\');
  if (n == 0 || n >= MAX_PATH || !slash) return 125;
  slash[1] = 0;

  wchar_t real[MAX_PATH];
  swprintf_s(real, MAX_PATH, L"%slink-real.exe", dir);
  const wchar_t *tail = GetCommandLineW();
  const wchar_t *quote = *tail == L'"' ? wcschr(tail + 1, L'"') : NULL;
  tail = quote ? quote + 1 : tail + wcscspn(tail, L" ");
  static wchar_t cmd[32768];
  swprintf_s(cmd, 32768, L"\"%s\"%s", real, tail);

  STARTUPINFOW si = {.cb = sizeof(si)};
  PROCESS_INFORMATION pi;
  const ULONGLONG t0 = GetTickCount64();
  DWORD code = 125;
  if (CreateProcessW(real, cmd, NULL, NULL, TRUE, 0, NULL, NULL, &si, &pi)) {
    WaitForSingleObject(pi.hProcess, INFINITE);
    GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
  } else code = 0xFFFF0000u | GetLastError();
  if (code == 0) return 0;

  const DWORD pid = GetCurrentProcessId();
  keepInputs(pid);
  wchar_t logPath[MAX_PATH];
  swprintf_s(logPath, MAX_PATH, L"%slink-shim.log", dir);
  FILE *f = NULL;
  if (_wfopen_s(&f, logPath, L"a") == 0 && f) {
    fwprintf(f, L"pid %lu exit 0x%08lX after %llu ms\n  %s\n", pid, code, GetTickCount64() - t0, GetCommandLineW());
    fclose(f);
  }
  return (int)code;
}
