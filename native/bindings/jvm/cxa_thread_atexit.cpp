#include <cxxabi.h>

// See https://hg.mozilla.org/integration/autoland/rev/98efceb86ec5
// Specifically, the diff at
// https://hg.mozilla.org/integration/autoland/diff/98efceb86ec55e39c4306bca0ec27486366ec9ad/build/unix/stdc%2B%2Bcompat/stdc%2B%2Bcompat.cpp

extern "C" int __cxa_thread_atexit_impl(void (*dtor)(void *), void *obj, void *dso_handle) {
  return __cxxabiv1::__cxa_thread_atexit(dtor, obj, dso_handle);
}