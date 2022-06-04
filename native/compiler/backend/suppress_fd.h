#pragma once
#include <cstdio>

// XXX see https://stackoverflow.com/a/13498238/896997

#ifdef _WIN32
  #include <io.h>
const char *nulFileName = "NUL";
  #define CROSS_DUP(fd) _dup(fd)
  #define CROSS_DUP2(fd, newfd) _dup2(fd, newfd)
  #define STDIN_FILENO 0
  #define STDOUT_FILENO 1
  #define STDERR_FILENO 2
#else
  #include <unistd.h>
  #define CROSS_DUP(fd) dup(fd)
  #define CROSS_DUP2(fd, newfd) dup2(fd, newfd)
const char *nulFileName = "/dev/null";
#endif

namespace suppress_fd {

constexpr int STDOUT = STDOUT_FILENO;

template <int FD> class Suppressor {

  int saved;

public:
  Suppressor() : saved(CROSS_DUP(FD)) {
    std::fflush(stdout);
    FILE *nullOut = std::fopen(nulFileName, "w");
    CROSS_DUP2(fileno(nullOut), FD);
  }

  void restore() {
    CROSS_DUP2(saved, FD);
    close(saved);
  }
};

} // namespace suppress_fd