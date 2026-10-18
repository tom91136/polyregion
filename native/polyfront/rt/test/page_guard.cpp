#ifndef _WIN32

  #include <algorithm>
  #include <atomic>
  #include <cerrno>
  #include <csignal>
  #include <cstdlib>
  #include <cstring>
  #include <functional>
  #include <mutex>
  #include <string>
  #include <thread>
  #include <vector>

  #include <fcntl.h>
  #include <unistd.h>

  #include <sys/mman.h>
  #include <sys/wait.h>

  #ifdef __APPLE__
    #include <mach-o/dyld.h>
  #endif

  #include <catch2/catch_test_macros.hpp>

  #include "polyrt/page_guard.hpp"

namespace guard = polyregion::polyrt::guard;

namespace {

// every scenario runs in a fresh process: a broken handler hangs, loops, or corrupts process-wide signal state
struct Outcome {
  int exitCode = -1;
  int signal = 0;
  std::string output;
};

std::string selfPath() {
  #ifdef __APPLE__
  char path[4096];
  uint32_t size = sizeof(path);
  return _NSGetExecutablePath(path, &size) == 0 ? std::string(path) : std::string{};
  #else
  char path[4096];
  const auto n = readlink("/proc/self/exe", path, sizeof(path) - 1);
  return n > 0 ? std::string(path, static_cast<size_t>(n)) : std::string{};
  #endif
}

Outcome runScenario(const std::string &name) {
  const auto self = selfPath();
  REQUIRE_FALSE(self.empty());
  int fds[2];
  REQUIRE(pipe(fds) == 0);
  const pid_t pid = fork();
  REQUIRE(pid >= 0);
  if (pid == 0) {
    dup2(fds[1], STDOUT_FILENO);
    dup2(fds[1], STDERR_FILENO);
    close(fds[0]);
    close(fds[1]);
    execl(self.c_str(), self.c_str(), name.c_str(), static_cast<char *>(nullptr));
    _exit(127);
  }
  close(fds[1]);
  Outcome outcome;
  char buffer[4096];
  for (ssize_t n; (n = read(fds[0], buffer, sizeof(buffer))) != 0;) {
    if (n < 0 && errno == EINTR) continue;
    if (n < 0) break;
    outcome.output.append(buffer, static_cast<size_t>(n));
  }
  close(fds[0]);
  int status = 0;
  while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {
  }
  if (WIFEXITED(status)) outcome.exitCode = WEXITSTATUS(status);
  if (WIFSIGNALED(status)) outcome.signal = WTERMSIG(status);
  return outcome;
}

void say(const char *message) { [[maybe_unused]] const auto written = write(STDERR_FILENO, message, std::strlen(message)); }

[[noreturn]] void fail(const char *message) {
  say(message);
  _exit(1);
}

// the signal a PROT_NONE access raises
  #ifdef __APPLE__
constexpr int ProtectionFault = SIGBUS;
  #else
constexpr int ProtectionFault = SIGSEGV;
  #endif

constexpr int markerExit(const int signal) { return signal == SIGBUS ? 43 : 42; }

void markerHandler(const int signal, siginfo_t *, void *) {
  say(signal == SIGBUS ? "prior SIGBUS\n" : "prior SIGSEGV\n");
  _exit(markerExit(signal));
}

void installMarker(const int signal) {
  struct sigaction action{};
  action.sa_sigaction = markerHandler;
  action.sa_flags = SA_SIGINFO;
  sigemptyset(&action.sa_mask);
  sigaction(signal, &action, nullptr);
}

char *mapPages(const size_t pages, const int prot) {
  void *p = mmap(nullptr, pages * guard::pageSize(), prot, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (p == MAP_FAILED) fail("mmap failed\n");
  return static_cast<char *>(p);
}

std::atomic<size_t> resolutions{0};

guard::Resolution countingResolver(void *) {
  resolutions.fetch_add(1);
  say("resolver called\n");
  return guard::Resolution::Declined;
}

uintptr_t page(const void *p) { return reinterpret_cast<uintptr_t>(p); }

[[noreturn]] void touchAndExit(const volatile char *p, const int code) {
  [[maybe_unused]] const char value = *p;
  _exit(code);
}

} // namespace

TEST_CASE("guard scenario: untracked fault reaches the earlier handler", "[.][guard-scenario]") {
  alarm(10);
  installMarker(ProtectionFault);
  guard::install(countingResolver);
  touchAndExit(mapPages(1, PROT_NONE), 0);
}

TEST_CASE("guard scenario: re-asserting install chains to a later handler", "[.][guard-scenario]") {
  alarm(10);
  signal(ProtectionFault, SIG_DFL);
  guard::install(countingResolver);
  installMarker(ProtectionFault);
  guard::install(countingResolver);
  struct sigaction current{};
  sigaction(ProtectionFault, nullptr, &current);
  if (!(current.sa_flags & SA_SIGINFO) || current.sa_sigaction == markerHandler) fail("guard did not re-assert\n");
  touchAndExit(mapPages(1, PROT_NONE), 0);
}

namespace {
struct sigaction chainedPrior{};

void chainingHandler(const int signal, siginfo_t *info, void *context) {
  say("chained\n");
  if (chainedPrior.sa_flags & SA_SIGINFO) chainedPrior.sa_sigaction(signal, info, context);
}
} // namespace

TEST_CASE("guard scenario: a later handler that chains back is not looped through", "[.][guard-scenario]") {
  alarm(10);
  signal(ProtectionFault, SIG_DFL);
  guard::install(countingResolver);
  struct sigaction action{};
  action.sa_sigaction = chainingHandler;
  action.sa_flags = SA_SIGINFO;
  sigemptyset(&action.sa_mask);
  sigaction(ProtectionFault, &action, &chainedPrior);
  guard::install(countingResolver);
  touchAndExit(mapPages(1, PROT_NONE), 0);
}

TEST_CASE("guard scenario: untracked SIGBUS reaches the earlier handler", "[.][guard-scenario]") {
  alarm(10);
  installMarker(SIGBUS);
  guard::install(countingResolver);
  char name[] = "/tmp/polyrt-guard-XXXXXX";
  const int fd = mkstemp(name);
  if (fd < 0) fail("mkstemp failed\n");
  unlink(name);
  void *p = mmap(nullptr, guard::pageSize(), PROT_READ, MAP_SHARED, fd, 0);
  if (p == MAP_FAILED) fail("mmap failed\n");
  touchAndExit(static_cast<const char *>(p), 0);
}

TEST_CASE("guard scenario: untracked fault without a handler takes the default action", "[.][guard-scenario]") {
  alarm(10);
  signal(ProtectionFault, SIG_DFL);
  guard::install(countingResolver);
  touchAndExit(mapPages(1, PROT_NONE), 0);
}

namespace {
std::atomic<std::thread::id> resolverThread{};
char *errnoRegion = nullptr;

guard::Resolution fillingResolver(void *address) {
  resolverThread.store(std::this_thread::get_id());
  const size_t bytes = 4 * guard::pageSize();
  if (page(address) < page(errnoRegion) || page(address) >= page(errnoRegion) + bytes) return guard::Resolution::Declined;
  guard::protect(errnoRegion, bytes, guard::Access::ReadWrite);
  std::memset(errnoRegion, 0x5A, bytes);
  errno = EIO;
  return guard::Resolution::Resolved;
}
} // namespace

TEST_CASE("guard scenario: tracked fault resolves off the faulting thread and keeps its errno", "[.][guard-scenario]") {
  alarm(10);
  errnoRegion = mapPages(4, PROT_NONE);
  const auto slot = guard::track(page(errnoRegion), page(errnoRegion) + 4 * guard::pageSize());
  if (!slot) fail("track failed\n");
  guard::install(fillingResolver);
  errno = ERANGE;
  const char value = *static_cast<volatile char *>(errnoRegion + guard::pageSize() + 7);
  const int observed = errno;
  if (value != 0x5A) fail("fault did not resolve to the resolver's data\n");
  if (observed != ERANGE) fail("errno changed across the fault\n");
  if (resolverThread.load() == std::this_thread::get_id()) fail("fault resolved on the faulting thread\n");
  guard::untrack(*slot);
  _exit(0);
}

namespace {
constexpr size_t ConcurrentRegions = 32, ConcurrentPages = 16, ConcurrentThreads = 16;
std::vector<char *> concurrentRegions;
std::atomic<size_t> concurrentResolutions{0};

guard::Resolution regionResolver(void *address) {
  const size_t bytes = ConcurrentPages * guard::pageSize();
  for (size_t i = 0; i < concurrentRegions.size(); ++i) {
    char *region = concurrentRegions[i];
    if (page(address) < page(region) || page(address) >= page(region) + bytes) continue;
    if (!guard::publish(region, bytes, guard::Access::Read, [&](void *staging) { std::memset(staging, static_cast<int>(i + 1), bytes); }))
      fail("publish failed\n");
    concurrentResolutions.fetch_add(1);
    return guard::Resolution::Resolved;
  }
  return guard::Resolution::Declined;
}
} // namespace

TEST_CASE("guard scenario: concurrent faults from many threads all resolve", "[.][guard-scenario]") {
  alarm(20);
  for (size_t i = 0; i < ConcurrentRegions; ++i) {
    char *region = mapPages(ConcurrentPages, PROT_NONE);
    if (!guard::track(page(region), page(region) + ConcurrentPages * guard::pageSize())) fail("track failed\n");
    concurrentRegions.push_back(region);
  }
  guard::install(regionResolver);
  std::atomic<size_t> wrong{0};
  std::vector<std::thread> threads;
  for (size_t t = 0; t < ConcurrentThreads; ++t)
    threads.emplace_back([&, t] {
      for (size_t k = 0; k < ConcurrentRegions; ++k) {
        const size_t i = (k + t) % ConcurrentRegions;
        const auto *region = static_cast<volatile char *>(concurrentRegions[i]);
        for (size_t p = 0; p < ConcurrentPages; ++p)
          if (region[p * guard::pageSize() + t] != static_cast<char>(i + 1)) wrong.fetch_add(1);
      }
    });
  for (auto &thread : threads)
    thread.join();
  if (wrong.load() != 0) fail("a thread read unresolved data\n");
  if (concurrentResolutions.load() < ConcurrentRegions) fail("fewer resolutions than regions\n");
  _exit(0);
}

namespace {
char *goneRegion = nullptr, *declinedRegion = nullptr;

guard::Resolution goneOrDeclined(void *address) {
  if (page(address) >= page(goneRegion) && page(address) < page(goneRegion) + guard::pageSize()) {
    guard::protect(goneRegion, guard::pageSize(), guard::Access::ReadWrite);
    return guard::Resolution::Gone;
  }
  return guard::Resolution::Declined;
}
} // namespace

TEST_CASE("guard scenario: gone retries the access and declined forwards it", "[.][guard-scenario]") {
  alarm(10);
  installMarker(ProtectionFault);
  goneRegion = mapPages(1, PROT_NONE);
  declinedRegion = mapPages(1, PROT_NONE);
  if (!guard::track(page(goneRegion), page(goneRegion) + guard::pageSize())) fail("track failed\n");
  if (!guard::track(page(declinedRegion), page(declinedRegion) + guard::pageSize())) fail("track failed\n");
  guard::install(goneOrDeclined);
  [[maybe_unused]] const char value = *static_cast<volatile char *>(goneRegion);
  say("gone retried\n");
  touchAndExit(declinedRegion, 0);
}

TEST_CASE("guard scenario: guarded fault in a forked child aborts with a diagnostic", "[.][guard-scenario]") {
  alarm(10);
  char *region = mapPages(1, PROT_NONE);
  if (!guard::track(page(region), page(region) + guard::pageSize())) fail("track failed\n");
  guard::install(countingResolver);
  const pid_t pid = fork();
  if (pid == 0) touchAndExit(region, 0);
  int status = 0;
  while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {
  }
  if (!WIFSIGNALED(status) || WTERMSIG(status) != SIGABRT) fail("forked child did not abort\n");
  if (resolutions.load() != 0) fail("parent resolver served the child's fault\n");
  _exit(0);
}

namespace {
std::mutex stallMutex;
char *stallRegion = nullptr;

guard::Resolution stallingResolver(void *) {
  const std::lock_guard lock(stallMutex);
  return guard::Resolution::Resolved;
}
} // namespace

TEST_CASE("guard scenario: a resolution blocked by the faulting thread is reported", "[.][guard-scenario]") {
  alarm(3);
  guard::details::stallReportMillis = 100;
  stallRegion = mapPages(1, PROT_NONE);
  if (!guard::track(page(stallRegion), page(stallRegion) + guard::pageSize())) fail("track failed\n");
  guard::install(stallingResolver);
  const std::lock_guard lock(stallMutex);
  touchAndExit(stallRegion, 0);
}

TEST_CASE("page guard forwards untracked faults to the earlier handler") {
  const auto outcome = runScenario("guard scenario: untracked fault reaches the earlier handler");
  INFO(outcome.output);
  CHECK(outcome.exitCode == markerExit(ProtectionFault));
  CHECK(outcome.output.find("resolver called") == std::string::npos);
}

TEST_CASE("page guard re-asserts itself and chains to a later handler") {
  const auto outcome = runScenario("guard scenario: re-asserting install chains to a later handler");
  INFO(outcome.output);
  CHECK(outcome.exitCode == markerExit(ProtectionFault));
}

TEST_CASE("page guard does not loop through a later handler that chains back to it") {
  const auto outcome = runScenario("guard scenario: a later handler that chains back is not looped through");
  size_t chained = 0;
  for (size_t at = outcome.output.find("chained"); at != std::string::npos; at = outcome.output.find("chained", at + 1))
    ++chained;
  INFO(outcome.output.substr(0, 512));
  CHECK(chained == 1);
  CHECK(outcome.signal == ProtectionFault);
}

TEST_CASE("page guard forwards untracked SIGBUS") {
  const auto outcome = runScenario("guard scenario: untracked SIGBUS reaches the earlier handler");
  INFO(outcome.output);
  CHECK(outcome.exitCode == 43);
}

TEST_CASE("page guard leaves untracked faults to the default action") {
  const auto outcome = runScenario("guard scenario: untracked fault without a handler takes the default action");
  INFO(outcome.output);
  CHECK(outcome.signal == ProtectionFault);
}

TEST_CASE("page guard resolves tracked faults off the faulting thread without clobbering errno") {
  const auto outcome = runScenario("guard scenario: tracked fault resolves off the faulting thread and keeps its errno");
  INFO(outcome.output);
  CHECK(outcome.exitCode == 0);
}

TEST_CASE("page guard resolves concurrent faults from many threads") {
  const auto outcome = runScenario("guard scenario: concurrent faults from many threads all resolve");
  INFO(outcome.output);
  CHECK(outcome.exitCode == 0);
}

TEST_CASE("page guard retries gone faults and forwards declined ones") {
  const auto outcome = runScenario("guard scenario: gone retries the access and declined forwards it");
  INFO(outcome.output);
  CHECK(outcome.exitCode == markerExit(ProtectionFault));
  CHECK(outcome.output.find("gone retried") != std::string::npos);
}

TEST_CASE("page guard aborts a forked child that touches a guarded page") {
  const auto outcome = runScenario("guard scenario: guarded fault in a forked child aborts with a diagnostic");
  INFO(outcome.output);
  CHECK(outcome.exitCode == 0);
  CHECK(outcome.output.find("forked child") != std::string::npos);
}

TEST_CASE("page guard reports a resolution the faulting thread blocks") {
  const auto outcome = runScenario("guard scenario: a resolution blocked by the faulting thread is reported");
  INFO(outcome.output);
  CHECK(outcome.signal == SIGALRM);
  CHECK(outcome.output.find("still resolving a guarded page fault at 0x") != std::string::npos);
}

#endif
