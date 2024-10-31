#include "backend.h"

polyregion::backend::BackendException::BackendException(const char *what, const char *file, const char *fn, const int line)
    : BackendException(std::string(what), file, fn, line) {}

polyregion::backend::BackendException::BackendException(const std::string &what, const char *file, const char *fn, const int line)
    : std::logic_error("[Backend] In " + std::string(file) + ":" + std::to_string(line) + " (" + std::string(fn) + ")\n" + what) {}
