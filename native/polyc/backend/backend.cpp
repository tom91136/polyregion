#include "backend.h"

#include "fmt/core.h"

polyregion::backend::BackendException::BackendException(const char *what, const char *file, const char *fn, const int line)
    : BackendException(std::string(what), file, fn, line) {}

polyregion::backend::BackendException::BackendException(const std::string &what, const char *file, const char *fn, const int line)
    : std::logic_error("[Backend] In " + std::string(file) + ":" + std::to_string(line) + " (" + std::string(fn) + ")\n" + what) {}

void polyregion::backend::requireGroupNative(const compiletime::Target target, const std::string &arch, const compiletime::GroupKind kind,
                                             const polyast::AtomicOp::Any &op, const polyast::Type::Any &type) {
  const auto opName = polyast::repr(op);
  const auto typeName = polyast::canonicalName(type);
  if (!compiletime::groupNative(target, arch, kind, opName, typeName))
    throw BackendException(fmt::format("work-group {} {} on {} is not native, SubgroupLower must lower it",
                                       compiletime::groupKindName(kind), opName, typeName));
}
