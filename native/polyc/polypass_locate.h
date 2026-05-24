#pragma once

#include <optional>
#include <string>
#include <string_view>

#include "polyregion/aliases.h"
#include "polyregion/export.h"

namespace polyregion::polypass {

enum class PluginKind { Js, Dso };

struct PluginRef {
  String path;
  PluginKind kind;
};

POLYREGION_EXPORT std::optional<PluginKind> pluginKindFor(std::string_view path);
POLYREGION_EXPORT Vector<PluginRef> resolvePlugins(String &error);

} // namespace polyregion::polypass
