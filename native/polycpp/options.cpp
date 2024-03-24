#include "options.h"

#include "aspartame/string.hpp"
#include "aspartame/variant.hpp"
#include "aspartame/vector.hpp"

std::
    variant<                                                                                    //
        std::vector<std::string>,                                                               //
        std::pair<std::vector<const char *>, std::optional<polyregion::polystl::StdParOptions>> //
        >
    polyregion::polystl::StdParOptions::stripAndParse(const std::vector<const char *> &args) {

  using namespace aspartame;

  auto parseEqFlags = [](const char *arg, const std::string &argNameWithEq) -> std::optional<std::string> {
    if (!(arg ^ starts_with(argNameWithEq))) return std::nullopt;
    else return arg ^ drop(argNameWithEq.length());
  };

  auto markError = [](std::vector<std::string> &drain, const std::string &prefix) {
    return [&](const std::string &x) { drain.push_back("\"" + prefix + "\": " + x); };
  };

  const std::string fStdParFlag = "-fstdpar";
  const std::string fStdParQuietFlag = "-fstdpar-quiet";
  const std::string fStdParArchNoCompressFlag = "-fstdpar-no-compress";
  const std::string fStdParInterposeMallocFlag = "-fstdpar-interpose-malloc";
  const std::string fStdParInterposeAllocaFlag = "-fstdpar-interpose-alloca";
  const std::string fStdParArchFlag = "-fstdpar-arch=";
  const std::string fStdParRtFlag = "-fstdpar-rt=";
  const std::string fStdParJitFlag = "-fstdpar-jit=";

  auto fStdPar = false, fStdParDependents = false;
  StdParOptions options;
  std::vector<std::string> errors;
  auto stripped =
      args ^ filter([&](auto arg) {
        if (arg == fStdParFlag) {
          fStdPar = true;
          return false;
        }
        if (arg == fStdParQuietFlag) {
          fStdParDependents = true;
          options.quiet = true;
          return false;
        }
        if (arg == fStdParArchNoCompressFlag) {
          fStdParDependents = true;
          options.noCompress = true;
          return false;
        }
        if (arg == fStdParInterposeMallocFlag) {
          fStdParDependents = true;
          options.interposeMalloc = true;
          return false;
        }
        if (arg == fStdParInterposeAllocaFlag) {
          fStdParDependents = true;
          options.interposeAlloca = true;
          return false;
        }
        if (auto arch = parseEqFlags(arg, fStdParArchFlag)) {
          fStdParDependents = true;
          parseTargets(*arch) ^ foreach_total(markError(errors, fStdParArchFlag), [&](const Targets &xs) { options.targets = xs; });
          return false;
        }
        if (auto rt = parseEqFlags(arg, fStdParRtFlag)) {
          fStdParDependents = true;
          parseLinkKind(*rt) ^ foreach_total(markError(errors, fStdParRtFlag), [&](const LinkKind &x) { options.rt = x; });
          return false;
        }
        if (auto jit = parseEqFlags(arg, fStdParJitFlag)) {
          fStdParDependents = true;
          parseLinkKind(*jit) ^ foreach_total(markError(errors, fStdParJitFlag), [&](const LinkKind &x) { options.jit = x; });
          return false;
        }
        return true;
      });

  if (!fStdPar && fStdParDependents)
    errors.insert(errors.begin(), fStdParFlag + " not specified but StdPar dependent flags used, pleased add " + fStdParFlag);

  if (errors.empty()) {
    return fStdPar ? std::pair{stripped, std::optional{options}} : std::pair{stripped, std::nullopt};
  } else {
    return errors;
  }
}

std::variant<std::string, polyregion::polystl::StdParOptions::LinkKind>
polyregion::polystl::StdParOptions::parseLinkKind(const std::string &arg) {
  using namespace aspartame;
  if (auto v = arg ^ to_lower(); v == "static") return LinkKind::Static;
  else if (v == "dynamic") return LinkKind::Dynamic;
  else if (v == "disabled") return LinkKind::Disabled;
  return "Unknown link kind `" + arg + "`";
}
std::variant<std::string, polyregion::polystl::StdParOptions::Targets>
polyregion::polystl::StdParOptions::parseTargets(const std::string &arg) {
  using namespace aspartame;
  StdParOptions::Targets result;
  for (auto &rawArchAndFeaturesCSV : arg ^ split(':')) {
    auto archAndFeaturesCSV = rawArchAndFeaturesCSV ^ split('@');
    if (archAndFeaturesCSV.size() != 2) {
      return "Missing or invalid placement of arch and feature separator '@' in `" + rawArchAndFeaturesCSV + "`";
    }
    if (auto t = polyregion::compiletime::parseTarget(archAndFeaturesCSV[0]); t) {
      for (auto &feature : archAndFeaturesCSV[1] ^ split(','))
        result.emplace_back(*t, feature);
    } else return "Unknown arch `" + archAndFeaturesCSV[0] + "`";
  }
  return result;
}
