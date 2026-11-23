#pragma once

#include <memory>

#include "llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h"

namespace polyregion::invoke::object {

std::unique_ptr<llvm::orc::LinkGraphLinkingLayer::Plugin> makeCoffPlugin();

} // namespace polyregion::invoke::object
