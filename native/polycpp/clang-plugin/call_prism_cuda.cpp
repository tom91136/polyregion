#include <string_view>

#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"

#include "call_prism_internal.hpp"

namespace polyregion::polystl::call_prism {

using namespace aspartame;

namespace {

constexpr std::string_view cubShuffleDownLegacy = "cub::SHFL_DOWN_SYNC";
constexpr std::string_view cubShuffleUpLegacy = "cub::SHFL_UP_SYNC";
constexpr std::string_view cubVaPrintf = "cub::va_printf";
constexpr std::string_view thrustThrowOnError = "thrust::cuda_cub::throw_on_error";
constexpr std::string_view thrustTerminateWithMessage = "thrust::system::cuda::detail::terminate_with_message";
constexpr std::string_view cubShuffleDown = "cub::ShuffleDown";
constexpr std::string_view cubShuffleUp = "cub::ShuffleUp";
constexpr std::string_view cubShuffleIndex = "cub::ShuffleIndex";
constexpr std::string_view cubWarpAny = "cub::WARP_ANY";
constexpr std::string_view cubWarpAll = "cub::WARP_ALL";
constexpr std::string_view cubWarpBallot = "cub::WARP_BALLOT";
constexpr std::string_view cubWarpSync = "cub::WARP_SYNC";
constexpr std::string_view cubLaneId = "cub::LaneId";
constexpr std::string_view cubWarpSize = "cub::WarpSize";
constexpr std::string_view cubLaneMaskGe = "cub::LaneMaskGe";
constexpr std::string_view cubPtxVersion = "cub::PtxVersion";
constexpr std::string_view cubPtxVersionUncached = "cub::PtxVersionUncached";
constexpr std::string_view cubMaxSmOccupancy = "cub::MaxSmOccupancy";
constexpr std::string_view cubThreadLoad = "cub::ThreadLoad";
constexpr std::string_view cubThreadStore = "cub::ThreadStore";
constexpr std::string_view thrustUninitializedCopy = "thrust::cuda_cub::uninitialized_copy_n";

[[nodiscard]] bool shuffleDown(const std::string_view name) { return name == cubShuffleDownLegacy || name == cubShuffleDown; }
[[nodiscard]] bool shuffleUp(const std::string_view name) { return name == cubShuffleUpLegacy || name == cubShuffleUp; }

[[nodiscard]] Opt<uint32_t> firstIntegralTemplateArgument(const clang::FunctionDecl &decl) {
  const auto *arguments = decl.getTemplateSpecializationArgs();
  if (!arguments) return {};
  for (const auto &argument : arguments->asArray())
    if (argument.getKind() == clang::TemplateArgument::Integral) return static_cast<uint32_t>(argument.getAsIntegral().getZExtValue());
  return {};
}

[[nodiscard]] bool validLogicalWarpWidth(const uint64_t width) { return width > 0 && width <= 32 && (width & (width - 1)) == 0; }

[[nodiscard]] Opt<uint32_t> warpScanWidth(const clang::FunctionDecl &decl) {
  const auto *specialisation = llvm::dyn_cast<clang::ClassTemplateSpecializationDecl>(decl.getParent());
  if (!specialisation || specialisation->getTemplateArgs().size() < 2
      || specialisation->getTemplateArgs().get(1).getKind() != clang::TemplateArgument::Integral)
    return {};
  const auto width = specialisation->getTemplateArgs().get(1).getAsIntegral().getZExtValue();
  if (!validLogicalWarpWidth(width)) return {};
  return static_cast<uint32_t>(width);
}

Expr::Any boundedShuffle(Remapper &self, Remapper::RemapContext &r, const bool down, Term::Any value, Term::Any delta, Term::Any boundary,
                         Term::Any width, Term::Any mask, const Type::Any &result) {
  delta = r.newVar(self.conform(r, Expr::Alias(delta), Type::IntU32()));
  boundary = r.newVar(self.conform(r, Expr::Alias(boundary), Type::IntU32()));
  width = r.newVar(self.conform(r, Expr::Alias(width), Type::IntU32()));
  mask = r.newVar(self.conform(r, Expr::Alias(mask), Type::IntU32()));
  const auto physicalLane = r.newVar(Expr::SpecOp(Spec::GpuLaneIdx()));
  const auto lane = r.newVar(Expr::IntrOp(Intr::BAnd(physicalLane, width, Type::IntU32())));
  const auto threshold = r.newVar(Expr::IntrOp(Intr::Add(down ? lane : boundary, delta, Type::IntU32())));
  const auto valid =
      r.newVar(Expr::IntrOp(down ? Intr::Any(Intr::LogicLte(threshold, boundary)) : Intr::Any(Intr::LogicGte(lane, threshold))));
  const auto shuffled = r.newVar(Expr::SpecOp(down ? Spec::Any(Spec::GpuShuffleDown(value, delta, width, mask, result))
                                                   : Spec::Any(Spec::GpuShuffleUp(value, delta, width, mask, result))));
  const auto output = r.newName(result);
  r.push(Stmt::Var(output, Expr::Alias(value), true));
  r.push(Stmt::Cond(valid, {Stmt::Mut(Term::Select(output, {}, result), Expr::Alias(shuffled))}, {}));
  return Expr::Alias(Term::Select(output, {}, result));
}

[[nodiscard]] bool isPtxVersionQuery(const std::string_view name) { return name == cubPtxVersion || name == cubPtxVersionUncached; }

[[nodiscard]] bool isOccupancyQuery(const std::string_view name) {
  return name == cubMaxSmOccupancy || name == "cudaOccupancyMaxActiveBlocksPerMultiprocessor";
}

[[nodiscard]] Opt<std::string> deviceAttributeName(const clang::Expr &expression) {
  const auto *reference = llvm::dyn_cast<clang::DeclRefExpr>(expression.IgnoreParenImpCasts());
  const auto *constant = reference ? llvm::dyn_cast<clang::EnumConstantDecl>(reference->getDecl()) : nullptr;
  return constant ? Opt<std::string>{constant->getNameAsString()} : std::nullopt;
}

struct BitOps {
  Remapper::RemapContext &r;
  Type::Any type;
  Term::Any add(const Term::Any &a, const Term::Any &b) const { return r.newVar(Expr::IntrOp(Intr::Add(a, b, type))); }
  Term::Any sub(const Term::Any &a, const Term::Any &b) const { return r.newVar(Expr::IntrOp(Intr::Sub(a, b, type))); }
  Term::Any mul(const Term::Any &a, const Term::Any &b) const { return r.newVar(Expr::IntrOp(Intr::Mul(a, b, type))); }
  Term::Any band(const Term::Any &a, const Term::Any &b) const { return r.newVar(Expr::IntrOp(Intr::BAnd(a, b, type))); }
  Term::Any bor(const Term::Any &a, const Term::Any &b) const { return r.newVar(Expr::IntrOp(Intr::BOr(a, b, type))); }
  Term::Any shl(const Term::Any &a, const Term::Any &b) const { return r.newVar(Expr::IntrOp(Intr::BSL(a, b, type))); }
  Term::Any shr(const Term::Any &a, const Term::Any &b) const { return r.newVar(Expr::IntrOp(Intr::BZSR(a, b, type))); }
};

static Term::Any populationCount(Remapper::RemapContext &r, const Type::Any &type, Term::Any value) {
  const BitOps op{r, type};
  const auto constant = [&](const uint64_t x) { return r.newVar(Remapper::integralConstOfType(type, x)); };
  const auto bits = static_cast<uint64_t>(primitiveSize(type).value_or(4) * 8);
  value = op.sub(value, op.band(op.shr(value, constant(1)), constant(0x5555555555555555ull)));
  value = op.add(op.band(value, constant(0x3333333333333333ull)), op.band(op.shr(value, constant(2)), constant(0x3333333333333333ull)));
  value = op.band(op.add(value, op.shr(value, constant(4))), constant(0x0f0f0f0f0f0f0f0full));
  return op.shr(op.mul(value, constant(0x0101010101010101ull)), constant(bits - 8));
}

} // namespace

static Opt<MatchedCall> cubLegacyShuffle(const clang::CallExpr &call, const clang::FunctionDecl &decl) {
  const auto name = decl.getQualifiedNameAsString();
  if (call.getNumArgs() != 4 || (!shuffleDown(name) && !shuffleUp(name))) return {};
  const auto down = shuffleDown(name);
  const auto legacy = name == cubShuffleDownLegacy || name == cubShuffleUpLegacy;
  const auto logicalWidth = legacy ? Opt<uint32_t>{} : firstIntegralTemplateArgument(decl);
  if (!legacy && (!logicalWidth || !validLogicalWarpWidth(*logicalWidth))) raise("CUB shuffle has no valid logical-warp width");
  const auto *expression = &call;
  return MatchedCall{Lowering{[expression, down, legacy, logicalWidth](Remapper &self, Remapper::RemapContext &r) -> Expr::Any {
                       const auto args = lowerArguments(*expression, self, r);
                       const auto tpe = self.handleType(expression->getType(), r);
                       if (!legacy)
                         return boundedShuffle(self, r, down, args[0], args[1], args[2], Term::IntU32Const(*logicalWidth - 1), args[3],
                                               tpe);
                       const auto control = r.newVar(self.conform(r, Expr::Alias(args[2]), Type::IntU32()));
                       const auto shifted = r.newVar(Expr::IntrOp(Intr::BZSR(control, Term::IntU32Const(8), Type::IntU32())));
                       const auto inverted = r.newVar(Expr::IntrOp(Intr::BNot(shifted, Type::IntU32())));
                       const auto width = r.newVar(Expr::IntrOp(Intr::BAnd(inverted, Term::IntU32Const(31), Type::IntU32())));
                       const auto boundary = r.newVar(Expr::IntrOp(Intr::BAnd(control, Term::IntU32Const(31), Type::IntU32())));
                       return boundedShuffle(self, r, down, args[0], args[1], boundary, width, args[3], tpe);
                     }},
                     false};
}

static Opt<MatchedCall> cubThreadAccess(const clang::CallExpr &call, const clang::FunctionDecl &decl) {
  const auto name = decl.getQualifiedNameAsString();
  const bool load = name == cubThreadLoad;
  const bool store = name == cubThreadStore;
  if (!load && !store) return {};
  const auto *expression = &call;
  return MatchedCall{Lowering{[expression, load](Remapper &self, Remapper::RemapContext &r) -> Expr::Any {
                       const auto arguments = lowerArguments(*expression, self, r);
                       const auto result = self.handleType(expression->getType(), r);
                       Opt<size_t> pointerIndex;
                       Opt<size_t> valueIndex;
                       for (size_t i = 0; i < arguments.size(); ++i)
                         if (const auto pointer = arguments[i].tpe().get<Type::Ptr>();
                             pointer && !pointerIndex && (load ? pointer->comp == result : true))
                           pointerIndex = i;
                       if (pointerIndex && !load) {
                         const auto pointee = arguments[*pointerIndex].tpe().get<Type::Ptr>()->comp;
                         for (size_t i = 0; i < arguments.size(); ++i)
                           if (i != *pointerIndex && arguments[i].tpe() == pointee) {
                             valueIndex = i;
                             break;
                           }
                       }
                       if (!pointerIndex || (!load && !valueIndex)) raise("CUB thread access has no compatible pointer/value arguments");
                       if (load) return Expr::SpecOp(Spec::GpuVolatileLoad(arguments[*pointerIndex], result));
                       return Expr::SpecOp(Spec::GpuVolatileStore(arguments[*pointerIndex], arguments[*valueIndex]));
                     }},
                     false};
}

static Opt<MatchedCall> cudaAtomic(const clang::CallExpr &call, const clang::FunctionDecl &decl) {
  auto name = decl.getQualifiedNameAsString();
  MemScope::Any scope = MemScope::Device();
  if (name.ends_with("_block")) {
    scope = MemScope::Workgroup();
    name.resize(name.size() - 6);
  } else if (name.ends_with("_system")) {
    scope = MemScope::System();
    name.resize(name.size() - 7);
  }
  const static Map<std::string, AtomicOp::Any> operations{
      {"atomicAdd", AtomicOp::Add()}, {"atomicSub", AtomicOp::Sub()}, {"atomicMin", AtomicOp::Min()}, {"atomicMax", AtomicOp::Max()},
      {"atomicAnd", AtomicOp::And()}, {"atomicOr", AtomicOp::Or()},   {"atomicXor", AtomicOp::Xor()}, {"atomicExch", AtomicOp::Xchg()}};
  const auto operation = operations ^ get_maybe(name);
  const bool compareExchange = name == "atomicCAS";
  if ((!operation || call.getNumArgs() != 2) && (!compareExchange || call.getNumArgs() != 3)) return {};
  const auto *expression = &call;
  return MatchedCall{Lowering{[expression, operation, compareExchange, scope](Remapper &self, Remapper::RemapContext &r) -> Expr::Any {
                       const auto arguments = lowerArguments(*expression, self, r);
                       const auto result = self.handleType(expression->getType(), r);
                       if (compareExchange)
                         return Expr::SpecOp(
                             Spec::GpuAtomicCAS(arguments[0], arguments[1], arguments[2], scope, MemOrder::Relaxed(), result));
                       return Expr::SpecOp(Spec::GpuAtomicRMW(*operation, arguments[0], arguments[1], scope, MemOrder::Relaxed(), result));
                     }},
                     false};
}

static Opt<MatchedCall> cudaBitOperation(const clang::CallExpr &call, const clang::FunctionDecl &decl) {
  const auto name = decl.getQualifiedNameAsString();
  const bool reverse = name == "__nv_brev";
  const bool leadingZeros = name == "__nv_clz";
  const bool count = name == "__nv_popc" || name == "__nv_popcll";
  const bool trailingZeros = name == "__builtin_ctz" || name == "__builtin_ctzll";
  if ((!reverse && !leadingZeros && !count && !trailingZeros) || call.getNumArgs() != 1) return {};
  const auto *expression = &call;
  return MatchedCall{
      Lowering{[expression, reverse, leadingZeros, count](Remapper &self, Remapper::RemapContext &r) -> Expr::Any {
        const auto wide = expression->getArg(0)->getType()->isIntegerType()
                          && self.context.getTypeSizeInChars(expression->getArg(0)->getType()).getQuantity() == 8;
        const auto type = wide ? Type::IntU64().widen() : Type::IntU32().widen();
        const BitOps op{r, type};
        const auto constant = [&](const uint64_t x) { return r.newVar(Remapper::integralConstOfType(type, x)); };
        auto value = r.newVar(Expr::Cast(r.newVar(self.handleExpr(expression->getArg(0), r)), type));
        if (reverse) {
          for (const auto [shift, mask] : {Pair<uint64_t, uint64_t>{1, 0x55555555u}, {2, 0x33333333u}, {4, 0x0f0f0f0fu}, {8, 0x00ff00ffu}})
            value =
                op.bor(op.band(op.shr(value, constant(shift)), constant(mask)), op.shl(op.band(value, constant(mask)), constant(shift)));
          value = op.bor(op.shr(value, constant(16)), op.shl(value, constant(16)));
        } else if (leadingZeros) {
          for (const uint64_t shift : {1, 2, 4, 8, 16})
            value = op.bor(value, op.shr(value, constant(shift)));
          value = op.sub(constant(32), populationCount(r, type, value));
        } else if (count) {
          value = populationCount(r, type, value);
        } else {
          const auto lowBit = op.band(value, op.sub(constant(0), value));
          value = populationCount(r, type, op.sub(lowBit, constant(1)));
        }
        return self.conform(r, Expr::Alias(value), self.handleType(expression->getType(), r));
      }},
      false};
}

static Opt<MatchedCall> thrustRelocate(const clang::CallExpr &call, const clang::FunctionDecl &decl) {
  if (decl.getQualifiedNameAsString() != thrustUninitializedCopy || call.getNumArgs() != 4 || !call.getArg(1)->getType()->isPointerType())
    return {};
  const auto *policy = call.getArg(0)->getType().getCanonicalType()->getAsCXXRecordDecl();
  if (!policy || !policy->getQualifiedNameAsString().starts_with("thrust::cuda_cub::"))
    raise("thrust::uninitialized_copy_n requires a CUDA device execution policy");
  if (call.getArg(3)->getType()->isPointerType()) raise("thrust::uninitialized_copy_n raw destinations have ambiguous memory provenance");
  if (!call.getArg(1)->getType()->getPointeeType().isTriviallyCopyableType(decl.getASTContext()))
    raise("thrust::uninitialized_copy_n requires a trivially-copyable value type");
  const auto *expression = &call;
  return MatchedCall{
      Lowering{[expression](Remapper &self, Remapper::RemapContext &r) -> Expr::Any {
        (void)r.newVar(self.handleExpr(expression->getArg(0), r));
        const auto *sourceExpression = expression->getArg(1);
        const auto *countExpression = expression->getArg(2);
        const auto *destinationExpression = expression->getArg(3);
        const auto elementBytes =
            static_cast<uint64_t>(self.context.getTypeSizeInChars(sourceExpression->getType()->getPointeeType()).getQuantity());
        const auto pointerType = self.handleType(sourceExpression->getType(), r);
        const auto pointer = pointerType.get<Type::Ptr>();
        if (!pointer) raise("Thrust relocation source did not lower to a pointer");
        const auto source = r.newVar(self.conform(r, self.handleExpr(sourceExpression, r), pointerType));
        Opt<Named> destinationHolder;
        Opt<Term::Select> destinationIterator;
        const Term::Any destination = [&]() -> Term::Any {
          if (destinationExpression->getType()->isPointerType()) {
            const auto destinationType = self.handleType(destinationExpression->getType(), r);
            if (destinationType != pointerType) raise("Thrust relocation requires identical source and destination pointer types");
            return r.newVar(self.handleExpr(destinationExpression, r));
          }
          const auto holder = r.newName(self.handleType(destinationExpression->getType(), r));
          r.push(Stmt::Var(holder, self.handleExpr(destinationExpression, r), false));
          const auto root = holder.tpe.get<Type::Struct>();
          if (!root) raise("Thrust relocation output is not a struct or pointer");
          Set<std::string> seen;
          std::function<Opt<Vector<Named>>(const Type::Struct &)> findIterator = [&](const Type::Struct &structure) -> Opt<Vector<Named>> {
            if (!seen.emplace(fqcn(structure.name)).second) return {};
            const auto definition = r.findStruct(fqcn(structure.name), "Thrust relocation output");
            if (const auto member = definition->members ^ find([](const auto &candidate) {
                                      return candidate.symbol == "m_iterator" || candidate.symbol.ends_with("::m_iterator");
                                    }))
              return Vector<Named>{*member};
            for (const auto &member : definition->members)
              if (const auto nested = member.tpe.get<Type::Struct>())
                if (const auto path = findIterator(*nested)) return Vector<Named>{member} ^ concat(*path);
            return {};
          };
          const auto path = findIterator(*root);
          if (!path) raise("Thrust relocation output has no m_iterator field");
          destinationHolder = holder;
          auto prefix = Vector<Named>{holder} | concat(*path | take(path->size() - 1)) | to_vector();
          destinationIterator = self.selectPath(r, prefix, path->back());
          if (destinationIterator->tpe != pointerType)
            raise("Thrust relocation requires identical source and destination iterator pointer types");
          return r.newVar(Expr::Alias(*destinationIterator));
        }();
        const auto count = r.newVar(self.conform(r, self.handleExpr(countExpression, r), Type::IntS64()));
        const auto safeCount =
            r.newVar(Expr::IntrOp(Intr::Max(count, r.newVar(Remapper::integralConstOfType(Type::IntS64(), 0)), Type::IntS64())));
        const auto bytes = r.newVar(
            Expr::IntrOp(Intr::Mul(safeCount, r.newVar(Remapper::integralConstOfType(Type::IntS64(), elementBytes)), Type::IntS64())));
        (void)r.newVar(Expr::SpecOp(Spec::RemoteMemcpy(packageContext(), destination, source, r.newVar(Expr::Cast(bytes, Type::IntU64())),
                                                       Direction::RemoteToRemote())));
        const auto advanced =
            r.newVar(Expr::RefTo(termToSelect(destination, r), safeCount, pointer->comp, pointer->space, Region::Opaque()));
        if (!destinationHolder) return Expr::Alias(advanced);
        r.push(Stmt::Mut(*destinationIterator, Expr::Alias(advanced)));
        return Expr::Alias(Term::Select(*destinationHolder, {}, destinationHolder->tpe));
      }},
      false};
}

static Opt<MatchedCall> cubWarpScan(const clang::CallExpr &call, const clang::FunctionDecl &decl) {
  const auto *member = llvm::dyn_cast<clang::CXXMemberCallExpr>(&call);
  const auto *owner = llvm::dyn_cast<clang::CXXRecordDecl>(decl.getDeclContext());
  const auto ownerName = owner ? owner->getQualifiedNameAsString() : std::string{};
  const auto *operationRecord = call.getNumArgs() > 1 ? call.getArg(1)->getType().getCanonicalType()->getAsCXXRecordDecl() : nullptr;
  const auto *operationSpecialisation = llvm::dyn_cast_or_null<clang::ClassTemplateSpecializationDecl>(operationRecord);
  const auto operationName = operationSpecialisation ? operationSpecialisation->getSpecializedTemplate()->getQualifiedNameAsString()
                             : operationRecord       ? operationRecord->getQualifiedNameAsString()
                                                     : std::string{};
  const bool additive = operationName == "cub::Sum" || (operationName.starts_with("cub::") && operationName.ends_with("::Sum"))
                        || operationName == "cuda::std::plus"
                        || (operationName.starts_with("cuda::std::__") && operationName.ends_with("::plus")) || operationName == "std::plus"
                        || (operationName.starts_with("std::__") && operationName.ends_with("::plus"));
  if (!member || !owner || !ownerName.starts_with("cub::") || owner->getName() != "WarpScanShfl" || decl.getName() != "InclusiveScanStep"
      || (call.getNumArgs() != 4 && call.getNumArgs() != 5))
    return {};
  if (!additive) raise("CUB WarpScan currently supports only the standard additive operation");
  const auto logicalWidth = warpScanWidth(decl);
  if (!logicalWidth) raise("CUB WarpScan has no valid logical-warp width");
  const auto *expression = member;
  return MatchedCall{Lowering{[expression, logicalWidth](Remapper &self, Remapper::RemapContext &r) -> Expr::Any {
                       const auto result = self.handleType(expression->getType(), r);
                       if (!(result.kind().is<TypeKind::Integral>() || result.kind().is<TypeKind::Fractional>()))
                         raise("CUB WarpScan scalar result is not numeric");
                       const auto receiver = r.newVar(self.handleExpr(expression->getImplicitObjectArgument(), r));
                       const auto input = r.newVar(self.handleExpr(expression->getArg(0), r));
                       (void)r.newVar(self.handleExpr(expression->getArg(1), r));
                       const auto firstLane = r.newVar(self.handleExpr(expression->getArg(2), r));
                       const auto offset = r.newVar(self.handleExpr(expression->getArg(3), r));
                       for (unsigned i = 4; i < expression->getNumArgs(); ++i)
                         (void)r.newVar(self.handleExpr(expression->getArg(i), r));
                       const auto selected = receiver.get<Term::Select>();
                       if (!selected) raise("CUB WarpScan receiver is not selectable");
                       const auto field = [&](const std::string_view name) -> Term::Any {
                         const auto structure = selected->tpe.get<Type::Struct>();
                         if (!structure) raise("CUB WarpScan receiver is not a struct");
                         const auto definition = r.findStruct(fqcn(structure->name), "CUB WarpScan field");
                         const auto member = definition->members ^ find([&](const auto &candidate) {
                                               return candidate.symbol == name || candidate.symbol.ends_with(fmt::format("::{}", name));
                                             });
                         if (!member) raise(fmt::format("CUB WarpScan receiver has no {} field", name));
                         auto steps = selected->steps;
                         steps.emplace_back(PathStep::Field(member->symbol));
                         return Term::Select(selected->root, std::move(steps), member->tpe);
                       };
                       const auto mask = field("member_mask");
                       const auto lane = field("lane_id");
                       const auto shuffled =
                           r.newVar(Expr::SpecOp(Spec::GpuShuffleUp(input, offset, Term::IntU32Const(*logicalWidth - 1), mask, result)));
                       const auto sum = r.newVar(Expr::IntrOp(Intr::Add(shuffled, input, result)));
                       const auto bound = r.newVar(Expr::IntrOp(Intr::Add(firstLane, offset, Type::IntS32())));
                       const auto laneSigned = r.newVar(Expr::Cast(lane, Type::IntS32()));
                       const auto below = r.newVar(Expr::IntrOp(Intr::LogicLt(laneSigned, bound)));
                       const auto output = r.newName(result);
                       r.push(Stmt::Var(output, Expr::Alias(input), true));
                       r.push(Stmt::Cond(below, {}, {Stmt::Mut(Term::Select(output, {}, result), Expr::Alias(sum))}));
                       return Expr::Alias(Term::Select(output, {}, result));
                     }},
                     false};
}

static Opt<MatchedCall> cudaWarp(const clang::CallExpr &call, const clang::FunctionDecl &decl) {
  const auto name = decl.getQualifiedNameAsString();
  const bool ballot = (name == "__ballot_sync" || name == "__nvvm_vote_ballot_sync" || name == cubWarpBallot) && call.getNumArgs() == 2;
  const bool any = (name == "__any_sync" || name == "__nvvm_vote_any_sync" || name == cubWarpAny) && call.getNumArgs() == 2;
  const bool all = (name == "__all_sync" || name == "__nvvm_vote_all_sync" || name == cubWarpAll) && call.getNumArgs() == 2;
  const bool barrier = (name == "__syncwarp" || name == "__nvvm_bar_warp_sync" || name == cubWarpSync) && call.getNumArgs() == 1;
  const bool shuffleXor = (name == "__shfl_xor_sync" || name == "__nvvm_shfl_sync_bfly_i32") && call.getNumArgs() == 4;
  const bool shuffleIndex = name == cubShuffleIndex && call.getNumArgs() == 3;
  const bool lane = name == cubLaneId && call.getNumArgs() == 0;
  const bool width = name == cubWarpSize && call.getNumArgs() == 0;
  const bool laneMask = name == cubLaneMaskGe && call.getNumArgs() == 0;
  if (!ballot && !any && !all && !barrier && !shuffleXor && !shuffleIndex && !lane && !width && !laneMask) return {};
  const bool cub = name.starts_with("cub::");
  const auto indexWidth = shuffleIndex ? firstIntegralTemplateArgument(decl) : Opt<uint32_t>{};
  if (shuffleIndex && (!indexWidth || !validLogicalWarpWidth(*indexWidth))) raise("CUB ShuffleIndex has no valid logical-warp width");
  const auto *expression = &call;
  const bool highLevelShuffleXor = name == "__shfl_xor_sync";
  return MatchedCall{Lowering{[expression, ballot, any, all, barrier, shuffleXor, highLevelShuffleXor, shuffleIndex, lane, width, cub,
                               indexWidth](Remapper &self, Remapper::RemapContext &r) -> Expr::Any {
                       const auto arguments = lowerArguments(*expression, self, r);
                       if (barrier) return Expr::SpecOp(Spec::GpuSubgroupBarrier(arguments[0]));
                       if (lane) {
                         const auto physical = r.newVar(Expr::SpecOp(Spec::GpuLaneIdx()));
                         return Expr::IntrOp(Intr::BAnd(physical, Term::IntU32Const(31), Type::IntU32()));
                       }
                       if (width) return Expr::Alias(Term::IntU32Const(32));
                       if (ballot || any || all) {
                         const auto mask = cub ? arguments[1] : arguments[0];
                         const auto predicate = cub ? arguments[0] : arguments[1];
                         if (ballot) return Expr::SpecOp(Spec::GpuBallot(mask, predicate));
                         if (any) return Expr::SpecOp(Spec::GpuVoteAny(mask, predicate));
                         return Expr::SpecOp(Spec::GpuVoteAll(mask, predicate));
                       }
                       if (shuffleXor) {
                         const auto control = r.newVar(self.conform(r, Expr::Alias(arguments[3]), Type::IntU32()));
                         const auto clamp = highLevelShuffleXor
                                                ? r.newVar(Expr::IntrOp(Intr::Sub(control, Term::IntU32Const(1), Type::IntU32())))
                                                : control;
                         return Expr::SpecOp(Spec::GpuShuffleXor(arguments[1], arguments[2], clamp, arguments[0],
                                                                 self.handleType(expression->getType(), r)));
                       }
                       if (shuffleIndex)
                         return Expr::SpecOp(Spec::GpuShuffleIdx(arguments[0], arguments[1], Term::IntU32Const(*indexWidth - 1),
                                                                 arguments[2], self.handleType(expression->getType(), r)));
                       const auto physicalLane = r.newVar(Expr::SpecOp(Spec::GpuLaneIdx()));
                       const auto laneIndex = r.newVar(Expr::IntrOp(Intr::BAnd(physicalLane, Term::IntU32Const(31), Type::IntU32())));
                       return Expr::IntrOp(Intr::BSL(Term::IntU32Const(0xFFFFFFFFu), laneIndex, Type::IntU32()));
                     }},
                     false};
}

static Opt<MatchedCall> cudaBarrier(const clang::CallExpr &call, const clang::FunctionDecl &decl) {
  const auto name = decl.getQualifiedNameAsString();
  const bool workgroup = name == "__syncthreads" || name == "__nvvm_barrier_sync";
  const bool localFence = name == "__threadfence_block";
  const bool globalFence = name == "__threadfence";
  const bool allFence = name == "__threadfence_system";
  const bool countedBarrier = name == "__barrier_sync_count" || name == "__nvvm_barrier_sync_cnt";
  if (!workgroup && !localFence && !globalFence && !allFence && !countedBarrier) return {};
  if (countedBarrier) raise("Counted CUDA barriers are not supported in package code");
  if (call.getNumArgs() != 0) raise("CUDA barrier or fence has an unsupported arity");
  const auto *expression = &call;
  return MatchedCall{Lowering{[expression, localFence, globalFence, allFence](Remapper &self, Remapper::RemapContext &r) -> Expr::Any {
                       (void)lowerArguments(*expression, self, r);
                       if (localFence) return Expr::SpecOp(Spec::GpuFenceLocal());
                       if (globalFence) return Expr::SpecOp(Spec::GpuFenceGlobal());
                       if (allFence) return Expr::SpecOp(Spec::GpuFenceAll());
                       return Expr::SpecOp(Spec::GpuBarrierLocal());
                     }},
                     false};
}

static Opt<MatchedCall> cudaSynchronise(const clang::CallExpr &call, const clang::FunctionDecl &decl) {
  const auto name = decl.getQualifiedNameAsString();
  const bool stream = name == "cudaStreamSynchronize" && call.getNumArgs() == 1;
  const bool device = name == "cudaDeviceSynchronize" && call.getNumArgs() == 0;
  if (!stream && !device) return {};
  const auto *expression = &call;
  return MatchedCall{Lowering{[expression](Remapper &self, Remapper::RemapContext &r) -> Expr::Any {
                       (void)lowerArguments(*expression, self, r);
                       if (expression->getNumArgs() == 1
                           && !expression->getArg(0)->isNullPointerConstant(self.context, clang::Expr::NPC_ValueDependentIsNotNull))
                         raise("Non-default CUDA streams are not supported in package code");
                       (void)r.newVar(Expr::SpecOp(Spec::RemoteSync(packageContext())));
                       return self.integralConstOfType(self.handleType(expression->getType(), r), 0);
                     }},
                     false};
}

static Opt<MatchedCall> cudaRuntime(const clang::CallExpr &call, const clang::FunctionDecl &decl) {
  return remoteRuntimePrism(call, decl, "cuda");
}

static Opt<MatchedCall> cudaHostQuery(const clang::CallExpr &call, const clang::FunctionDecl &decl) {
  const auto name = decl.getQualifiedNameAsString();
  const bool ptxVersion = isPtxVersionQuery(name);
  const bool occupancy = isOccupancyQuery(name);
  const bool deviceAttribute = name == "cudaDeviceGetAttribute" && call.getNumArgs() == 3;
  const auto attribute = deviceAttribute ? deviceAttributeName(*call.getArg(1)) : Opt<std::string>{};
  if ((!ptxVersion && !occupancy && !deviceAttribute) || call.getNumArgs() < 1) return {};
  const auto *expression = &call;
  return MatchedCall{
      Lowering{[expression, ptxVersion, occupancy, deviceAttribute, attribute](Remapper &self, Remapper::RemapContext &r) -> Expr::Any {
        const auto arguments = lowerArguments(*expression, self, r);
        const auto pointer = arguments.front().tpe().get<Type::Ptr>();
        if (pointer) {
          Expr::Any queried = Expr::Alias(Term::Poison(pointer->comp));
          if (ptxVersion) {
            const auto major = r.newVar(Expr::ForeignCall("polyrt_device_cuda_architecture_major", {packageContext()}, Type::IntU64()));
            const auto minor = r.newVar(Expr::ForeignCall("polyrt_device_cuda_architecture_minor", {packageContext()}, Type::IntU64()));
            const auto hundreds = r.newVar(Expr::IntrOp(Intr::Mul(major, Term::IntU64Const(100), Type::IntU64())));
            const auto tens = r.newVar(Expr::IntrOp(Intr::Mul(minor, Term::IntU64Const(10), Type::IntU64())));
            queried = self.conform(r, Expr::IntrOp(Intr::Add(hundreds, tens, Type::IntU64())), pointer->comp);
          } else if (occupancy) {
            if (expression->getDirectCallee()->getQualifiedNameAsString() == "cudaOccupancyMaxActiveBlocksPerMultiprocessor")
              raise("cudaOccupancyMaxActiveBlocksPerMultiprocessor requires a target-specific occupancy query");
            // One resident block is the conservative target-independent lower bound for a launchable kernel.
            queried = self.integralConstOfType(pointer->comp, 1);
          }
          if (deviceAttribute) {
            if (!attribute) raise("CUDA device attribute is not a named constant");
            if (*attribute == "cudaDevAttrWarpSize") queried = self.integralConstOfType(pointer->comp, 32);
            else if (*attribute == "cudaDevAttrMultiProcessorCount")
              queried =
                  self.conform(r, Expr::ForeignCall("polyrt_device_compute_units", {packageContext()}, Type::IntU64()), pointer->comp);
            else if (*attribute == "cudaDevAttrMaxSharedMemoryPerBlock" || *attribute == "cudaDevAttrMaxSharedMemoryPerBlockOptin")
              queried =
                  self.conform(r, Expr::ForeignCall("polyrt_device_local_memory_bytes", {packageContext()}, Type::IntU64()), pointer->comp);
            else if (*attribute == "cudaDevAttrMaxThreadsPerBlock")
              queried = self.conform(r, Expr::ForeignCall("polyrt_device_max_threads_per_block_u64", {packageContext()}, Type::IntU64()),
                                     pointer->comp);
            else if (*attribute == "cudaDevAttrMaxGridDimX") queried = self.integralConstOfType(pointer->comp, 0x7fffffff);
            else if (*attribute == "cudaDevAttrComputeCapabilityMajor")
              queried = self.conform(r, Expr::ForeignCall("polyrt_device_cuda_architecture_major", {packageContext()}, Type::IntU64()),
                                     pointer->comp);
            else if (*attribute == "cudaDevAttrComputeCapabilityMinor")
              queried = self.conform(r, Expr::ForeignCall("polyrt_device_cuda_architecture_minor", {packageContext()}, Type::IntU64()),
                                     pointer->comp);
            else raise("Unsupported CUDA device attribute query: " + *attribute);
          }
          const auto stored = r.newVar(queried);
          const auto base = arguments.front().get<Term::Select>();
          if (!base) raise("CUDA host query output did not lower to a selectable pointer");
          r.push(Stmt::Update(*base, Term::IntU64Const(0), stored));
        }
        return self.integralConstOfType(self.handleType(expression->getType(), r), 0);
      }},
      false};
}

static Opt<MatchedCall> cudaIgnoredHelper(const clang::CallExpr &call, const clang::FunctionDecl &decl) {
  const auto name = decl.getQualifiedNameAsString();
  const bool error = name == thrustThrowOnError && call.getNumArgs() >= 1;
  const auto *method = llvm::dyn_cast<clang::CXXMethodDecl>(&decl);
  const bool printInfo =
      method && method->getName() == "print_info"
      && (method->getParent()->getQualifiedNameAsString().find("thrust::cuda_cub::agent_launcher") != std::string::npos
          || method->getParent()->getQualifiedNameAsString().find("thrust::cuda_cub::core::AgentLauncher") != std::string::npos);
  const bool sink = name == thrustTerminateWithMessage || name == cubVaPrintf || printInfo;
  if (!error && !sink) return {};
  const auto *expression = &call;
  return MatchedCall{Lowering{[expression](Remapper &self, Remapper::RemapContext &r) { return unitAfterArguments(*expression, self, r); }},
                     false};
}

Vector<CallPrism> cudaPrisms() {
  return {cubLegacyShuffle, cubThreadAccess, cudaAtomic,  cudaBitOperation, thrustRelocate, cubWarpScan,
          cudaWarp,         cudaBarrier,     cudaRuntime, cudaSynchronise,  cudaHostQuery,  cudaIgnoredHelper};
}

} // namespace polyregion::polystl::call_prism
