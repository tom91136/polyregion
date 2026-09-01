
#include "c_source.h"

#include <cctype>
#include <cmath>
#include <functional>
#include <limits>
#include <set>

#include "aspartame/all.hpp"
#include "fmt/core.h"

#include "polyregion/conventions.h"
#include "polyregion/env_keys.h"

using namespace aspartame;
using namespace polyregion;

namespace {
// XXX inf/nan have no numeric spelling; a finite integral-reading value gets a `.0` to stay floating-point
std::string cFloatLiteral(double v, const std::string &suffix) {
  if (std::isinf(v)) return v < 0 ? "-INFINITY" : "INFINITY";
  if (std::isnan(v)) return "NAN";
  auto s = fmt::format("{}", v);
  if (s.find_first_of(".eE") == std::string::npos) s += ".0";
  return s + suffix;
}

std::string escapeCString(const std::string &s) {
  return s ^ mk_string("", [](char c) -> std::string {
           if (c == '"' || c == '\\') return fmt::format("\\{}", c);
           const auto u = static_cast<unsigned char>(c);
           if (u < 0x20 || u >= 0x7f) return fmt::format("\\{:03o}", u);
           return std::string(1, c);
         });
}
} // namespace
using namespace polyast;
using namespace std::string_literals;

static bool isLocalArr(const Type::Any &t) {
  return t.template get<Type::Arr>() ^ exists([](const auto &a) { return a.space.template is<TypeSpace::Local>(); });
}

static bool isPoisonInit(const Expr::Any &e) {
  const auto alias = e.template get<Expr::Alias>();
  return alias && alias->ref.template is<Term::Poison>();
}

static bool isNullPtrInit(const Expr::Any &e) {
  const auto alias = e.template get<Expr::Alias>();
  return alias && alias->ref.template is<Term::NullPtrConst>();
}

static std::optional<Sym> structNameOf(const Type::Any &t) {
  Type::Any base = t;
  if (const auto ptr = base.template get<Type::Ptr>()) base = ptr->comp;
  if (const auto structure = base.template get<Type::Struct>()) return structure->name;
  return std::nullopt;
}

static std::optional<Term::Select> aliasSelect(const Expr::Any &e) {
  if (const auto alias = e.template get<Expr::Alias>()) return alias->ref.template get<Term::Select>();
  return std::nullopt;
}

static std::optional<Term::Select> initSelect(const Expr::Any &e) {
  if (const auto alias = e.template get<Expr::Alias>()) return alias->ref.template get<Term::Select>();
  if (const auto ref = e.template get<Expr::RefTo>()) return ref->lhs.template get<Term::Select>();
  if (const auto cast = e.template get<Expr::Cast>()) return cast->from.template get<Term::Select>();
  return std::nullopt;
}

static std::string volatileHelperName(const bool load, const std::string &space, const std::string &element) {
  return fmt::format("_pr_v{}_{}_{}", load ? "ld" : "st", space, element);
}

static std::string atomicMinMaxHelperName(const bool minimum, const std::string &element) {
  return fmt::format("_pr_atomic_{}_{}", minimum ? "min" : "max", element);
}

struct SlotUnion {
  Map<std::string, std::string> parent;
  std::string find(const std::string &key) {
    const auto it = parent.find(key);
    if (it == parent.end()) return parent[key] = key;
    return it->second == key ? key : parent[key] = find(it->second);
  }
  void unite(const std::string &lhs, const std::string &rhs) {
    const auto l = find(lhs), r = find(rhs);
    if (l != r) parent[l] = r;
  }
};

static std::optional<uint64_t> scalarBytes(const Type::Any &t) {
  if (t.template is<Type::Bool1>() || t.template is<Type::IntU8>() || t.template is<Type::IntS8>()) return 1;
  if (t.template is<Type::Float16>() || t.template is<Type::IntU16>() || t.template is<Type::IntS16>()) return 2;
  if (t.template is<Type::Float32>() || t.template is<Type::IntU32>() || t.template is<Type::IntS32>()) return 4;
  if (t.template is<Type::Float64>() || t.template is<Type::IntU64>() || t.template is<Type::IntS64>()) return 8;
  return std::nullopt;
}

struct ArrayExtent {
  Type::Any element;
  uint64_t count;
};

static std::optional<ArrayExtent> arrayExtent(const Type::Any &t) {
  Type::Any element = t;
  uint64_t count = 1;
  bool found = false;
  while (const auto a = element.template get<Type::Arr>()) {
    found = true;
    if (a->length < 0) return std::nullopt;
    if (a->length != 0 && count > std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(a->length)) return std::nullopt;
    count *= static_cast<uint64_t>(a->length);
    element = a->comp;
  }
  return found ? std::optional<ArrayExtent>{{element, count}} : std::nullopt;
}

template <typename T> static bool usesTpe(const std::vector<Function> &fns, const std::vector<StructDef> &defs) {
  return (fns ^ exists([](const auto &f) { return !f.template collect_all<T>().empty(); }))
         || (defs ^ exists([](const auto &d) {
               return d.members ^ exists([](const auto &m) { return !m.tpe.template collect_all<T>().empty(); });
             }));
}

struct CLAddressSpaceTracePass {

  bool strictSpaces = true;

  struct StackScope {
    Map<std::string, Named> vars;
  };

  struct SpacedTerm {
    Term::Any actual;
    TypeSpace::Any space = TypeSpace::Private();
  };

  struct SpacedExpr {
    Expr::Any actual;
    TypeSpace::Any space = TypeSpace::Private();
  };

  Map<Sym, Map<std::string, Type::Any>> fields;

  std::optional<Type::Any> fieldType(const Type::Any &owner, const std::string &name) const {
    Type::Any t = owner;
    if (const auto p = t.template get<Type::Ptr>()) t = p->comp;
    const auto s = t.template get<Type::Struct>();
    if (!s) return std::nullopt;
    if (const auto members = fields ^ get_maybe(s->name)) return *members ^ get_maybe(name);
    return std::nullopt;
  }

  struct PathWalk {
    Type::Any type;
    TypeSpace::Any space;
  };

  PathWalk walkPath(const Type::Any &rootTpe, const std::vector<PathStep::Any> &steps, size_t upto) const {
    Type::Any current = rootTpe;
    TypeSpace::Any live = rootTpe.template get<Type::Ptr>() | map([](const auto &p) { return p.space; })
                          | get_or_else(isLocalArr(rootTpe) ? TypeSpace::Local().widen() : TypeSpace::Private().widen());
    for (size_t i = 0; i < upto && i < steps.size(); ++i) {
      steps[i].match_total(
          [&](const PathStep::Field &f) {
            if (const auto p = current.template get<Type::Ptr>()) live = p->space, current = p->comp;
            if (const auto tpe = fieldType(current, f.name)) current = *tpe;
          },
          [&](const PathStep::Deref &) {
            if (const auto p = current.template get<Type::Ptr>()) live = p->space, current = p->comp;
          },
          [&](const PathStep::Index &) {
            if (const auto p = current.template get<Type::Ptr>()) live = p->space, current = p->comp;
            else if (const auto a = current.template get<Type::Arr>()) current = a->comp;
          },
          [&](const PathStep::IndexDyn &) {
            if (const auto p = current.template get<Type::Ptr>()) live = p->space, current = p->comp;
            else if (const auto a = current.template get<Type::Arr>()) current = a->comp;
          });
    }
    return {current, live};
  }

  // Map a Term: only Term::Select needs scope rewiring; all other Term variants are pure atoms.
  SpacedTerm mapTerm(const Term::Any &term, StackScope &scope) {
    if (auto sel = term.template get<Term::Select>()) {
      const auto rebound = scope.vars ^ get_or_default(sel->root.symbol, sel->root);
      const auto walked = walkPath(rebound.tpe, sel->steps, sel->steps.size());
      const auto liveSpace = walked.space;
      auto declared = sel->steps.empty() ? std::optional<Type::Ptr>{} : sel->tpe.template get<Type::Ptr>();
      if (declared)
        if (const auto field = sel->steps.back().template get<PathStep::Field>())
          if (const auto tpe = fieldType(walkPath(rebound.tpe, sel->steps, sel->steps.size() - 1).type, field->name))
            declared = tpe->template get<Type::Ptr>() ^ get_or_else(*declared);
      const auto space = declared | map([](const auto &p) { return p.space; }) | get_or_else(liveSpace);
      // The conflict-splitting pass can replace the root with an address-space-specialised
      // struct. Carry that structural replacement through aliases; retaining sel->tpe here
      // silently converts `private Box_asp*` back to `private Box*`. Preserve the declared
      // type otherwise because non-entry helper arguments deliberately retain their original
      // address-space contract even though their local binding is private.
      auto actualTpe = sel->tpe;
      const auto declaredStruct = structNameOf(actualTpe), walkedStruct = structNameOf(walked.type);
      if (declaredStruct && walkedStruct && *declaredStruct != *walkedStruct) actualTpe = walked.type;
      return SpacedTerm{Term::Select(rebound, sel->steps, actualTpe), space};
    }
    // XXX null seeds its space from the pointee (refTpe=Global) so `T* p = nullptr` later infers Global not Private
    if (auto np = term.template get<Term::NullPtrConst>()) return SpacedTerm{term, np->space};
    if (term.template is<Term::StringConst>()) return SpacedTerm{term, TypeSpace::Constant()};
    return SpacedTerm{term};
  }

  SpacedExpr mapExpr(const Expr::Any &expr, StackScope &scope) {
    auto mapTerm_ = [&](const Term::Any &t) { return mapTerm(t, scope); };
    auto mapTerm0_ = [&](const Term::Any &t) { return mapTerm(t, scope).actual; };
    return expr.match_total(
        [&](const Expr::Alias &x) -> SpacedExpr {
          auto st = mapTerm_(x.ref);
          return {Expr::Alias(st.actual), st.space};
        },
        [&](const Expr::SpecOp &x) -> SpacedExpr { return {Expr::SpecOp(x.op.modify_all<Term::Any>(mapTerm0_))}; },
        [&](const Expr::IntrOp &x) -> SpacedExpr { return {Expr::IntrOp(x.op.modify_all<Term::Any>(mapTerm0_))}; },
        [&](const Expr::MathOp &x) -> SpacedExpr { return {Expr::MathOp(x.op.modify_all<Term::Any>(mapTerm0_))}; },
        [&](const Expr::Cast &x) -> SpacedExpr {
          auto st = mapTerm_(x.from);
          if (auto asPtr = x.as.template get<Type::Ptr>(); asPtr && !x.from.tpe().template is<Type::Ptr>())
            return {Expr::Cast(st.actual, x.as), asPtr->space};
          // re-space the cast target so OpenCL doesn't see a global<-private mismatch
          auto as = x.as.template get<Type::Ptr>()                                            //
                    | map([&](const auto &p) { return Type::Ptr(p.comp, st.space).widen(); }) //
                    | get_or_else(x.as);
          return {Expr::Cast(st.actual, as), st.space};
        },
        [&](const Expr::Invoke &x) -> SpacedExpr { return {x.modify_all<Term::Any>(mapTerm0_)}; },
        [&](const Expr::Index &x) -> SpacedExpr {
          auto stLhs = mapTerm_(x.lhs);
          auto stIdx = mapTerm_(x.idx);
          const auto space = x.comp.template get<Type::Ptr>() ^ fold([](const auto &p) { return p.space; }, [&] { return stLhs.space; });
          return {Expr::Index(stLhs.actual, stIdx.actual, x.comp), space};
        },
        [&](const Expr::RefTo &x) -> SpacedExpr {
          auto stLhs = mapTerm_(x.lhs);
          const auto space = x.space.template is<TypeSpace::Private>() ? x.space : stLhs.space;
          return {Expr::RefTo(stLhs.actual, x.idx ^ map(mapTerm0_), x.comp, space, Region::Opaque()), space};
        },
        [&](const Expr::Alloc &x) -> SpacedExpr { return {Expr::Alloc(x.comp, mapTerm0_(x.size), x.space, Region::Opaque())}; },
        [&](const Expr::ForeignCall &x) -> SpacedExpr { return {x.modify_all<Term::Any>(mapTerm0_)}; },
        [&](const Expr::OffsetOf &x) -> SpacedExpr { return {Expr::OffsetOf(x.structTpe, x.field)}; },
        [&](const Expr::SizeOf &x) -> SpacedExpr { return {Expr::SizeOf(x.forTpe)}; });
  }

  std::optional<std::pair<Sym, std::string>> memberStoreTarget(const Term::Select &select, StackScope &scope) const {
    if (select.steps.empty()) return std::nullopt;
    const auto field = select.steps.back().template get<PathStep::Field>();
    if (!field) return std::nullopt;
    const auto rebound = scope.vars ^ get_or_default(select.root.symbol, select.root);
    Type::Any owner = walkPath(rebound.tpe, select.steps, select.steps.size() - 1).type;
    if (const auto ptr = owner.template get<Type::Ptr>()) owner = ptr->comp;
    const auto structure = owner.template get<Type::Struct>();
    if (!structure) return std::nullopt;
    const auto tpe = fieldType(owner, field->name);
    return tpe && tpe->template is<Type::Ptr>() ? std::optional<std::pair<Sym, std::string>>{{structure->name, field->name}} : std::nullopt;
  }

  struct MemberStores {
    Map<Sym, Set<std::string>> global, constant, local, priv;
  };

  static void eachStmt(const std::vector<Stmt::Any> &stmts, const std::function<void(const Stmt::Any &)> &f) {
    for (const auto &stmt : stmts) {
      f(stmt);
      if (const auto c = stmt.template get<Stmt::Cond>()) eachStmt(c->trueBr, f), eachStmt(c->falseBr, f);
      else if (const auto w = stmt.template get<Stmt::While>()) eachStmt(w->body, f);
      else if (const auto r = stmt.template get<Stmt::ForRange>()) eachStmt(r->body, f);
      else if (const auto a = stmt.template get<Stmt::Annotated>()) eachStmt({a->inner}, f);
    }
  }

  Function mapFn(const Function &fn, MemberStores *memberStores = nullptr) {

    StackScope scope{.vars = fn.decl.args                                                                   //
                             | flat_map([&](const auto &arg) { return arg.template collect_all<Named>(); }) //
                             | filter([](const auto &n) { return n.tpe.template is<Type::Ptr>(); })         //
                             | map([](const auto &n) { return std::pair(n.symbol, n); })                    //
                             | to<Map>()};

    // a phi pointer var (declared with no initialiser, assigned in branches) keeps its declared space;
    // pre-scan its Mut assignments so the decl takes the assigned value's space, else OpenCL rejects the
    // `global* = private*` of e.g. std::min(&a, &b) over stack scalars in basic_string::max_size
    Map<std::string, TypeSpace::Any> phiSpace;
    Map<std::string, int> phiKinds;
    {
      auto spaceKind = [](const TypeSpace::Any &s) {
        return s.match_total([](const TypeSpace::Global &) { return 0; }, [](const TypeSpace::Constant &) { return 1; },
                             [](const TypeSpace::Local &) { return 2; }, [](const TypeSpace::Private &) { return 3; });
      };
      // a no-initialiser phi pointer's space comes from its branch Muts, which can read OTHER phi vars
      // over a loop back-edge (swg's std::max row-swap); iterate to a fixpoint so a phi-var read uses the
      // inferred (not declared) space and the chain converges
      bool changed = true;
      for (size_t iter = 0; changed && iter <= scope.vars.size() + 2; ++iter) {
        changed = false;
        Map<std::string, TypeSpace::Any> next;
        Map<std::string, int> nextKinds;
        StackScope scan{.vars = scope.vars};
        std::function<void(const std::vector<Stmt::Any> &)> walk = [&](const std::vector<Stmt::Any> &stmts) {
          for (auto &s : stmts) {
            if (auto var = s.template get<Stmt::Var>()) {
              auto name = var->name;
              if (auto expr = var->expr ^ map([&](const auto &e) { return mapExpr(e, scan); })) {
                if (auto ptr = expr->actual.tpe().template get<Type::Ptr>()) {
                  const auto space = isNullPtrInit(*var->expr) ? phiSpace ^ get_or_default(var->name.symbol, expr->space) : expr->space;
                  name = Named(var->name.symbol, Type::Ptr(ptr->comp, space));
                }
              } else if (auto ptr = var->name.tpe.template get<Type::Ptr>()) {
                if (auto sp = phiSpace ^ get_maybe(var->name.symbol)) name = Named(var->name.symbol, Type::Ptr(ptr->comp, *sp));
              }
              scan.vars.insert_or_assign(name.symbol, name);
            } else if (auto mut = s.template get<Stmt::Mut>()) {
              if (mut->name.steps.empty() && mut->name.root.tpe.template is<Type::Ptr>()) {
                const auto sp = mapExpr(mut->expr, scan).space;
                nextKinds[mut->name.root.symbol] |= 1 << spaceKind(sp);
                // a private<-private+global merge can't be a global pointer in CL 1.2; keep it private
                const auto prev = next ^ get_maybe(mut->name.root.symbol);
                next.insert_or_assign(mut->name.root.symbol,
                                      (prev && (prev->template is<TypeSpace::Private>() || sp.template is<TypeSpace::Private>()))
                                          ? TypeSpace::Private().widen()
                                          : sp);
              }
            } else if (auto c = s.template get<Stmt::Cond>()) {
              walk(c->trueBr);
              walk(c->falseBr);
            } else if (auto w = s.template get<Stmt::While>()) walk(w->body);
            else if (auto fr = s.template get<Stmt::ForRange>()) walk(fr->body);
            else if (auto an = s.template get<Stmt::Annotated>()) walk(std::vector<Stmt::Any>{an->inner});
          }
        };
        walk(fn.body);
        if (next.size() != phiSpace.size()) changed = true;
        else
          for (auto &[k, v] : next) {
            const auto old = phiSpace ^ get_maybe(k);
            if (!old || spaceKind(*old) != spaceKind(v)) {
              changed = true;
              break;
            }
          }
        phiSpace = next;
        phiKinds = nextKinds;
      }
    }

    if (strictSpaces) {
      const auto conflicted =
          phiKinds | filter([](const auto &, const auto mask) { return (mask & (mask - 1)) != 0; }) | keys() | to<Set>();
      if (!conflicted.empty()) throw backend::BackendException("cross-address-space pointer merge escapes read-only use");
    }

    auto body = fn.body ^ map([&](const auto &s) {
                  return s
                      .template modify_all<Stmt::Var>([&](const auto &var) { //
                        if (auto expr = var.expr ^ map([&](const auto &e) { return mapExpr(e, scope); })) {
                          auto name = var.name;
                          if (auto ptr = expr->actual.tpe().template get<Type::Ptr>()) {
                            const auto space =
                                isNullPtrInit(*var.expr) ? phiSpace ^ get_or_default(var.name.symbol, expr->space) : expr->space;
                            name = Named(var.name.symbol, Type::Ptr(ptr->comp, space));
                          }
                          scope.vars.emplace(name.symbol, name);
                          return Stmt::Var(name, expr->actual, var.isMutable);
                        }
                        auto name = var.name;
                        if (auto ptr = var.name.tpe.template get<Type::Ptr>())
                          if (auto sp = phiSpace ^ get_maybe(var.name.symbol)) name = Named(var.name.symbol, Type::Ptr(ptr->comp, *sp));
                        scope.vars.emplace(name.symbol, name);
                        return Stmt::Var(name, {}, var.isMutable);
                      })
                      .template modify_all<Expr::Any>([&](const auto &e) { return mapExpr(e, scope).actual; }); //
                });

    if (memberStores) {
      eachStmt(fn.body, [&](const Stmt::Any &stmt) {
        const auto mut = stmt.template get<Stmt::Mut>();
        if (!mut) return;
        if (const auto target = memberStoreTarget(mut->name, scope)) {
          const auto space = mapExpr(mut->expr, scope).space;
          auto &bucket = space.template is<TypeSpace::Global>()     ? memberStores->global
                         : space.template is<TypeSpace::Constant>() ? memberStores->constant
                         : space.template is<TypeSpace::Local>()    ? memberStores->local
                                                                    : memberStores->priv;
          bucket[target->first].insert(target->second);
        }
      });
    }

    const auto tracedRtnTpes = body                                                                              //
                               | flat_map([&](const auto &s) { return s.template collect_all<Stmt::Return>(); }) //
                               | map([&](const auto &r) { return r.value.tpe(); })                               //
                               | distinct()                                                                      //
                               | to_vector();
    return Function(fn.decl.withRtn(tracedRtnTpes[0]), body, fn.visibility, fn.fpMode, fn.convention, fn.implements,
                    fn.requiredCapabilities);
  }

  struct ConflictSplit {
    Map<Sym, StructDef> clones;
    Map<Sym, Map<std::string, Sym>> memberRetype;
    Map<std::string, Map<std::string, Sym>> fnVarRetype;
  };

  static std::string functionKey(const Function &fn) {
    const auto types = [](const std::vector<Arg> &args) {
      std::vector<Type::Any> result;
      result.reserve(args.size());
      for (const auto &arg : args)
        result.push_back(arg.named.tpe);
      return result;
    };
    std::optional<Type::Any> receiver;
    if (fn.decl.receiver) receiver = fn.decl.receiver->named.tpe;
    return signatureKey(Signature(fn.decl.name, fn.decl.tpeVars, receiver, types(fn.decl.args), types(fn.decl.moduleCaptures),
                                  types(fn.decl.termCaptures), Type::Unit0()));
  }

  static Type::Any retypeStructOccurrence(const Type::Any &tpe, const Sym &clone) {
    if (const auto structure = tpe.template get<Type::Struct>()) return Type::Struct(clone, structure->args).widen();
    if (const auto ptr = tpe.template get<Type::Ptr>())
      if (const auto structure = ptr->comp.template get<Type::Struct>())
        return Type::Ptr(Type::Struct(clone, structure->args).widen(), ptr->space).widen();
    return tpe;
  }

  static int spaceCode(const TypeSpace::Any &space) {
    return space.match_total([](const TypeSpace::Global &) { return 0; }, [](const TypeSpace::Constant &) { return 1; },
                             [](const TypeSpace::Local &) { return 2; }, [](const TypeSpace::Private &) { return 3; });
  }

  static TypeSpace::Any spaceFromCode(int code) {
    return code == 1   ? TypeSpace::Constant().widen()
           : code == 2 ? TypeSpace::Local().widen()
           : code == 3 ? TypeSpace::Private().widen()
                       : TypeSpace::Global().widen();
  }

  std::optional<ConflictSplit> planConflictSplit(const std::vector<Function> &functions, const Set<Sym> &conflicted,
                                                 const Map<Sym, StructDef> &structDefs) {
    const auto carries = [&](const Type::Any &tpe) -> std::optional<Sym> {
      const auto structure = structNameOf(tpe);
      return structure && (conflicted ^ contains(*structure)) ? structure : std::nullopt;
    };

    SlotUnion slots;
    Map<std::string, Sym> slotStruct;
    Map<std::string, std::pair<std::string, std::string>> variableSlots;
    Map<std::string, std::pair<Sym, std::string>> memberSlots;
    const auto variableKey = [](const std::string &fn, const std::string &symbol) { return "V\x1f" + fn + "\x1f" + symbol; };
    const auto memberKey = [](const Sym &owner, const std::string &member) { return "M\x1f" + fqcn(owner) + "\x1f" + member; };
    const auto registerVariable = [&](const std::string &fn, const std::string &symbol, const Sym &structure) {
      const auto key = variableKey(fn, symbol);
      slotStruct.insert_or_assign(key, structure);
      variableSlots.insert_or_assign(key, std::pair{fn, symbol});
      slots.find(key);
      return key;
    };
    const auto registerMember = [&](const Sym &owner, const std::string &member, const Sym &structure) {
      const auto key = memberKey(owner, member);
      slotStruct.insert_or_assign(key, structure);
      memberSlots.insert_or_assign(key, std::pair{owner, member});
      slots.find(key);
      return key;
    };

    const auto designated = [&](const std::string &fn, const Term::Select &select) -> std::optional<std::string> {
      if (select.steps.empty()) {
        if (const auto structure = carries(select.root.tpe)) return registerVariable(fn, select.root.symbol, *structure);
        return std::nullopt;
      }
      const auto field = select.steps.back().template get<PathStep::Field>();
      if (!field) return std::nullopt;
      Type::Any owner = walkPath(select.root.tpe, select.steps, select.steps.size() - 1).type;
      if (const auto ptr = owner.template get<Type::Ptr>()) owner = ptr->comp;
      const auto structure = owner.template get<Type::Struct>();
      if (!structure) return std::nullopt;
      if (const auto tpe = fieldType(owner, field->name))
        if (const auto carried = carries(*tpe)) return registerMember(structure->name, field->name, *carried);
      return std::nullopt;
    };
    const auto expressionSpace = [&](const Expr::Any &expr) {
      if (const auto alias = expr.template get<Expr::Alias>())
        if (const auto select = alias->ref.template get<Term::Select>())
          return spaceCode(walkPath(select->root.tpe, select->steps, select->steps.size()).space);
      if (const auto ref = expr.template get<Expr::RefTo>()) return spaceCode(ref->space);
      if (const auto ptr = expr.tpe().template get<Type::Ptr>()) return spaceCode(ptr->space);
      return 3;
    };

    bool valid = true;
    std::vector<std::tuple<std::string, std::string, int>> colours;
    struct NestedCopy {
      std::string function;
      Sym owner;
      std::string destination, member, source;
    };
    std::vector<NestedCopy> nestedCopies;
    struct AggregateCopy {
      std::string function, destination, source;
    };
    std::vector<AggregateCopy> aggregateCopies;
    for (const auto &function : functions) {
      const auto fn = functionKey(function);
      eachStmt(function.body, [&](const Stmt::Any &stmt) {
        if (const auto var = stmt.template get<Stmt::Var>()) {
          if (const auto structure = carries(var->name.tpe)) {
            const auto dst = registerVariable(fn, var->name.symbol, *structure);
            if (var->expr && !isPoisonInit(*var->expr)) {
              const auto source = initSelect(*var->expr);
              const auto src = source ^ flat_map([&](const auto &select) { return designated(fn, select); });
              if (src) slots.unite(dst, *src);
              else valid = false;
            }
          }
          if (var->name.tpe.template is<Type::Struct>() && var->expr && !isPoisonInit(*var->expr))
            if (const auto source = initSelect(*var->expr); source && source->steps.empty() && source->root.tpe == var->name.tpe)
              aggregateCopies.push_back({fn, var->name.symbol, source->root.symbol});
          return;
        }
        const auto mut = stmt.template get<Stmt::Mut>();
        if (!mut) return;
        const auto &select = mut->name;
        if (select.steps.empty()) {
          if (const auto structure = carries(select.root.tpe)) {
            const auto source = aliasSelect(mut->expr);
            const auto src = source ^ flat_map([&](const auto &value) { return designated(fn, value); });
            if (src) slots.unite(registerVariable(fn, select.root.symbol, *structure), *src);
            else valid = false;
          }
          if (select.root.tpe.template is<Type::Struct>())
            if (const auto source = aliasSelect(mut->expr); source && source->steps.empty() && source->root.tpe == select.root.tpe)
              aggregateCopies.push_back({fn, select.root.symbol, source->root.symbol});
          return;
        }
        const auto field = select.steps.back().template get<PathStep::Field>();
        if (!field) return;
        Type::Any owner = walkPath(select.root.tpe, select.steps, select.steps.size() - 1).type;
        if (const auto ptr = owner.template get<Type::Ptr>()) owner = ptr->comp;
        const auto structure = owner.template get<Type::Struct>();
        const auto tpe = structure ? fieldType(owner, field->name) : std::nullopt;
        if (structure && tpe && tpe->template is<Type::Ptr>() && (conflicted ^ contains(structure->name))) {
          std::optional<std::string> slot;
          if (select.steps.size() == 1) {
            slot = registerVariable(fn, select.root.symbol, structure->name);
          } else {
            auto ownerSteps = select.steps;
            ownerSteps.pop_back();
            const Term::Select ownerSelect(select.root, ownerSteps, walkPath(select.root.tpe, ownerSteps, ownerSteps.size()).type);
            slot = designated(fn, ownerSelect);
          }
          if (!slot) throw backend::BackendException("cannot specialise indirect conflicting pointer field");
          colours.emplace_back(*slot, field->name, expressionSpace(mut->expr));
        } else if (structure && tpe) {
          if (const auto carried = carries(*tpe)) {
            const auto source = aliasSelect(mut->expr);
            if (source && select.steps.size() == 1 && source->steps.empty() && select.root.tpe.template is<Type::Struct>())
              nestedCopies.push_back({fn, structure->name, select.root.symbol, field->name, source->root.symbol});
            else {
              const auto src = source ^ flat_map([&](const auto &value) { return designated(fn, value); });
              if (src) slots.unite(registerMember(structure->name, field->name, *carried), *src);
              else valid = false;
            }
          }
        }
      });
    }
    if (!valid) return std::nullopt;

    Map<std::string, Map<std::string, int>> componentColours;
    for (const auto &[slot, member, code] : colours) {
      auto &signature = componentColours[slots.find(slot)];
      if (const auto it = signature.find(member); it != signature.end() && it->second != code) return std::nullopt;
      signature[member] = code;
    }

    ConflictSplit plan;
    Map<std::string, Sym> clones;
    for (const auto &[slot, structure] : slotStruct) {
      const auto component = slots.find(slot);
      const auto coloursIt = componentColours.find(component);
      if (coloursIt == componentColours.end()) continue;
      const auto fieldsIt = fields.find(structure);
      bool differs = false;
      for (const auto &[member, code] : coloursIt->second) {
        const auto tpe = fieldsIt != fields.end() ? fieldsIt->second ^ get_maybe(member) : std::nullopt;
        differs |= !tpe || !tpe->template is<Type::Ptr>() || spaceCode(tpe->template get<Type::Ptr>()->space) != code;
      }
      if (!differs) continue;

      std::map<std::string, int> signature;
      if (const auto def = structDefs.find(structure); def != structDefs.end())
        for (const auto &member : def->second.members)
          if (const auto ptr = member.tpe.template get<Type::Ptr>()) signature.emplace(member.symbol, spaceCode(ptr->space));
      for (const auto &[member, code] : coloursIt->second)
        signature.insert_or_assign(member, code);
      const auto suffix = signature | values() | fold_left(std::string("_as"), [](const auto &acc, const auto code) {
                            return acc + (code == 1 ? "c" : code == 2 ? "l" : code == 3 ? "p" : "g");
                          });
      auto cloneName = structure.fqn;
      if (cloneName.empty()) cloneName.push_back(suffix);
      else cloneName.back() += suffix;
      const auto clone = clones ^ get_or_default(component, Sym(cloneName));
      clones.emplace(component, clone);
      if (!plan.clones.contains(clone))
        if (const auto def = structDefs.find(structure); def != structDefs.end())
          plan.clones.emplace(clone, def->second.withName(clone).withMembers(
                                         def->second.members ^ map([&](const Named &member) {
                                           const auto code = signature ^ get_maybe(member.symbol);
                                           const auto ptr = member.tpe.template get<Type::Ptr>();
                                           return code && ptr ? member.withTpe(Type::Ptr(ptr->comp, spaceFromCode(*code)).widen()) : member;
                                         })));
      if (const auto variable = variableSlots.find(slot); variable != variableSlots.end())
        plan.fnVarRetype[variable->second.first].insert_or_assign(variable->second.second, clone);
      if (const auto member = memberSlots.find(slot); member != memberSlots.end())
        plan.memberRetype[member->second.first].insert_or_assign(member->second.second, clone);
    }

    Map<std::string, Sym> nestedClones;
    if (strictSpaces)
      for (bool changed = true; changed;) {
        changed = false;
        Map<std::string, std::vector<NestedCopy>> groups;
        for (const auto &copy : nestedCopies)
          groups[copy.function + "\x1f" + copy.destination].push_back(copy);
        for (const auto &[_, copies] : groups) {
          if (copies.empty()) continue;
          auto function = plan.fnVarRetype.find(copies.front().function);
          const auto owner = structDefs.find(copies.front().owner);
          if (function == plan.fnVarRetype.end() || owner == structDefs.end()) continue;
          std::map<std::string, Sym> replacements;
          for (const auto &copy : copies)
            if (const auto source = function->second.find(copy.source); source != function->second.end())
              replacements.insert_or_assign(copy.member, source->second);
          if (replacements.empty()) continue;

          std::string suffix = "_as", key = fqcn(copies.front().owner);
          for (const auto &member : owner->second.members) {
            const auto carried = structNameOf(member.tpe);
            if (!carried || !(conflicted ^ contains(*carried))) continue;
            if (const auto replacement = replacements.find(member.symbol); replacement != replacements.end()) {
              const auto child = fqcn(replacement->second);
              const auto parts = child ^ split("_as");
              const auto code = parts.size() == 1 ? "g" : parts.back();
              suffix += code;
              key += "\x1f" + member.symbol + "\x1f" + child;
            } else {
              suffix += "g";
              key += "\x1f" + member.symbol + "\x1f";
            }
          }
          auto clone = nestedClones ^ get_maybe(key);
          if (!clone) {
            auto name = copies.front().owner.fqn;
            if (name.empty()) name.push_back(suffix);
            else name.back() += suffix;
            clone = Sym(name);
            nestedClones.emplace(key, *clone);
            plan.clones.emplace(*clone, owner->second.withName(*clone).withMembers(
                                            owner->second.members ^ map([&](const Named &member) {
                                              const auto replacement = replacements.find(member.symbol);
                                              return replacement == replacements.end()
                                                         ? member
                                                         : member.withTpe(retypeStructOccurrence(member.tpe, replacement->second));
                                            })));
          }
          const auto current = function->second.find(copies.front().destination);
          if (current == function->second.end() || current->second != *clone) {
            function->second.insert_or_assign(copies.front().destination, *clone);
            changed = true;
          }
        }
        for (const auto &copy : aggregateCopies) {
          const auto function = plan.fnVarRetype.find(copy.function);
          if (function == plan.fnVarRetype.end()) continue;
          const auto source = function->second.find(copy.source);
          if (source == function->second.end()) continue;
          const auto destination = function->second.find(copy.destination);
          if (destination == function->second.end() || destination->second != source->second) {
            function->second.insert_or_assign(copy.destination, source->second);
            changed = true;
          }
        }
      }
    return plan;
  }

  Function retypeConflicted(const Function &fn, const ConflictSplit &plan) {
    const auto it = plan.fnVarRetype.find(functionKey(fn));
    if (it == plan.fnVarRetype.end()) return fn;
    const auto &types = it->second;
    auto retyped = fn.template modify_all<Named>([&](const Named &named) {
      const auto type = types.find(named.symbol);
      return type == types.end() ? named : named.withTpe(retypeStructOccurrence(named.tpe, type->second));
    });
    return retyped.template modify_all<Expr::RefTo>([&](const Expr::RefTo &ref) {
      const auto select = ref.lhs.template get<Term::Select>();
      if (!select || !(types ^ get_maybe(select->root.symbol))) return ref;
      return Expr::RefTo(ref.lhs, ref.idx, walkPath(select->root.tpe, select->steps, select->steps.size()).type, ref.space, ref.region);
    });
  }

  static Program execute(const Program &p, bool strictSpaces) {
    CLAddressSpaceTracePass pass;
    pass.strictSpaces = strictSpaces;
    for (const auto &def : p.defs)
      pass.fields.emplace(def.name, def.members | map([](const auto &member) { return std::pair{member.symbol, member.tpe}; }) | to<Map>());

    const auto argRespace = [](const Function &f) {
      const auto offloadEntry = f.convention.is<CallConvention::OffloadEntry>();
      auto remapSpace = [&](const auto &s) {
        return s.match_total(
            [&](const TypeSpace::Global &) { return offloadEntry ? TypeSpace::Global().widen() : TypeSpace::Private().widen(); }, //
            [&](const TypeSpace::Constant &) { return offloadEntry ? TypeSpace::Global().widen() : TypeSpace::Private().widen(); },
            [&](const TypeSpace::Local &x) { return x.widen(); }, //
            [&](const TypeSpace::Private &x) { return x.widen(); });
      };
      return f.withDecl(
          f.decl.withArgs(f.decl.args ^ map([&](const auto &arg) { return arg.template modify_all<TypeSpace::Any>(remapSpace); })));
    };

    auto seeds = p.functions ^ map(argRespace);
    if (p.entry) seeds ^= prepend(argRespace(*p.entry));
    for (bool changed = true; changed;) {
      changed = false;
      MemberStores stores;
      for (const auto &function : seeds)
        pass.mapFn(function, &stores);
      const auto has = [](const auto &members, const Sym &structure, const std::string &field) {
        const auto it = members.find(structure);
        return it != members.end() && (it->second ^ contains(field));
      };
      const auto respace = [&](const auto &members, const TypeSpace::Any &space, const auto &skip) {
        for (const auto &[structure, names] : members)
          for (const auto &name : names) {
            if (skip(structure, name)) continue;
            const auto owner = pass.fields.find(structure);
            if (owner == pass.fields.end()) continue;
            const auto field = owner->second.find(name);
            if (field == owner->second.end()) continue;
            if (const auto ptr = field->second.template get<Type::Ptr>(); ptr && ptr->space != space)
              field->second = Type::Ptr(ptr->comp, space).widen(), changed = true;
          }
      };
      respace(stores.constant, TypeSpace::Constant().widen(), [&](const auto &structure, const auto &field) {
        return has(stores.global, structure, field) || has(stores.local, structure, field) || has(stores.priv, structure, field);
      });
      respace(stores.local, TypeSpace::Local().widen(), [](const auto &, const auto &) { return false; });
      respace(stores.priv, TypeSpace::Private().widen(), [&](const auto &structure, const auto &field) {
        return has(stores.global, structure, field) || has(stores.constant, structure, field) || has(stores.local, structure, field);
      });
    }

    MemberStores stores;
    if (strictSpaces)
      for (const auto &function : seeds)
        pass.mapFn(function, &stores);
    const auto has = [](const auto &members, const Sym &structure, const std::string &field) {
      const auto it = members.find(structure);
      return it != members.end() && (it->second ^ contains(field));
    };
    Set<Sym> conflicted;
    for (const auto *bucket : {&stores.global, &stores.constant, &stores.local, &stores.priv})
      for (const auto &[structure, fields] : *bucket)
        for (const auto &field : fields)
          if (static_cast<int>(has(stores.global, structure, field)) + static_cast<int>(has(stores.constant, structure, field))
                  + static_cast<int>(has(stores.local, structure, field)) + static_cast<int>(has(stores.priv, structure, field))
              >= 2)
            conflicted.insert(structure);

    ConflictSplit split;
    if (!conflicted.empty()) {
      const auto structDefs = p.defs | map([](const auto &def) { return std::pair{def.name, def}; }) | to<Map>();
      const auto planned =
          pass.planConflictSplit(seeds ^ map([&](const auto &function) { return pass.mapFn(function); }), conflicted, structDefs);
      if (!planned) throw backend::BackendException("address-space-specialized struct escapes representable storage");
      split = *planned;
    }
    for (const auto &[owner, members] : split.memberRetype)
      if (const auto fields = pass.fields.find(owner); fields != pass.fields.end())
        for (const auto &[member, clone] : members)
          if (const auto field = fields->second.find(member); field != fields->second.end())
            field->second = retypeStructOccurrence(field->second, clone);
    for (const auto &[name, def] : split.clones)
      pass.fields.insert_or_assign(name,
                                   def.members | map([](const auto &member) { return std::pair{member.symbol, member.tpe}; }) | to<Map>());
    const auto reify = [&](const StructDef &def) {
      const auto fields = pass.fields.find(def.name);
      return fields == pass.fields.end() ? def : def.withMembers(def.members ^ map([&](const auto &member) {
                                                                   const auto field = fields->second.find(member.symbol);
                                                                   return field == fields->second.end() ? member
                                                                                                        : member.withTpe(field->second);
                                                                 }));
    };
    auto defs = p.defs ^ map(reify);
    auto cloneDefs = split.clones | values() | map(reify) | to_vector();
    std::sort(cloneDefs.begin(), cloneDefs.end(), [](const auto &lhs, const auto &rhs) { return fqcn(lhs.name) < fqcn(rhs.name); });
    defs ^= concat(cloneDefs);

    const auto remap = [&](const Function &function) { return pass.mapFn(pass.retypeConflicted(argRespace(function), split)); };
    auto entry = p.entry;
    if (entry) entry = remap(*entry);
    auto fns = p.functions ^ map(remap);

    auto sigOf = [](const Expr::Invoke &inv) {
      return Signature(calleeName(inv), /*tpeVars*/ {}, /*receiver*/ {}, inv.args ^ map([](const auto &e) { return e.tpe(); }),
                       /*moduleCaptures*/ {}, /*termCaptures*/ {}, inv.rtn);
    };

    Map<Signature, std::shared_ptr<Function>> functionTable;
    for (const auto &f : fns) {
      const Signature sig(f.decl.name, /*tpeVars*/ {}, /*receiver*/ {}, f.decl.args ^ map([](const auto &e) { return e.named.tpe; }),
                          /*moduleCaptures*/ {},
                          /*termCaptures*/ {}, f.decl.rtn);
      functionTable[sig] = std::make_shared<Function>(f);
    }

    while (true) {
      const auto specialised = functionTable                                                                                      //
                               | flat_map([&](const auto &, const auto &f) { return f->template collect_all<Expr::Invoke>(); })   //
                               | collect([&](const auto &inv) -> std::optional<std::pair<Signature, std::shared_ptr<Function>>> { //
                                   if (const auto sig = sigOf(inv); !(functionTable ^ get_maybe(sig))) {
                                     if (auto spec = functionTable ^ find([&](const auto &lhs, const auto &) {
                                                       return lhs.name == sig.name && lhs.args.size() == sig.args.size();
                                                     })) {
                                       const auto fn = *spec->second;
                                       const auto args =
                                           fn.decl.args                                                                                  //
                                           | zip(sig.args)                                                                               //
                                           | map([](const auto &arg, const auto &tpe) { return arg.withNamed(arg.named.withTpe(tpe)); }) //
                                           | to_vector();

                                       return std::pair{sig, std::make_shared<Function>(pass.mapFn(fn.withDecl(fn.decl.withArgs(args))))};
                                     }
                                   }
                                   return {};
                                 }) //
                               | to<Map>();
      if (specialised.empty()) break;
      functionTable.insert(specialised.begin(), specialised.end());
    }

    const auto spaceSpecialisedName = [](const Sym &name, const std::vector<TypeSpace::Any> &ts) -> Sym {
      auto suffix = ts ^ mk_string("", [&](const auto &s) {
                      return s.match_total([&](TypeSpace::Global) { return "g"; },                                          //
                                           [&](TypeSpace::Constant) { return "g"; }, [&](TypeSpace::Local) { return "l"; }, //
                                           [&](TypeSpace::Private) { return "p"; });
                    });
      auto fqn = name.fqn;
      if (!fqn.empty()) fqn.back() = fqn.back() + "_" + suffix;
      else fqn.push_back("_" + suffix);
      return Sym(fqn);
    };

    auto spaces = [&](const auto &a) { return a.template collect_all<TypeSpace::Any>(); };
    const auto renameInvokes = [&](const Function &f) {
      return f.template modify_all<Expr::Invoke>(
          [&](const auto &inv) { return inv.withCallee(Type::FnRef(spaceSpecialisedName(calleeName(inv), inv.args ^ flat_map(spaces)))); });
    };
    const auto spaceSpecialisedFns =           //
        functionTable                          //
        | values()                             //
        | map([&](const auto &f) -> Function { //
            const auto name = f->convention.template is<CallConvention::OffloadEntry>()
                                  ? f->decl.name
                                  : spaceSpecialisedName(f->decl.name, f->decl.args ^ flat_map(spaces));
            const auto renamed = renameInvokes(*f);
            return renamed.withDecl(renamed.decl.withName(name));
          }) //
        | to_vector();

    if (entry) entry = renameInvokes(*entry);
    return Program(std::move(entry), spaceSpecialisedFns, defs, p.phase, p.metadata);
  }
};

std::string backend::CSource::normalise(const std::string &s) const {
  // a member named `long4` would otherwise make `x.long4__x` parse as an illegal vector swizzle
  static const Set<std::string> reserved = [] {
    Set<std::string> ws = {"global", "local", "kernel", "constant", "private"};
    for (const auto *base : {"char", "uchar", "short", "ushort", "int", "uint", "long", "ulong", "float", "double", "half"})
      for (const auto *width : {"2", "3", "4", "8", "16"})
        ws.emplace(std::string(base) + width);
    return ws;
  }();
  static const Set<std::string> mslReserved = {"device", "threadgroup", "thread"};
  // allowlist non-identifier chars to `_`: a stray `=` from `operator=` parses as an OpenCL assignment
  auto out = s ^ map([](const auto &c) {
               return ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_') ? c : '_';
             });
  // escape whole identifiers only: a symbol merely containing `kernel` must stay verbatim to match the launch lookup
  if ((reserved ^ contains(out)) || (dialect == Dialect::MSL1_0 && (mslReserved ^ contains(out)))) out = "_" + out;
  return out;
}

std::string backend::CSource::normalise(const Sym &s) const { return normalise(fqcn(s)); }

static std::string sourceIdent(const Origin &origin) {
  if (!origin.source) return {};
  const auto ignored = [](const char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n'; };
  const auto first = *origin.source ^ index_where([&](const char c) { return !ignored(c); });
  if (first < 0) return {};
  const auto last = *origin.source ^ last_index_where([&](const char c) { return !ignored(c); });
  const auto s = *origin.source ^ slice(first, last + 1);
  if (s.empty() || (!std::isalpha(static_cast<unsigned char>(s.front())) && s.front() != '_')) return {};
  if (s ^ exists([](const char c) { return !std::isalnum(static_cast<unsigned char>(c)) && c != '_'; })) return {};
  return s;
}

static std::string denseName(size_t n) {
  std::string s;
  do {
    s.insert(s.begin(), "0123456789abcdefghijklmnopqrstuvwxyz"[n % 36]);
    n /= 36;
  } while (n);
  return "_v" + s;
}

std::string backend::CSource::localName(const std::string &symbol) {
  if (const auto it = localNames.find(symbol); it != localNames.end()) return it->second;
  std::string name;
  do
    name = denseName(localNameCounter++);
  while (fileScopeNames ^ contains(name));
  return localNames.emplace(symbol, name).first->second;
}

void backend::CSource::bindLocalNames(const Function &fn) {
  localNames.clear();
  localNameCounter = 0;
  Set<std::string> used = fileScopeNames;
  const bool verbose = std::getenv(polyregion::env::PolycVerboseNames) != nullptr;
  const auto bind = [&](const Named &named) {
    if (localNames.contains(named.symbol)) return;
    auto name = verbose ? sourceIdent(named.origin) : std::string{};
    if (!name.empty()) {
      const auto base = normalise(name);
      name = base;
      for (size_t suffix = 1; used ^ contains(name); ++suffix)
        name = base + "_" + std::to_string(suffix);
    } else {
      do
        name = denseName(localNameCounter++);
      while (used ^ contains(name));
    }
    used.emplace(name);
    localNames.emplace(named.symbol, name);
  };
  for (const auto &arg : fn.decl.args)
    bind(arg.named);
  for (const auto &named : fn.template collect_all<Named>())
    bind(named);
}

Type::Any backend::CSource::resolveFieldType(const Type::Any &owner, const std::string &fieldName) const {
  if (auto s = owner.get<Type::Struct>()) {
    if (auto it = structDefsByName.find(normalise(s->name)); it != structDefsByName.end()) {
      const auto field = normalise(fieldName);
      if (auto m = it->second ^ find([&](const auto &name, const auto &) { return name == field; })) return m->second;
    }
    throw std::logic_error("field " + fieldName + " not found on struct " + repr(s->name));
  }
  throw std::logic_error("field " + fieldName + " selected on non-struct type " + repr(owner));
}

std::string backend::CSource::mkTpe(const Type::Any &tpe) {
  // metal requires an address space on every pointer, struct fields included
  auto mslPtrPrefix = [&](const TypeSpace::Any &space) {
    return space.match_total([&](TypeSpace::Global) { return "device"; },                                                      //
                             [&](TypeSpace::Constant) { return "constant"; }, [&](TypeSpace::Local) { return "threadgroup"; }, //
                             [&](TypeSpace::Private) { return "thread"; }                                                      //
    );
  };
  switch (dialect) {
    case Dialect::C11:
    case Dialect::MSL1_0:
      return tpe.match_total([&](const Type::Float16 &) { return "__fp16"s; }, //
                             [&](const Type::Float32 &) { return "float"s; },  //
                             [&](const Type::Float64 &) { return "double"s; }, //

                             [&](const Type::IntU8 &) { return "uint8_t"s; },   //
                             [&](const Type::IntU16 &) { return "uint16_t"s; }, //
                             [&](const Type::IntU32 &) { return "uint32_t"s; }, //
                             [&](const Type::IntU64 &) { return "uint64_t"s; }, //

                             [&](const Type::IntS8 &) { return "int8_t"s; },   //
                             [&](const Type::IntS16 &) { return "int16_t"s; }, //
                             [&](const Type::IntS32 &) { return "int32_t"s; }, //
                             [&](const Type::IntS64 &) { return "int64_t"s; }, //

                             [&](const Type::Nothing &) { return "/*nothing*/"s; }, //
                             [&](const Type::Unit0 &) { return "void"s; },          //
                             [&](const Type::Bool1 &) { return "bool"s; },          //

                             [&](const Type::Struct &x) { return normalise(x.name); }, //
                             [&](const Type::Ptr &x) {
                               if (x.comp.template is<Type::Nothing>()) {
                                 if (dialect == Dialect::MSL1_0) return fmt::format("{} char*", mslPtrPrefix(x.space));
                                 return "int8_t*"s;
                               }
                               // a pointer to an array needs the `c(*)[n]` form; `c[n]*` is not valid C
                               if (auto arr = x.comp.template get<Type::Arr>(); arr) {
                                 const std::string pfx = dialect == Dialect::MSL1_0 ? std::string(mslPtrPrefix(x.space)) + " " : "";
                                 return fmt::format("{}{} (*)[{}]", pfx, mkTpe(arr->comp), arr->length);
                               }
                               if (dialect == Dialect::MSL1_0) {
                                 // each level qualified at its own `*` (`device T * device *`), else the outer `*` is unqualified
                                 if (x.comp.template is<Type::Ptr>()) return fmt::format("{} {} *", mkTpe(x.comp), mslPtrPrefix(x.space));
                                 return fmt::format("{} {}*", mslPtrPrefix(x.space), mkTpe(x.comp));
                               }
                               return fmt::format("{}*", mkTpe(x.comp));
                             },                                                                                  //
                             [&](const Type::Arr &x) { return fmt::format("{}[{}]", mkTpe(x.comp), x.length); }, //
                             [&](const Type::Var &x) -> std::string { throw std::logic_error("Type::Var should be erased"); },
                             [&](const Type::Exec &x) -> std::string { throw std::logic_error("Type::Exec should be erased"); },
                             [&](const Type::FnRef &x) -> std::string { throw std::logic_error("Type::FnRef should be erased"); });
    case Dialect::OpenCL1_1:
      return tpe.match_total([&](const Type::Float16 &) { return "half"s; },   //
                             [&](const Type::Float32 &) { return "float"s; },  //
                             [&](const Type::Float64 &) { return "double"s; }, //

                             [&](const Type::IntU8 &) { return "uchar"s; },   //
                             [&](const Type::IntU16 &) { return "ushort"s; }, //
                             [&](const Type::IntU32 &) { return "uint"s; },   //
                             [&](const Type::IntU64 &) { return "ulong"s; },  //

                             [&](const Type::IntS8 &) { return "char"s; },   //
                             [&](const Type::IntS16 &) { return "short"s; }, //
                             [&](const Type::IntS32 &) { return "int"s; },   //
                             [&](const Type::IntS64 &) { return "long"s; },  //

                             [&](const Type::Nothing &) { return "/*nothing*/"s; }, //
                             [&](const Type::Unit0 &) { return "void"s; },          //
                             [&](const Type::Bool1 &) { return "char"s; },          //

                             [&](const Type::Struct &x) { return normalise(x.name); }, //
                             [&](const Type::Ptr &x) {
                               auto prefix = x.space.match_total([&](TypeSpace::Global) { return "global"; },     //
                                                                 [&](TypeSpace::Constant) { return "constant"; }, //
                                                                 [&](TypeSpace::Local) { return "local"; },       //
                                                                 [&](TypeSpace::Private) { return "private"; }    //
                               );
                               if (x.comp.template is<Type::Nothing>()) return fmt::format("{} char*", prefix);
                               // a pointer to an array needs the `c(*)[n]` form; `c[n]*` is not valid C
                               if (auto arr = x.comp.template get<Type::Arr>(); arr)
                                 return fmt::format("{} {} (*)[{}]", prefix, mkTpe(arr->comp), arr->length);
                               // each pointer level carries its own space at its own `*`: `global T * global *`
                               // not `global global T**` (the latter leaves the outer `*` private, breaking an arena cast)
                               if (x.comp.template is<Type::Ptr>()) return fmt::format("{} {} *", mkTpe(x.comp), prefix);
                               return fmt::format("{} {}*", prefix, mkTpe(x.comp));
                             }, //
                             // an array carries no own address-space qualifier; it lives in its container's space
                             [&](const Type::Arr &x) { return fmt::format("{}[{}]", mkTpe(x.comp), x.length); }, //
                             [&](const Type::Var &x) -> std::string { throw std::logic_error("Type::Var should be erased"); },
                             [&](const Type::Exec &x) -> std::string { throw std::logic_error("Type::Exec should be erased"); },
                             [&](const Type::FnRef &x) -> std::string { throw std::logic_error("Type::FnRef should be erased"); });
  }
}

std::string backend::CSource::mslPtrSpace(const Term::Any &ptr) const {
  const auto tpe = ptr.tpe().template get<Type::Ptr>();
  if (!tpe) throw BackendException("MSL memory operation requires a pointer operand");
  return tpe->space.match_total(
      [](const TypeSpace::Global &) { return "device"s; }, [](const TypeSpace::Constant &) { return "constant"s; },
      [](const TypeSpace::Local &) { return "threadgroup"s; }, [](const TypeSpace::Private &) { return "thread"s; });
}

std::string backend::CSource::mkArrayDecl(const Type::Any &element, const TypeSpace::Any &space, const std::string &name,
                                          const std::string &extent) {
  std::string dims = fmt::format("[{}]", extent);
  Type::Any base = element;
  while (auto a = base.template get<Type::Arr>()) {
    dims += fmt::format("[{}]", a->length);
    base = a->comp;
  }
  const auto q = space.template is<TypeSpace::Local>() ? dialect == Dialect::MSL1_0 ? "threadgroup " : "local " : "";
  return fmt::format("{}{} {}{}", q, mkTpe(base), name, dims);
}

// a C declarator places array extents AFTER the identifier (`T n[N][M]`), unlike mkTpe
std::string backend::CSource::mkDecl(const Type::Any &tpe, const std::string &name) {
  if (const auto a = tpe.template get<Type::Arr>()) return mkArrayDecl(a->comp, a->space, name, std::to_string(a->length));
  if (auto p = tpe.template get<Type::Ptr>(); p && p->comp.template is<Type::Arr>()) {
    // pointer-to-array `T (*name)[d1][d2]` keeps all pointee extents so `&base[0][idx]` strides by sub-array
    std::string dims;
    Type::Any base = p->comp;
    while (auto a = base.template get<Type::Arr>()) {
      dims += fmt::format("[{}]", a->length);
      base = a->comp;
    }
    const auto q = p->space.match_total([&](TypeSpace::Global) { return dialect == Dialect::MSL1_0 ? "device "s : "global "s; },    //
                                        [&](TypeSpace::Constant) { return "constant "s; },                                          //
                                        [&](TypeSpace::Local) { return dialect == Dialect::MSL1_0 ? "threadgroup "s : "local "s; }, //
                                        [&](TypeSpace::Private) { return dialect == Dialect::MSL1_0 ? "thread "s : "private "s; });
    return fmt::format("{}{} (*{}){}", q, mkTpe(base), name, dims);
  }
  return fmt::format("{} {}", mkTpe(tpe), name);
}

std::string backend::CSource::mkTerm(const Term::Any &term) {
  return term.match_total([](const Term::Float16Const &x) { return cFloatLiteral(x.value, ""); },  //
                          [](const Term::Float32Const &x) { return cFloatLiteral(x.value, "f"); }, //
                          [](const Term::Float64Const &x) { return cFloatLiteral(x.value, ""); },  //

                          [](const Term::IntU8Const &x) { return fmt::format("{}", x.value); },  //
                          [](const Term::IntU16Const &x) { return fmt::format("{}", x.value); }, //
                          [](const Term::IntU32Const &x) { return fmt::format("{}", x.value); }, //
                          [](const Term::IntU64Const &x) { return fmt::format("{}", x.value); }, //

                          [](const Term::IntS8Const &x) { return fmt::format("{}", x.value); },  //
                          [](const Term::IntS16Const &x) { return fmt::format("{}", x.value); }, //
                          [](const Term::IntS32Const &x) { return fmt::format("{}", x.value); }, //
                          [](const Term::IntS64Const &x) { return fmt::format("{}", x.value); }, //

                          [](const Term::Unit0Const &) { return "/*void*/"s; },                   //
                          [](const Term::Bool1Const &x) { return x.value ? "true"s : "false"s; }, //
                          [](const Term::NullPtrConst &) { return "0"s; },                        //
                          [&](const Term::Poison &x) {
                            // `0` not `NULL`: comgr doesn't predefine NULL for AMD kernel sources (non-ptr poison still casts)
                            if (x.tpe.is<Type::Ptr>()) return fmt::format("(0 /*{}*/)", repr(x.tpe));
                            return fmt::format("(({})0 /*poison {}*/)", mkTpe(x.tpe), repr(x.tpe));
                          }, //
                          [&](const Term::StringConst &x) {
                            // an inline OpenCL literal has no addressable storage so it must be referenced by name
                            return stringConstNames ^ get_or_default(x.value, fmt::format("\"{}\"", escapeCString(x.value)));
                          }, //
                          [&](const Term::Select &x) {
                            std::string acc = localName(x.root.symbol);
                            // the AST omits the implicit deref of a Field through a pointer; insert `(*...)` here
                            Type::Any current = x.root.tpe;
                            for (auto &step : x.steps) {
                              step.match_total(
                                  [&](const PathStep::Field &f) {
                                    if (auto p = current.template get<Type::Ptr>()) {
                                      acc = "(*" + acc + ")";
                                      current = p->comp;
                                    }
                                    acc += ".";
                                    acc += normalise(f.name);
                                    current = resolveFieldType(current, f.name);
                                  },
                                  [&](const PathStep::Deref &) {
                                    acc = "(*" + acc + ")";
                                    if (auto p = current.template get<Type::Ptr>()) current = p->comp;
                                  },
                                  [&](const PathStep::Index &i) {
                                    acc += "[" + std::to_string(i.idx) + "]";
                                    if (auto p = current.template get<Type::Ptr>()) current = p->comp;
                                    else if (auto a = current.template get<Type::Arr>()) current = a->comp;
                                  },
                                  [&](const PathStep::IndexDyn &i) {
                                    acc += "[" + mkTerm(i.idx) + "]";
                                    if (auto p = current.template get<Type::Ptr>()) current = p->comp;
                                    else if (auto a = current.template get<Type::Arr>()) current = a->comp;
                                  });
                            }
                            return acc;
                          });
}

std::string backend::CSource::mkExpr(const Expr::Any &expr) {
  return expr.match_total(
      [&](const Expr::Alias &x) { return mkTerm(x.ref); },
      [&](const Expr::SpecOp &x) {
        struct DialectAccessor {
          std::string c11, cl, msl;
        };
        const auto gpuIntr = [&](const DialectAccessor &accessor) -> std::string {
          switch (dialect) {
            case Dialect::C11: return accessor.c11;
            case Dialect::MSL1_0: return accessor.msl;
            case Dialect::OpenCL1_1: return accessor.cl;
          }
        };
        const auto gpuDimIntr = [&](const DialectAccessor &accessor, const Term::Any &index) -> std::string {
          switch (dialect) {
            case Dialect::C11: return accessor.c11;
            case Dialect::MSL1_0: return fmt::format("{}[{}]", accessor.msl, mkTerm(index));
            case Dialect::OpenCL1_1: return fmt::format("{}({})", accessor.cl, mkTerm(index));
          }
        };
        return x.op.match_total(
            [&](const Spec::Assert &v) -> std::string {
              throw BackendException("assert reached codegen; the StructuredExit pass must run before the backend");
            }, //
            [&](const Spec::GpuBarrierGlobal &v) {
              return gpuIntr({.c11 = "((void)0)",
                              .cl = "barrier(CLK_GLOBAL_MEM_FENCE)", //
                              .msl = "threadgroup_barrier(metal::mem_flags::mem_device)"});
            },
            [&](const Spec::GpuBarrierLocal &v) {
              return gpuIntr({.c11 = "((void)0)",
                              .cl = "barrier(CLK_LOCAL_MEM_FENCE)", //
                              .msl = "threadgroup_barrier(metal::mem_flags::mem_threadgroup)"});
            },
            [&](const Spec::GpuBarrierAll &v) {
              return gpuIntr({.c11 = "((void)0)",
                              .cl = "barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE)", //
                              .msl = "threadgroup_barrier(metal::mem_flags::mem_threadgroup | metal::mem_flags::mem_device)"});
            },
            [&](const Spec::GpuFenceGlobal &v) {
              return gpuIntr({.c11 = "((void)0)",
                              .cl = "mem_fence(CLK_GLOBAL_MEM_FENCE)", //
                              .msl = "threadgroup_barrier(metal::mem_flags::mem_device)"});
            },
            [&](const Spec::GpuFenceLocal &v) {
              return gpuIntr({.c11 = "((void)0)",
                              .cl = "mem_fence(CLK_LOCAL_MEM_FENCE)", //
                              .msl = "threadgroup_barrier(metal::mem_flags::mem_threadgroup)"});
            },
            [&](const Spec::GpuFenceAll &v) {
              return gpuIntr({.c11 = "((void)0)",
                              .cl = "mem_fence(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE)", //
                              .msl = "threadgroup_barrier(metal::mem_flags::mem_device)"});
            },
            [&](const Spec::GpuGlobalIdx &v) { return gpuDimIntr({.c11 = "0", .cl = "get_global_id", .msl = "__get_global_id__"}, v.dim); },
            [&](const Spec::GpuGlobalSize &v) {
              return gpuDimIntr({.c11 = "1", .cl = "get_global_size", .msl = "__get_global_size__"}, v.dim);
            },
            [&](const Spec::GpuGroupIdx &v) { return gpuDimIntr({.c11 = "0", .cl = "get_group_id", .msl = "__get_group_id__"}, v.dim); },
            [&](const Spec::GpuGroupSize &v) {
              return gpuDimIntr({.c11 = "1", .cl = "get_num_groups", .msl = "__get_num_groups__"}, v.dim);
            },
            [&](const Spec::GpuLocalIdx &v) { return gpuDimIntr({.c11 = "0", .cl = "get_local_id", .msl = "__get_local_id__"}, v.dim); },
            [&](const Spec::GpuLocalSize &v) {
              return gpuDimIntr({.c11 = "1", .cl = "get_local_size", .msl = "__get_local_size__"}, v.dim);
            },
            [&](const Spec::GpuLaneIdx &) -> std::string {
              if (dialect == Dialect::C11) return "0";
              throw BackendException("Spec::GpuLaneIdx requires SubgroupLower");
            },
            [&](const Spec::GpuSubgroupSize &) -> std::string {
              if (dialect == Dialect::C11) return "1";
              throw BackendException("Spec::GpuSubgroupSize requires SubgroupLower");
            },
            [&](const Spec::GpuShuffleDown &v) -> std::string {
              if (dialect == Dialect::C11) return mkTerm(v.value);
              throw BackendException("Spec::GpuShuffleDown requires SubgroupLower");
            },
            [&](const Spec::GpuShuffleUp &v) -> std::string {
              if (dialect == Dialect::C11) return mkTerm(v.value);
              throw BackendException("Spec::GpuShuffleUp requires SubgroupLower");
            },
            [&](const Spec::GpuShuffleIdx &v) -> std::string {
              if (dialect == Dialect::C11) return mkTerm(v.value);
              throw BackendException("Spec::GpuShuffleIdx requires SubgroupLower");
            },
            [&](const Spec::GpuShuffleXor &v) -> std::string {
              if (dialect == Dialect::C11) return mkTerm(v.value);
              throw BackendException("Spec::GpuShuffleXor requires SubgroupLower");
            },
            [&](const Spec::GpuSubgroupBarrier &) -> std::string {
              if (dialect == Dialect::C11) return "((void)0)";
              if (dialect == Dialect::OpenCL1_1) return "sub_group_barrier(CLK_LOCAL_MEM_FENCE)";
              throw BackendException("Spec::GpuSubgroupBarrier is unsupported for this C source dialect");
            },
            [&](const Spec::GpuBallot &v) -> std::string {
              if (dialect == Dialect::C11) return fmt::format("(({} && (({} & 1u) != 0u)) ? 1u : 0u)", mkTerm(v.pred), mkTerm(v.mask));
              throw BackendException("Spec::GpuBallot requires SubgroupLower");
            },
            [&](const Spec::GpuVoteAny &v) -> std::string {
              if (dialect == Dialect::C11) return fmt::format("({} && (({} & 1u) != 0u))", mkTerm(v.pred), mkTerm(v.mask));
              throw BackendException("Spec::GpuVoteAny requires SubgroupLower");
            },
            [&](const Spec::GpuVoteAll &v) -> std::string {
              if (dialect == Dialect::C11) return fmt::format("({} || (({} & 1u) == 0u))", mkTerm(v.pred), mkTerm(v.mask));
              throw BackendException("Spec::GpuVoteAll requires SubgroupLower");
            },
            [&](const Spec::GpuAtomicRMW &v) -> std::string {
              if (dialect == Dialect::C11) {
                if (!v.rtn.template is<Type::IntS32>() && !v.rtn.template is<Type::IntU32>())
                  throw BackendException("C11 supports only 32-bit integer atomic RMW");
                if (!v.order.template is<MemOrder::Relaxed>()) throw BackendException("C11 atomic RMW supports only relaxed ordering");
                const auto type = mkTpe(v.rtn), ptr = mkTerm(v.ptr), value = mkTerm(v.value);
                return v.op.match_total(
                    [&](const AtomicOp::Xchg &) {
                      return fmt::format("atomic_exchange_explicit((volatile _Atomic({})*){}, ({}){}, memory_order_relaxed)", type, ptr,
                                         type, value);
                    },
                    [&](const AtomicOp::Add &) {
                      return fmt::format("atomic_fetch_add_explicit((volatile _Atomic({})*){}, ({}){}, memory_order_relaxed)", type, ptr,
                                         type, value);
                    },
                    [&](const AtomicOp::Sub &) {
                      return fmt::format("atomic_fetch_sub_explicit((volatile _Atomic({})*){}, ({}){}, memory_order_relaxed)", type, ptr,
                                         type, value);
                    },
                    [&](const AtomicOp::And &) {
                      return fmt::format("atomic_fetch_and_explicit((volatile _Atomic({})*){}, ({}){}, memory_order_relaxed)", type, ptr,
                                         type, value);
                    },
                    [&](const AtomicOp::Or &) {
                      return fmt::format("atomic_fetch_or_explicit((volatile _Atomic({})*){}, ({}){}, memory_order_relaxed)", type, ptr,
                                         type, value);
                    },
                    [&](const AtomicOp::Xor &) {
                      return fmt::format("atomic_fetch_xor_explicit((volatile _Atomic({})*){}, ({}){}, memory_order_relaxed)", type, ptr,
                                         type, value);
                    },
                    [&](const AtomicOp::Min &) {
                      return fmt::format("{}((volatile _Atomic({})*){}, ({}){})", atomicMinMaxHelperName(true, type), type, ptr, type,
                                         value);
                    },
                    [&](const AtomicOp::Max &) {
                      return fmt::format("{}((volatile _Atomic({})*){}, ({}){})", atomicMinMaxHelperName(false, type), type, ptr, type,
                                         value);
                    });
              }
              if (dialect == Dialect::OpenCL1_1) {
                if (!v.rtn.template is<Type::IntS32>() && !v.rtn.template is<Type::IntU32>())
                  throw BackendException("OpenCL 1.1 supports only 32-bit integer atomic RMW");
                if (!v.order.template is<MemOrder::Relaxed>())
                  throw BackendException("OpenCL 1.1 atomic RMW supports only relaxed ordering");
                const auto p = v.ptr.tpe().template get<Type::Ptr>();
                if (!p) throw BackendException("OpenCL atomic RMW requires a pointer operand");
                const auto space = p->space.match_total(
                    [](const TypeSpace::Global &) { return "global"s; }, [](const TypeSpace::Constant &) { return "constant"s; },
                    [](const TypeSpace::Local &) { return "local"s; }, [](const TypeSpace::Private &) { return "private"s; });
                if (space != "global" && space != "local") throw BackendException("OpenCL atomic RMW requires global or local storage");
                const auto function = v.op.match_total(
                    [](const AtomicOp::Xchg &) { return "atomic_xchg"s; }, [](const AtomicOp::Add &) { return "atomic_add"s; },
                    [](const AtomicOp::Sub &) { return "atomic_sub"s; }, [](const AtomicOp::And &) { return "atomic_and"s; },
                    [](const AtomicOp::Or &) { return "atomic_or"s; }, [](const AtomicOp::Xor &) { return "atomic_xor"s; },
                    [](const AtomicOp::Min &) { return "atomic_min"s; }, [](const AtomicOp::Max &) { return "atomic_max"s; });
                const auto type = mkTpe(v.rtn);
                return fmt::format("{}((volatile {} {}*){}, ({}){})", function, space, type, mkTerm(v.ptr), type, mkTerm(v.value));
              }
              if (!v.rtn.template is<Type::IntS32>() && !v.rtn.template is<Type::IntU32>())
                throw BackendException("MSL supports only 32-bit integer atomic RMW");
              const auto space = mslPtrSpace(v.ptr);
              if (space != "device" && space != "threadgroup")
                throw BackendException("MSL atomic RMW requires device or threadgroup storage");
              const auto function = v.op.match_total([](const AtomicOp::Xchg &) { return "atomic_exchange_explicit"s; },
                                                     [](const AtomicOp::Add &) { return "atomic_fetch_add_explicit"s; },
                                                     [](const AtomicOp::Sub &) { return "atomic_fetch_sub_explicit"s; },
                                                     [](const AtomicOp::And &) { return "atomic_fetch_and_explicit"s; },
                                                     [](const AtomicOp::Or &) { return "atomic_fetch_or_explicit"s; },
                                                     [](const AtomicOp::Xor &) { return "atomic_fetch_xor_explicit"s; },
                                                     [](const AtomicOp::Min &) { return "atomic_fetch_min_explicit"s; },
                                                     [](const AtomicOp::Max &) { return "atomic_fetch_max_explicit"s; });
              if (!v.order.template is<MemOrder::Relaxed>()) throw BackendException("MSL atomic RMW supports only relaxed ordering");
              const auto atomic = v.rtn.template is<Type::IntU32>() ? "metal::atomic_uint" : "metal::atomic_int";
              const auto value = v.rtn.template is<Type::IntU32>() ? "uint32_t" : "int32_t";
              return fmt::format("metal::{}(({} {}*){}, ({}){}, metal::memory_order_relaxed)", function, space, atomic, mkTerm(v.ptr),
                                 value, mkTerm(v.value));
            },
            [&](const Spec::GpuAtomicCAS &) -> std::string {
              throw BackendException("Spec::GpuAtomicCAS lowering is not available for this C source dialect");
            },
            [&](const Spec::GpuGroupReduce &) -> std::string {
              throw BackendException("Spec::GpuGroupReduce lowering is not available for this C source dialect");
            },
            [&](const Spec::GpuGroupInclusiveScan &) -> std::string {
              throw BackendException("Spec::GpuGroupInclusiveScan lowering is not available for this C source dialect");
            },
            [&](const Spec::GpuGroupExclusiveScan &) -> std::string {
              throw BackendException("Spec::GpuGroupExclusiveScan lowering is not available for this C source dialect");
            },
            [&](const Spec::RemoteLaunch &) -> std::string {
              throw BackendException("Spec::RemoteLaunch is a local orchestration operation");
            },
            [&](const Spec::RemoteAlloc &) -> std::string {
              throw BackendException("Spec::RemoteAlloc is a local orchestration operation");
            },
            [&](const Spec::RemoteFree &) -> std::string { throw BackendException("Spec::RemoteFree is a local orchestration operation"); },
            [&](const Spec::RemoteMemcpy &) -> std::string {
              throw BackendException("Spec::RemoteMemcpy is a local orchestration operation");
            },
            [&](const Spec::RemoteSync &) -> std::string { throw BackendException("Spec::RemoteSync is a local orchestration operation"); },
            [&](const Spec::GpuVolatileLoad &v) -> std::string {
              const auto ptr = mkTerm(v.ptr), type = mkTpe(v.rtn);
              if (dialect == Dialect::MSL1_0) {
                const auto space = mslPtrSpace(v.ptr);
                if (v.rtn.template is<Type::Struct>())
                  return fmt::format("{}((volatile {} {}*){})", volatileHelperName(true, space, type), space, type, ptr);
                return fmt::format("(*((volatile {} {}*){}))", space, type, ptr);
              }
              if (dialect == Dialect::OpenCL1_1) {
                const auto p = v.ptr.tpe().template get<Type::Ptr>();
                if (!p) throw BackendException("volatile load requires a pointer operand");
                const auto space = p->space.match_total(
                    [](const TypeSpace::Global &) { return "global"s; }, [](const TypeSpace::Constant &) { return "constant"s; },
                    [](const TypeSpace::Local &) { return "local"s; }, [](const TypeSpace::Private &) { return "private"s; });
                return fmt::format("(*((volatile {} {}*){}))", space, type, ptr);
              }
              return fmt::format("(*((volatile {}*){}))", type, ptr);
            },
            [&](const Spec::GpuVolatileStore &v) -> std::string {
              const auto ptr = mkTerm(v.ptr), value = mkTerm(v.value), type = mkTpe(v.value.tpe());
              if (dialect == Dialect::MSL1_0) {
                const auto space = mslPtrSpace(v.ptr);
                if (space == "constant") throw BackendException("volatile store to constant storage is unsupported for MSL");
                if (v.value.tpe().template is<Type::Struct>())
                  return fmt::format("{}((volatile {} {}*){}, {})", volatileHelperName(false, space, type), space, type, ptr, value);
                return fmt::format("(*((volatile {} {}*){}) = {})", space, type, ptr, value);
              }
              if (dialect == Dialect::OpenCL1_1) {
                const auto p = v.ptr.tpe().template get<Type::Ptr>();
                if (!p) throw BackendException("volatile store requires a pointer operand");
                if (p->space.template is<TypeSpace::Constant>())
                  throw BackendException("volatile store to constant storage is unsupported for OpenCL");
                const auto space = p->space.match_total(
                    [](const TypeSpace::Global &) { return "global"s; }, [](const TypeSpace::Constant &) { return "constant"s; },
                    [](const TypeSpace::Local &) { return "local"s; }, [](const TypeSpace::Private &) { return "private"s; });
                return fmt::format("(*((volatile {} {}*){}) = {})", space, type, ptr, value);
              }
              return fmt::format("(*((volatile {}*){}) = {})", type, ptr, value);
            } //
        );
      },
      [&](const Expr::IntrOp &x) {
        const auto intrFn = [&](std::string_view name) {
          return dialect == Dialect::MSL1_0 ? "metal::" + std::string(name) : std::string(name);
        };
        return x.op.match_total([&](const Intr::Pos &v) { return fmt::format("(+{})", mkTerm(v.x)); },
                                [&](const Intr::Neg &v) { return fmt::format("(-{})", mkTerm(v.x)); },
                                [&](const Intr::BNot &v) { return fmt::format("(~{})", mkTerm(v.x)); },
                                [&](const Intr::LogicNot &v) { return fmt::format("(!{})", mkTerm(v.x)); },
                                [&](const Intr::Add &v) { return fmt::format("({} + {})", mkTerm(v.x), mkTerm(v.y)); },
                                [&](const Intr::Sub &v) { return fmt::format("({} - {})", mkTerm(v.x), mkTerm(v.y)); },
                                [&](const Intr::Mul &v) { return fmt::format("({} * {})", mkTerm(v.x), mkTerm(v.y)); },
                                [&](const Intr::Div &v) { return fmt::format("({} / {})", mkTerm(v.x), mkTerm(v.y)); },
                                [&](const Intr::Rem &v) { return fmt::format("({} % {})", mkTerm(v.x), mkTerm(v.y)); },
                                [&](const Intr::Min &v) { return fmt::format("{}({}, {})", intrFn("min"), mkTerm(v.x), mkTerm(v.y)); },
                                [&](const Intr::Max &v) { return fmt::format("{}({}, {})", intrFn("max"), mkTerm(v.x), mkTerm(v.y)); },
                                [&](const Intr::BAnd &v) { return fmt::format("({} & {})", mkTerm(v.x), mkTerm(v.y)); },
                                [&](const Intr::BOr &v) { return fmt::format("({} | {})", mkTerm(v.x), mkTerm(v.y)); },
                                [&](const Intr::BXor &v) { return fmt::format("({} ^ {})", mkTerm(v.x), mkTerm(v.y)); },
                                [&](const Intr::BSL &v) { return fmt::format("({} << {})", mkTerm(v.x), mkTerm(v.y)); },
                                [&](const Intr::BSR &v) { return fmt::format("({} >> {})", mkTerm(v.x), mkTerm(v.y)); },
                                [&](const Intr::BZSR &v) { return fmt::format("({} >> {})", mkTerm(v.x), mkTerm(v.y)); },
                                [&](const Intr::PopCount &v) {
                                  if (dialect == Dialect::MSL1_0) return fmt::format("metal::popcount({})", mkTerm(v.x));
                                  const auto unsignedType = [&]() -> Type::Any {
                                    if (v.rtn.template is<Type::IntU8>() || v.rtn.template is<Type::IntS8>()) return Type::IntU8();
                                    if (v.rtn.template is<Type::IntU16>() || v.rtn.template is<Type::IntS16>()) return Type::IntU16();
                                    if (v.rtn.template is<Type::IntU32>() || v.rtn.template is<Type::IntS32>()) return Type::IntU32();
                                    if (v.rtn.template is<Type::IntU64>() || v.rtn.template is<Type::IntS64>()) return Type::IntU64();
                                    throw BackendException("popcount requires an integral operand");
                                  }();
                                  const bool wide = unsignedType.template is<Type::IntU64>();
                                  const auto helperType = wide ? Type::IntU64().widen() : Type::IntU32().widen();
                                  return fmt::format("(({}) POLY_POPCOUNT{}(({}) (({}) {})))", mkTpe(v.rtn), wide ? "64" : "32",
                                                     mkTpe(helperType), mkTpe(unsignedType), mkTerm(v.x));
                                },
                                [&](const Intr::LogicAnd &v) { return fmt::format("({} && {})", mkTerm(v.x), mkTerm(v.y)); },
                                [&](const Intr::LogicOr &v) { return fmt::format("({} || {})", mkTerm(v.x), mkTerm(v.y)); },
                                [&](const Intr::LogicEq &v) { return fmt::format("({} == {})", mkTerm(v.x), mkTerm(v.y)); },
                                [&](const Intr::LogicNeq &v) { return fmt::format("({} != {})", mkTerm(v.x), mkTerm(v.y)); },
                                [&](const Intr::LogicLte &v) { return fmt::format("({} <= {})", mkTerm(v.x), mkTerm(v.y)); },
                                [&](const Intr::LogicGte &v) { return fmt::format("({} >= {})", mkTerm(v.x), mkTerm(v.y)); },
                                [&](const Intr::LogicLt &v) { return fmt::format("({} < {})", mkTerm(v.x), mkTerm(v.y)); },
                                [&](const Intr::LogicGt &v) { return fmt::format("({} > {})", mkTerm(v.x), mkTerm(v.y)); });
      },
      [&](const Expr::MathOp &x) {
        const auto fp = [](const Type::Any &t) {
          return t.template is<Type::Float16>() || t.template is<Type::Float32>() || t.template is<Type::Float64>();
        };
        const auto mathFn = [&](std::string_view name) {
          return dialect == Dialect::MSL1_0 ? "metal::" + std::string(name) : std::string(name);
        };
        return x.op.match_total(
            // OpenCL/C `abs` is integer-only; floats need `fabs`
            [&](const Math::Abs &v) { return fmt::format("{}({})", mathFn(fp(v.tpe) ? "fabs" : "abs"), mkTerm(v.x)); },
            // POLY_* macro on OpenCL: llvmpipe libclc crashes on precise sin/cos/tan range-reduction
            [&](const Math::Sin &v) {
              return fmt::format("{}({})", dialect == Dialect::OpenCL1_1 ? "POLY_SIN" : mathFn("sin"), mkTerm(v.x));
            },
            [&](const Math::Cos &v) {
              return fmt::format("{}({})", dialect == Dialect::OpenCL1_1 ? "POLY_COS" : mathFn("cos"), mkTerm(v.x));
            },
            [&](const Math::Tan &v) {
              return fmt::format("{}({})", dialect == Dialect::OpenCL1_1 ? "POLY_TAN" : mathFn("tan"), mkTerm(v.x));
            },
            [&](const Math::Asin &v) { return fmt::format("{}({})", mathFn("asin"), mkTerm(v.x)); },
            [&](const Math::Acos &v) { return fmt::format("{}({})", mathFn("acos"), mkTerm(v.x)); },
            [&](const Math::Atan &v) { return fmt::format("{}({})", mathFn("atan"), mkTerm(v.x)); },
            [&](const Math::Sinh &v) { return fmt::format("{}({})", mathFn("sinh"), mkTerm(v.x)); },
            [&](const Math::Cosh &v) { return fmt::format("{}({})", mathFn("cosh"), mkTerm(v.x)); },
            [&](const Math::Tanh &v) { return fmt::format("{}({})", mathFn("tanh"), mkTerm(v.x)); },
            [&](const Math::Signum &v) { return fmt::format("{}({})", mathFn("signum"), mkTerm(v.x)); },
            [&](const Math::Round &v) { return fmt::format("{}({})", mathFn("round"), mkTerm(v.x)); },
            [&](const Math::Ceil &v) { return fmt::format("{}({})", mathFn("ceil"), mkTerm(v.x)); },
            [&](const Math::Floor &v) { return fmt::format("{}({})", mathFn("floor"), mkTerm(v.x)); },
            [&](const Math::Rint &v) { return fmt::format("{}({})", mathFn("rint"), mkTerm(v.x)); },
            [&](const Math::Sqrt &v) { return fmt::format("{}({})", mathFn("sqrt"), mkTerm(v.x)); },
            [&](const Math::Cbrt &v) { return fmt::format("{}({})", mathFn("cbrt"), mkTerm(v.x)); },
            [&](const Math::Exp &v) { return fmt::format("{}({})", mathFn("exp"), mkTerm(v.x)); },
            [&](const Math::Expm1 &v) { return fmt::format("{}({})", mathFn("expm1"), mkTerm(v.x)); },
            [&](const Math::Log &v) { return fmt::format("{}({})", mathFn("log"), mkTerm(v.x)); },
            [&](const Math::Log1p &v) { return fmt::format("{}({})", mathFn("log1p"), mkTerm(v.x)); },
            [&](const Math::Log10 &v) { return fmt::format("{}({})", mathFn("log10"), mkTerm(v.x)); },
            [&](const Math::Pow &v) { return fmt::format("{}({}, {})", mathFn("pow"), mkTerm(v.x), mkTerm(v.y)); },
            [&](const Math::Atan2 &v) { return fmt::format("{}({}, {})", mathFn("atan2"), mkTerm(v.x), mkTerm(v.y)); },
            [&](const Math::Hypot &v) { return fmt::format("{}({}, {})", mathFn("hypot"), mkTerm(v.x), mkTerm(v.y)); });
      },
      [&](const Expr::Cast &x) { return fmt::format("(({}) {})", mkTpe(x.as), mkTerm(x.from)); },
      [&](const Expr::Invoke &x) {
        return fmt::format("{}({})", normalise(calleeName(x)), x.args ^ mk_string(", ", [&](const auto &arg) { return mkTerm(arg); }));
      }, //
      [&](const Expr::Index &x) { return fmt::format("{}[{}]", mkTerm(x.lhs), mkTerm(x.idx)); },
      [&](const Expr::RefTo &x) {
        // pointer-to-array: `&base[0][idx]` matches the mkDecl declarator; `&base[idx]` would stride by sub-array
        if (x.idx)
          if (auto pt = x.lhs.tpe().template get<Type::Ptr>(); pt && pt->comp.template is<Type::Arr>())
            return fmt::format("&({})[0][{}]", mkTerm(x.lhs), mkTerm(*x.idx));
        std::string str;
        // OpenCL C has no C++ temporary materialisation: `&(1)` is invalid.  A C99
        // compound literal provides the required private lvalue and preserves RefTo's
        // lifetime for the enclosing block.
        if (dialect == Dialect::OpenCL1_1 && !x.lhs.template is<Term::Select>() && !x.lhs.template is<Term::StringConst>())
          str = fmt::format("&((private {}){{{}}})", mkTpe(x.comp), mkTerm(x.lhs));
        else str = fmt::format("&({} /*{}*/)", mkTerm(x.lhs), mkTpe(x.comp));
        // a value lhs would make `&value[idx]` illegal C, so drop the idx; a pointer/array lhs keeps it
        const bool valueLhs = !x.lhs.tpe().template is<Type::Ptr>() && !x.lhs.tpe().template is<Type::Arr>();
        if (x.idx && !valueLhs) str += fmt::format("[{}]", mkTerm(*x.idx));
        const auto lhsSel = x.lhs.template get<Term::Select>();
        const auto lastField = lhsSel && !lhsSel->steps.empty() ? lhsSel->steps.back().template get<PathStep::Field>() : std::nullopt;
        // EBO empty base addressed as the base type: cast to the declared pointer type so Rusticl accepts it
        if (lastField && (lastField->name ^ starts_with(conventions::BaseFieldPrefix)) && x.comp.template is<Type::Struct>()) {
          // an empty base is elided from the struct, so `&obj.#base_X` is dangling: address the parent (offset
          // 0) instead. the Select is typed as the logical base, so key elision on the field's owner-declared
          // type (#empty for an EBO base), not the Select type
          Type::Any owner = lhsSel->root.tpe;
          std::vector<bool> elidedBaseSteps;
          for (const auto &step : lhsSel->steps)
            step.match_total(
                [&](const PathStep::Field &f) {
                  if (auto p = owner.template get<Type::Ptr>()) owner = p->comp;
                  const auto selected = resolveFieldType(owner, f.name);
                  const bool elided = (f.name ^ starts_with(conventions::BaseFieldPrefix))
                                      && (selected.template get<Type::Struct>() //
                                          ^ exists([&](const auto &s) { return zeroSizeStructNames ^ contains(normalise(s.name)); }));
                  elidedBaseSteps.emplace_back(elided);
                  owner = selected;
                },
                [&](const PathStep::Deref &) {
                  elidedBaseSteps.emplace_back(false);
                  if (auto p = owner.template get<Type::Ptr>()) owner = p->comp;
                },
                [&](const PathStep::Index &) {
                  elidedBaseSteps.emplace_back(false);
                  if (auto p = owner.template get<Type::Ptr>()) owner = p->comp;
                  else if (auto a = owner.template get<Type::Arr>()) owner = a->comp;
                },
                [&](const PathStep::IndexDyn &) {
                  elidedBaseSteps.emplace_back(false);
                  if (auto p = owner.template get<Type::Ptr>()) owner = p->comp;
                  else if (auto a = owner.template get<Type::Arr>()) owner = a->comp;
                });
          if (!elidedBaseSteps.empty() && elidedBaseSteps.back()) {
            auto parent = mkTerm(x.lhs);
            for (auto it = elidedBaseSteps.rbegin(); it != elidedBaseSteps.rend() && *it; ++it) {
              const auto cut = parent ^ last_index_of('.');
              if (cut < 0) break;
              parent.resize(cut);
            }
            str = fmt::format("&({})", parent);
          }
          str = fmt::format("(({}) {})", mkTpe(Type::Ptr(x.comp, x.space).widen()), str);
        }
        return str;
      },
      [&](const Expr::Alloc &x) { return fmt::format("{{/*{}*/}}", to_string(x)); },
      [&](const Expr::ForeignCall &x) {
        return fmt::format("{}({})", x.name, x.args ^ mk_string(", ", [&](const auto &arg) { return mkTerm(arg); }));
      },
      [&](const Expr::OffsetOf &x) { return fmt::format("__builtin_offsetof({}, {})", mkTpe(x.structTpe), normalise(x.field)); },
      [&](const Expr::SizeOf &x) { return fmt::format("sizeof({})", mkTpe(x.forTpe)); });
}

// C/OpenCL forbid whole-array assignment; copy element-wise (a nested loop per array level)
std::string backend::CSource::mkValueCopy(const std::string &lhs, const std::string &rhs, const Type::Any &tpe, int depth) const {
  if (auto a = tpe.template get<Type::Arr>()) {
    const auto i = fmt::format("_ac{}", depth);
    return fmt::format("for (int {} = 0; {} < {}; {}++) {{ {} }}", i, i, a->length, i,
                       mkValueCopy(fmt::format("{}[{}]", lhs, i), fmt::format("{}[{}]", rhs, i), a->comp, depth + 1));
  }
  // XXX rusticl zeroes a whole-struct read of a private var in a loop, so copy scalar leaves:
  // XXX   S s; s.off = x; for (...) { S t = s; }  ->  t.off reads back 0
  if (auto s = tpe.template get<Type::Struct>(); s && dialect == Dialect::OpenCL1_1) {
    const auto name = normalise(s->name);
    // a populated union stays whole; naming its members would not preserve the active one
    // a zero-size member has no storage in the emitted body, so it must not be named
    if (auto it = structDefsByName.find(name); it != structDefsByName.end() && (it->second.empty() || !(unionDefNames ^ contains(name))))
      return it->second | filter([&](const auto &m) {
               return !m.second.template is<Type::FnRef>() && !(m.second.template get<Type::Struct>() ^ exists([&](const auto &nested) {
                                                                  return zeroSizeStructNames ^ contains(normalise(nested.name));
                                                                }));
             })
             | map([&](const auto &m) {
                 return mkValueCopy(fmt::format("{}.{}", lhs, m.first), fmt::format("{}.{}", rhs, m.first), m.second, depth);
               })                                                     //
             | filter([](const auto &copy) { return !copy.empty(); }) //
             | mk_string(" ");
  }
  return fmt::format("{} = {};", lhs, rhs);
}

std::string backend::CSource::mkVolatileCopy(const std::string &lhs, const std::string &rhs, const Type::Any &tpe, int depth) const {
  if (const auto array = tpe.template get<Type::Arr>()) {
    const auto index = fmt::format("_vc{}", depth);
    return fmt::format("for (int {} = 0; {} < {}; {}++) {{ {} }}", index, index, array->length, index,
                       mkVolatileCopy(fmt::format("{}[{}]", lhs, index), fmt::format("{}[{}]", rhs, index), array->comp, depth + 1));
  }
  if (const auto structure = tpe.template get<Type::Struct>()) {
    const auto name = normalise(structure->name);
    if (unionDefNames ^ contains(name))
      throw BackendException("volatile access to union " + repr(structure->name) + " is unsupported for MSL");
    const auto members = structDefsByName.find(name);
    if (members == structDefsByName.end()) throw BackendException("volatile access to undeclared struct " + repr(structure->name));
    return members->second | filter([&](const auto &member) {
             return !member.second.template is<Type::FnRef>()
                    && !(member.second.template get<Type::Struct>()
                         ^ exists([&](const auto &nested) { return zeroSizeStructNames ^ contains(normalise(nested.name)); }));
           })
           | map([&](const auto &member) {
               return mkVolatileCopy(fmt::format("{}.{}", lhs, member.first), fmt::format("{}.{}", rhs, member.first), member.second,
                                     depth + 1);
             })
           | mk_string(" ");
  }
  return fmt::format("{} = {};", lhs, rhs);
}

std::string backend::CSource::mkVolatileHelper(const bool load, const Type::Any &tpe, const std::string &space) {
  const auto element = mkTpe(tpe), name = volatileHelperName(load, space, element);
  if (load)
    return fmt::format("{} {}(volatile {} {} *p) {{\n  {} r;\n  {}\n  return r;\n}}", element, name, space, element, element,
                       mkVolatileCopy("r", "(*p)", tpe, 0));
  return fmt::format("void {}(volatile {} {} *p, {} v) {{\n  {}\n}}", name, space, element, element, mkVolatileCopy("(*p)", "v", tpe, 0));
}

std::optional<std::string> backend::CSource::mkZeroInit(const Type::Any &tpe) const {
  if (dialect == Dialect::MSL1_0) return "{}"s;
  const std::function<bool(const Type::Any &, size_t)> reachesScalar = [&](const Type::Any &t, const size_t depth) -> bool {
    if (depth > 32) return false;
    if (const auto a = t.template get<Type::Arr>()) return a->length > 0 && reachesScalar(a->comp, depth + 1);
    if (const auto s = t.template get<Type::Struct>()) {
      const auto members = structDefsByName.find(normalise(s->name));
      if (members == structDefsByName.end()) return false;
      const auto first =
          members->second ^ find([&](const auto &member) {
            const auto &memberTpe = member.second;
            return !memberTpe.template is<Type::FnRef>() && !(memberTpe.template get<Type::Struct>() ^ exists([&](const auto &nested) {
                                                                return zeroSizeStructNames ^ contains(normalise(nested.name));
                                                              }));
          });
      return first ^ exists([&](const auto &member) { return reachesScalar(member.second, depth + 1); });
    }
    return !t.template is<Type::FnRef>() && !t.template is<Type::Unit0>() && !t.template is<Type::Nothing>();
  };
  return reachesScalar(tpe, 0) ? std::optional{"{0}"s} : std::nullopt;
}

std::string backend::CSource::mkStmt(const Stmt::Any &stmt) {
  // member-wise reads repeat side effects, so a struct only decomposes for a plain lvalue source
  const auto memberwise = [&](const Type::Any &tpe, const bool lvalueSource) {
    return tpe.is<Type::Arr>() || (dialect == Dialect::OpenCL1_1 && tpe.is<Type::Struct>() && lvalueSource);
  };
  return stmt.match_total( //
      [&](const Stmt::Var &x) {
        if (x.name.tpe.is<Type::FnRef>()) return ""s;
        if (x.name.tpe.is<Type::Unit0>()) return x.expr ? fmt::format("{};", mkExpr(*x.expr)) : ""s;
        if (isLocalArr(x.name.tpe)) {
          if (!x.expr || isPoisonInit(*x.expr)) return ""s;
          if (memberwise(x.name.tpe, x.expr->template is<Expr::Alias>()))
            return mkValueCopy(localName(x.name.symbol), mkExpr(*x.expr), x.name.tpe, 0);
          throw BackendException("workgroup array initializer is not representable");
        }
        if (x.expr && isPoisonInit(*x.expr) && (x.name.tpe.is<Type::Struct>() || x.name.tpe.is<Type::Arr>())) {
          return fmt::format("{};", mkDecl(x.name.tpe, localName(x.name.symbol)));
        }
        if (x.expr && memberwise(x.name.tpe, x.expr->template is<Expr::Alias>()))
          return fmt::format("{}; {}", mkDecl(x.name.tpe, localName(x.name.symbol)),
                             mkValueCopy(localName(x.name.symbol), mkExpr(*x.expr), x.name.tpe, 0));
        if (!x.expr && x.name.tpe.is<Type::Struct>())
          if (const auto init = mkZeroInit(x.name.tpe)) return fmt::format("{} = {};", mkDecl(x.name.tpe, localName(x.name.symbol)), *init);
        return fmt::format("{}{};", mkDecl(x.name.tpe, localName(x.name.symbol)), x.expr ? " = " + mkExpr(*x.expr) : "");
      },
      [&](const Stmt::Mut &x) {
        if (x.name.tpe.template is<Type::FnRef>()) return ""s;
        if (isPoisonInit(x.expr) && (x.name.tpe.template is<Type::Struct>() || x.name.tpe.template is<Type::Arr>())) return ""s;
        if (x.name.tpe.template is<Type::Unit0>()) return fmt::format("{};", mkExpr(x.expr));
        if (memberwise(x.name.tpe, x.expr.template is<Expr::Alias>())) return mkValueCopy(mkTerm(x.name), mkExpr(x.expr), x.name.tpe, 0);
        return fmt::format("{} = {};", mkTerm(x.name), mkExpr(x.expr));
      },
      [&](const Stmt::Update &x) {
        if (memberwise(x.value.tpe(), true)) // a Term source is always a plain lvalue
          return mkValueCopy(fmt::format("{}[{}]", mkTerm(x.lhs), mkTerm(x.idx)), mkTerm(x.value), x.value.tpe(), 0);
        return fmt::format("{}[{}] = {};", mkTerm(x.lhs), mkTerm(x.idx), mkTerm(x.value));
      },
      [&](const Stmt::While &x) {
        const auto body = x.body ^ mk_string("\n", [&](const auto &s) { return mkStmt(s); });
        return fmt::format("while({}) {{\n{}\n}}", mkTerm(x.cond), body ^ indent(2));
      },
      [&](const Stmt::ForRange &x) {
        const auto body = x.body ^ mk_string("\n", [&](const auto &s) { return mkStmt(s); });
        const auto induction = localName(x.induction.symbol);
        return fmt::format("for({} {} = {}; {} < {}; {} += {}) {{\n{}\n}}",     //
                           mkTpe(x.induction.tpe), induction, mkTerm(x.lbIncl), //
                           induction, mkTerm(x.ubExcl), induction, mkTerm(x.step), body ^ indent(2));
      },
      [&](const Stmt::Break &) { return "break;"s; },   //
      [&](const Stmt::Cont &) { return "continue;"s; }, //
      [&](const Stmt::Cond &x) {
        auto trueBr = x.trueBr ^ mk_string("{\n", "\n", "\n}", [&](const auto &s) { return mkStmt(s) ^ indent(2); });
        if (x.falseBr.empty()) {
          return fmt::format("if ({}) {}", mkTerm(x.cond), trueBr);
        } else {
          auto falseBr = x.falseBr ^ mk_string("{\n", "\n", "\n}", [&](const auto &s) { return mkStmt(s) ^ indent(2); });
          // Metal can miscompile an empty taken arm guarding a loop latch through a mutable flag.
          if (dialect == Dialect::MSL1_0 && x.trueBr.empty()) return fmt::format("if (!({})) {}", mkTerm(x.cond), falseBr);
          return fmt::format("if ({}) {} else {}", mkTerm(x.cond), trueBr, falseBr);
        }
      },
      [&](const Stmt::Return &x) { return "return " + mkExpr(x.value) + ";"; }, //
      // Annotations carry no codegen meaning; unwrap and recurse.
      [&](const Stmt::Annotated &x) { return mkStmt(x.inner); },
      [&](const Stmt::Try &) -> std::string { throw std::logic_error("Stmt::Try should be erased"); },
      [&](const Stmt::Raise &) -> std::string { throw std::logic_error("Stmt::Raise should be erased"); },
      [&](const Stmt::Rethrow &) -> std::string { throw std::logic_error("Stmt::Rethrow should be erased"); });
}

std::string backend::CSource::mkFnProto(const Function &fnTree) {
  bindLocalNames(fnTree);

  const auto entry = fnTree.convention.is<CallConvention::OffloadEntry>();

  std::vector<std::string> argExprs =
      fnTree.decl.args   //
      | zip_with_index() //
      | map([&](const auto &arg, const auto &idx) {
          auto tpe = mkTpe(arg.named.tpe);
          auto name = localName(arg.named.symbol);
          std::string decl;
          switch (dialect) {
            case Dialect::OpenCL1_1: {
              decl = mkDecl(arg.named.tpe, name);
              break;
            }
            case Dialect::MSL1_0: {
              if (auto arr = arg.named.tpe.template get<Type::Ptr>()) {
                decl = arr->space.match_total([&](TypeSpace::Global) { return fmt::format("{} {} [[buffer({})]]", tpe, name, idx); }, //
                                              [&](TypeSpace::Constant) { return fmt::format("{} {} [[buffer({})]]", tpe, name, idx); },
                                              [&](TypeSpace::Local) { return fmt::format("{} {} [[threadgroup({})]]", tpe, name, idx); }, //
                                              [&](TypeSpace::Private) { return fmt::format("{} &{} [[buffer({})]]", tpe, name, idx); }    //
                );
              } else decl = fmt::format("device {} &{} [[buffer({})]]", tpe, name, idx);

              break;
            }
            default: break;
          }
          return decl;
        }) //
      | to_vector();

  if (dialect == Dialect::MSL1_0) {

    std::set<std::pair<std::string, std::string>> iargs; // ordered set for consistency
    // a SpecOp can nest in a loop/branch body, not just a top-level Var/Mut, so scan the whole function
    for (const auto &expr : fnTree.collect_all<Expr::Any>()) {
      auto spec = expr.template get<Expr::SpecOp>();
      if (!spec) continue;
      if (spec->op.is<Spec::GpuGlobalIdx>()) iargs.emplace("get_global_id", "thread_position_in_grid");
      if (spec->op.is<Spec::GpuGlobalSize>()) iargs.emplace("get_global_size", "threads_per_grid");
      if (spec->op.is<Spec::GpuGroupIdx>()) iargs.emplace("get_group_id", "threadgroup_position_in_grid");
      if (spec->op.is<Spec::GpuGroupSize>()) iargs.emplace("get_num_groups", "threadgroups_per_grid");
      if (spec->op.is<Spec::GpuLocalIdx>()) iargs.emplace("get_local_id", "thread_position_in_threadgroup");
      if (spec->op.is<Spec::GpuLocalSize>()) iargs.emplace("get_local_size", "threads_per_threadgroup");
    }
    argExprs ^= concat(iargs ^ map([](const auto &name, const auto &attr) { return fmt::format("uint3 __{}__ [[ {} ]]", name, attr); }));
  }

  std::string fnPrefix;
  switch (dialect) {
    case Dialect::C11: fnPrefix = ""; break;
    case Dialect::MSL1_0:
    case Dialect::OpenCL1_1:
      if (entry) {
        fnPrefix = "kernel ";
      }
      break;
    default: fnPrefix = "";
  }

  return fmt::format("{}{} {}({})",
                     fnPrefix,                    //
                     mkTpe(fnTree.decl.rtn),      //
                     normalise(fnTree.decl.name), //
                     argExprs ^ mk_string(", "));
}

std::string backend::CSource::mkFn(const Function &fnTree) {
  bindLocalNames(fnTree);
  const auto allVars = fnTree.body ^ flat_map([](const auto &s) { return s.template collect_all<Stmt::Var>(); });
  Set<std::string> seen;
  std::vector<Stmt::Var> localVars;
  localVars.reserve(allVars.size());
  for (const auto &v : allVars)
    if (isLocalArr(v.name.tpe) && seen.insert(v.name.symbol).second) localVars.emplace_back(v);
  struct Usage {
    uint64_t fixedBytes = 0;
    std::vector<std::string> fixedSizeExprs;
    const Named *dynamic = nullptr;
  };
  const auto usage = localVars ^ fold_left(Usage{0, {}, nullptr}, [&](Usage acc, const auto &v) {
                       const auto extent = arrayExtent(v.name.tpe);
                       if (!extent) throw BackendException("workgroup array extent overflow");
                       if (extent->count == 0) {
                         if (!acc.dynamic) acc.dynamic = &v.name;
                       } else if (const auto bytes = scalarBytes(extent->element)) {
                         if (extent->count > std::numeric_limits<uint64_t>::max() / *bytes)
                           throw BackendException("workgroup array extent overflow");
                         const auto total = extent->count * *bytes;
                         if (acc.fixedBytes > std::numeric_limits<uint64_t>::max() - total)
                           throw BackendException("workgroup array extent overflow");
                         acc.fixedBytes += total;
                       } else acc.fixedSizeExprs.push_back(fmt::format("({} * sizeof({}))", extent->count, mkTpe(extent->element)));
                       return acc;
                     });
  if (usage.fixedBytes > workgroupMemoryBytes || (usage.dynamic && usage.fixedBytes >= workgroupMemoryBytes))
    throw BackendException(fmt::format("workgroup storage exceeds configured capacity of {} bytes", workgroupMemoryBytes));

  const bool inPlace = usage.dynamic && scalarBytes(usage.dynamic->tpe.template get<Type::Arr>()->comp) == 1;
  const auto regionName = !usage.dynamic ? ""s : inPlace ? localName(usage.dynamic->symbol) : localName("#workgroup_region");
  const auto fixedExpr = usage.fixedSizeExprs ^ mk_string("", " + ", "", [](const auto &x) { return x; });
  const auto remaining = workgroupMemoryBytes - usage.fixedBytes;
  const auto available = fixedExpr.empty() ? std::to_string(remaining) : fmt::format("{} - ({})", remaining, fixedExpr);
  std::vector<std::string> regionConditions;
  if (!fixedExpr.empty()) regionConditions.push_back(fmt::format("({}) <= {}", fixedExpr, remaining));
  if (usage.dynamic) {
    const auto required = fnTree.template collect_all<Expr::Cast>() | collect([&](const auto &cast) -> std::optional<std::string> {
                            if (const auto ptr = cast.as.template get<Type::Ptr>(); ptr && ptr->space.template is<TypeSpace::Local>())
                              if (const auto structure = ptr->comp.template get<Type::Struct>()) return mkTpe(structure->widen());
                            return std::nullopt;
                          })
                          | to<Set>();
    regionConditions ^=
        concat(required ^ map([&](const auto &structure) { return fmt::format("sizeof({}) <= ({})", structure, available); }));
  }
  const auto condition = regionConditions ^ mk_string("", " && ", "", [](const auto &x) { return x; });
  const auto regionExtent = condition.empty() ? available : fmt::format("(({}) ? {} : -1)", condition, available);
  const auto regionDecl = [&](const Type::Any &element, const TypeSpace::Any &space, const std::string &name) {
    return fmt::format("__attribute__((aligned(16))) {};", mkArrayDecl(element, space, name, regionExtent));
  };

  std::vector<std::string> regionDecls;
  if (usage.dynamic && !inPlace) regionDecls.push_back(regionDecl(Type::IntS8(), TypeSpace::Local(), regionName));

  const auto localDecls = localVars ^ map([&](const auto &v) {
                            const auto a = v.name.tpe.template get<Type::Arr>();
                            if (!a || a->length != 0) return fmt::format("{};", mkDecl(v.name.tpe, localName(v.name.symbol)));
                            const auto name = localName(v.name.symbol);
                            if (name == regionName) return regionDecl(a->comp, a->space, name);
                            const auto ptr = Type::Ptr(a->comp, a->space).widen();
                            return fmt::format("{} = (({}) {});", mkDecl(ptr, name), mkTpe(ptr), regionName);
                          });
  if (!usage.fixedSizeExprs.empty() && !usage.dynamic)
    regionDecls.push_back(fmt::format("typedef char _polyregion_workgroup_capacity[({}) <= {} ? 1 : -1];", fixedExpr, remaining));
  const auto stmts = concat(concat(regionDecls, localDecls), fnTree.body ^ map([&](const auto &s) { return mkStmt(s); }));
  return fmt::format("{} {}", mkFnProto(fnTree), stmts ^ mk_string("{\n", "\n", "\n}", [&](const auto &s) { return s ^ indent(2); }));
}

CompileResult backend::CSource::compileProgram(const Program &program_, const compiletime::OptLevel &opt) {
  const auto tracePassStart = compiler::nowMono();
  auto program = CLAddressSpaceTracePass::execute(program_, dialect != Dialect::C11);
  CompileEvent cltpEvent(compiler::nowMs(), compiler::elapsedNs(tracePassStart), "polyast_cltp", repr(program), {});

  const auto start = compiler::nowMono();

  structDefsByName =
      program.defs | map([&](const auto &def) {
        return std::pair{normalise(def.name), def.members ^ map([&](const auto &m) { return std::pair{normalise(m.symbol), m.tpe}; })};
      }) //
      | to<Map>();
  unionDefNames = program.defs                                                //
                  | filter([](const auto &def) { return def.isUnion; })       //
                  | map([&](const auto &def) { return normalise(def.name); }) //
                  | to<Set>();
  Set<Sym> zeroSizeStructs = program.defs                                                  //
                             | filter([](const auto &def) { return def.members.empty(); }) //
                             | map([](const auto &def) { return def.name; })               //
                             | to<Set>();
  auto zeroSizeMember = [&](const Named &m) {
    return m.tpe.template get<Type::Struct>() ^ exists([&](const auto &s) { return zeroSizeStructs ^ contains(s.name); });
  };
  // metal rejects an all-zero-size body as a `[[buffer]]` pointee; OpenCL-C tolerates the empty member
  if (dialect == Dialect::MSL1_0)
    for (bool changed = true; changed;) {
      const auto seen = zeroSizeStructs.size();
      for (const auto &def : program.defs)
        if (def.members ^ forall(zeroSizeMember)) zeroSizeStructs.emplace(def.name);
      changed = zeroSizeStructs.size() != seen;
    }
  zeroSizeStructNames = zeroSizeStructs ^ map([&](const auto &name) { return normalise(name); });
  auto realStorageMember = [&](const Named &m) { return !zeroSizeMember(m); };
  auto renderStorageMember = [&](const Named &m) {
    // FnRef is an erased stateless callable, but its originating C++ object still occupies one byte when
    // captured as a data member.  Keeping that byte is ABI-significant: otherwise every later capture is read
    // at the preceding offset (for example `fn, init, step` becomes `init, step, ...`).  Calls and standalone
    // FnRef values remain erased; this is only their aggregate storage slot.
    const auto storageTpe = m.tpe.template is<Type::FnRef>() ? Type::IntU8().widen() : m.tpe;
    return fmt::format("  {};", mkDecl(storageTpe, normalise(m.symbol)));
  };

  // only by-value members create a definition-order dependency; pointer members resolve via the forward decl
  auto structsAndDeps = program.defs | map([&](const auto &def) {
                          const auto deps = def.members ^ collect([&](const auto &m) -> std::optional<Sym> {
                                              Type::Any base = m.tpe;
                                              while (auto a = base.template get<Type::Arr>())
                                                base = a->comp;
                                              return base.template get<Type::Struct>() ^ map([](const auto &s) { return s.name; });
                                            });
                          return std::pair{def, deps};
                        }) //
                        | to<Map>();

  const auto includes =
      dialect == Dialect::C11
          ? std::vector<std::string>{"#include <stdint.h>\n#include <stdbool.h>\n#include <math.h>\n#include <stdatomic.h>"}
          : std::vector<std::string>{};
  // forward-declare every struct so pointer members (including cyclic ones) resolve
  const auto typedefs =
      program.defs ^ map([&](const auto &def) {
        return fmt::format("typedef {} {} {};", def.isUnion ? "union" : "struct", normalise(def.name), normalise(def.name));
      });

  // emit struct bodies in by-value dependency order; a recursive cycle bails with a note
  std::vector<std::string> structBodies;
  Set<Sym> resolved;
  while (resolved.size() != program.defs.size()) {
    const auto noDeps = structsAndDeps                                  //
                        | filter([&](const auto &s, const auto &deps) { //
                            return !(resolved ^ contains(s.name)) && deps ^ forall([&](const auto &d) { return resolved ^ contains(d); });
                          })     //
                        | keys() //
                        | to_vector();
    if (noDeps.empty()) {
      structBodies ^= concat(std::vector<std::string>{"// Some structs cannot be resolved due to recursive by-value dependencies"});
      break;
    }
    structBodies ^= concat(noDeps ^ map([&](const auto &s) {
                             return fmt::format("{} {} {};\n", s.isUnion ? "union" : "struct", normalise(s.name),
                                                s.members | filter(realStorageMember) | mk_string("{\n", "\n", "\n}", renderStorageMember));
                           }));
    resolved ^= concat(noDeps ^ map([](const auto &s) { return s.name; }));
  }

  auto allFns = program.functions;
  if (program.entry) allFns ^= prepend(*program.entry);

  // hoist string literals to named program-scope constant arrays (collection order is deterministic); an inline
  // OpenCL literal has no addressable storage so reading it through a pointer yields garbage
  const char *constQual = dialect == Dialect::OpenCL1_1 ? "__constant " : dialect == Dialect::MSL1_0 ? "constant " : "static const ";
  stringConstNames.clear();
  const auto stringDecls =                                                                    //
      allFns                                                                                  //
      | flat_map([](const auto &fn) { return fn.template collect_all<Term::StringConst>(); }) //
      | map([](const auto &sc) { return sc.value; })                                          //
      | distinct()                                                                            //
      | map([&](const auto &value) {                                                          //
          const auto name = fmt::format("_polyregion_str_{}", stringConstNames.size());
          stringConstNames.emplace(value, name);
          // MSL is C++ so char and int8_t do not convert
          return fmt::format("{}{} {}[] = \"{}\";", constQual, mkTpe(Type::IntS8()), name, escapeCString(value));
        }) //
      | to_vector();

  const auto typeNames = program.defs ^ flat_map([&](const auto &def) {
                           return std::vector<std::string>{normalise(def.name)}
                                  ^ concat(def.members ^ map([&](const auto &member) { return normalise(member.symbol); }));
                         });
  fileScopeNames = typeNames                                                                       //
                   | concat(allFns ^ map([&](const auto &fn) { return normalise(fn.decl.name); })) //
                   | concat(stringConstNames ^ values())                                           //
                   | to<Set>();

  std::vector<std::string> volatileHelpers;
  if (dialect == Dialect::MSL1_0) {
    Set<std::string> emitted;
    const auto add = [&](const bool load, const Type::Any &tpe, const Term::Any &ptr) {
      if (!tpe.template is<Type::Struct>()) return;
      const auto space = mslPtrSpace(ptr), name = volatileHelperName(load, space, mkTpe(tpe));
      if (emitted.insert(name).second) volatileHelpers.push_back(mkVolatileHelper(load, tpe, space));
    };
    for (const auto &fn : allFns) {
      for (const auto &load : fn.template collect_all<Spec::GpuVolatileLoad>())
        add(true, load.rtn, load.ptr);
      for (const auto &store : fn.template collect_all<Spec::GpuVolatileStore>())
        add(false, store.value.tpe(), store.ptr);
    }
  }

  const auto atomicHelpers = dialect == Dialect::C11
                                 ? allFns                                                                                       //
                                       | flat_map([](const auto &fn) { return fn.template collect_all<Spec::GpuAtomicRMW>(); }) //
                                       | collect([&](const auto &atomic) -> std::optional<std::string> {
                                           const bool minimum = atomic.op.template is<AtomicOp::Min>();
                                           if (!minimum && !atomic.op.template is<AtomicOp::Max>()) return std::nullopt;
                                           const auto type = mkTpe(atomic.rtn), name = atomicMinMaxHelperName(minimum, type);
                                           return fmt::format("static {} {}(volatile _Atomic({}) *p, {} v) {{\n"
                                                              "  {} old = atomic_load_explicit(p, memory_order_relaxed);\n"
                                                              "  while (v {} old && !atomic_compare_exchange_weak_explicit(p, &old, v, "
                                                              "memory_order_relaxed, memory_order_relaxed)) {{}}\n"
                                                              "  return old;\n"
                                                              "}}",
                                                              type, name, type, type, type, minimum ? "<" : ">");
                                         })         //
                                       | distinct() //
                                       | to_vector()
                                 : std::vector<std::string>{};

  std::vector<std::string> popCountHelpers;
  if (dialect != Dialect::MSL1_0) {
    // C11 has no standard population count and OpenCL added its builtin in 1.2. Emit the same
    // width-specific SWAR fallback for both, then let newer OpenCL compilers select popcount.
    const auto popCounts = allFns ^ flat_map([](const auto &fn) { return fn.template collect_all<Intr::PopCount>(); });
    const auto wide = [](const Intr::PopCount &op) { return op.rtn.template is<Type::IntU64>() || op.rtn.template is<Type::IntS64>(); };
    const bool needs32 = popCounts ^ exists([&](const auto &op) { return !wide(op); });
    const bool needs64 = popCounts ^ exists(wide);
    const auto u32 = mkTpe(Type::IntU32()), u64 = mkTpe(Type::IntU64());
    if (needs32)
      popCountHelpers.emplace_back(fmt::format("static {} _polyregion_popcount_u32({} x) {{\n"
                                               "  x -= (x >> 1) & (({}) 0x55555555);\n"
                                               "  x = (x & (({}) 0x33333333)) + ((x >> 2) & (({}) 0x33333333));\n"
                                               "  x = (x + (x >> 4)) & (({}) 0x0f0f0f0f);\n"
                                               "  return (x * (({}) 0x01010101)) >> 24;\n"
                                               "}}",
                                               u32, u32, u32, u32, u32, u32, u32));
    if (needs64)
      popCountHelpers.emplace_back(fmt::format("static {} _polyregion_popcount_u64({} x) {{\n"
                                               "  x -= (x >> 1) & (({}) 0x5555555555555555);\n"
                                               "  x = (x & (({}) 0x3333333333333333)) + ((x >> 2) & (({}) 0x3333333333333333));\n"
                                               "  x = (x + (x >> 4)) & (({}) 0x0f0f0f0f0f0f0f0f);\n"
                                               "  return (x * (({}) 0x0101010101010101)) >> 56;\n"
                                               "}}",
                                               u64, u64, u64, u64, u64, u64, u64));
    if (needs32 || needs64) {
      std::string native, fallback;
      if (needs32) {
        native += "#define POLY_POPCOUNT32(x) popcount(x)\n";
        fallback += "#define POLY_POPCOUNT32(x) _polyregion_popcount_u32(x)\n";
      }
      if (needs64) {
        native += "#define POLY_POPCOUNT64(x) popcount(x)\n";
        fallback += "#define POLY_POPCOUNT64(x) _polyregion_popcount_u64(x)\n";
      }
      popCountHelpers.emplace_back(dialect == Dialect::OpenCL1_1 ? "#if defined(__OPENCL_C_VERSION__) && __OPENCL_C_VERSION__ >= 120\n"
                                                                       + native + "#else\n" + fallback + "#endif"
                                                                 : fallback);
    }
  }

  const auto protos = allFns ^ mk_string("\n", [&](const auto &fn) { return fmt::format("{};", mkFnProto(fn)); });
  auto code = includes                                                       //
              | concat(typedefs)                                             //
              | concat(structBodies)                                         //
              | concat(stringDecls)                                          //
              | concat(volatileHelpers)                                      //
              | concat(atomicHelpers)                                        //
              | concat(popCountHelpers)                                      //
              | append(protos)                                               //
              | append(std::string("\n"))                                    //
              | concat(allFns ^ map([&](const auto &f) { return mkFn(f); })) //
              | mk_string("\n");

  std::vector<std::string> features;
  if (usesTpe<Type::Float64>(allFns, program.defs)) features.emplace_back("fp64");

  // OpenCL half/double is behind cl_khr_fp16/cl_khr_fp64.
  if (dialect == Dialect::OpenCL1_1) {
    std::string pragmas;
    if (usesTpe<Type::Float64>(allFns, program.defs)) {
      pragmas += "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";
    }
    if (usesTpe<Type::Float16>(allFns, program.defs)) {
      pragmas += "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n";
      features.emplace_back("fp16");
    }
    // do NOT key "int64" off `long`: it maps to cl_khr_int64_base_atomics, which plain long arithmetic
    // does not need, so it would wrongly SKIP on Rusticl
    // POLY_NATIVE_TRIG routes the precise trig builtins to native_* on llvmpipe
    pragmas += "#ifdef POLY_NATIVE_TRIG\n"
               "#define POLY_SIN native_sin\n#define POLY_COS native_cos\n#define POLY_TAN native_tan\n"
               "#else\n"
               "#define POLY_SIN sin\n#define POLY_COS cos\n#define POLY_TAN tan\n"
               "#endif\n";
    code = pragmas + code;
  }

  std::string dialectName;
  switch (dialect) {
    case Dialect::C11: dialectName = "c11"; break;
    case Dialect::OpenCL1_1: dialectName = "opencl1_1"; break;
    case Dialect::MSL1_0: dialectName = "msl1"; break;
    default: dialectName = "unknown";
  }

  return {std::vector<int8_t>(code.begin(), code.end()),
          features,
          {cltpEvent, {compiler::nowMs(), compiler::elapsedNs(start), fmt::format("polyast_to_{}_c", dialectName), code, {}}},
          {},
          "",
          {}};
}
std::vector<StructLayout> backend::CSource::resolveLayouts(const std::vector<StructDef> &defs) { return std::vector<StructLayout>(); }
