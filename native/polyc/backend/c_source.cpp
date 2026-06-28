
#include "c_source.h"

#include <cmath>
#include <functional>
#include <set>

#include "aspartame/all.hpp"
#include "fmt/core.h"

#include "polyregion/conventions.h"

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
} // namespace
using namespace polyast;
using namespace std::string_literals;

static bool isLocalArr(const Type::Any &t) {
  return t.template get<Type::Arr>() ^ exists([](auto &a) { return a.space.template is<TypeSpace::Local>(); });
}

struct CLAddressSpaceTracePass {

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

  // Map a Term: only Term::Select needs scope rewiring; all other Term variants are pure atoms.
  static SpacedTerm mapTerm(const Term::Any &term, StackScope &scope) {
    if (auto sel = term.template get<Term::Select>()) {
      const auto rebound = scope.vars ^ get_or_default(sel->root.symbol, sel->root);
      // a final pointer field keeps its own space; a value subobject takes the root's space (local for a
      // declared local array, private otherwise)
      const auto rootSpace = rebound.tpe.template get<Type::Ptr>() ^ map([](auto &p) { return p.space; }) ^
                             get_or_else(isLocalArr(rebound.tpe) ? TypeSpace::Local().widen() : TypeSpace::Private().widen());
      const auto space = (sel->steps.empty() ? std::optional<Type::Ptr>{} : sel->tpe.template get<Type::Ptr>()) ^
                         map([](auto &p) { return p.space; }) ^ //
                         get_or_else(rootSpace);
      return SpacedTerm{Term::Select(rebound, sel->steps, sel->tpe), space};
    }
    // XXX null seeds its space from the pointee (refTpe=Global) so `T* p = nullptr` later infers Global not Private
    if (auto np = term.template get<Term::NullPtrConst>()) return SpacedTerm{term, np->space};
    return SpacedTerm{term};
  }

  static SpacedExpr mapExpr(const Expr::Any &expr, StackScope &scope) {
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
          auto as = x.as.template get<Type::Ptr>() ^ map([&](auto &p) { return Type::Ptr(p.comp, st.space).widen(); }) ^ get_or_else(x.as);
          return {Expr::Cast(st.actual, as), st.space};
        },
        [&](const Expr::Invoke &x) -> SpacedExpr { return {x.modify_all<Term::Any>(mapTerm0_)}; },
        [&](const Expr::Index &x) -> SpacedExpr {
          auto stLhs = mapTerm_(x.lhs);
          auto stIdx = mapTerm_(x.idx);
          return {Expr::Index(stLhs.actual, stIdx.actual, x.comp), stLhs.space};
        },
        [&](const Expr::RefTo &x) -> SpacedExpr {
          auto stLhs = mapTerm_(x.lhs);
          return {Expr::RefTo(stLhs.actual, x.idx ^ map(mapTerm0_), x.comp, stLhs.space, Region::Opaque()), stLhs.space};
        },
        [&](const Expr::Alloc &x) -> SpacedExpr { return {Expr::Alloc(x.comp, mapTerm0_(x.size), x.space, Region::Opaque())}; },
        [&](const Expr::ForeignCall &x) -> SpacedExpr { return {x.modify_all<Term::Any>(mapTerm0_)}; },
        [&](const Expr::OffsetOf &x) -> SpacedExpr { return {Expr::OffsetOf(x.structTpe, x.field)}; },
        [&](const Expr::SizeOf &x) -> SpacedExpr { return {Expr::SizeOf(x.forTpe)}; });
  }

  static Function mapFn(const Function &fn) {

    StackScope scope{.vars = fn.args                                                                  //
                             | flat_map([&](auto &arg) { return arg.template collect_all<Named>(); }) //
                             | filter([](auto &n) { return n.tpe.template is<Type::Ptr>(); })         //
                             | map([](auto &n) { return std::pair(n.symbol, n); })                    //
                             | to<Map>()};

    // a phi pointer var (declared with no initialiser, assigned in branches) keeps its declared space;
    // pre-scan its Mut assignments so the decl takes the assigned value's space, else OpenCL rejects the
    // `global* = private*` of e.g. std::min(&a, &b) over stack scalars in basic_string::max_size
    Map<std::string, TypeSpace::Any> phiSpace;
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
        StackScope scan{.vars = scope.vars};
        std::function<void(const std::vector<Stmt::Any> &)> walk = [&](const std::vector<Stmt::Any> &stmts) {
          for (auto &s : stmts) {
            if (auto var = s.template get<Stmt::Var>()) {
              auto name = var->name;
              if (auto expr = var->expr ^ map([&](auto &e) { return mapExpr(e, scan); })) {
                if (auto ptr = expr->actual.tpe().template get<Type::Ptr>())
                  name = Named(var->name.symbol, Type::Ptr(ptr->comp, expr->space));
              } else if (auto ptr = var->name.tpe.template get<Type::Ptr>()) {
                if (auto sp = phiSpace ^ get_maybe(var->name.symbol)) name = Named(var->name.symbol, Type::Ptr(ptr->comp, *sp));
              }
              scan.vars.insert_or_assign(name.symbol, name);
            } else if (auto mut = s.template get<Stmt::Mut>()) {
              if (mut->name.steps.empty() && mut->name.root.tpe.template is<Type::Ptr>()) {
                const auto sp = mapExpr(mut->expr, scan).space;
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
      }
    }

    auto body = fn.body ^ map([&](auto &s) {
                  return s
                      .template modify_all<Stmt::Var>([&](auto &var) { //
                        if (auto expr = var.expr ^ map([&](auto &e) { return mapExpr(e, scope); })) {
                          auto name = var.name;
                          if (auto ptr = expr->actual.tpe().template get<Type::Ptr>()) {
                            name = Named(var.name.symbol, Type::Ptr(ptr->comp, expr->space));
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
                      .template modify_all<Expr::Any>([&](auto &e) { return mapExpr(e, scope).actual; }); //
                });

    const auto tracedRtnTpes = body                                                                        //
                               | flat_map([&](auto &s) { return s.template collect_all<Stmt::Return>(); }) //
                               | map([&](auto &r) { return r.value.tpe(); })                               //
                               | distinct()                                                                //
                               | to_vector();                                                              //
    return Function(fn.name, fn.tpeVars, fn.receiver, fn.args, fn.moduleCaptures, fn.termCaptures, tracedRtnTpes[0], body, fn.visibility,
                    fn.fpMode, fn.isEntry, fn.affinity);
  }

  static Program execute(const Program &p) {
    // address-space inference must run on the ENTRY: OpenCL 1.2 rejects a global* initialised from a private*
    const auto remap = [](const Function &f) {
      const auto kernel = f.isEntry;
      auto remapSpace = [&](auto &s) {
        return s.match_total(
            [&](const TypeSpace::Global &) { return kernel ? TypeSpace::Global().widen() : TypeSpace::Private().widen(); }, //
            [&](const TypeSpace::Constant &) { return kernel ? TypeSpace::Global().widen() : TypeSpace::Private().widen(); },
            [&](const TypeSpace::Local &x) { return x.widen(); }, //
            [&](const TypeSpace::Private &x) { return x.widen(); });
      };
      return CLAddressSpaceTracePass::mapFn(
          f.withArgs(f.args ^ map([&](auto &arg) { return arg.template modify_all<TypeSpace::Any>(remapSpace); })));
    };
    auto entry = remap(p.entry);
    auto fns = p.functions ^ filter([](const Function &f) { return !f.isEntry; }) ^ map(remap);

    auto sigOf = [](const Expr::Invoke &inv) {
      return Signature(inv.name, /*tpeVars*/ {}, /*receiver*/ {}, inv.args ^ map([](auto &e) { return e.tpe(); }),
                       /*moduleCaptures*/ {}, /*termCaptures*/ {}, inv.rtn);
    };

    Map<Signature, std::shared_ptr<Function>> functionTable;
    fns | for_each([&](auto &f) {
      const Signature sig(f.name, /*tpeVars*/ {}, /*receiver*/ {}, f.args ^ map([](auto &e) { return e.named.tpe; }),
                          /*moduleCaptures*/ {}, /*termCaptures*/ {}, f.rtn);
      functionTable[sig] = std::make_shared<Function>(f);
    });

    while (true) {
      const auto specialised =
          functionTable                                                                                //
          | flat_map([&](auto, auto &f) { return f->template collect_all<Expr::Invoke>(); })           //
          | collect([&](auto &inv) -> std::optional<std::pair<Signature, std::shared_ptr<Function>>> { //
              if (const auto sig = sigOf(inv); !functionTable.contains(sig)) {
                if (auto spec =
                        functionTable ^ find([&](auto &lhs, auto) { return lhs.name == sig.name && lhs.args.size() == sig.args.size(); })) {
                  const auto fn = *spec->second;
                  const auto args = fn.args                                                                           //
                                    | zip(sig.args)                                                                   //
                                    | map([](auto &arg, auto &tpe) { return arg.withNamed(arg.named.withTpe(tpe)); }) //
                                    | to_vector();

                  return std::pair{sig, std::make_shared<Function>(CLAddressSpaceTracePass::mapFn(fn.withArgs(args)))};
                }
              }
              return {};
            }) //
          | to<Map>();
      if (specialised.empty()) break;
      functionTable.insert(specialised.begin(), specialised.end());
    }

    const auto spaceSpecialisedName = [](const Sym &name, const std::vector<TypeSpace::Any> &ts) -> Sym {
      auto suffix = ts ^ mk_string("", [&](auto &s) {
                      return s.match_total([&](TypeSpace::Global) { return "g"; },                                          //
                                           [&](TypeSpace::Constant) { return "g"; }, [&](TypeSpace::Local) { return "l"; }, //
                                           [&](TypeSpace::Private) { return "p"; });
                    });
      auto fqn = name.fqn;
      if (!fqn.empty()) fqn.back() = fqn.back() + "_" + suffix;
      else fqn.push_back("_" + suffix);
      return Sym(fqn);
    };

    auto spaces = [&](auto &a) { return a.template collect_all<TypeSpace::Any>(); };
    const auto renameInvokes = [&](const Function &f) {
      return f.template modify_all<Expr::Invoke>(
          [&](auto &inv) { return inv.withName(spaceSpecialisedName(inv.name, inv.args ^ flat_map(spaces))); });
    };
    const auto spaceSpecialisedFns =     //
        functionTable                    //
        | values()                       //
        | map([&](auto &f) -> Function { //
            return renameInvokes(*f).withName(f->isEntry ? f->name : spaceSpecialisedName(f->name, f->args ^ flat_map(spaces)));
          }) //
        | to_vector();

    return Program(renameInvokes(entry), spaceSpecialisedFns, p.defs, p.phase, p.metadata);
  }
};

static std::string normalise(const std::string &s) {
  // allowlist non-identifier chars to `_`: a stray `=` from `operator=` parses as an OpenCL assignment
  return (s ^ map([](const char c) {
            return ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_') ? c : '_';
          }))                                //
         ^ replace_all("global", "_global")  //
         ^ replace_all("local", "_local")    //
         ^ replace_all("kernel", "_kernel"); //
}

static std::string normalise(const Sym &s) { return normalise(repr(s)); }

Type::Any backend::CSource::resolveFieldType(const Type::Any &owner, const std::string &fieldName) const {
  if (auto s = owner.get<Type::Struct>()) {
    if (auto it = structDefsByName.find(normalise(s->name)); it != structDefsByName.end()) {
      const auto field = normalise(fieldName);
      if (auto m = it->second ^ find([&](auto &member) { return member.first == field; })) return m->second;
    }
    throw std::logic_error("field " + fieldName + " not found on struct " + repr(s->name));
  }
  throw std::logic_error("field " + fieldName + " selected on non-struct type " + repr(owner));
}

std::string backend::CSource::mkTpe(const Type::Any &tpe) {
  // metal requires an address space on every pointer, struct fields included
  auto mslPtrPrefix = [&](const TypeSpace::Any &space) {
    return space.match_total([&](TypeSpace::Global) { return "device"; },                                                    //
                             [&](TypeSpace::Constant) { return "device"; }, [&](TypeSpace::Local) { return "threadgroup"; }, //
                             [&](TypeSpace::Private) { return "thread"; }                                                    //
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
                             [&](const Type::Exec &x) -> std::string { throw std::logic_error("Type::Exec should be erased"); });
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
                               auto prefix = x.space.match_total([&](TypeSpace::Global) { return "global"; }, //
                                                                 [&](TypeSpace::Constant) { return "global"; },
                                                                 [&](TypeSpace::Local) { return "local"; },    //
                                                                 [&](TypeSpace::Private) { return "private"; } //
                               );
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
                             [&](const Type::Exec &x) -> std::string { throw std::logic_error("Type::Exec should be erased"); });
  }
}

// a C declarator places array extents AFTER the identifier (`T n[N][M]`), unlike mkTpe
std::string backend::CSource::mkDecl(const Type::Any &tpe, const std::string &name) {
  if (tpe.template is<Type::Arr>()) {
    std::string dims;
    Type::Any base = tpe;
    std::optional<TypeSpace::Any> space;
    while (auto a = base.template get<Type::Arr>()) {
      if (!space) space = a->space;
      dims += fmt::format("[{}]", a->length);
      base = a->comp;
    }
    // workgroup arrays need the `local`/`threadgroup` qualifier; private (the default) needs none
    std::string q;
    if (space && space->template is<TypeSpace::Local>()) q = dialect == Dialect::MSL1_0 ? "threadgroup " : "local ";
    return fmt::format("{}{} {}{}", q, mkTpe(base), name, dims);
  }
  if (auto p = tpe.template get<Type::Ptr>(); p && p->comp.template is<Type::Arr>()) {
    // pointer-to-array `T (*name)[d1][d2]` keeps all pointee extents so `&base[0][idx]` strides by sub-array
    std::string dims;
    Type::Any base = p->comp;
    while (auto a = base.template get<Type::Arr>()) {
      dims += fmt::format("[{}]", a->length);
      base = a->comp;
    }
    const auto q = p->space.match_total([&](TypeSpace::Global) { return dialect == Dialect::MSL1_0 ? "device "s : "global "s; },    //
                                        [&](TypeSpace::Constant) { return dialect == Dialect::MSL1_0 ? "device "s : "global "s; },  //
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

                          [](const Term::Unit0Const &) { return "/*void*/"s; },                                //
                          [](const Term::Bool1Const &x) { return x.value ? "true"s : "false"s; },              //
                          [&](const Term::NullPtrConst &x) { return fmt::format("0 /*{}*/", mkTpe(x.comp)); }, //
                          [&](const Term::Poison &x) {
                            // `0` not `NULL`: comgr doesn't predefine NULL for AMD kernel sources (non-ptr poison still casts)
                            if (x.tpe.is<Type::Ptr>()) return fmt::format("(0 /*{}*/)", repr(x.tpe));
                            return fmt::format("(({})0 /*poison {}*/)", mkTpe(x.tpe), repr(x.tpe));
                          }, //
                          [&](const Term::Select &x) {
                            std::string acc = normalise(x.root.symbol);
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
                                  [&](const PathStep::Index &) { throw std::logic_error("PathStep::Index not supported in c_source"); },
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
          std::string cl, msl;
        };
        const auto gpuIntr = [&](const DialectAccessor &accessor) -> std::string {
          switch (dialect) {
            case Dialect::C11: throw std::logic_error(to_string(x) + " not supported for C11");
            case Dialect::MSL1_0: return accessor.msl;
            case Dialect::OpenCL1_1: return accessor.cl;
          }
        };
        const auto gpuDimIntr = [&](const DialectAccessor &accessor, const Term::Any &index) -> std::string {
          switch (dialect) {
            case Dialect::C11: throw std::logic_error(to_string(x) + " not supported for C11");
            case Dialect::MSL1_0: return fmt::format("{}[{}]", accessor.msl, mkTerm(index));
            case Dialect::OpenCL1_1: return fmt::format("{}({})", accessor.cl, mkTerm(index));
          }
        };
        return x.op.match_total(
            [&](const Spec::Assert &v) { return "abort()"s; }, //
            [&](const Spec::GpuBarrierGlobal &v) {
              return gpuIntr({.cl = "barrier(CLK_GLOBAL_MEM_FENCE)", //
                              .msl = "threadgroup_barrier(metal::mem_flags::mem_none)"});
            },
            [&](const Spec::GpuBarrierLocal &v) {
              return gpuIntr({.cl = "barrier(CLK_LOCAL_MEM_FENCE)", //
                              .msl = "threadgroup_barrier(metal::mem_flags::mem_none)"});
            },
            [&](const Spec::GpuBarrierAll &v) {
              return gpuIntr({.cl = "barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE)", //
                              .msl = "threadgroup_barrier(metal::mem_flags::mem_none)"});
            },
            [&](const Spec::GpuFenceGlobal &v) {
              return gpuIntr({.cl = "mem_fence(CLK_GLOBAL_MEM_FENCE)", //
                              .msl = "threadgroup_barrier(metal::mem_flags::mem_device)"});
            },
            [&](const Spec::GpuFenceLocal &v) {
              return gpuIntr({.cl = "mem_fence(CLK_LOCAL_MEM_FENCE)", //
                              .msl = "threadgroup_barrier(metal::mem_flags::mem_threadgroup)"});
            },
            [&](const Spec::GpuFenceAll &v) {
              return gpuIntr({.cl = "mem_fence(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE)", //
                              .msl = "threadgroup_barrier(metal::mem_flags::mem_device)"});
            },
            [&](const Spec::GpuGlobalIdx &v) { return gpuDimIntr({.cl = "get_global_id", .msl = "__get_global_id__"}, v.dim); },      //
            [&](const Spec::GpuGlobalSize &v) { return gpuDimIntr({.cl = "get_global_size", .msl = "__get_global_size__"}, v.dim); }, //
            [&](const Spec::GpuGroupIdx &v) { return gpuDimIntr({.cl = "get_group_id", .msl = "__get_group_id__"}, v.dim); },         //
            [&](const Spec::GpuGroupSize &v) { return gpuDimIntr({.cl = "get_num_groups", .msl = "__get_num_groups__"}, v.dim); },    //
            [&](const Spec::GpuLocalIdx &v) { return gpuDimIntr({.cl = "get_local_id", .msl = "__get_local_id__"}, v.dim); },         //
            [&](const Spec::GpuLocalSize &v) { return gpuDimIntr({.cl = "get_local_size", .msl = "__get_local_size__"}, v.dim); }     //
        );
      },
      [&](const Expr::IntrOp &x) {
        return x.op.match_total([&](const Intr::Pos &v) { return fmt::format("(+{})", mkTerm(v.x)); },
                                [&](const Intr::Neg &v) { return fmt::format("(-{})", mkTerm(v.x)); },
                                [&](const Intr::BNot &v) { return fmt::format("(~{})", mkTerm(v.x)); },
                                [&](const Intr::LogicNot &v) { return fmt::format("(!{})", mkTerm(v.x)); },
                                [&](const Intr::Add &v) { return fmt::format("({} + {})", mkTerm(v.x), mkTerm(v.y)); },
                                [&](const Intr::Sub &v) { return fmt::format("({} - {})", mkTerm(v.x), mkTerm(v.y)); },
                                [&](const Intr::Mul &v) { return fmt::format("({} * {})", mkTerm(v.x), mkTerm(v.y)); },
                                [&](const Intr::Div &v) { return fmt::format("({} / {})", mkTerm(v.x), mkTerm(v.y)); },
                                [&](const Intr::Rem &v) { return fmt::format("({} % {})", mkTerm(v.x), mkTerm(v.y)); },
                                [&](const Intr::Min &v) { return fmt::format("min({}, {})", mkTerm(v.x), mkTerm(v.y)); },
                                [&](const Intr::Max &v) { return fmt::format("max({}, {})", mkTerm(v.x), mkTerm(v.y)); },
                                [&](const Intr::BAnd &v) { return fmt::format("({} & {})", mkTerm(v.x), mkTerm(v.y)); },
                                [&](const Intr::BOr &v) { return fmt::format("({} | {})", mkTerm(v.x), mkTerm(v.y)); },
                                [&](const Intr::BXor &v) { return fmt::format("({} ^ {})", mkTerm(v.x), mkTerm(v.y)); },
                                [&](const Intr::BSL &v) { return fmt::format("({} << {})", mkTerm(v.x), mkTerm(v.y)); },
                                [&](const Intr::BSR &v) { return fmt::format("({} >> {})", mkTerm(v.x), mkTerm(v.y)); },
                                [&](const Intr::BZSR &v) { return fmt::format("({} >> {})", mkTerm(v.x), mkTerm(v.y)); },
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
        return fmt::format("{}({})", normalise(x.name), x.args ^ mk_string(", ", [&](auto &arg) { return mkTerm(arg); }));
      }, //
      [&](const Expr::Index &x) { return fmt::format("{}[{}]", mkTerm(x.lhs), mkTerm(x.idx)); },
      [&](const Expr::RefTo &x) {
        // pointer-to-array: `&base[0][idx]` matches the mkDecl declarator; `&base[idx]` would stride by sub-array
        if (x.idx)
          if (auto pt = x.lhs.tpe().template get<Type::Ptr>(); pt && pt->comp.template is<Type::Arr>())
            return fmt::format("&({})[0][{}]", mkTerm(x.lhs), mkTerm(*x.idx));
        std::string str = fmt::format("&({} /*{}*/)", mkTerm(x.lhs), mkTpe(x.comp));
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
          for (size_t i = 0; i + 1 < lhsSel->steps.size(); ++i)
            lhsSel->steps[i].match_total(
                [&](const PathStep::Field &f) {
                  if (auto p = owner.template get<Type::Ptr>()) owner = p->comp;
                  owner = resolveFieldType(owner, f.name);
                },
                [&](const PathStep::Deref &) {
                  if (auto p = owner.template get<Type::Ptr>()) owner = p->comp;
                },
                [&](const PathStep::Index &) {},
                [&](const PathStep::IndexDyn &) {
                  if (auto p = owner.template get<Type::Ptr>()) owner = p->comp;
                  else if (auto a = owner.template get<Type::Arr>()) owner = a->comp;
                });
          if (auto p = owner.template get<Type::Ptr>()) owner = p->comp;
          const bool emptyBase = resolveFieldType(owner, lastField->name).template get<Type::Struct>() ^ exists([&](auto &s) {
                                   const auto it = structDefsByName.find(normalise(s.name));
                                   return it != structDefsByName.end() && it->second.empty();
                                 });
          if (emptyBase) {
            const auto full = mkTerm(x.lhs);
            const auto cut = full.rfind('.');
            str = fmt::format("&({})", cut == std::string::npos ? full : full.substr(0, cut));
          }
          str = fmt::format("(({}) {})", mkTpe(Type::Ptr(x.comp, x.space).widen()), str);
        }
        return str;
      },
      [&](const Expr::Alloc &x) { return fmt::format("{{/*{}*/}}", to_string(x)); },
      [&](const Expr::ForeignCall &x) {
        return fmt::format("{}({})", x.name, x.args ^ mk_string(", ", [&](auto &arg) { return mkTerm(arg); }));
      },
      [&](const Expr::OffsetOf &x) { return fmt::format("__builtin_offsetof({}, {})", mkTpe(x.structTpe), normalise(x.field)); },
      [&](const Expr::SizeOf &x) { return fmt::format("sizeof({})", mkTpe(x.forTpe)); });
}

// C/OpenCL forbid whole-array assignment; copy element-wise (a nested loop per array level)
static std::string mkArrayCopy(const std::string &lhs, const std::string &rhs, const Type::Any &tpe, int depth) {
  if (auto a = tpe.template get<Type::Arr>()) {
    const auto i = fmt::format("_ac{}", depth);
    return fmt::format("for (int {} = 0; {} < {}; {}++) {{ {} }}", i, i, a->length, i,
                       mkArrayCopy(fmt::format("{}[{}]", lhs, i), fmt::format("{}[{}]", rhs, i), a->comp, depth + 1));
  }
  return fmt::format("{} = {};", lhs, rhs);
}

std::string backend::CSource::mkStmt(const Stmt::Any &stmt) {
  return stmt.match_total( //
      [&](const Stmt::Var &x) {
        if (x.name.tpe.is<Type::Unit0>()) return x.expr ? fmt::format("{};", mkExpr(*x.expr)) : ""s;
        // hoisted to the kernel's outermost scope by mkFn; drop the nested decl
        if (!x.expr && isLocalArr(x.name.tpe)) return ""s;
        if (x.expr && x.name.tpe.is<Type::Arr>())
          return fmt::format("{}; {}", mkDecl(x.name.tpe, normalise(x.name.symbol)),
                             mkArrayCopy(normalise(x.name.symbol), mkExpr(*x.expr), x.name.tpe, 0));
        return fmt::format("{}{};", mkDecl(x.name.tpe, normalise(x.name.symbol)), x.expr ? " = " + mkExpr(*x.expr) : "");
      },
      [&](const Stmt::Mut &x) {
        if (x.name.tpe.template is<Type::Unit0>()) return fmt::format("{};", mkExpr(x.expr));
        if (x.name.tpe.template is<Type::Arr>()) return mkArrayCopy(mkTerm(x.name), mkExpr(x.expr), x.name.tpe, 0);
        return fmt::format("{} = {};", mkTerm(x.name), mkExpr(x.expr));
      },
      [&](const Stmt::Update &x) {
        if (x.value.tpe().template is<Type::Arr>())
          return mkArrayCopy(fmt::format("{}[{}]", mkTerm(x.lhs), mkTerm(x.idx)), mkTerm(x.value), x.value.tpe(), 0);
        return fmt::format("{}[{}] = {};", mkTerm(x.lhs), mkTerm(x.idx), mkTerm(x.value));
      },
      [&](const Stmt::While &x) {
        const auto body = x.body | mk_string("\n", [&](auto &s) { return mkStmt(s); });
        return fmt::format("while({}) {{\n{}\n}}", mkTerm(x.cond), body ^ indent(2));
      },
      [&](const Stmt::ForRange &x) {
        const auto body = x.body | mk_string("\n", [&](auto &s) { return mkStmt(s); });
        const auto induction = normalise(x.induction.symbol);
        return fmt::format("for({} {} = {}; {} < {}; {} += {}) {{\n{}\n}}",     //
                           mkTpe(x.induction.tpe), induction, mkTerm(x.lbIncl), //
                           induction, mkTerm(x.ubExcl), induction, mkTerm(x.step), body ^ indent(2));
      },
      [&](const Stmt::Break &) { return "break;"s; },   //
      [&](const Stmt::Cont &) { return "continue;"s; }, //
      [&](const Stmt::Cond &x) {
        auto trueBr = x.trueBr ^ mk_string("{\n", "\n", "\n}", [&](auto &s) { return mkStmt(s) ^ indent(2); });
        if (x.falseBr.empty()) {
          return fmt::format("if ({}) {}", mkTerm(x.cond), trueBr);
        } else {
          auto falseBr = x.falseBr ^ mk_string("{\n", "\n", "\n}", [&](auto &s) { return mkStmt(s) ^ indent(2); });
          return fmt::format("if ({}) {} else {}", mkTerm(x.cond), trueBr, falseBr);
        }
      },
      [&](const Stmt::Return &x) { return "return " + mkExpr(x.value) + ";"; }, //
      // Annotations carry no codegen meaning; unwrap and recurse.
      [&](const Stmt::Annotated &x) { return mkStmt(x.inner); });
}

std::string backend::CSource::mkFnProto(const Function &fnTree) {

  const auto entry = fnTree.isEntry;

  std::vector<std::string> argExprs =
      fnTree.args | zip_with_index() | map([&](auto &arg, auto idx) {
        auto tpe = mkTpe(arg.named.tpe);
        auto name = normalise(arg.named.symbol);
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
      }) |
      to_vector();

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
    for (auto &[name, attr] : iargs)
      argExprs.emplace_back(fmt::format("uint3 __{}__ [[ {} ]]", name, attr));
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
                     fnPrefix,               //
                     mkTpe(fnTree.rtn),      //
                     normalise(fnTree.name), //
                     argExprs | mk_string(", "));
}

std::string backend::CSource::mkFn(const Function &fnTree) {
  // OpenCL local-AS declarations must sit in the kernel's outermost scope
  const auto allVars = fnTree.body | flat_map([](auto &s) { return s.template collect_all<Stmt::Var>(); }) | to_vector();
  const auto localDecls = allVars ^ collect([&](auto &v) -> std::optional<std::string> {
                            if (!v.expr && isLocalArr(v.name.tpe)) return fmt::format("{};", mkDecl(v.name.tpe, normalise(v.name.symbol)));
                            return std::nullopt;
                          });
  const auto stmts = concat(localDecls, fnTree.body ^ map([&](auto &s) { return mkStmt(s); }));
  return fmt::format("{} {}", mkFnProto(fnTree), stmts ^ mk_string("{\n", "\n", "\n}", [&](auto &s) { return s ^ indent(2); }));
}

CompileResult backend::CSource::compileProgram(const Program &program_, const compiletime::OptLevel &opt) {
  const auto tracePassStart = compiler::nowMono();
  auto program = CLAddressSpaceTracePass::execute(program_);
  CompileEvent cltpEvent(compiler::nowMs(), compiler::elapsedNs(tracePassStart), "polyast_cltp", repr(program), {});

  const auto start = compiler::nowMono();

  structDefsByName =
      program.defs ^ map([&](auto &def) {
        return std::pair{normalise(def.name), def.members ^ map([&](auto &m) { return std::pair{normalise(m.symbol), m.tpe}; })};
      }) ^
      to<Map>();
  const Set<Sym> zeroSizeStructs =
      program.defs ^ filter([](auto &def) { return def.members.empty(); }) ^ map([](auto &def) { return def.name; }) ^ to<Set>();
  auto realStorageMember = [&](const Named &m) {
    return !(m.tpe.template get<Type::Struct>() ^ exists([&](auto &s) { return zeroSizeStructs.contains(s.name); }));
  };

  // only by-value members create a definition-order dependency; pointer members resolve via the forward decl
  auto structsAndDeps = program.defs ^ map([&](auto &def) {
                          const auto deps = def.members ^ collect([&](auto &m) -> std::optional<Sym> {
                                              Type::Any base = m.tpe;
                                              while (auto a = base.template get<Type::Arr>())
                                                base = a->comp;
                                              return base.template get<Type::Struct>() ^ map([](auto &s) { return s.name; });
                                            });
                          return std::pair{def, deps};
                        }) ^
                        to<Map>();

  const auto includes = dialect == Dialect::C11 ? std::vector<std::string>{"#include <stdint.h>\n#include <stdbool.h>\n#include <math.h>"}
                                                : std::vector<std::string>{};
  // forward-declare every struct so pointer members (including cyclic ones) resolve
  const auto typedefs =
      program.defs ^ map([&](auto &def) {
        return fmt::format("typedef {} {} {};", def.isUnion ? "union" : "struct", normalise(def.name), normalise(def.name));
      });

  // emit struct bodies in by-value dependency order; a recursive cycle bails with a note
  std::vector<std::string> structBodies;
  Set<Sym> resolved;
  while (resolved.size() != program.defs.size()) {
    const auto noDeps = structsAndDeps                      //
                        | filter([&](auto &s, auto &deps) { //
                            return !resolved.contains(s.name) && deps | forall([&](auto &d) { return resolved.contains(d); });
                          }) //
                        | keys() | to_vector();
    if (noDeps.empty()) {
      structBodies ^= concat(std::vector<std::string>{"// Some structs cannot be resolved due to recursive by-value dependencies"});
      break;
    }
    structBodies ^= concat(noDeps | map([&](auto &s) {
                             return fmt::format("{} {} {};\n", s.isUnion ? "union" : "struct", normalise(s.name),
                                                s.members | filter(realStorageMember) | mk_string("{\n", "\n", "\n}", [&](auto &m) {
                                                  return fmt::format("  {};", mkDecl(m.tpe, normalise(m.symbol)));
                                                }));
                           }) |
                           to_vector());
    resolved ^= concat(noDeps | map([](auto &s) { return s.name; }) | to_vector());
  }

  const auto allFns = std::vector<Function>{program.entry} ^ concat(program.functions);
  const auto protos = allFns | mk_string("\n", [&](auto &fn) { return fmt::format("{};", mkFnProto(fn)); });
  auto code = includes | concat(typedefs) | concat(structBodies) | append(protos) | append(std::string("\n")) |
              concat(allFns ^ map([&](auto &f) { return mkFn(f); })) | mk_string("\n");

  const auto usesTpe = [&](auto pred) {
    return (allFns ^ exists([&](const Function &f) { return f.collect_all<Type::Any>() ^ exists(pred); })) ||
           (program.defs ^ exists([&](const StructDef &d) {
              return d.members ^ exists([&](const Named &m) { return m.tpe.collect_all<Type::Any>() ^ exists(pred); });
            }));
  };

  std::vector<std::string> features;
  if (usesTpe([](const Type::Any &t) { return t.is<Type::Float64>(); })) features.emplace_back("fp64");

  // OpenCL half/double is behind cl_khr_fp16/cl_khr_fp64.
  if (dialect == Dialect::OpenCL1_1) {
    std::string pragmas;
    if (usesTpe([](const Type::Any &t) { return t.is<Type::Float64>(); })) {
      pragmas += "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";
    }
    if (usesTpe([](const Type::Any &t) { return t.is<Type::Float16>(); })) {
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
          ""};
}
std::vector<StructLayout> backend::CSource::resolveLayouts(const std::vector<StructDef> &defs) { return std::vector<StructLayout>(); }
