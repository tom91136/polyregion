
#include "c_source.h"

#include <cctype>
#include <cmath>
#include <cstdlib>
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

static std::string volatileHelperName(const bool load, const std::string &space, const std::string &element) {
  return fmt::format("_pr_v{}_{}_{}", load ? "ld" : "st", space, element);
}

static std::string atomicMinMaxHelperName(const bool minimum, const std::string &element) {
  return fmt::format("_pr_atomic_{}_{}", minimum ? "min" : "max", element);
}

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

static std::string safeLocalIdentifier(const std::string &s) {
  static const Set<std::string> reserved = {"global", "local", "kernel", "constant", "private", "device", "threadgroup", "thread"};
  return reserved ^ contains(s) ? "_" + s : s;
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
      const auto base = safeLocalIdentifier(name);
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
    if (auto it = structDefsByName.find(fqcn(s->name)); it != structDefsByName.end()) {
      if (auto m = it->second ^ find([&](const auto &name, const auto &) { return name == fieldName; })) return m->second;
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

                             [&](const Type::Struct &x) { return fqcn(x.name); }, //
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

                             [&](const Type::Struct &x) { return fqcn(x.name); }, //
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

std::optional<std::string> backend::CSource::mkArrayAliasDecl(const Type::Any &tpe, const Term::Any &source, const std::string &name) {
  // An immutable array alias only needs the source address.  Materialising every element of a
  // large field array in a private temporary is both unnecessary and, on PoCL, enough to exhaust
  // the worker stack.  Keep mutable aliases as value copies below because rebinding their storage
  // would change the language-level copy semantics.
  if (dialect != Dialect::OpenCL1_1) return std::nullopt;
  const auto array = tpe.template get<Type::Arr>();
  const auto select = source.template get<Term::Select>();
  // Keep ordinary local-array initialisation by value.  The optimisation is for a field reached
  // through a pointer (the derived-type capture shape emitted by polyfc), where the selected
  // storage is the intended immutable view and is already addressable without a copy.
  if (!array || !select || select->steps.empty() || !select->root.tpe.template is<Type::Ptr>()) return std::nullopt;

  Type::Any current = select->root.tpe;
  TypeSpace::Any space =
      current.template get<Type::Ptr>() ^ map([](const auto &p) { return p.space; }) ^ get_or_else(TypeSpace::Private().widen());
  for (const auto &step : select->steps)
    step.match_total(
        [&](const PathStep::Field &field) {
          if (const auto pointer = current.template get<Type::Ptr>()) space = pointer->space, current = pointer->comp;
          current = resolveFieldType(current, field.name);
        },
        [&](const PathStep::Deref &) {
          if (const auto pointer = current.template get<Type::Ptr>()) space = pointer->space, current = pointer->comp;
        },
        [&](const PathStep::Index &) {
          if (const auto pointer = current.template get<Type::Ptr>()) space = pointer->space, current = pointer->comp;
          else if (const auto nested = current.template get<Type::Arr>()) current = nested->comp;
        },
        [&](const PathStep::IndexDyn &) {
          if (const auto pointer = current.template get<Type::Ptr>()) space = pointer->space, current = pointer->comp;
          else if (const auto nested = current.template get<Type::Arr>()) current = nested->comp;
        });
  return fmt::format("{} = {}", mkDecl(Type::Ptr(array->comp, space).widen(), name), mkTerm(*select));
}

std::string backend::CSource::reinterpretScalar(const std::string &value, const Type::Any &from, const Type::Any &to) {
  switch (dialect) {
    case Dialect::C11: return fmt::format("((union {{ {} from; {} to; }}){{ .from = ({}) }}).to", mkTpe(from), mkTpe(to), value);
    case Dialect::OpenCL1_1: return fmt::format("as_{}({})", mkTpe(to), value);
    case Dialect::MSL1_0: return fmt::format("metal::as_type<{}>({})", mkTpe(to), value);
  }
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
                          },                                                                              //
                          [&](const Term::Defer &x) { return fmt::format("(({}){{0}})", mkTpe(x.tpe)); }, //
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
                                    acc += f.name;
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
      [&](const Expr::BitCast &x) { return reinterpretScalar(mkTerm(x.from), x.from.tpe(), x.as); },
      [&](const Expr::Invoke &x) {
        return fmt::format("{}({})", fqcn(calleeName(x)), x.args ^ mk_string(", ", [&](const auto &arg) { return mkTerm(arg); }));
      }, //
      [&](const Expr::Index &x) { return fmt::format("{}[{}]", mkTerm(x.lhs), mkTerm(x.idx)); },
      [&](const Expr::RefTo &x) {
        // pointer-to-array: `&base[0][idx]` matches the mkDecl declarator; `&base[idx]` would stride by sub-array
        if (x.idx)
          if (auto pt = x.lhs.tpe().template get<Type::Ptr>(); pt && pt->comp.template is<Type::Arr>())
            return fmt::format("&({})[0][{}]", mkTerm(x.lhs), mkTerm(*x.idx));
        auto str = fmt::format("&({} /*{}*/)", mkTerm(x.lhs), mkTpe(x.comp));
        // a value lhs would make `&value[idx]` illegal C, so drop the idx; a pointer/array lhs keeps it
        const bool valueLhs = !x.lhs.tpe().template is<Type::Ptr>() && !x.lhs.tpe().template is<Type::Arr>();
        if (x.idx && !valueLhs) str += fmt::format("[{}]", mkTerm(*x.idx));
        return str;
      },
      [&](const Expr::Alloc &x) { return fmt::format("{{/*{}*/}}", to_string(x)); },
      [&](const Expr::ForeignCall &x) {
        return fmt::format("{}({})", x.name, x.args ^ mk_string(", ", [&](const auto &arg) { return mkTerm(arg); }));
      },
      [&](const Expr::OffsetOf &x) { return fmt::format("__builtin_offsetof({}, {})", mkTpe(x.structTpe), x.field); },
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
    const auto name = fqcn(s->name);
    // a populated union stays whole; naming its members would not preserve the active one
    // a zero-size member has no storage in the emitted body, so it must not be named
    if (auto it = structDefsByName.find(name); it != structDefsByName.end() && (it->second.empty() || !(unionDefNames ^ contains(name))))
      return it->second | map([&](const auto &m) {
               return mkValueCopy(fmt::format("{}.{}", lhs, m.first), fmt::format("{}.{}", rhs, m.first), m.second, depth);
             })                                                       //
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
    const auto name = fqcn(structure->name);
    if (unionDefNames ^ contains(name))
      throw BackendException("volatile access to union " + repr(structure->name) + " is unsupported for MSL");
    const auto members = structDefsByName.find(name);
    if (members == structDefsByName.end()) throw BackendException("volatile access to undeclared struct " + repr(structure->name));
    return members->second | map([&](const auto &member) {
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
      const auto members = structDefsByName.find(fqcn(s->name));
      if (members == structDefsByName.end()) return false;
      const auto first = members->second ^ find([](const auto &) { return true; });
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
        if (x.expr)
          if (const auto alias = x.expr->template get<Expr::Alias>(); alias && !x.isMutable)
            if (const auto decl = mkArrayAliasDecl(x.name.tpe, alias->ref, localName(x.name.symbol))) return *decl + ";";
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

  std::vector<std::string> argExprs;
  argExprs.reserve(fnTree.decl.args.size() * 2);
  for (size_t idx = 0; idx < fnTree.decl.args.size(); ++idx) {
    const auto &arg = fnTree.decl.args[idx];
    const auto tpe = mkTpe(arg.named.tpe);
    const auto name = localName(arg.named.symbol);
    switch (dialect) {
      case Dialect::OpenCL1_1: {
        // clSetKernelArg binds an owning cl_mem, not an arbitrary interior pointer.  Source kernels receive
        // each Global/Constant pointer as owner+byte-offset and reconstruct the logical typed pointer in mkFn.
        const auto ptr = arg.named.tpe.template get<Type::Ptr>();
        const bool offsetAbi =
            entry && ptr && (ptr->space.template is<TypeSpace::Global>() || ptr->space.template is<TypeSpace::Constant>());
        if (offsetAbi) {
          argExprs.push_back(mkDecl(arg.named.tpe, fmt::format("_polyregion_arg_base_{}", idx)));
          argExprs.push_back(fmt::format("ulong _polyregion_arg_byte_offset_{}", idx));
        } else argExprs.push_back(mkDecl(arg.named.tpe, name));
        break;
      }
      case Dialect::MSL1_0: {
        if (auto arr = arg.named.tpe.template get<Type::Ptr>()) {
          argExprs.push_back(
              arr->space.match_total([&](TypeSpace::Global) { return fmt::format("{} {} [[buffer({})]]", tpe, name, idx); }, //
                                     [&](TypeSpace::Constant) { return fmt::format("{} {} [[buffer({})]]", tpe, name, idx); },
                                     [&](TypeSpace::Local) { return fmt::format("{} {} [[threadgroup({})]]", tpe, name, idx); }, //
                                     [&](TypeSpace::Private) { return fmt::format("{} &{} [[buffer({})]]", tpe, name, idx); }));
        } else argExprs.push_back(fmt::format("device {} &{} [[buffer({})]]", tpe, name, idx));
        break;
      }
      default: break;
    }
  }

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

  return fmt::format("{}{} {}({})", fnPrefix, mkTpe(fnTree.decl.rtn), fqcn(fnTree.decl.name), argExprs ^ mk_string(", "));
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

  std::vector<std::string> entryAbiDecls;
  if (dialect == Dialect::OpenCL1_1 && fnTree.convention.is<CallConvention::OffloadEntry>()) {
    for (size_t idx = 0; idx < fnTree.decl.args.size(); ++idx) {
      const auto &arg = fnTree.decl.args[idx];
      const auto ptr = arg.named.tpe.template get<Type::Ptr>();
      if (!ptr || (!ptr->space.template is<TypeSpace::Global>() && !ptr->space.template is<TypeSpace::Constant>())) continue;
      const auto bytePtr = ptr->space.template is<TypeSpace::Global>() ? "global uchar*" : "constant uchar*";
      const auto base = fmt::format("_polyregion_arg_base_{}", idx);
      const auto offset = fmt::format("_polyregion_arg_byte_offset_{}", idx);
      entryAbiDecls.push_back(fmt::format("{} = {} == POLYREGION_OPENCL_NULL_POINTER_OFFSET ? (({}) 0) : (({}) ((({}) {}) + {}));",
                                          mkDecl(arg.named.tpe, localName(arg.named.symbol)), offset, mkTpe(arg.named.tpe),
                                          mkTpe(arg.named.tpe), bytePtr, base, offset));
    }
  }

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
  const auto stmts =
      concat(concat(concat(entryAbiDecls, regionDecls), localDecls), fnTree.body ^ map([&](const auto &s) { return mkStmt(s); }));
  return fmt::format("{} {}", mkFnProto(fnTree), stmts ^ mk_string("{\n", "\n", "\n}", [&](const auto &s) { return s ^ indent(2); }));
}

CompileResult backend::CSource::compileProgram(const Program &program_, const compiletime::OptLevel &opt) {
  auto program = program_;

  const auto start = compiler::nowMono();

  structDefsByName = program.defs | map([&](const auto &def) {
                       return std::pair{fqcn(def.name), def.members ^ map([&](const auto &m) { return std::pair{m.symbol, m.tpe}; })};
                     }) //
                     | to<Map>();
  unionDefNames = program.defs                                           //
                  | filter([](const auto &def) { return def.isUnion; })  //
                  | map([&](const auto &def) { return fqcn(def.name); }) //
                  | to<Set>();
  auto renderStorageMember = [&](const Named &m) { return fmt::format("  {};", mkDecl(m.tpe, m.symbol)); };

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
  const auto typedefs = program.defs ^ map([&](const auto &def) {
                          return fmt::format("typedef {} {} {};", def.isUnion ? "union" : "struct", fqcn(def.name), fqcn(def.name));
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
                             return fmt::format("{} {} {};\n", s.isUnion ? "union" : "struct", fqcn(s.name),
                                                s.members | mk_string("{\n", "\n", "\n}", renderStorageMember));
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

  const auto typeNames =
      program.defs ^ flat_map([&](const auto &def) {
        return std::vector<std::string>{fqcn(def.name)} ^ concat(def.members ^ map([&](const auto &member) { return member.symbol; }));
      });
  fileScopeNames = typeNames                                                                  //
                   | concat(allFns ^ map([&](const auto &fn) { return fqcn(fn.decl.name); })) //
                   | concat(stringConstNames ^ values())                                      //
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
               "#endif\n"
               "#define POLYREGION_OPENCL_NULL_POINTER_OFFSET ((ulong)-1)\n";
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
          {{compiler::nowMs(), compiler::elapsedNs(start), fmt::format("polyast_to_{}_c", dialectName), code, {}}},
          {},
          "",
          {}};
}
std::vector<StructLayout> backend::CSource::resolveLayouts(const std::vector<StructDef> &defs) { return std::vector<StructLayout>(); }
