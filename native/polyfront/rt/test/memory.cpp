#include <catch2/catch_test_macros.hpp>

#include <fstream>
#include <numeric>

#include "aspartame/all.hpp"
#include "fmt/args.h"
#include "polyregion/show.hpp"
#include "polyrt/mem.hpp"

using namespace polyregion::runtime;

template <typename T> constexpr size_t indirections() {
  if constexpr (std::is_pointer_v<T>) return 1 + indirections<std::remove_pointer_t<T>>();
  else return 0;
}

template <typename T> constexpr size_t componentSize() {
  if constexpr (std::is_pointer_v<T>) return componentSize<std::remove_pointer_t<T>>();
  else return sizeof(T);
}

struct StructWithStorage {
  TypeLayout s;
  std::unique_ptr<AggregateMember[]> storage;

  const TypeLayout &operator*() const { return s; }
  TypeLayout &operator*() { return s; }
};

template <typename T> StructWithStorage liftToStruct(const char *name, std::initializer_list<AggregateMember> members) {
  auto storage = std::make_unique<AggregateMember[]>(members.size());
  std::copy(members.begin(), members.end(), storage.get());

  TypeLayout s{
      .name = name,
      .sizeInBytes = sizeof(T),
      .alignmentInBytes = alignof(T),
      .attrs = LayoutAttrs::None,
      .memberCount = members.size(),
      .members = storage.get(),
  };

  return StructWithStorage{s, std::move(storage)};
}

#define NamedType(type) TypeLayout::named<type>(#type)
#define StructMember_(type_, member_, typePtr_)                                                                                            \
  AggregateMember {                                                                                                                        \
    .name = #member_,                                               /**/                                                                   \
        .offsetInBytes = offsetof(type_, member_),                  /**/                                                                   \
        .sizeInBytes = sizeof(type_::member_),                      /**/                                                                   \
        .ptrIndirection = indirections<decltype(type_::member_)>(), /**/                                                                   \
        .componentSize = componentSize<decltype(type_::member_)>(), /**/                                                                   \
        .type = typePtr_,                                           /**/                                                                   \
        .resolvePtrSizeInBytes = nullptr                                                                                                   \
  }
#define Struct_(type_, ...) liftToStruct<type_>(#type_, {__VA_ARGS__})

const static TypeLayout floatType = NamedType(float);
const static TypeLayout int32Type = NamedType(int32_t);
const static TypeLayout int64Type = NamedType(int64_t);

struct Fixture {
  std::unordered_map<uintptr_t, size_t> localAllocations, remoteAllocations;

  using QueryPtr = std::function<polyregion::polyrt::PtrQuery(const void *)>;
  using AllocateRemote = std::function<uintptr_t(size_t)>;
  using ReadRemote = std::function<void(void *, uintptr_t, size_t, size_t)>;
  using WriteRemote = std::function<void(const void *, uintptr_t, size_t, size_t)>;
  using FreeRemote = std::function<void(uintptr_t)>;
  polyregion::polyrt::SynchronisedMemAllocation<QueryPtr, AllocateRemote, ReadRemote, WriteRemote, FreeRemote> allocation;

  Fixture()
      : allocation(
            [&](const void *ptr) -> polyregion::polyrt::PtrQuery {
              const uintptr_t localPtr = reinterpret_cast<uintptr_t>(ptr);
              if (const auto it = localAllocations.find(localPtr); it != localAllocations.end()) {
                return polyregion::polyrt::PtrQuery{it->second, 0};
              }
              for (auto [p, size] : localAllocations) {
                if (localPtr >= static_cast<uintptr_t>(p) && localPtr < static_cast<uintptr_t>(p) + size) {
                  return polyregion::polyrt::PtrQuery{size, localPtr - static_cast<uintptr_t>(p)};
                }
              }
              return polyregion::polyrt::PtrQuery{0, 0};
            }, //
            [&](size_t size) {
              auto p = std::malloc(size);
              fprintf(stderr, "                               Remote %p = malloc(%ld)\n", p, size);
              remoteAllocations.emplace(reinterpret_cast<uintptr_t>(p), size);
              return reinterpret_cast<uintptr_t>(p);
            }, //
            [](void *dst, uintptr_t src, size_t srcOffset, size_t size) {
              fprintf(stderr, "Local %p <|[%4ld]- Remote [%p + %4ld]\n", dst, size, reinterpret_cast<void *>(src), srcOffset);
              return std::memcpy(dst, reinterpret_cast<char *>(src) + srcOffset, size);
            }, //
            [](const void *src, uintptr_t dst, size_t dstOffset, size_t size) {
              fprintf(stderr, "Local %p -[%4ld]|> Remote [%p + %4ld]\n", src, size, reinterpret_cast<void *>(dst), dstOffset);
              return std::memcpy(reinterpret_cast<char *>(dst) + dstOffset, src, size);
            }, //
            [](uintptr_t remotePtr) {
              fprintf(stderr, "                               Remote free(%p)\n", reinterpret_cast<void *>(remotePtr));
              std::free(reinterpret_cast<void *>(remotePtr));
            },
            true) {}

  template <typename T> T *mallocLocal(size_t count = 1) {
    auto p = std::malloc(sizeof(T) * count);
    localAllocations.emplace(reinterpret_cast<uintptr_t>(p), sizeof(T) * count);
    return static_cast<T *>(p);
  };

  template <typename T> std::enable_if_t<!std::is_pointer_v<T>, T> localToRemote(const T &t, const TypeLayout &s) {
    return *reinterpret_cast<T *>(allocation.syncLocalToRemote(&t, s));
  }
  template <typename T> T *localToRemote(const T *t, const TypeLayout &s) {
    return reinterpret_cast<T *>(allocation.syncLocalToRemote(t, s));
  }

  template <typename T> std::optional<uintptr_t> localToRemote(const T *t) { return allocation.syncLocalToRemote(t); }

  template <typename T> std::optional<uintptr_t> remoteToLocal(T *t) { return allocation.syncRemoteToLocal(t); }

  ~Fixture() {
    for (auto &[k, _] : localAllocations)
      std::free(reinterpret_cast<void *>(k));
    for (auto &[k, _] : remoteAllocations)
      std::free(reinterpret_cast<void *>(k));
  }
};

TEST_CASE("ptr-indirect-2-star") {
  struct Foo {
    float **a;
  };
  auto fooMeta = Struct_(Foo, StructMember_(Foo, a, &floatType));
  Fixture fixture;
  size_t N = 10;
  size_t M = 5;
  auto as = fixture.mallocLocal<float *>(N);
  for (size_t i = 0; i < N; ++i) {
    as[i] = fixture.mallocLocal<float>(M);
    for (size_t j = 0; j < M; ++j) {
      as[i][j] = j;
    }
  }
  auto local = new (fixture.mallocLocal<Foo>()) Foo{as};
  Foo *remote = fixture.localToRemote(local, *fooMeta);
  CHECK(&remote != &local);
  CHECK(local->a != remote->a);
  for (size_t i = 0; i < N; ++i) {
    CHECK(local->a[i] != remote->a[i]);
    for (size_t j = 0; j < M; ++j) {
      CHECK(local->a[i][j] == remote->a[i][j]);
    }
  }

  remote->a[N - 1][M - 1] = 42;
  CHECK(local->a[N - 1][M - 1] == M - 1);
  CHECK(fixture.remoteToLocal(local->a[N - 1]) == std::optional{reinterpret_cast<uintptr_t>(remote->a[N - 1])});
  CHECK(local->a[N - 1][M - 1] == 42);

  local->a[N - 1][M - 1] = 43;

  CHECK(remote->a[N - 1][M - 1] == 42);
  CHECK(fixture.allocation.invalidateLocal(local->a[N - 1]) == std::optional{reinterpret_cast<uintptr_t>(remote->a[N - 1])});
  fixture.localToRemote(local->a[N - 1]);
  CHECK(remote->a[N - 1][M - 1] == 43);
}

TEST_CASE("ptr-indirect-nested-2-star") {
  struct check_array {
    int32_t **xs;
  };

  struct test_utils {
    int32_t *result;
    check_array f;
  };

  auto fooMeta = Struct_(check_array, StructMember_(check_array, xs, &int32Type));
  auto barMeta = Struct_(test_utils,                                    //
                         StructMember_(test_utils, result, &int32Type), //
                         StructMember_(test_utils, f, &*fooMeta), );

  Fixture fixture;
  size_t N = 10;
  const auto as = fixture.mallocLocal<int32_t>(N);
  std::iota(as, as + N, 0);

  const auto asRef = fixture.mallocLocal<int32_t *>();
  *asRef = as;

  const auto result = fixture.mallocLocal<int32_t>();
  *result = 42;

  auto local = new (fixture.mallocLocal<test_utils>()) test_utils{result, check_array{asRef}};

  test_utils *remote = fixture.localToRemote(local, *barMeta);

  (*barMeta).visualise(stderr);
  (*fooMeta).visualise(stderr);

  //
  // int p[1]={42};
  //
  // fprintf(stderr, "a = %p\n", p);
  // const auto ff = [=]()   {
  //   fprintf(stderr, "Lam: a = %p\n", p);
  // };
  // static_assert(sizeof(decltype(ff)) == sizeof(void*));
  //
  //
  // auto raw = reinterpret_cast<const char*>(&ff);
  // void* value=0;
  // std::memcpy(&value, raw, sizeof(value));
  // fprintf(stderr, "f = %p\n", value);
  //
  // fprintf(stderr, "f = %d !\n", *static_cast<int32_t*>(value));
}

TEST_CASE("ptr-indirect-3-star") {
  struct Bar {
    int32_t a;
  };
  struct Foo {
    Bar ***a;
  };
  auto barMeta = Struct_(Bar, StructMember_(Bar, a, &int32Type));
  auto fooMeta = Struct_(Foo, StructMember_(Foo, a, &*barMeta));
  Fixture fixture;
  auto bar = new (fixture.mallocLocal<Bar>()) Bar(42);
  auto barPtr = new (fixture.mallocLocal<Bar *>()) Bar *;
  auto barPtrPtr = new (fixture.mallocLocal<Bar **>()) Bar **;
  barPtr[0] = bar;
  barPtrPtr[0] = barPtr;
  Foo expected{barPtrPtr};
  Foo actual = fixture.localToRemote(expected, *fooMeta);
  CHECK(&actual != &expected);
  CHECK(expected.a != actual.a);
  CHECK(*expected.a != *actual.a);
  CHECK((*expected.a) != (*actual.a));
  CHECK((*(*expected.a)) != (*(*actual.a)));
  CHECK((*(*expected.a))->a == (*(*actual.a))->a);
}

TEST_CASE("simple") {
  struct Foo {
    int32_t a;
    int32_t b;
  };
  auto fooMeta = Struct_(Foo,                               //
                         StructMember_(Foo, a, &int32Type), //
                         StructMember_(Foo, b, &int32Type), //
  );
  Foo expected{42, 43};
  Fixture fixture;
  Foo actual = fixture.localToRemote(expected, *fooMeta);
  CHECK(&actual != &expected);
  CHECK(expected.a == actual.a);
  CHECK(expected.b == actual.b);
  CHECK(fixture.remoteAllocations.size() == 1);
}

TEST_CASE("simple-nested") {
  struct Foo {
    int32_t a;
    int32_t b;
  };

  struct Bar {
    Foo foo;
    int32_t c;
    int32_t d;
  };

  auto fooMeta = Struct_(Foo,                               //
                         StructMember_(Foo, a, &int32Type), //
                         StructMember_(Foo, b, &int32Type)  //
  );

  auto barMeta = Struct_(Bar,                                //
                         StructMember_(Bar, foo, &*fooMeta), //
                         StructMember_(Bar, c, &int32Type),  //
                         StructMember_(Bar, d, &int32Type)   //
  );

  Fixture fixture;
  auto expected = new (fixture.mallocLocal<Bar>()) Bar{{42, 43}, 44, 45};
  Bar *actual = fixture.localToRemote(expected, *barMeta);

  CHECK(actual != expected);
  CHECK(expected->foo.a == actual->foo.a);
  CHECK(expected->foo.b == actual->foo.b);
  CHECK(expected->c == actual->c);
  CHECK(expected->d == actual->d);
  CHECK(fixture.remoteAllocations.size() == 1);

  actual->foo.a = 0;
  actual->foo.b = 1;
  actual->c = 2;
  actual->d = 3;
  CHECK(fixture.remoteToLocal(expected) == std::optional{reinterpret_cast<uintptr_t>(actual)});
  CHECK(expected->foo.a == 0);
  CHECK(expected->foo.b == 1);
  CHECK(expected->c == 2);
  CHECK(expected->d == 3);
}

TEST_CASE("linkedlist") {
  struct Node {
    int32_t data;
    Node *next;
  };

  auto nodeMetaDeferred = Struct_(Node);
  auto nodeMeta = Struct_(Node,                                         //
                          StructMember_(Node, data, &int32Type),        //
                          StructMember_(Node, next, &*nodeMetaDeferred) //
  );
  (*nodeMetaDeferred).memberCount = (*nodeMeta).memberCount;
  (*nodeMetaDeferred).members = (*nodeMeta).members;

  Fixture fixture;

  auto node3 = new (fixture.mallocLocal<Node>()) Node{3, nullptr};
  auto node2 = new (fixture.mallocLocal<Node>()) Node{2, node3};
  auto node1 = new (fixture.mallocLocal<Node>()) Node{1, node2};
  auto actual = fixture.localToRemote(*node1, *nodeMeta);
  CHECK(&actual != node1);
  CHECK(actual.data == 1);
  CHECK(actual.next != node2);
  CHECK(actual.next->data == 2);
  CHECK(actual.next->next != node3);
  CHECK(actual.next->next->data == 3);
  CHECK(actual.next->next->next == nullptr);
  CHECK(fixture.remoteAllocations.size() == 3); // 3 allocations: 3 Nodes

  actual.next->next->data = 42;
  CHECK(fixture.remoteToLocal(node3) == std::optional{reinterpret_cast<uintptr_t>(actual.next->next)});

  CHECK(node3->next == nullptr);
  CHECK(node3->data == 42);
  CHECK(node2->next == node3);
  CHECK(node2->data == 2);
  CHECK(node1->next == node2);
  CHECK(node1->data == 1);
}

TEST_CASE("linkedlist-indirect") {
  struct Node {
    int32_t data;
    struct Other *other;
  };

  struct Other {
    Node *value;
  };

  auto nodeMetaDeferred = Struct_(Node);
  auto otherMetaDeferred = Struct_(Other);
  auto otherMeta = Struct_(Other,                                          //
                           StructMember_(Other, value, &*nodeMetaDeferred) //
  );

  auto nodeMeta = Struct_(Node,                                           //
                          StructMember_(Node, data, &int32Type),          //
                          StructMember_(Node, other, &*otherMetaDeferred) //
  );
  (*nodeMetaDeferred).memberCount = (*nodeMeta).memberCount;
  (*nodeMetaDeferred).members = (*nodeMeta).members;
  (*otherMetaDeferred).memberCount = (*otherMeta).memberCount;
  (*otherMetaDeferred).members = (*otherMeta).members;

  Fixture fixture;
  auto node3 = new (fixture.mallocLocal<Node>()) Node{3, new (fixture.mallocLocal<Other>()) Other{nullptr}};
  auto node2 = new (fixture.mallocLocal<Node>()) Node{2, new (fixture.mallocLocal<Other>()) Other{node3}};
  auto node1 = new (fixture.mallocLocal<Node>()) Node{1, new (fixture.mallocLocal<Other>()) Other{node2}};

  auto remote = fixture.localToRemote(*node1, *nodeMeta);

  const char *p = reinterpret_cast<char *>(node1);
  (*nodeMeta).visualise(stderr, [&](size_t offset, const AggregateMember &m) {
    auto x = p + offset;
    std::fprintf(stderr, "value=");
    if (m.ptrIndirection != 0) {
      polyregion::compiletime::showPtr(stderr, sizeof(void *), x);
    } else {
      polyregion::compiletime::showInt(stderr, false, m.type->sizeInBytes, x);
    }
  });

  CHECK(&remote != node1);
  CHECK(remote.data == 1);
  CHECK(remote.other != node1->other);
  CHECK(remote.other->value != node2);
  CHECK(remote.other->value->data == 2);
  CHECK(remote.other->value->other->value != node3);
  CHECK(remote.other->value->other->value->data == 3);
  CHECK(remote.other->value->other->value->other->value == nullptr);
  CHECK(fixture.remoteAllocations.size() == 6); // 6 allocations: 3 Nodes + 3 Others
  CHECK(node3->other->value == nullptr);

  remote.other->value->other->value->data = 42;
  CHECK(fixture.remoteToLocal(node3) == std::optional{reinterpret_cast<uintptr_t>(remote.other->value->other->value)});

  CHECK(node3->other->value == nullptr);
  CHECK(node3->data == 42);
  CHECK(node2->other->value == node3);
  CHECK(node2->data == 2);
  CHECK(node1->other->value == node2);
  CHECK(node1->data == 1);
}

TEST_CASE("ptr") {
  struct Bar {
    int64_t x;
    float *a;
  };
  struct Foo {
    float *a;
    int32_t b;
    float *c;
    Bar *bar;
    Bar barOpaque;
  };
  auto barMeta = Struct_(Bar,                               //
                         StructMember_(Bar, x, &int64Type), //
                         StructMember_(Bar, a, &floatType), //
  );
  auto fooMeta = Struct_(Foo,                                     //
                         StructMember_(Foo, a, &floatType),       //
                         StructMember_(Foo, b, &int32Type),       //
                         StructMember_(Foo, c, &floatType),       //
                         StructMember_(Foo, bar, &*barMeta),      //
                         StructMember_(Foo, barOpaque, &*barMeta) //
  );

  Fixture fixture;
  int N = 10;
  auto as = fixture.mallocLocal<float>(N);
  for (int i = 0; i < N; ++i)
    as[i] = i;
  auto bar = new (fixture.mallocLocal<Bar>()) Bar{.x = 43, .a = as};
  Foo expected{.a = as, .b = 42, .c = nullptr, .bar = bar, .barOpaque = *bar};
  auto actual = fixture.localToRemote(expected, *fooMeta);
  CHECK(&actual != &expected);
  CHECK(actual.a != expected.a);
  CHECK(actual.b == expected.b);
  CHECK(actual.c == expected.c);
  CHECK(actual.bar != expected.bar);
  CHECK(actual.bar->x == expected.bar->x);
  CHECK(actual.bar->a != expected.bar->a);
  CHECK(actual.barOpaque.x == expected.barOpaque.x);
  CHECK(actual.barOpaque.a != expected.barOpaque.a);
  CHECK(std::memcmp(actual.a, expected.a, sizeof(float) * N) == 0);
  CHECK(fixture.remoteAllocations.size() == 3);
}

TEST_CASE("ptr-offset-simple") {
  struct Foo {
    float *a;
  };
  auto fooMeta = Struct_(Foo, StructMember_(Foo, a, &floatType));

  Fixture fixture;
  size_t N = 10;
  size_t Offset = 5;
  auto as = fixture.mallocLocal<float>(N);
  std::iota(as, as + N, 0);

  Foo expected{as + Offset};
  auto actual = fixture.localToRemote(expected, *fooMeta);
  CHECK(&actual != &expected);
  CHECK(actual.a != expected.a);

  CHECK(std::memcmp(actual.a, expected.a, sizeof(float) * (N - Offset)) == 0);
  CHECK(fixture.remoteAllocations.size() == 2);
}

TEST_CASE("ptr-offset-internal") {
  struct Foo {
    float *a;
    float *b;
  };
  auto fooMeta = Struct_(Foo,                               //
                         StructMember_(Foo, a, &floatType), //
                         StructMember_(Foo, b, &floatType)  //

  );
  Fixture fixture;
  size_t N = 10;
  size_t Offset = 5;
  auto as = fixture.mallocLocal<float>(N);
  std::iota(as, as + N, 0);

  Foo expected{as, as + Offset};
  auto actual = fixture.localToRemote(expected, *fooMeta);
  CHECK(&actual != &expected);
  CHECK(actual.a != expected.a);
  CHECK(actual.b != expected.b);

  CHECK(std::memcmp(actual.a, expected.a, sizeof(float) * N) == 0);

  CHECK(std::memcmp(actual.b, expected.b, sizeof(float) * (N - Offset)) == 0);
  CHECK(fixture.remoteAllocations.size() == 2); // struct and a*, no b*

  CHECK(expected.a[0] == 0);
  CHECK(expected.b[0] == 5);

  actual.a[0] = 42;
  actual.b[0] = 43;

  CHECK(fixture.remoteToLocal(expected.b) == std::optional{reinterpret_cast<uintptr_t>(actual.b)});
  CHECK(expected.a[0] == 0);
  CHECK(expected.b[0] == 43);

  CHECK(fixture.remoteToLocal(expected.a) == std::optional{reinterpret_cast<uintptr_t>(actual.a)});
  CHECK(expected.a[0] == 42);
  CHECK(expected.b[0] == 43);
}