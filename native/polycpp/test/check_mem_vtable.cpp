#pragma region case: vtable
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -o {output} {input} -g0
#pragma region do: {output}
#pragma region requires: 42 42 42 42 42 43

#include <cstddef>
#include <cstdio>
#include <cstring>

#include "test_utils.h"

struct Base {
  int *data;
  size_t size;
  [[clang::noinline]] Base(int *data, size_t size) : data(data), size(size) {}
  [[clang::noinline]] virtual int foo(int value) = 0;
  [[clang::noinline]] virtual void bar() = 0;
  virtual ~Base() {};
};

struct Derived : Base {
  [[clang::noinline]] Derived(int *data, size_t size) : Base(data, size) {}
  [[clang::noinline]] int foo(int value) override {
    return 1 + __polyregion_offload_f1__([=, data = this->data, size = this->size]() {
             for (size_t i = 0; i < size; i++) {
               data[i] = value;
             }
             return value;
           });
  }

  [[clang::noinline]] void bar() override {
    for (size_t i = 0; i < size; i++) {
      printf("%d ", data[i]);
    }
  }
};

int main() {

  size_t size = 5;
  auto data = static_cast<int *>(malloc(size * sizeof(int)));
  std::memset(data, -1, size * sizeof(int));
  Base *instance = new Derived(data, size);
  //  auto laundered = instance; //std::launder(instance);

  auto result = instance->foo(42);
  instance->bar();
  printf("%d", result);

  delete instance;
  std::free(data);
  return 0;
}
