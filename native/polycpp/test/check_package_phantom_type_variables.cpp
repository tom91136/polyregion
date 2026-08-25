#pragma region case: package-phantom-type-variables
#pragma region offload-only
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -fstdpar-emit-library={output}.polyast -c -o {output}.o {input}

#define POLYREGION_EXPORT_AS(name) [[clang::annotate("polyregion_export:" name)]]
#define POLYREGION_TYPE_VARIABLE(name) [[clang::annotate("polyregion_type_variable:" name)]]
#define POLYREGION_CALLABLE_VARIABLE(name) [[clang::annotate("polyregion_callable_variable:" name)]]

struct POLYREGION_TYPE_VARIABLE("Element") Element {
  int value;
};

struct POLYREGION_CALLABLE_VARIABLE("Callable0") Callable {
  int value;
};

struct Closure {
  Callable callable;
};

template <class T> struct Box {};

template <class T> Box<T> makeBox(Box<T> box) {
  T value{};
  (void)value;
  return box;
}

template <class T> struct Right;

template <class T> struct Left {
  Right<T> *right;
};

template <class T> struct Right {
  Left<T> *left;
};

template <class T> struct Ref {
  Left<T> *left;
};

template <class T> Ref<T> makeRef(Left<T> *left) {
  T value{};
  (void)value;
  return {left};
}

int increment(int value) { return value + 1; }
int invoke(int (*fn)(int), int value);

struct Sink {
  int invoke(int (*fn)(int), int value);
};

struct OperatorSink {
  int operator<<(int (*fn)(int));
};

POLYREGION_EXPORT_AS("foo.implementation.apply") Element apply(Element value, Callable callable) {
  Left<Element> left{};
  (void)makeRef(&left);
  Box<Closure> box{};
  (void)makeBox(box);
  (void)invoke(&increment, 1);
  Sink sink{};
  (void)sink.invoke(&increment, 1);
  OperatorSink operatorSink{};
  (void)(operatorSink << &increment);
  (void)callable;
  return value;
}
