#pragma region case: package-phantom-type-variables
#pragma region offload-only
#pragma region do: polycpp {polycpp_defaults} {polycpp_stdpar} -fstdpar-emit-library={output}.polyast -c -o {output}.o {input}
#pragma region do: {package_fixture} --assert-no-value-pointer-assignments {output}.polyast
#pragma region do: {package_fixture} --assert-zero-type-variable {output}.polyast
#pragma region do: {package_fixture} --assert-zero-type-variable-return {output}.polyast

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

Element copyElement(Element value) {
  Element copy(value);
  Element moved(static_cast<Element &&>(copy));
  return moved;
}

POLYREGION_EXPORT_AS("foo.implementation.zero") Element zeroElement() { return Element{}; }

Element chooseElement(bool takeFirst, Element *values) { return takeFirst ? values[0] : values[1]; }

int increment(int value) { return value + 1; }
int invoke(int (*fn)(int), int value);

struct Sink {
  int invoke(int (*fn)(int), int value);
};

struct OperatorSink {
  int operator<<(int (*fn)(int));
};

POLYREGION_EXPORT_AS("foo.implementation.apply") Element apply(Element value, Callable callable) {
  Element values[2]{};
  values[0] = value;
  values[1] = values[0];
  auto incrementPointer = &increment;
  Left<Element> left{};
  (void)makeRef(&left);
  Box<Closure> box{};
  (void)makeBox(box);
  (void)invoke(&increment, 1);
  Sink sink{};
  (void)sink.invoke(&increment, 1);
  OperatorSink operatorSink{};
  (void)(operatorSink << &increment);
  (void)incrementPointer;
  (void)callable;
  return copyElement(chooseElement(incrementPointer != nullptr, values));
}
