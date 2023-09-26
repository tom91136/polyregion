//#include <vector>
//#include <string>
//
#include "offload.hpp"

template<typename F>
struct wrapper1 {
  F f;
  wrapper1(F f) :f(f) {}
  auto operator ()() const { return f(); }
};

//template<typename F>
//wrapper1<F> wrap0(F f) {
//  return wrapper1<F>{f};
//}


template < typename T, typename F> void assertOffload1(F f){
  assertOffload<T>([&]() {
    return f();
  });
}

template< typename>
void m(){
  int aaaa = 42;
  int bbbbbb = 12;
//  std::vector as{1};



//  struct lam2{
//    int aaaa;
//    int bbbbbb;
//    int operator()(){
//      return aaaa + bbbbbb;
//    }
//  };


  int c = 12;
  const auto mmm = [&](auto a, auto b){
    return a + b + c;
  };



//
  assertOffload1<int>( [&]() {
    printf("A\n");
    int ccc = 123;
    ccc += mmm(12, 34) + mmm(56,78);
    return ccc;
  });


  assertOffload1<short>([&]() {
    printf("B\n");
    return 4242;
  });
}

int main (){
//TEST_CASE("Invoke arith") {
 printf("In\n");
 m<int>();
 printf("===\n");
 m<double>();
 printf("Done\n");

}

// list(FILES  offload_arith.hpp)
// offload_arith.cpp*