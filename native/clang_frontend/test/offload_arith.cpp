//#include <vector>
//#include <string>
//



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
  //  assertOffload1<int>( [&]() {
  ////    printf("A\n");
  //    int ccc = 123;
  //    ccc += mmm(12, 34) + mmm(56,78);
  //    return ccc;
  //  });

  assertOffload1<short>([&]() {
    //    printf("B\n");
    return 42 + aaaa;
  });
}

struct A{
  int& xs;
  A(int&aa): xs(aa){}
};


int twice(int* a){
  return *a*2;
}

struct Foo{
  int a;
};
int main (){
//TEST_CASE("Invoke arith") {
  //  printf("In\n");
  //  m<int>();
  //  printf("===\n");
  //  m<double>();

  int32_t a = 42;
  int32_t b = 42;
  auto x = []{ return 1; };
  Foo aa;

  int32_t c = __polyregion_offload_f1__([&]() {
//    A aa(a);
//    aa.xs = 2;
//    a = 3;
//    b +=2;

//    a++;


   return 1;
  });
  printf("a = %d\n", a);
  printf("b = %d\n", b);
  printf("c = %d\n", c);
  printf("aa = %d\n", aa.a);
  printf("Done\n");

}

// list(FILES  offload_arith.hpp)
// offload_arith.cpp*