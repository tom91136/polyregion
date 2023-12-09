//#include <vector>
//#include <string>
//
#include <array>
#include <string>
#include <vector>

struct foobarbaz{int a;};

int main (){

  struct foobar{int a;};
  foobar value;
  int foo[12];
  int a = __polyregion_offload_f1__([&]() {
    foo[1] = 42;

//std::vector<int> a(2);
//a.emplace_back(2);
//    std::opt<int, 3> foo{};
//    foobar that;
//    that.a = 123;
//
////    foobar aaa;
////    aaa.a = 12;
//
////    that = value;
    return 42;
  });
//  printf("a = %d\n", value.a);
//  printf("Done\n");

}

