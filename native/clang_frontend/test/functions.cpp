#include <execution>
#include <stdlib.h>
#include <ranges>

int main() {
  //  std::vector<int> xs{1, 2, 3};
  //  std::vector<int> ys(xs.size());

  struct A {
    int memberA = 42;

    void run() {
      auto xs = (int *)malloc(42);
      int* zs = (int *)malloc(42);
         int* ys = (int *)malloc(42);

      int hello = 42;
      int bar = 9;
      auto &u = bar;



      std::array<int, 3> foo{1,2,3};
      std::vector<int> arr{1};

      auto rng = std::ranges::iota_view {1,2};

#pragma split
      auto m = [](){

      };

      std::transform(std::execution::par, xs, xs + 10, zs, //
                     [&](auto &foobar) { // TODO implement lambda args!!!
                       //

//                       int v = rng[120];
//                       arr[0] = 42;

//                       int _a = 3;
//                       int &a = _a;
//                       a = 32;
//
//                       int back;
//                       back = a;




//                       ys[1] = 1;
//                       int &aaa = ys[12];
////                       aaa = 3;
//                       int &mmmm = ys[42];
//                       aaa = mmmm;
//                       auto &uuu = ys[12];
//                       uuu = 3;

                       int &v = ys[42];
                       int m = foobar;


                       int &vr = v;
                       int vx = v;

                       int &mr = m;
                       int mx = m;

                       int o = mr + 1;
                        mr += mr;

//                       w = 42;
//                       ys[123] = 456;


                       //x + hello + bar + u + this->memberA
                       return  mr;
                     });
    }
  };
}