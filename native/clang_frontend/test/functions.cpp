// #include <algorithm>
#include <execution>
// #include <vector>
#include <stdlib.h>

namespace polystl {
template <class _ExecutionPolicy, //
          class _ForwardIterator1, class _ForwardIterator2,
          class _UnaryOperation>
_ForwardIterator2 transform(_ExecutionPolicy &&__exec, //
                            _ForwardIterator1 __first, //
                            _ForwardIterator1 __last,  //
                            _ForwardIterator2 __result, _UnaryOperation __op42) {
  //  auto N = std::distance(__first, __last);

  for (int i = 0; i < 10; ++i) {
    __result[i] = __op42(__first[i]);
  }
  //        [[lift]] std::vector<Type::> m = __op;
  // enqueueInvokeAsync("a", "b", {Type::Ptr, Type::Ptr, <captureTpes...>}, {&__first, &__result, <captures...>}, {N, 1, 1});
  //          void kernel(__first : Struct__First){   __result[i] = __op(__first[i])    }
  //        while (__first != __last) *__result++ = __op(*__first++);
  return __result;
}
} // namespace polystl

template <typename F> void shim(int a, F fff) {
  std::for_each_n(std::execution::par_unseq, (int *)0, 1, [&](auto &v) {
    auto m = fff(v);
    ;;
    //
    /**/
#pragma
  return m;});
}

//int &add(int a){
//  return a;
//}

int main() {
  auto xs = (short *)malloc(42);
  auto ys = (short *)malloc(42);
  //  std::vector<int> xs{1, 2, 3};
  //  std::vector<int> ys(xs.size());
  int hello = 42;
  int bar = 9;

    std::transform(std::execution::par, xs , xs+ 10, ys , [&](auto &x) {

    // Ref[T]
//    int xs[2];
//    int &uuu = xs[0];
//    uuu = 12;

//
//    int &foo = xs[1];
//    foo = 2;
//    int bar  ;
//    bar = 3;

//    add(1) =2;



    std::array<int, 2> a;
    int vv = (a[0] = 2) = 4;

////    a[1] = 42;
////    auto it = a.begin();
//    int u = 1;
    return 0;

//    int xs[] = {0, x};
//    int ys[3] = {42};
//
//    int m = xs[0];
//
//
//
//   uint a = 111;
//
//   // var i : Array[uint] = View(a)
//   uint* xxx = &a;
//
//   uint8_t b;
//   uint16_t c;
//   uint32_t d;
//   uint64_t e;
//
//
//   int a_ = 22;
//   int8_t b_;
//   int16_t c_;
//   int32_t d_;
//   int64_t e_;
//
//   float aa;
//   double bb;
//   bool aaa = true;
//
//
//   for (int i = 0; i < a_; ++i) {
//     x+=1;
//   }
//   int cc = 0;
//   while(cc < 10){
//     if(a_ == c) break;
//     if(a_ == d) continue ;
//     cc++;
//   }
//
//   if(aaa){
//     return 1;
//   }else if (a_ == 1){
//     return x + hello + hello + bar + 2;
//   }else {
//     return 3;
//   }

  });

  struct a {



    void m() {
      double xxx = 42;
      int baz = 42;




      auto xs1 = (double *)malloc(42);
      auto ys1 = (double *)malloc(42);
      auto xs = (int *)malloc(42);
      auto ys = (int *)malloc(42);

      auto ff = [=](auto &x) { return x * baz + baz + x + xxx; };

      //      std::transform(std::execution::par_unseq, xs1 , xs1+ 10, ys1 , ff);
      //      static constexpr auto X = std::execution::par_unseq;
      //      std::transform(X,   xs , xs+ 10, ys ,   (ff) );
      //      std::for_each_n(X,   xs ,   10,    (ff) );
      //      auto a = [=](auto x){
      //        std::for_each_n(X,   (short*)0 ,   10,    x );
      //      };
      //      a(ff);

//      shim(10, [](auto x) { return x + 42; });
    }
  };
}