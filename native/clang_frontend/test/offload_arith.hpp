#include "offload.hpp"





TEST_CASE("Invoke arith") {

  int aaaa = 42;
  int bbbbbb = 12;
  std::vector as{1};



  struct lam2{
    int aaaa;
    int bbbbbb;
    int operator()(){
      return aaaa + bbbbbb;
    }
  };


  assertOffload<int>([&]() {
    int ccc = 23;
    return aaaa + bbbbbb + ccc;
  });

}

// list(FILES  offload_arith.hpp)
// offload_arith.cpp*