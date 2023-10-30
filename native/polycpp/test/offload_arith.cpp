//#include <vector>
//#include <string>
//

struct foobarbaz{int a;};

int main (){

  struct foobar{int a;};
  foobar value;
  foobar c = __polyregion_offload_f1__([&]() {

    foobar that;
    that.a = 123;

//    foobar aaa;
//    aaa.a = 12;

//    that = value;
    return that;
  });
  printf("a = %d\n", value.a);
  printf("Done\n");

}

