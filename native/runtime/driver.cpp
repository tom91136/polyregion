#include "runtime.h"

int main(int argc, char*argv[]) {
  std::string path = "/home/tom/polyregion/native/the_obj2.o";
  //  std::string path = "/home/tom/Desktop/prime.o";
  std::fstream s(path, std::ios::binary | std::ios::in);
  if (!s.good()) {
    throw std::invalid_argument("Bad file: " + path);
  }
  s.ignore(std::numeric_limits<std::streamsize>::max());
  auto len = s.gcount();
  s.clear();
  s.seekg(0, std::ios::beg);
  std::vector<uint8_t> xs(len / sizeof(uint8_t));
  s.read(reinterpret_cast<char *>(xs.data()), len);
  s.close();

  int u;
  polyregion_data rtn{.type = polyregion_type::Void, .ptr = &u};

  std::vector<float> data = {1.1, 2.1, 3.1};
  auto ptr = data.data();
  polyregion_data arg1{.type = polyregion_type::Ptr, .ptr = &ptr}; // XXX pointer to T, so ** for pointers

  auto err = polyregion_invoke(xs.data(), xs.size(), "lambda", &arg1, 1, &rtn);

  //  int exp = 0;
  //  int in = 99;
  //  polyregion_data arg1{.type = polyregion_type::Int, .ptr = &in};
  //  polyregion_data rtn{.type = polyregion_type::Int, .ptr = &exp};
  //  polyregion_invoke(xs.data(), xs.size(), "lambda", &arg1, 1, &rtn, &err);

  auto mk = [](float f) {
    long long unsigned int f_as_int = 0;
    std::memcpy(&f_as_int, &f, sizeof(float));            // 2.
    std::bitset<8 * sizeof(float)> f_as_bitset{f_as_int}; // 3.
    return f_as_bitset;
  };

  if (err) {
    std::cerr << "Err:" << err << std::endl;
  } else {

    std::bitset<32> act1 = mk(data[0]);
    std::bitset<32> act2 = mk(data[1]);

    float d = 1;
    std::bitset<32> exp = mk(d);
    std::cout << exp << '\n';
    std::cout << act1 << '\n';
    std::cout << act2 << '\n';

    std::cout << "r=" << data[0] << std::endl;
    std::cout << "r=" << data[1] << std::endl;
    std::cout << "r=" << data[2] << std::endl;
    std::cout << "OK" << err << std::endl;
  }
  polyregion_consume_error(err);


}