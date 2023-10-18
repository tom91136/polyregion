#ifdef RUNTIME_ENABLE_METAL
  #define METALCPP_SYMBOL_VISIBILITY_HIDDEN
  #define NS_PRIVATE_IMPLEMENTATION
  #define MTL_PRIVATE_IMPLEMENTATION
  #include <Foundation/Foundation.hpp>
  #include <Metal/Metal.hpp>
#endif