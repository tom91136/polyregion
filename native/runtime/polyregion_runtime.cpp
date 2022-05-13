#include "polyregion_runtime.h"
#include "runtime.h"
#include "utils.hpp"
#include <iostream>

using namespace polyregion;

static_assert(                                //
    std::is_same_v<                           //
        decltype(polyregion_type::ordinal),   //
        std::underlying_type_t<runtime::Type> //
        >);

const polyregion_type POLYREGION_BOOL = {to_underlying(runtime::Type::Bool8)};
const polyregion_type POLYREGION_BYTE = {to_underlying(runtime::Type::Byte8)};
const polyregion_type POLYREGION_CHAR = {to_underlying(runtime::Type::CharU16)};
const polyregion_type POLYREGION_SHORT = {to_underlying(runtime::Type::Short16)};
const polyregion_type POLYREGION_INT = {to_underlying(runtime::Type::Int32)};
const polyregion_type POLYREGION_LONG = {to_underlying(runtime::Type::Long64)};
const polyregion_type POLYREGION_FLOAT = {to_underlying(runtime::Type::Float32)};
const polyregion_type POLYREGION_DOUBLE = {to_underlying(runtime::Type::Double64)};
const polyregion_type POLYREGION_PTR = {to_underlying(runtime::Type::Ptr)};
const polyregion_type POLYREGION_VOID = {to_underlying(runtime::Type::Void)};

struct polyregion_object {
  std::unique_ptr<runtime::Object> data;
  explicit polyregion_object(std::unique_ptr<runtime::Object> data) : data(std::move(data)) {}
};

void polyregion_release_object(polyregion_object_ref *ref) {
  if (ref) {
    polyregion::free_str(ref->message);
    delete ref->object;
    delete ref;
  }
}

polyregion_object_ref *polyregion_load_object(const uint8_t *object, size_t object_size) {
  auto ref = new polyregion_object_ref{};
  try {
    ref->object =
        new polyregion_object(std::make_unique<runtime::Object>(std::vector<uint8_t>(object, object + object_size)));
  } catch (const std::exception &e) {
    ref->message = new_str(e.what());
  }
  return ref;
}

void polyregion_release_enumerate(polyregion_symbol_table *table) {
  if (table) {
    for (size_t i = 0; i < table->size; ++i) {
      polyregion::free_str(table->symbols[i].name);
    }
    delete[] table->symbols;
    delete table;
  }
}

polyregion_symbol_table *polyregion_enumerate(const polyregion_object *object) {
  auto table = object->data->enumerate();
  auto xs = new polyregion_symbol[table.size()];
  std::transform(table.begin(), table.end(), xs, [](auto &p) {
    // copy name here as the symbol table is deleted with dyld
    return polyregion_symbol{polyregion::new_str(p.first), p.second};
  });
  return new polyregion_symbol_table{xs, table.size()};
}

void polyregion_release_invoke(char *err) { polyregion::free_str(err); }

int runit(  )
{

  runtime::run();


//  auto api = ( oroApi )( ORO_API_CUDA   );
//
//
//  auto check = [](oroError & e)  {
//    if(e != oroSuccess){
//      const char* pStr;
//      oroGetErrorString( e, &pStr );
//      printf("error %d: %s\n", e, pStr);
//    }
//  };
//
//  int a = oroInitialize( api, 0 );
//  if( a != 0 )
//  {
//    printf("initialization failed\n");
//    return 0;
//  }
//  printf( ">> executing on %s\n", ( api == ORO_API_HIP )? "hip":"cuda" );
//
//  printf(">> testing initialization\n");
//  oroError e;
//
//  e = oroInit( 0 );
//  check(e);
//  oroDevice device;
//  e = oroDeviceGet( &device, 0 );
//  check(e);
//  oroCtx ctx;
//  e = oroCtxCreate( &ctx, 0, device );
//  check(e);
//
//  printf(">> testing device props\n");
//  {
//    oroDeviceProp props;
//    oroGetDeviceProperties( &props, device );
//    printf("executing on %s (%s)\n", props.name, props.gcnArchName );
//  }
//  printf(">> testing kernel execution\n");
//  {
//    oroFunction function;
//    {
//      const char* code = "extern \"C\" __global__ void testKernel(float c, float *out) { int a = threadIdx.x; out[a] = c; }";
//      const char* funcName = "testKernel";
//      orortcProgram prog;
//      orortcResult e;
//      e = orortcCreateProgram( &prog, code, funcName, 0, 0, 0 );
//      std::vector<const char*> opts;
//      opts.push_back( "-I ../ -arch=sm_61" );
//
//      e = orortcCompileProgram( prog, opts.size(), opts.data() );
//      if( e != ORORTC_SUCCESS )
//      {
//        size_t logSize;
//        orortcGetProgramLogSize(prog, &logSize);
//        if (logSize)
//        {
//          std::string log(logSize, '\0');
//          orortcGetProgramLog(prog, &log[0]);
//          std::cout << log << '\n';
//        };
//      }
//      size_t codeSize;
//      e = orortcGetCodeSize(prog, &codeSize);
//
//      std::vector<char> codec(codeSize);
//      e = orortcGetCode(prog, codec.data());
//      e = orortcDestroyProgram(&prog);
//      oroModule module;
//      oroError ee = oroModuleLoadData(&module, codec.data());
//      check(ee);
//      ee = oroModuleGetFunction(&function, module, funcName);
//      check(ee);
//      printf("Compiled\n");
//    }
//
//    auto* a  = static_cast<float *>(std::malloc(sizeof(float) * 10));
//    oroDeviceptr d_a  ;
//    auto v = oroMalloc( &d_a, sizeof(float) * 10);
//
//
//    float ans = 42;
//    void* args[] = {
//        &ans,
//        &d_a
//    };
//    oroError e = oroModuleLaunchKernel( function, 1,1,1, 10,1,1, 0, nullptr, args, nullptr );
//    check(e);
//
//
//
//    oroMemcpyDtoH(a,d_a, sizeof(float) * 10 );
//    oroDeviceSynchronize();
//
//    for (int i = 0; i < 10; ++i) {
//      printf("[%d] = %f\n", i, a[i]);
//    }
//  }
//  printf(">> done\n");
  return 0;
}

char *polyregion_invoke(const polyregion_object *object,
                        const char *symbol,                        //
                        const polyregion_data *args, size_t nargs, //
                        polyregion_data *rtn                       //
) {


  runit();
  std::exit(0);

  auto toTyped = [](const auto &data) -> runtime::TypedPointer {
    return std::make_pair(static_cast<runtime::Type>(data.type.ordinal), data.ptr);
  };

  std::vector<runtime::TypedPointer> typedArgs(nargs);
  std::transform(args, args + nargs, typedArgs.begin(), toTyped);

  try {
    object->data->invoke(
        symbol, [](size_t size) { return std::malloc(size); }, typedArgs, toTyped(*rtn));
    return nullptr;
  } catch (const std::exception &e) {
    return new_str(e.what());
  }
}
