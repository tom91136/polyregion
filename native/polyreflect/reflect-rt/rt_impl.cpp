// Single owner of the polyreflect-rt runtime. Defines `_rt_record` / `_rt_release` /
// `_rt_reflect_*` and the leaked `ReflectService` instance; consumers link against this .so.
#define __RT_IMPL
#include "reflect-rt/rt_reflect.hpp"
