#pragma once

#include <concepts>
#include <functional>
#include <type_traits>
#include <variant>

namespace polyregion::variants {

namespace details {

template <typename F, typename Ret, typename A, typename... Rest> //
A arg1_(Ret (F::*)(A, Rest...));
template <typename F, typename Ret, typename A, typename... Rest> //
A arg1_(Ret (F::*)(A, Rest...) const);

template <typename F> struct arg1 { using type = decltype(arg1_(&F::operator())); };

} // namespace details

template <typename T, typename Variant> struct is_variant_member;
template <typename T, typename... Ts>
struct is_variant_member<T, std::variant<Ts...>> : public std::disjunction<std::is_same<T, Ts>...> {};

template <class... Ts> struct overloaded : Ts... { using Ts::operator()...; };
template <class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

template <typename Variant, typename... Ts,                                               //
          typename =                                                                      //
          typename std::enable_if<                                                        //
              std::conjunction<                                                           //
                  is_variant_member<typename details::arg1<Ts>::type, Variant>...>::value //
              >::type                                                                     //
          >                                                                               //
constexpr auto total(Variant &&v, Ts &&...ts) {
  return std::visit(overloaded{ts...}, v);
}

}; // namespace polyregion::variants