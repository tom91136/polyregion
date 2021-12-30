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
template <typename T> using arg1_t = typename arg1<T>::type;

} // namespace details

template <typename T, typename Variant> struct is_variant_member;
template <typename T, typename... Ts>
struct is_variant_member<T, std::variant<Ts...>> : public std::disjunction<std::is_same<T, Ts>...> {};

template <class... Ts> struct overloaded : Ts... { using Ts::operator()...; };
template <class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

template <                                                                                  //
    typename Variant, typename... Ts,                                                       //
    typename = std::enable_if_t<                                                            //
        std::conjunction_v<                                                                 //
            is_variant_member<std::decay_t<details::arg1_t<Ts>>, std::decay_t<Variant>>...> //
        >                                                                                   //
    >                                                                                       //
constexpr auto total(Variant &&v, Ts &&...ts) {
  return std::visit(overloaded{ts...}, std::forward<Variant>(v));
}

}; // namespace polyregion::variants