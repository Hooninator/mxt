#ifndef SHAPE_CUH
#define SHAPE_CUH

#include "common.cuh"


template <size_t... Dims>
struct Shape
{
    static constexpr std::array<size_t , sizeof...(Dims)> dims = {Dims...};
};

template <typename std::size_t Exclude, typename ShapeT>
struct RemoveOneToShape;

template <std::size_t Exclude, std::size_t... Dims>
struct RemoveOneToShape<Exclude, Shape<Dims...>>  
{
    static constexpr std::size_t N = sizeof...(Dims);
    static constexpr std::array<std::size_t, N> Arr = {Dims...};
    static constexpr bool IgnoreRemove = (Exclude == N); 

    template <std::size_t... Is>
    static constexpr Shape<(IgnoreRemove ? Arr[Is] : Arr[Is < Exclude ? Is : Is + 1])...> filter(std::index_sequence<Is...>)
    {
        return {};
    }

    using type = decltype(filter(std::make_index_sequence<IgnoreRemove ? N : N - 1>{}));
};



#endif
