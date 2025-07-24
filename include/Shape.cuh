#ifndef SHAPE_CUH
#define SHAPE_CUH

#include "common.cuh"


template <size_t... Dims>
struct Shape
{
    static constexpr std::array<size_t , sizeof...(Dims)> dims = {Dims...};
};


template <std::size_t Mode, typename ShapeT>
struct AdjustShape;

template <std::size_t Mode, std::size_t... Dims>
struct AdjustShape<Mode, Shape<Dims...>>  
{
    static constexpr std::size_t N = sizeof...(Dims);
    static constexpr std::array<std::size_t, N> Arr = {Dims...};

    template <std::size_t... Is>
    static constexpr Shape<(Arr[(Is < Mode ? Is : Is + 1)])...> filter(std::index_sequence<Is...>)
    {
        return {};
    }

    using type = decltype(filter(std::make_index_sequence<N - 1>{}));
};



#endif
