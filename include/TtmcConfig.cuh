
#ifndef TTMC_CONFIG_HPP
#define TTMC_CONFIG_HPP

#include <string>
#include <sstream>
#include <iostream>

namespace mxt
{

template <typename T>
std::string get_type()
{
    std::string name = __PRETTY_FUNCTION__;
    auto start = name.find("T = ") + 4;
    auto end = name.find(";", start);
    return name.substr(start, end - start); 
}


template <typename MatrixCols, typename MatrixRows, typename HighU, typename LowU, int ComputeTypeT>
struct TtmcConfig
{
    using MatrixCols_t = MatrixCols;
    using MatrixRows_t = MatrixRows;
    using HighU_t = HighU;
    using LowU_t = LowU;

    static_assert(MatrixCols_t::dims.size() == MatrixRows_t::dims.size());

    static constexpr uint32_t Order = MatrixCols_t::dims.size();
    static constexpr cublasComputeType_t ComputeType = static_cast<cublasComputeType_t>(ComputeTypeT);

    inline static const std::map<const int, std::string> compute_type_map
    {
        {CUBLAS_COMPUTE_64F, "compute64f"},
        {CUBLAS_COMPUTE_32F, "compute32f"},
        {CUBLAS_COMPUTE_32F_FAST_16F, "compute32f16f"},
        {CUBLAS_COMPUTE_16F, "compute16f"}
    };

    static void print(const std::string& tensor)
    {
        std::cout<<"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"<<std::endl;
        std::cout<<"\tTENSOR: "<<tensor<<std::endl;
        std::cout<<"\tINPUT PRECISION: "<<get_type<HighU_t>()<<std::endl;
        std::cout<<"\tLOW PRECISION: "<<get_type<LowU_t>()<<std::endl;
        std::cout<<"\tCOMPUTE TYPE: "<<compute_type()<<std::endl;
        std::cout<<"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"<<std::endl;
    }


    static std::string str()
    {
        return get_type<LowU_t>() + "-" + std::to_string(ComputeTypeT);
    }


    static std::string compute_type()
    {
        return compute_type_map.at(ComputeTypeT);
    }


};
} //mxt

#endif
