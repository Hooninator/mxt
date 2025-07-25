
#ifndef TTMC_CONFIG_HPP
#define TTMC_CONFIG_HPP

#include <string>
#include <sstream>
#include <iostream>

#include "MatrixCollection.cuh"

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


template <typename MatrixCols, typename MatrixRows, typename InputType, typename ComputeTypeT, typename AccumType, typename Normalizer, int ComputeTypeIT, int GenTypeT>
struct TtmcConfig
{
    using MatrixCols_t = MatrixCols;
    using MatrixRows_t = MatrixRows;
    using InputType_t = InputType;
    using AccumType_t = AccumType;
    using ComputeType_t = ComputeTypeT;
    using Normalizer_t = Normalizer;

    static_assert(MatrixCols_t::dims.size() == MatrixRows_t::dims.size());

    static constexpr uint32_t Order = MatrixCols_t::dims.size();
    static constexpr cublasComputeType_t ComputeType = static_cast<cublasComputeType_t>(ComputeTypeIT);
    static constexpr MatrixGenerator_t Generator = static_cast<MatrixGenerator_t>(GenTypeT);

    inline static const std::map<const int, std::string> compute_type_map
    {
        {CUBLAS_COMPUTE_64F, "compute64f"},
        {CUBLAS_COMPUTE_32F, "compute32f"},
        {CUBLAS_COMPUTE_32F_FAST_16F, "compute32f16f"},
        {CUBLAS_COMPUTE_16F, "compute16f"}
    };


    inline static const std::map<const int, std::string> gen_type_map
    {
        {GEN_SMALL, "gen_small"},
        {GEN_RANDN, "gen_randn"},
        {GEN_BIG, "gen_big"}
    };


    static void print(const std::string& tensor)
    {
        std::cout<<"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"<<std::endl;
        std::cout<<"\tTENSOR: "<<tensor<<std::endl;
        std::cout<<"\tMATRIX ROWS: "<<rows_str()<<std::endl;
        std::cout<<"\tACCUMULATION TYPE: "<<get_type<AccumType_t>()<<std::endl;
        std::cout<<"\tCOMPUTE TYPE: "<<compute_type()<<std::endl;
        std::cout<<"\tMATRIX GENERATION TYPE: "<<gen_type()<<std::endl;
        std::cout<<"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"<<std::endl;
    }


    static std::string str()
    {
        std::stringstream ss;
        ss<<"lowu-"<<get_type<AccumType_t>()<<"_"
          <<"compute-"<<compute_type()<<"_"
          <<"gen-"<<gen_type()<<"_"
          <<"rows-"<<rows_str();

        return ss.str();
    }

    static std::string rows_str()
    {
        std::stringstream ss;
        for (int i=0; i<Order; i++)
        {
            ss<<MatrixRows::dims[i];
            if (i < Order - 1)
            {
                ss<<"x";
            }
        }
        return ss.str();
    }

    static std::string compute_type()
    {
        return compute_type_map.at(ComputeTypeIT);
    }


    static std::string gen_type()
    {
        return gen_type_map.at(GenTypeT);
    }

};
} //mxt

#endif
