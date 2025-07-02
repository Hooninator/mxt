
#ifndef CONFIG_HPP
#define CONFIG_HPP

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


template <typename InputModes, typename TuckerRanks, typename HighU, typename LowU, typename LraU, typename CoreTensorU, typename Idx>
struct Config
{
    using InputModes_t = InputModes;
    using TuckerRanks_t = TuckerRanks;
    using HighU_t = HighU;
    using LowU_t = LowU;
    using CoreTensorU_t = CoreTensorU;
    using LraU_t = LraU;
    using Idx_t = Idx;

    static constexpr uint32_t Order = InputModes_t::dims.size();


    static void print(const std::string& tensor)
    {
        std::cout<<"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"<<std::endl;
        std::cout<<"\tTENSOR: "<<tensor<<std::endl;
        std::cout<<"\tINPUT PRECISION: "<<get_type<HighU_t>()<<std::endl;
        std::cout<<"\tTTMC PRECISION: "<<get_type<LowU_t>()<<std::endl;
        std::cout<<"\tCORE TENSOR PRECISION: "<<get_type<CoreTensorU_t>()<<std::endl;
        std::cout<<"\tSVD PRECISION: "<<get_type<LraU_t>()<<std::endl;
        std::cout<<"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"<<std::endl;
    }


    static std::string str()
    {
        std::stringstream ss;
        ss<<"ttmc:"<<get_type<LowU_t>()<<"_core:"<<get_type<CoreTensorU_t>()
            <<"_lra:"<<get_type<LraU_t>();
        return ss.str();
    }


};
} //mxt

#endif
