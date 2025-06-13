#ifndef TENSOR_IO_CUH
#define TENSOR_IO_CUH

#include "SparseTensor.cuh"

namespace mxt
{
namespace io
{


template <typename Index_t, typename Value_t, uint32_t Order>
void parse_frostt_line(std::string& line, Index_t& idx, Value_t& val)
{
    size_t spos = 0;
    size_t epos = 0;
    for (uint32_t i=0; i<Order; i++)
    {
        epos = line.find_first_of(' ', spos);
        idx[i] = std::stoul(line.substr(spos, (epos - spos)));
        spos = epos;
    }
    val = static_cast<Value_t>(std::stod(line.substr(epos - spos)));
}


template <typename SparseTensor_t>
SparseTensor_t read_tensor_frostt(const char * fpath)
{

    using Index_t = typename SparseTensor_t::Index;
    using Value_t = typename SparseTensor_t::ValueType;

    std::ifstream infile;
    infile.open(fpath);
    
    //First, get the tensor order
    std::string line;
    std::getline(infile, line);

    uint32_t order = 0;
    for (char c : line)
    {
        if (c == ' ')
            order++;
    }

    ASSERT( (order == SparseTensor_t::Order), "Specified tensor order %lu not same as order %lu in file", SparseTensor_t::Order, order);


    // Reset back to start
    infile.seekg(0);

    Index_t idx;
    Value_t val;

    // Parse entries
    std::vector<Index_t> inds;
    std::vector<Value_t> vals;
    while (std::getline(infile, line))
    {
        parse_frostt_line<Index_t, Value_t, SparseTensor_t::Order>(line, idx, val);
        inds.push_back(idx);
        vals.push_back(val);
    }

    SparseTensor_t tensor(std::move(inds), std::move(vals));

    return tensor;
}



} //io
} //mxt

#endif
