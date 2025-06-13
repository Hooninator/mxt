#ifndef SPARSE_TENSOR_CUH
#define SPARSE_TENSOR_CUH

#include "common.cuh"
#include "utils.cuh"


namespace mxt
{

template <uint32_t O, typename VT, typename IT>
class SparseTensor
{
public:

    using ValueType = VT;
    using IndexType = IT;
    static constexpr uint32_t Order = O;

    using Index = std::array<IndexType, Order>;

    SparseTensor(std::vector<Index>&& inds, std::vector<ValueType>&& vals):
        inds(std::move(inds)),
        vals(std::move(vals))
    {
        ASSERT( (inds.size() == vals.size()), "Number of indices %zu is not equal to number of nonzeros %zu", inds.size(), vals.size() ); 
        this->nnz = inds.size();
    }


    void dump(std::ofstream& ofs)
    {
        for (size_t i=0; i<nnz; i++)
        {
            Index& index = inds[i];
            ValueType& val = vals[i];

            ofs<<"(";
            for (uint32_t j = 0; j < Order; j++)
            {
                ofs<<index[j]<<",";
            }
            ofs<<") -> "<<val<<'\n';
        }
        std::flush(ofs);
    }


private:


    std::vector<Index> inds;
    std::vector<ValueType> vals;

    Index mode_sizes;
    size_t nnz;

};


} //mxt
#endif
