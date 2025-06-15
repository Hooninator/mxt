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

    SparseTensor(){}

    SparseTensor(std::vector<Index>& inds, std::vector<ValueType>& vals, Index& modes):
        nnz(inds.size()), 
        mode_sizes(modes),
        d_vals(utils::h2d_cpy(vals)),
        d_inds(utils::h2d_cpy(inds))
    {
        ASSERT( (inds.size() == vals.size()), "Number of indices %zu is not equal to number of nonzeros %zu", inds.size(), vals.size() ); 
    }


    inline size_t bytes_vals()
    {
        return sizeof(ValueType) * this->nnz;
    }


    inline size_t bytes_inds()
    {
        return sizeof(Index) * this->nnz;
    }


    void dump(std::ofstream& ofs)
    {
        Index * inds = utils::d2h_cpy(d_inds, nnz);
        ValueType * vals = utils::d2h_cpy(d_vals, nnz);
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
        delete[] inds;
        delete[] vals;
    }


    inline ValueType * get_d_vals() {return d_vals;}
    inline Index * get_d_inds() {return d_inds;}
    inline Index get_mode_sizes() {return mode_sizes;}
    inline size_t get_nnz() const {return nnz;}


private:

    ValueType * d_vals;
    Index * d_inds;

    Index mode_sizes;
    size_t nnz;

};


} //mxt
#endif
