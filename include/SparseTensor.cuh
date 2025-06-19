#ifndef SPARSE_TENSOR_CUH
#define SPARSE_TENSOR_CUH

#include "common.cuh"
#include "Shape.cuh"
#include "utils.cuh"


namespace mxt
{


template <typename VT, typename IT, uint32_t _Order, typename Shape>
class SparseTensor
{
public:

    using ValueType_t = VT;
    using IndexType_t = IT;

    static constexpr uint32_t Order = _Order;

    using Index_t = std::array<IndexType_t, Order>;

    static constexpr Index_t Modes = Shape::dims;
    using ShapeType_t = Shape;


    SparseTensor(){}

    SparseTensor(std::vector<Index_t>& inds, std::vector<ValueType_t>& vals):
        nnz(inds.size()), 
        d_vals(utils::h2d_cpy(vals)),
        d_inds(utils::h2d_cpy(inds))
    {
        ASSERT( (inds.size() == vals.size()), "Number of indices %zu is not equal to number of nonzeros %zu", inds.size(), vals.size() ); 
    }


    inline size_t bytes_vals()
    {
        return sizeof(ValueType_t) * this->nnz;
    }


    inline size_t bytes_inds()
    {
        return sizeof(Index_t) * this->nnz;
    }


    void dump(std::ofstream& ofs)
    {
        Index_t * inds = utils::d2h_cpy(d_inds, nnz);
        ValueType_t * vals = utils::d2h_cpy(d_vals, nnz);
        for (size_t i=0; i<nnz; i++)
        {
            Index_t& index = inds[i];
            ValueType_t& val = vals[i];

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


    inline ValueType_t * get_d_vals() {return d_vals;}
    inline Index_t * get_d_inds() {return d_inds;}
    inline size_t get_nnz() const {return nnz;}


private:

    ValueType_t * d_vals;
    Index_t * d_inds;

    size_t nnz;

};


} //mxt
#endif
