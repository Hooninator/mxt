#ifndef SPARSE_TENSOR_CUH
#define SPARSE_TENSOR_CUH

#include "common.cuh"
#include "Shape.cuh"
#include "utils.cuh"


namespace mxt
{


template <typename VT, typename VTL, typename IT, uint32_t _Order, typename Shape>
class SparseTensor
{
public:

    using ValueType_t = VT;
    using ValueTypeLow_t = VTL;
    using IndexType_t = IT;

    static constexpr uint32_t Order = _Order;

    using Index_t = std::array<IndexType_t, Order>;

    static constexpr Index_t Modes = Shape::dims;
    using ShapeType_t = Shape;


    SparseTensor(){}

    SparseTensor(std::vector<Index_t>& inds, std::vector<ValueType_t>& vals):
        nnz(inds.size())
    {
        ASSERT( (inds.size() == vals.size()), "Number of indices %zu is not equal to number of nonzeros %zu", inds.size(), vals.size() ); 

        sort_inds_vals(std::move(inds), std::move(vals));

        for (size_t i = 0; i<nnz; i++)
        {
            indmap[inds[i]] = i;
        }

        d_inds = utils::h2d_cpy(inds);
        d_vals = utils::h2d_cpy(vals);
        d_vals_low = utils::d_to_u<ValueType_t, ValueTypeLow_t>(d_vals, nnz);
    }


    void sort_inds_vals(std::vector<Index_t>&& inds, std::vector<ValueType_t>&& vals)
    {
        std::vector<IndexType_t> sort_inds(nnz);
        std::iota(sort_inds.begin(), sort_inds.end(), 0);

        std::sort(sort_inds.begin(), sort_inds.end(), [&](auto i, auto k) {return std::lexicographical_compare(inds[i].begin(), inds[i].end(), inds[k].begin(), inds[k].end());});

        std::vector<Index_t> sorted_inds(nnz);
        std::vector<ValueType_t> sorted_vals(nnz);

        for (size_t i = 0; i<nnz; i++)
        {
            sorted_inds[i] = inds[sort_inds[i]];
            sorted_vals[i] = vals[sort_inds[i]];
        }

        inds = std::move(sorted_inds);
        vals = std::move(sorted_vals);
    }


    void make_unfolding0_csr(std::vector<Index_t>& inds, std::vector<ValueType_t>& vals)
    {
        static constexpr IndexType_t M = Modes[0];

        std::vector<IndexType_t> colinds(nnz);
        std::vector<IndexType_t> rowptrs(M + 1, 0);

        for (size_t i = 0; i<nnz; i++)
        {
            rowptrs[inds[i][0] + 1]++;
            colinds[i] = colidx(inds[i]);
        }

        std::inclusive_scan(rowptrs.begin() + 1, rowptrs.end(), rowptrs.begin() + 1);

        d_colinds = utils::h2d_cpy(colinds);
        d_rowptrs = utils::h2d_cpy(rowptrs);

        CUSPARSE_CHECK(cusparseCreateCsr(&cusparse_descr,
                                         Modes[0], unfolding_cols<0>(), nnz, 
                                         d_rowptrs, d_colinds, d_vals,
                                         utils::to_cusparse_idx<IndexType_t>(),
                                         utils::to_cusparse_idx<IndexType_t>(),
                                         CUSPARSE_INDEX_BASE_ZERO,
                                         utils::to_cuda_dtype<ValueType_t>()));
    }


    IndexType_t colidx(Index_t& idx)
    {
        IndexType_t result = 1;
        IndexType_t offset = 1;
        for (size_t n = 1; n < Order; n++)
        {
            result += Modes[n] * offset;
            offset *= Modes[n];
        }
        return result;
    }


    template <uint32_t Mode>
    constexpr inline IndexType_t unfolding_cols()
    {
        return std::reduce(Modes.begin(), Modes.end(), 1, std::multiplies<IndexType_t>{}) / Modes[Mode];
    }


    inline size_t bytes_vals()
    {
        return sizeof(ValueType_t) * this->nnz;
    }


    inline size_t bytes_inds()
    {
        return sizeof(Index_t) * this->nnz;
    }


    bool has_idx(Index_t& idx)
    {
        return indmap.contains(idx);
    }


    size_t get_idx_idx(Index_t& idx)
    {
        if (indmap.contains(idx))
        {
            return indmap[idx];
        }
        return nnz + 1;
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


    ~SparseTensor()
    {
        CUDA_FREE(d_vals);
        CUDA_FREE(d_inds);
        CUDA_FREE(d_vals_low);
    }


    inline ValueType_t * get_d_vals() {return d_vals;}
    inline ValueTypeLow_t * get_d_vals_low() {return d_vals_low;}
    inline Index_t * get_d_inds() {return d_inds;}
    inline IndexType_t * get_d_colinds() {return d_colinds;}
    inline IndexType_t * get_d_rowptrs() {return d_rowptrs;}
    inline size_t get_nnz() const {return nnz;}
    inline cusparseSpMatDescr_t get_cusparse_descr() { return cusparse_descr; }


private:
    


    struct IndexHash
    {

        std::size_t operator()(const Index_t& idx) const
        {
            std::size_t seed = 0;
            for (IndexType_t v : idx)
            {
                seed ^= v + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            }
            return seed;
        }
    };


    ValueType_t * d_vals;
    ValueTypeLow_t * d_vals_low;
    Index_t * d_inds;
    IndexType_t * d_colinds;
    IndexType_t * d_rowptrs;
    cusparseSpMatDescr_t cusparse_descr;

    std::unordered_map<Index_t, std::size_t, IndexHash> indmap;

    size_t nnz;

};


} //mxt
#endif
