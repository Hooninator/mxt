#ifndef SYMBOLIC_TTMC_CUH
#define SYMBOLIC_TTMC_CUH

#include "DeviceWorkspace.cuh"

namespace mxt
{

struct SymbolicTTMC
{

    //TODO: Add more fields to this, some static ones that tell us info about dimensions of X

    template <typename SparseTensor_t>
    SymbolicTTMC (SparseTensor_t& X)
    {
        using Index_t = SparseTensor_t::Index_t;
        using IndexType_t = SparseTensor_t::IndexType_t;
        using Workspace = DeviceWorkspace<size_t>;

        static constexpr uint32_t Order = SparseTensor_t::Order;
        static constexpr Index_t Modes = SparseTensor_t::Modes;

        /* Y_n_inds[n][i] -- indices of the ith mode-n slice X(:, :, ..., i_n, ..., :) */
        const size_t nnz = X.get_nnz();

        // First index determines mode unfolding, second determines row index of Y_n, then you have the acutal index list
        std::vector<std::vector<std::vector<size_t> > > h_Y_n_inds;

        static constexpr size_t ModeSum = std::reduce(Modes.begin(), Modes.end(), 0);
        std::vector<size_t> h_Y_n_offsets(ModeSum + 1, 0);

        std::vector<size_t> mode_offsets(Order, 0);
        std::exclusive_scan(Modes.begin(), Modes.end(), mode_offsets.begin(), 0);

        h_Y_n_inds.resize(Order);
        for (size_t n=0; n<Order; n++)
        {
            h_Y_n_inds[n].resize(Modes[n]);
        }

        Index_t * h_inds = utils::d2h_cpy(X.get_d_inds(), nnz);

        for (size_t i=0; i<nnz; i++)
        {
            Index_t idx = h_inds[i];
            for (uint32_t n=0; n < Order; n++)
            {
                h_Y_n_inds[n][idx[n]].push_back(i);
                size_t offset = idx[n] + mode_offsets[n];
                h_Y_n_offsets[offset + 1] += 1;
            }
        }

        std::inclusive_scan(h_Y_n_offsets.begin(), h_Y_n_offsets.end(), h_Y_n_offsets.begin());

        d_Y_n_inds.alloc(nnz * Order);
        d_Y_n_offsets.alloc(ModeSum + 1);
        d_Y_mode_offsets.alloc(Order);

        // Move to device
        d_Y_n_offsets.h2d_cpy(h_Y_n_offsets.data(), h_Y_n_offsets.size());
        d_Y_mode_offsets.h2d_cpy(mode_offsets.data(), mode_offsets.size());

        // Indices to device
        size_t offset = 0;
        for (size_t n=0; n < Order; n++)
        {
            for (size_t i=0; i < Modes[n]; i++)
            {
                d_Y_n_inds.h2d_cpy(h_Y_n_inds[n][i].data(), h_Y_n_inds[n][i].size(), offset);
                offset += h_Y_n_inds[n][i].size();
            }
        }

        delete[] h_inds;

    }


    void dump(std::ofstream& ofs)
    {
        d_Y_n_inds.dump(ofs, "!!!Indices!!!");
        d_Y_n_offsets.dump(ofs, "!!!Offsets!!!");
        d_Y_mode_offsets.dump(ofs, "!!!Mode Offsets!!!");
    }

    DeviceWorkspace<size_t> d_Y_n_inds;
    DeviceWorkspace<size_t> d_Y_n_offsets; // Starting index in d_Y_n_inds of each row of all unfoldings
    DeviceWorkspace<size_t> d_Y_mode_offsets; // Starting index in d_Y_n_offsets of rows corresponding to each unfolding
};

} //mxt

#endif
