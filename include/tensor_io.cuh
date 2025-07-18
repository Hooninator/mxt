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
        idx[i] = std::stoul(line.substr(spos, (epos - spos))) - 1; // 1-indexing
        spos = epos+1;
    }
    epos = line.size();
    val = static_cast<Value_t>(std::stod(line.substr(spos, epos - spos)));

}


template <typename T>
T * read_matrix_frostt(const char * fpath, const size_t M, const size_t N, const bool transpose=false)
{
    std::ifstream infile;
    infile.open(fpath);

    T * vals = new T[M * N];

    std::string line;
    T val;
    std::array<size_t, 2> idx;
    while (std::getline(infile, line))
    {
        parse_frostt_line<std::array<size_t, 2>, T, 2>(line, idx, val);
        vals[idx[0] * N + idx[1]] = val;
    }

    infile.close();

    if (transpose)
    {
        utils::transpose(vals, N, M);
    }

    T * d_vals = utils::h2d_cpy(vals, M * N);
    delete[] vals;

    return d_vals;
}


template <typename T>
T * read_matrix_dns(const char * fpath, const size_t M, const size_t N, const bool transpose=false)
{
    std::ifstream infile;
    infile.open(fpath);

    T * vals = new T[M * N];

    std::string line;

    std::getline(infile, line);
    std::getline(infile, line);
    std::getline(infile, line);

    T val;
    std::array<size_t, 2> idx;
    size_t offset = 0;
    while (std::getline(infile, line))
    {
        val = std::stod(line);
        idx = utils::multidx_reverse<2>(offset, {M, N});
        vals[idx[0] + idx[1] * M] = val;
        offset++;
    }

    infile.close();

    if (transpose)
    {
        utils::transpose(vals, N, M);
    }

    T * d_vals = utils::h2d_cpy(vals, M * N);
    delete[] vals;

    return d_vals;
}


template <typename T, typename ShapeT>
T * read_dense_tensor_frostt(const char * fpath, const bool natural_order=false)
{
    std::ifstream infile;
    infile.open(fpath);

    static constexpr auto Dims = ShapeT::dims;
    static constexpr size_t N = Dims.size();
    static constexpr size_t In = std::reduce(Dims.begin(), Dims.end(), 1, std::multiplies<size_t>{});

    T * vals = new T[In];
    std::memset(vals, 0, sizeof(T)*In);

    std::string line;

    // Skip first three lines
    std::getline(infile, line);
    std::getline(infile, line);
    std::getline(infile, line);

    std::array<size_t, N> idx;
    size_t offset = 0;
    T val;
    while (std::getline(infile, line))
    {
        parse_frostt_line<std::array<size_t, N>, T, N>(line, idx, val);
        offset = (natural_order) ? utils::linear_index(Dims.data(), idx.data(), N) : offset + 1;
        vals[offset] = val;
    }

    infile.close();

    T * d_vals = utils::h2d_cpy(vals, In);
    delete[] vals;

    return d_vals;
}


template <typename T, typename ShapeT>
T * read_dense_tensor_dns(const char * fpath)
{
    std::ifstream infile;
    infile.open(fpath);

    static constexpr auto Dims = ShapeT::dims;
    static constexpr size_t N = Dims.size();
    static constexpr size_t In = std::reduce(Dims.begin(), Dims.end(), 1, std::multiplies<size_t>{});

    T * vals = new T[In];
    std::memset(vals, 0, sizeof(T)*In);

    std::string line;

    // Skip first three lines
    std::getline(infile, line);
    std::getline(infile, line);
    std::getline(infile, line);

    std::array<size_t, N> idx;
    size_t offset1 = 0;
    size_t offset2 = 0;
    T val;
    while (std::getline(infile, line))
    {
        val = std::stod(line);
        idx = utils::multidx_reverse<Dims.size()>(offset1, Dims);
        offset2 = utils::linear_index(Dims.data(), idx.data(), N);
        vals[offset2] = val;
        offset1++;
    }

    infile.close();

    T * d_vals = utils::h2d_cpy(vals, In);
    delete[] vals;

    return d_vals;
}


template <typename SparseTensor_t>
SparseTensor_t read_tensor_frostt(const char * fpath)
{

    using Index_t = typename SparseTensor_t::Index_t;
    using Value_t = typename SparseTensor_t::ValueType_t;

    std::ifstream infile;
    infile.open(fpath);
    
    //First, get the tensor order
    std::string line;
    std::getline(infile, line);
    std::getline(infile, line);
    std::getline(infile, line);
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
    std::getline(infile, line);
    std::getline(infile, line);
    std::getline(infile, line);
    std::getline(infile, line);

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
