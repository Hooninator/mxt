#include "mxt.cuh"
#include "TtmcConfig.cuh"

#include <map>
#include <string>
#include <sstream>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>


using namespace mxt;
static const char * base = "../test/correctness/ttmc_golden/";


template <typename Conf>
void run_correctness(std::string& path, std::string& tensorname)
{
    using DenseTensor_t = DenseTensor<typename Conf::HighU_t, typename Conf::MatrixCols_t>;
    using MatrixCollection_t = MatrixCollection<typename Conf::LowU_t, typename Conf::MatrixRows_t, typename Conf::MatrixCols_t>;
    using OutputDenseTensor_t = DenseTensor<typename DenseTensor_t::ValueType_t, typename Conf::MatrixRows_t>;

    utils::print_separator("Beginning IO");
    DenseTensor_t X(path.c_str()); 
    utils::print_separator("Done IO");

    std::string golden_dir = std::string(base);
    golden_dir.append(tensorname);
    golden_dir.append("/");

    MatrixCollection_t matrices(golden_dir.c_str());

    utils::print_separator("Beginning TTMc");
    OutputDenseTensor_t Y = ttmc_mixed<DenseTensor_t, MatrixCollection_t, OutputDenseTensor_t>(X, matrices);
    utils::print_separator("Done TTMc");

    /* Correctness check */
    std::string correct_output = golden_dir + "output.dns";
    OutputDenseTensor_t Y_correct(correct_output.c_str());
    bool correct = Y == Y_correct;

    if (correct)
    {
        std::cout<<GREEN<<"Correctness passed!"<<RESET<<std::endl;
    }
    else
    {
        std::cout<<RED<<"Correctness failed"<<RESET<<std::endl;
    }
}



using SmallDense = TtmcConfig<Shape<3, 3, 3>, 
                     Shape<2,2,2>, 
                     double, double, CUBLAS_COMPUTE_64F, GEN_RANDN>;


using IndianPines = TtmcConfig<Shape<145, 145, 200>, 
                        Shape<20, 20, 20>,
                        double, double, CUBLAS_COMPUTE_64F, GEN_RANDN>;


int main(int argc, char ** argv)
{
    if (argc < 2)
    {
        std::cerr<<"Usage: ./correctness <tensor_name>"<<std::endl;
        std::abort();
    }

    std::string tensor = std::string(argv[1]);
    std::stringstream ss;
    ss<<"../tensors/"<<tensor<<".dns";
    std::string path = ss.str();

    mxt_init();

    if (tensor.compare("small_dense")==0)
    {
        run_correctness<SmallDense>(path, tensor);
    }
    else if (tensor.compare("indian_pines")==0)
    {
        run_correctness<IndianPines>(path, tensor);
    }
    else
    {
        std::cerr<<"Invalid tensor: "<<tensor<<std::endl;
        std::abort();
    }

    mxt_finalize();

    return 0;
}
