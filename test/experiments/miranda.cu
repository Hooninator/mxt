#include "run_ttmc.cuh"
using Conf0 = TtmcConfig<Shape<256,256,2048>,
                                                      Shape<100,100,100>,
                                                      double, double, CUBLAS_COMPUTE_64F>;

                using Conf1 = TtmcConfig<Shape<256,256,2048>,
                                                      Shape<100,100,100>,
                                                      double, float, CUBLAS_COMPUTE_32F>;

                using Conf2 = TtmcConfig<Shape<256,256,2048>,
                                                      Shape<100,100,100>,
                                                      double, float, CUBLAS_COMPUTE_32F_FAST_16F>;

                using Conf3 = TtmcConfig<Shape<256,256,2048>,
                                                      Shape<100,100,100>,
                                                      double, __half, CUBLAS_COMPUTE_16F>;

                
            int main(int argc, char ** argv)
            {
                std::string path("../tensors/miranda.dns");
                mxt_init();
                run<Conf0,Conf1,Conf2,Conf3>(path);
                mxt_finalize();
                return 0;
            }

            