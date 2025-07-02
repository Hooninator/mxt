#include "run.cuh"
using Conf0 = Config<Shape<2482,2862,14036,17>,
                                                      Shape<10,10,10,10>,
                                                      double, double, double, double,
                                                      uint64_t>;

                using Conf1 = Config<Shape<2482,2862,14036,17>,
                                                      Shape<10,10,10,10>,
                                                      double, float, float, float,
                                                      uint64_t>;

                using Conf2 = Config<Shape<2482,2862,14036,17>,
                                                      Shape<10,10,10,10>,
                                                      double, __half, float, __half,
                                                      uint64_t>;

                
            int main(int argc, char ** argv)
            {
                std::string path("../tensors/nips.tns");
                mxt_init();
                run<Conf0,Conf1,Conf2>(path);
                mxt_finalize();
                return 0;
            }

            