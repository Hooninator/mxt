#include "run.cuh"
using Conf0 = Config<Shape<100,100,100,100>,
                                                      Shape<10,10,10,10>,
                                                      double, double, double, double,
                                                      uint64_t>;

                using Conf1 = Config<Shape<100,100,100,100>,
                                                      Shape<10,10,10,10>,
                                                      double, float, float, float,
                                                      uint64_t>;

                using Conf2 = Config<Shape<100,100,100,100>,
                                                      Shape<10,10,10,10>,
                                                      double, __half, float, float,
                                                      uint64_t>;

                
            int main(int argc, char ** argv)
            {
                std::string path("../tensors/evil_fp16.tns");
                mxt_init();
                run<Conf0,Conf1,Conf2>(path);
                mxt_finalize();
                return 0;
            }

            