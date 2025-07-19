#include "run_ttmc.cuh"
using Conf0 = TtmcConfig<Shape<256,256,2048>,
                                                              Shape<100,100,100>,
                                                              double, double, CUBLAS_COMPUTE_64F,
                                                              GEN_SMALL>;

                        using Conf1 = TtmcConfig<Shape<256,256,2048>,
                                                              Shape<100,100,100>,
                                                              double, double, CUBLAS_COMPUTE_64F,
                                                              GEN_RANDN>;

                        using Conf2 = TtmcConfig<Shape<256,256,2048>,
                                                              Shape<100,100,100>,
                                                              double, double, CUBLAS_COMPUTE_64F,
                                                              GEN_BIG>;

                        using Conf3 = TtmcConfig<Shape<256,256,2048>,
                                                              Shape<200,200,200>,
                                                              double, double, CUBLAS_COMPUTE_64F,
                                                              GEN_SMALL>;

                        using Conf4 = TtmcConfig<Shape<256,256,2048>,
                                                              Shape<200,200,200>,
                                                              double, double, CUBLAS_COMPUTE_64F,
                                                              GEN_RANDN>;

                        using Conf5 = TtmcConfig<Shape<256,256,2048>,
                                                              Shape<200,200,200>,
                                                              double, double, CUBLAS_COMPUTE_64F,
                                                              GEN_BIG>;

                        using Conf6 = TtmcConfig<Shape<256,256,2048>,
                                                              Shape<500,500,500>,
                                                              double, double, CUBLAS_COMPUTE_64F,
                                                              GEN_SMALL>;

                        using Conf7 = TtmcConfig<Shape<256,256,2048>,
                                                              Shape<500,500,500>,
                                                              double, double, CUBLAS_COMPUTE_64F,
                                                              GEN_RANDN>;

                        using Conf8 = TtmcConfig<Shape<256,256,2048>,
                                                              Shape<500,500,500>,
                                                              double, double, CUBLAS_COMPUTE_64F,
                                                              GEN_BIG>;

                        using Conf9 = TtmcConfig<Shape<256,256,2048>,
                                                              Shape<100,100,100>,
                                                              double, float, CUBLAS_COMPUTE_32F,
                                                              GEN_SMALL>;

                        using Conf10 = TtmcConfig<Shape<256,256,2048>,
                                                              Shape<100,100,100>,
                                                              double, float, CUBLAS_COMPUTE_32F,
                                                              GEN_RANDN>;

                        using Conf11 = TtmcConfig<Shape<256,256,2048>,
                                                              Shape<100,100,100>,
                                                              double, float, CUBLAS_COMPUTE_32F,
                                                              GEN_BIG>;

                        using Conf12 = TtmcConfig<Shape<256,256,2048>,
                                                              Shape<200,200,200>,
                                                              double, float, CUBLAS_COMPUTE_32F,
                                                              GEN_SMALL>;

                        using Conf13 = TtmcConfig<Shape<256,256,2048>,
                                                              Shape<200,200,200>,
                                                              double, float, CUBLAS_COMPUTE_32F,
                                                              GEN_RANDN>;

                        using Conf14 = TtmcConfig<Shape<256,256,2048>,
                                                              Shape<200,200,200>,
                                                              double, float, CUBLAS_COMPUTE_32F,
                                                              GEN_BIG>;

                        using Conf15 = TtmcConfig<Shape<256,256,2048>,
                                                              Shape<500,500,500>,
                                                              double, float, CUBLAS_COMPUTE_32F,
                                                              GEN_SMALL>;

                        using Conf16 = TtmcConfig<Shape<256,256,2048>,
                                                              Shape<500,500,500>,
                                                              double, float, CUBLAS_COMPUTE_32F,
                                                              GEN_RANDN>;

                        using Conf17 = TtmcConfig<Shape<256,256,2048>,
                                                              Shape<500,500,500>,
                                                              double, float, CUBLAS_COMPUTE_32F,
                                                              GEN_BIG>;

                        using Conf18 = TtmcConfig<Shape<256,256,2048>,
                                                              Shape<100,100,100>,
                                                              double, float, CUBLAS_COMPUTE_32F_FAST_16F,
                                                              GEN_SMALL>;

                        using Conf19 = TtmcConfig<Shape<256,256,2048>,
                                                              Shape<100,100,100>,
                                                              double, float, CUBLAS_COMPUTE_32F_FAST_16F,
                                                              GEN_RANDN>;

                        using Conf20 = TtmcConfig<Shape<256,256,2048>,
                                                              Shape<100,100,100>,
                                                              double, float, CUBLAS_COMPUTE_32F_FAST_16F,
                                                              GEN_BIG>;

                        using Conf21 = TtmcConfig<Shape<256,256,2048>,
                                                              Shape<200,200,200>,
                                                              double, float, CUBLAS_COMPUTE_32F_FAST_16F,
                                                              GEN_SMALL>;

                        using Conf22 = TtmcConfig<Shape<256,256,2048>,
                                                              Shape<200,200,200>,
                                                              double, float, CUBLAS_COMPUTE_32F_FAST_16F,
                                                              GEN_RANDN>;

                        using Conf23 = TtmcConfig<Shape<256,256,2048>,
                                                              Shape<200,200,200>,
                                                              double, float, CUBLAS_COMPUTE_32F_FAST_16F,
                                                              GEN_BIG>;

                        using Conf24 = TtmcConfig<Shape<256,256,2048>,
                                                              Shape<500,500,500>,
                                                              double, float, CUBLAS_COMPUTE_32F_FAST_16F,
                                                              GEN_SMALL>;

                        using Conf25 = TtmcConfig<Shape<256,256,2048>,
                                                              Shape<500,500,500>,
                                                              double, float, CUBLAS_COMPUTE_32F_FAST_16F,
                                                              GEN_RANDN>;

                        using Conf26 = TtmcConfig<Shape<256,256,2048>,
                                                              Shape<500,500,500>,
                                                              double, float, CUBLAS_COMPUTE_32F_FAST_16F,
                                                              GEN_BIG>;

                        using Conf27 = TtmcConfig<Shape<256,256,2048>,
                                                              Shape<100,100,100>,
                                                              double, __half, CUBLAS_COMPUTE_16F,
                                                              GEN_SMALL>;

                        using Conf28 = TtmcConfig<Shape<256,256,2048>,
                                                              Shape<100,100,100>,
                                                              double, __half, CUBLAS_COMPUTE_16F,
                                                              GEN_RANDN>;

                        using Conf29 = TtmcConfig<Shape<256,256,2048>,
                                                              Shape<100,100,100>,
                                                              double, __half, CUBLAS_COMPUTE_16F,
                                                              GEN_BIG>;

                        using Conf30 = TtmcConfig<Shape<256,256,2048>,
                                                              Shape<200,200,200>,
                                                              double, __half, CUBLAS_COMPUTE_16F,
                                                              GEN_SMALL>;

                        using Conf31 = TtmcConfig<Shape<256,256,2048>,
                                                              Shape<200,200,200>,
                                                              double, __half, CUBLAS_COMPUTE_16F,
                                                              GEN_RANDN>;

                        using Conf32 = TtmcConfig<Shape<256,256,2048>,
                                                              Shape<200,200,200>,
                                                              double, __half, CUBLAS_COMPUTE_16F,
                                                              GEN_BIG>;

                        using Conf33 = TtmcConfig<Shape<256,256,2048>,
                                                              Shape<500,500,500>,
                                                              double, __half, CUBLAS_COMPUTE_16F,
                                                              GEN_SMALL>;

                        using Conf34 = TtmcConfig<Shape<256,256,2048>,
                                                              Shape<500,500,500>,
                                                              double, __half, CUBLAS_COMPUTE_16F,
                                                              GEN_RANDN>;

                        using Conf35 = TtmcConfig<Shape<256,256,2048>,
                                                              Shape<500,500,500>,
                                                              double, __half, CUBLAS_COMPUTE_16F,
                                                              GEN_BIG>;

                        
            int main(int argc, char ** argv)
            {
                std::string path("../tensors/miranda.dns");
                mxt_init();
                run<Conf0,Conf1,Conf2,Conf3,Conf4,Conf5,Conf6,Conf7,Conf8,Conf9,Conf10,Conf11,Conf12,Conf13,Conf14,Conf15,Conf16,Conf17,Conf18,Conf19,Conf20,Conf21,Conf22,Conf23,Conf24,Conf25,Conf26,Conf27,Conf28,Conf29,Conf30,Conf31,Conf32,Conf33,Conf34,Conf35>(path);
                mxt_finalize();
                return 0;
            }

            